import torch
import neptune
import numpy
import torch.nn.functional as F
from torch.distributions.utils import broadcast_all
from typing import Union, Callable, Optional, Tuple
from collections import OrderedDict
from torch.distributions.distribution import Distribution
from torch.distributions import constraints

from MODULES.utilities import compute_average_in_box, compute_ranking, convert_to_box_list
from MODULES.utilities import pass_bernoulli
from MODULES.utilities_visualization import show_batch
from MODULES.namedtuple import DIST, MetricMiniBatch, BB
from MODULES.utilities_neptune import log_dict_metrics

def are_broadcastable(a: torch.Tensor, b: torch.Tensor) -> bool:
    """ Return True if tensor are broadcastable to each other, False otherwise """
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(a.shape[::-1], b.shape[::-1]))


def sample_and_kl_diagonal_normal(posterior_mu: torch.Tensor,
                                  posterior_std: torch.Tensor,
                                  prior_mu: torch.Tensor,
                                  prior_std: torch.Tensor,
                                  noisy_sampling: bool,
                                  sample_from_prior: bool) -> DIST:

    post_mu, post_std, pr_mu, pr_std = broadcast_all(posterior_mu, posterior_std, prior_mu, prior_std)
    if sample_from_prior:
        # working with the prior
        sample = pr_mu + pr_std * torch.randn_like(pr_mu) if noisy_sampling else pr_mu
        kl = torch.zeros_like(pr_mu)
    else:
        # working with the posterior
        sample = post_mu + post_std * torch.randn_like(post_mu) if noisy_sampling else post_mu
        tmp = (post_std + pr_std) * (post_std - pr_std) + (post_mu - pr_mu).pow(2)
        kl = tmp / (2 * pr_std * pr_std) - post_std.log() + pr_std.log()

    return DIST(sample=sample, kl=kl)


def sample_and_kl_prob(logit_map: torch.Tensor,
                       similarity_kernel: torch.Tensor,
                       images: torch.Tensor,
                       background: torch.Tensor,
                       bounding_box_no_noise: BB,
                       prob_corr_factor: float,
                       noisy_sampling: bool,
                       sample_from_prior: bool) -> Tuple[DIST, torch.Tensor]:

    # Correction factor
    if sample_from_prior:
        with torch.no_grad():
            batch_size = torch.Size([logit_map.shape[0]])
            s = similarity_kernel.requires_grad_(False)
            c_all = FiniteDPP(L=s).sample(sample_shape=batch_size).transpose(-1, -2).float()
            kl = torch.zeros(logit_map.shape[0])
            q_all = torch.zeros_like(c_all).float()
    else:
        # Work with posterior
        if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):
            with torch.no_grad():
                av_intensity = compute_average_in_box((images - background).abs(), bounding_box_no_noise)
                assert len(av_intensity.shape) == 2
                n_boxes_all, batch_size = av_intensity.shape
                ranking = compute_ranking(av_intensity)  # n_boxes_all, batch. It is in [0,n_box_all-1]
                tmp = ((ranking + 1).float() / (n_boxes_all + 1))
                q_approx = tmp.pow(10)

            q_uncorrected = torch.sigmoid(convert_to_box_list(logit_map).squeeze(-1))
            q_all = ((1 - prob_corr_factor) * q_uncorrected + prob_corr_factor * q_approx).clamp(min=1E-4, max=1 - 1E-4)
            log_q = torch.log(q_all)
            log_one_minus_q = torch.log1p(-q_all)
        else:
            logit_reshaped = convert_to_box_list(logit_map).squeeze(-1)
            q_all = torch.sigmoid(logit_reshaped)
            log_q = F.logsigmoid(logit_reshaped)
            log_one_minus_q = F.logsigmoid(-logit_reshaped)

        c_all = pass_bernoulli(prob=q_all, noisy_sampling=noisy_sampling)  # float variable which requires grad

        # Here the gradients are only through log_q and similarity_kernel not c
        c_no_grad = c_all.bool().detach()  # bool variable has requires_grad = False
        log_prob_posterior = (c_no_grad * log_q + ~c_no_grad * log_one_minus_q).sum(dim=0)
        log_prob_prior = FiniteDPP(L=similarity_kernel).log_prob(c_no_grad.transpose(-1, -2))  # shape: batch_shape
        assert log_prob_posterior.shape == log_prob_prior.shape
        kl = log_prob_posterior - log_prob_prior

    return DIST(sample=c_all, kl=kl), q_all


class SimilarityKernel(torch.nn.Module):
    """ Similarity based on sum of gaussian kernels of different strength and length_scales """

    def __init__(self, n_kernels: int = 4, eps: float = 1E-4):
        super().__init__()

        self.n_kernels = n_kernels
        self.eps = eps
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.similarity_w = torch.nn.Parameter(data=torch.ones(self.n_kernels,
                                                               device=self.device,
                                                               dtype=torch.float)/self.n_kernels, requires_grad=True)
        self.similarity_s2 = torch.nn.Parameter(data=100*torch.randn(self.n_kernels,
                                                                     device=self.device,
                                                                     dtype=torch.float), requires_grad=True)

        # Initialization
        self.n_width = -1
        self.n_height = -1
        self.d2 = None
        self.diag = None

    def _compute_d2_diag(self, n_width: int, n_height: int):
        with torch.no_grad():
            ix_array = torch.arange(start=0, end=n_width, dtype=torch.int, device=self.device)
            iy_array = torch.arange(start=0, end=n_height, dtype=torch.int, device=self.device)
            ix_grid, iy_grid = torch.meshgrid([ix_array, iy_array])
            map_points = torch.stack((ix_grid, iy_grid), dim=-1)  # n_width, n_height, 2
            locations = map_points.flatten(start_dim=0, end_dim=-2)  # (n_width*n_height, 2)
            d2 = (locations.unsqueeze(-2) - locations.unsqueeze(-3)).pow(2).sum(dim=-1).float()
            diag = torch.eye(d2.shape[-2],
                             dtype=torch.float,
                             device=self.device,
                             requires_grad=False) * self.eps
            return d2, diag

    def sample_2_mask(self, sample):
        independent_dims = list(sample.shape[:-1])
        mask = sample.view(independent_dims + [self.n_width, self.n_height])
        return mask

    def get_sigma2_w(self):
        return F.softplus(self.similarity_s2), F.softplus(self.similarity_w)

    def forward(self, n_width: int, n_height: int):
        """ Implement L = sum_i a_i exp[-b_i d2] """
        sigma2, w = self.get_sigma2_w()

        if (n_width != self.n_width) or (n_height != self.n_height):
            self.n_width = n_width
            self.n_height = n_height
            self.d2, self.diag = self._compute_d2_diag(n_width=n_width, n_height=n_height)

        likelihood_kernel = (w[..., None, None] * torch.exp(-0.5*self.d2/sigma2[..., None, None])).sum(dim=-3) + self.diag
        return likelihood_kernel  # shape (n_width*n_height, n_width*n_height)


class FiniteDPP(Distribution):
    """ Finite DPP distribution defined via:
        1. L = likelihood kernel of shape *,n,n
        2. K = correlation kernel of shape *,n,n

        The constraints are:
        K = positive semidefinite, symmetric, eigenvalues in [0,1]
        L = positive semidefinite, symmetric, eigenvalues >= 0
    """

    arg_constraints = {'K': constraints.positive_definite,
                       'L': constraints.positive_definite}
    support = constraints.boolean
    has_rsample = False

    def __init__(self, K=None, L=None, validate_args=None):

        if (K is None and L is None) or (K is not None and L is not None):
            raise Exception("only one among K and L need to be defined")

        elif K is not None:
            self.K = 0.5 * (K + K.transpose(-1, -2))  # make sure it is symmetrized
            u, s_k, v = torch.svd(self.K)
            s_l = s_k / (1.0 - s_k)
            self.L = torch.matmul(u * s_l.unsqueeze(-2), v.transpose(-1, -2))

            tmp = torch.matmul(u * s_k.unsqueeze(-2), v.transpose(-1, -2))
            check = (tmp - self.K).abs().max()
            # print("check ->",check)
            assert check < 1E-4

        else:
            self.L = 0.5 * (L + L.transpose(-1, -2))  # make sure it is symmetrized
            u, s_l, v = torch.svd(self.L)
            s_k = s_l / (1.0 + s_l)
            self.K = torch.matmul(u * s_k.unsqueeze(-2), v.transpose(-1, -2))

            tmp = torch.matmul(u * s_l.unsqueeze(-2), v.transpose(-1, -2))
            check = (tmp - self.L).abs().max()
            # print("check ->",check)
            assert check < 1E-4

        self.s_l = s_l
        batch_shape, event_shape = self.K.shape[:-2], self.K.shape[-1:]
        super(FiniteDPP, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(FiniteDPP, _instance)
        batch_shape = torch.Size(batch_shape)
        kernel_shape = batch_shape + self.event_shape + self.event_shape
        value_shape = batch_shape + self.event_shape
        new.s_l = self.s_l.expand(value_shape)
        new.L = self.L.expand(kernel_shape)
        new.K = self.K.expand(kernel_shape)
        super(FiniteDPP, new).__init__(batch_shape,
                                       self.event_shape,
                                       validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape_value = self._extended_shape(sample_shape)  # shape = sample_shape + batch_shape + event_shape
        shape_kernel = shape_value + self._event_shape  # shape = sample_shape + batch_shape + event_shape + event_shape

        with torch.no_grad():
            K = self.K.expand(shape_kernel).clone()
            value = torch.zeros(shape_value, dtype=torch.bool, device=K.device)
            rand = torch.rand(shape_value, dtype=K.dtype, device=K.device)

            for j in range(rand.shape[-1]):
                c = rand[..., j] < K[..., j, j]
                value[..., j] = c
                K[..., j, j] -= (~c).to(K.dtype)
                K[..., j + 1:, j] /= K[..., j, j].unsqueeze(-1)
                K[..., j + 1:, j + 1:] -= K[..., j + 1:, j].unsqueeze(-1) * K[..., j, j + 1:].unsqueeze(-2)

            return value

    def log_prob(self, value):
        """ log_prob = logdet(Ls) - logdet(L+I)
            I am using the fact that eigen(L+I) = eigen(L)+1
            -> logdet(L+I)=log prod[ eigen(L+I) ] = sum log(eigen(L+I)) = sum log(eigen(L)+1)

            # value.shape = sample_shape + batch_shape + event_shape
            # logdet(L+I).shape = batch_shape
            :rtype:
        """
        assert are_broadcastable(value, self.L[..., 0])
        assert self.L.device == value.device
        assert value.dtype == torch.bool

        if self._validate_args:
            self._validate_sample(value)

        logdet_L_plus_I = (self.s_l + 1).log().sum(dim=-1)  # batch_shape

        # Reshapes
        independet_dims = list(value.shape[:-1])
        value = value.flatten(start_dim=0, end_dim=-2)  # *, event_shape
        L = self.L.expand(independet_dims + [-1, -1]).flatten(start_dim=0, end_dim=-3)  # *, event_shape, event_shape
        logdet_Ls = torch.zeros(independet_dims, dtype=self.L.dtype, device=value.device).view(-1)  # *

        # Select rows and columns of the matrix which correspond to selected particles
        for i in range(logdet_Ls.shape[0]):
            tmp = L[i, value[i], :][:, value[i]]
            logdet_Ls[i] = torch.logdet(tmp)
        logdet_Ls = logdet_Ls.view(independet_dims)  # sample_shape, batch_shape
        return logdet_Ls - logdet_L_plus_I


class Moving_Average_Calculator(object):
    """ Compute the moving average of a dictionary.
        Return the dictionary with the moving average up to that point

        beta is the factor multiplying the moving average.
        Approximately we average the last 1/(1-beta) points.
        For example:
        beta = 0.9 -> 10 points
        beta = 0.99 -> 100 points
        The larger beta the longer the time average.
    """

    def __init__(self, beta):
        super().__init__()
        self._bias = None
        self._steps = 0
        self._beta = beta
        self._dict_accumulate = {}
        self._dict_MA = {}

    def accumulate(self, input_dict):
        self._steps += 1
        self._bias = 1 - self._beta ** self._steps

        for key, value in input_dict.items():
            try:
                tmp = self._beta * self._dict_accumulate[key] + (1 - self._beta) * value
                self._dict_accumulate[key] = tmp
            except KeyError:
                self._dict_accumulate[key] = (1 - self._beta) * value
            self._dict_MA[key] = self._dict_accumulate[key] / self._bias
        return self._dict_MA


class Accumulator(object):
    """ accumulate a tuple or dictionary into a dictionary """

    def __init__(self):
        super().__init__()
        self._counter = 0
        self._dict_accumulate = OrderedDict()

    def _accumulate_key_value(self, key, value, counter_increment):
        if isinstance(value, torch.Tensor):
            x = value.detach().item() * counter_increment
        elif isinstance(value, float):
            x = value * counter_increment
        elif isinstance(value, numpy.ndarray):
            x = value * counter_increment
        else:
            print(type(value))
            raise Exception
        self._dict_accumulate[key] = x + self._dict_accumulate.get(key, 0)

    def accumulate(self, source: Union[tuple, dict], counter_increment: int = 1):
        self._counter += counter_increment

        if isinstance(source, tuple):
            for key in source._fields:
                value = getattr(source, key)
                self._accumulate_key_value(key, value, counter_increment)
        else:
            for key, value in source.items():
                self._accumulate_key_value(key, value, counter_increment)

    def get_average(self):
        tmp = self._dict_accumulate.copy()
        for k, v in self._dict_accumulate.items():
            tmp[k] = v/self._counter
        return tmp


class ConditionalRandomCrop(object):
    """ Crop a torch Tensor at random locations to obtain output of given size.
        The random crop is accepted only if it is inside the Region Of Interest (ROI) """

    def __init__(self, desired_w, desired_h, min_roi_fraction: float = 0.0, n_crops_per_image: int = 1):
        super().__init__()
        self.desired_w = desired_w
        self.desired_h = desired_h
        self.min_roi_fraction = min_roi_fraction
        self.desired_area = desired_w * desired_h
        self.n_crops_per_image = n_crops_per_image

    @staticmethod
    def get_smallest_corner_for_crop(w_raw: int, h_raw: int, w_desired: int, h_desired: int):
        assert w_desired <= w_raw
        assert h_desired <= h_raw

        if w_raw == w_desired and h_raw == h_desired:
            return 0, 0
        else:
            i = torch.randint(low=0, high=w_raw - w_desired + 1, size=[1]).item()
            j = torch.randint(low=0, high=h_raw - h_desired + 1, size=[1]).item()
            return i, j

    def get_index(self,
                  img: torch.Tensor,
                  roi_mask: Optional[torch.Tensor] = None,
                  cum_sum_roi_mask: Optional[torch.Tensor] = None,
                  n_crops_per_image: Optional[int] = None):
        """ img.shape: *,c,w,h where * might or might not be present
            roi_mask:  *,1,w,h where * might or might not be present
            cum_sum_roi_mask: *,1,w,h where * might or might not be present

            return a list of images
        """
        n_crops_per_image = self.n_crops_per_image if n_crops_per_image is None else n_crops_per_image

        if roi_mask is not None and cum_sum_roi_mask is not None:
            raise Exception("Only one between roi_mask and cum_sum_roi_mask can be specified")

        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if cum_sum_roi_mask is not None and len(cum_sum_roi_mask.shape) == 3:
            cum_sum_roi_mask = cum_sum_roi_mask.unsqueeze(0)
        if roi_mask is not None and len(roi_mask.shape) == 3:
            roi_mask = roi_mask.unsqueeze(0)

        assert len(img.shape) == 4
        assert (roi_mask is None or len(roi_mask.shape) == 4)
        assert (cum_sum_roi_mask is None or len(cum_sum_roi_mask.shape) == 4)

        with torch.no_grad():

            bij_list = []
            for b in range(img.shape[0]):
                for n in range(n_crops_per_image):
                    fraction = 0
                    while fraction < self.min_roi_fraction:
                        i, j = self.get_smallest_corner_for_crop(w_raw=img[b].shape[-2],
                                                                 h_raw=img[b].shape[-1],
                                                                 w_desired=self.desired_w,
                                                                 h_desired=self.desired_h)

                        if cum_sum_roi_mask is not None:
                            term1 = cum_sum_roi_mask[b, 0, i + self.desired_w - 1, j + self.desired_h - 1].item()
                            term2 = 0 if i < 1 else cum_sum_roi_mask[b, 0, i - 1, j + self.desired_h - 1].item()
                            term3 = 0 if j < 1 else cum_sum_roi_mask[b, 0, i + self.desired_w - 1, j - 1].item()
                            term4 = 0 if (i < 1 or j < 1) else cum_sum_roi_mask[b, 0, i - 1, j - 1].item()
                            fraction = float(term1 - term2 - term3 + term4) / self.desired_area
                        elif roi_mask is not None:
                            fraction = roi_mask[b, 0, i:i+self.desired_w,
                                       j:j+self.desired_h].sum().float()/self.desired_area
                        else:
                            fraction = 1.0

                    bij_list.append([b, i, j])
        return bij_list

    def collate_crops_from_list(self, img: torch.Tensor, bij_list: list):
        return torch.stack([img[b, :, i:i+self.desired_w, j:j+self.desired_h] for b, i, j in bij_list], dim=-4)

    def crop(self,
             img: torch.Tensor,
             roi_mask: Optional[torch.Tensor] = None,
             cum_sum_roi_mask: Optional[torch.Tensor] = None,
             n_crops_per_image: Optional[int] = None):

        n_crops_per_image = self.n_crops_per_image if n_crops_per_image is None else n_crops_per_image
        bij_list = self.get_index(img, roi_mask, cum_sum_roi_mask, n_crops_per_image)
        return self.collate_crops_from_list(img, bij_list)


class SpecialDataSet(object):
    def __init__(self,
                 img: torch.Tensor,
                 roi_mask: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None,
                 data_augmentation: Optional[ConditionalRandomCrop] = None,
                 store_in_cuda: bool = False,
                 drop_last=False,
                 batch_size=4,
                 shuffle=False):
        """ :param device: 'cpu' or 'cuda:0'
            Dataset returns random crops of a given size inside the Region Of Interest.
            The function getitem returns imgs, labels and indeces
        """
        assert len(img.shape) == 4
        assert (roi_mask is None or len(roi_mask.shape) == 4)
        assert (labels is None or labels.shape[0] == img.shape[0])

        storing_device = torch.device('cuda') if store_in_cuda else torch.device('cpu')

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Expand the dataset so that I can do one crop per image
        if data_augmentation is None:
            new_batch_size = img.shape[0]
            self.data_augmentaion = None
        else:
            new_batch_size = img.shape[0] * data_augmentation.n_crops_per_image
            self.data_augmentaion = data_augmentation

        if store_in_cuda:
            self.img = img.cuda().detach().expand(new_batch_size, -1, -1, -1)
        else:
            self.img = img.cpu().detach().expand(new_batch_size, -1, -1, -1)

        if labels is None:
            self.labels = -1*torch.ones(self.img.shape[0], device=storing_device).detach()
        else:
            self.labels = labels.to(storing_device).detach()
        self.labels = self.labels.expand(new_batch_size)

        if roi_mask is None:
            self.roi_mask = None
            self.cum_roi_mask = None
        else:
            self.roi_mask = roi_mask.to(storing_device).detach().expand(new_batch_size, -1, -1, -1)
            self.cum_roi_mask = roi_mask.to(storing_device).detach().cumsum(dim=-1).cumsum(
                dim=-2).expand(new_batch_size, -1, -1, -1)

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index: torch.Tensor):
        assert isinstance(index, torch.Tensor)

        if self.data_augmentaion is None:
            return self.img[index], self.labels[index], index
        else:
            bij_list = []
            for i in index:
                bij_list += self.data_augmentaion.get_index(img=self.img[i],
                                                            cum_sum_roi_mask=self.cum_roi_mask[i],
                                                            n_crops_per_image=1)
            return self.data_augmentaion.collate_crops_from_list(self.img, bij_list), self.labels[index], index

    def __iter__(self, batch_size=None, drop_last=None, shuffle=None):
        # If not specified use defaults
        batch_size = self.batch_size if batch_size is None else batch_size
        drop_last = self.drop_last if drop_last is None else drop_last
        shuffle = self.shuffle if shuffle is None else shuffle

        # Actual generation of iterator
        n_max = max(1, self.__len__() - (self.__len__() % batch_size) if drop_last else self.__len__())
        index = torch.randperm(self.__len__()).long() if shuffle else torch.arange(self.__len__()).long()
        for pos in range(0, n_max, batch_size):
            yield self.__getitem__(index[pos:pos + batch_size])
            
    def load(self, batch_size=None, index=None):
        if (batch_size is None and index is None) or (batch_size is not None and index is not None):
            raise Exception("Only one between batch_size and index must be specified")
        index = torch.randint(low=0, high=self.__len__(), size=(batch_size,)).long() if index is None else index
        return self.__getitem__(index)

    def check_batch(self, batch_size: int = 8):
        print("Dataset lenght:", self.__len__())
        print("img.shape", self.img.shape)
        print("img.dtype", self.img.dtype)
        print("img.device", self.img.device)
        index = torch.randperm(self.__len__(), dtype=torch.long, device=self.img.device, requires_grad=False)
        # grab one minibatch
        img, labels, index = self.__getitem__(index[:batch_size])
        print("MINIBATCH: img.shapes labels.shape, index.shape ->", img.shape, labels.shape, index.shape)
        print("MINIBATCH: min and max of minibatch", torch.min(img), torch.max(img))
        return show_batch(img, n_col=4, n_padding=4, pad_value=1, figsize=(24, 24))


def process_one_epoch(model: torch.nn.Module,
                      dataloader: SpecialDataSet,
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      weight_clipper: Optional[Callable[[None], None]] = None,
                      verbose: bool = False,
                      neptune_experiment: Optional[neptune.experiments.Experiment] = None,
                      neptune_prefix: Optional[str] = None) -> MetricMiniBatch:
    """ return a tuple with all the metrics averaged over a epoch """
    metric_accumulator = Accumulator()

    for i, (imgs, labels, index) in enumerate(dataloader):
        
        # Put data in GPU if available
        if torch.cuda.is_available() and (imgs.device == torch.device('cpu')): 
            imgs = imgs.cuda()

        metrics = model.forward(imgs_in=imgs).metrics  # the forward function returns metric and other stuff
        if verbose:
            print("i = %3d train_loss=%.5f" % (i, metrics.loss))

        # Accumulate metrics over an epoch
        with torch.no_grad():
            metric_accumulator.accumulate(source=metrics, counter_increment=len(index))

        # Only if training I apply backward
        if model.training:
            optimizer.zero_grad()
            metrics.loss.backward()  # do back_prop and compute all the gradients
            optimizer.step()  # update the parameters
        
            # apply the weight clipper
            if weight_clipper is not None:
                model.__self__.apply(weight_clipper)
                
        # Delete stuff from GPU
        # del imgs
        # del labels
        # del index
        # del metrics
        # torch.cuda.empty_cache()

    # At the end of the loop compute the average of the metrics
    with torch.no_grad():
        metric_one_epoch = metric_accumulator.get_average()
        if neptune_experiment is not None:
            log_dict_metrics(metrics=metric_one_epoch,
                             prefix=neptune_prefix,
                             experiment=neptune_experiment,
                             verbose=False)
        return MetricMiniBatch._make(metric_one_epoch.values())
