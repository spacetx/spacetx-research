import torch
import neptune
from torch.distributions.utils import broadcast_all
from typing import Union, Callable, Optional
from collections import OrderedDict

from .utilities_visualization import show_batch
from .namedtuple import DIST, MetricMiniBatch
from .utilities_neptune import log_dict_metrics


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


def _batch_mv(bmat, bvec):
    """ Performs a batched matrix-vector product, with compatible but different batch shapes.

        bmat shape (*, n, n)
        bvec shape (*, n)
        result = MatrixVectorMultiplication(bmat,bvec) of shape (*, n)

        * represents all the batched dimensions which might or might not be presents

        Very simple procedure
        b = bvec.unsqueeze(-1) -> (*, n, 1)
        c = torch.matmul(bmat, b) = (*, n, n) x (*, n , 1) -> (*, n, 1)
        result = c.squeeze(-1) -> (*, n)
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def sample_and_kl_multivariate_normal(posterior_mu: torch.Tensor,
                                      posterior_L_cov: torch.Tensor,
                                      prior_mu: torch.Tensor,
                                      prior_L_cov: torch.Tensor,
                                      noisy_sampling: bool,
                                      sample_from_prior: bool) -> DIST:

    post_L, prior_L = broadcast_all(posterior_L_cov, prior_L_cov)  # (*, n, n)
    post_mu, prior_mu = broadcast_all(posterior_mu, prior_mu)  # (*, n)
    assert post_L.shape[-1] == post_L.shape[-2] == post_mu.shape[-1]  # number of grid points are the same
    assert post_L.shape[:-2] == post_mu.shape[:-1]  # batch_size is the same

    if sample_from_prior:
        # working with the prior
        eps = torch.randn_like(prior_mu)
        sample = prior_mu + _batch_mv(prior_L, eps) if noisy_sampling else prior_mu  # size: *, n
        kl = torch.zeros_like(prior_mu[..., 0])  # size: *
    else:
        # working with the posterior
        eps = torch.randn_like(post_mu)
        sample = post_mu + _batch_mv(post_L, eps) if noisy_sampling else post_mu
        kl = kl_multivariate_normal0_normal1(mu0=post_mu, mu1=prior_mu, L_cov0=post_L, L_cov1=prior_L)
    return DIST(sample=sample, kl=kl)


def kl_multivariate_normal0_normal1(mu0: torch.Tensor,
                                    mu1: torch.Tensor,
                                    L_cov0: torch.Tensor,
                                    L_cov1: torch.Tensor) -> torch.Tensor:
    """
    Function that analytically computes the KL divergence between two MultivariateNormal distributions
    (see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence)

    Each MultivariateNormal is defined in terms of its mean and the cholesky decomposition of the covariance matrix
    :param mu0: array with mean value of posterior, size (*, n)
    :param mu1: array with mean value of prior, size (*, n)
    :param L_cov0: lower triangular matrix with the decomposition on the covariance for the posterior, size (*, n, n)
    :param L_cov1: lower triangular matrix with the decomposition on the covariance for the prior, size (*, n, n)
    :return: kl: array with kl divergence between posterior and prior, size (*)

    Note that n is the number of locations where the MultivariateNormal is evaluated,
    * represents all the batched dimensions which might or might not be presents
    """

    assert are_broadcastable(mu0, mu1)  # (*, n)
    assert are_broadcastable(L_cov0, L_cov1)  # (*, n, n)
    n = L_cov0.shape[-1]

    # Tr[cov1^(-1)cov0] = Tr[L L^T] = sum_of_element_wise_square(L)
    # where L = L1^(-1) L0 -> Solve trilinear problem: L1 L = L0
    L = torch.triangular_solve(L_cov0, A=L_cov1, upper=False, transpose=False, unitriangular=False)[0]  # (*,n,n)
    trace_term = torch.sum(L.pow(2), dim=(-1, -2))  # (*)
    # print("trace_term",trace_term.shape, trace_term)

    # x^T conv1^(-1) x = z^T z where z = L1^(-1) x -> solve trilinear problem L1 z = x
    dmu = (mu0 - mu1).unsqueeze(-1)  # (*,n,1)
    z = torch.triangular_solve(dmu, A=L_cov1, upper=False, transpose=False, unitriangular=False)[0]  # (*,n,1)
    # Now z.t*z is the sum over both n_points and dimension
    square_term = z.pow(2).sum(dim=(-1, -2))  # (*)
    # print("square_term",square_term.shape, square_term)

    # log[det(cov)]= log[det(L L^T)] = logdet(L) + logdet(L^T) = 2 logdet(L)
    # where logdet casn be computed as the sum of the diagonal elements
    logdet1 = torch.diagonal(L_cov1, dim1=-1, dim2=-2).log().sum(-1)
    logdet0 = torch.diagonal(L_cov0, dim1=-1, dim2=-2).log().sum(-1)
    logdet_term = 2 * (logdet1 - logdet0)  # (*)  factor of 2 b/c log[det(L L^T)]= 2 log[det(L)]
    # print("logdet_term",logdet_term.shape, logdet_term)

    return 0.5 * (trace_term + square_term - n + logdet_term)


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
        else:
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

    for i, data in enumerate(dataloader):
        imgs, labels, index = data
        
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
