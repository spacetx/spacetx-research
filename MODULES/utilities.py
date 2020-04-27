import torch
import json
from PIL import Image as pilimage
from PIL import ImageDraw as pilimagedraw
import pickle
import numpy as np
from torchvision import utils
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
from torch.distributions.utils import broadcast_all
from typing import Union, Callable, Optional, List, Tuple
from .namedtuple import BB, DIST


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_json_as_dict(path):
    with open(path, 'rb') as f:
        return json.load(f)


def save_dict_as_json(my_dict, path):
    with open(path, 'w') as f:
        return json.dump(my_dict, f)


def reset_parameters(parent_module, verbose):
    for m in parent_module.modules():
        try:
            m.reset_parameters()
            if verbose:
                print("reset -> ", m)
        except AttributeError:
            pass


def are_boradcastable(a: torch.Tensor, b: torch.Tensor) -> bool:
    """ Return True if tensor are broadcastable to each other, False otherwise """
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(a.shape[::-1], b.shape[::-1]))


def append_dict_to_dict(source, target, prefix_include=None, prefix_exclude=None, prefix_to_add=None):
    """ Use typing.
        For now: prefix_include is str or tuple of str
        For now: prefix_exclude is str or tuple of str
        For now: prefix_to_add is str """

    for k, v in source.items():

        if (prefix_include is None or k.startswith(prefix_include)) and (prefix_exclude is None or
                                                                         not k.startswith(prefix_exclude)):
            new_k = k if prefix_to_add is None else prefix_to_add+k
            try:
                target[new_k].append(v)
            except KeyError:
                target[new_k] = [v]

    return target


def sample_and_kl_diagonal_normal(posterior_mu: torch.Tensor,
                                  posterior_std: torch.Tensor,
                                  prior_mu: torch.Tensor,
                                  prior_std: torch.Tensor,
                                  noisy_sampling: bool = True,
                                  sample_from_prior: bool = False) -> DIST:

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
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def sample_and_kl_multivariate_normal(posterior_mu: torch.Tensor,
                                      posterior_L_cov: torch.Tensor,
                                      prior_mu: torch.Tensor,
                                      prior_L_cov: torch.Tensor,
                                      noisy_sampling: bool = True,
                                      sample_from_prior: bool = False) -> DIST:

    post_L, prior_L = broadcast_all(posterior_L_cov, prior_L_cov)  # (*, n, n)
    post_mu, prior_mu = broadcast_all(posterior_mu, prior_mu)  # (*, n)
    assert post_L.shape[-1] == post_L.shape[-2] == post_mu.shape[-1]
    assert post_L.shape[:-2] == post_mu.shape[:-1]

    if sample_from_prior:
        # working with the prior
        eps = torch.randn_like(prior_mu)
        sample = prior_mu + _batch_mv(prior_L, eps) if noisy_sampling else prior_mu
        kl = torch.zeros_like(prior_mu)
    else:
        # working with the posterior
        eps = torch.randn_like(post_mu)
        sample = post_mu + _batch_mv(post_L, eps) if noisy_sampling else post_mu
        kl = kl_multivariate_normal0_normal1(mu0=post_mu, mu1=prior_mu, L_cov0=post_L, L_cov1=prior_L)
    return DIST(sample=sample, kl=kl)


# def sample_normal(mu: Union[float, torch.Tensor],
#                   std: Union[float, torch.Tensor],
#                   noisy_sampling: bool) -> Union[float, torch.Tensor]:
#     new_mu, new_std = broadcast_all(mu, std)
#     if noisy_sampling:
#         return new_mu + new_std * torch.randn_like(new_mu)
#     else:
#         return new_mu
#
#
# def kl_normal0_normal1(mu0: Union[float, torch.Tensor],
#                        mu1: Union[float, torch.Tensor],
#                        std0: Union[float, torch.Tensor],
#                        std1: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
#     tmp = (std0 + std1) * (std0 - std1) + (mu0 - mu1).pow(2)
#     return tmp / (2 * std1 * std1) - torch.log(std0 / std1)


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

    assert are_boradcastable(mu0, mu1)  # (*, n)
    assert are_boradcastable(L_cov0, L_cov1)  # (*, n, n)
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


def linear_interpolation(t: Union[np.array, float], values: tuple, times: tuple) -> Union[np.array, float]:
    """ Makes an interpolation between (t_in,v_in) and (t_fin,v_fin)
        For time t>t_fin and t<t_in the value of v is clamped to either v_in or v_fin
        Usage:
        epoch = np.arange(0,100,1)
        v = linear_interpolation(epoch, values=[0.0,0.5], times=[20,40])
        plt.plot(epoch,v)
    """
    v_in, v_fin = values  # initial and final values
    t_in, t_fin = times   # initial and final times

    if t_fin >= t_in:
        den = max(t_fin-t_in, 1E-8)
        m = (v_fin-v_in)/den
        v = v_in + m*(t-t_in)
    else:
        raise Exception("t_fin should be greater than t_in")

    v_min = min(v_in, v_fin)
    v_max = max(v_in, v_fin)
    return np.clip(v, v_min, v_max)


def accumulate_counting_accuracy(indices_wrong_examples: list,
                                 indices_right_examples: list,
                                 dict_accuracy: dict) -> dict:
    dict_accuracy["wrong_examples"] = indices_wrong_examples + dict_accuracy.get("wrong_examples", [])  # concat lists
    dict_accuracy["right_examples"] = indices_right_examples + dict_accuracy.get("right_examples", [])  # concat lists
    return dict_accuracy


def flatten_list(my_list: List[List]) -> list:
    flat_list: list = []
    for sublist in my_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list


class DatasetInMemory(Dataset):
    """ Typical usage:

        synthetic_data_test  = dataset_in_memory(root_dir,"synthetic_data_DISK_test_v1",use_cuda=False)
        for epoch in range(5):
            print("EPOCH")
            batch_iterator = synthetic_data_test.generate_batch_iterator(batch_size)
            for i, x in enumerate(batch_iterator):
                print(x)
                blablabla
                ......
    """

    def __init__(self, path, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda

        with torch.no_grad():
            # this is when I have both imgs and labels
            imgs, labels = load_obj(path)
            self.n = imgs.shape[0]

            if self.use_cuda:
                self.data = imgs.cuda()
                self.labels = labels.cuda()
            else:
                self.data = imgs.cpu().detach()
                self.labels = labels.cpu().detach()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.data[index, ...], self.labels[index]

    def load(self, batch_size=8, indices=None):
        if indices is None:
            indices = torch.randint(low=0, high=self.n, size=(batch_size,)).long()
        return self.__getitem__(indices[:batch_size])

    def generate_batch_iterator(self, batch_size):
        indices = torch.randperm(self.n).numpy()
        remainder = len(indices) % batch_size
        # n_max = len(indices) - remainder  # Note the trick so that all the minibatches have the same size
        n_max = len(indices)
        assert (n_max >= batch_size), "batch_size is too big for this dataset of size."
        batch_iterator = (indices[pos:pos + batch_size] for pos in range(0, n_max, batch_size))
        return batch_iterator

    def check(self):

        imgs, labels = self.load(batch_size=8)
        title = "# labels =" + str(labels.cpu().numpy().tolist())

        print("Dataset lenght:", self.__len__())
        print("imgs.shape", imgs.shape)
        print("type(imgs)", type(imgs))
        print("imgs.device", imgs.device)
        print("torch.max(imgs)", torch.max(imgs))
        print("torch.min(imgs)", torch.min(imgs))

        return show_batch(imgs, n_col=4, n_padding=4, title=title, pad_value=1)


def process_one_epoch(model: torch.nn.Module,
                      data_set: DatasetInMemory,
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      weight_clipper: Optional[Callable[[None], None]] = None,
                      batch_size: int = 64,
                      verbose: bool = False) -> dict:
    """ return a dictionary with all the metrics """
    n_term: int = 0
    dict_accumulate_accuracy: dict = {}
    dict_metric_av: dict = {}
    batch_iterator = data_set.generate_batch_iterator(batch_size)

    for i, indices in enumerate(batch_iterator):  # get the indices

        # get the data from the indices. Note that everything is already loaded into memory
        imgs_in, labels = data_set[indices]
        metrics = model.forward(imgs_in=imgs_in).metrics  # the forward function returns metric and other stuff
        if verbose:
            print("i = %3d train_loss=%.5f" % (i, metrics.loss))

        # Only if training I apply backward
        if model.training:
            optimizer.zero_grad()
            metrics.loss.backward()  # do back_prop and compute all the gradients
            optimizer.step()  # update the parameters

            # apply the weight clipper
            if weight_clipper is not None:
                model.__self__.apply(weight_clipper)

        # Accumulate over an epoch
        with torch.no_grad():

            # Accumulate metrics
            n_term += len(indices)
            for key in metrics._fields:
                if key == 'n_obj_counts':
                    counts = getattr(metrics, 'n_obj_counts').view_as(labels)
                else:
                    value = getattr(metrics, key).item() * len(indices)
                    dict_metric_av[key] = value + dict_metric_av.get(key, 0.0)

            # Accumulate counting accuracy
            index_wrong_tmp = (labels != counts).cpu()
            index_right_tmp = (labels == counts).cpu()
            indices_wrong_examples = indices[index_wrong_tmp].tolist()
            indices_right_examples = indices[index_right_tmp].tolist()
            dict_accumulate_accuracy = accumulate_counting_accuracy(indices_wrong_examples=indices_wrong_examples,
                                                                    indices_right_examples=indices_right_examples,
                                                                    dict_accuracy=dict_accumulate_accuracy)

    # At the end of the loop compute the average of the metrics
    with torch.no_grad():

        # compute the average of the metrics
        for k, v in dict_metric_av.items():
            dict_metric_av[k] = v / n_term

        # compute the accuracy
        n_right = len(dict_accumulate_accuracy["right_examples"])
        n_wrong = len(dict_accumulate_accuracy["wrong_examples"])
        dict_metric_av["accuracy"] = float(n_right) / (n_right + n_wrong)
        dict_metric_av["wrong_examples"] = dict_accumulate_accuracy["wrong_examples"]

        # join the two dictionary together
        return dict_metric_av


def show_batch(images: torch.Tensor,
               n_col: int = 4,
               n_padding: int = 10,
               title: Optional[str] = None,
               pad_value: int = 1):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    assert len(images.shape) == 4  # batch, ch, width, height
    if images.device != "cpu":
        images = images.cpu()
    grid = utils.make_grid(images, n_col, n_padding, normalize=True, range=(0.0, 1.0),
                           scale_each=False, pad_value=pad_value)
        
    fig = plt.figure()
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
    if isinstance(title, str):
        plt.title(title)
    plt.close(fig)
    fig.tight_layout()
    return fig


def compute_ranking(x: torch.Tensor) -> torch.Tensor:
    """ Given a vector of shape: n, batch_size
        For each batch dimension it ranks the n elements"""
    assert len(x.shape) == 2
    n, batch_size = x.shape
    _, order = torch.sort(x, dim=-2, descending=False)

    # this is the fast way which uses indexing on the left
    rank = torch.zeros_like(order)
    batch_index = torch.arange(batch_size, dtype=order.dtype, device=order.device).view(1, -1).expand(n, batch_size)
    rank[order, batch_index] = torch.arange(n, dtype=order.dtype, device=order.device).view(-1, 1).expand(n, batch_size)
    return rank


def compute_average_intensity_in_box(imgs: torch.Tensor, bounding_box: BB) -> torch.Tensor:
    """ Input batch of images: batch_size x ch x w x h
        z_where collections of [bx,by,bw,bh]
        bx.shape = batch x n_box
        similarly for by,bw,bh
        Output:
        av_intensity = n_box x batch_size
    """
    # cumulative sum in width and height, standard sum in channels
    cum = torch.sum(torch.cumsum(torch.cumsum(imgs, dim=-1), dim=-2), dim=-3)
    assert len(cum.shape) == 3
    batch_size, w, h = cum.shape

    # compute the x1,y1,x3,y3
    x1 = torch.clamp((bounding_box.bx - 0.5 * bounding_box.bw).long(), min=0, max=w - 1)
    x3 = torch.clamp((bounding_box.bx + 0.5 * bounding_box.bw).long(), min=0, max=w - 1)
    y1 = torch.clamp((bounding_box.by - 0.5 * bounding_box.bh).long(), min=0, max=h - 1)
    y3 = torch.clamp((bounding_box.by + 0.5 * bounding_box.bh).long(), min=0, max=h - 1)
    assert x1.shape == x3.shape == y1.shape == y3.shape

    # compute the area
    # Note that this way penalizes boxes that go out-of-bound
    # This is in contrast to area = (x3-x1)*(y3-y1) which does NOT penalize boxes out of bound
    area = bounding_box.bw * bounding_box.bh
    assert area.shape == x1.shape == x3.shape == y1.shape == y3.shape
    n_boxes, batch_size = area.shape

    # compute the total intensity in each box
    b_index = torch.arange(start=0, end=batch_size, step=1, device=x1.device,
                           dtype=x1.dtype).view(1, -1).expand(n_boxes, -1)
    assert b_index.shape == x1.shape

    tot_intensity = cum[b_index, x3, y3] + cum[b_index, x1, y1] - cum[b_index, x1, y3] - cum[b_index, x3, y1]

    # return the average intensity
    assert tot_intensity.shape == x1.shape
    return tot_intensity / area


def draw_bounding_boxes(prob: Optional[torch.Tensor], bounding_box: BB, width: int, height: int) -> torch.Tensor:

    # set all prob to one if they are not passed as input
    if prob is None:
        prob = torch.ones_like(bounding_box.bx)

    # checks
    assert prob.shape == bounding_box.bx.shape
    assert len(bounding_box.bx.shape) == 2
    n_boxes, batch_size = bounding_box.bx.shape

    # prepare the storage
    batch_bb_np = np.zeros((batch_size, width, height, 3))  # numpy storage for bounding box images

    # compute the coordinates of the bounding boxes and the probability of each box
    x1 = bounding_box.bx - 0.5 * bounding_box.bw
    x3 = bounding_box.bx + 0.5 * bounding_box.bw
    y1 = bounding_box.by - 0.5 * bounding_box.bh
    y3 = bounding_box.by + 0.5 * bounding_box.bh
    assert x1.shape == x3.shape == y1.shape == y3.shape  # n_boxes, batch_size
    x1y1x3y3 = torch.stack((x1, y1, x3, y3), dim=-1)

    # draw the bounding boxes
    for batch in range(batch_size):

        # Draw on PIL
        img = pilimage.new('RGB', (width, height), color=0)
        draw = pilimagedraw.Draw(img)
        for box in range(n_boxes):
            # if prob[box, batch] > -1:
            if prob[box, batch] > 0.5:
                draw.rectangle(x1y1x3y3[box, batch, :].cpu().numpy(), outline='red', fill=None)
        batch_bb_np[batch, ...] = np.array(img.getdata(), np.uint8).reshape((width, height, 3))

    # Transform np to torch, rescale from [0,255] to (0,1)
    batch_bb_torch = torch.from_numpy(batch_bb_np).permute(0, 3, 2, 1).float() / 255  # permute(0,3,2,1) is CORRECT
    return batch_bb_torch.to(bounding_box.bx.device)


def sample_from_constraints_dict(dict_soft_constraints: dict,
                                 var_name: str,
                                 var_value: Union[float, torch.Tensor],
                                 verbose: bool = False,
                                 chosen: Optional[int] = None) -> Union[float, torch.Tensor]:

    if isinstance(var_value, torch.Tensor):
        cost: torch.Tensor = torch.zeros_like(var_value)
    elif isinstance(var_value, float):
        cost: float = 0.0
    else:
        raise Exception

    var_constraint_params = dict_soft_constraints[var_name]

    if 'lower_bound_value' in var_constraint_params:
        left = var_constraint_params['lower_bound_value']
        width_low = var_constraint_params['lower_bound_width']
        exponent_low = var_constraint_params['lower_bound_exponent']
        strength_low = var_constraint_params['lower_bound_strength']
        activity_low = torch.clamp(left + width_low - var_value, min=0.) / width_low
        cost += strength_low * activity_low.pow(exponent_low)

    if 'upper_bound_value' in var_constraint_params:
        right = var_constraint_params['upper_bound_value']
        width_up = var_constraint_params['upper_bound_width']
        exponent_up = var_constraint_params['upper_bound_exponent']
        strength_up = var_constraint_params['upper_bound_strength']
        activity_up = torch.clamp(var_value - right + width_up, min=0.) / width_up
        cost += strength_up * activity_up.pow(exponent_up)

    if 'strength' in var_constraint_params:
        strength = var_constraint_params['strength']
        exponent = var_constraint_params['exponent']
        cost += strength * var_value.pow(exponent)

    if verbose:
        if chosen is None:
            print("constraint name ->", var_name)
            print("input value ->", var_value)
            print("cost ->", cost)
        else:
            print("constraint name ->", var_name)
            print("input value ->", var_value[..., chosen])
            print("cost ->", cost[..., chosen])

    return cost


class ConstraintBase(object):
    """ Base Constraints which implement the identity """
    def __init__(self):
        super().__init__()

    def to_unconstrained(self, value):
        return value

    def to_constrained(self, value):
        return value


class ConstraintBounded(ConstraintBase):
    def __init__(self, lower_bound: Union[float, torch.Tensor], upper_bound: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.delta = self.upper_bound - self.lower_bound

    def to_unconstrained(self, value):
        """ takes value in (lower_bound, upper_bound) and returns in (-Infinity, Infinity) """
        assert ((self.lower_bound <= value) & (value <= self.upper_bound)).all()
        x0 = (value - self.lower_bound) / self.delta  # this is in (0,1)
        x1 = 2 * x0 - 1  # this is in (-1,1)
        return torch.log1p(x1) - torch.log1p(-x1)  # when x1=+/- 1 result is +/- Infinity

    def to_constrained(self, value):
        """ takes value in (-Infinity, Infinity) and returns in (lower_bound, upper_bound) """
        return torch.sigmoid(value) * self.delta + self.lower_bound


class ConstrainedParam(torch.nn.Module):
    def __init__(self, initial_data: torch.Tensor, transformation: ConstraintBase = ConstraintBase):
        super().__init__()
        self.transformation = transformation
        with torch.no_grad():
            init_unconstrained = self.transformation.to_unconstrained(initial_data.detach())
        self.unconstrained = torch.nn.Parameter(init_unconstrained, requires_grad=True)

    def forward(self):
        # return constrained, self.unconstrained
        return self.transformation.to_constrained(self.unconstrained)



#####def param_constraint(module: torch.nn.Module,
#####                     name: str,
#####                     data: torch.Tensor,
#####                     transformation: ConstraintBase = ConstraintBase) -> Tuple[torch.Tensor, torch.Tensor]:
#####    """ define a constrained parameters inside a torch.module and return both constrained and unconstrained values """
#####
#####    # Make sure that a parameters with that name was initialized during the module initialization.
#####    # This is to guaranteed that the parameters is registered with the optimizer
#####    assert name in module._parameters
#####
#####    # Look for name in transformation_store which might not exist yet
#####    try:
#####        found = name in module.transformation_store
#####    except AttributeError:
#####        module.transformation_store = {}
#####        found = False
#####
#####    # If not found save both transformation and the unconstrained value in two dictionaries which live on the module
#####    if not found:
#####        module.transformation_store[name] = transformation
#####        with torch.no_grad():
#####            data_unconstrained = transformation.to_unconstrained(data.detach())
#####        # next line will automatically register the parameters in the module._parameters dictionary
#####        setattr(module, name, torch.nn.Parameter(data_unconstrained, requires_grad=True))
#####
#####    # Read transformation and unconstrained_values from dictionary
#####    transformation = module.transformation_store[name]
#####    unconstrained_value = getattr(module, name)
#####
#####    # Transform from unconstrained to constrained
#####    constrained_value = transformation.to_constrained(unconstrained_value)
#####    return constrained_value, unconstrained_value


###class MSE_learn_sigma(torch.nn.Module):
###    """ MSE which learns the right value for sigma unless sigma is passed in which case that external value is used
###        I use the expression: sigma = e^x + eps to guarantee that sigma >= eps.
###        x is the unconstrained torch.parameter
###    """
###    def __init__(self,
###                 initial_value: torch.Tensor,
###                 eps: Union[float, torch.Tensor] = 1E-3):
###        super().__init__()
###        self.eps = eps * torch.ones_like(initial_value)
###        self.unconstrained = torch.nn.Parameter(torch.log(initial_value), requires_grad=True)
###
###    def __get_sigma__(self):
###        # sigma = e^x + eps
###        return self.unconstrained.exp_() + self.eps
###
###    def __get_log_sigma__(self):
###        # log(sigma) = log(e^x + eps) = x + log(1+ eps/e^x) = x + log1p(eps/e^x)
###        f = self.eps / self.unconstrained.exp_()
###        return self.unconstrained + torch.log1p(f)
###
###    def forward(self,
###                output: torch.Tensor,
###                target: torch.Tensor,
###                sigma: Union[float, torch.Tensor, None] = None) -> torch.Tensor:
###
###        if sigma is None:
###
###            # If nothing is passed use self.sigma and make sure to learn it
###            if self.unconstrained.device != output.device:
###                self.unconstrained = self.unconstrained.to(output.device)
###                self.eps = self.eps.to(output.device)
###
###            sigma = self.__get_sigma__()
###            log_sigma = self.__get_log_sigma__()
###            return ((output-target)/sigma).pow(2) + 2*log_sigma - 2*log_sigma.detach()
###
###        else:
###
###            # Use whatever it was passed
###            return ((output - target) / sigma).pow(2)
