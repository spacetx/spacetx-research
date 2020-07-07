import torch
import numpy
from typing import NamedTuple

#  ----------------------------------------------------------------  #
#  ------- Stuff defined in terms of native types -----------------  #
#  ----------------------------------------------------------------  #

class SimplifiedPartition(NamedTuple):
    sizes: list
    membership: list
    resolution_parameter: float
    modularity: float


class COMMUNITY(NamedTuple):
    mask: numpy.array
    n: int
    modularity: float
    resolution: float


class Adjacency(NamedTuple):
    edge_weight: list
    source: list
    destination: list


class TILING(NamedTuple):
    co_object: torch.Tensor  # NN, w, h   where NN = (2r+1)*(2*r+1)
    raw_img: torch.Tensor  # ch,w,h


class DIST(NamedTuple):
    sample: torch.Tensor
    kl: torch.Tensor


class ZZ(NamedTuple):
    mu: torch.Tensor
    std: torch.Tensor


class BB(NamedTuple):
    bx: torch.Tensor  # shape: n_box, batch_size
    by: torch.Tensor
    bw: torch.Tensor
    bh: torch.Tensor


class NMSoutput(NamedTuple):
    nms_mask: torch.Tensor
    index_top_k: torch.Tensor


class Checkpoint(NamedTuple):
    epoch: int
    hyperparams_dict: dict
    history_dict: dict

#  ----------------------------------------------------------------  #
#  -------Stuff defined in term of other sutff --------------------  #
#  ----------------------------------------------------------------  #

class UNEToutput(NamedTuple):
    zwhere: ZZ
    logit: ZZ
    zbg: ZZ
    features: torch.Tensor


class Inference(NamedTuple):
    length_scale_GP: torch.Tensor
    p_map: torch.Tensor
    area_map: torch.Tensor
    big_bg: torch.Tensor
    big_img: torch.Tensor
    big_mask: torch.Tensor
    big_mask_NON_interacting: torch.Tensor
    prob: torch.Tensor
    bounding_box: BB
    kl_zinstance_each_obj: torch.Tensor
    kl_zwhere_map: torch.Tensor
    kl_logit_map: torch.Tensor


class MetricMiniBatch(NamedTuple):
    loss: torch.Tensor
    nll: torch.Tensor
    reg: torch.Tensor
    kl_tot: torch.Tensor
    kl_instance: torch.Tensor
    kl_where: torch.Tensor
    kl_logit: torch.Tensor
    sparsity: torch.Tensor
    fg_fraction: torch.Tensor
    geco_sparsity: torch.Tensor
    geco_balance: torch.Tensor
    delta_1: torch.Tensor
    delta_2: torch.Tensor
    length_GP: torch.Tensor
    n_obj_counts: torch.Tensor


class RegMiniBatch(NamedTuple):
    # cost_fg_pixel_fraction: torch.Tensor
    cost_overlap: torch.Tensor
    cost_vol_absolute: torch.Tensor
    # cost_volume_mask_fraction: torch.Tensor
    # cost_prob_map_integral: torch.Tensor
    # cost_prob_map_fraction: torch.Tensor
    # cost_prob_map_TV: torch.Tensor


class Metric_and_Reg(NamedTuple):
    # MetricMiniBatch (in the same order as underlying class)
    loss: torch.Tensor
    nll: torch.Tensor
    reg: torch.Tensor
    kl_tot: torch.Tensor
    kl_instance: torch.Tensor
    kl_where: torch.Tensor
    kl_logit: torch.Tensor
    sparsity: torch.Tensor
    fg_fraction: torch.Tensor
    geco_sparsity: torch.Tensor
    geco_balance: torch.Tensor
    delta_1: torch.Tensor
    delta_2: torch.Tensor
    length_GP: torch.Tensor
    n_obj_counts: torch.Tensor
    # RegMiniBatch (in the same order as underlying class)
    # cost_fg_pixel_fraction: torch.Tensor
    cost_overlap: torch.Tensor
    cost_vol_absolute: torch.Tensor
    # cost_volume_mask_fraction: torch.Tensor
    # cost_prob_map_integral: torch.Tensor
    # cost_prob_map_fraction: torch.Tensor
    # cost_prob_map_TV: torch.Tensor

    @classmethod
    def from_merge(cls, metrics, regularizations):
        assert isinstance(metrics, MetricMiniBatch) and isinstance(regularizations, RegMiniBatch)
        assert cls._fields == MetricMiniBatch._fields + RegMiniBatch._fields
        return cls._make([*metrics, *regularizations])


class Output(NamedTuple):
    metrics: Metric_and_Reg
    inference: Inference
    imgs: torch.Tensor
