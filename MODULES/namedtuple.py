import torch
import numpy
from typing import NamedTuple, Optional, Tuple

#  ----------------------------------------------------------------  #
#  ------- Stuff defined in terms of native types -----------------  #
#  ----------------------------------------------------------------  #


class Partition(NamedTuple):
    type: str
    membership: torch.tensor  # bg=0, fg=1,2,3,.....
    sizes: torch.tensor  # both for bg and fg. It is simply obtained by numpy.bincount(membership)
    params: dict


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


class Similarity(NamedTuple):
    data: torch.tensor  # *, nn, w, h where nn = ((2*r+1)^2 -1)//2
    ch_to_dxdy: list  # [(dx0,dy0),(dx1,dy1),....]  of lenght nn

    def reduce_similarity_radius(self, new_radius: int):

        # Check which element should be selected
        to_select = [(abs(dxdy[0]) <= new_radius and abs(dxdy[1]) <= new_radius) for dxdy in self.ch_to_dxdy]
        all_true = sum(to_select) == len(to_select)
        if all_true:
            raise Exception("new radius should be smaller than current radius. No subsetting left to do")

        # Subsample the ch_to_dxdy
        new_ch_to_dxdy = []
        for selected, dxdy in zip(to_select, self.ch_to_dxdy):
            if selected:
                new_ch_to_dxdy.append(dxdy)

        # Subsample the similarity matrix
        index = torch.arange(len(to_select), dtype=torch.long)[to_select]

        return Similarity(data=torch.index_select(self.data, dim=-3, index=index), ch_to_dxdy=new_ch_to_dxdy)

    def one_over(self):
        return self._replace(data=1.0/self.data)


#  ----------------------------------------------------------------  #
#  -------Stuff defined in term of other sutff --------------------  #
#  ----------------------------------------------------------------  #


class Segmentation(NamedTuple):
    """ Where * is the batch dimension which might be NOT present """
    raw_image: torch.Tensor  # *,ch,w,h
    fg_prob: torch.Tensor  # *,1,w,h
    integer_mask: torch.Tensor  # *,1,w,h
    bounding_boxes: Optional[torch.Tensor]  # *,3,w,h
    similarity: Similarity

    def reduce_similarity_radius(self, new_radius: int):
        return self._replace(similarity=self.similarity.reduce_similarity_radius(new_radius=new_radius))

    def one_over_similarity(self):
        return self._replace(similarity=self.similarity.one_over())


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
