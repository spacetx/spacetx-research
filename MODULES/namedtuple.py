import torch
import numpy
from typing import NamedTuple, Optional, Tuple

#  ----------------------------------------------------------------  #
#  ------- Stuff defined in terms of native types -----------------  #
#  ----------------------------------------------------------------  #


class Concordance(NamedTuple):
    joint_distribution: torch.tensor
    mutual_information: float
    delta_n: int
    iou: float


class Partition(NamedTuple):
    type: str
    membership: torch.tensor  # bg=0, fg=1,2,3,.....
    sizes: torch.tensor  # both for bg and fg. It is simply obtained by numpy.bincount(membership)
    params: dict

    def filter_by_size(self, min_size: Optional[int] = None, max_size: Optional[int] = None):
        """ If a cluster is too small or too large, its label is set to zero (i.e. background value).
            The other labels are adjusted so that there are no gaps in the labels number.
            Min_size and Max_size are integers specifying the number of pixels.
        """
        if (min_size is None) and (max_size is None):
            raise Exception("At least one among min_size and max_size should be specified")
        if (min_size is not None) and (max_size is not None):
            assert max_size > min_size > 0, "Condition max_size > min_size > 0 failed."

        my_filter = (self.sizes > min_size) * (self.sizes < max_size)
        count = torch.cumsum(my_filter, dim=-1)
        old_2_new = (count - count[0]) * my_filter  # this makes sure that label 0 is always mapped to 0

        new_dict = self.params
        new_dict["filter_by_size"] = min_size
        new_membership = old_2_new[self.membership]
        return self._replace(membership=new_membership, params=new_dict, sizes=torch.bincount(new_membership))

    def concordance_with_partition(self, other_partition) -> Concordance:
        """ Compute measure of concordance between two partitions:
            joint_distribution
            mutual_information
            delta_n
            iou

            We use the peaks of the join distribution to extract the mapping between membership labels.
        """

        assert self.membership.shape == other_partition.membership.shape
        nx = len(other_partition.sizes)
        ny = len(self.sizes)

        pxy = torch.zeros((nx, ny), dtype=torch.float, device=self.membership.device)
        for ix in range(0, nx):
            counts = torch.bincount(self.membership[other_partition.membership == ix])
            pxy[ix, :counts.shape[0]] = counts.float()
        pxy /= torch.sum(pxy)
        px = torch.sum(pxy, dim=-1)
        py = torch.sum(pxy, dim=-2)

        # Compute the mutual information
        term_xy = pxy * torch.log(pxy)
        term_x = px * torch.log(px)
        term_y = py * torch.log(py)
        mutual_information = term_xy[pxy > 0].sum() - term_x[px > 0].sum() - term_y[py > 0].sum()

        # Extract the most likely mappings
        # print(nx, ny)
        to_other = torch.max(pxy, dim=-2)[1]
        from_other = torch.max(pxy, dim=-1)[1]

        # Find one-to-one correspondence among instance IDs
        original_instance_id = torch.arange(ny, device=self.membership.device, dtype=torch.long)
        is_id_one_to_one = (from_other[to_other[original_instance_id]] == original_instance_id)

        # Define a mapping that changes all bad (i.e. not unique or background) instance IDs to -1
        original_to_good_id = original_instance_id
        original_to_good_id[~is_id_one_to_one] = -1
        original_to_good_id[0] = -1  # Exclude the background
        # print(original_to_good_id)

        # compute the intersection among all fg_pixels_with_unique_id
        pixel_with_same_id = (to_other[self.membership] == other_partition.membership)
        fg_pixels_with_unique_id = (original_to_good_id[self.membership] > 0)
        intersection = torch.sum(pixel_with_same_id[fg_pixels_with_unique_id])
        union = torch.sum(self.sizes[1:]) + torch.sum(other_partition.sizes[1:]) - intersection  # exclude background
        iou = intersection.float()/union

        return Concordance(joint_distribution=pxy,
                           mutual_information=mutual_information.item(),
                           delta_n=ny - nx,
                           iou=iou.item())


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
