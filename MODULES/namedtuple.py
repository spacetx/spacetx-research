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
    mean_IoU: float
    mean_IoU_reverse: float


class Partition(NamedTuple):
    type: str
    membership: torch.tensor  # bg=0, fg=1,2,3,.....
    sizes: torch.tensor  # both for bg and fg. It is simply obtained by numpy.bincount(membership)
    params: dict

    def filter_by_size(self, min_size: int):
        """ If a cluster is too small, its label is set to zero (i.e. background value).
            The subsequent labels are shifted so that there are no gaps in the labels number.
        """
        my_filter = self.sizes > min_size
        count = torch.cumsum(my_filter, dim=-1)
        old_2_new = (count - count[0]) * my_filter  # this makes sure that label 0 is always mapped to 0

        new_dict = self.params
        new_dict["filter_by_size"] = min_size
        new_membership = old_2_new[self.membership]
        return self._replace(membership=new_membership, params=new_dict, sizes=torch.bincount(new_membership))

    def concordance_with_partition(self, other_partition) -> Concordance:
        """ Compute Mutual_Information.
            From the peaks of the join distribution extract the mapping between membership labels.
            Compute Intersection over Union
        """

        assert self.membership.shape == other_partition.membership.shape
        nx = len(other_partition.sizes)
        px = other_partition.sizes.float() / torch.sum(other_partition.sizes)

        ny = len(self.sizes)
        py = self.sizes.float() / torch.sum(self.sizes)

        pxy = torch.zeros((nx, ny), dtype=torch.float, device=self.membership.device)
        for ix in range(0, nx):
            counts = torch.bincount(self.membership[other_partition.membership == ix])
            pxy[ix, :counts.shape[0]] = counts
        pxy /= torch.sum(pxy)

        # Compute the mutual information
        log_term = torch.log(pxy) - torch.log(px.view(-1, 1) * py.view(1, -1))
        log_term[pxy == 0] = 0
        mutual_information = torch.sum(pxy * log_term).item()

        # Extract the most likely mapping and compute the two types of Intersection_over_Union
        # The asymmetry to IoU is due to the fact that mappings target_to_trial and
        # trial_to_target might be one to many
        target_to_trial = torch.max(pxy, dim=-1)[1]
        trial_to_target = torch.max(pxy, dim=-2)[1]

        same = (target_to_trial[other_partition.membership.long()] == self.membership)
        IoU = torch.zeros(nx, dtype=torch.float, device=self.membership.device)
        for ix in range(0, nx):
            intersection = torch.sum(same[other_partition.membership == ix]).float()
            union = other_partition.sizes[ix] + self.sizes[target_to_trial[ix]] - intersection
            IoU[ix] = intersection / union
        #IoU_avg = torch.sum(IoU*other_partition.sizes)/torch.sum(other_partition.sizes)
        IoU_avg = torch.mean(IoU)

        same_reverse = other_partition.membership == (trial_to_target[self.membership.long()])
        IoU_reverse = torch.zeros(ny, dtype=torch.float, device=self.membership.device)
        for iy in range(0, ny):
            intersection = torch.sum(same_reverse[self.membership == iy]).float()
            union = other_partition.sizes[trial_to_target[iy]] + self.sizes[iy] - intersection
            IoU_reverse[iy] = intersection / union
        #IoU_reverse_avg = torch.sum(IoU_reverse*self.sizes)/torch.sum(self.sizes)
        IoU_reverse_avg = torch.mean(IoU_reverse)



        return Concordance(joint_distribution=pxy,
                           mutual_information=mutual_information,
                           delta_n=ny - nx,
                           mean_IoU=IoU_avg.item(),
                           mean_IoU_reverse=IoU_reverse_avg.item())


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
