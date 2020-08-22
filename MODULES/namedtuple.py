import torch
import numpy
from typing import NamedTuple, Optional
import skimage.color
import matplotlib.pyplot as plt

#  ----------------------------------------------------------------  #
#  ------- Stuff Related to PreProcessing -------------------------  #
#  ----------------------------------------------------------------  #


class ImageBbox(NamedTuple):
    """ Follows Scikit Image convention. Pixels belonging to the bounding box are in the half-open interval:
        [min_row;max_row) and [min_col;max_col). """
    min_row: int
    min_col: int
    max_row: int
    max_col: int


class PreProcess(NamedTuple):
    img: torch.Tensor
    roi_mask: torch.Tensor
    bbox_original: ImageBbox
    bbox_crop: ImageBbox

#  --------------------------------------------------------------------------------------  #
#  ------- Stuff Related to PostProcessing (i.e. Graph Clustering Based on Modularity) --  #
#  --------------------------------------------------------------------------------------  #


class Suggestion(NamedTuple):
    best_resolution: float
    best_index: int
    sweep_resolution: numpy.ndarray
    sweep_mi: numpy.ndarray
    sweep_iou: numpy.ndarray
    sweep_delta_n: numpy.ndarray
    sweep_seg_mask: numpy.ndarray
    sweep_n_cells: numpy.ndarray
    sweep_sizes: list
        
    def show_index(self, index: int, figsize: tuple = (20, 20), fontsize: int = 20):
        figure, ax = plt.subplots(figsize=figsize)
        ax.imshow(skimage.color.label2rgb(label=self.sweep_seg_mask[index], bg_label=0))
        ax.set_title('resolution = {0:.3f}, \
                      iou = {1:.3f}, \
                      delta_n = {2:3d}, \
                      n_cells = {3:3d}'.format(self.sweep_resolution[index],
                                               self.sweep_iou[index],
                                               self.sweep_delta_n[index],
                                               self.sweep_n_cells[index]),
                     fontsize=fontsize)
        
    def show_best(self, figsize: tuple = (20, 20), fontsize: int = 20):
        return self.show_index(self.best_index, figsize, fontsize)
        
    def show_graph(self, figsize: tuple = (20, 20), fontsize: int = 20):
        figure, ax = plt.subplots(figsize=figsize)
        ax.set_title('Resolution sweep', fontsize=fontsize)
        ax.set_xlabel("resolution", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        
        color = 'tab:red'
        _ = ax.plot(self.sweep_resolution, self.sweep_n_cells, '.--', label="n_cell", color=color)
        ax.set_ylabel('n_cell', color=color, fontsize=fontsize)
        ax.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
        ax.legend(loc='upper left', fontsize=fontsize)
        ax.grid()

        ax_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        _ = ax_2.plot(self.sweep_resolution, self.sweep_iou, '-', label="iou", color=color)
        ax_2.set_ylabel('Intersection Over Union', color=color, fontsize=fontsize)
        ax_2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
        ax_2.legend(loc='upper right', fontsize=fontsize)


class ConcordancePartition(NamedTuple):
    joint_distribution: torch.tensor
    mutual_information: float
    delta_n: int
    iou: float
    n_reversible_mapping: int


class Partition(NamedTuple):
    sizes: torch.tensor  # both for bg and fg. It is simply obtained by numpy.bincount(membership)
    membership: torch.tensor  # bg=0, fg=1,2,3,.....

    def compactify(self):
        """ if there are gaps in the SIZES, then shift both membership and sizes accordingly"""
        if (self.sizes[1:] > 0).all():
            return self
        else:
            my_filter = self.sizes > 0
            my_filter[0] = True
            count = torch.cumsum(my_filter, dim=-1)
            old_2_new = ((count - count[0]) * my_filter).to(self.membership.dtype)
            return Partition(sizes=self.sizes[my_filter], membership=old_2_new[self.membership])

    def filter_by_vertex(self, keep_vertex: torch.tensor):
        assert (~keep_vertex).any()  # check that there is at least one vertex to drop
        assert self.membership.shape == keep_vertex.shape
        assert keep_vertex.dtype == torch.bool
            
        """ Put all the bad vertices in the background cluster """
        if keep_vertex.all():
            return self
        else:
            new_membership = self.membership * keep_vertex
            return Partition(sizes=torch.bincount(new_membership),
                             membership=new_membership).compactify()

    def filter_by_size(self,
                       min_size: Optional[int] = None,
                       max_size: Optional[int] = None):
        """ If a cluster is too small or too large, its label is set to zero (i.e. background value).
            The other labels are adjusted so that there are no gaps in the labels number.
            Min_size and Max_size are integers specifying the number of pixels.
        """
        if (min_size is None) and (max_size is None):
            return self
        elif (min_size is not None) and (max_size is not None):
            assert max_size > min_size > 0, "Condition max_size > min_size > 0 failed."
            my_filter = (self.sizes > min_size) * (self.sizes < max_size)
        elif min_size is not None:
            assert min_size > 0, "Condition min_size > 0 failed."
            my_filter = (self.sizes > min_size)
        elif max_size is not None:
            assert max_size > 0, "Condition max_size > 0 failed."
            my_filter = (self.sizes < max_size)
        else:
            raise Exception("you should never be here!!")

        my_filter[0] = True  # always keep the bg
        if my_filter.all():
            return self
        else:
            return Partition(sizes=self.sizes * my_filter,
                             membership=self.membership).compactify()

    def concordance_with_partition(self, other_partition) -> ConcordancePartition:
        """ Compute measure of concordance between two partitions:
            joint_distribution
            mutual_information
            delta_n
            iou

            We use the peaks of the join distribution to extract the mapping between membership labels.
        """

        assert self.membership.shape == other_partition.membership.shape

        # Use z = x + y * nx
        # whose reciprocal is:
        # x = z % nx
        # y = z // nx

        nx = self.sizes.shape[0]
        ny = other_partition.sizes.shape[0]
        z = self.membership + other_partition.membership * nx  # if z = x + y * nx
        count_z = torch.bincount(z).float()
        z_to_x = torch.arange(count_z.shape[0]) % nx  # then x = z % nx
        z_to_y = torch.arange(count_z.shape[0]) // nx  # and y = z // nx

        pxy = torch.zeros((nx, ny), dtype=torch.float, device=self.membership.device)
        pxy[z_to_x, z_to_y] = count_z
        pxy /= torch.sum(pxy)
        px = torch.sum(pxy, dim=-1)
        py = torch.sum(pxy, dim=-2)

        # Compute the mutual information
        term_xy = pxy * torch.log(pxy)
        term_x = px * torch.log(px)
        term_y = py * torch.log(py)
        mutual_information = term_xy[pxy > 0].sum() - term_x[px > 0].sum() - term_y[py > 0].sum()

        # Extract the most likely mappings
        to_other = torch.max(pxy, dim=-1)[1]
        from_other = torch.max(pxy, dim=-2)[1]

        # Find one-to-one correspondence among instance IDs
        original_instance_id = torch.arange(nx, device=self.membership.device, dtype=torch.long)
        is_id_one_to_one = (from_other[to_other[original_instance_id]] == original_instance_id)
        n_reversible_mapping = is_id_one_to_one.sum().item()

        # Define a mapping that changes all bad (i.e. not unique or background) instance IDs to -1
        original_to_good_id = original_instance_id
        original_to_good_id[~is_id_one_to_one] = -1
        original_to_good_id[0] = -1  # Exclude the background
        filter_fg_pixels_with_reversible_id = (original_to_good_id[self.membership] > 0)

        # Even if the ID is reversible the pixel might have the wrong one.
        pixel_with_same_id = (to_other[self.membership] == other_partition.membership)
        intersection = torch.sum(pixel_with_same_id[filter_fg_pixels_with_reversible_id])
        union = torch.sum(self.sizes[1:]) + torch.sum(other_partition.sizes[1:]) - intersection  # exclude background
        iou = intersection.float()/union

        return ConcordancePartition(joint_distribution=pxy,
                                    mutual_information=mutual_information.item(),
                                    delta_n=ny - nx,
                                    iou=iou.item(),
                                    n_reversible_mapping=n_reversible_mapping)


#  ----------------------------------------------------------------  #
#  ------- Stuff Related to Processing (i.e. CompositionalVAE) ----  #
#  ----------------------------------------------------------------  #


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


class SparseSimilarity(NamedTuple):
    sparse_matrix: torch.sparse.FloatTensor
    index_matrix: Optional[torch.tensor]


class Segmentation(NamedTuple):
    """ Where * is the batch dimension which might be NOT present """
    raw_image: torch.Tensor  # *,ch,w,h
    fg_prob: torch.Tensor  # *,1,w,h
    integer_mask: torch.Tensor  # *,1,w,h
    bounding_boxes: Optional[torch.Tensor]  # *,3,w,h
    similarity: Optional[SparseSimilarity] = None


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
    big_mask_NON_interacting: torch.Tensor  # Use exclusively to compute overlap penalty
    prob: torch.Tensor
    bounding_box: BB
    zinstance_each_obj: torch.Tensor
    kl_zinstance_each_obj: torch.Tensor
    kl_zwhere_map: torch.Tensor
    kl_logit_map: torch.Tensor


class MetricMiniBatch(NamedTuple):
    loss: torch.Tensor
    mse: torch.Tensor
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
    cost_overlap: torch.Tensor
    cost_vol_absolute: torch.Tensor
    cost_total: torch.Tensor


class Metric_and_Reg(NamedTuple):
    # MetricMiniBatch (in the same order as underlying class)
    loss: torch.Tensor
    mse: torch.Tensor
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
    cost_overlap: torch.Tensor
    cost_vol_absolute: torch.Tensor
    cost_total: torch.Tensor

    @classmethod
    def from_merge(cls, metrics, regularizations):
        assert isinstance(metrics, MetricMiniBatch) and isinstance(regularizations, RegMiniBatch)
        assert cls._fields == MetricMiniBatch._fields + RegMiniBatch._fields
        return cls._make([*metrics, *regularizations])


class Output(NamedTuple):
    metrics: Metric_and_Reg
    inference: Inference
    imgs: torch.Tensor
