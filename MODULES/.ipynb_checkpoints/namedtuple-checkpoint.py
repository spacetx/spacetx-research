import torch
import numpy
from typing import NamedTuple, Optional, Union
import skimage.color
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.sparse

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


class Partition(NamedTuple):
    which: str
    membership: torch.tensor  # bg=0, fg=1,2,3,.....
    sizes: torch.tensor  # both for bg and fg. It is simply obtained by numpy.bincount(membership)
    params: dict

    @staticmethod
    def is_old_2_new_identity(old_2_new: torch.tensor):
        diff = old_2_new - torch.arange(old_2_new.shape[0], device=old_2_new.device, dtype=old_2_new.dtype)
        check = (diff.abs().sum().item() == 0)
        return check

    def filter_by_vertex(self, keep_vertex: torch.tensor):
        assert self.membership.shape == keep_vertex.shape
        assert torch.min(self.membership).item() >= 0
        assert keep_vertex.dtype == torch.bool
            
        """ Put all the bad vertices in the background cluster """
        if keep_vertex.sum().item() == torch.numel(keep_vertex):
            # keep all vertex. Nothing to do
            return self
        else:
            my_filter = torch.bincount(self.membership * keep_vertex) > 0
            count = torch.cumsum(my_filter, dim=-1)
            old_2_new = ((count - count[0]) * my_filter).to(self.membership.dtype)
            
            if Partition.is_old_2_new_identity(old_2_new):
                # nothing to do
                return self
            else:
                new_membership = old_2_new[self.membership * keep_vertex]
                new_dict = self.params
                new_dict["filter_by_vertex"] = True
                return self._replace(membership=new_membership,
                                     params=new_dict,
                                     sizes=torch.bincount(new_membership))

    def filter_by_size(self, min_size: Optional[int] = None, max_size: Optional[int] = None):
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

        count = torch.cumsum(my_filter, dim=-1)
        old_2_new = (count - count[0]) * my_filter  # this makes sure that label 0 is always mapped to 0
        
        if Partition.is_old_2_new_identity(old_2_new):
            # nothing to do
            return self
        else:
            #TODO: this might be too slow. Eliminate torch.bincount.
            new_dict = self.params
            new_dict["filter_by_size"] = (min_size, max_size)
            new_membership = old_2_new[self.membership]
            return self._replace(membership=new_membership, params=new_dict, sizes=torch.bincount(new_membership))

    def concordance_with_partition(self, other_partition) -> ConcordancePartition:
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

        return ConcordancePartition(joint_distribution=pxy,
                                    mutual_information=mutual_information.item(),
                                    delta_n=ny - nx,
                                    iou=iou.item())


###    def to_sparse_similarity(self, index_matrix: torch.Tensor,
###                             index_max: Optional[int] = None,
###                             min_edge_weight: float = 0.01) -> SparseSimilarity:
###        """ Create a sparse CSR matrix:
###            CSR = csr_matrix((data, (row_ind, col_ind)), [shape = (M, N)])
###            where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k]
###            The batch dimension will be summed.
###            Only pixel with index_matrix >= 0 will be used.
###            Therefore index_matrix can be used to pass a fg_mask
###            (i.e. just set the value outside the mask to -1 to exclude)
###        """
###
###        assert len(index_matrix.shape) == len(self.data.shape) == 4
###        b, ch, w, h = self.data.shape
###        assert (b, 1, w, h) == index_matrix.shape
###
###        index_max = torch.max(index_matrix)[0].item() if index_max is None else index_max
###        radius = numpy.max(self.ch_to_dxdy)
###        pad_list = [radius + 1] * 4
###        pad_index_matrix = F.pad(index_matrix, pad=pad_list, mode="constant", value=-1)
###        pad_weight = F.pad(self.data, pad=pad_list, mode="constant", value=0.0).to(pad_index_matrix.device)
###
###        # Prepare the storage
###        i_list, j_list, e_list = [], [], []  # i,j are verteces, e=edges
###
###        for ch, dxdy in enumerate(self.ch_to_dxdy):
###            pad_index_matrix_shifted = torch.roll(torch.roll(pad_index_matrix, dxdy[0], dims=-2), dxdy[1],
###                                                  dims=-1)
###
###            data = pad_weight[:, ch, :, :].flatten()
###            row_ind = pad_index_matrix[:, 0, :, :].flatten()
###            col_ind = pad_index_matrix_shifted[:, 0, :, :].flatten()
###
###            # Do not add loops.
###            my_filter = (row_ind >= 0) * (col_ind >= 0) * (data > min_edge_weight)
###
###            e_list += data[my_filter].tolist()
###            i_list += row_ind[my_filter].tolist()
###            j_list += col_ind[my_filter].tolist()
###
###        return SparseSimilarity(csr_matrix=scipy.sparse.csr_matrix((e_list, (i_list, j_list)),
###                                                                   shape=(index_max, index_max)),
###                                index_matrix=None)
###

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
