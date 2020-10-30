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


class ConcordanceIntMask(NamedTuple):
    intersection_mask: torch.tensor
    joint_distribution: torch.tensor
    mutual_information: float
    delta_n: int
    iou: float
    n_reversible_instances: int


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
        assert self.membership.shape == keep_vertex.shape
        assert keep_vertex.dtype == torch.bool
            
        if keep_vertex.all():
            return self
        else:
            new_membership = self.membership * keep_vertex  # put all bad vertices in the background cluster
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
    zbg: ZZ
    logit: torch.Tensor
    features: torch.Tensor


class Inference(NamedTuple):
    area_map: torch.Tensor  # shape -> batch_size, 1, w, h
    prob_map: torch.Tensor  # shape -> batch_size, 1, w, h
    prob_few: torch.Tensor  # shape -> boxes_few, batch_size
    big_bg: torch.Tensor
    big_img: torch.Tensor
    mixing: torch.Tensor
    mixing_non_interacting: torch.Tensor  # Use exclusively to compute overlap penalty
    # the samples of the 3 latent variables
    sample_c_map: torch.Tensor      # shape -> batch, 1, width, height
    sample_c: torch.Tensor          # boxes_few, batch_size
    sample_bb: BB                   # each bx,by,bw,bh has shape -> boxes_few, batch_size
    sample_zwhere: torch.Tensor     # boxes_few, batch_size, latent_dim
    sample_zinstance: torch.Tensor  # boxes_few, batch_size, latent_dim
    sample_zbg: torch.Tensor        # batch_size, latent_dim
    # kl of the 3 latent variables
    kl_zbg: torch.Tensor        # batch_size, latent_dim
    kl_logit: torch.Tensor      # batch_size
    kl_zwhere: torch.Tensor     # boxes_few, batch_size, latent_dim
    kl_zinstance: torch.Tensor  # boxes_few, batch_size, latent_dim
    # similarity DPP
    similarity_l: torch.Tensor
    similarity_w: torch.Tensor


class RegMiniBatch(NamedTuple):
    # All entries should be scalars obtained by averaging over minibatch
    reg_overlap: torch.Tensor
    reg_area_obj: torch.Tensor

    def total(self):
        tot = None
        for x in self:
            if tot is None:
                tot = x
            else:
                tot += x
        return tot


class MetricMiniBatch(NamedTuple):
    # All entries should be scalars obtained by averaging over minibatch
    loss: torch.Tensor  # this is the only tensor b/c I need to take gradients
    mse_tot: float
    reg_tot: float
    kl_tot: float
    sparsity_tot: float

    kl_zbg: float
    kl_instance: float
    kl_where: float
    kl_logit: float
    sparsity_mask: float
    sparsity_box: float
    sparsity_prob: float
    reg_overlap: float
    reg_area_obj: float

    fg_fraction: float
    geco_sparsity: float
    geco_balance: float
    delta_1: float
    delta_2: float

    similarity_l: numpy.ndarray
    similarity_w: numpy.ndarray
    lambda_logit: float


    def pretty_print(self, epoch: int=0) -> str:
        s = "[epoch {0:4d}] loss={1:.3f}, mse={2:.3f}, \
        reg={3:.3f}, kl_tot={4:.3f}, sparsity={5:.3f}, \
        fg_fraction={6:.3f}, geco_sp={7:.3f}, geco_bal={8:.3f}".format(epoch,
                                                                       self.loss,
                                                                       self.mse_tot,
                                                                       self.reg_tot,
                                                                       self.kl_tot,
                                                                       self.sparsity_tot,
                                                                       self.fg_fraction,
                                                                       self.geco_sparsity,
                                                                       self.geco_balance)
        return s


class Output(NamedTuple):
    metrics: MetricMiniBatch
    inference: Inference
    imgs: torch.Tensor
