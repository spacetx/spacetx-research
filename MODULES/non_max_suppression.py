import torch
from .namedtuple import BB, NMSoutput
from .utilities import weighted_sampling_without_replacement

class NonMaxSuppression(object):
    """ Use Intersection_over_Union criteria to put most of the entries to zero while leaving few detection unchanged.
        INPUT  has shape: BATCH x N_BOXES x .... 
        OUTPUT has shape: BATCH x K_MAX   x ....
                
        The non_max_suppression algorithm is as follows:
        1. Discard all voxels with p<p_threshold
        2. While there are any remaining voxels:
            - Pick the voxel with largest p and output it as a prediction
            - Discard any voxels which has a IoU> IoU_threshold with the box just exported.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def perform_nms_selection(mask_overlap: torch.Tensor,
                              score: torch.Tensor,
                              possible: torch.Tensor,
                              n_objects_max: int) -> torch.Tensor:
        """ Input:
            mask_overlap: n x n x batch
            score: n x batch
            possible: n x batch
            n_objects_max: integer
            Output: n x batch

            It assumes score >= 0.0
        """

        # check input formats
        assert len(mask_overlap.shape) == 3
        n_boxes, n_boxes, batch_size = mask_overlap.shape
        assert score.shape == (n_boxes, batch_size)
        assert possible.shape == (n_boxes, batch_size)

        # reshape
        score: torch.Tensor = score.unsqueeze(0)                                                     # 1 x n_box x batch
        possible: torch.Tensor = possible.unsqueeze(0)                                               # 1 x n_box x batch
        idx: torch.Tensor = torch.arange(start=0, end=n_boxes, step=1,
                            device=score.device).view(n_boxes, 1, 1).long()             # n_box x 1 x 1
        selected: torch.Tensor = torch.zeros((n_boxes, 1, batch_size), device=score.device).float()  # n_box x 1 x batch

        # Loop
        for l in range(n_objects_max):  # you never need more than n_objects_max proposals
            score_mask: torch.Tensor = mask_overlap*(score*possible)                       # n_box x n_box x batch
            index = torch.max(score_mask, keepdim=True, dim=-2)[1]           # n_box x 1 x batch
            selected += possible.permute(1, 0, 2)*(idx == index).float()     # n_box x 1 x batch
            blocks = torch.sum(mask_overlap*selected, keepdim=True, dim=-3)  # 1 x n_box x batch
            possible *= (blocks == 0).float()                                # 1 x n_box x batch

        # return
        return selected.squeeze(-2)  # shape: n_box x batch

    @staticmethod
    def unroll_and_compare(x: torch.Tensor, label: str) -> torch.Tensor:
        """ Given a vector of size: batch x n_boxes
        it creates a matrix of size: batch x n_boxes x n_boxes
        obtained by comparing all vecotr entries with all other vector entries
        The comparison is either: MIN,MAX """
        assert len(x.shape) == 2
        n_box, batch_size = x.shape
        tmp_a = x.view(1, n_box, batch_size)
        tmp_b = x.view(n_box, 1, batch_size)
        if label == "MAX":
            return torch.max(tmp_a, tmp_b)
        elif label == "MIN":
            return torch.min(tmp_a, tmp_b)
        else:
            raise Exception("label is unknown. It is ", label)

    @staticmethod
    def compute_box_intersection_over_min_area(bounding_box: BB) -> torch.Tensor:
        """ compute the matrix of shape: batch x n_boxes x n_boxes with the Intersection Over Unions """

        # compute x1,x3,y1,y3
        x1: torch.Tensor = bounding_box.bx - 0.5 * bounding_box.bw
        x3: torch.Tensor = bounding_box.bx + 0.5 * bounding_box.bw
        y1: torch.Tensor = bounding_box.by - 0.5 * bounding_box.bh
        y3: torch.Tensor = bounding_box.by + 0.5 * bounding_box.bh
        area: torch.Tensor = bounding_box.bw * bounding_box.bh

        min_area: torch.Tensor = NonMaxSuppression.unroll_and_compare(area, "MIN")  # min of area between box1 and box2
        xi1: torch.Tensor = NonMaxSuppression.unroll_and_compare(x1, "MAX")  # max of x1 between box1 and box2
        yi1: torch.Tensor = NonMaxSuppression.unroll_and_compare(y1, "MAX")  # max of y1 between box1 and box2
        xi3: torch.Tensor = NonMaxSuppression.unroll_and_compare(x3, "MIN")  # min of x3 between box1 and box2
        yi3: torch.Tensor = NonMaxSuppression.unroll_and_compare(y3, "MIN")  # min of y3 between box1 and box2

        intersection_area: torch.Tensor = torch.clamp(xi3 - xi1, min=0) * torch.clamp(yi3 - yi1, min=0)
        return intersection_area / min_area

    @staticmethod
    def compute_mask_and_index(prob: torch.Tensor,
                               bounding_box: BB,
                               overlap_threshold: float,
                               n_objects_max: int,
                               topk_only: bool) -> NMSoutput:

        # compute the indices to do nms + topk filter based on noisy probabilities
        assert len(prob.shape) == 2
        n_boxes, batch_size = prob.shape

        # this is O(N^2) algorithm
        overlap_measure = NonMaxSuppression.compute_box_intersection_over_min_area(bounding_box=bounding_box)
        binarized_overlap = (overlap_measure > overlap_threshold).float()
        assert binarized_overlap.shape == (n_boxes, n_boxes, batch_size)

        if topk_only:
            # If nms_mask = 1 then this is equivalent to do topk only
            chosen_nms_mask = torch.ones_like(prob)
        else:
            possible = (prob > 0.0).float()
            chosen_nms_mask: torch.Tensor = NonMaxSuppression.perform_nms_selection(mask_overlap=binarized_overlap,
                                                                                    score=prob,
                                                                                    possible=possible,
                                                                                    n_objects_max=n_objects_max)

            assert chosen_nms_mask.shape == (n_boxes, batch_size)

        # select the indices of the top boxes according to the masked prob.
        # Note that masked_score:
        # 1. is based on the prob
        # 2. is zero for boxes which underwent NMS
        masked_prob = chosen_nms_mask * prob
        k = min(n_objects_max, n_boxes)
        indices_top_k: torch.Tensor = torch.topk(masked_prob, k=k, dim=-2, largest=True, sorted=True)[1]

        return NMSoutput(nms_mask=chosen_nms_mask,
                         index_top_k=indices_top_k)

