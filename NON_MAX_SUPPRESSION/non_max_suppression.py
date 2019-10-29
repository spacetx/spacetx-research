# %load NON_MAX_SUPPRESSION/non_max_suppression.py
import torch
import collections

class Non_Max_Suppression(torch.nn.Module):
    """ Use Intersection_over_Union criteria to put most of the entries to zero while leaving few detection unchanged.
        INPUT  has shape: BATCH x N_BOXES x .... 
        OUTPUT has shape: BATCH x K_MAX   x ....
                
        The non_max_suppression algorithm is as follows:
        1. Discard all voxels with p<p_threshold
        2. While there are any remaining voxels:
            - Pick the voxel with largest p and output it as a prediction
            - Discard any voxels which has a IoU> IoU_threshold with the box just exported.
    """

    def __init__(self,params: dict):
        super().__init__()
        self.score_threshold       = params['NMS.score_threshold']

    @staticmethod
    def perform_nms_selection(mask_overlap, score, possible, n_objects_max):
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
        score = score.unsqueeze(0)                                                     # 1 x n_box x batch
        possible = possible.unsqueeze(0)                                               # 1 x n_box x batch
        idx = torch.arange(start=0, end=n_boxes, step=1,
                           device=score.device).view(n_boxes, 1, 1).long()             # n_box x 1 x 1
        selected = torch.zeros((n_boxes, 1, batch_size), device=score.device).float()  # n_box x 1 x batch

        # Loop
        for l in range(n_objects_max):  # you never need more than n_objects_max proposals
            score_mask = mask_overlap*(score*possible)                       # n_box x n_box x batch
            index = torch.max(score_mask, keepdim=True, dim=-2)[1]           # n_box x 1 x batch
            selected += possible.permute(1, 0, 2)*(idx == index).float()     # n_box x 1 x batch
            blocks = torch.sum(mask_overlap*selected, keepdim=True, dim=-3)  # 1 x n_box x batch
            possible *= (blocks == 0).float()                                # 1 x n_box x batch

        # return
        return selected.squeeze(-2)  # shape: n_box x batch


    @staticmethod
    def unroll_and_compare(x, label):
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

    def compute_box_intersection_over_min_area(self, z_where=None):
        """ compute the matrix of shape: batch x n_boxes x n_boxes with the Intersection Over Unions """

        # compute x1,x3,y1,y3
        x1 = (z_where.bx - 0.5 * z_where.bw).squeeze(-1)
        x3 = (z_where.bx + 0.5 * z_where.bw).squeeze(-1)
        y1 = (z_where.by - 0.5 * z_where.bh).squeeze(-1)
        y3 = (z_where.by + 0.5 * z_where.bh).squeeze(-1)
        area = (z_where.bw * z_where.bh).squeeze(-1)

        min_area = self.unroll_and_compare(area, "MIN")  # min of area between box1 and box2
        xi1 = self.unroll_and_compare(x1, "MAX")  # max of x1 between box1 and box2
        yi1 = self.unroll_and_compare(y1, "MAX")  # max of y1 between box1 and box2
        xi3 = self.unroll_and_compare(x3, "MIN")  # min of x3 between box1 and box2
        yi3 = self.unroll_and_compare(y3, "MIN")  # min of y3 between box1 and box2

        intersection_area = torch.clamp(xi3 - xi1, min=0) * torch.clamp(yi3 - yi1, min=0)
        return intersection_area / min_area

    def compute_nms_mask(self, prob=None, z_where=None, overlap_threshold=None, randomize_nms_factor=None,
                         n_objects_max=None, topk_only=None):

        # compute the indices to do nms + topk filter based on noisy probabilities
        prob_all = prob.squeeze(-1)
        assert len(prob_all.shape) == 2
        n_boxes, batch_size = prob_all.shape

        overlap_measure = self.compute_box_intersection_over_min_area(z_where=z_where)  # this is O(N^2) algorithm
        binarized_overlap_measure = (overlap_measure > overlap_threshold).float()
        assert binarized_overlap_measure.shape == (n_boxes, n_boxes, batch_size)

        # The noise need to be added to the probabilities
        noisy_score = torch.clamp(prob_all + randomize_nms_factor * torch.randn_like(prob_all), min=0.0)
        assert noisy_score.shape == (n_boxes, batch_size)

        if topk_only:
            # If nms_mask = 1 then this is equivalent to do topk only
            chosen_nms_mask = torch.ones_like(noisy_score)
        else:
            possible = (noisy_score > self.score_threshold).float()
            chosen_nms_mask = self.perform_nms_selection(binarized_overlap_measure, noisy_score, possible, n_objects_max)
        assert chosen_nms_mask.shape == (n_boxes, batch_size)

        # select the indices of the top boxes according to the masked scores.
        # Note that masked_score:
        # 1. is based on the noisy score
        # 2. is zero for boxes which underwent NMS
        masked_score = chosen_nms_mask * noisy_score
        k = min(n_objects_max, n_boxes)
        scores_tmp, top_k_indices = torch.topk(masked_score, k=k, dim=-2, largest=True, sorted=True)
        batch_indices = torch.arange(start=0, end=batch_size, step=1, dtype=top_k_indices.dtype,
                                     device=top_k_indices.device).view(1, -1).expand(n_objects_max, -1)

        return chosen_nms_mask, top_k_indices, batch_indices

    def forward(self, prob=None, z_where=None, overlap_threshold=None, randomize_nms_factor=None, n_objects_max=None, topk_only=None):

        # package the output
        with torch.no_grad():
            nms_mask, top_k_indices, batch_indices = self.compute_nms_mask(prob=prob,
                                                                           z_where=z_where,
                                                                           overlap_threshold=overlap_threshold,
                                                                           randomize_nms_factor=randomize_nms_factor,
                                                                           n_objects_max=n_objects_max,
                                                                           topk_only=topk_only)

        return nms_mask, top_k_indices, batch_indices
