import torch
from .namedtuple import BB, NMSoutput


class NonMaxSuppression(object):
    """ Use Intersection_over_Minimum criteria to filter out overlapping proposals. """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _perform_nms_selection(mask_overlap_nnb: torch.Tensor,
                               score_nb: torch.Tensor,
                               possible_nb: torch.Tensor,
                               k_objects_max: int) -> torch.Tensor:
        """ This algorithm does greedy NMS in parallel if possible """

        # reshape
        n_boxes, batch_size = score_nb.shape
        score_1nb = score_nb.unsqueeze(0)
        possible_1nb = possible_nb.unsqueeze(0)
        idx_n11 = torch.arange(start=0, end=n_boxes, step=1, device=score_nb.device).view(n_boxes, 1, 1).long()
        selected_n1b = torch.zeros((n_boxes, 1, batch_size), device=score_nb.device, dtype=torch.bool)

        # Loop
        counter = 0
        while counter <= k_objects_max:  # and possible.any():  # you never need more than n_objects_max proposals
            score_mask_nnb = mask_overlap_nnb * (score_1nb * possible_1nb)
            index_n1b = torch.max(score_mask_nnb, keepdim=True, dim=-2)[1]
            selected_n1b += possible_1nb.permute(1, 0, 2) * (idx_n11 == index_n1b)
            blocks_1nb = torch.sum(mask_overlap_nnb * selected_n1b, keepdim=True, dim=-3)
            possible_1nb *= (blocks_1nb == 0)
            counter += 1
            if possible_1nb.sum() == 0:
                # print("leaving NMS with counter =", counter)
                break

        # return
        return selected_n1b.squeeze(-2)

    @staticmethod
    def _unroll_and_compare(x_nb: torch.Tensor, label: str) -> torch.Tensor:
        """ Given a vector of size: batch x n_boxes it creates a matrix of size: n_boxes x n_boxes x batches
            obtained by comparing all vector entries with all other vector entries
            The comparison is either: MIN,MAX """
        if label == "MAX":
            y_nnb = torch.max(x_nb.unsqueeze(0), x_nb.unsqueeze(1))
        elif label == "MIN":
            y_nnb = torch.min(x_nb.unsqueeze(0), x_nb.unsqueeze(1))
        else:
            raise Exception("label is unknown. It is ", label)
        return y_nnb

    @staticmethod
    def _compute_box_intersection_over_min_area(bounding_box: BB) -> torch.Tensor:
        """ compute the matrix of shape: batch x n_boxes x n_boxes with the Intersection Over Unions """

        # compute x1,x3,y1,y3
        x1: torch.Tensor = bounding_box.bx - 0.5 * bounding_box.bw
        x3: torch.Tensor = bounding_box.bx + 0.5 * bounding_box.bw
        y1: torch.Tensor = bounding_box.by - 0.5 * bounding_box.bh
        y3: torch.Tensor = bounding_box.by + 0.5 * bounding_box.bh
        area: torch.Tensor = bounding_box.bw * bounding_box.bh

        min_area: torch.Tensor = NonMaxSuppression._unroll_and_compare(area, "MIN")  # min of area between box1 and box2
        xi1: torch.Tensor = NonMaxSuppression._unroll_and_compare(x1, "MAX")  # max of x1 between box1 and box2
        yi1: torch.Tensor = NonMaxSuppression._unroll_and_compare(y1, "MAX")  # max of y1 between box1 and box2
        xi3: torch.Tensor = NonMaxSuppression._unroll_and_compare(x3, "MIN")  # min of x3 between box1 and box2
        yi3: torch.Tensor = NonMaxSuppression._unroll_and_compare(y3, "MIN")  # min of y3 between box1 and box2

        intersection_area: torch.Tensor = torch.clamp(xi3 - xi1, min=0) * torch.clamp(yi3 - yi1, min=0)
        return intersection_area / min_area

    @staticmethod
    def compute_mask_and_index(score_nb: torch.Tensor,
                               bounding_box_nb: BB,
                               iom_threshold: float,
                               k_objects_max: int,
                               topk_only: bool) -> NMSoutput:
        """ Filter the proposals according to their score and their Intersection over Minimum.

            Args:
                score_nb: score used to sort the proposals
                bounding_box_nb: bounding boxes for the proposals
                iom_threshold: threshold of Intersection over Minimum. If IoM is larger than this value the boxes
                    will be suppressed during NMS. It is imporatant only if :attr:`topk_only` is False.
                k_objects_max: maximum number of proposal to consider.
                topk_only: if True, this function performs a top-K filter and returns the indices of the k-highest
                    scoring proposals regardless of their IoU. If False, the function perform NMS and returns the
                     indices of the k-highest scoring weakly-overlapping proposals.

            Returns:
                The container of type :class:`NmsOutput` with the value of the selected score and their
                indices of shape :math:`(K,B)`
        """
        assert score_nb.shape == bounding_box_nb.bx.shape
        n_boxes, batch_size = score_nb.shape

        if topk_only:
            # If nms_mask = 1 then this is equivalent to do topk only
            chosen_nms_mask_nb = torch.ones_like(score_nb)
        else:
            # this is O(N^2) algorithm (all boxes compared to all other boxes) but it is very simple
            overlap_measure_nnb = NonMaxSuppression._compute_box_intersection_over_min_area(
                bounding_box=bounding_box_nb)
            # Next greedy NMS
            binarized_overlap_nnb = (overlap_measure_nnb > iom_threshold).float()
            chosen_nms_mask_nb = NonMaxSuppression._perform_nms_selection(mask_overlap_nnb=binarized_overlap_nnb,
                                                                          score_nb=score_nb,
                                                                          possible_nb=torch.ones_like(score_nb).bool(),
                                                                          k_objects_max=k_objects_max)

        # select the indices of the top boxes according to the masked_score.
        # Note that masked_score are zero for the boxes which underwent NMS
        assert chosen_nms_mask_nb.shape == score_nb.shape
        masked_score_nb = chosen_nms_mask_nb * score_nb
        k = min(k_objects_max, n_boxes)
        masked_score_kb, indices_kb = torch.topk(masked_score_nb, k=k, dim=-2, largest=True, sorted=True)

        return NMSoutput(nms_mask=chosen_nms_mask_nb, index_top_k=indices_kb)