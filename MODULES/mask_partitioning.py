import torch
import torch.nn as nn
import collections
from .unet_model import MaskUnet
from .cropper_uncropper import Uncropper, Cropper

MASKS = collections.namedtuple("masks", "squares overlapping not_overlapping")

class MaskPartitioning(nn.Module):
    """ mask_k = b_k * fg_mask
        w_k = unet(mask_k, raw_img)
        new_mask_k = mask_k * [ b_k * w_k / sum_k (b_k * w_k) ] """

    def __init__(self, params):
        super().__init__()
        self.ch_raw_img = len(params["input_image"]["ch_in_description"])
        self.cropped_width = params["global"]["cropped_width"]
        self.unet_masks_resolution = MaskUnet(ch_in=self.ch_raw_img+2, ch_out=1)
        self.uncropper = Uncropper()
        self.cropper = Cropper(params)

    def forward(self, fg_mask=None, imgs_in=None, z_where=None):

        n_box, batch_size = z_where.bx.shape
        small_squares = torch.ones((1, 1, 1, 1, 1),
                                   dtype=fg_mask.dtype,
                                   layout=fg_mask.layout,
                                   device=fg_mask.device,
                                   requires_grad=False).expand(n_box, batch_size, 1,
                                                               self.cropped_width,
                                                               self.cropped_width)

        big_squares = self.uncropper.forward(z_where=z_where,
                                             small_stuff=small_squares,
                                             width_big=fg_mask.shape[-2],
                                             height_big=fg_mask.shape[-1])

        all_other_squares = (torch.sum(big_squares, dim=-5, keepdim=True) - big_squares).clamp(max=1.0)
        raw_img = imgs_in.unsqueeze(0).expand(n_box, -1, -1, -1, -1)  # expand the n_box dimension

        # cat along channel dimension
        # SHOULD I DETACH ? Yes b/c this netwrok learns to partition given proposal. Can not change proposal.
        input_for_resolution = torch.cat((fg_mask * big_squares, fg_mask * all_other_squares, raw_img), dim=-3).detach()
        w_k = self.unet_masks_resolution.forward(input_for_resolution)
        assert w_k.shape == big_squares.shape

        # partition the fg_mask
        denominator = torch.sum(big_squares * w_k, dim=-5) + 1E-6
        masks_overlapping = fg_mask * big_squares
        masks_not_overlapping = masks_overlapping * w_k / denominator

        return MASKS(squares=big_squares,
                     overlapping=masks_overlapping,
                     not_overlapping=masks_not_overlapping)
