
import torch
import torch.nn as nn
import collections
from UNET.unet_model import UNet
from NON_MAX_SUPPRESSION.non_max_suppression import Non_Max_Suppression 
from ENCODERS_DECODERS.encoders_decoders import EncoderConv, DecoderConv
from CROPPER_UNCROPPER.cropper_uncropper import Uncropper, Cropper


P_AND_ZWHERE = collections.namedtuple('zwhere', 'prob bx by bw bh')
ZWHERE = collections.namedtuple('zwhere', 'bx by bw bh')
KL = collections.namedtuple('kl', "tp tx ty tw th zwhat zmask")
RESULT = collections.namedtuple('result', "prob z_where z_what z_mask")  # this has results: prob, bx, by, bw, bh, ..
GEN = collections.namedtuple("generated", "small_mu_raw small_w_raw big_mu_sigmoid big_w_sigmoid")


def compute_ranking(x):
    """ Given a vector of shape: n, batch_size
        For each batch dimension it ranks the n elements"""
    assert len(x.shape) == 2
    n, batch_size = x.shape
    _, order = torch.sort(x, dim=-2, descending=False)

    # this is the fast way which uses indexing on the left
    rank = torch.zeros_like(order)
    batch_index = torch.arange(batch_size, dtype=order.dtype, device=order.device).view(1, -1).expand(n, batch_size)
    rank[order, batch_index] = torch.arange(n, dtype=order.dtype, device=order.device).view(-1, 1).expand(n, batch_size)
    return rank

def compute_average_intensity_in_box(imgs,z_where):
    """ Input batch of images: batch x ch x w x h
        z_where collections of [prob,bx,by,bw,bh]
        z_where.prob.shape = batch x n_box x 1 
        similarly for bx,by,bw,bh
        
        Output: 
        av_intensity = batch x n_box
    """
    # cumulative sum in width and height, standard sum in channels
    cum = torch.sum(torch.cumsum(torch.cumsum(imgs, dim=-1), dim=-2), dim=-3)
    assert len(cum.shape) == 3
    batch_size, w, h = cum.shape

    # compute the x1,y1,x3,y3
    x1 = torch.clamp((z_where.bx_dimfull - 0.5 * z_where.bw_dimfull).long(), min=0, max=w-1).squeeze(-1)
    x3 = torch.clamp((z_where.bx_dimfull + 0.5 * z_where.bw_dimfull).long(), min=0, max=w-1).squeeze(-1)
    y1 = torch.clamp((z_where.by_dimfull - 0.5 * z_where.bh_dimfull).long(), min=0, max=h-1).squeeze(-1)
    y3 = torch.clamp((z_where.by_dimfull + 0.5 * z_where.bh_dimfull).long(), min=0, max=h-1).squeeze(-1)
    assert x1.shape == x3.shape == y1.shape == y3.shape

    # compute the area
    # Note that this way penalizes boxes that go out-of-bound
    # This is in contrast to area = (x3-x1)*(y3-y1) which does NOT penalize boxes out of bound
    area = (z_where.bw_dimfull*z_where.bh_dimfull).squeeze(-1)
    assert area.shape == x1.shape == x3.shape == y1.shape == y3.shape
    n_boxes, batch_size = area.shape

    # compute the total intensity in each box
    batch_index = torch.arange(start=0, end=batch_size, step=1, device=x1.device,
                               dtype=x1.dtype).view(1, -1).expand(n_boxes, -1)
    assert batch_index.shape == x1.shape

    tot_intensity = cum[batch_index, x3, y3] \
        + cum[batch_index, x1, y1] \
        - cum[batch_index, x1, y3] \
        - cum[batch_index, x3, y1]

    # return the average intensity
    assert tot_intensity.shape == x1.shape
    return tot_intensity/area


class Inference(torch.nn.Module):
    
    def __init__(self,params):
        super().__init__()        
        self.unet = UNet(params)
        self.nms = Non_Max_Suppression(params)
        self.cropper = Cropper(params)
        
        # encoders z_what,z_mask
        self.encoder_zwhat = EncoderConv(params, dim_z=params['ZWHAT.dim'], name="z_what")
        self.encoder_zmask = EncoderConv(params, dim_z=params['ZMASK.dim'], name="z_mask")

    def forward(self, imgs_in, prob_corr_factor=0.0, overlap_threshold=0.2,
                randomize_nms_factor=0.0, n_objects_max=6, topk_only=False):

        # UNET
        #print("start UNET")
        z_where_all = self.unet.forward(imgs_in)
        #print("end UNET")

        # ANNEAL THE PROBABILITIES IF NECESSARY
        if prob_corr_factor>0:
            with torch.no_grad():
                av_intensity = compute_average_intensity_in_box(imgs_in, z_where_all)
                assert len(av_intensity.shape) == 2
                n_boxes_all, batch_size = av_intensity.shape
                ranking = compute_ranking(av_intensity)  # shape: n_boxes_all, batch. This is in [0,n_box_all-1]
                tmp = ((ranking + 1).float() / (n_boxes_all + 1)).view_as(z_where_all.prob)
                p_approx = tmp.pow(10)

            # weighted average of the prob by the inference netwrok and probs by the comulative
            new_p = (1-prob_corr_factor)*z_where_all.prob+prob_corr_factor*p_approx
            z_where_all = z_where_all._replace(prob=new_p)


        # NMS
        #print("start NMS")
        z_where = self.nms.forward(z_where=z_where_all,
                                   overlap_threshold=overlap_threshold,
                                   randomize_nms_factor=randomize_nms_factor,
                                   n_objects_max=n_objects_max,
                                   topk_only=topk_only)
        #print("end NMS")

        # CROP
        cropped_imgs = self.cropper.forward(z_where,imgs_in) 
        
        # ENCODE
        z_what = self.encoder_zwhat.forward(cropped_imgs)
        z_mask = self.encoder_zmask.forward(cropped_imgs)
        
        # COMBINE THE TUPLE
        return collections.namedtuple('z_prediction', "z_where z_mask z_what")._make([z_where, z_mask, z_what])


class Generator(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.uncropper = Uncropper()
        self.decoder_masks = DecoderConv(params,
                                         dim_z=params['ZMASK.dim'],
                                         ch_out=1)
        self.decoder_imgs = DecoderConv(params,
                                        dim_z=params['ZWHAT.dim'],
                                        ch_out=len(params['IMG.ch_in_description']))

    def forward(self, z_where=None, z_what=None, z_mask=None, width_raw=None, height_raw=None):

        small_mu_raw = self.decoder_imgs.forward(z_what)
        small_w_raw = self.decoder_masks.forward(z_mask)

        # Use uncropper to generate the big stuff
        # from the small stuff obtain dby catting along the channel dims
        ch_dims = (small_mu_raw.shape[-3], small_w_raw.shape[-3])
        big_stuff = self.uncropper.forward(z_where=z_where,
                                           small_stuff=torch.sigmoid(torch.cat((small_mu_raw, small_w_raw), dim=-3)),  # cat ch dimension
                                           width_big=width_raw,
                                           height_big=height_raw)

        big_mu_sigmoid, big_w_sigmoid = torch.split(big_stuff, ch_dims, dim=-3)  # split ch dimension

        return GEN(small_mu_raw=small_mu_raw,
                   small_w_raw=small_w_raw,
                   big_mu_sigmoid=big_mu_sigmoid,
                   big_w_sigmoid=big_w_sigmoid)
