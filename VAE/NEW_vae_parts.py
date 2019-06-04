
import torch
import collections
from UNET.unet_model import UNet 
from ROI.roi import Roi
from NON_MAX_SUPPRESSION.non_max_suppression import Non_Max_Suppression 
from ENCODERS_DECODERS.encoders import Encoder_MLP
from ENCODERS_DECODERS.decoders import Multi_Channel_Img_Decoder, Mask_Decoder_MLP
from CROPPER_UNCROPPER.cropper_uncropper import Uncropper, Cropper

class Inference(torch.nn.Module):
    
    def __init__(self,params):
        super().__init__()        
        self.unet = UNet(params)
        ch_features = self.unet.ch_feature_stack
        assert(len(ch_features) >= 1) 
        self.roi = Roi(ch_features,params)
        self.nms = Non_Max_Suppression(params)
        self.cropper = Cropper(params)
        
        # encoders z_what,z_mask
        ch_raw_image = len(params["IMG.ch_in_description"])
        width  = params["SD.width"]
        dim_in = width*width*ch_raw_image
        dim_h1 = params["SD.dim_h1"] 
        dim_h2 = params["SD.dim_h2"]
        dim_zwhat = params['ZWHAT.dim']
        dim_zmask = params['ZMASK.dim']
        self.encoder_zwhat = Encoder_MLP(dim_in,dim_h1,dim_h2,dim_zwhat,"z_what")
        self.encoder_zmask = Encoder_MLP(dim_in,dim_h1,dim_h2,dim_zmask,"z_mask")
    
    def forward(self,imgs_in,p_corr_factor=0.0):
        raw_width,raw_height = imgs_in.shape[-2:]
        
        # FEATURES EXTRACTION VIA A UNET
        features = self.unet.forward(imgs_in)
        
        # TRANSFROM THE FEATURES INTO z_where predictions
        z_where_all = self.roi.forward(features,raw_width,raw_height)
        
        # ANNEAL THE PROBABILITIES IF NECESSARY
        if(p_corr_factor>0):
            with torch.no_grad():
                box_intensities = compute_average_intensity_in_box(imgs_in,z_where_all)
                rank = torch.sort(box_intensities.view(-1),dim=-1, descending=False).float()/torch.numel(box_intensities)
                p_approx = rank.view_as(z_where_all.prob)
            # weighted average of the prob by the inference netwrok and probs by the comulative 
            z_where_all.prob = (1-p_corr_factor)*z_where_all.prob+p_corr_factor*p_approx
        
        # PERFORM Non-Max-Suppression    
        z_where = self.nms.forward(z_where_all) 
        
        # CROP
        cropped_imgs = self.cropper.forward(z_where,imgs_in) 
        
        # ENCODE
        z_what = self.encoder_zwhat.forward(cropped_imgs)
        z_mask = self.encoder_zmask.forward(cropped_imgs)
        
        # COMBINE THE TUPLE
        return collections.namedtuple('z_prediction', 'z_where z_mask z_what')._make([z_where, z_mask, z_what])

    
class Imgs_Generator(torch.nn.Module):
    def __init__(self,params):
        super().__init__()
        self.decoder = Multi_Channel_Img_Decoder(params)
        self.uncropper = Uncropper(params)
        
    def forward(self,z_where,z_what,width_raw,height_raw):
    
        # DECODER
        batch_size,n_boxes,dim_z_what = z_what.shape
        cropped_imgs    = self.decoder.forward(z_what.view(-1,dim_z_what))
        uncropped_imgs  = self.uncropper(z_where,cropped_imgs,width_raw,height_raw)             
        return uncropped_imgs

class Masks_Generator(torch.nn.Module):
    def __init__(self,params):
        super().__init__()
        self.decoder = Mask_Decoder_MLP(params)
        self.uncropper = Uncropper(params)
        
    def forward(self,z_where,z_mask,width_raw,height_raw):
    
        # DECODER
        batch_size,n_boxes,dim_z_mask = z_mask.shape
        cropped_masks    = self.decoder.forward(z_mask.view(-1,dim_z_mask))
        uncropped_masks  = self.uncropper(z_where,cropped_masks,width_raw,height_raw)             
        return uncropped_masks
    
def compute_average_intensity_in_box(imgs,z_where):
    """ Input batch of images: batch x ch x w x h
        z_where collections of [prob,bx,by,bw,bh]
        z_where.prob.shape = batch x n_box x 1 
        similarly for bx,by,bw,bh
        
        Output: 
        av_intensity = batch x n_box
    """
    
    # sum the channels, do cumulative sum in width and height
    cum = torch.cumsum(torch.cumsum(torch.sum(imgs, dim=-3),dim=-1),dim=-2)
    assert len(cum.shape)==3
    batch_size, w, h = cum.shape 
    
    # compute the x1,y1,x3,y3
    x1 = torch.clamp((z_where.bx_dimfull-0.5*z_where.bw_dimfull).long(),min=0,max=w-1).squeeze(-1) 
    x3 = torch.clamp((z_where.bx_dimfull+0.5*z_where.bw_dimfull).long(),min=0,max=w-1).squeeze(-1) 
    x3 = torch.clamp((z_where.by_dimfull-0.5*z_where.bh_dimfull).long(),min=0,max=h-1).squeeze(-1) 
    x1 = torch.clamp((z_where.by_dimfull+0.5*z_where.bh_dimfull).long(),min=0,max=h-1).squeeze(-1) 
    
    # compute the area
    # Note that this way penalizes boxes that go out-of-bound
    # This is in contrast to area = (x3-x1)*(y3-y1) which does NOT penalize boxes out of bound
    area = (z_where.bw_dimfull*z_where.bh_dimfull).squeeze(-1) 

    # Make some checks
    assert x1.shape == x3.shape == y1.shape == y3.shape == area.shape
    assert len(area.shape)==2
    assert area.shape[0] == cum.shape[0]
    batch_size, n_boxes = area.shape
    
    batch_index = torch.arange(0,batch_size).view(-1,1).expand(-1,n_boxes)
    
    result = cum[batch_index,x3,y3]-cum[batch_index,x1,y3]-cum[batch_index,x3,y1]+cum[batch_index,x1,y1]
    
    assert (batch_size, n_boxes) == result.shape
    
    return result/area

    
    
    
    
    
###class Generator(torch.nn.Module):
###    def __init__(self,params):
###        super().__init__()
###        self.img_decoder  = Multi_Channel_Img_Decoder(params)
###        self.mask_decoder = Mask_Decoder_MLP(params)
###        self.uncropper    = Uncropper(params)
###        
###    def forward(self,z_what,z_mask,box_dimfull,width_raw,height_raw):
###    
###        # IMAGES 
###        batch_size,n_boxes,dim_z_what = z_what.shape
###        cropped_imgs    = self.img_decoder.forward(z_what.view(-1,dim_z_what))
###        uncropped_imgs  = self.uncropper(cropped_imgs,box_dimfull,width_raw,height_raw)             
###
###        # MASKS
###        batch_size,n_boxes,dim_z_mask = z_mask.shape
###        cropped_masks    = self.mask_decoder.forward(z_mask.view(-1,dim_z_mask))
###        uncropped_masks  = self.uncropper(cropped_masks,box_dimfull,width_raw,height_raw)             
###
###        return collections.namedtuple('generated', "imgs masks")._make([uncropped_imgs,uncropped_masks]) 
