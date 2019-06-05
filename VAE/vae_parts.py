
import torch
import collections
from UNET.unet_model import UNet
from NON_MAX_SUPPRESSION.non_max_suppression import Non_Max_Suppression 
from ENCODERS_DECODERS.encoders import Encoder_MLP, Encoder_CONV
from ENCODERS_DECODERS.decoders import Multi_Channel_Img_Decoder, Decoder_MLP, Decoder_CONV
from CROPPER_UNCROPPER.cropper_uncropper import Uncropper, Cropper

        
def compute_ranking(x,dim=-1):
    _, order = torch.sort(x,    dim=dim, descending=False)
    
    # this is the brute force but slow way (b/c it has a second argsort)
    # _, rank  = torch.sort(order,dim=dim, descending=False)

    # rhis is the fast way which uses indexing on the left
    n1,n2 = order.shape
    rank = torch.zeros_like(order)
    batch_index = torch.arange(n1,dtype=order.dtype,device=order.device).view(-1,1).expand(n1,n2)
    rank[batch_index,order]=torch.arange(n2,dtype=order.dtype,device=order.device).view(1,-1).expand(n1,n2)
    
    return rank
        
def compute_average_intensity_in_box(imgs,z_where):
    """ Input batch of images: batch x ch x w x h
        z_where collections of [prob,bx,by,bw,bh]
        z_where.prob.shape = batch x n_box x 1 
        similarly for bx,by,bw,bh
        
        Output: 
        av_intensity = batch x n_box
    """
    
    # sum the channels, do cumulative sum in width and height
    cum = torch.cumsum(torch.cumsum(imgs,dim=-2),dim=-1)[:,0,:,:]
    assert len(cum.shape)==3
    batch_size, w, h = cum.shape 
    
    # compute the x1,y1,x3,y3
    x1 = torch.clamp((z_where.bx_dimfull-0.5*z_where.bw_dimfull).long(),min=0,max=w-1).squeeze(-1) 
    x3 = torch.clamp((z_where.bx_dimfull+0.5*z_where.bw_dimfull).long(),min=0,max=w-1).squeeze(-1) 
    y1 = torch.clamp((z_where.by_dimfull-0.5*z_where.bh_dimfull).long(),min=0,max=h-1).squeeze(-1) 
    y3 = torch.clamp((z_where.by_dimfull+0.5*z_where.bh_dimfull).long(),min=0,max=h-1).squeeze(-1) 
    
    # compute the area
    # Note that this way penalizes boxes that go out-of-bound
    # This is in contrast to area = (x3-x1)*(y3-y1) which does NOT penalize boxes out of bound
    area = (z_where.bw_dimfull*z_where.bh_dimfull).squeeze(-1) 

    # Make some checks
    #assert x1.shape == x3.shape == y1.shape == y3.shape == area.shape
    #assert len(area.shape)==2
    #assert area.shape[0] == cum.shape[0]
    batch_size, n_boxes = area.shape
    
    # compute the total intensity in each box
    batch_index = torch.arange(0,batch_size).view(-1,1).expand(-1,n_boxes).to(x1.device)
    tot_intensity = cum[batch_index,x3,y3]-cum[batch_index,x1,y3]-cum[batch_index,x3,y1]+cum[batch_index,x1,y1]
    #assert (batch_size, n_boxes) == tot_intensity.shape
   
    # return the average intensity
    return tot_intensity/area 


def select_top_boxes_by_prob(z_where_all,n_max_object):
    
    p=z_where_all.prob.squeeze(-1)
    batch_size,n_boxes = p.shape 
    p_top_k, top_k_indeces = torch.topk(p, k=min(n_max_object,n_boxes), dim=-1, largest=True, sorted=True)
    batch_size, k = top_k_indeces.shape 
    batch_indeces = torch.arange(batch_size).unsqueeze(-1).expand(-1,k).to(top_k_indeces.device)
               
    # package the output
    return collections.namedtuple('z_where', 'prob bx_dimfull by_dimfull bw_dimfull bh_dimfull')._make(
            [z_where_all.prob[batch_indeces,top_k_indeces],
             z_where_all.bx_dimfull[batch_indeces,top_k_indeces],
             z_where_all.by_dimfull[batch_indeces,top_k_indeces],
             z_where_all.bw_dimfull[batch_indeces,top_k_indeces],
             z_where_all.bh_dimfull[batch_indeces,top_k_indeces]])


        
class Inference(torch.nn.Module):
    
    def __init__(self,params):
        super().__init__()        
        self.unet = UNet(params)
        self.nms = Non_Max_Suppression(params)
        self.cropper = Cropper(params)
        
        # stuff for the select_top_boxes_by_prob
        self.n_max_objects = params["PRIOR.n_max_objects"]

        # encoders z_what,z_mask
        #self.encoder_zwhat = Encoder_MLP(params,is_zwhat=True)
        #self.encoder_zmask = Encoder_MLP(params,is_zwhat=False)
        self.encoder_zwhat = Encoder_CONV(params,is_zwhat=True)
        self.encoder_zmask = Encoder_CONV(params,is_zwhat=False)        
    
    def forward(self,imgs_in,p_corr_factor=0.0):
        raw_width,raw_height = imgs_in.shape[-2:]
        
        # UNET
        z_where_all = self.unet.forward(imgs_in)
       
        # ANNEAL THE PROBABILITIES IF NECESSARY
        if(p_corr_factor>0):
            with torch.no_grad():
                x = compute_average_intensity_in_box(imgs_in,z_where_all)
                tmp = (compute_ranking(x,dim=-1).float()/(x.shape[-1]-1)).view_as(z_where_all.prob)  # this is rescaled ranking
                #print("x.shape, tmp.shape",x.shape,tmp.shape)
                #print("print tmp[0]",tmp[0])  
                #p_approx = tmp
                p_approx = tmp.pow(10) #this is to make p_approx even more peaked
                
            # weighted average of the prob by the inference netwrok and probs by the comulative 
            new_p = (1-p_corr_factor)*z_where_all.prob+p_corr_factor*p_approx
            z_where_all = z_where_all._replace(prob=new_p)
        
        # NMS   
        z_where = self.nms.forward(z_where_all)
        #z_where = select_top_boxes_by_prob(z_where_all,self.n_max_objects)
        #print("z_where.prob[0]",z_where.prob[0])
        
        # CROP
        cropped_imgs = self.cropper.forward(z_where,imgs_in) 
        
        # ENCODE
        z_what = self.encoder_zwhat.forward(cropped_imgs)
        z_mask = self.encoder_zmask.forward(cropped_imgs)
        
        # COMBINE THE TUPLE
        return collections.namedtuple('z_prediction', "z_where z_mask z_what")._make([z_where, z_mask, z_what])

    
class Imgs_Generator(torch.nn.Module):
    def __init__(self,params):
        super().__init__()
        self.imgs_decoder = Multi_Channel_Img_Decoder(params)
        self.uncropper = Uncropper(params)
        
    def forward(self,z_where,z_what,width_raw,height_raw):
    
        # DECODER
        batch_size,n_boxes,dim_z_what = z_what.shape
        cropped_imgs    = self.imgs_decoder.forward(z_what.view(-1,dim_z_what))
        uncropped_imgs  = self.uncropper(z_where,cropped_imgs,width_raw,height_raw)             
        return uncropped_imgs

class Masks_Generator(torch.nn.Module):
    def __init__(self,params):
        super().__init__()
        #self.mask_decoder = Decoder_MLP(params,is_zwhat=False)
        self.mask_decoder = Decoder_CONV(params,is_zwhat=False)
        self.uncropper = Uncropper(params)
        
    def forward(self,z_where,z_mask,width_raw,height_raw):
    
        # DECODER
        batch_size,n_boxes,dim_z_mask = z_mask.shape
        cropped_masks    = self.mask_decoder.forward(z_mask.view(-1,dim_z_mask))
        uncropped_masks  = self.uncropper(z_where,cropped_masks,width_raw,height_raw)             
        return uncropped_masks
    
   
   
    
    
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
