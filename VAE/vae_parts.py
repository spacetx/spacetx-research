
import torch
import torch.nn as nn
import collections
from UNET.unet_model import UNet
from NON_MAX_SUPPRESSION.non_max_suppression import Non_Max_Suppression 
from ENCODERS_DECODERS.encoders import EncoderConv
from ENCODERS_DECODERS.decoders import DecoderConv
from CROPPER_UNCROPPER.cropper_uncropper import Uncropper, Cropper


P_AND_ZWHERE = collections.namedtuple('zwhere', 'prob bx by bw bh')
ZWHERE = collections.namedtuple('zwhere', 'bx by bw bh')
KL = collections.namedtuple('kl', "tp tx ty tw th zwhat zmask")
RESULT = collections.namedtuple('result', "prob z_where z_what z_mask")  # this has results: prob, bx, by, bw, bh, ..
GEN = collections.namedtuple("generated", "small_mu_raw small_w_raw big_mu_sigmoid big_w_sigmoid")


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
    batch_index = torch.arange(start=0,end=batch_size,step=1,device=x1.device,dtype=x1.dtype).view(-1,1).expand(-1,n_boxes)
    tot_intensity = cum[batch_index,x3,y3]-cum[batch_index,x1,y3]-cum[batch_index,x3,y1]+cum[batch_index,x1,y1]
    #assert (batch_size, n_boxes) == tot_intensity.shape
   
    # return the average intensity
    return tot_intensity/area 


def select_top_boxes_by_prob(z_where_all,n_max_object):
    
    p=z_where_all.prob.squeeze(-1)
    batch_size,n_boxes = p.shape 
    p_top_k, top_k_indeces = torch.topk(p, k=min(n_max_object,n_boxes), dim=-1, largest=True, sorted=True)
    batch_size, k = top_k_indeces.shape 
    batch_indeces = torch.arange(start=0,end=batch_size,step=1,
                                 device=top_k_indeces.device,dtype=top_k_indeces.dtype).view(-1,1).expand(-1,k)
               
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
        self.encoder_zwhat = EncoderConv(params, dim_z=params['ZWHAT.dim'], name="z_what")
        self.encoder_zmask = EncoderConv(params, dim_z=params['ZMASK.dim'], name="z_mask")

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
        # from the small stuff obtaine dby catting along the channel dims
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
