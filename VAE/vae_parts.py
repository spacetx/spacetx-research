
import torch
import collections
from UNET.unet_model import UNet
from NON_MAX_SUPPRESSION.non_max_suppression import Non_Max_Suppression 
from ENCODERS_DECODERS.encoders import Encoder_CONV #,Encoder_MLP 
from ENCODERS_DECODERS.decoders import Multi_Channel_Img_Decoder, Decoder_CONV #,Decoder_MLP 
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
    area = (z_where.bw_dimfull*z_where.bh_dimfull).squeeze(-1) #this area is big for boxes out of bounds
    #area = ((x3-x1)*(y3-y1)).float()                            #this area is small for boxes out of bounds
    batch_size, n_boxes = area.shape
    
    # compute the total intensity in each box
    batch_index = torch.arange(start=0,end=batch_size,step=1,device=x1.device,dtype=x1.dtype).view(-1,1).expand(-1,n_boxes)
    tot_intensity = cum[batch_index,x3,y3]-cum[batch_index,x1,y3]-cum[batch_index,x3,y1]+cum[batch_index,x1,y1]
    assert (batch_size, n_boxes) == tot_intensity.shape
   
    # return the average intensity
    return tot_intensity/area 


def select_top_boxes_by_prob(z_where_all,n_objects_max):
    
    p=z_where_all.prob.squeeze(-1)
    batch_size,n_boxes = p.shape 
    p_top_k, top_k_indeces = torch.topk(p, k=min(n_objects_max,n_boxes), dim=-1, largest=True, sorted=True)
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

        # encoders z_what,z_mask
        #self.encoder_zwhat = Encoder_MLP(params,is_zwhat=True)
        #self.encoder_zmask = Encoder_MLP(params,is_zwhat=False)
        self.encoder_zwhat = Encoder_CONV(params,is_zwhat=True)
        self.encoder_zmask = Encoder_CONV(params,is_zwhat=False)        
    
    def forward(self,imgs_in,p_corr_factor=0.0,randomize_nms_factor=0.0,n_objects_max=10):
        raw_width,raw_height = imgs_in.shape[-2:]
        
        # UNET
        z_where_all = self.unet.forward(imgs_in,verbose=False)
        
        # CHECK
        try:
            assert torch.max(z_where_all.prob) <= 1.0
            assert torch.min(z_where_all.prob) >= 0.0
        except:
            print("JUST AFTER UNET")
            print(z_where_all.prob)
            exit()
       
        # ANNEAL THE PROBABILITIES IF NECESSARY
        if(p_corr_factor>0):
            with torch.no_grad():
                x = compute_average_intensity_in_box(imgs_in,z_where_all)
                tmp = (compute_ranking(x,dim=-1).float()/(x.shape[-1]-1)).view_as(z_where_all.prob)  # this is rescaled ranking
                #print("x.shape, tmp.shape",x.shape,tmp.shape)
                #print("print tmp[0]",tmp[0])  
                #p_approx = tmp
                p_approx = tmp.pow(10) #this is to make p_approx even more peaked
                
                # CHECK
                try:
                    assert torch.max(p_approx) <= 1.0
                    assert torch.min(p_approx) >= 0.0
                except:
                    print("inside annel probabilities")
                    print(p_approx.shape)
                    print(p_approx)
                    exit()
                
            # weighted average of the prob by the inference netwrok and probs by the comulative 
            new_p = (1-p_corr_factor)*z_where_all.prob+p_corr_factor*p_approx
            z_where_all = z_where_all._replace(prob=new_p)
            

        # CHECK
        try:
            assert torch.max(z_where_all.prob) <= 1.0
            assert torch.min(z_where_all.prob) >= 0.0
        except:
            print("BEFORE NMS")
            print(z_where_all.prob)
            exit()
        
        # NMS   
        z_where = self.nms.forward(z_where_all,randomize_nms_factor,n_objects_max)
        #z_where = select_top_boxes_by_prob(z_where_all,self.n_objects_max)
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
        cropped_imgs,sigma  = self.imgs_decoder.forward(z_what.view(-1,dim_z_what))     # shape: N,ch,Win,Hin and N,1,1,1
        uncropped_imgs      = self.uncropper(z_where,cropped_imgs,width_raw,height_raw) # shape: N,ch,Wout,Hout   
        
        # reshape
        ch,width,height = uncropped_imgs.shape[-3:]
        return uncropped_imgs.view(batch_size,n_boxes,ch,width,height),sigma.view(batch_size,n_boxes,1,1,1)
        

class Masks_Generator(torch.nn.Module):
    def __init__(self,params):
        super().__init__()
        self.mask_decoder = Decoder_CONV(params,is_zwhat=False)
        self.uncropper = Uncropper(params)
        
    def forward(self,z_where,z_mask,width_raw,height_raw):
    
        # DECODER
        batch_size,n_boxes,dim_z_mask = z_mask.shape
        cropped_masks   = self.mask_decoder.forward(z_mask.view(-1,dim_z_mask))      # shape: N,ch,Win,Hin
        uncropped_masks = self.uncropper(z_where,cropped_masks,width_raw,height_raw) # shape: N,ch,Wout,Hout 

        # reshape
        ch,width,height = uncropped_masks.shape[-3:]
        return uncropped_masks.view(batch_size,n_boxes,ch,width,height)
    

## This is just a combination of the previous two    
#class Generator(torch.nn.Module):
#    def __init__(self,params):
#        super().__init__()
#        self.imgs_decoder = Multi_Channel_Img_Decoder(params)
#        self.mask_decoder = Decoder_CONV(params,is_zwhat=False)
#        self.uncropper = Uncropper(params)
#        
#        
#    def forward(self,z_where,z_mask,z_what,width_raw,height_raw):
#        
#        assert z_what.shape[:2] == z_mask.shape[:2]
#        batch_size,n_boxes  = z_what.shape[:2]
#        cropped_masks       = self.mask_decoder.forward(z_mask.view(batch_size*n_boxes,-1)) # shape: N, 1,Win,Hin 
#        cropped_imgs,sigma  = self.imgs_decoder.forward(z_what.view(batch_size*n_boxes,-1)) # shape: N,ch,Win,Hin and N,1,1,1
#        
#        # do the uncrop 
#        to_uncrop = torch.cat((cropped_masks,cropped_imgs),dim=1) # cat along the channel dimension. Mask is first
#        uncropped = self.uncropper(z_where,to_uncrop,width_raw,height_raw)
#        
#        # reshape
#        ch_p1,width,height = uncropped.shape[-3:]
#        uncropped = uncropped.view(batch_size,n_boxes,ch_p1,width,height)
#        sigma     = sigma.view(batch_size,n_boxes,1,1,1)
#        
#        return uncropped[:,:1],uncropped[:,1:],sigma