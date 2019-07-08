# sub-parts of the Unet

import torch
import collections

def convert_to_box_list(x):
    """ takes x of shape: (batch x ch x width x height) 
        and returns a list: batch x n_list x ch
        where n_list = width x height
    """
    batch_size, ch, width, height = x.shape
    return x.permute(0,2,3,1).view(batch_size,-1,ch)
  
class prediction_Zwhere_Zmask(torch.nn.Module):
    def __init__(self, channel_in, params: dict):
        super().__init__()
        self.compute_zwhere = prediction_Zwhere(channel_in,params)
        self.compute_zmask = prediction_Zmask(channel_in,params)
        
    def forward(self, x, width_raw_image,height_raw_image):
        z_where_prediction = self.compute_zwhere(x,width_raw_image,height_raw_image)
        z_mask_prediction  = self.compute_zmask(x)
        return collections.namedtuple('z_prediction', 'z_where z_mask')._make([z_where_prediction,z_mask_prediction])

class prediction_Zwhere_Zwhat_Zmask(torch.nn.Module):
    def __init__(self, channel_in, params: dict):
        super().__init__()
        self.compute_zwhere = prediction_Zwhere(channel_in,params)
        self.compute_zwhat = prediction_Zwhat(channel_in,params)
        self.compute_zmask = prediction_Zmask(channel_in,params)
        
    def forward(self, x, width_raw_image,height_raw_image):
        z_where_prediction = self.compute_zwhere(x,width_raw_image,height_raw_image)
        z_what_prediction  = self.compute_zwhat(x)
        z_mask_prediction  = self.compute_zmask(x)
        return collections.namedtuple('z_prediction', 'z_where z_what z_mask')._make([z_where,z_what,z_mask])

class prediction_Zmask(torch.nn.Module):
    """ Input  shape: batch x ch x width x height 
        Output: namedtuple with gamma_theta,z_std
        Each one of them has shape: batch x n_boxes x Zwhat_dim
        where n_boxes = width x height
    """ 
    def __init__(self, channel_in, params: dict):
        super().__init__()
        self.ch_in = channel_in
        self.Zmask_dim = params['ZMASK.dim']
        self.comp_z_mu  = torch.nn.Conv2d(self.ch_in,self.Zmask_dim,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_z_std = torch.nn.Conv2d(self.ch_in,self.Zmask_dim,kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self,x):
        z_mu  = self.comp_z_mu(x) 
        z_std = torch.exp(self.comp_z_std(x)) 
        return collections.namedtuple('z_mask', 'z_mu z_std')._make([convert_to_box_list(z_mu),convert_to_box_list(z_std)])
    
class prediction_Zwhat(torch.nn.Module):
    """ Input  shape: batch x ch x width x height 
        Output: namedtuple with z_mu,z_std
        Each one of them has shape: batch x n_boxes x Zwhat_dim
        where n_boxes = width x height
    """ 
    def __init__(self, channel_in, params: dict):
        super().__init__()
        self.ch_in = channel_in
        self.Zwhat_dim = params['ZWHAT.dim']
        self.comp_z_mu  = torch.nn.Conv2d(self.ch_in,self.Zwhat_dim,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_z_std = torch.nn.Conv2d(self.ch_in,self.Zwhat_dim,kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self,x):
        z_mu  = self.comp_z_mu(x) 
        z_std = torch.exp(self.comp_z_std(x))
        return collections.namedtuple('z_what', 'z_mu z_std')._make([convert_to_box_list(z_mu),convert_to_box_list(z_std)])

class prediction_Zwhere(torch.nn.Module):
    """ Input  shape: batch x ch x width x height 
        Output: namedtuple with all dimless stuff: prob,bx,by,bw,bh
        Each one of them has shape: batch x n_boxes x 1
    """ 
    def __init__(self, channel_in,params: dict):
        super().__init__()
        self.ch_in = channel_in
        self.comp_p     = torch.nn.Conv2d(self.ch_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_tx    = torch.nn.Conv2d(self.ch_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_ty    = torch.nn.Conv2d(self.ch_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_tw    = torch.nn.Conv2d(self.ch_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_th    = torch.nn.Conv2d(self.ch_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.size_min   = params['PRIOR.size_object_min']
        self.size_max   = params['PRIOR.size_object_max']
        self.size_delta = self.size_max - self.size_min
    
        # Here I am initializing the bias with large value so that p also has large value
        # This in turns helps the model not to get stuck in the empty configuration which is a local minimum
        self.comp_p.bias.data += 1.0


    def forward(self,x,width_raw_image,height_raw_image):
        
        batch, ch, n_width, n_height = x.shape 
        
        with torch.no_grad():
            ix = torch.arange(0,n_width,  dtype=x.dtype, device=x.device).view(1,1,-1,1) #between 0 and 1
            iy = torch.arange(0,n_height, dtype=x.dtype, device=x.device).view(1,1,1,-1) #between 0 and 1
            
        # probability
        logit_p = torch.sigmoid(self.comp_p(x))

        # center of bounding box from dimfull to dimless and reshaping
        bx = torch.sigmoid(self.comp_tx(x)) + ix #-- in (0,n_width)
        by = torch.sigmoid(self.comp_ty(x)) + iy #-- in (0,n_height)
        bx_dimfull = width_raw_image*bx/n_width  # in (0,width_raw_image)
        by_dimfull = height_raw_image*by/n_height # in (0,height_raw_image)  
        
        # size of the bounding box
        bw_dimless = torch.sigmoid(self.comp_tw(x)) # between 0 and 1
        bh_dimless = torch.sigmoid(self.comp_th(x)) # between 0 and 1
        bw_dimfull = self.size_min + self.size_delta*bw_dimless # in (min_size,max_size)
        bh_dimfull = self.size_min + self.size_delta*bh_dimless # in (min_size,max_size)
        
        return collections.namedtuple('z_where', 'prob bx_dimfull by_dimfull bw_dimfull bh_dimfull')._make(
            [convert_to_box_list(logit_p), 
             convert_to_box_list(bx_dimfull), 
             convert_to_box_list(by_dimfull), 
             convert_to_box_list(bw_dimfull), 
             convert_to_box_list(bh_dimfull)])
    

class double_convolution_block(torch.nn.Module):
    """ [ conv(f=3,p=1,s=1) => BN => ReLU ] x 2
        The spatial extension of the input is unchanged (because f=3, p=1, s=1).
        The convolutional layers have NO bias since they are followed by BATCH NORM
        The number of channels is passed as a parameter: ch_in -> ch_out -> ch_out 
    """
    # Is this the way to define class constants?
    PADDING = 1
    FILTER_SIZE = 3

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.ReplicationPad2d(self.PADDING), #reflection padding
            torch.nn.Conv2d(ch_in,ch_out,self.FILTER_SIZE, bias=False),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.ReLU(),
            torch.nn.ReplicationPad2d(self.PADDING), #reflection padding
            torch.nn.Conv2d(ch_out,ch_out,self.FILTER_SIZE, bias=False),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.ReLU()
        )
        self.s_p_k = [1.0,2.0,5.0]
        
    def __add_to_spk_list__(self,spk_list):
        spk_list.append(self.s_p_k)
        return spk_list
        
    def forward(self, x):
        y = self.double_conv(x)
        #assert(torch.isnan(x).any() == False)
        #assert(torch.isnan(y).any() == False)
        return y
    

class down_block(torch.nn.Module):
    """ Performs:  max_pool(2x2) + double_convolutions
        Note that at initialization you need to specify: ch_in, ch_out
    """
        
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.max_pool_layer = torch.nn.MaxPool2d(2, 2)
        self.double_conv = double_convolution_block(ch_in, ch_out)
        self.s_p_k = [[2.0,0.0,2.0],[1.0,2.0,5.0]]

    def __add_to_spk_list__(self,spk_list):
        spk_list.append(self.s_p_k[0])
        spk_list.append(self.s_p_k[1])
        return spk_list
        
    def forward(self, x):
        y = self.max_pool_layer(x)
        return self.double_conv(y)

class up_block(torch.nn.Module):
    """ Performs: up_conv(f=2,s=2) + concatenation + double_convolutions
        During upconv the channels go from ch_in to ch_in/2
        Since I am concatenating with something which has ch_in/2
        During the double_convolution the channels go from ch_in,ch_out
        Therefore during initialization only ch_in,ch_out of the double convolution need to be specified.
        The forward function takes two tensors: x_from_contracting_path , x_to_upconv
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up_conv_layer = torch.nn.ConvTranspose2d(ch_in, int(ch_in/2), kernel_size=2, stride=2, padding=0)
        self.double_conv = double_convolution_block(ch_in, ch_out)
        self.s_p_k = [[0.5,0.75,2.0],[1.0,2.0,5.0]]
    
    def __add_to_spk_list__(self,spk_list):
        spk_list.append(self.s_p_k[0])
        spk_list.append(self.s_p_k[1])
        return spk_list

    def forward(self, x_from_compressing_path, x_to_upconv):
        x = self.up_conv_layer(x_to_upconv)
        y = torch.cat((x_from_compressing_path,x), dim=1) #here dim=1 means that I am concatenating the channel dimension 
        return self.double_conv(y) 
