# sub-parts of the Unet

import torch
import torch.nn as nn
import collections


def convert_to_namedtuple(x):
    """ takes input of shape: n_objects, batch, ch
        and split the channel dimension and give them names.
        This function FIX the conventions
    """
    assert x.shape[-1] == 9
    return collections.namedtuple('unet_results',
           'ix iy W_over_nw H_over_nh tp_mu tx_mu ty_mu tw_mu th_mu')._make(torch.split(x, 1, dim=-1))


def convert_to_box_list(x):
    """ takes input of shape: (batch, ch, width, height)
        and returns output of shape: (n_list, batch, ch)
        where n_list = width x height
    """
    batch_size, ch, width, height = x.shape
    return x.permute(2, 3, 0, 1).view(width*height, batch_size, ch)
  

class PredictionZwhereAndProb(nn.Module):
    """ Input  shape: batch, ch_in  , width , height
        Output shape: (width x height), bathc_size, ch_out
        where ch_out = 5 because: tp, tx, ty, tw, th
    """
    def __init__(self, channel_in: int, params: dict):
        super().__init__()
        self.ch_in = channel_in
        self.comp_tmu = torch.nn.Conv2d(self.ch_in, 5, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, width_raw_image, height_raw_image):
        
        batch, ch, n_width, n_height = x.shape

        # raw_results
        t_mu = self.comp_tmu(x)

        with torch.no_grad():
            ix = torch.arange(start=0, end=n_width,  dtype=x.dtype,
                              device=x.device).view(1, 1, -1, 1).expand(batch, 1, n_width, n_height)
            iy = torch.arange(start=0, end=n_height, dtype=x.dtype,
                              device=x.device).view(1, 1, 1, -1).expand(batch, 1, n_width, n_height)
            W_over_nw = width_raw_image*torch.ones_like(ix)/n_width
            H_over_nh = height_raw_image*torch.ones_like(ix)/n_width


        return convert_to_box_list(torch.cat((ix, iy, W_over_nw, H_over_nh, t_mu), dim=-3))  # cat along ch dimension


class DoubleConvolutionBlock(torch.nn.Module):
    """ [ conv(f=3,p=1,s=1) => BN => ReLU ] x 2
        NOT: [ BN => conv(f=3,p=1,s=1) => ReLU ] x 2
        The spatial extension of the input is unchanged (because f=3, p=1, s=1).
        The convolutional layers have NO bias since they are followed by BATCH NORM
        The number of channels is passed as a parameter: ch_in -> ch_out -> ch_out 
    """
    # Is this the way to define class constants?
    PADDING = 1
    FILTER_SIZE = 3

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.ReplicationPad2d(self.PADDING),  # reflection padding
            nn.Conv2d(ch_in, ch_out, self.FILTER_SIZE, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.ReplicationPad2d(self.PADDING),  # reflection padding
            nn.Conv2d(ch_out, ch_out, self.FILTER_SIZE, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.s_p_k = [1.0, 2.0, 5.0]
        
    def __add_to_spk_list__(self, spk_list):
        spk_list.append(self.s_p_k)
        return spk_list
        
    def forward(self, x):
        y = self.double_conv(x)
        return y
    

class DownBlock(torch.nn.Module):
    """ Performs:  max_pool(2x2) + double_convolutions
        Note that at initialization you need to specify: ch_in, ch_out
    """
        
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.max_pool_layer = torch.nn.MaxPool2d(2, 2)
        self.double_conv = DoubleConvolutionBlock(ch_in, ch_out)
        self.s_p_k = [[2.0, 0.0, 2.0], [1.0, 2.0, 5.0]]  # max_pool, double_conv values in a list

    def __add_to_spk_list__(self, spk_list):
        spk_list.append(self.s_p_k[0])
        spk_list.append(self.s_p_k[1])
        return spk_list
        
    def forward(self, x):
        y = self.max_pool_layer.forward(x)
        return self.double_conv.forward(y)


class UpBlock(torch.nn.Module):
    """ Performs: up_conv(f=2,s=2) + concatenation + double_convolutions
        During upconv the channels go from ch_in to ch_in/2
        Since I am concatenating with something which has ch_in/2
        During the double_convolution the channels go from ch_in,ch_out
        Therefore during initialization only ch_in,ch_out of the double convolution need to be specified.
        The forward function takes two tensors: x_from_contracting_path , x_to_upconv
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up_conv_layer = nn.ConvTranspose2d(ch_in, int(ch_in/2), kernel_size=2, stride=2, padding=0)
        self.double_conv = DoubleConvolutionBlock(ch_in, ch_out)
        self.s_p_k = [[0.5, 0.75, 2.0], [1.0, 2.0, 5.0]]
    
    def __add_to_spk_list__(self, spk_list):
        spk_list.append(self.s_p_k[0])
        spk_list.append(self.s_p_k[1])
        return spk_list

    def forward(self, x_from_compressing_path, x_to_upconv):
        x = self.up_conv_layer.forward(x_to_upconv)
        y = torch.cat((x_from_compressing_path, x), dim=1)  # concatenate along the channel dimension
        return self.double_conv.forward(y)
