# sub-parts of the Unet

import torch
import collections

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
