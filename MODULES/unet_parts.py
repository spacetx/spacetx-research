import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_1by1(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch_hidden = (self.ch_in + self.ch_out) // 2
        self.mlp_1by1 = nn.Sequential(
            nn.Conv2d(self.ch_in, self.ch_hidden, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ch_hidden, self.ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_1by1(x)


class PredictBackground(nn.Module):
    """ Predict the bg_mu in (0,1) by applying sigmoid"""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.predict = MLP_1by1(ch_in=ch_in, ch_out=ch_out)

    def forward(self, x, desired_w, desired_h):
        small_bg_mu = torch.sigmoid(self.predict(x))
        return F.interpolate(small_bg_mu, size=[desired_w, desired_h], mode='bilinear', align_corners=True)


class DoubleConvolutionBlock(nn.Module):
    """ [ conv(f=3,p=1,s=1) => BN => ReLU ] x 2
        NOT: [ BN => conv(f=3,p=1,s=1) => ReLU ] x 2
        The spatial extension of the input is unchanged (because f=3, p=1, s=1).
        The convolutional layers have NO bias since they are followed by BATCH NORM
        The number of channels is passed as a parameter: ch_in -> ch_out -> ch_out 
    """
    # Is this the way to define class constants?
    PADDING = 1
    FILTER_SIZE = 3

    def __init__(self, ch_in: int, ch_out: int):
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
        
    def forward(self, x, verbose=False):
        y = self.double_conv(x)
        if verbose:
            print("input -> output", x.shape, y.shape)
        return y
    

class DownBlock(nn.Module):
    """ Performs:  max_pool(2x2) + double_convolutions
        Note that at initialization you need to specify: ch_in, ch_out
    """
        
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.max_pool_layer = nn.MaxPool2d(2, 2)
        self.double_conv = DoubleConvolutionBlock(ch_in, ch_out)
        self.s_p_k = [[2.0, 0.0, 2.0], [1.0, 2.0, 5.0]]  # max_pool, double_conv values in a list

    def __add_to_spk_list__(self, spk_list):
        spk_list.append(self.s_p_k[0])
        spk_list.append(self.s_p_k[1])
        return spk_list
        
    def forward(self, x0, verbose=False):

        x1 = self.max_pool_layer.forward(x0)
        x2 = self.double_conv.forward(x1)

        if verbose:
            print("input -> output", x0.shape, x2.shape)

        return x2


class UpBlock(nn.Module):
    """ Performs: up_conv(f=2,s=2) + concatenation + double_convolutions
        During upconv the channels go from ch_in to ch_in/2
        Since I am concatenating with something which has ch_in/2
        During the double_convolution the channels go from ch_in,ch_out
        Therefore during initialization only ch_in,ch_out of the double convolution need to be specified.
        The forward function takes two tensors: x_from_contracting_path , x_to_upconv
    """
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.ch_in = ch_in
        self.up_conv_layer = nn.ConvTranspose2d(ch_in, int(ch_in/2), kernel_size=2, stride=2, padding=0)
        self.double_conv = DoubleConvolutionBlock(ch_in, ch_out)
        self.s_p_k = [[0.5, 0.75, 2.0], [1.0, 2.0, 5.0]]
    
    def __add_to_spk_list__(self, spk_list):
        spk_list.append(self.s_p_k[0])
        spk_list.append(self.s_p_k[1])
        return spk_list

    def forward(self, x_from_compressing_path, x_to_upconv, verbose=False):
        x = self.up_conv_layer.forward(x_to_upconv)
        x1 = torch.cat((x_from_compressing_path, x), dim=-3)  # concatenate along the channel dimension

        if verbose:
            print("x_from_compressing_path", x_from_compressing_path.shape)
            print("x_to_upconv", x_to_upconv.shape)
            print("after_upconv", x.shape)
            print("after concat", x1.shape)
            print("ch_in", self.ch_in)

        x2 = self.double_conv.forward(x1)

        return x2
