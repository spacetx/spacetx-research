import torch
import torch.nn as nn
import torch.nn.functional as F
from .namedtuple import ZZ, EPS_STD
from typing import List, Optional


class MLP_1by1(nn.Module):
    """ Use 1x1 convolution, if ch_hidden <= 0 there is hidden layer, if ch_hidden = None automatic choice"""
    def __init__(self, ch_in: int, ch_out: int, ch_hidden: Optional[int] = None):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch_hidden = (self.ch_in + self.ch_out) // 2 if ch_hidden is None else ch_hidden
        if self.ch_hidden <= 0:
            self.mlp_1by1 = nn.Conv2d(self.ch_in, self.ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.mlp_1by1 = nn.Sequential(
                nn.Conv2d(self.ch_in, self.ch_hidden, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(self.ch_hidden, self.ch_out, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_1by1(x)


class PredictBackground(nn.Module):
    """ Predict the bg_mu in (0,1) by applying sigmoid"""
    def __init__(self, ch_in: int, ch_out: int, ch_hidden: Optional[int] = None):
        super().__init__()
        self.ch_out = ch_out
        self.predict = MLP_1by1(ch_in=ch_in, ch_out=2*ch_out, ch_hidden=ch_hidden)

    def forward(self, x: torch.Tensor) -> ZZ:
        mu, std = torch.split(self.predict(x), self.ch_out, dim=-3)
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)




class Decoder1by1Linear(nn.Module):
    def __init__(self, dim_z: int, ch_out: int, groups: int):
        super().__init__()
        self.dim_z: int = dim_z
        self.ch_out: int = ch_out
        self.groups: int = groups
        # if groups=1 all inputs convolved to produce all outputs
        # if groups=in_channels each channel is convolved with its set of filters
        self.predict = nn.Conv2d(self.dim_z,
                                 self.ch_out,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True,
                                 groups=self.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)
    

class DecoderConv(nn.Module):
    """ Decode z -> x
        INPUT:  z of shape: ..., dim_z 
        OUTPUT: image of shape: ..., ch_out, width, height 
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
    """
    def __init__(self, size: int, dim_z: int, ch_out: int):
        super().__init__()
        self.width = size
        assert self.width == 28
        self.dim_z: int = dim_z
        self.ch_out: int = ch_out
        self.upsample = nn.Linear(self.dim_z, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, self.ch_out, 4, 1, 2)  # B, ch, 28, 28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        independent_dim = list(z.shape[:-1])
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 7, 7)
        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])


class EncoderConv(nn.Module):
    """ Encode x -> z_mu, z_std
        INPUT  x of shape: ..., ch_raw_image, width, height 
        OUTPUT z_mu, z_std of shape: ..., latent_dim
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
    """ 
    
    def __init__(self, size: int, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in: int = ch_in
        self.width: int = size
        assert self.width == 28
        self.dim_z = dim_z

        self.conv = nn.Sequential(
            torch.nn.Conv2d(self.ch_in, 32, 4, 1, 2),  # B, 32, 28, 28
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),  # B, 32, 14, 14
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, 1),  # B, 64,  7, 7
        )
        self.compute_mu = nn.Linear(64 * 7 * 7, self.dim_z)
        self.compute_std = nn.Linear(64 * 7 * 7, self.dim_z)

    def forward(self, x: torch.Tensor) -> ZZ:  # this is right

        independent_dim = list(x.shape[:-3])  # this might includes: enumeration, n_boxes, batch_size
        dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height
        # assert dependent_dim == [self.ch_raw_image, self.width, self.width]
        x1 = x.view([-1] + dependent_dim)  # flatten the independent dimensions
        x2 = self.conv(x1).view(-1, 64*7*7)  # flatten the dependent dimension
        mu = self.compute_mu(x2).view(independent_dim + [self.dim_z])
        std = F.softplus(self.compute_std(x2)).view(independent_dim + [self.dim_z])
        return ZZ(mu=mu, std=std + EPS_STD)

# class Encoder1by1(nn.Module):
#     def __init__(self, ch_in: int, dim_z: int):
#         super().__init__()
#         self.ch_in: int = ch_in
#         self.dim_z: int = dim_z
#         self.ch_hidden = (self.ch_in + self.dim_z) // 2
#         self.predict = nn.Sequential(
#             nn.Conv2d(self.ch_in, self.ch_hidden, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.ReLU(),
#             nn.Conv2d(self.ch_hidden, 2 * self.dim_z, kernel_size=1, stride=1, padding=0, bias=True)
#         )
#
#     def forward(self, x: torch.Tensor) -> ZZ:
#         mu, std = torch.split(self.predict(x), self.dim_z, dim=-3)
#         # Apply non-linearity and return
#         return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)
#
# class MLP_to_ZZ(nn.Module):
#     def __init__(self, in_features: int, dim_z: int):
#         super().__init__()
#         self.ch_in: int = in_features
#         self.dim_z: int = dim_z
#         self.ch_hidden = (self.ch_in + self.dim_z) // 2
#         self.predict = nn.Sequential(
#             nn.Linear(in_features=self.ch_in, out_features=self.ch_hidden, bias=True),
#             nn.ReLU(),
#             nn.Linear(in_features=self.ch_hidden, out_features=2 * self.dim_z, bias=True)
#         )
#
#     def forward(self, x: torch.Tensor) -> ZZ:
#         mu, std = torch.split(self.predict(x), self.dim_z, dim=-1)
#         # Apply non-linearity and return
#         return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)
