import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from MODULES.namedtuple import ZZ

EPS_STD = 1E-3  # standard_deviation = F.softplus(x) + EPS_STD >= EPS_STD
LOW_RESOLUTION_BG = (5, 5)
CH_BG_MAP = 32


class MLP_1by1(nn.Module):
    """ Use 1x1 convolution, if ch_hidden <= 0 there is NO hidden layer """
    def __init__(self, ch_in: int, ch_out: int, ch_hidden: int):
        super().__init__()
        if ch_hidden <= 0:
            self.mlp_1by1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.mlp_1by1 = nn.Sequential(
                nn.Conv2d(ch_in, ch_hidden, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_hidden, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_1by1(x)


class Encoder1by1(nn.Module):
    def __init__(self, ch_in: int, dim_z: int, ch_hidden: int):
        super().__init__()
        self.dim_z = dim_z
        self.predict = MLP_1by1(ch_in=ch_in, ch_out=2*self.dim_z, ch_hidden=ch_hidden)

    def forward(self, x: torch.Tensor) -> ZZ:
        mu, std = torch.split(self.predict(x), self.dim_z, dim=-3)
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


class Decoder1by1Linear(nn.Module):
    def __init__(self, dim_z: int, ch_out: int):
        super().__init__()
        # if groups=1 all inputs convolved to produce all outputs
        # if groups=in_channels each channel is convolved with its set of filters
        self.predict = nn.Conv2d(dim_z,
                                 ch_out,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True,
                                 groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)


class EncoderBackground(nn.Module):
    """ Encode bg_map into -> bg_mu, bg_std
        Use  adaptive_avg_2D adaptive_max_2D so that any input spatial resolution can work
    """

    def __init__(self, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in = ch_in
        self.dim_z = dim_z

        # self.global_avg_2D = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.global_max_2D = nn.AdaptiveMaxPool2d(output_size=1)
        self.adaptive_avg_2D = nn.AdaptiveAvgPool2d(output_size=LOW_RESOLUTION_BG)
        self.adaptive_max_2D = nn.AdaptiveMaxPool2d(output_size=LOW_RESOLUTION_BG)

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=2 * ch_in, out_channels=CH_BG_MAP, kernel_size=1, stride=1, padding=0, bias=True),  # 5x5
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH_BG_MAP, out_channels=CH_BG_MAP, kernel_size=5),  # 1x1
            nn.Flatten(start_dim=-3),  # batch, channels
            nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=CH_BG_MAP, out_features=2*self.dim_z))

    def forward(self, x: torch.Tensor) -> ZZ:

        # y_global = torch.cat((self.global_avg_2D(x), self.global_max_2D(x)), dim=-3) # B, ch_in , 1 , 1
        print("DEBUG",x.shape)
        y_spatial = torch.cat((self.adaptive_avg_2D(x), self.adaptive_max_2D(x)), dim=-3)  # B, ch_in, 5, 5
        mu, std = torch.split(self.encode(y_spatial), self.dim_z, dim=-1)  # B, dim_z
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


class DecoderBackground(nn.Module):
    """ Encode x -> z_mu, z_std
        INPUT  x of shape: ..., ch_raw_image, width, height
        OUTPUT z_mu, z_std of shape: ..., latent_dim
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.

        Observation ConvTranspose2D with:
        1. k=4, s=2, p=1 -> double the spatial dimension
    """
    def __init__(self, ch_out: int, dim_z: int):
        super().__init__()
        self.dim_z = dim_z
        self.ch_out = ch_out
        self.upsample = nn.Linear(self.dim_z, CH_BG_MAP * LOW_RESOLUTION_BG[0] * LOW_RESOLUTION_BG[1])
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=CH_BG_MAP, out_channels=32, kernel_size=4, stride=2, padding=1),  # 10,10
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 20,20
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),  # 40,40
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=self.ch_out, kernel_size=4, stride=2, padding=1),  # 80,80
        )

    def forward(self, z: torch.Tensor, high_resolution: tuple) -> torch.Tensor:
        # From (B, dim_z) to (B, ch_out, 28, 28) to (B, ch_out, w_raw, h_raw)
        x0 = self.upsample(z).view(-1, CH_BG_MAP, LOW_RESOLUTION_BG[0], LOW_RESOLUTION_BG[0])
        x1 = self.decoder(x0)  # B, ch_out, 80, 80
        return F.interpolate(x1, size=high_resolution, mode='bilinear', align_corners=True)


class DecoderConv(nn.Module):
    def __init__(self, size: int, dim_z: int, ch_out: int):
        super().__init__()
        self.width = size
        assert (self.width == 28 or self.width == 56)
        self.dim_z: int = dim_z
        self.ch_out: int = ch_out
        self.upsample = nn.Linear(self.dim_z, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, self.ch_out, 4, 1, 2)  # B, ch, 28, 28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        independent_dim = list(z.shape[:-1])
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 7, 7)
        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])


class EncoderConv(nn.Module):
    def __init__(self, size: int, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in: int = ch_in
        self.width: int = size
        assert (self.width == 28 or self.width == 56)
        self.dim_z = dim_z

        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.ch_in, out_channels=32, kernel_size=4, stride=1, padding=2),  # 28,28
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 14,14
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 7,7
        )

        self.compute_mu = nn.Linear(64 * 7 * 7, self.dim_z)
        self.compute_std = nn.Linear(64 * 7 * 7, self.dim_z)

    def forward(self, x: torch.Tensor) -> ZZ:  # this is right

        independent_dim = list(x.shape[:-3])  # this might includes: enumeration, n_boxes, batch_size
        dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height
        # assert dependent_dim == [self.ch_raw_image, self.width, self.width]
        x1 = x.view([-1] + dependent_dim)  # flatten the independent dimensions
        x2 = self.conv(x1).view(-1, 64 * 7 * 7)  # flatten the dependent dimension
        mu = self.compute_mu(x2).view(independent_dim + [self.dim_z])
        std = F.softplus(self.compute_std(x2)).view(independent_dim + [self.dim_z])
        return ZZ(mu=mu, std=std + EPS_STD)


####class DecoderConv(nn.Module):
####    """ Decode z -> x
####        INPUT:  z of shape: ..., dim_z
####        OUTPUT: image of shape: ..., ch_out, width, height
####        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
####
####        Observation ConvTranspose2D with:
####        1. k=4, s=2, p=1 -> double spatial dimensions
####        2. k=3, s=1, p=1 -> keep spatial dimensions
####    """
####
####    def __init__(self, size: int, dim_z: int, ch_out: int):
####        super().__init__()
####        self.dim_z = dim_z
####        self.ch_out = ch_out
####        self.width = size
####        assert (self.width == 28 or self.width == 56)
####
####        self.upsample = nn.Linear(self.dim_z, 64 * 7 * 7)
####
####        if self.width == 56:
####            self.decoder = nn.Sequential(
####                torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),  # 14,14
####                torch.nn.ReLU(inplace=True),
####                torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 28,28
####                torch.nn.ReLU(inplace=True),
####                torch.nn.ConvTranspose2d(in_channels=32, out_channels=self.ch_out, kernel_size=4, stride=2, padding=1)  # 56,56
####            )
####        else:
####            self.decoder = nn.Sequential(
####                torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),  # 14,14
####                torch.nn.ReLU(inplace=True),
####                torch.nn.ConvTranspose2d(in_channels=32, out_channels=self.ch_out, kernel_size=4, stride=2, padding=1)  # 28,28
####            )
####
####    def forward(self, z: torch.Tensor) -> torch.Tensor:
####        independent_dim = list(z.shape[:-1])
####        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 7, 7)
####        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])
####
####
####class EncoderConv(nn.Module):
####    """ Encode x -> z_mu, z_std
####        INPUT  x of shape: ..., ch_raw_image, width, height
####        OUTPUT z_mu, z_std of shape: ..., latent_dim
####        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
####
####        Observation Conv2D with:
####        1. k=4, p=2, s=1 -> keep the same spatial dimension
####        2. k=4, p=1, s=2 -> reduce the spatial dimension in half
####    """
####
####    def __init__(self, size: int, ch_in: int, dim_z: int):
####        super().__init__()
####        self.dim_z = dim_z
####        self.ch_in = ch_in
####        self.width = size
####        assert (self.width == 28 or self.width == 56)
####
####        if self.width == 56:
####            self.conv = nn.Sequential(
####                torch.nn.Conv2d(in_channels=self.ch_in, out_channels=32, kernel_size=4, stride=1, padding=2),  # 56,56
####                torch.nn.ReLU(inplace=True),
####                torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 28,28
####                torch.nn.ReLU(inplace=True),
####                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 14,14
####                torch.nn.ReLU(inplace=True),
####                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),  # 7,7
####            )
####
####        elif self.width == 28:
####            self.conv = nn.Sequential(
####                torch.nn.Conv2d(in_channels=self.ch_in, out_channels=32, kernel_size=4, stride=1, padding=2),  # 28,28
####                torch.nn.ReLU(inplace=True),
####                torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 14,14
####                torch.nn.ReLU(inplace=True),
####                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 7,7
####            )
####
####        self.compute_mu = nn.Linear(64 * 7 * 7, self.dim_z)
####        self.compute_std = nn.Linear(64 * 7 * 7, self.dim_z)
####
####    def forward(self, x: torch.Tensor) -> ZZ:  # this is right
####        independent_dim = list(x.shape[:-3])  # this might includes: enumeration, n_boxes, batch_size
####        dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height
####        # assert dependent_dim == [self.ch_raw_image, self.width, self.width]
####        x1 = x.view([-1] + dependent_dim)  # flatten the independent dimensions
####        x2 = self.conv(x1).view(-1, 64 * 7 * 7)  # flatten the dependent dimension
####        mu = self.compute_mu(x2).view(independent_dim + [self.dim_z])
####        std = F.softplus(self.compute_std(x2)).view(independent_dim + [self.dim_z])
####        return ZZ(mu=mu, std=std + EPS_STD)
####
