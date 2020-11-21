import torch
import torch.nn as nn
import torch.nn.functional as F
from MODULES.namedtuple import ZZ, BB
from MODULES.utilities import tmaps_to_bb

EPS_STD = 1E-3  # standard_deviation = F.softplus(x) + EPS_STD >= EPS_STD
LOW_RESOLUTION_BG = (5, 5)
CH_BG_MAP = 16


# --------- HELPER FUNCTION ---------------------

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


# ---------- ENCODERS -------------------------------

class EncoderWhere(nn.Module):
    """ This  is a wrapper around MLP_1by1 with non-linearities for mu,std """
    def __init__(self, ch_in: int, dim_z: int, ch_hidden: int):
        super().__init__()
        self.dim_z = dim_z
        self.predict = MLP_1by1(ch_in=ch_in, ch_out=2*self.dim_z, ch_hidden=ch_hidden)

    def forward(self, x: torch.Tensor) -> ZZ:
        mu, std = torch.split(self.predict(x), self.dim_z, dim=-3)
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


class EncoderBackground(nn.Module):
    """ Encode bg_map into -> bg_mu, bg_std
        Use  adaptive_avg_2D adaptive_max_2D so that any input spatial resolution can work
    """

    def __init__(self, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in = ch_in
        self.dim_z = dim_z

        ch_hidden = (CH_BG_MAP + dim_z)//2

        self.bg_map_before = nn.Conv2d(in_channels=ch_in, out_channels=CH_BG_MAP, kernel_size=1)
        self.adaptive_avg_2D = nn.AdaptiveAvgPool2d(output_size=LOW_RESOLUTION_BG)
        self.adaptive_max_2D = nn.AdaptiveAvgPool2d(output_size=LOW_RESOLUTION_BG)

        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=2 * CH_BG_MAP, out_channels=CH_BG_MAP, kernel_size=1),  # 5x5
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH_BG_MAP, out_channels=CH_BG_MAP, kernel_size=3),  # 3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH_BG_MAP, out_channels=CH_BG_MAP, kernel_size=3),  # 1x1
            nn.ReLU(inplace=True))

        self.linear = nn.Sequential(
            nn.Linear(in_features=CH_BG_MAP, out_features=ch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=ch_hidden, out_features=2*self.dim_z))

    def forward(self, x: torch.Tensor) -> ZZ:
        # TODO: see how fast.ai does UNET
        y1 = self.bg_map_before(x)  # B, 32, small_w, small_h
        y2 = torch.cat((self.adaptive_avg_2D(y1), self.adaptive_max_2D(y1)), dim=-3)  # 2*ch_bg_map , low_res, low_res
        y3 = self.convolutional(y2)  # B, 32, 1, 1
        mu, std = torch.split(self.linear(y3.flatten(start_dim=-3)), self.dim_z, dim=-1)  # B, dim_z
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


class EncoderInstance(nn.Module):
    def __init__(self, size: int, ch_in: int, dim_z: int, leaky: bool = True):
        super().__init__()
        self.ch_in: int = ch_in
        self.width: int = size
        assert self.width == 28
        # assert (self.width == 28 or self.width == 56)
        self.dim_z = dim_z

        if leaky:
            activation = nn.LeakyReLU(inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.ch_in, out_channels=32, kernel_size=4, stride=1, padding=2),  # 28,28
            activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 14,14
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 7,7
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


# ------ DECODER --------------------

class DecoderWhere(nn.Module):
    def __init__(self, dim_z: int, leaky: bool = True):
        super().__init__()
        self.dim_z = dim_z

        if leaky:
            activation = nn.LeakyReLU(inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_z, out_channels=4, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor,
                width_raw_image: int,
                height_raw_image: int,
                min_box_size: int,
                max_box_size: int) -> BB:

        return tmaps_to_bb(tmaps=self.transform(z),
                           width_raw_image=width_raw_image,
                           height_raw_image=height_raw_image,
                           min_box_size=min_box_size,
                           max_box_size=max_box_size)


class DecoderBackground(nn.Module):
    """ Encode x -> z_mu, z_std
        INPUT  x of shape: ..., ch_raw_image, width, height
        OUTPUT z_mu, z_std of shape: ..., latent_dim
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.

        Observation ConvTranspose2D with:
        1. k=4, s=2, p=1 -> double the spatial dimension
    """
    def __init__(self, dim_z: int, ch_out: int, leaky: bool = False):
        super().__init__()
        self.dim_z = dim_z
        self.ch_out = ch_out

        if leaky:
            activation = nn.LeakyReLU(inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.upsample = nn.Sequential(
            nn.Linear(in_features=self.dim_z, out_features=28),
            activation,
            nn.Linear(in_features=28, out_features=56),
            activation,
            nn.Linear(in_features=56, out_features=128),
            activation,
            nn.Linear(in_features=128, out_features=256),
            activation,
            nn.Linear(in_features=256, out_features=32 * 4 * 4)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 8,8
            activation,
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 16,16
            activation,
            nn.ConvTranspose2d(in_channels=32, out_channels=ch_out, kernel_size=4, stride=2, padding=1)  # 32,32
        )

    def forward(self, z: torch.Tensor, high_resolution: tuple) -> torch.Tensor:
        # From (B, dim_z) to (B, ch_out, 28, 28) to (B, ch_out, w_raw, h_raw)
        x0 = self.upsample(z).view(-1, 32, 4, 4)
        x1 = self.decoder(x0)  # B, ch_out, 80, 80
        return F.interpolate(x1, size=high_resolution, mode='bilinear', align_corners=True)


class DecoderInstance(nn.Module):
    def __init__(self, size: int, dim_z: int, ch_out: int, leaky: bool = False):
        super().__init__()
        self.width = size
        assert self.width == 28
        self.dim_z = dim_z
        self.ch_out = ch_out

        if leaky:
            activation = nn.LeakyReLU(inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.upsample = nn.Sequential(
            nn.Linear(in_features=self.dim_z, out_features=28),
            activation,
            nn.Linear(in_features=28, out_features=56),
            activation,
            nn.Linear(in_features=56, out_features=128),
            activation,
            nn.Linear(in_features=128, out_features=256),
            activation,
            nn.Linear(in_features=256, out_features=32 * 4 * 4)
        )

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 8,8
                activation,
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2),  # 14,14
                activation,
                nn.ConvTranspose2d(in_channels=32, out_channels=ch_out, kernel_size=4, stride=2, padding=1)  # 28,28
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        independent_dim = list(z.shape[:-1])
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 32, 4, 4)
        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])


