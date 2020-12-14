import torch
import torch.nn as nn
import torch.nn.functional as F
from MODULES.namedtuple import ZZ, BB
from MODULES.utilities import tmaps_to_bb

EPS_STD = 1E-3  # standard_deviation = F.softplus(x) + EPS_STD >= EPS_STD
LOW_RESOLUTION_BG = (5, 5)
CH_BG_MAP = 32


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

        self.bg_map_before = nn.Conv2d(in_channels=ch_in, out_channels=CH_BG_MAP, kernel_size=1)
        self.adaptive_avg_2D = nn.AdaptiveAvgPool2d(output_size=LOW_RESOLUTION_BG)
        self.adaptive_max_2D = nn.AdaptiveAvgPool2d(output_size=LOW_RESOLUTION_BG)

        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=2 * CH_BG_MAP, out_channels=4 * CH_BG_MAP, kernel_size=1),  # 5x5
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4 * CH_BG_MAP, out_channels=8 * CH_BG_MAP, kernel_size=3),  # 3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8 * CH_BG_MAP, out_channels=8 * CH_BG_MAP, kernel_size=3),  # 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8 * CH_BG_MAP, out_channels=2 * self.dim_z, kernel_size=1)  # 1x1
        )

    def forward(self, x: torch.Tensor) -> ZZ:
        # TODO: see how fast.ai does UNET
        y1 = self.bg_map_before(x)  # B, 32, small_w, small_h
        y2 = torch.cat((self.adaptive_avg_2D(y1), self.adaptive_max_2D(y1)), dim=-3)  # B, 2*ch_bg_map, low_res, low_res
        y3 = self.convolutional(y2)  # B, 2*dim_z, 1, 1
        mu, std = torch.split(y3[..., 0, 0], self.dim_z, dim=-1)  # B, dim_z
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


class EncoderInstance(nn.Module):
    def __init__(self, size: int, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in: int = ch_in
        self.width: int = size
        assert self.width == 28
        # assert (self.width == 28 or self.width == 56)
        self.dim_z = dim_z

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.ch_in, out_channels=32, kernel_size=4, stride=1, padding=2),  # 28,28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 14,14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 7,7
        )

        self.linear = nn.Linear(in_features=64*7*7, out_features=2*self.dim_z)

    def forward(self, x: torch.Tensor) -> ZZ:  # this is right

        independent_dim = list(x.shape[:-3])  # this might includes: enumeration, n_boxes, batch_size
        # dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height
        # assert dependent_dim == [self.ch_raw_image, self.width, self.width]
        x1 = x.flatten(end_dim=-4)  # flatten the independent dimensions so that can apply conv2D -> (*,ch,w,h)
        x2 = self.conv(x1).flatten(start_dim=-3)  # flatten the dependent dimension -> (*, ch*w*h)
        x3 = self.linear(x2).view(independent_dim + [2*self.dim_z])  # reshape the independent dimensions
        mu, std = torch.split(x3, self.dim_z, dim=-1)  # B, dim_z
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


# ------ DECODER --------------------

class DecoderWhere(nn.Module):
    def __init__(self, dim_z: int):
        super().__init__()
        self.dim_z = dim_z

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_z, out_channels=8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    @staticmethod
    def tmaps_to_bb(self, tmaps, width_raw_image: int, height_raw_image: int, min_box_size: float, max_box_size: float):
        tx_map, ty_map, tw_map, th_map = torch.split(tmaps, 1, dim=-3)
        n_width, n_height = tx_map.shape[-2:]
        ix_array = torch.arange(start=0, end=n_width, dtype=tx_map.dtype, device=tx_map.device)
        iy_array = torch.arange(start=0, end=n_height, dtype=tx_map.dtype, device=tx_map.device)
        ix_grid, iy_grid = torch.meshgrid([ix_array, iy_array])

        bx_map: torch.Tensor = width_raw_image * (ix_grid + tx_map) / n_width
        by_map: torch.Tensor = height_raw_image * (iy_grid + ty_map) / n_height
        bw_map: torch.Tensor = min_box_size + (max_box_size - min_box_size) * tw_map
        bh_map: torch.Tensor = min_box_size + (max_box_size - min_box_size) * th_map
        return BB(bx=convert_to_box_list(bx_map).squeeze(-1),
                  by=convert_to_box_list(by_map).squeeze(-1),
                  bw=convert_to_box_list(bw_map).squeeze(-1),
                  bh=convert_to_box_list(bh_map).squeeze(-1))

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
    def __init__(self, dim_z: int, ch_out: int):
        super().__init__()
        self.dim_z = dim_z
        self.ch_out = ch_out

        self.upsample = nn.Sequential(
            nn.Linear(in_features=self.dim_z, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=5 * 5 * 32),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 10,10
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 20,20
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),  # 40,40
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=self.ch_out, kernel_size=4, stride=2, padding=1)  # 80,80
        )

    def forward(self, z: torch.Tensor, high_resolution: tuple) -> torch.Tensor:
        # From (B, dim_z) to (B, ch_out, 28, 28) to (B, ch_out, w_raw, h_raw)
        x0 = self.upsample(z).view(-1, 32, 5, 5)
        x1 = self.decoder(x0)  # B, ch_out, 80, 80
        return F.interpolate(x1, size=high_resolution, mode='bilinear', align_corners=True)


class DecoderInstance(nn.Module):
    def __init__(self, size: int, dim_z: int, ch_out: int):
        super().__init__()
        self.width = size
        assert self.width == 28
        self.dim_z = dim_z
        self.ch_out = ch_out

        self.upsample = nn.Linear(in_features=self.dim_z, out_features=5 * 5 * 64)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=2),  # 8,8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2),  # 14,14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.ch_out, kernel_size=4, stride=2, padding=1)  # 28,28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        independent_dim = list(z.shape[:-1])
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 5, 5)
        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])


