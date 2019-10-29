import torch
import torch.nn as nn
import collections
import torch.nn.functional as F


class DecoderConv(torch.nn.Module):
    """ Decode z -> x
        INPUT:  z of shape: ..., dim_z 
        OUTPUT: image of shape: ..., ch_out, width, height 
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
    """
    def __init__(self, params, dim_z=None, ch_out=None):
        super().__init__()
        self.width = params['SD.width']
        assert self.width == 28
        self.dim_z = dim_z
        self.ch_out = ch_out

        self.upsample = torch.nn.Linear(self.dim_z, 64 * 7 * 7)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, self.ch_out, 4, 1, 2)  # B, ch, 28, 28
        )

    def forward(self, z):
        # assert z.shape[-1] == self.dim_z
        independent_dim = list(z.shape[:-1])
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 7, 7)
        # assert x1.shape[-3:] == (self.ch_out, self.width, self.width)
        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])


class EncoderConv(torch.nn.Module):
    """ Encode x -> z_mu, z_std
        INPUT  x of shape: ..., ch_raw_image, width, height 
        OUTPUT z_mu, z_std of shape: ..., latent_dim
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
    """ 
    
    def __init__(self, params, dim_z=None):
        super().__init__()
        self.ch_raw_image = len(params["IMG.ch_in_description"])
        self.width = params["SD.width"]
        assert self.width == 28
        self.dim_z = dim_z

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.ch_raw_image, 32, 4, 1, 2),  # B, 32, 28, 28
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),  # B, 32, 14, 14
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, 1),  # B, 64,  7, 7
        )
        self.compute_mu = nn.Linear(64 * 7 * 7, self.dim_z)
        self.compute_std = nn.Linear(64 * 7 * 7, self.dim_z)

    def forward(self, x):  # this is right

        independent_dim = list(x.shape[:-3])  # this might includes: enumeration, n_boxes, batch_size
        dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height
        # assert dependent_dim == [self.ch_raw_image, self.width, self.width]
        x1 = x.view([-1] + dependent_dim)  # flatten the independent dimensions
        x2 = self.conv(x1).view(-1, 64*7*7)  # flatten the dependent dimension
        mu = self.compute_mu(x2).view(independent_dim + [self.dim_z])
        std = F.softplus(self.compute_std(x2)).view(independent_dim + [self.dim_z])
        return mu, std
