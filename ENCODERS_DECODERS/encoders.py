import torch
import torch.nn as nn
import collections
import torch.nn.functional as F


class EncoderConv(torch.nn.Module):
    """ Encode cropped stuff into z_mu, z_std 
        INPUT  SHAPE: batch x n_boxes x ch x width x height 
        OUTPUT SHAPE: batch x n_boxes x latent_dim
        
        Architecture inspired by:
        https://www.datacamp.com/community/tutorials/cyclical-learning-neural-nets
    """ 
    
    def __init__(self, params, dim_z=None, name=None):
        super().__init__()
        self.ch_raw_image = len(params["IMG.ch_in_description"])
        self.width = params["SD.width"]
        assert self.width == 28
        self.dim_z = dim_z
        self.result = collections.namedtuple(name, "z_mu z_std")

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.ch_raw_image, 32, 4, 1, 2),   # B,  32, 28, 28
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )
        self.compute_mu = nn.Linear(64 * 7 * 7, self.dim_z)
        self.compute_std = nn.Linear(64 * 7 * 7, self.dim_z)

    def forward(self, x):

        independent_dim = list(x.shape[:-3])  # this includes: n_boxes, batch
        dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height

        x1 = x.view([-1] + dependent_dim)  # flatten the independent dimensions
        x2 = self.conv(x1).view(independent_dim + [-1])  # flatten the dependent dimensions

        return self.result(z_mu=self.compute_mu(x2), z_std=F.softplus(self.compute_std(x2)))
