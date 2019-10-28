import torch
import collections
import torch.nn.functional as F


class DecoderConv(torch.nn.Module):
    """ Decode z -> x
        INPUT:  z of shape: batch x dim_z 
        OUTPUT: image of shape: batch x 1 x width x height (where width = height)
    """
    def __init__(self, params, dim_z=None, ch_out=None):
        super().__init__()
        self.width = params['SD.width']
        assert self.width == 28
        self.dim_z = dim_z
        self.ch_out = ch_out

        self.upsample = torch.nn.Linear(self.dim_z, 64 * 7 * 7)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  64,  14,  14
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1), # B,  32, 28, 28
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, self.ch_out, 4, 1, 2)   # B, 1, 28, 28
        )

    def forward(self, z):
        #assert z.shape[-1] == self.dim_z
        #independent_dim = list(z.shape[:-1])
        #x1 = self.upsample(z)
        #assert x1.shape == torch.Size(independent_dim + [64*7*7])
        #x2 = x1.view(-1, 64, 7, 7)
        #x3 = self.decoder(x2)
        #assert (self.ch_out, self.width, self.width) == x3.shape[-3:]
        #return x3.view(independent_dim + [self.ch_out, self.width, self.width])

        assert z.shape[-1] == self.dim_z
        independent_dim = list(z.shape[:-1])
        dependent_dim = [self.ch_out, self.width, self.width]
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 7, 7)
        x2 = self.decoder(x1)
        assert x2.shape[-3:] == torch.Size(dependent_dim)
        return x2.view(independent_dim + dependent_dim)
    
