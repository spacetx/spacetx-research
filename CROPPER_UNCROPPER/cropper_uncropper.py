import torch
import torch.nn.functional as F
import collections    

class Cropper(torch.nn.Module):
    """ Use STN to crop out a patch of the original images according to z_where. 
        It uses REFLECTION padding """
    
    def __init__(self, params: dict):
        super().__init__()
        self.cropped_width = params['SD.width']
        self.cropped_height = params['SD.width']
        
    def forward(self, z_where=None, uncropped_imgs=None):

        # Prepare the shapes
        assert len(uncropped_imgs.shape) == 4  # batch, ch, width, height
        ch, width_raw, height_raw = uncropped_imgs.shape[-3:]
        large_dependent_dim = list(uncropped_imgs.shape[-3:])  # ch, width, height
        small_dependent_dim = [ch, self.cropped_width, self.cropped_height]

        # Compute the affine matrix
        affine = self.compute_affine_cropper(z_where=z_where, width_raw=width_raw, height_raw=height_raw)
        independent_dim = list(affine.shape[:-2])  # this extract n_boxes, batch

        # The cropped and uncropped imgs have:
        # a. same independent dimension (boxes, batch)
        # b. same channels
        # c. different width and height
        #  Note that I replicate the uncropped image n_boxes times
        uncropped_imgs = uncropped_imgs.unsqueeze(0).expand(independent_dim +
                                                            large_dependent_dim).reshape([-1] + large_dependent_dim)
        cropped_images = torch.zeros(independent_dim + small_dependent_dim, dtype=uncropped_imgs.dtype,
                                     device=uncropped_imgs.device).view([-1] + small_dependent_dim)
        grid = F.affine_grid(affine.view(-1, 2, 3), cropped_images.shape)
        cropped_images = F.grid_sample(uncropped_imgs, grid, mode='bilinear', padding_mode='reflection')
        return cropped_images.view(independent_dim + small_dependent_dim)

    @staticmethod
    def compute_affine_cropper(z_where=None, width_raw=None, height_raw=None):
        """ Source is UNCROPPED (large) image
            Target is CROPPED (small) image.
            
            The equations are:
            | x_s |   | sx  0   kx | | x_t |   | sx  0  | | x_t |   | kx |
            |     | = |            | | y_t | = | 0   sy | | y_t | + | ky |
            | y_s |   | 0   sy  ky | | 1   |     
            We can evaluate the expression above at:
            a. target (0,0) <===> source (-1+2*bx_dimfull/WIDTH,-1+2*by_dimfull/HEIGHT)
            b. target (1,1) <===> source (-1+2*(bx_dimfull+0.5*bw_dimfull)/WIDTH,-1+2*(by_dimfull+0.5*bh_dimfull)/HEIGHT)
        
            This leads to:
            a. kx = -1+2*bx_dimfull/WIDTH
            b. ky = -1+2*by_dimfull/HEIGHT
            c. sx = bw_dimfull/WIDTH
            d. sy = bh_dimfull/HEIGHT
        """ 
        independent_dim = list(z_where.bx.shape[:-1])

        kx = (-1.0+2*z_where.bx/width_raw).view(-1, 1)
        ky = (-1.0+2*z_where.by/height_raw).view(-1, 1)
        sx = (z_where.bw/width_raw).view(-1, 1)
        sy = (z_where.bh/height_raw).view(-1, 1)
        zero = torch.zeros_like(kx)

        affine = torch.cat((sy, zero, ky, zero, sx, kx), dim=-1)
        return affine.view(independent_dim + [2, 3])

    
class Uncropper(torch.nn.Module):
    """ Use STN to uncrop the original images according to z_where. """
    
    def __init__(self):
        super().__init__()

    def forward(self, z_where=None, small_stuff=None, width_big=None, height_big=None):

        # Check and prepare the sizes
        ch = small_stuff.shape[-3]  # this is the channels
        independent_dim = list(small_stuff.shape[:-3])  # this includes: boxes, batch
        small_dependent_dim = list(small_stuff.shape[-3:])  # this includes: ch, small_width, small_height
        large_dependent_dim = [ch, width_big, height_big]  # these are the dependent dimensions

        # Compute the affine matrix. Note that z_where has only independent dimensions
        affine_matrix = self.compute_affine_uncropper(z_where=z_where, width_raw=width_big, height_raw=height_big)
        assert affine_matrix.shape == torch.Size(independent_dim + [2, 3])

        # The cropped and uncropped imgs have:
        # a. same independent dimension (enumeration, boxes, batch)
        # b. same channels
        # c. different width and height
        uncropped_stuff = torch.zeros(independent_dim + large_dependent_dim,
                                      dtype=small_stuff.dtype,
                                      device=small_stuff.device).view([-1] + large_dependent_dim)
        grid = F.affine_grid(affine_matrix.view(-1, 2, 3), uncropped_stuff.shape)
        uncropped_stuff = F.grid_sample(small_stuff.view([-1] + small_dependent_dim), grid,
                                        mode='bilinear', padding_mode='zeros')
        return uncropped_stuff.view(independent_dim + large_dependent_dim)

    @staticmethod
    def compute_affine_uncropper(z_where=None, width_raw=None, height_raw=None):
        """ Source is CROPPED (small) image
            Target is UNCROPPED (large) image.
            
            The equations are:
            | x_s |   | sx  0   kx | | x_t |   | sx  0  | | x_t |   | kx |
            |     | = |            | | y_t | = | 0   sy | | y_t | + | ky |
            | y_s |   | 0   sy  ky | | 1   |     
            We can evaluate the expression above at:
            a. source (0,0) <===> target (-1+2*bx_dimfull/WIDTH,-1+2*by_dimfull/HEIGHT)
            b. source (1,1) <===> target (-1+2*(bx_dimfull+0.5*bw_dimfull)/WIDTH,-1+2*(by_dimfull+0.5*bh_dimfull)/HEIGHT)
        
            This leads to:
            a. kx = (WIDTH-2*bx_dimfull)/bw_dimfull
            b. ky = (WIDTH-2*bx_dimfull)/bh_dimfull
            c. sx = WIDTH/bw_dimfull
            d. sy = HEIGHT/bh_dimfull
        """
        independent_dim = list(z_where.bx.shape[:-1])

        kx = ((width_raw-2*z_where.bx)/z_where.bw).view(-1, 1)
        ky = ((height_raw-2*z_where.by)/z_where.bh).view(-1, 1)
        sx = (width_raw/z_where.bw).view(-1, 1)
        sy = (height_raw/z_where.bh).view(-1, 1)
        zero = torch.zeros_like(kx)

        affine = torch.cat((sy, zero, ky, zero, sx, kx), dim=-1)
        return affine.view(independent_dim + [2, 3])
