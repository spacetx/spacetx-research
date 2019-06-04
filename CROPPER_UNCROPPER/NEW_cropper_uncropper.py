import torch
import torch.nn.functional as F
import collections    

class Cropper(torch.nn.Module):
    """ Use STN to crop out a patch of the original images according to z_where. 
        It uses REFLECTION padding """
    
    def __init__(self, params: dict):
        super().__init__()
        self.cropped_width  = params['SD.width']
        self.cropped_height = params['SD.width']

    def forward(self,uncropped_stuff,box_dimfull,width_dimfull,height_dimfull):
        batch_size,ch,width_raw,height_raw = uncropped_stuff.shape
       
        # Compute the affine matrix
        affine_matrix = self.compute_affine_cropper(box_dimfull,width_dimfull,height_dimfull) 

        # Add an index for the n_boxes and extend it.
        N = affine_matrix.shape[0]
        n_boxes = int(N/batch_size)
        uncropped_stuff = uncropped_stuff.unsqueeze(1).expand(-1,n_boxes,-1,-1,-1).contiguous().view(N,ch,width_raw,height_raw)
                
        # Create grid to sample the input at all the location necessary for the output
        cropped_stuff = uncropped_stuff.new_zeros((N,ch,self.cropped_width,self.cropped_height)) 
        grid = F.affine_grid(affine_matrix, cropped_stuff.shape) 
        cropped_stuff = F.grid_sample(uncropped_stuff, grid, mode='bilinear', padding_mode='reflection')         
        return cropped_stuff.view(batch_size,n_boxes,ch,self.cropped_width,self.cropped_height)
        
        
    def compute_affine_cropper(self,box_dimfull,width_dimfull,height_dimfull):
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
        kx = (-1.0+2*box_dimfull.bx_dimfull/width_dimfull).view(-1,1)
        ky = (-1.0+2*box_dimfull.by_dimfull/height_dimfull).view(-1,1)
        sx = (box_dimfull.bw_dimfull/width_dimfull).view(-1,1)
        sy = (box_dimfull.bh_dimfull/height_dimfull).view(-1,1)
        zero = torch.zeros_like(kx)
        affine = torch.cat((zero,sx,sy,kx,ky), dim=-1)
        indeces_resampling = torch.LongTensor([1, 0, 3, 0, 2, 4]).to(affine.device) # indeces to sample: sx,0,kx,0,sy,ky
        return torch.index_select(affine, 1, indeces_resampling).view(-1,2,3) 

    
class Uncropper(torch.nn.Module):
    """ Use STN to uncrop the original images according to z_where. """
    
    def __init__(self, params: dict):
        super().__init__()

    def forward(self,cropped_stuff,box_dimfull,width_raw,height_raw):
        batch_size,ch,cropped_width,cropped_height = cropped_stuff.shape
       
        # Compute the affine matrix
        affine_matrix = self.compute_affine_uncropper(box_dimfull,width_raw,height_raw) 
        assert(affine_matrix.shape[0] == batch_size)

        # The cropped and uncropped stuff have:
        # a. same batch and channel dimension
        # b. different width and height
        uncropped_stuff = cropped_stuff.new_zeros((batch_size,ch,width_raw,height_raw)) 
        grid = F.affine_grid(affine_matrix, uncropped_stuff.shape) 
        uncropped_stuff = F.grid_sample(cropped_stuff, grid, mode='bilinear', padding_mode='zeros')  
        
        # unbox the first two dimensions and return
        n1,n2 = box_dimfull.bx_dimfull.shape[:2]
        return uncropped_stuff.view(n1,n2,ch,width_raw,height_raw) 
  
    def compute_affine_uncropper(self,box_dimfull,width_raw,height_raw):
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
        kx = ((width_raw-2*box_dimfull.bx_dimfull)/box_dimfull.bw_dimfull).view(-1,1)
        ky = ((height_raw-2*box_dimfull.by_dimfull)/box_dimfull.bh_dimfull).view(-1,1)
        sx = (width_raw/box_dimfull.bw_dimfull).view(-1,1)
        sy = (height_raw/box_dimfull.bh_dimfull).view(-1,1)
        zero = torch.zeros_like(kx)
        affine = torch.cat((zero,sx,sy,kx,ky), dim=-1)
        indeces_resampling = torch.LongTensor([1, 0, 3, 0, 2, 4]).to(affine.device) # indeces to sample: sx,0,kx,0,sy,ky
        return torch.index_select(affine, 1, indeces_resampling).view(-1,2,3) 
