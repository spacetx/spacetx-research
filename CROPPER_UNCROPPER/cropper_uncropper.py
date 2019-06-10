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
        
    def forward(self,z_where,uncropped_imgs):
        batch_size,ch,width_raw,height_raw = uncropped_imgs.shape
       
        # Compute the affine matrix
        affine_matrix = self.compute_affine_cropper(z_where,width_raw,height_raw) 

        # Add an index for the n_boxes and extend it.
        with torch.no_grad():
            N = affine_matrix.shape[0]
            n_boxes = int(N/batch_size)
        uncropped_imgs = uncropped_imgs.unsqueeze(1).expand(-1,n_boxes,-1,-1,-1).contiguous().view(N,ch,width_raw,height_raw)
                
        # Create grid to sample the input at all the location necessary for the output
        cropped_images = torch.zeros((N,ch,self.cropped_width,self.cropped_height),
                                    dtype=uncropped_imgs.dtype,device=uncropped_imgs.device)  
        grid = F.affine_grid(affine_matrix, cropped_images.shape) 
        cropped_images = F.grid_sample(uncropped_imgs, grid, mode='bilinear', padding_mode='reflection')         
        return cropped_images.view(batch_size,n_boxes,ch,self.cropped_width,self.cropped_height)
        
        
    def compute_affine_cropper(self,z_where,width_raw,height_raw):
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
        kx = (-1.0+2*z_where.bx_dimfull/width_raw).view(-1,1)
        ky = (-1.0+2*z_where.by_dimfull/height_raw).view(-1,1)
        sx = (z_where.bw_dimfull/width_raw).view(-1,1)
        sy = (z_where.bh_dimfull/height_raw).view(-1,1)
        zero = torch.zeros_like(kx)
        
        # old version (slow)
        #affine = torch.cat((zero,sy,sx,ky,kx), dim=-1)
        #indeces_resampling = torch.LongTensor([1, 0, 3, 0, 2, 4]).to(affine.device) # indeces to sample: sx,0,kx,0,sy,ky
        #old = torch.index_select(affine, 1, indeces_resampling).view(-1,2,3) 
    
        # newer version is equivalent and faster
        return torch.cat((sy,zero,ky,zero,sx,kx),dim=-1).view(-1,2,3)

    
class Uncropper(torch.nn.Module):
    """ Use STN to uncrop the original images according to z_where. """
    
    def __init__(self, params: dict):
        super().__init__()      

    def forward(self,z_where,cropped_imgs,width_raw,height_raw):
        batch_size,ch,cropped_width,cropped_height = cropped_imgs.shape
       
        # Compute the affine matrix
        affine_matrix = self.compute_affine_uncropper(z_where,width_raw,height_raw) 
        assert(affine_matrix.shape[0] == batch_size)

        # The cropped and uncropped imgs have:
        # a. same batch and channel dimension
        # b. different width and height
        uncropped_imgs = torch.zeros((batch_size,ch,width_raw,height_raw),
                                    dtype=cropped_imgs.dtype,device=cropped_imgs.device)          
        grid = F.affine_grid(affine_matrix, uncropped_imgs.shape) 
        uncropped_imgs = F.grid_sample(cropped_imgs, grid, mode='bilinear', padding_mode='zeros')  
        
        # unbox the first two dimensions and return
        n1,n2 = z_where.bx_dimfull.shape[:2]
        return uncropped_imgs.view(n1,n2,ch,width_raw,height_raw) 
  
    def compute_affine_uncropper(self,z_where,width_raw,height_raw):
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
        kx = ((width_raw-2*z_where.bx_dimfull)/z_where.bw_dimfull).view(-1,1)
        ky = ((height_raw-2*z_where.by_dimfull)/z_where.bh_dimfull).view(-1,1)
        sx = (width_raw/z_where.bw_dimfull).view(-1,1)
        sy = (height_raw/z_where.bh_dimfull).view(-1,1)
        zero = torch.zeros_like(kx)    
        
        #old version (slow)
        #affine = torch.cat((zero,sy,sx,ky,kx), dim=-1)
        #indeces_resampling = torch.LongTensor([1, 0, 3, 0, 2, 4]).to(affine.device) # indeces to sample: sx,0,kx,0,sy,ky
        #old = torch.index_select(affine, 1, indeces_resampling).view(-1,2,3) 

        # newer version is equivalent and faster
        return torch.cat((sy,zero,ky,zero,sx,kx),dim=-1).view(-1,2,3)