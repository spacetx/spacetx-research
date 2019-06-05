import torch
import collections
import torch.nn.functional as F



class Encoder_CONV(torch.nn.Module):
    """ Encode cropped stuff into z_mu, z_std 
        INPUT  SHAPE: batch x n_boxes x ch x width x height 
        OUTPUT SHAPE: batch x n_boxes x latent_dim
        
        Architecture inspired by:
        https://www.datacamp.com/community/tutorials/cyclical-learning-neural-nets
    """ 
    
    def __init__(self, params,is_zwhat):
        super().__init__()
        self.ch_raw_image = len(params["IMG.ch_in_description"])
        self.width  = params["SD.width"]
        assert self.width == 28
        
        if(is_zwhat):
            self.dim_z = params['ZWHAT.dim']
            self.name    = "z_what"
        else:
            self.dim_z = params['ZMASK.dim']
            self.name    = "z_mask"
        

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.ch_raw_image, 32, 4, 1, 2),   # B,  32, 28, 28
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )
        
        self.linear = torch.nn.Linear(64 * 7 * 7, 2*self.dim_z)

    def forward(self,x):
        assert len(x.shape) == 5
        batch_size,n_boxes,ch,width,height = x.shape
        x = x.view(batch_size*n_boxes,ch,width,height)
        
        x1 = self.conv(x).view(batch_size,n_boxes,-1)
        z = self.linear(x1)        
        z_mu  = z[...,:self.dim_z]
        z_std = F.softplus(z[...,self.dim_z:])
        return collections.namedtuple(self.name, "z_mu z_std")._make([z_mu,z_std])
 

class Encoder_MLP(torch.nn.Module):
    """ Encode cropped stuff into z_mu, z_std 
        INPUT  SHAPE: batch x n_boxes x ch x width x height 
        OUTPUT SHAPE: batch x n_boxes x latent_dim
    """
    
    def __init__(self,params,is_zwhat):
        
        super().__init__()    
        ch_raw_image = len(params["IMG.ch_in_description"])
        width  = params["SD.width"]
        self.dim_in = width*width*ch_raw_image
        self.dim_h1 = params["SD.dim_h1"] 
        self.dim_h2 = params["SD.dim_h2"]
        
        if(is_zwhat):
            self.dim_z = params['ZWHAT.dim']
            self.name    = "z_what"
        else:
            self.dim_z = params['ZMASK.dim']
            self.name    = "z_mask"

        if(self.dim_h1>0 and self.dim_h2>0):
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h1, self.dim_h2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h2, 2*self.dim_z)
            )
        elif(self.dim_h1 > 0 and self.dim_h2 <= 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h1, 2*self.dim_z)
            )
        elif(self.dim_h1 <= 0 and self.dim_h2 > 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h2, 2*self.dim_z)
            )
        elif(self.dim_h1 <= 0 and self.dim_h2 <= 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, 2*self.dim_z)
            ) 
            
    def forward(self,x):
        
        # input shape: batch x boxes x ch x width x height  
        assert len(x.shape) == 5
        batch,nboxes = x.shape[:2]
        
        # actual encoding
        z = self.compute_z(x.view(batch,nboxes,-1)) 
        z_mu  = z[...,:self.dim_z]
        z_std = F.softplus(z[...,self.dim_z:])
        return collections.namedtuple(self.name, "z_mu z_std")._make([z_mu,z_std])
    