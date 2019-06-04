import torch
import collections
import torch.nn.functional as F


class Encoder_CONV(torch.nn.Module):
    """ Encode cropped stuff into z_mu, z_std 
        INPUT  SHAPE: batch x n_boxes x ch x width x height 
        OUTPUT SHAPE: batch x n_boxes x latent_dim
        
        Architecture is inspired by:
        https://github.com/abelusha/MNIST-Fashion-CNN/blob/master/Fashon_MNIST_CNN_using_Keras_10_Runs.ipynb
    """
    
    def __init__(self,params,is_zwhat):
        
        super().__init__()            
        self.nb_filters     = 32
        self.pool_size      = 2
        self.kernel_size    = 3
        self.ch_raw_image = len(params["IMG.ch_in_description"])
        self.width  = params["SD.width"]
        assert(self.width == 28)
        self.dim_in = 800
        self.dim_h1 = params["SD.dim_h1"] 
        self.dim_h2 = params["SD.dim_h2"]
        if(is_zwhat):
            self.dim_out = params['ZWHAT.dim']
            self.name    = "z_what"
        else:
            self.dim_out = params['ZMASK.dim']
            self.name    = "z_mask"
            
            
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = self.ch_raw_image, out_channels=self.nb_filters, kernel_size=self.kernel_size),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size),
                torch.nn.Conv2d(in_channels = self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
            )
        
        
        if(self.dim_h1>0 and self.dim_h2>0):
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h1, self.dim_h2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h2, 2*self.dim_out)
            )
        elif(self.dim_h1 > 0 and self.dim_h2 <= 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h1, 2*self.dim_out)
            )
        elif(self.dim_h1 <= 0 and self.dim_h2 > 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h2, 2*self.dim_out)
            )
        elif(self.dim_h1 <= 0 and self.dim_h2 <= 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, 2*self.dim_out)
            ) 
            
            
    def forward(self,x):
        
        # input reshape from: batch x boxes x ch x width x height ->  batch x boxes x -1
        assert len(x.shape) == 5
        batch,nboxes,ch,width,height = x.shape
        x = x.view(batch*nboxes,ch,width,height)
        
        # actual encoding
        x1 = self.conv(x).view(batch,nboxes,-1)
        #print("x1.shape in encoders",x1.shape)
        
        z = self.compute_z(x1)
        z_mu  = z[...,:self.dim_out]
        z_std = F.softplus(z[...,self.dim_out:])
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
            self.dim_out = params['ZWHAT.dim']
            self.name    = "z_what"
        else:
            self.dim_out = params['ZMASK.dim']
            self.name    = "z_mask"

        if(self.dim_h1>0 and self.dim_h2>0):
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h1, self.dim_h2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h2, 2*self.dim_out)
            )
        elif(self.dim_h1 > 0 and self.dim_h2 <= 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h1, 2*self.dim_out)
            )
        elif(self.dim_h1 <= 0 and self.dim_h2 > 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, self.dim_h2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_h2, 2*self.dim_out)
            )
        elif(self.dim_h1 <= 0 and self.dim_h2 <= 0):    
            self.compute_z = torch.nn.Sequential(
                torch.nn.Linear(self.dim_in, 2*self.dim_out)
            ) 
            
    def forward(self,x):
        
        # input shape: batch x boxes x ch x width x height  
        assert len(x.shape) == 5
        batch,nboxes = x.shape[:2]
        
        # actual encoding
        z = self.compute_z(x.view(batch,nboxes,-1)) 
        z_mu  = z[...,:self.dim_out]
        z_std = F.softplus(z[...,self.dim_out:])
        return collections.namedtuple(self.name, "z_mu z_std")._make([z_mu,z_std])
    
    
