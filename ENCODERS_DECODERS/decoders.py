import torch
import collections
import torch.nn.functional as F

class Multi_Channel_Img_Decoder(torch.nn.Module):
    """ This is a wrapper class which uses a different decoder type 
        to predict each channel in the input image"""
    def __init__(self, params: dict):
        super().__init__()
        self.dim_zwhat         = params['ZWHAT.dim']
        self.ch_in_description = params['IMG.ch_in_description']
        self.use_cuda  = params['use_cuda']
        self.multi_channels_dec = torch.nn.ModuleList()
        for i in self.ch_in_description:
            if(i == 'DAPI'):
                #self.multi_channels_dec.append(Decoder_MLP(params,is_zwhat=True))
                self.multi_channels_dec.append(Decoder_CONV(params,is_zwhat=True))                 
            elif(i == 'DISK'):
                self.multi_channels_dec.append(Decoder_Disk(params))
            elif(i == 'SQUARE'):
                self.multi_channels_dec.append(Decoder_Square(params))
            elif( i == 'FISH'):
                raise NotImplementedError
            else:
                raise Exception 

    def forward(self,z_what):
        output = list()
        for i, decoder in enumerate(self.multi_channels_dec):
            output.append(decoder(z_what))
        if(len(self.multi_channels_dec)>1):
            return torch.cat(output,dim = -3) #glue along the channel dimension
        elif(len(self.multi_channels_dec)==1):
            return output[0]
        
class Decoder_CONV(torch.nn.Module):
    """ Decode z -> x 
        INPUT:  z of shape: batch x dim_z 
        OUTPUT: image of shape: batch x 1 x width x height (where width = height)
    """
    def __init__(self, params, is_zwhat=True):
        super().__init__()
        self.width    = params['SD.width']
        self.is_zwhat = is_zwhat
        
        assert self.width == 28
        
        if(self.is_zwhat):
            self.dim_z = params['ZWHAT.dim']
            self.ch    = len(params["IMG.ch_in_description"])
            self.compute_sigma = torch.nn.Linear(self.dim_z, 1)
        else:
            self.dim_z = params['ZMASK.dim']
            self.ch    = 1
            
        
        self.upsample = torch.nn.Linear(self.dim_z, 64 * 7 * 7)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  64,  14,  14
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1), # B,  32, 28, 28
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, self.ch, 4, 1, 2)   # B, ch, 28, 28
        )

    def forward(self,z):
        assert len(z.shape) == 2 
        batch_size = z.shape[0]
        x1    = self.upsample(z).view(batch_size,64,7,7)
        x     = torch.sigmoid(self.decoder(x1)) # use sigmoid so pixel intensity is in (0,1). 
        if(self.is_zwhat):
            sigma = F.softplus(self.compute_sigma(z)) 
            return x,sigma
        else:
            return x
        
###class Decoder_MLP(torch.nn.Module):
###    """This is a MLP which deencodes a latent representation into a single general object
###       INPUT: z of shape: batch x dim_z 
###       OUTPUT: an image of shape: batch x 1 x width x height (where width = height)
###    """
###    def __init__(self, params, is_zwhat=True):
###        super().__init__()
###        self.use_cuda  = params['use_cuda']     
###        self.dim_h1    = params['SD.dim_h1']
###        self.dim_h2    = params['SD.dim_h2']
###        self.width     = params['SD.width']
###        self.dim_out   = self.width*self.width
###        
###        if(is_zwhat):
###            self.dim_in = params['ZWHAT.dim']
###        else:
###            self.dim_in = params['ZMASK.dim']
###
###        # Depending on the value of self.dim_he I do 1 or 2 hidden layers
###        if(self.dim_h1>0 and self.dim_h2>0):
###            self.single_decoder = torch.nn.Sequential(
###                torch.nn.Linear(self.dim_in, self.dim_h1),
###                torch.nn.ReLU(),
###                torch.nn.Linear(self.dim_h1, self.dim_h2),
###                torch.nn.ReLU(),
###                torch.nn.Linear(self.dim_h2, self.dim_out)
###            )
###        elif(self.dim_h1 > 0 and self.dim_h2 <= 0):    
###            self.single_decoder = torch.nn.Sequential(
###                torch.nn.Linear(self.dim_in, self.dim_h1),
###                torch.nn.ReLU(),
###                torch.nn.Linear(self.dim_h1, self.dim_out)
###            )
###        elif(self.dim_h1 <= 0 and self.dim_h2 > 0):    
###            self.single_decoder = torch.nn.Sequential(
###                torch.nn.Linear(self.dim_in, self.dim_h2),
###                torch.nn.ReLU(),
###                torch.nn.Linear(self.dim_h2, self.dim_out)
###            )
###        elif(self.dim_h1 <= 0 and self.dim_h2 <= 0):    
###            self.single_decoder = torch.nn.Sequential(
###                torch.nn.Linear(self.dim_in, self.dim_out)
###            )    
###            
###        # Here I am initializing the bias with small values so that initially the decoder produce empty stuff
###        #m = list(self.single_decoder.children())
###        #m[-1].bias.data += -1.0
###        
###        
###    def forward(self, z):
###        assert len(z.shape) == 2
###        assert self.dim_in == z.shape[-1]
###        x = torch.sigmoid(self.single_decoder(z)) # use sigmoid so pixel intensity is in (0,1). 
###        return x.view(-1,1,self.width,self.width)
###    
###
###class Decoder_Disk(torch.nn.Module):
###    """ Decoder which ALWAYS generates a circle of unifrom (but adapting) intensity in a BLACK background """
###    def __init__(self, params: dict):
###        super().__init__()
###        self.width   = params['SD.width']   
###        x_mat = torch.linspace(-1.0, 1.0,steps=self.width).view(-1,1).expand(self.width,self.width).float()
###        y_mat = torch.linspace(-1.0, 1.0,steps=self.width).view(1,-1).expand(self.width,self.width).float()
###        r2   = (x_mat**2 + y_mat**2)
###        disk = (r2<1.0).float()
###        self.unit_disk_mask = disk[None,None,...] #shape (1,1,width,width)
###                       
###    def forward(self, z):
###        assert len(z.shape) == 2
###        batch_size = z.shape[0]
###        return z.new_ones(batch_size,1,self.width,self.width)*self.unit_disk_mask.to(z.device)
###    
###    
###class Decoder_Square(torch.nn.Module):
###    """ Decoder which ALWAYS generates a square of unifrom (but adapting) intensity in a BLACK background """
###    def __init__(self, params: dict):
###        super().__init__()
###        self.width  = params['SD.width']
###                       
###    def forward(self, z):
###        assert len(z.shape) == 2
###        batch_size = z.shape[0]
###        return  z.new_ones((batch_size,1,self.width,self.width))
###    
###    
###
###