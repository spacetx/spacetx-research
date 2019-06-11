import torch

class SimulationDictionary(dict):
    """ Specify the value to use in the simulation. Things will break badly unless dim_STN=5 (for p,x1,x2,y1,y2) 
        and the latent space is organized so that p,x1,x2,y1,y2 are at the beginning followed by latent_z
        I also assume that the WIDTH and HEIGHT are the same for raw_image and at any step of the net
    """
    def __init__(self):
        super().__init__()
        
        # Stuff for single decoder (MLP)
        self['SD.width'] = 28  
        self['SD.dim_h1'] = -1 # int(self['SD.dim_output']/4)
        self['SD.dim_h2'] = -1 # int(self['SD.dim_output']/2)   
        
        # Parameters for ZWHAT, ZMASK
        self['ZWHAT.dim'] = 50
        self['ZMASK.dim'] = 50 
        
        # Parameters regularizations
        self['REGULARIZATION.p_corr_factor']=0.0
        self['REGULARIZATION.lambda_small_box_size']=0.0
        self['REGULARIZATION.lambda_big_mask_volume']=1.0
        self['REGULARIZATION.lambda_tot_var_mask']=0.0
        self['REGULARIZATION.lambda_overlap']=0.0
        self['REGULARIZATION.LOSS_ZMASK']=1.0
        self['REGULARIZATION.LOSS_ZWHAT']=10.0
        
        # Parameters for the PRIOR in the VAE model
        self['PRIOR.n_max_objects'] = 6
        self['PRIOR.min_object_size'] = 15 #in pixels
        self['PRIOR.max_object_size'] = 35 #in pixels
        self['PRIOR.expected_object_size'] = 20 #in pixels
        
        # Parameters for the description of the raw image
        self['IMG.size_raw_image'] = 80 # for now it only take square images as input
        self['IMG.ch_in_description'] = ["DAPI"]
        
        # Stuff for UNET
        self['UNET.N_max_pool'] = 4
        self['UNET.N_up_conv'] = 2
        self['UNET.CH_after_first_two_conv'] = 32
        self['UNET.N_prediction_maps'] = 1 # if 1 only the rightmost layer export the prediction
        
        # Stuff for YOLO FILTER (a.k.a. non max suppression)
        self['NMS.p_threshold']=0.0 # it might be that there are no boxes. Then only the background is present
        self['NMS.overlap_threshold']=0.2

        # Enviromental variable
        self["use_cuda"] = torch.cuda.is_available() 
    
    def check_consistency(self):
        
        assert( isinstance(self['ZWHAT.dim'],int) )
        assert( isinstance(self['ZMASK.dim'],int) )
        assert( self['ZWHAT.dim'] > 0)
        assert( self['ZMASK.dim'] > 0)
        
        assert(self['PRIOR.max_object_size'] >  self['PRIOR.expected_object_size'] > self['PRIOR.min_object_size'] >  0.0)

        assert( isinstance(self['UNET.N_max_pool'],int) )
        assert( isinstance(self['UNET.N_up_conv'],int) )
        assert( isinstance(self['UNET.N_prediction_maps'],int) )
        assert( self['UNET.N_max_pool'] >= self['UNET.N_up_conv'])
        assert( 1<= self['UNET.N_prediction_maps'] <= (self['UNET.N_up_conv']+1))
        
        assert( 0<=self['NMS.p_threshold']<=1.0 )
        assert( 0<=self['NMS.overlap_threshold']<=1.0 )