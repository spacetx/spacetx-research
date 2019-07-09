import numpy as np
import torch
import pyro
import pyro.distributions as dist
from PIL import Image, ImageDraw
from LOW_LEVEL_UTILITIES.distributions import UniformWithTails, Indicator, NullScoreDistribution_Bernoulli, UnitCauchy
from LOW_LEVEL_UTILITIES.utilities import reset_parameters,save_obj,load_obj
from VAE.vae_parts import *
import matplotlib.pyplot as plt
import collections
from torch.distributions import constraints
import pyro.poutine as poutine



class Compositional_VAE(torch.nn.Module):
    
    # a class method to create a Compositional_VAE from file 
    @classmethod
    def load(cls,params,root_dir,name):
        vae = cls(params)
        vae.load_everything(root_dir,name)
        return vae 
    
    def reset(self):
        reset_parameters(self.inference)
        reset_parameters(self.generator_imgs)
        reset_parameters(self.generator_masks)
        
    def save_everything(self,root_dir,name):
    
        # pyro params
        pyro.get_param_store().save(root_dir + name + "_params_store.pt")
        
        # dictionaty with member variables
        dictionary = self.__dict__
        member_var = {}
        for k,v in dictionary.items():
            if((isinstance(v,bool) or isinstance(v,int) or isinstance(v,float) or isinstance(v,list) or isinstance(v,str))
               and k != 'training' and k != 'key'):
                member_var[k]=v  
        #print("member_var -->",member_var)
        save_obj(member_var,root_dir,name+"_member_var")            
                
        # dictionary with modules (including batch norm)
        save_obj(self.state_dict(),root_dir,name+"_modules")

    def load_everything(self,root_dir,name):
    
        # load pyro params (but do not update the modules)
        pyro.get_param_store().load(root_dir + name + "_params_store.pt")
        
        # load the member variables
        member_var = load_obj(root_dir,name+"_member_var")   
        for key, value in member_var.items():
            #print("OVERLOADING ->",key)
            setattr(self,key,value)
            
        # load the modules
        current_state  = self.state_dict()
        previous_state = load_obj(root_dir,name+"_modules")
        # overwrite the modules
        for k, v in previous_state.items():
            if(k in current_state.keys()):
                #print("OVERLOADING ->",k)
                current_state[k]=v
        # reload the updated modules
        self.load_state_dict(current_state)
       
    
    
    def __init__(self, params):
        super().__init__()
        
        # Instantiate the encoder and decoder
        self.inference = Inference(params)
        self.generator_imgs  = Imgs_Generator(params)
        self.generator_masks = Masks_Generator(params)
 
        #-------------------------#
        #--- Global Parameters ---#
        #-------------------------#
        self.Zwhat_dim         = params['ZWHAT.dim']
        self.Zmask_dim         = params['ZMASK.dim']
        self.size_raw_image    = params['IMG.size_raw_image'] 
        self.ch_in_description = params['IMG.ch_in_description'] 
        self.ch_raw_image      = len(self.ch_in_description) 
        
        #--------------------------------#
        #--- Pramatres Regolurization ---#
        #--------------------------------#
        self.volume_mask_min             = params['REGULARIZATION.volume_mask_min']
        self.volume_mask_expected        = params['REGULARIZATION.volume_mask_expected']
        self.volume_mask_max             = params['REGULARIZATION.volume_mask_max']

        self.p_corr_factor               = params['REGULARIZATION.p_corr_factor']
        self.randomize_nms_factor        = params['REGULARIZATION.randomize_nms_factor']
        self.lambda_small_box_size       = params['REGULARIZATION.lambda_small_box_size']
        self.lambda_mask_volume_fraction = params['REGULARIZATION.lambda_mask_volume_fraction']
        self.lambda_mask_volume_absolute = params['REGULARIZATION.lambda_mask_volume_absolute']
        self.lambda_tot_var_mask         = params['REGULARIZATION.lambda_tot_var_mask']
        self.lambda_overlap              = params['REGULARIZATION.lambda_overlap']  
        self.LOSS_ZMASK                  = params['REGULARIZATION.LOSS_ZMASK']
        self.LOSS_ZWHAT                  = params['REGULARIZATION.LOSS_ZWHAT']
        
        #------------------------------------#
        #----------- PRIORS -----------------#
        #------------------------------------#
        self.width_zmask        = params['PRIOR.width_zmask']
        self.width_zwhat        = params['PRIOR.width_zwhat']
        self.n_objects_max      = params['PRIOR.n_objects_max'] 
        self.n_objects_expected = params['PRIOR.n_objects_expected'] 

        self.size_min      = params['PRIOR.size_object_min'] 
        self.size_max      = params['PRIOR.size_object_max'] 
        self.size_expected = params['PRIOR.size_object_expected'] 
        
        # Put everything on the cuda if necessary
        self.use_cuda = params["use_cuda"]
        self.device = 'cuda' if self.use_cuda else 'cpu'
        if self.use_cuda:
            self.cuda()
            
    def mask_argmin_argmax(self,w,label=None):   
        """ Return the mask with one 1 along the direction dim=-4 where the value of w is either max or min
            This is usefull to find the object a pixel belongs to 
        """
        with torch.no_grad():
            assert len(w.shape) == 5 #batch_size,n_box,1,width,height = w.shape
            k = w.shape[-4]
            
            if(label == "argmax"):
                real_indeces = torch.argmax(w,dim=-4,keepdim=True) #max over boxes dimension
            elif(label == "argmin"):
                real_indeces = torch.argmin(w,dim=-4,keepdim=True) #min over boxes dimension
            else:
                raise Exception('invalid label is neither argmin nor argmax. Label = {}'.format(label))
                
            fake_indeces = torch.arange(start=0, end=k, step=1, dtype=real_indeces.dtype, device=real_indeces.device).view(1,-1,1,1,1)
            mask = (real_indeces == fake_indeces).float()
            return mask
        
        
    def score_observations(self,box_dimfull,sigma_imgs,putative_imgs,putative_masks,
                           mask_pixel_assignment,definitely_bg_mask,imgs):   
        
        # get the parameters 
        #normal_sigma = pyro.param("normal_sigma")
        bg_mu = pyro.param("bg_mu")
        fg_mu = pyro.param("fg_mu")
        bg_sigma = pyro.param("bg_sigma")
        fg_sigma = pyro.param("fg_sigma")
        
        # The foreground/background should be drawn from a Cauchy distribtion with scalar parameters 
        # TO IMPROVE: parameter should be different for each channel
        obs_imgs = imgs.unsqueeze(-4) # also add a singleton for the n_object dimension
        log_p_definitely_bg_cauchy = UnitCauchy(bg_mu,bg_sigma).expand(obs_imgs.shape).mask(definitely_bg_mask).log_prob(obs_imgs)
        obs_imgs = obs_imgs.expand(-1,self.n_objects_max,-1,-1,-1) # expand for dimension over n_boxes
        log_p_given_bg_cauchy = UnitCauchy(bg_mu,bg_sigma).expand(obs_imgs.shape).mask(mask_pixel_assignment).log_prob(obs_imgs)
        log_p_given_fg_cauchy = UnitCauchy(fg_mu,fg_sigma).expand(obs_imgs.shape).mask(mask_pixel_assignment).log_prob(obs_imgs)
        log_p_given_fg_normal = dist.Normal(putative_imgs,sigma_imgs).mask(mask_pixel_assignment).log_prob(obs_imgs)   
                    
        # technically incorrect but it speeds up training and lead to binarized masks
        # The probability of each pixel value is:
        # P(x) = w * P(x|FG) + (1-w) * P(x|BG)
        #log_w   = torch.log(putative_masks)
        #log_1mw = get_log_prob_compl(log_w)
        #log_partial_pixel_cauchy = Log_Add_Exp(log_p_given_fg_cauchy,log_p_given_bg_cauchy,log_w,log_1mw)
        #log_partial_pixel_normal = Log_Add_Exp(log_p_given_fg_normal,log_p_given_bg_cauchy,log_w,log_1mw)
        log_partial_pixel_cauchy = putative_masks*log_p_given_fg_cauchy+(1.0-putative_masks)*log_p_given_bg_cauchy 
        log_partial_pixel_normal = putative_masks*log_p_given_fg_normal+(1.0-putative_masks)*log_p_given_bg_cauchy 
                        
        # compute logp
        logp_definitely_bg = torch.sum(log_p_definitely_bg_cauchy,dim=(-1,-2,-3)) 
        logp_box_off       = torch.sum(log_p_given_bg_cauchy,dim=(-1,-2,-3))
        logp_box_on_cauchy = torch.sum(log_partial_pixel_cauchy,dim=(-1,-2,-3))
        logp_box_on_normal = torch.sum(log_partial_pixel_normal,dim=(-1,-2,-3))  
        
        # package the logp
        common_logp    = logp_definitely_bg/self.n_objects_max
        log_probs = collections.namedtuple('logp', 'logp_off, logp_on_cauchy, logp_on_normal')._make(
                [common_logp+logp_box_off, common_logp + logp_box_on_cauchy, common_logp + logp_box_on_normal])           

        
        # compute the regularizations
        volume_box  = (box_dimfull.bw_dimfull*box_dimfull.bh_dimfull).squeeze(-1)
        volume_mask = torch.sum(mask_pixel_assignment*putative_masks,dim=(-1,-2,-3))
        
        
        #- reg1: bounding box should be as small as possible --#
        #- Note that volume_mask is detached from computation graph, 
        #- therefore this regolarization can only make box_volume smaller 
        #- not make the mask_volume larger.
        with torch.no_grad():
            volume_box_min = torch.tensor(self.size_min*self.size_min, device=volume_mask.device, dtype=volume_mask.dtype)
            volume_min     = torch.max(volume_mask,volume_box_min)
        reg_small_box_size = (volume_box/volume_min - 1.0)**2
        
        #- reg 2: mask should occupy at least 10% of the box -#
        #- Note that the box volume is detached from computation graph, 
        #- therefore this regolarization can only make mask_volume larger not the 
        #- box_volume smaller.
        of     = volume_mask/volume_box.detach() # occupaid fraction
        tmp_of = torch.clamp(50*(0.1-of),min=0)
        reg_mask_volume_fraction = torch.expm1(tmp_of)
        
        #- reg 3: mask volume should be between min and max 
        tmp_volume_absolute = torch.clamp((self.volume_mask_min-volume_mask)/self.volume_mask_expected,min=0) + \
                              torch.clamp((volume_mask-self.volume_mask_max)/self.volume_mask_expected,min=0)
        reg_mask_volume_absolute = (50*tmp_volume_absolute).pow(2)
         
       
        #- reg 4: mask should have small total variations -#
        #- TotVar = integral of the absolute gradient -----#
        #- This is L1 b/c we want discountinuity ----------#
        pixel_weights = putative_masks*mask_pixel_assignment
        grad_x = torch.sum(torch.abs(pixel_weights[:,:,:,:,:-1] - pixel_weights[:,:,:,:,1:]),dim=(-1,-2,-3))
        grad_y = torch.sum(torch.abs(pixel_weights[:,:,:,:-1,:] - pixel_weights[:,:,:,1:,:]),dim=(-1,-2,-3))
        reg_tot_var_mask = (grad_x+grad_y)
  
            
        #- reg 5: mask should have small or no overlap ---------------#
        #- Question: Assign the cost to the second most likely mask? -#
        values, indeces = torch.topk(putative_masks, k=2, dim=1, largest=True) # shape: batch x 2 x 1 x width x height
        prod = torch.prod(values,dim=1,keepdim=True) # shape batch x 1 x 1 x width x height
        with torch.no_grad():
            fake_indeces = torch.arange(start=0,end=self.n_objects_max,step=1,
                                        dtype=indeces.dtype,device=indeces.device).view(1,-1,1,1,1)
            assignment_mask = (indeces[:,-1:,:,:,:] == fake_indeces).float()
        reg_overlap_mask = torch.sum(prod*assignment_mask,dim=(-1,-2,-3))**2
            
        regularizations = collections.namedtuple('reg', "small_box_size mask_volume_fraction mask_volume_absolute tot_var_mask overlap_mask")._make(
            [reg_small_box_size, reg_mask_volume_fraction, reg_mask_volume_absolute, reg_tot_var_mask, reg_overlap_mask])

        return log_probs,regularizations
    
    
    def guide(self, imgs=None, epoch=None):
        
        """ The GUIDE takes a mini-batch of images and: 
            1. run the inference to get: zwhere,zwhat
            2. sample:
                - z ~ N(z_mu, z_std) where each component of z is drawn independently
                - c ~ Bernulli(p)
                - cxcy ~ N(cxcy_mu,0.1) 
                - dxdy ~ gamma(dxdy_mu,0.1) 
        """
        
        #-----------------------#
        #--------  Trick -------#
        #-----------------------#
        if(imgs is None):
            imgs = torch.zeros(8,self.ch_raw_image,self.size_raw_image,self.size_raw_image)
            if(self.use_cuda):
                imgs=imgs.cuda()
        assert(len(imgs.shape)==4)
        batch_size,ch,width,height = imgs.shape    
        assert(width == height) 
        one  = torch.ones(1,dtype=imgs.dtype,device=imgs.device)
        zero = torch.zeros(1,dtype=imgs.dtype,device=imgs.device)
        #-----------------------#
        #----- Enf of Trick ----#
        #-----------------------#
        
        # register the modules
        pyro.module("inference",self.inference)
        
        with pyro.plate("batch", batch_size, dim =-2 ):
            
            #--------------------------#
            #-- 1. run the inference --#
            #--------------------------#          
            z_nms = self.inference.forward(imgs,
                                           p_corr_factor=self.p_corr_factor,
                                           randomize_nms_factor = self.randomize_nms_factor)

            with pyro.plate("n_objects", self.n_objects_max, dim =-1 ):
                     
                #---------------#    
                #-- 2. Sample --#
                #---------------#

                # bounding box
                pyro.sample("bx_dimfull",dist.Delta(z_nms.z_where.bx_dimfull).to_event(1))
                pyro.sample("by_dimfull",dist.Delta(z_nms.z_where.by_dimfull).to_event(1))
                pyro.sample("bw_dimfull",dist.Delta(z_nms.z_where.bw_dimfull).to_event(1)) 
                pyro.sample("bh_dimfull",dist.Delta(z_nms.z_where.bh_dimfull).to_event(1))
                
                # Probability of a box being active
                p = z_nms.z_where.prob.squeeze(-1)
                c = pyro.sample("prob_object",dist.Bernoulli(probs = p))     
                c_mask = c[...,None] # add a singleton for the event dimension
                
                # GATING of Z_WHAT and Z_MASK
                z_what_mu  = (1-c_mask)*zero             + c_mask*z_nms.z_what.z_mu
                z_what_std = (1-c_mask)*self.width_zwhat + c_mask*z_nms.z_what.z_std
                z_mask_mu  = (1-c_mask)*zero             + c_mask*z_nms.z_mask.z_mu
                z_mask_std = (1-c_mask)*self.width_zmask + c_mask*z_nms.z_mask.z_std
                
                #print("p.shape",p.shape)
                #print("c.shape",c.shape)
                #print("z_nms.z_what.z_mu.shape",z_nms.z_what.z_mu.shape)
                #print("z_mask_mu.shape",z_mask_mu.shape)
                
                pyro.sample("z_what",dist.Normal(z_what_mu, z_what_std).to_event(1))
                pyro.sample("z_mask",dist.Normal(z_mask_mu, z_mask_std).to_event(1))
                

    def model(self, imgs=None, epoch=None):
        """ The MODEL takes a mini-batch of images and:
            1.  sample the latent from the prior:
                - z ~ N(0,1)
                - presence of a cell ~ Bernulli(p)
                - cxcy ~ uniform (-1,+1)
                - dxdy ~ gamma(alpha,beta) 
            2.  runs the generative model
            3A. If imgs = None then
                score the generative model against actual data 
                (it requires a noise model and fix the reconstruction loss)
            3B. Else
                return the generated image
                
            This is a trick so that by passing imgs=None I can:
            1. effectively test the priors
            2. debug shapes by using poutine.trace
        """
        #-----------------------#
        #--------  Trick -------#
        #-----------------------#
        if(imgs is None):
            observed = False
            imgs = torch.zeros(8,self.ch_raw_image,self.size_raw_image,self.size_raw_image)
            if(self.use_cuda):
                imgs=imgs.cuda()
        else:
            observed = True
        assert(len(imgs.shape)==4)
        batch_size,ch,width,height = imgs.shape    
        assert(width == height) 
        one  = torch.ones(1,dtype=imgs.dtype,device=imgs.device)
        zero = torch.zeros(1,dtype=imgs.dtype,device=imgs.device)
        #-----------------------#
        #----- Enf of Trick ----#
        #-----------------------#
        
        # register the modules 
        pyro.module("generator_imgs",  self.generator_imgs)
        pyro.module("generator_masks", self.generator_masks)

        # register the parameters of the distribution used to score the results
        bg_mu        = pyro.param("bg_mu", 0.1*one, constraint=constraints.unit_interval)
        fg_mu        = pyro.param("fg_mu", 0.9*one, constraint=constraints.unit_interval)
        bg_sigma     = pyro.param("bg_sigma", 0.2*one, constraint=constraints.interval(0.01,0.25))
        fg_sigma     = pyro.param("fg_sigma", 0.2*one, constraint=constraints.interval(0.01,0.25))
        #normal_sigma = pyro.param("normal_sigma", 0.2*one, constraint=constraints.interval(0.01,0.25))

        with pyro.plate("batch", batch_size, dim=-2):
            
            with pyro.plate("n_objects", self.n_objects_max, dim =-1):
            
                #------------------------#
                # 1. Sample from priors -#
                #------------------------#
                
                #- Z_WHERE 
                c          = pyro.sample("prob_object",dist.Bernoulli(probs=0.5*one))
                bx_dimfull = pyro.sample("bx_dimfull",dist.Uniform(zero,width).expand([1]).to_event(1))
                by_dimfull = pyro.sample("by_dimfull",dist.Uniform(zero,width).expand([1]).to_event(1))
                bw_dimfull = pyro.sample("bw_dimfull",dist.Uniform(self.size_min*one,self.size_max).expand([1]).to_event(1))
                bh_dimfull = pyro.sample("bh_dimfull",dist.Uniform(self.size_min*one,self.size_max).expand([1]).to_event(1))
                
                #- Z_WHAT, Z_WHERE 
                z_what = pyro.sample("z_what",dist.Normal(zero,self.width_zwhat).expand([self.Zwhat_dim]).to_event(1)) 
                z_mask = pyro.sample("z_mask",dist.Normal(zero,self.width_zmask).expand([self.Zmask_dim]).to_event(1)) 
                       
                    
                #------------------------------#
                # 2. Run the generative model -#
                #------------------------------#
                box_dimfull = collections.namedtuple('box_dimfull', 'bx_dimfull by_dimfull bw_dimfull bh_dimfull')._make(
                                                    [bx_dimfull,by_dimfull,bw_dimfull,bh_dimfull])

                if(len(z_mask.shape) == 4 and z_mask.shape[0] == 2):
                    # This means that I have an extra enumerate dimension in front to account for box being ON/OFF.
                    # I need to reconstruct the image only if the box is ON therefore I pass z_mask[1], z_what[1]
                    putative_masks           = self.generator_masks.forward(box_dimfull,z_mask[1],width,height)
                    putative_imgs,sigma_imgs = self.generator_imgs.forward(box_dimfull,z_what[1],width,height) 
                elif(len(z_mask.shape) == 3):
                    putative_masks           = self.generator_masks.forward(box_dimfull,z_mask,width,height)
                    putative_imgs,sigma_imgs = self.generator_imgs.forward(box_dimfull,z_what,width,height) 
                
                assert putative_masks.shape == (batch_size,self.n_objects_max,1,width,height)
                assert putative_imgs.shape  == (batch_size,self.n_objects_max,ch,width,height)
                assert sigma_imgs.shape     == (batch_size,self.n_objects_max,1,1,1)
                
                
                # Resolve the conflict. Each pixel belongs to only one FG object
                # If a pixel does not belong to any object it belongs to the background
                mask_pixel_assignment = self.mask_argmin_argmax(putative_masks,"argmax") 
                #mask_pixel_assignment = self.mask_argmin_argmax(putative_masks*p_inferred[...,None,None,None],"argmax") 
                definitely_bg_mask = (torch.sum(mask_pixel_assignment,dim=-4,keepdim=True) == 0.0) 
                
                # sample the background 
                background_sample = torch.clamp(dist.Cauchy(bg_mu,bg_sigma).expand(imgs.shape).sample(),min=0.0,max=1.0)

                    
                if(observed):
                    logp,reg = self.score_observations(box_dimfull,sigma_imgs,putative_imgs,putative_masks,
                                                       mask_pixel_assignment,definitely_bg_mask,imgs)
                    
                    total_reg = self.lambda_small_box_size*reg.small_box_size + \
                                self.lambda_mask_volume_fraction*reg.mask_volume_fraction + \
                                self.lambda_mask_volume_absolute*reg.mask_volume_absolute + \
                                self.lambda_tot_var_mask*reg.tot_var_mask + \
                                self.lambda_overlap*reg.overlap_mask      
                                                                         
                    log_prob_ZMASK = self.LOSS_ZMASK*torch.stack((logp.logp_off,logp.logp_on_cauchy-total_reg),dim=-1) 
                    log_prob_ZWHAT = self.LOSS_ZWHAT*torch.stack((logp.logp_off,logp.logp_on_normal-total_reg),dim=-1) 

                    pyro.sample("LOSS", Indicator(log_probs=log_prob_ZMASK+log_prob_ZWHAT), obs = c)
                    
                return putative_imgs,putative_masks,background_sample,c
    

    def reconstruct_img(self,original_image,bounding_box=False):
        if(self.use_cuda):
            original_image=original_image.cuda()
        
        batch_size,ch,width,height = original_image.shape
        assert(width==height)
        self.eval() # set the model into evaluation mode
        self.randomize_score_nms = False
        with torch.no_grad(): # do not keep track of the gradients
            
            #--------------------------#
            #-- 1. run the inference --#
            #--------------------------#        
            z_nms = self.inference.forward(original_image,
                                           p_corr_factor=self.p_corr_factor,
                                           randomize_nms_factor = self.randomize_nms_factor)
                
            p     = z_nms.z_where.prob 
            assert p.shape == (batch_size,self.n_objects_max,1)

            #--------------------------------#
            #--- 2. Run the model forward ---#
            #--------------------------------#
            putative_masks = self.generator_masks.forward(z_nms.z_where,z_nms.z_mask.z_mu,width,height)                
            putative_imgs,sigma_imgs  = self.generator_imgs.forward(z_nms.z_where,z_nms.z_what.z_mu,width,height) 
            mask_pixel_assignment = self.mask_argmin_argmax(putative_masks,"argmax")   
            definitely_bg_mask = (torch.sum(mask_pixel_assignment,dim=-4,keepdim=True) == 0.0) 
            
            #---------------------------------#
            #--- 3. Score the model ----------#
            #---------------------------------#
            logp,reg = self.score_observations(z_nms.z_where,sigma_imgs,putative_imgs,putative_masks,
                                               mask_pixel_assignment,definitely_bg_mask,original_image)
           
            #---------------------------------#
            #----- 4. Reconstruct images -----#
            #---------------------------------#
            box_is_active = (p>0.5).float()[...,None,None]
            fg_mask = (mask_pixel_assignment*putative_masks > 0.0).float()
            reconstructed_image = torch.sum(box_is_active*fg_mask*putative_imgs,dim=-4,keepdim=False)
            
            # 3. If bounding_box == True compute the bounding box
            if(bounding_box == False):
                return reconstructed_image,z_nms.z_where,putative_imgs,putative_masks,logp,reg
            elif(bounding_box == True):
                bounding_boxes = self.draw_batch_of_images_with_bb_only(z_nms.z_where,width,height)
                reconstructed_image_with_bb = bounding_boxes + reconstructed_image
                return reconstructed_image_with_bb,z_nms.z_where,putative_imgs,putative_masks,logp,reg
    
    def draw_batch_of_images_with_bb_only(self,z_where,width,height):
       
        # Extract the probabilities for each box
        batch_size,n_boxes = z_where.prob.shape[:2]
        p = z_where.prob.view(batch_size,n_boxes,-1)
        
        # prepare the storage
        batch_bb_np    = np.zeros((batch_size,width,height,3)) # numpy storage for bounding box images
        
        # compute the coordinates of the bounding boxes and the probability of each box
        x1 = (z_where.bx_dimfull-0.5*z_where.bw_dimfull).view(batch_size,n_boxes,-1)
        x3 = (z_where.bx_dimfull+0.5*z_where.bw_dimfull).view(batch_size,n_boxes,-1)
        y1 = (z_where.by_dimfull-0.5*z_where.bh_dimfull).view(batch_size,n_boxes,-1)
        y3 = (z_where.by_dimfull+0.5*z_where.bh_dimfull).view(batch_size,n_boxes,-1)

        assert( len(x1.shape) == 3) 
        x1y1x3y3 = torch.cat((x1,y1,x3,y3),dim=2)
                
        # draw the bounding boxes
        for b in range(batch_size):
        
            # Draw on PIL
            img = Image.new('RGB', (width,height), color=0)
            draw = ImageDraw.Draw(img)
            for i in range(n_boxes):
                #if(p[b,i,0]>0.0):
                draw.rectangle(x1y1x3y3[b,i,:].cpu().numpy(), outline='red', fill=None)
            batch_bb_np[b,...] = np.array(img.getdata(),np.uint8).reshape(width,height,3)

        # Transform np to torch, rescale from [0,255] to (0,1) 
        batch_bb_torch = torch.from_numpy(batch_bb_np).permute(0,3,2,1).float()/255 # permute(0,3,2,1) is CORRECT
        return batch_bb_torch.to(p.device)   
           
    def generate_synthetic_data(self, N=100):
        
        # prepare the storage 
        putative_imgs,putative_masks,background,c = self.model()
        assert putative_masks.shape[-2:] == background.shape[-2:] == putative_imgs.shape[-2:] # width, height are the same
        batch_size, n_boxes, ch, width, height = putative_imgs.shape
        synthetic_data = torch.zeros((N, ch, width, height), 
                                     dtype=torch.float32, 
                                     device='cpu', 
                                     requires_grad=False) 
            
        # loop to generate the images
        l = 0 
        while (l<N):         
            
            # generate the images and mask
            putative_imgs,putative_masks,background,c = self.model()
            box_is_active = (c == 1).float()[...,None,None,None] # add singleton for ch,width,height 
            imgs_prior = torch.sum((box_is_active*putative_imgs),dim=-4) + background
            
            # Compute left and right indeces
            r = min(l+batch_size,N)
            d = r-l           
            synthetic_data[l:r,:,:,:]=imgs_prior[0:d,:,:,:].detach().cpu()
            l = r
            
        return torch.clamp(synthetic_data,min=0.0, max=1.0)
