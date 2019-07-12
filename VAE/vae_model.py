import numpy as np
import torch
import pyro
import pyro.distributions as dist
from PIL import Image, ImageDraw
from LOW_LEVEL_UTILITIES.distributions import Indicator, NullScoreDistribution_Unit, UnitCauchy
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
        self.lambda_overlap              = params['REGULARIZATION.lambda_overlap']  
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
        
        
    def score_observations(self,z_where,sigma_imgs,putative_imgs,putative_masks,original_imgs): 
               
        # get the parameters 
        fg_mu    = pyro.param("fg_mu")
        bg_mu    = pyro.param("bg_mu")
        fg_sigma = pyro.param("fg_sigma")
        bg_sigma = pyro.param("bg_sigma")      
        
        # Extract the shapes 
        batch_size, n_boxes, ch, width, height = putative_imgs.shape
        
        # Resolve the conflict. Each pixel belongs to only one FG object.
        # Select the FG object with highest product of probability and mask value
        p_inferred = z_where.prob[...,None,None] #add 2 singletons for width,height 
        mask_pixel_assignment = self.mask_argmin_argmax(putative_masks*p_inferred,"argmax") 
        
        # If a pixel does not belong to any object it belongs to the background
        # Use these (probably very few) pixels to learn the background parameters)
        definitely_bg_mask = (torch.sum(mask_pixel_assignment,dim=-4,keepdim=True) == 0.0) 
        obs_imgs = original_imgs.unsqueeze(-4) # add a singleton in the n_boxes dimension             
        logp_definitely_bg = UnitCauchy(bg_mu,bg_sigma).expand(obs_imgs.shape).mask(definitely_bg_mask).log_prob(obs_imgs).sum(dim=(-1,-2,-3))
        
        # Expand the n_boxes dimension so that each box can be scored indepenently       
        obs_imgs = obs_imgs.expand(-1,n_boxes,-1,-1,-1) # expand for dimension over n_boxes
        logp_given_bg = UnitCauchy(bg_mu,bg_sigma).expand(obs_imgs.shape).mask(mask_pixel_assignment).log_prob(obs_imgs)
        logp_given_fg = dist.Normal(putative_imgs,sigma_imgs).mask(mask_pixel_assignment).log_prob(obs_imgs)   
        
        # technically for each pixel:
        # P(x) = w * P(x|FG) + (1-w) * P(x|BG) -> log P = log[ w * P(x|FG) + (1-w) * P(x|BG) ] 
        #
        #log_w   = torch.log(putative_masks)
        #log_1mw = get_log_prob_compl(log_w)
        #logp_box_on  = Log_Add_Exp(log_p_given_fg_cauchy,log_p_given_bg_cauchy,log_w,log_1mw)
        #logp_box_off  = Log_Add_Exp(log_p_given_fg_normal,log_p_given_bg_cauchy,log_w,log_1mw)
        #
        # We do the trick (technically wrong but it speeds up training and lead to binarized masks):
        # log P = w * log P(x|FG) + (1-w) * log P(x|BG)
        logp_box_on  = (putative_masks*logp_given_fg+(1.0-putative_masks)*logp_given_bg).sum(dim=(-1,-2,-3))
        logp_box_off = logp_given_bg.sum(dim=(-1,-2,-3))
        
        # package the logp
        log_probs = collections.namedtuple('logp', 'definitely_bg box_on box_off')._make(
                [logp_definitely_bg, logp_box_on, logp_box_off])           

        
        # compute the regularizations
        volume_box  = (z_where.bw_dimfull*z_where.bh_dimfull).squeeze(-1)
        volume_mask = torch.sum(putative_masks,dim=(-1,-2,-3))
        
        #- reg 1: MAKE SURE THAT THE MASK DOES NOT DISAPPEAR 
        #- Mask should occupy at least 10% of the box 
        #- Note that the box volume is detached from computation graph, 
        #- therefore this regolarization can only make mask_volume larger not the 
        #- box_volume smaller.
        #- This regularization is very strong. 
        #- It basically is a HARD constraint on the generation of masks
        of     = volume_mask/volume_box.detach() # occupied fraction
        tmp_of = torch.clamp(50*(0.1-of),min=0)
        reg_mask_volume_fraction = torch.expm1(tmp_of)
        
        #- reg2: REDUCE BOX SIZE TO FIT THE MASK
        #- Note that volume_mask is detached from computation graph, 
        #- therefore this regolarization can only make box_volume smaller 
        #- not make the mask_volume larger.
        # This regularization need to be gentle since I do not want to split an object in multiple small boxes
        with torch.no_grad():
            volume_box_min = torch.tensor(self.size_min*self.size_min, device=volume_mask.device, dtype=volume_mask.dtype)
            volume_min     = torch.max(volume_mask,volume_box_min)
        reg_small_box_size = (volume_box/volume_min - 1.0)**2
        
        #- reg 3: MASK VOLUME SHOULD BE BETWEEN MIN and MAX
        tmp_volume_absolute = torch.clamp((self.volume_mask_min-volume_mask)/self.volume_mask_expected,min=0) + \
                              torch.clamp((volume_mask-self.volume_mask_max)/self.volume_mask_expected,min=0)
        reg_mask_volume_absolute = (50*tmp_volume_absolute).pow(2)
        
        #- reg 4: MASK SHOULD NOT OVERLAP ---------------#
        #- This is the only INTERACTION term which will be treated in MEAN FIELD
        #- penalty_i = c_i \sum_{j ne i} U(IoMIN_{i,j}) p_j/2
        #- Here U(IoMIN) is the repulsive potential which depends on the amount of overlap 
        m1 = putative_masks.view(batch_size, n_boxes,1,-1)
        m2 = putative_masks.view(batch_size, 1,n_boxes,-1)
        Intersection_vol = torch.sum(m1*m2,dim=-1)
        mask_diagonal = (1.0 - torch.eye(n_boxes, dtype=m1.dtype, device=m1.device, requires_grad=False)).unsqueeze(0)

        v1 = volume_mask.view(batch_size, n_boxes,1)
        v2 = volume_mask.view(batch_size, 1,n_boxes)
        Min_vol = torch.min(v1,v2)
        U = torch.clamp(0.2-(Intersection_vol/Min_vol),min=0)
        
        p_j = z_where.prob.detach().view(batch_size, 1, n_boxes)
                
        reg_overlap_mask = torch.sum(0.5*mask_diagonal*U*p_j,dim=-1)
        
        # package the regularizations
        regularizations = collections.namedtuple('reg', "small_box_size mask_volume_fraction mask_volume_absolute overlap_mask")._make(
            [reg_small_box_size, reg_mask_volume_fraction, reg_mask_volume_absolute, reg_overlap_mask])
        
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
                                           randomize_nms_factor = self.randomize_nms_factor,
                                           n_objects_max = self.n_objects_max)

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
                pyro.sample("prob_object",dist.Delta(p))
                c = pyro.sample("presence_object",dist.Bernoulli(probs = p))   
                                  
                # GATING of Z_WHAT and Z_MASK
                c_mask = c[...,None] # add a singleton for the event dimension
                z_what_mu  = (1-c_mask)*zero             + c_mask*z_nms.z_what.z_mu
                z_what_std = (1-c_mask)*self.width_zwhat + c_mask*z_nms.z_what.z_std
                z_mask_mu  = (1-c_mask)*zero             + c_mask*z_nms.z_mask.z_mu
                z_mask_std = (1-c_mask)*self.width_zmask + c_mask*z_nms.z_mask.z_std
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
        #-----------------------#
        #----- Enf of Trick ----#
        #-----------------------#
        
        # register the modules 
        pyro.module("generator_imgs",  self.generator_imgs)
        pyro.module("generator_masks", self.generator_masks)

        # register the parameters of the distribution used to score the results
        
        # vector of ones used to create a vector of parameters.
        # Each entry corresponds to a different channels
        one_chs  = torch.ones(ch,dtype=imgs.dtype,device=imgs.device) 
        fg_mu    = pyro.param("fg_mu", 0.9*one_chs, constraint=constraints.unit_interval)
        bg_mu    = pyro.param("bg_mu", 0.1*one_chs, constraint=constraints.unit_interval)
        fg_sigma = pyro.param("fg_sigma", 0.2*one_chs, constraint=constraints.interval(0.01,0.25))
        bg_sigma = pyro.param("bg_sigma", 0.2*one_chs, constraint=constraints.interval(0.01,0.25))
                
        with pyro.plate("batch", batch_size, dim=-2):
            
            with pyro.plate("n_objects", self.n_objects_max, dim =-1):
            
                #------------------------#
                # 1. Sample from priors -#
                #------------------------#
                
                #- Trick to get the value of p from the inference 
                one  = torch.ones(1,dtype=imgs.dtype,device=imgs.device)
                prob = pyro.sample("prob_object",NullScoreDistribution_Unit()).to(one.device)

                #- Z_WHERE 
                c          = pyro.sample("presence_object",dist.Bernoulli(probs=0.5*one))                    
                bx_dimfull = pyro.sample("bx_dimfull",dist.Uniform(0,width*one).expand([1]).to_event(1))
                by_dimfull = pyro.sample("by_dimfull",dist.Uniform(0,width*one).expand([1]).to_event(1))
                bw_dimfull = pyro.sample("bw_dimfull",dist.Uniform(self.size_min*one,self.size_max).expand([1]).to_event(1))
                bh_dimfull = pyro.sample("bh_dimfull",dist.Uniform(self.size_min*one,self.size_max).expand([1]).to_event(1))
                z_where = collections.namedtuple('z_where', 'prob bx_dimfull by_dimfull bw_dimfull bh_dimfull')._make(
                                                    [prob.unsqueeze(-1), bx_dimfull,by_dimfull,bw_dimfull,bh_dimfull])
                                                # Note unsqueeze(-1) so that all tensors have shape: batch x nboxes x latent_dim 
                
                #- Z_WHAT, Z_WHERE 
                z_what = pyro.sample("z_what",dist.Normal(0,self.width_zwhat*one).expand([self.Zwhat_dim]).to_event(1)) 
                z_mask = pyro.sample("z_mask",dist.Normal(0,self.width_zmask*one).expand([self.Zmask_dim]).to_event(1)) 
                       
                #------------------------------#
                # 2. Run the generative model -#
                #------------------------------#
                
                if(len(z_mask.shape) == 4 and z_mask.shape[0] == 2):
                    # This means that I have an extra enumerate dimension in front to account for box being ON/OFF.
                    # I need to reconstruct the image only if the box is ON therefore I pass z_mask[1], z_what[1]
                    putative_masks           = self.generator_masks.forward(z_where,z_mask[1],width,height)
                    putative_imgs,sigma_imgs = self.generator_imgs.forward(z_where,z_what[1],width,height) 
                elif(len(z_mask.shape) == 3):
                    putative_masks           = self.generator_masks.forward(z_where,z_mask,width,height)
                    putative_imgs,sigma_imgs = self.generator_imgs.forward(z_where,z_what,width,height) 
                
                assert putative_masks.shape == (batch_size,self.n_objects_max,1,width,height)
                assert putative_imgs.shape  == (batch_size,self.n_objects_max,ch,width,height)
                assert sigma_imgs.shape     == (batch_size,self.n_objects_max,1,1,1)
                                
                    
                if(observed):
                    logp,reg = self.score_observations(z_where,sigma_imgs,putative_imgs,putative_masks,imgs)
                    
                    # PUT EVERYTHING TOGETHER TO COMPUTE THE LOSS:
                    # Note that putting the regularization into the common logp term 
                    # means that the resularizations do not change box on/off
                    common_logp = logp.definitely_bg/self.n_objects_max - \
                                  self.lambda_mask_volume_fraction*reg.mask_volume_fraction - \
                                  self.lambda_small_box_size*reg.small_box_size - \
                                  self.lambda_mask_volume_absolute*reg.mask_volume_absolute - \
                                  self.lambda_overlap*reg.overlap_mask  
                                                
                    log_probs_ZWHAT = self.LOSS_ZWHAT*torch.stack((common_logp+logp.box_off,common_logp+logp.box_on),dim=-1) 
                    pyro.sample("LOSS", Indicator(log_probs=log_probs_ZWHAT), obs = c)
        
                    
                # sample the background 
                background_sample = dist.Cauchy(bg_mu,bg_sigma).expand(imgs.shape).sample()
                #background_sample = dist.Normal(bg_mu_normal,bg_sigma_normal).expand(imgs.shape).sample()
                background_sample_clamped = torch.clamp(background_sample,min=0.0,max=1.0)

                return putative_imgs,putative_masks,background_sample_clamped,c
    

    def reconstruct_img(self,original_image,bounding_box=False):
        if(self.use_cuda):
            original_image=original_image.cuda()
        
        batch_size,ch,width,height = original_image.shape
        assert(width==height)
        with torch.no_grad(): # do not keep track of the gradients
            
            #--------------------------#
            #-- 1. run the inference --#
            #--------------------------#  
            self.eval() # set the model into evaluation mode
            
            z_nms = self.inference.forward(original_image,
                                           p_corr_factor=self.p_corr_factor,
                                           randomize_nms_factor = self.randomize_nms_factor,
                                           n_objects_max = self.n_objects_max)

            #--------------------------------#
            #--- 2. Run the model forward ---#
            #--------------------------------#
            if(len(z_nms.z_mask.z_mu.shape) == 4 and z_nms.z_mask.z_mu.shape[0] == 2):
                # This means that I have an extra enumerate dimension in front to account for box being ON/OFF.
                # I need to reconstruct the image only if the box is ON therefore I pass z_mask[1], z_what[1]
                putative_masks           = self.generator_masks.forward(z_nms.z_where,z_nms.z_mask.z_mu[1],width,height)
                putative_imgs,sigma_imgs = self.generator_imgs.forward(z_nms.z_where,z_nms.z_what.z_mu[1],width,height) 
            elif(len(z_nms.z_mask.z_mu.shape) == 3):
                putative_masks           = self.generator_masks.forward(z_nms.z_where,z_nms.z_mask.z_mu,width,height)
                putative_imgs,sigma_imgs = self.generator_imgs.forward(z_nms.z_where,z_nms.z_what.z_mu,width,height) 
        
            #---------------------------------#
            #--- 3. Score the model ----------#
            #---------------------------------#    
            logp,reg = self.score_observations(z_nms.z_where,sigma_imgs,putative_imgs,putative_masks,original_image)
        
            #---------------------------------#
            #----- 4. Reconstruct images -----#
            #---------------------------------#
            p     = z_nms.z_where.prob[...,None,None] 
            box_is_active = (p>0.5).float()
            mask_pixel_assignment = self.mask_argmin_argmax(putative_masks*p,"argmax")   
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
