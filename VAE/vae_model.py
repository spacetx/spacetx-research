import numpy as np
from PIL import Image, ImageDraw
from LOW_LEVEL_UTILITIES.distributions import CustomLogProbTerm, UnitCauchy
from LOW_LEVEL_UTILITIES.utilities import reset_parameters, save_obj, load_obj
from VAE.vae_parts import *
import collections
from torch.distributions import constraints
import matplotlib.pyplot as plt
import pyro.poutine as poutine
import torch
import pyro
import pyro.distributions as dist


class Compositional_VAE(torch.nn.Module):
    
    # a class method to create a Compositional_VAE from file 
    @classmethod
    def load(cls,params,root_dir,name):
        vae = cls(params)
        vae.load_everything(root_dir,name)
        return vae 
    
    def reset(self):
        reset_parameters(self.inference)
        reset_parameters(self.generator)

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
        self.generator = Generator(params)

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
        self.prob_corr_factor          = params['REGULARIZATION.prob_corr_factor']
        self.lambda_small_box_size  = params['REGULARIZATION.lambda_small_box_size']
        self.lambda_big_mask_volume = params['REGULARIZATION.lambda_big_mask_volume']
        self.lambda_tot_var_mask    = params['REGULARIZATION.lambda_tot_var_mask']
        self.lambda_overlap         = params['REGULARIZATION.lambda_overlap']  
        self.LOSS_ZMASK             = params['REGULARIZATION.LOSS_ZMASK']
        self.LOSS_ZWHAT             = params['REGULARIZATION.LOSS_ZWHAT']
        
        #------------------------------------#
        #----------- PRIORS -----------------#
        #------------------------------------#
        self.n_objects_max = params['PRIOR.n_objects_max']
        self.min_size      = params['PRIOR.min_object_size'] 
        self.max_size      = params['PRIOR.max_object_size'] 
        self.expected_size = params['PRIOR.expected_object_size'] 
        
        # Size of a object is uniform with tails between min_object_size and max_object_size  
        self.tails_dist_size = 0.1*self.expected_size 
        
        # Location of the BoundingBox centers is uniform in (-1,1) WITH exponential tails
        self.tails_dist_center = 0.1*self.expected_size 
        
        # Put everything on the cude if necessary
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
            k = w.shape[-5]
            
            if(label == "argmax"):
                real_indeces = torch.argmax(w,dim=-5,keepdim=True) #max over boxes dimension
            elif(label == "argmin"):
                real_indeces = torch.argmin(w,dim=-5,keepdim=True) #min over boxes dimension
            else:
                raise Exception('invalid label is neither argmin nor argmax. Label = {}'.format(label))
                
            fake_indeces = torch.arange(start=0, end=k, step=1, dtype=real_indeces.dtype, device=real_indeces.device).view(-1,1,1,1,1)
            mask = (real_indeces == fake_indeces).float()
            return mask
        
        
    def score_observations(self,box_dimfull,putative_imgs,putative_masks,
                           mask_pixel_assignment,definitely_bg_mask,imgs):   
        
        # get the parameters 
        normal_sigma = pyro.param("normal_sigma")
        bg_mu = pyro.param("bg_mu")
        fg_mu = pyro.param("fg_mu")
        bg_sigma = pyro.param("bg_sigma")
        fg_sigma = pyro.param("fg_sigma")
        n_box = putative_imgs.shape[-5]

        # The foreground/background should be drawn from a Cauchy distribtion with scalar parameters 
        # TO IMPROVE: parameter should be different for each channel
        obs_imgs = imgs.unsqueeze(-5).expand(n_box, -1, -1, -1, -1)
        log_p_definitely_bg_cauchy = UnitCauchy(bg_mu,bg_sigma).expand(obs_imgs.shape).mask(definitely_bg_mask).log_prob(obs_imgs)
        log_p_given_bg_cauchy = UnitCauchy(bg_mu,bg_sigma).expand(obs_imgs.shape).mask(mask_pixel_assignment).log_prob(obs_imgs)
        log_p_given_fg_cauchy = UnitCauchy(fg_mu,fg_sigma).expand(obs_imgs.shape).mask(mask_pixel_assignment).log_prob(obs_imgs)
        log_p_given_fg_normal = dist.Normal(putative_imgs,normal_sigma).mask(mask_pixel_assignment).log_prob(obs_imgs)   
                    
        # technically incorrect but it speeds up training and lead to binarized masks
        # The probability of each pixel value is:
        # P(x) = w * P(x|FG) + (1-w) * P(x|BG)
        log_partial_pixel_cauchy = putative_masks*log_p_given_fg_cauchy+(1.0-putative_masks)*log_p_given_bg_cauchy 
        log_partial_pixel_normal = putative_masks*log_p_given_fg_normal+(1.0-putative_masks)*log_p_given_bg_cauchy 
                        
        # compute logp
        logp_definitely_bg = torch.sum(log_p_definitely_bg_cauchy,dim=(-1,-2,-3)) 
        logp_box_off       = torch.sum(log_p_given_bg_cauchy,dim=(-1,-2,-3))
        logp_box_on_cauchy = torch.sum(log_partial_pixel_cauchy,dim=(-1,-2,-3))
        logp_box_on_normal = torch.sum(log_partial_pixel_normal,dim=(-1,-2,-3))

        assert logp_definitely_bg.shape == logp_box_off.shape == logp_box_on_cauchy.shape == logp_box_on_normal.shape

        # package the logp
        common_logp    = logp_definitely_bg/n_box
        log_probs = collections.namedtuple('logp', 'logp_off, logp_on_cauchy, logp_on_normal')._make(
                [common_logp+logp_box_off, common_logp + logp_box_on_cauchy, common_logp + logp_box_on_normal])           

        
        # compute the regularizations
        volume_box  = (box_dimfull.bw*box_dimfull.bh).squeeze(-1)
        volume_mask = torch.sum(mask_pixel_assignment*putative_masks,dim=(-1,-2,-3))
        
        
        #- reg1: bounding box should be as small as possible --#
        #- Note that volume_mask is detached from computation graph, 
        #- therefore this regolarization can only make box_volume smaller 
        #- not make the mask_volume larger.
        with torch.no_grad():
            volume_box_min = torch.tensor(self.min_size*self.min_size, device=volume_mask.device, dtype=volume_mask.dtype)
            volume_min     = torch.max(volume_mask,volume_box_min)
        reg_small_box_size = (volume_box/volume_min - 1.0)**2
        
        #- reg 2: mask should occupy at least 10% of the box -#
        #- Note that the box volume is detached from computation graph, 
        #- therefore this regolarization can only make mask_volume larger not the 
        #- box_volume smaller.
        #- This is the continuous version of:
        #- a. if of>0.1 cost = 0
        #- b. is of<0.1 cost is exponential increasing 
        of = volume_mask/volume_box.detach() # occupaid fraction
        reg_big_mask_volume = torch.exp(50*(0.1-of))
        
        #- reg 3: mask should have small total variations -#
        #- TotVar = integral of the absolute gradient -----#
        #- This is L1 b/c we want discountinuity ----------#
        pixel_weights = putative_masks*mask_pixel_assignment
        grad_x = torch.sum(torch.abs(pixel_weights[:,:,:,:,:-1] - pixel_weights[:,:,:,:,1:]),dim=(-1,-2,-3))
        grad_y = torch.sum(torch.abs(pixel_weights[:,:,:,:-1,:] - pixel_weights[:,:,:,1:,:]),dim=(-1,-2,-3))
        reg_tot_var_mask = (grad_x+grad_y)
  
            
        # - reg 4: mask should have small or no overlap ---------------#
        # - Question: Assign the cost to the second most likely mask? -#
        values, indeces = torch.topk(putative_masks, k=2, dim=0, largest=True)  # shape: batch x 2 x 1 x width x height
        prod = torch.prod(values, dim=0, keepdim=True)  # shape batch x 1 x 1 x width x height
        with torch.no_grad():
            fake_indeces = torch.arange(start=0, end=n_box, step=1,
                                        dtype=indeces.dtype, device=indeces.device).view(-1, 1, 1, 1, 1)
        assignment_mask = (indeces[-1:, :, :, :, :] == fake_indeces).float()
        reg_overlap_mask = torch.sum(prod * assignment_mask, dim=(-1, -2, -3)) ** 2
        regularizations = collections.namedtuple('reg', "small_box_size big_mask_volume tot_var_mask overlap_mask")._make(
            [reg_small_box_size, reg_big_mask_volume, reg_tot_var_mask, reg_overlap_mask])

        assert reg_small_box_size.shape == reg_big_mask_volume.shape == reg_tot_var_mask.shape == reg_overlap_mask.shape
        return log_probs, regularizations
    
    def guide(self, imgs=None, epoch=None):
        
        """ The GUIDE takes a mini-batch of images and: 
            1. run the inference to get: zwhere,zwhat
            2. sample:
                - z ~ N(z_mu, z_std) where each component of z is drawn independently
                - c ~ Bernulli(p)
                - cxcy ~ N(cxcy_mu,0.1) 
                - dxdy ~ gamma(dxdy_mu,0.1) 
        """

        # Trick #
        if imgs is None:
            imgs = torch.zeros(8, self.ch_raw_image, self.size_raw_image, self.size_raw_image)
            if self.use_cuda:
                imgs = imgs.cuda()
        assert len(imgs.shape) == 4
        batch_size, ch, width, height = imgs.shape
        assert(width == height)
        one = torch.ones([1], dtype=imgs.dtype, device=imgs.device)
        # End of Trick #

        # register the modules
        pyro.module("inference", self.inference)
        
        #register the oparameters
        pyro.param("std_bx", one, constraint=constraints.greater_than(0.1))
        pyro.param("std_by", one, constraint=constraints.greater_than(0.1))
        pyro.param("std_bw", one, constraint=constraints.greater_than(0.1))
        pyro.param("std_bh", one, constraint=constraints.greater_than(0.1))

        with pyro.plate("batch", batch_size, dim =-1 ):
            
            # run the inference #
            self.guide_results, self.guide_kl = self.inference.forward(imgs,
                                                                       prob_corr_factor=self.prob_corr_factor,
                                                                       overlap_threshold=self.overlap_threshold,
                                                                       randomize_nms_factor=self.randomize_nms_factor,
                                                                       n_objects_max=self.n_objects_max,
                                                                       topk_only=False,
                                                                       noisy_sampling=True)

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
        # Trick #
        if imgs is None:
            observed = False
            imgs = torch.zeros(8, self.ch_raw_image, self.size_raw_image, self.size_raw_image)
            if self.use_cuda:
                imgs=imgs.cuda()
        else:
            observed = True
        assert len(imgs.shape) == 4
        batch_size, ch, width, height = imgs.shape
        assert width == height
        one = torch.ones(1, dtype=imgs.dtype, device=imgs.device)
        # End of Trick #

        # register the modules 
        pyro.module("generator",  self.generator)

        # register the parameters of the distribution used to score the results
        pyro.param("bg_mu", 0.1*one, constraint=constraints.unit_interval)
        pyro.param("fg_mu", 0.9*one, constraint=constraints.unit_interval)
        pyro.param("bg_sigma", 0.2*one, constraint=constraints.interval(0.01, 0.25))
        pyro.param("fg_sigma", 0.2*one, constraint=constraints.interval(0.01, 0.25))
        pyro.param("normal_sigma", 0.2*one, constraint=constraints.interval(0.01, 0.25))

        with pyro.plate("batch", batch_size, dim=-1):
            
            generated = self.generator(z_where=self.guide_results.z_where,
                                       z_what=self.guide_results.z_what,
                                       z_mask=self.guide_results.z_mask,
                                       width_raw=width,
                                       height_raw=height)
            putative_imgs = generated.big_mu_sigmoid
            putative_masks = generated.big_w_sigmoid

            # Resolve the conflict. Each pixel belongs to only one FG object
            # If a pixel does not belong to any object it belongs to the background
            mask_pixel_assignment = self.mask_argmin_argmax(putative_masks, "argmax")
            # mask_pixel_assignment = self.mask_argmin_argmax(putative_masks*p_inferred[...,None,None,None],"argmax")
            definitely_bg_mask = (torch.sum(mask_pixel_assignment, dim=-5, keepdim=True) == 0.0)

            if observed:
                logp, reg = self.score_observations(self.guide_results.z_where, putative_imgs, putative_masks,
                                                    mask_pixel_assignment, definitely_bg_mask, imgs)

                total_reg = self.lambda_small_box_size*reg.small_box_size + \
                            self.lambda_big_mask_volume*reg.big_mask_volume + \
                            self.lambda_tot_var_mask*reg.tot_var_mask + \
                            self.lambda_overlap*reg.overlap_mask

                kl_tot = self.guide_kl.bx + \
                         self.guide_kl.by + \
                         self.guide_kl.bw + \
                         self.guide_kl.bh + \
                         self.guide_kl.zwhat + \
                         self.guide_kl.zmask

                p = self.guide_results.prob

                total_off = (getattr(self, 'LOSS_ZMASK', 0.0) + getattr(self, 'LOSS_ZWHAT', 0.0)) * logp.logp_off
                total_on = getattr(self, 'LOSS_ZMASK', 0.0) * (logp.logp_on_cauchy - total_reg) + \
                           getattr(self, 'LOSS_ZWHAT', 0.0) * (logp.logp_on_normal - total_reg)
                objective = (1 - p) * total_off + p * (total_on - kl_tot)
                # print("objective.shape", objective.shape)
                with pyro.plate("n_objects", objective.shape[0], dim=-2):
                    pyro.sample("OBJECTIVE", CustomLogProbTerm(custom_log_prob=objective), obs=objective)

    def reconstruct_img(self, original_image, bounding_box=False):
        if self.use_cuda:
            original_image = original_image.cuda()
        
        batch_size, ch, width, height = original_image.shape
        assert width == height
        self.eval()  # set the model into evaluation mode
        with torch.no_grad():  # do not keep track of the gradients
            
            #--------------------------#
            #-- 1. run the inference --#
            #--------------------------#      
            results, kl = self.inference.forward(original_image,
                                                 prob_corr_factor=self.prob_corr_factor,
                                                 overlap_threshold=self.overlap_threshold,
                                                 randomize_nms_factor=self.randomize_nms_factor,
                                                 n_objects_max=self.n_objects_max,
                                                 topk_only=False,
                                                 noisy_sampling=False)

            p = results.prob
            assert p.shape == (self.n_objects_max, batch_size)

            #--------------------------------#
            #--- 2. Run the model forward ---#
            #--------------------------------#
            generated = self.generator(z_where=results.z_where,
                                       z_what=results.z_what,
                                       z_mask=results.z_mask,
                                       width_raw=width,
                                       height_raw=height)
            putative_imgs = generated.big_mu_sigmoid
            putative_masks = generated.big_w_sigmoid
                              
            mask_pixel_assignment = self.mask_argmin_argmax(putative_masks, "argmax")
            definitely_bg_mask = (torch.sum(mask_pixel_assignment, dim=-5, keepdim=True) == 0.0)
            
            #---------------------------------#
            #--- 3. Score the model ----------#
            #---------------------------------#
            logp, reg = self.score_observations(results.z_where, putative_imgs, putative_masks,
                                               mask_pixel_assignment, definitely_bg_mask, original_image)
            
            total_reg = self.lambda_small_box_size*reg.small_box_size + \
                        self.lambda_big_mask_volume*reg.big_mask_volume + \
                        self.lambda_tot_var_mask*reg.tot_var_mask + \
                        self.lambda_overlap*reg.overlap_mask
           
            #---------------------------------#
            #----- 4. Reconstruct images -----#
            #---------------------------------#
            box_is_active = (p > 0.5).float()[..., None, None, None]  # add singleton for ch,w,h
            fg_mask = (mask_pixel_assignment*putative_masks > 0.0).float()
            reconstructed_image = torch.sum(box_is_active*fg_mask*putative_imgs, dim=-5)
            
            # 3. If bounding_box == True compute the bounding box
            if bounding_box == False:
                return reconstructed_image, results, putative_imgs, putative_masks, logp, reg, total_reg
            elif bounding_box == True:
                bounding_boxes = self.draw_batch_of_images_with_bb_only(prob=results.prob,
                                                                        z_where=results.z_where,
                                                                        width=width,
                                                                        height=height)
                reconstructed_image_with_bb = bounding_boxes + reconstructed_image
                return reconstructed_image_with_bb, results, putative_imgs, putative_masks, logp, reg, total_reg

    def draw_batch_of_images_with_bb_only(self, prob=None, z_where=None, width=None, height=None):
       
        # Exttract the probabilities for each box
        assert len(prob.shape) == 2
        n_boxes, batch_size = prob.shape

        # prepare the storage
        batch_bb_np = np.zeros((batch_size, width, height, 3)) # numpy storage for bounding box images
        
        # compute the coordinates of the bounding boxes and the probability of each box
        x1 = z_where.bx - 0.5*z_where.bw
        x3 = z_where.bx + 0.5*z_where.bw
        y1 = z_where.by - 0.5*z_where.bh
        y3 = z_where.by + 0.5*z_where.bh

        assert x1.shape == x3.shape == y1.shape == y3.shape == prob.shape
        x1y1x3y3 = torch.stack((x1, y1, x3, y3), dim=-1)

        # draw the bounding boxes
        for b in range(batch_size):
        
            # Draw on PIL
            img = Image.new('RGB', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            for n in range(n_boxes):
                #if(prob[n,b,0]>0.0):
                draw.rectangle(x1y1x3y3[n, b, :].cpu().numpy(), outline='red', fill=None)
            batch_bb_np[b, ...] = np.array(img.getdata(), np.uint8).reshape(width, height, 3)

        # Transform np to torch, rescale from [0,255] to (0,1) 
        batch_bb_torch = torch.from_numpy(batch_bb_np).permute(0, 3, 2, 1).float()/255  # permute(0,3,2,1) is CORRECT
        return batch_bb_torch.to(prob.device)
