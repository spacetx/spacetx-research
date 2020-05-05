from .utilities import *
from .vae_parts import *
from .namedtuple import *
from typing import Union


def pretty_print_metrics(epoch: int,
                         metric: tuple,
                         is_train: bool = True) -> str:
    if is_train:
        s = 'Train [epoch {0:4d}] loss={1[loss]:.3f}, nll={1[nll]:.3f}, reg={1[reg]:.3f}, kl_tot={1[kl_tot]:.3f}, sparsity={1[sparsity]:.3f}, acc={1[accuracy]:.3f}, n_obj={1[n_obj]:.3f}, geco_sp={1[geco_sparsity]:.3f}, geco_nll={1[geco_nll]:.3f}'.format(epoch, metric)
    else:
        s = 'Test  [epoch {0:4d}] loss={1[loss]:.3f}, nll={1[nll]:.3f}, reg={1[reg]:.3f}, kl_tot={1[kl_tot]:.3f}, sparsity={1[sparsity]:.3f}, acc={1[accuracy]:.3f}, n_obj={1[n_obj]:.3f}, geco_sp={1[geco_sparsity]:.3f}, geco_nll={1[geco_nll]:.3f}'.format(epoch, metric)
    return s
    

def save_everything(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    history_dict: dict,
                    hyperparams_dict: dict, epoch: int) -> None:

    all_member_var = model.__dict__
    member_var_to_save = {}
    for k, v in all_member_var.items():
        if not k.startswith("_") and k != 'training':
            member_var_to_save[k] = v

    torch.save({'epoch': epoch,
                'model_member_var': member_var_to_save,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history_dict': history_dict,
                'hyperparam_dict': hyperparams_dict}, path)


def load_info(path: str,
              load_params: bool = False,
              load_epoch: bool = False,
              load_history: bool = False) -> tuple:

    if torch.cuda.is_available():
        resumed = torch.load(path, map_location="cuda:0")
    else:
        resumed = torch.load(path, map_location=torch.device('cpu'))

    epoch = resumed['epoch'] if load_epoch else None
    hyperparams_dict = resumed['hyperparams_dict'] if load_params else None
    history_dict = resumed['history_dict'] if load_history else None

    return Checkpoint(history_dict=history_dict, epoch=epoch, hyperparams_dict=hyperparams_dict)


def load_model_optimizer(path: str,
                         model: Union[None, torch.nn.Module] = None,
                         optimizer: Union[None, torch.optim.Optimizer] = None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        resumed = torch.load(path, map_location="cuda:0")
    else:
        device = torch.device('cpu')
        resumed = torch.load(path, map_location=device)

    if model is not None:

        # load member variables
        member_var = resumed['model_member_var']
        for key, value in member_var.items():
            setattr(model, key, value)

        # load the modules
        model.load_state_dict(resumed['model_state_dict'])
        model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(resumed['optimizer_state_dict'])


def instantiate_optimizer(model: torch.nn.Module,
                          dict_params_optimizer: dict) -> torch.optim.Optimizer:
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(params=model.parameters(),
    #                              lr=dict_params_optimizer["base_lr"],
    #                              betas=dict_params_optimizer["betas"],
    #                              eps=dict_params_optimizer["eps"])
    #return optimizer
    
    # split the parameters between GECO and NOT_GECO
    geco_params, other_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("geco"):
            geco_params.append(param)
        else:
            other_params.append(param)

    if dict_params_optimizer["type"] == "adam":
        optimizer = torch.optim.Adam([{'params': geco_params, 'lr': dict_params_optimizer["base_lr_geco"], 'betas' : dict_params_optimizer["betas_geco"]},
                                      {'params': other_params, 'lr': dict_params_optimizer["base_lr"], 'betas' : dict_params_optimizer["betas"]}],
                                     eps=dict_params_optimizer["eps"],
                                     weight_decay=dict_params_optimizer["weight_decay"])
        
    elif dict_params_optimizer["type"] == "SGD":
        optimizer = torch.optim.SGD([{'params': geco_params, 'lr': dict_params_optimizer["base_lr_geco"]},
                                     {'params': other_params, 'lr': dict_params_optimizer["base_lr"]}],
                                    weight_decay=dict_params_optimizer["weight_decay"])
    else:
        raise Exception
    return optimizer


def instantiate_scheduler(optimizer: torch.optim.Optimizer,
                          dict_params_scheduler: dict) -> torch.optim.lr_scheduler:
    if dict_params_scheduler["scheduler_type"] == "step_LR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=dict_params_scheduler["scheduler_step_size"],
                                                    gamma=dict_params_scheduler["scheduler_gamma"],
                                                    last_epoch=-1)
    else:
        raise Exception
    return scheduler


class CompositionalVae(torch.nn.Module):

    def __init__(self, params: dict) -> None:
        super().__init__()

        # Instantiate all the modules
        self.inference_and_generator = Inference_and_Generation(params)
        self.ma_calculator = Moving_Average_Calculator(beta=0.99)  # i.e. average over the last 100 mini-batches

        # Raw image parameters
        self.dict_soft_constraints = params["soft_constraint"]
        self.nms_dict = params["nms"]
        self.sigma_fg = torch.tensor(params["loss"]["fg_std"])[..., None, None]  # add singleton for width, height
        self.sigma_bg = torch.tensor(params["loss"]["bg_std"])[..., None, None]  # add singleton for width, height

        self.geco_dict = params["GECO"]
        self.input_img_dict = params["input_image"]

        self.geco_factor_sparsity = ConstrainedParam(initial_data=
                                                     torch.tensor(np.median(self.geco_dict["factor_sparsity_range"])),
                                                     transformation=ConstraintBounded(
                                                         lower_bound=np.min(self.geco_dict["factor_sparsity_range"]),
                                                         upper_bound=np.max(self.geco_dict["factor_sparsity_range"])))
        
        self.geco_factor_nll = ConstrainedParam(initial_data=torch.tensor(np.median(self.geco_dict["factor_nll_range"])),
                                                transformation=ConstraintBounded(
                                                    lower_bound=np.min(self.geco_dict["factor_nll_range"]),
                                                    upper_bound=np.max(self.geco_dict["factor_nll_range"])))

        # Put everything on the cude if necessary
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            self.sigma_fg = self.sigma_fg.cuda()
            self.sigma_bg = self.sigma_bg.cuda()

    def draw_img(self, inference: Inference, draw_bounding_box: bool = False) -> torch.tensor:

        rec_imgs_no_bb = torch.sum(inference.prob[..., None, None, None] *
                                   inference.big_mask * inference.big_img, dim=-5)  # sum over boxes
        width, height = rec_imgs_no_bb.shape[-2:]
        
        bounding_boxes = draw_bounding_boxes(prob=inference.prob,
                                             bounding_box=inference.bounding_box,
                                             width=width,
                                             height=height) if draw_bounding_box else torch.zeros_like(rec_imgs_no_bb)
        return bounding_boxes + rec_imgs_no_bb

    def compute_regularizations(self,
                                inference: Inference,
                                verbose: bool = False,
                                chosen: int = None) -> RegMiniBatch:
        """ Compute the mean regularization over each image. 
            These regularizations are written only in terms of p.detached and big_masks.
            1. fg_pixel_fraction determines the overall foreground budget
            2. overlap make sure that mask do not overlap
        """
        
        p_detached_times_mask = inference.prob[..., None, None, None].detach() * inference.big_mask  # singleton ch,w,h
        weighted_fg_mask = torch.sum(p_detached_times_mask, dim=-5)  # sum over boxes
        
        # 1. fg_pixel_fraction should be in a range. This is a foreground budget. 
        # The model should allocate the foreground pixel where most necessary (i.e. where description based on background is bad)
        fraction_fg_pixels = torch.mean(weighted_fg_mask, dim=(-1, -2, -3))  # mean over: ch,w,h
        cost_fg_pixel_fraction = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                              var_name="fg_pixel_fraction",
                                                              var_value=fraction_fg_pixels,
                                                              verbose=verbose,
                                                              chosen=chosen)
        
        # 2. Masks should not overlap:
        # This should change the mask but NOT the probabilities therefore prob are DETACHED.
        # cost_i = 0.5 * sum_{x,y} \sum_{i \ne j} (p_i M_i * p_j M_j)
        #        = 0.5 * sum_{x,y} (\sum_i p_i M_i)^2  - \sum_i (p_i M_i)^2
        square_first_sum_later = torch.sum(p_detached_times_mask * p_detached_times_mask, dim=-5)  # sum over boxes
        sum_first_square_later = weighted_fg_mask * weighted_fg_mask
        # Now I sum over ch,w,h and make sure that the min value is > 0 (due to rounding error I might get <0)
        overlap = 0.5 * torch.sum((sum_first_square_later - square_first_sum_later), dim=(-1, -2, -3)).clamp(min=0)
        cost_overlap = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                    var_name="overlap",
                                                    var_value=overlap,
                                                    verbose=verbose,
                                                    chosen=chosen)

        # Note that before returning I am computing the mean over the batch_size (which is the only dimension left)
        assert cost_fg_pixel_fraction.shape == cost_overlap.shape 
        return RegMiniBatch(cost_fg_pixel_fraction=cost_fg_pixel_fraction.mean(),
                            cost_overlap=cost_overlap.mean())
    
    def NLL_MSE(self, output: torch.tensor, target: torch.tensor, sigma: torch.tensor) -> torch.Tensor:
        return ((output-target)/sigma).pow(2)

    def compute_metrics(self,
                        imgs_in: torch.Tensor,
                        inference: Inference,
                        reg: RegMiniBatch) -> MetricMiniBatch:

        # preparation
        typical_box_size = self.input_img_dict["size_object_expected"] * self.input_img_dict["size_object_expected"]
        n_obj_counts = (inference.prob > 0.5).float().sum(-2)  # sum over boxes dimension
        n_obj_av = n_obj_counts.mean()  # mean over batch_size. This will be used as a label

        # Sum together all the regularization
        reg_av: torch.Tensor = torch.zeros(1, device=imgs_in.device, dtype=imgs_in.dtype)  # shape: 1
        for f in reg._fields:
            reg_av += getattr(reg, f)
        reg_av = reg_av.mean()  # shape []

        # if the observation_std is fixed then normalization 1.0/sqrt(2*pi*sigma^2) is irrelevant.
        # We are better off using MeanSquareError metric
        # Note that nll_off_bg is detached when appears with the minus sign. 
        # This is b/c we want the best possible background on top of which to add FOREGROUD objects
        nll_on = self.NLL_MSE(output=inference.big_img, target=imgs_in, sigma=self.sigma_fg)
        nll_off = self.NLL_MSE(output=inference.bg_mu, target=imgs_in, sigma=self.sigma_bg)  # batch_size, ch, w, h
        delta_nll_obj = torch.mean(inference.big_mask * (nll_on - nll_off), dim=(-1, -2, -3))  # average over: ch, w, h
        delta_nll = torch.sum(inference.prob * delta_nll_obj, dim=0)  # sum over boxes -> batch_size
        nll_av = nll_off.mean() + delta_nll.mean()  # first term has means over batch_size, ch, w, h. Second term meand over batch_size

        # compute the KL for each image
        kl_zmask_av = torch.mean(inference.kl_zmask_each_obj)  # mean over: boxes, batch_size, latent_zmask
        kl_zwhat_av = torch.mean(inference.kl_zwhat_each_obj)  # mean over: boxes, batch_size, latent_zwhat
        small_w, small_h = inference.kl_zwhere_map.shape[-2:]
        kl_zwhere_av = torch.mean(inference.kl_zwhere_map)  # mean over: batch_size, latent_zwhere, width, height
        kl_logit_av = torch.mean(inference.kl_logit_map)/(small_w * small_h)  # mean over: batch_size. Division by area
        
        kl_zwhat_and_mask = 2 * torch.mean(torch.max(inference.kl_zwhat_each_obj,
                                                     inference.kl_zmask_each_obj))  # encourage even split
        kl_total_av = kl_zwhat_and_mask + kl_zwhere_av  #+ kl_logit_av  # one kl MIGHT be much bigger than others

        # compute the sparsity term: (sum over batch, ch=1, w, h and divide by n_instances * typical_size)
        n_box, batch_size, latent_zwhat = inference.kl_zwhat_each_obj.shape
        sparsity_av = torch.sum(inference.p_map * inference.area_map)/(batch_size * typical_box_size * n_box)
        # sparsity_av = torch.sum(inference.p_map)/(batch_size * n_box)

        # For debug I print the metrics
        # print("nll_av.shape", nll_av.shape, nll_av)
        # print("kl_zmask_av.shape", kl_zmask_av.shape, kl_zmask_av)
        # print("kl_zwhat_av.shape", kl_zwhat_av.shape, kl_zwhat_av)
        # print("kl_zwhere_av.shape", kl_zwhere_av.shape, kl_zwhere_av)
        # print("kl_logit_av.shape", kl_logit_av.shape, kl_logit_av)
        # print("kl_total_av.shape", kl_total_av.shape, kl_total_av)
        # print("sparsity_av.shape", sparsity_av.shape, sparsity_av)
        # print("reg_av.shape", reg_av.shape, reg_av)

        # Compute the moving averages
#        with torch.no_grad():
#            # here I am computing additional average over batch_size
#            input_dict = { "n_obj_av": n_obj_av.item(), "nll_av": nll_av.item()}
#            ma_dict = self.ma_calculator.accumulate(input_dict)
#            print("input_dict", input_dict)
#            print("ma_dict", ma_dict)
            
        # Now I have to make the loss using GECO or not
        assert nll_av.shape == reg_av.shape == kl_total_av.shape == sparsity_av.shape

        fspar_c = self.geco_factor_sparsity.forward()
        fnll_c = self.geco_factor_nll.forward()

        loss_vae = kl_total_av + fspar_c.detach() * sparsity_av + fnll_c.detach() * (nll_av + reg_av)
        #loss_vae = kl_total_av + reg_av + fspar_c.detach() * sparsity_av + fnll_c.detach() * nll_av

        if self.geco_dict["is_active"]:

            # If n_obj_av > max(target_n_obj) -> tmp1 > 0 -> too many objects 
            # If n_obj_av < min(target_n_obj) -> tmp2 > 0 -> too few objects
            tmp1 = (n_obj_av - max(self.geco_dict["target_n_obj"])).clamp(min=0.0)
            tmp2 = (min(self.geco_dict["target_n_obj"]) - n_obj_av).clamp(min=0.0)
            #delta_1 = torch.sign(tmp2 - tmp1).detach()
            delta_1 = (tmp2 - tmp1).detach()
            loss_1 = (fspar_c - fnll_c) * delta_1

            # If nll_av > max(target_nll) -> tmp3 > 0 -> bad reconstruction
            # If nll_av < min(target_nll) -> tmp4 > 0 -> too good reconstruction
            tmp3 = (nll_av - max(self.geco_dict["target_nll"])).clamp(min=0.0)
            tmp4 = (min(self.geco_dict["target_nll"]) - nll_av).clamp(min=0.0)
            # delta_2 = torch.sign(tmp4 - tmp3).detach()
            delta_2 = (tmp4 - tmp3).detach()
            loss_2 = fnll_c * delta_2

            loss_av = loss_vae + loss_1 + loss_2 - (loss_1 + loss_2).detach()     
        else:
            loss_av = loss_vae
            delta_1 = 0
            delta_2 = 0

        # add everything you want as long as there is one loss
        return MetricMiniBatch(loss=loss_av,
                               nll=nll_av,
                               reg=reg_av,
                               kl_tot=kl_total_av,
                               kl_what=kl_zwhat_av,
                               kl_mask=kl_zmask_av,
                               kl_where=kl_zwhere_av,
                               kl_logit=kl_logit_av,
                               sparsity=sparsity_av,
                               n_obj=n_obj_av,
                               geco_sparsity=fspar_c,
                               geco_nll=fnll_c,
                               delta_1=delta_1,
                               delta_2=delta_2,
                               n_obj_counts=n_obj_counts)

    def segment(self, img, n_objects_max=None, draw_bounding_box=False):



        n_objects_max = self.input_img_dict["n_objects_max"] if n_objects_max is None else n_objects_max
        self.eval()
        with torch.no_grad():
            results = self.inference_and_generator(imgs_in=img,
                                                   generate_synthetic_data=False,
                                                   prob_corr_factor=getattr(self, "prob_corr_factor", 0.0),
                                                   overlap_threshold=self.nms_dict["overlap_threshold"],
                                                   score_threshold=self.nms_dict["score_threshold"],
                                                   randomize_nms_factor=0.0,  # self.nms_dict["randomize_nms_factor"]
                                                   n_objects_max=n_objects_max,
                                                   topk_only=False,
                                                   noisy_sampling=False)

            # make the segmentation mask (one integer for each object)
            prob_times_big_mask = results.prob[..., None, None, None] * results.big_mask
            most_likely_mask, index = torch.max(prob_times_big_mask, dim=-5, keepdim=True)
            integer_segmentation_mask = ((most_likely_mask > 0.5) * (index + 1)).squeeze(-5)  # bg = 0 fg = 1,2,3,...
       
            if draw_bounding_box:
                bounding_boxes = draw_bounding_boxes(prob=results.prob,
                                                     bounding_box=results.bounding_box,
                                                     width=integer_segmentation_mask.shape[-2],
                                                     height=integer_segmentation_mask.shape[-1]) 
            else:
                bounding_boxes = torch.zeros_like(integer_segmentation_mask)


        return bounding_boxes + integer_segmentation_mask

    def segment_with_tiling(self, img, crop_w, crop_h, stride_w, stride_h, n_objects_max_per_patch=None, draw_bounding_box=False):

        # Initialization
        batch_size, ch, w_raw, h_raw = img.shape
        n_obj_tot = torch.zeros(batch_size, device=img.device, dtype=img.dtype)  # add singleton for ch, w, h
        if n_objects_max_per_patch is None:
            n_objects_max_per_patch = self.input_img_dict["n_objects_max"]

        # solve overlap = (crop - stride) / 2
        # Recompute stride b/c overlap is rounded
        overlap_w = int(np.ceil(0.5*(crop_w - stride_w)))
        overlap_h = int(np.ceil(0.5*(crop_h - stride_h)))
        str_w = crop_w - 2*overlap_w  # new stride
        str_h = crop_h - 2*overlap_h  # new stride
        filter_keep = torch.zeros((crop_w, crop_h), device=img.device, dtype=img.dtype)
        filter_keep[overlap_w:crop_w-overlap_w, overlap_h:crop_h-overlap_h] = 1
        print("overlaps", overlap_w, overlap_h)
        print("strides", str_w, str_h)

        # solve crop + n * stride >= size + 2 * overlap
        n_w = int(np.ceil(float(w_raw + 2*overlap_w - crop_w)/str_w))
        n_h = int(np.ceil(float(h_raw + 2*overlap_h - crop_h)/str_h))
        print("ntiles ->", n_w, n_h)
        print("area covered by tiles", crop_w + n_w*str_w, crop_h + n_h*str_h)
        print("area to cover", w_raw + 2*overlap_w, h_raw + 2*overlap_h)

        # compute how much padding to do to the right (padding to the left is exactly the overlap
        pad_w = n_w * str_w + crop_w - overlap_w - w_raw
        pad_h = n_h * str_h + crop_h - overlap_h - h_raw
        print("pad_w pad_h ->", pad_w, pad_h)

        # assertions
        assert len(img.shape) == 4
        assert isinstance(img, torch.Tensor)
        assert 0.5*crop_w <= str_w <= crop_w
        assert 0.5*crop_h <= str_h <= crop_h

        #try:
        img_padded = F.pad(img, pad=[overlap_w, pad_w, overlap_h, pad_h], mode='reflect')
        print("img_padded",img_padded.shape)
        #except:
        #    img_padded = F.pad(img, pad=[overlap_w, pad_w, overlap_h, pad_h], mode='constant', value=0)

        stitched_segmentation_mask = None

        self.eval()  # do I need this?
        with torch.no_grad():
            for i_w in range(n_w+1):
                w1 = i_w * str_w
                w2 = w1 + crop_w
                for i_h in range(n_h+1):
                    h1 = i_h * str_h
                    h2 = h1 + crop_h
                    # print(h1,h2)
                    integer_mask = self.segment(img=img_padded[..., w1:w2, h1:h2],
                                                n_objects_max=n_objects_max_per_patch,
                                                draw_bounding_box=draw_bounding_box)

                    integer_mask = integer_mask * filter_keep  # this need to be improved
                    integer_mask_shifted = torch.where(integer_mask > 0,
                                                       integer_mask + n_obj_tot.view(-1, 1, 1, 1),
                                                       torch.zeros_like(integer_mask))  # shift the integers
                    n_obj_tot = torch.max(integer_mask_shifted.view(batch_size, -1), dim=-1, keepdim=False)[0]
                    if stitched_segmentation_mask is None:
                        stitched_segmentation_mask = torch.zeros((batch_size, integer_mask.shape[-3], 
                            img_padded.shape[-2], img_padded.shape[-1]), device=img.device, dtype=img.dtype)
                    stitched_segmentation_mask[..., w1:w2, h1:h2] += integer_mask_shifted

        return stitched_segmentation_mask[..., overlap_w:overlap_w+w_raw, overlap_h:overlap_h+h_raw]

    # this is the generic function which has all the options unspecified
    def process_batch_imgs(self,
                           imgs_in: torch.tensor,
                           generate_synthetic_data: bool,
                           topk_only: bool,
                           draw_image: bool,
                           draw_bounding_box: bool,
                           verbose: bool,
                           noisy_sampling: bool,
                           prob_corr_factor: float,
                           overlap_threshold: float,
                           score_threshold: float,
                           randomize_nms_factor: float,
                           n_objects_max: int):
        """ It needs to return: metric (with a .loss member) and whatever else """

        # Checks
        assert len(imgs_in.shape) == 4
        assert self.input_img_dict["ch_in"] == imgs_in.shape[-3]
        # End of Checks #

        results = self.inference_and_generator(imgs_in=imgs_in,
                                               generate_synthetic_data=generate_synthetic_data,
                                               prob_corr_factor=prob_corr_factor,
                                               overlap_threshold=overlap_threshold,
                                               score_threshold=score_threshold,
                                               randomize_nms_factor=randomize_nms_factor,
                                               n_objects_max=n_objects_max,
                                               topk_only=topk_only,
                                               noisy_sampling=noisy_sampling)

        regularizations = self.compute_regularizations(inference=results,
                                                       verbose=verbose)

        metrics = self.compute_metrics(imgs_in=imgs_in,
                                       inference=results,
                                       reg=regularizations)

        all_metrics = Metric_and_Reg.from_merge(metrics=metrics, regularizations=regularizations)

        with torch.no_grad():
            imgs_rec = self.draw_img(inference=results,
                                     draw_bounding_box=draw_bounding_box) if draw_image else torch.zeros_like(imgs_in)

        return Output(metrics=all_metrics, inference=results, imgs=imgs_rec)

    def forward(self,
                imgs_in: torch.tensor,
                draw_image: bool = False,
                draw_bounding_box: bool = False,
                verbose: bool = False):

        noisy_sampling: bool = True if self.training else False
        prob_corr_factor: float = 0.0 if self.prob_corr_factor is None else self.prob_corr_factor

        return self.process_batch_imgs(imgs_in=imgs_in,
                                       generate_synthetic_data=False,
                                       topk_only=False,
                                       draw_image=draw_image,
                                       draw_bounding_box=draw_bounding_box,
                                       verbose=verbose,
                                       noisy_sampling=noisy_sampling,
                                       prob_corr_factor=prob_corr_factor,
                                       overlap_threshold=self.nms_dict["overlap_threshold"],
                                       score_threshold=self.nms_dict["score_threshold"],
                                       randomize_nms_factor=self.nms_dict["randomize_nms_factor"],
                                       n_objects_max=self.input_img_dict["n_objects_max"])

    def generate(self,
                 imgs_in: Optional[torch.tensor] = None,
                 batch_size: int = 4,
                 draw_bounding_box: bool = False,
                 verbose: bool = False):

        with torch.no_grad():
            if imgs_in is None:
                imgs_in = torch.sigmoid(torch.randn(batch_size,
                                                    self.input_img_dict["ch_in"],
                                                    self.input_img_dict["size_raw_image"],
                                                    self.input_img_dict["size_raw_image"]))
                if self.use_cuda:
                    imgs_in = imgs_in.cuda()

            else:
                imgs_in = torch.sigmoid(torch.randn_like(imgs_in))

            return self.process_batch_imgs(imgs_in=imgs_in,
                                           generate_synthetic_data=True,
                                           topk_only=False,
                                           draw_image=True,
                                           draw_bounding_box=draw_bounding_box,
                                           verbose=verbose,
                                           noisy_sampling=True,
                                           prob_corr_factor=0.0,
                                           overlap_threshold=self.nms_dict["overlap_threshold"],
                                           score_threshold=self.nms_dict["score_threshold"],
                                           randomize_nms_factor=self.nms_dict["randomize_nms_factor"],
                                           n_objects_max=self.input_img_dict["n_objects_max"])

