from .utilities import *
from .vae_parts import *
from .namedtuple import *
from typing import Union


def pretty_print_metrics(epoch: int,
                         metric: tuple,
                         is_train: bool = True) -> str:
    if is_train:
        s = 'Train [epoch {0:4d}] loss={1[loss]:.3f}, nll={1[nll]:.3f}, reg={1[reg]:.3f}, kl_tot={1[kl_tot]:.3f}, sparsity={1[sparsity]:.3f}, acc={1[accuracy]:.3f}, fg_fraction={1[fg_fraction]:.3f}, geco_sp={1[geco_sparsity]:.3f}, geco_bal={1[geco_balance]:.3f}'.format(epoch, metric)
    else:
        s = 'Test  [epoch {0:4d}] loss={1[loss]:.3f}, nll={1[nll]:.3f}, reg={1[reg]:.3f}, kl_tot={1[kl_tot]:.3f}, sparsity={1[sparsity]:.3f}, acc={1[accuracy]:.3f}, fg_fraction={1[fg_fraction]:.3f}, geco_sp={1[geco_sparsity]:.3f}, geco_bal={1[geco_balance]:.3f}'.format(epoch, metric)
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


def file2resumed(path: str, device: Optional[str]=None):
    """ wrapper around torch.load """
    if device is None:
        resumed = torch.load(path)
    elif device == 'cuda':
        resumed = torch.load(path, map_location="cuda:0")
    elif device == 'cpu':
        resumed = torch.load(path, map_location=torch.device('cpu'))
    else:
        raise Exception("device is not recognized")
    return resumed


def load_info(resumed,
              load_params: bool = False,
              load_epoch: bool = False,
              load_history: bool = False) -> Checkpoint:

    epoch = resumed['epoch'] if load_epoch else None
    hyperparam_dict = resumed['hyperparam_dict'] if load_params else None
    history_dict = resumed['history_dict'] if load_history else None

    return Checkpoint(history_dict=history_dict, epoch=epoch, hyperparams_dict=hyperparam_dict)


def load_model_optimizer(resumed,
                         model: Union[None, torch.nn.Module] = None,
                         optimizer: Union[None, torch.optim.Optimizer] = None,
                         overwrite_member_var: bool = False):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model is not None:

        # load member variables
        if overwrite_member_var:
            for key, value in resumed['model_member_var'].items():
                setattr(model, key, value)

        # load the modules
        model.load_state_dict(resumed['model_state_dict'])
        model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(resumed['optimizer_state_dict'])


def instantiate_optimizer(model: torch.nn.Module,
                          dict_params_optimizer: dict) -> torch.optim.Optimizer:
    
    # split the parameters between GECO and NOT_GECO
    geco_params, other_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("geco"):
            geco_params.append(param)
        else:
            other_params.append(param)

    if dict_params_optimizer["type"] == "adam":
        optimizer = torch.optim.Adam([{'params': geco_params, 'lr': dict_params_optimizer["base_lr_geco"],
                                       'betas': dict_params_optimizer["betas_geco"]},
                                      {'params': other_params, 'lr': dict_params_optimizer["base_lr"],
                                       'betas': dict_params_optimizer["betas"]}],
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
        self.ma_calculator = Moving_Average_Calculator(beta=0.999)  # i.e. average over the last 100 mini-batches

        # Raw image parameters
        self.dict_soft_constraints = params["soft_constraint"]
        self.nms_dict = params["nms"]
        self.sigma_fg = torch.tensor(params["loss"]["fg_std"])[..., None, None]  # add singleton for width, height
        self.sigma_bg = torch.tensor(params["loss"]["bg_std"])[..., None, None]  # add singleton for width, height

        self.geco_dict = params["GECO"]
        self.input_img_dict = params["input_image"]

        self.geco_sparsity_factor = torch.nn.Parameter(data=torch.tensor(self.geco_dict["factor_sparsity_range"][1]),
                                                       requires_grad=True)
        self.geco_balance_factor = torch.nn.Parameter(data=torch.tensor(self.geco_dict["factor_balance_range"][1]),
                                                      requires_grad=True)

        # Put everything on the cude if necessary
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            self.sigma_fg = self.sigma_fg.cuda()
            self.sigma_bg = self.sigma_bg.cuda()

    def compute_regularizations(self,
                                inference: Inference,
                                verbose: bool = False,
                                chosen: int = None) -> RegMiniBatch:
        """ Compute the mean regularization over each image. 
            These regularizations are written only in terms of p.detached and big_masks.
            1. fg_pixel_fraction determines the overall foreground budget
            2. overlap make sure that mask do not overlap
        """

        # 1. Masks should not overlap:
        # A = (x1+x2+x3)^2 = x1^2 + x2^2 + x3^2 + 2 x1*x2 + 2 x1*x3 + 2 x2*x3
        # Therefore sum_{i \ne j} x_i x_j = x1*x2 + x1*x3 + x2*x3 = 0.5 * [(sum xi)^2 - (sum xi^2)]
        x = inference.prob[..., None, None, None] * inference.big_mask_NON_interacting
        sum_x = torch.sum(x, dim=-5)  # sum over boxes
        sum_x_squared = torch.sum(x * x, dim=-5)
        tmp_value = (sum_x * sum_x - sum_x_squared).clamp(min=0)
        overlap = 0.5 * torch.sum(tmp_value, dim=(-1, -2, -3))  # sum over ch, w, h
        cost_overlap = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                    var_name="overlap",
                                                    var_value=overlap,
                                                    verbose=verbose,
                                                    chosen=chosen)

        # Mask should have a min and max volume
        volume_mask_absolute = torch.sum(inference.big_mask, dim=(-1, -2, -3))  # sum over ch,w,h
        cost_volume_absolute = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                            var_name="mask_volume_absolute",
                                                            var_value=volume_mask_absolute,
                                                            verbose=verbose,
                                                            chosen=chosen)
        cost_volume_absolute_times_prob = torch.sum(cost_volume_absolute * inference.prob, dim=-2)  # sum over boxes

        # Note that before returning I am computing the mean over the batch_size (which is the only dimension left)
        # assert cost_fg_pixel_fraction.shape == cost_overlap.shape
        return RegMiniBatch(cost_overlap=cost_overlap.mean(),
                            cost_vol_absolute=cost_volume_absolute_times_prob.mean())
    
    def NLL_MSE(self, output: torch.tensor, target: torch.tensor, sigma: torch.tensor) -> torch.Tensor:
        return ((output-target)/sigma).pow(2)

    def compute_metrics(self,
                        imgs_in: torch.Tensor,
                        inference: Inference,
                        reg: RegMiniBatch) -> MetricMiniBatch:

        # Preparation
        batch_size, ch, w, h = imgs_in.shape
        n_boxes = inference.big_mask.shape[-5]
        n_obj_counts = (inference.prob > 0.5).float().sum(-2).detach()  # sum over boxes dimension
        p_times_area_map = inference.p_map * inference.area_map
        mixing_k = inference.big_mask * inference.prob[..., None, None, None]
        mixing_fg = torch.sum(mixing_k, dim=-5)
        mixing_bg = torch.ones_like(mixing_fg) - mixing_fg
        assert len(mixing_fg.shape) == 4  # batch, ch=1, w, h

        # 1. Observation model
        # if the observation_std is fixed then normalization 1.0/sqrt(2*pi*sigma^2) is irrelevant.
        # We are better off using MeanSquareError metric
        nll_k = self.NLL_MSE(output=inference.big_img, target=imgs_in, sigma=self.sigma_fg)
        nll_bg = self.NLL_MSE(output=inference.big_bg, target=imgs_in, sigma=self.sigma_bg)  # batch_size, ch, w, h
        nll_av = (torch.sum(mixing_k * nll_k, dim=-5) + mixing_bg * nll_bg).mean()  # mean over batch_size, ch, w, h

        # 2. Regularizations
        reg_av: torch.Tensor = torch.zeros(1, device=imgs_in.device, dtype=imgs_in.dtype)  # shape: 1
        for f in reg._fields:
            reg_av += getattr(reg, f)
        reg_av = reg_av.mean()  # shape []

        # 2. Sparsity should encourage:
        # 1. small probabilities
        # 2. tight bounding boxes
        # 3. tight masks
        # Old solution: sparsity = p * area_box -> leads to square masks b/c they have no cost and lead to lower kl_mask
        # Medium solution: sparsity = \sum_{i,j} (p * area_box) + \sum_k (p_chosen * area_mask_chosen)
        #                        = fg_fraction_box + fg_fraction_box
        #                        -> lead to many small object b/c there is no cost
        # New solution = add term sum p so that many objects are penalized
        fg_fraction_mask_av = torch.sum(mixing_fg) / torch.numel(mixing_fg)  # divide by # total pixel
        fg_fraction_box_av = torch.sum(p_times_area_map) / torch.numel(mixing_fg)  # divide by # total pixel
        prob_total_av = torch.sum(inference.p_map)/ (batch_size * n_boxes)  # quickly converge to order 1

        # 4. compute the KL for each image
        kl_zinstance_av = torch.mean(inference.kl_zinstance_each_obj)  # choose latent dim z so that this number is order 1.
        kl_zwhere_av = torch.sum(inference.kl_zwhere_map) / (batch_size * n_boxes * 4)  # order 1
        kl_logit_tot = torch.sum(inference.kl_logit_map) / batch_size  # this will be normalized by running average -> order 1

        # 5. compute the moving averages
        with torch.no_grad():
            input_dict = {"kl_logit_tot": kl_logit_tot.item()}
            
            # Only if in training mode I accumulate the moving average
            if self.training:
                ma_dict = self.ma_calculator.accumulate(input_dict)
            else:
                ma_dict = input_dict


        # Note that I clamp in_place
        with torch.no_grad():
            f_balance = self.geco_balance_factor.data.clamp_(min=min(self.geco_dict["factor_balance_range"]),
                                                             max=max(self.geco_dict["factor_balance_range"]))
            f_sparsity = self.geco_sparsity_factor.data.clamp_(min=min(self.geco_dict["factor_sparsity_range"]),
                                                               max=max(self.geco_dict["factor_sparsity_range"]))

        # 6. Loss_VAE
        # TODO:
        # 1. try: loss_vae = f_balance * (nll_av + reg_av) + (1.0-f_balance) * (kl_av + f_sparsity * sparsity_av)
        # 2. move reg_av to the other size, i.e. proportional to 1-f_balance
        kl_av = kl_zinstance_av + kl_zwhere_av + kl_logit_tot / (1E-3 + ma_dict["kl_logit_tot"])
        sparsity_av = fg_fraction_mask_av + fg_fraction_box_av + prob_total_av
        assert nll_av.shape == reg_av.shape == kl_av.shape == sparsity_av.shape

        loss_vae = f_sparsity * sparsity_av + f_balance * (nll_av + reg_av) + (1.0 - f_balance) * kl_av



        # GECO BUSINESS
        if self.geco_dict["is_active"]:
            with torch.no_grad():
                # If fg_fraction_av > max(target) -> tmp1 > 0 -> delta_1 < 0 -> too much fg -> increase sparsity
                # If fg_fraction_av < min(target) -> tmp2 > 0 -> delta_1 > 0 -> too little fg -> decrease sparsity
                fg_fraction_av = torch.mean(mixing_fg)
                tmp1 = max(0, fg_fraction_av - max(self.geco_dict["target_fg_fraction"]))
                tmp2 = max(0, min(self.geco_dict["target_fg_fraction"]) - fg_fraction_av)
                delta_1 = torch.tensor(tmp2 - tmp1, dtype=loss_vae.dtype, device=loss_vae.device, requires_grad=False)

                # If nll_av > max(target) -> tmp3 > 0 -> delta_2 < 0 -> bad reconstruction -> increase f_balance
                # If nll_av < min(target) -> tmp4 > 0 -> delta_2 > 0 -> too good reconstruction -> decrease f_balance
                tmp3 = max(0, nll_av - max(self.geco_dict["target_nll"]))
                tmp4 = max(0, min(self.geco_dict["target_nll"]) - nll_av)
                delta_2 = torch.tensor(tmp4 - tmp3, dtype=loss_vae.dtype, device=loss_vae.device, requires_grad=False)

            loss_1 = self.geco_sparsity_factor * delta_1
            loss_2 = self.geco_balance_factor * delta_2
            loss_av = loss_vae + loss_1 + loss_2 - (loss_1 + loss_2).detach()
        else:
            delta_1 = torch.tensor(0.0, dtype=loss_vae.dtype, device=loss_vae.device)
            delta_2 = torch.tensor(0.0, dtype=loss_vae.dtype, device=loss_vae.device)
            loss_av = loss_vae

        # add everything you want as long as there is one loss
        return MetricMiniBatch(loss=loss_av,
                               nll=nll_av.detach(),
                               reg=reg_av.detach(),
                               kl_tot=kl_av.detach(),
                               kl_instance=kl_zinstance_av.detach(),
                               kl_where=kl_zwhere_av.detach(),
                               kl_logit=kl_logit_tot.detach(),
                               sparsity=sparsity_av.detach(),
                               fg_fraction=torch.mean(mixing_fg).detach(),
                               geco_sparsity=f_sparsity,
                               geco_balance=f_balance,
                               delta_1=delta_1,
                               delta_2=delta_2,
                               length_GP=inference.length_scale_GP.detach(),
                               n_obj_counts=n_obj_counts)

    def compute_edges(self, mixing_k: torch.tensor, radius_nn: int = 2):
        """ INPUT:  mixing_k of shape --> n_boxes, batch_shape, 1, w, h
            OUTPUT: edge of shape ------>          batch_shape, (2*r+1)*(2*r+1), w, h
            where each channels contains the value of e_ij = sum_{k} sqrt(mixing_{i,k} mixing_{j,k})
            and j is the pixels shifted by ....
        """
        n_boxes, batch_shape, ch_in, w, h = mixing_k.shape
        assert ch_in == 1
        ch_out = (2 * radius_nn + 1) * (2 * radius_nn + 1)
        pad = radius_nn+1
        edges = torch.zeros((batch_shape, ch_out, w, h), device=mixing_k.device, dtype=mixing_k.dtype)

        # Pad width and height with zero before rolling to avoid spurious connections due to PBC
        pad_mixing_k = F.pad(mixing_k, pad=[pad, pad, pad, pad], mode="constant", value=0.0)
        for ch, pad_mixing_k_shifted in enumerate(roller_2d(pad_mixing_k, radius_nn=radius_nn)):
            edges[:, ch] = torch.sqrt(pad_mixing_k * pad_mixing_k_shifted).sum(dim=-5)[:, 0, pad:(pad+w), pad:(pad+h)]
        return edges

    def segment(self, batch_imgs,
                n_objects_max: Optional[int] = None,
                prob_corr_factor: Optional[float] = None,
                overlap_threshold: Optional[float] = None,
                draw_boxes: bool = False,
                noisy_sampling: bool = True,
                radius_nn: int = 2):
        """ Edges: shape batch, radius_NN * radius_NN, width, height
            In each channel stores

        """

        n_objects_max = self.input_img_dict["n_objects_max"] if n_objects_max is None else n_objects_max
        prob_corr_factor = getattr(self, "prob_corr_factor", 0.0) if prob_corr_factor is None else prob_corr_factor
        overlap_threshold = self.nms_dict["overlap_threshold"] if overlap_threshold is None else overlap_threshold

        with torch.no_grad():
            inference = self.inference_and_generator(imgs_in=batch_imgs,
                                                     generate_synthetic_data=False,
                                                     prob_corr_factor=prob_corr_factor,
                                                     overlap_threshold=overlap_threshold,
                                                     n_objects_max=n_objects_max,
                                                     topk_only=False,
                                                     noisy_sampling=noisy_sampling,
                                                     bg_is_zero=True,
                                                     bg_resolution=(1, 1))

            # make the segmentation mask (one integer for each object)
            mixing_k = inference.big_mask * inference.prob[..., None, None, None]
            edges = self.compute_edges(mixing_k, radius_nn=radius_nn)

            most_likely_mixing, index = torch.max(mixing_k, dim=-5, keepdim=True)  # 1, batch_size, 1, w, h
            int_seg_mask = ((most_likely_mixing > 0.5) * (index + 1)).squeeze(-5)  # bg = 0 fg = 1,2,3,...

            if draw_boxes:
                bounding_boxes = draw_bounding_boxes(prob=inference.prob,
                                                     bounding_box=inference.bounding_box,
                                                     width=int_seg_mask.shape[-2],
                                                     height=int_seg_mask.shape[-1])
            else:
                bounding_boxes = torch.zeros_like(int_seg_mask)

        return bounding_boxes + int_seg_mask, edges

    def tiling(self,
               single_img: torch.Tensor,
               crop_size: tuple,
               stride: tuple,
               n_objects_max_per_patch: Optional[int] = None,
               prob_corr_factor: Optional[float] = None,
               overlap_threshold: Optional[float] = None,
               radius_nn: int=2,
               batch: int=64):
        """ Uses a sliding window approach to collect a co_objectiveness information
            about the pixels of a large image """

        n_objects_max_per_patch = self.input_img_dict["n_objects_max"] if n_objects_max_per_patch is \
                                                                          None else n_objects_max_per_patch
        prob_corr_factor = getattr(self, "prob_corr_factor", 0.0) if prob_corr_factor is None else prob_corr_factor
        overlap_threshold = self.nms_dict["overlap_threshold"] if overlap_threshold is None else overlap_threshold

        assert crop_size[0] % stride[0] == 0, "crop and stride size are NOT compatible"
        assert crop_size[1] % stride[1] == 0, "crop and stride size are NOT compatible"
        assert len(single_img.shape) == 3

        with torch.no_grad():

            w_img, h_img = single_img.shape[-2:]
            ch_out = (2*radius_nn+1)*(2*radius_nn+1)
            n_prediction = (crop_size[0]//stride[0]) * (crop_size[1]//stride[1])
            print(f'Each pixel will be segmented {n_prediction} times')

            pad_w = crop_size[0] - stride[0]
            pad_h = crop_size[1] - stride[1]
            pad_list = [pad_w, crop_size[0], pad_h, crop_size[1]]
            try:
                img_padded = F.pad(single_img.unsqueeze(0),
                                   pad=pad_list, mode='reflect')  # 1, ch_in, w_pad, h_pad
            except RuntimeError:
                img_padded = F.pad(single_img.unsqueeze(0),
                                   pad=pad_list, mode='constant', value=0)  # 1, ch_in, w_pad, h_pad
            w_paddded, h_padded = img_padded.shape[-2:]

            segmentation_edges = torch.zeros((ch_out, w_paddded, h_padded),
                                             device=single_img.device,
                                             dtype=single_img.dtype)

            # Build a list with the locations of the corner of the images
            location_of_corner = []
            for i in range(0, w_img + pad_w, stride[0]):
                for j in range(0, h_img + pad_h, stride[1]):
                    location_of_corner.append([i, j])
            print(f'I am going to process {len(location_of_corner)} patches')

            # split the list in chunks of batch_size
            ij_list_of_list = [location_of_corner[n:n + batch] for n in range(0, len(location_of_corner), batch)]

            # Build a batch of images and process them
            for n, ij_list in enumerate(ij_list_of_list):
                
                # Build a batch of images
                batch_imgs = torch.cat([img_padded[..., ij[0]:(ij[0] + crop_size[0]), ij[1]:(ij[1] + crop_size[1])]
                                        for ij in ij_list], dim=0)
                
                if (n % 100 == 0) or (n == len(ij_list_of_list)-1):
                    print(f'{n} out of {len(ij_list_of_list)-1} -> batch_of_imgs.shape = {batch_imgs.shape}') 

                # Segment the batch 
                integer_mask, edges = self.segment(batch_imgs,
                                                   n_objects_max=n_objects_max_per_patch,
                                                   prob_corr_factor=prob_corr_factor,
                                                   overlap_threshold=overlap_threshold,
                                                   draw_boxes=False,
                                                   noisy_sampling=True,
                                                   radius_nn=radius_nn)

                # TODO Can we vectorize this?
                for b, ij in enumerate(ij_list):
                    segmentation_edges[:,
                                       ij[0]:(ij[0]+crop_size[0]),
                                       ij[1]:(ij[1]+crop_size[1])] += edges[b, :, :, :]

        return TILING(co_object=segmentation_edges[:, pad_w:pad_w + w_img, pad_h:pad_h + h_img]/n_prediction,
                      raw_img=single_img)

    # this is the generic function which has all the options unspecified
    def process_batch_imgs(self,
                           imgs_in: torch.tensor,
                           generate_synthetic_data: bool,
                           topk_only: bool,
                           draw_image: bool,
                           draw_boxes: bool,
                           verbose: bool,
                           noisy_sampling: bool,
                           prob_corr_factor: float,
                           overlap_threshold: float,
                           n_objects_max: int,
                           bg_is_zero: bool,
                           bg_resolution: tuple):
        """ It needs to return: metric (with a .loss member) and whatever else """

        # Checks
        assert len(imgs_in.shape) == 4
        assert self.input_img_dict["ch_in"] == imgs_in.shape[-3]
        # End of Checks #

        results = self.inference_and_generator(imgs_in=imgs_in,
                                               generate_synthetic_data=generate_synthetic_data,
                                               prob_corr_factor=prob_corr_factor,
                                               overlap_threshold=overlap_threshold,
                                               n_objects_max=n_objects_max,
                                               topk_only=topk_only,
                                               noisy_sampling=noisy_sampling,
                                               bg_is_zero=bg_is_zero,
                                               bg_resolution=bg_resolution)

        regularizations = self.compute_regularizations(inference=results,
                                                       verbose=verbose)

        metrics = self.compute_metrics(imgs_in=imgs_in,
                                       inference=results,
                                       reg=regularizations)

        all_metrics = Metric_and_Reg.from_merge(metrics=metrics, regularizations=regularizations)

        with torch.no_grad():
            if draw_image:
                imgs_rec = draw_img(prob=results.prob,
                                    bounding_box=results.bounding_box,
                                    big_mask=results.big_mask,
                                    big_img=results.big_img,
                                    draw_boxes=draw_boxes)
            else:
                imgs_rec = torch.zeros_like(imgs_in)

        return Output(metrics=all_metrics, inference=results, imgs=imgs_rec)

    def forward(self,
                imgs_in: torch.tensor,
                draw_image: bool = False,
                draw_boxes: bool = False,
                verbose: bool = False):

        return self.process_batch_imgs(imgs_in=imgs_in,
                                       generate_synthetic_data=False,
                                       topk_only=False,
                                       draw_image=draw_image,
                                       draw_boxes=draw_boxes,
                                       verbose=verbose,
                                       noisy_sampling=True, #True if self.training else False,
                                       prob_corr_factor=getattr(self, "prob_corr_factor", 0.0),
                                       overlap_threshold=self.nms_dict.get("overlap_threshold", 0.3),
                                       n_objects_max=self.input_img_dict["n_objects_max"],
                                       bg_is_zero=self.input_img_dict.get("bg_is_zero", True),
                                       bg_resolution=self.input_img_dict.get("bg_resolution", (2, 2)))

    def generate(self,
                 imgs_in: torch.tensor,
                 draw_boxes: bool = False,
                 verbose: bool = False):

        with torch.no_grad():

            return self.process_batch_imgs(imgs_in=torch.zeros_like(imgs_in),
                                           generate_synthetic_data=True,
                                           topk_only=False,
                                           draw_image=True,
                                           draw_boxes=draw_boxes,
                                           verbose=verbose,
                                           noisy_sampling=True,
                                           prob_corr_factor=0.0,
                                           overlap_threshold=self.nms_dict.get("overlap_threshold", 0.3),
                                           n_objects_max=self.input_img_dict["n_objects_max"],
                                           bg_is_zero=True,
                                           bg_resolution=(2, 2))
