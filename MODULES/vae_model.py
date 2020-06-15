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


def load_info(path: str,
              load_params: bool = False,
              load_epoch: bool = False,
              load_history: bool = False) -> Checkpoint:

    if torch.cuda.is_available():
        resumed = torch.load(path) #, map_location="cuda:0")
    else:
        resumed = torch.load(path, map_location=torch.device('cpu'))

    epoch = resumed['epoch'] if load_epoch else None
    hyperparams_dict = resumed['hyperparam_dict'] if load_params else None
    history_dict = resumed['history_dict'] if load_history else None

    return Checkpoint(history_dict=history_dict, epoch=epoch, hyperparams_dict=hyperparams_dict)


def load_ckpt(path: str, device: Optional[str]=None):
    if device is None:
        resumed = torch.load(path)
    elif device == 'cuda':
        resumed = torch.load(path, map_location="cuda:0")
        #device = torch.device("cuda")
        #resumed = torch.load(path, map_location=device)
    elif device == 'cpu':
        device = torch.device('cpu')
        resumed = torch.load(path, map_location=device)
    else:
        raise Exception("device is not recognized")

    return resumed


def load_model_optimizer(ckpt,
                         model: Union[None, torch.nn.Module] = None,
                         optimizer: Union[None, torch.optim.Optimizer] = None,
                         overwrite_member_var: bool = False):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model is not None:

        # load member variables
        if overwrite_member_var:
            for key, value in ckpt['model_member_var'].items():
                setattr(model, key, value)

        # load the modules
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])


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
        n_obj_counts = (inference.prob > 0.5).float().sum(-2).detach()  # sum over boxes dimension
        p_times_area_map = inference.p_map * inference.area_map
        mixing_k = inference.big_mask * inference.prob[..., None, None, None]
        mixing_fg = torch.sum(mixing_k, dim=-5)
        mixing_bg = 1.0 - mixing_fg
        assert len(mixing_fg.shape) == 4  # batch, ch=1, w, h

        # 1. Regularizations
        reg_av: torch.Tensor = torch.zeros(1, device=imgs_in.device, dtype=imgs_in.dtype)  # shape: 1
        for f in reg._fields:
            reg_av += getattr(reg, f)
        reg_av = reg_av.mean()  # shape []

        # 2. Sparsity should encourage:
        # 1. small probabilities
        # 2. tight bounding boxes
        # 3. tight masks
        # Old solution: sparsity = p * area_box -> leads to square masks b/c they have no cost and lead to lower kl_mask
        # New solution: sparsity = \sum_{i,j} (p * area_box) + \sum_k (p_chosen * area_mask_chosen)
        #                        = fg_fraction_box + fg_fraction_box
        fg_fraction_av = torch.sum(mixing_fg)/torch.numel(mixing_fg)  # equivalent to torch.mean
        fg_fraction_box = torch.sum(p_times_area_map)/torch.numel(mixing_fg)  # division by the same as above
        sparsity_av = fg_fraction_box + fg_fraction_av
        #print("fg_fraction_av vs fg_fraction_box", fg_fraction_av, fg_fraction_box)

        # 3. Observation model
        # if the observation_std is fixed then normalization 1.0/sqrt(2*pi*sigma^2) is irrelevant.
        # We are better off using MeanSquareError metric
        # Note that nll_off_bg is detached when appears with the minus sign. 
        # This is b/c we want the best possible background on top of which to add FOREGROUD objects
        nll_k = self.NLL_MSE(output=inference.big_img, target=imgs_in, sigma=self.sigma_fg)
        nll_bg = self.NLL_MSE(output=inference.big_bg, target=imgs_in, sigma=self.sigma_bg)  # batch_size, ch, w, h
        nll_av = (torch.sum(mixing_k * nll_k, dim=-5) + mixing_bg * nll_bg).mean()

        # 4. compute the KL for each image
        # TODO: NORMALIZE EVERYTHING BY THEIR RUNNING AVERAGE?
        kl_zinstance_av = torch.mean(inference.kl_zinstance_each_obj)  # mean over: boxes, batch_size, latent_zwhat
        kl_zwhere_av = torch.mean(inference.kl_zwhere_map)  # imean over: batch_size, ch=4, w, h
        kl_logit_tot = torch.sum(inference.kl_logit_map)  # will be normalized by its moving average
        
        
        # 5. compute the moving averages
        with torch.no_grad():

            # Compute the moving averages to normalize kl_logit
            input_dict = {"kl_logit_tot": kl_logit_tot.item()}
            # Only if in training mode I accumulate the moving average
            if self.training:
                ma_dict = self.ma_calculator.accumulate(input_dict)
            else:
                ma_dict = input_dict

        # 6. Loss_VAE
        kl_av = kl_zinstance_av + kl_zwhere_av + kl_logit_tot/ma_dict["kl_logit_tot"]
        assert nll_av.shape == reg_av.shape == kl_av.shape == sparsity_av.shape
        # print(nll_av, reg_av, kl_av, sparsity_av)

        # Note that I clamp in_place
        with torch.no_grad():
            f_balance = self.geco_balance_factor.data.clamp_(min=min(self.geco_dict["factor_balance_range"]),
                                                             max=max(self.geco_dict["factor_balance_range"]))
            f_sparsity = self.geco_sparsity_factor.data.clamp_(min=min(self.geco_dict["factor_sparsity_range"]),
                                                               max=max(self.geco_dict["factor_sparsity_range"]))
        loss_vae = f_sparsity * sparsity_av + \
                   f_balance * (nll_av + reg_av) + (1.0-f_balance) * kl_av


        # GECO BUSINESS
        if self.geco_dict["is_active"]:
            with torch.no_grad():
                # If fg_fraction_av > max(target) -> tmp1 > 0 -> delta_1 < 0 -> too much fg -> increase sparsity
                # If fg_fraction_av < min(target) -> tmp2 > 0 -> delta_1 > 0 -> too little fg -> decrease sparsity
                tmp1 = max(0, fg_fraction_av - max(self.geco_dict["target_fg_fraction"]))
                tmp2 = max(0, min(self.geco_dict["target_fg_fraction"]) - fg_fraction_av)
                delta_1 = torch.tensor(tmp2 - tmp1, dtype=loss_vae.dtype, device=loss_vae.device, requires_grad=False)

                # If nll_av > max(target) -> tmp3 > 0 -> delta_2 < 0 -> bad reconstruction -> increase f_balance
                # If nll_av < min(target) -> tmp4 > 0 -> delta_2 > 0 -> too good reconstruction -> decrease f_balance
                tmp3 = max(0, nll_av - max(self.geco_dict["target_nll"]))
                tmp4 = max(0, min(self.geco_dict["target_nll"]) - nll_av)
                delta_2 = torch.tensor(tmp4 - tmp3, dtype=loss_vae.dtype, device=loss_vae.device, requires_grad=False)

            #print("delta_1, f_sparsity", delta_1, f_sparsity, self.geco_sparsity_factor.data.item())
            #print("delta_2, f_balance", delta_2, f_balance, self.geco_balance_factor.data.item())

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
                               fg_fraction=fg_fraction_av.detach(),
                               geco_sparsity=f_sparsity,
                               geco_balance=f_balance,
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
                                                   noisy_sampling=False,
                                                   bg_is_zero=True,
                                                   bg_resolution=(1,1))

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

    def segment_with_tiling(self, img: torch.Tensor,
                            crop: tuple,
                            stride: tuple,
                            n_objects_max_per_patch: Optional[int] = None,
                            draw_bounding_box: bool = False):

        # Initialization
        batch_size, ch, w_raw, h_raw = img.shape
        n_obj_tot = torch.zeros(batch_size, device=img.device, dtype=img.dtype)  # add singleton for ch, w, h
        if n_objects_max_per_patch is None:
            n_objects_max_per_patch = self.input_img_dict["n_objects_max"]

        # solve overlap = (crop - stride) / 2
        # Recompute stride b/c overlap is rounded
        crop_w, crop_h = crop
        stride_w, stride_h = stride
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
                                               score_threshold=score_threshold,
                                               randomize_nms_factor=randomize_nms_factor,
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
            imgs_rec = self.draw_img(inference=results,
                                     draw_bounding_box=draw_bounding_box) if draw_image else torch.zeros_like(imgs_in)

        return Output(metrics=all_metrics, inference=results, imgs=imgs_rec)

    def forward(self,
                imgs_in: torch.tensor,
                draw_image: bool = False,
                draw_bounding_box: bool = False,
                verbose: bool = False):

        noisy_sampling: bool = True if self.training else False
        prob_corr_factor = getattr(self, "prob_corr_factor", 0.0)
        bg_is_zero = getattr(self, "bg_is_zero", True)
        bg_resolution = getattr(self, "bg_resolution", (2, 2))

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
                                       n_objects_max=self.input_img_dict["n_objects_max"],
                                       bg_is_zero=bg_is_zero,
                                       bg_resolution=bg_resolution)

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
                                           n_objects_max=self.input_img_dict["n_objects_max"],
                                           bg_is_zero=True,
                                           bg_resolution=(1, 1))

