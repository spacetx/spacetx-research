from .utilities import *
from .vae_parts import *
from .namedtuple import *
from typing import Union
import time


def pretty_print_metrics(epoch: int,
                         metric: tuple,
                         is_train: bool = True) -> str:
    if is_train:
        s = 'Train [epoch {0:4d}] loss={1[loss]:.3f}, mse={1[mse]:.3f}, reg={1[reg]:.3f}, kl_tot={1[kl_tot]:.3f}, sparsity={1[sparsity]:.3f}, fg_fraction={1[fg_fraction]:.3f}, geco_sp={1[geco_sparsity]:.3f}, geco_bal={1[geco_balance]:.3f}'.format(epoch, metric)
    else:
        s = 'Test  [epoch {0:4d}] loss={1[loss]:.3f}, mse={1[mse]:.3f}, reg={1[reg]:.3f}, kl_tot={1[kl_tot]:.3f}, sparsity={1[sparsity]:.3f}, fg_fraction={1[fg_fraction]:.3f}, geco_sp={1[geco_sparsity]:.3f}, geco_bal={1[geco_balance]:.3f}'.format(epoch, metric)
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
        self.sigma_fg = torch.tensor(params["GECO_loss"]["fg_std"])[..., None, None]  # add singleton for width, height
        self.sigma_bg = torch.tensor(params["GECO_loss"]["bg_std"])[..., None, None]  # add singleton for width, height

        self.geco_dict = params["GECO_loss"]
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
        minibatch_mean_cost_overlap = cost_overlap.mean()
        minibatch_mean_cost_volume_absolute_times_prob = cost_volume_absolute_times_prob.mean()
        minibatch_mean_total_cost = minibatch_mean_cost_overlap + minibatch_mean_cost_volume_absolute_times_prob
        return RegMiniBatch(cost_overlap=minibatch_mean_cost_overlap,
                            cost_vol_absolute=minibatch_mean_cost_volume_absolute_times_prob,
                            cost_total=minibatch_mean_total_cost)

    @staticmethod
    def NLL_MSE(output: torch.tensor, target: torch.tensor, sigma: torch.tensor) -> torch.Tensor:
        return ((output-target)/sigma).pow(2)

    def compute_metrics(self,
                        imgs_in: torch.Tensor,
                        inference: Inference,
                        reg_av: torch.Tensor) -> MetricMiniBatch:

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
        mse_k = CompositionalVae.NLL_MSE(output=inference.big_img, target=imgs_in, sigma=self.sigma_fg)
        mse_bg = CompositionalVae.NLL_MSE(output=inference.big_bg, target=imgs_in, sigma=self.sigma_bg)  # batch_size, ch, w, h
        mse_av = (torch.sum(mixing_k * mse_k, dim=-5) + mixing_bg * mse_bg).mean()  # mean over batch_size, ch, w, h

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
            one_minus_f_balance = torch.ones_like(f_balance) - f_balance

        # 6. Loss_VAE
        # TODO:
        # 1. try: loss_vae = f_balance * (nll_av + reg_av) + (1.0-f_balance) * (kl_av + f_sparsity * sparsity_av)
        # 2. move reg_av to the other size, i.e. proportional to 1-f_balance
        kl_av = kl_zinstance_av + kl_zwhere_av + kl_logit_tot / (1E-3 + ma_dict["kl_logit_tot"])
        sparsity_av = fg_fraction_mask_av + fg_fraction_box_av + prob_total_av
        assert mse_av.shape == reg_av.shape == kl_av.shape == sparsity_av.shape

        loss_vae = f_sparsity * sparsity_av + f_balance * (mse_av + reg_av) + one_minus_f_balance * kl_av

        # GECO BUSINESS
        if self.geco_dict["is_active"]:
            with torch.no_grad():
                # If fg_fraction_av > max(target) -> tmp1 > 0 -> delta_1 < 0 -> too much fg -> increase sparsity
                # If fg_fraction_av < min(target) -> tmp2 > 0 -> delta_1 > 0 -> too little fg -> decrease sparsity
                fg_fraction_av = torch.mean(mixing_fg)
                tmp1 = (fg_fraction_av - max(self.geco_dict["target_fg_fraction"])).clamp(min=0)
                tmp2 = (min(self.geco_dict["target_fg_fraction"]) - fg_fraction_av).clamp(min=0)
                delta_1 = (tmp2 - tmp1).requires_grad_(False).to(loss_vae.device)

                # If nll_av > max(target) -> tmp3 > 0 -> delta_2 < 0 -> bad reconstruction -> increase f_balance
                # If nll_av < min(target) -> tmp4 > 0 -> delta_2 > 0 -> too good reconstruction -> decrease f_balance
                tmp3 = (mse_av - max(self.geco_dict["target_mse"])).clamp(min=0)
                tmp4 = (min(self.geco_dict["target_mse"]) - mse_av).clamp(min=0)
                delta_2 = (tmp4 - tmp3).requires_grad_(False).to(loss_vae.device)

            loss_1 = self.geco_sparsity_factor * delta_1
            loss_2 = self.geco_balance_factor * delta_2
            loss_av = loss_vae + loss_1 + loss_2 - (loss_1 + loss_2).detach()
        else:
            delta_1 = torch.tensor(0.0, dtype=loss_vae.dtype, device=loss_vae.device)
            delta_2 = torch.tensor(0.0, dtype=loss_vae.dtype, device=loss_vae.device)
            loss_av = loss_vae

        # add everything you want as long as there is one loss
        return MetricMiniBatch(loss=loss_av,
                               mse=mse_av.detach(),
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

    @staticmethod
    def compute_similarity(mixing_k: torch.tensor, 
                           batch_of_index: torch.tensor, 
                           radius_nn: int, 
                           min_threshold: float = 0.01) -> torch.sparse.FloatTensor:
        """ Compute the similarity between two pixels by computing the product of mixing_k
            describing the probability that each pixel belong to a given foreground instance
            If the similarity is less that the similarity_threshold the value is not recorded (i.e. effectivdely zero)
            but saves memory.

            INPUT: mixing_k of shape --> n_boxes, batch_shape, 1, w, h
            OUTPUT: similarity of shape ------>   batch_shape, nnz, 2 
        """
        with torch.no_grad():
            
            n_boxes, batch_shape, ch_in, w, h = mixing_k.shape
            assert ch_in == 1
            assert (batch_shape, 1, w, h) == batch_of_index.shape

            # Pad width and height with zero before rolling to avoid spurious connections due to PBC
            pad = radius_nn + 1
            pad_mixing_k = F.pad(mixing_k, pad=[pad, pad, pad, pad], mode="constant", value=0.0)
            pad_index = F.pad(batch_of_index, pad=[pad, pad, pad, pad], mode="constant", value=-1)
        
            print("HERE")

            for pad_mixing_k_shifted, pad_index_shifted in roller_2d(a=pad_mixing_k, 
                                                                     b=pad_index, 
                                                                     radius=radius_nn):
                v = (pad_mixing_k * pad_mixing_k_shifted).sum(dim=-5)[:, 0, pad:(pad + w), pad:(pad + h)]  # shape: batch, w, h
                i = batch_of_index[:, 0] # shape: batch, w, h
                j = pad_index_shifted[:, 0, pad:(pad + w), pad:(pad + h)] # shape: batch, w, h
                
                mask = (v > min_value) * (i >= 0) * (j >= 0)
                
                print(v[mask].shape)
                print(i[mask].shape)
                print(j[mask].shape)
                
                assert 1==2

      #  # Prepare storage
      #  
      #  
      #  ch_out = int(((2 * radius_nn + 1) ** 2 - 1)/2)  # this is the number of point NN points
      #  ch_to_dxdy = []
      #  similarity = torch.zeros((batch_shape, ch_out, w, h), device=mixing_k.device, dtype=mixing_k.dtype)
#
      #  # compute dot product of the mixing_k between nearby vertices
      #  for ch, rolled in enumerate(roller_2d(pad_mixing_k, radius=radius_nn)):
      #      pad_mixing_k_shifted, dx, dy = rolled
      #      ch_to_dxdy.append([dx, dy])
      #      similarity[:, ch] = (pad_mixing_k *
      #                           pad_mixing_k_shifted).sum(dim=-5)[:, 0, pad:(pad + w), pad:(pad + h)]
#
      #  return DenseSimilarity(data=similarity, ch_to_dxdy=ch_to_dxdy)

    def segment(self, batch_imgs: torch.tensor,
                n_objects_max: Optional[int] = None,
                prob_corr_factor: Optional[float] = None,
                overlap_threshold: Optional[float] = None,
                noisy_sampling: bool = True,
                draw_boxes: bool = False,
                batch_of_index: Optional[torch.tensor] = None,
                radius_nn: int = 5) -> Segmentation:
        """ Segment the batch of images """

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

            mixing_k = inference.big_mask * inference.prob[..., None, None, None]

            # Now compute fg_prob, integer_segmentation_mask, similarity
            most_likely_mixing, index = torch.max(mixing_k, dim=-5, keepdim=True)  # 1, batch_size, 1, w, h
            integer_mask = ((most_likely_mixing > 0.5) * (index + 1)).squeeze(-5).to(dtype=torch.int32)  # bg = 0 fg = 1,2,3,...

            fg_prob = torch.sum(mixing_k, dim=-5)  # sum over instances

            bounding_boxes = draw_bounding_boxes(prob=inference.prob,
                                                 bounding_box=inference.bounding_box,
                                                 width=integer_mask.shape[-2],
                                                 height=integer_mask.shape[-1]) if draw_boxes else None

            if batch_of_index is None:
                print("not computing similarity")
                return Segmentation(raw_image=batch_imgs,
                                    fg_prob=fg_prob,
                                    integer_mask=integer_mask,
                                    bounding_boxes=bounding_boxes,
                                    similarity=None)

            else:
                print("computing similarity")
                return Segmentation(raw_image=batch_imgs,
                                    fg_prob=fg_prob,
                                    integer_mask=integer_mask,
                                    bounding_boxes=bounding_boxes,
                                    similarity=CompositionalVae.compute_similarity(mixing_k=mixing_k,
                                                                                   batch_of_index=batch_of_index,
                                                                                   radius_nn=radius_nn,
                                                                                   min_threshold=0.1))
                                                                     
    def segment_with_tiling(self,
                            single_img: torch.Tensor,
                            roi_mask: Optional[torch.Tensor],
                            crop_size: Optional[tuple] = None,
                            stride: Optional[tuple] = None,
                            n_objects_max_per_patch: Optional[int] = None,
                            prob_corr_factor: Optional[float] = None,
                            overlap_threshold: Optional[float] = None,
                            radius_nn: int = 5,
                            batch_size: int = 32) -> (Segmentation, SparseSimilarity):
        """ Uses a sliding window approach to collect a co_objectiveness information
            about the pixels of a large image """
        if roi_mask is not None:
            cum_roi_mask = torch.cumsum(torch.cumsum(roi_mask, dim=-1), dim=-2)
            assert len(cum_roi_mask.shape) == len(single_img.shape)
            assert cum_roi_mask.shape[-2:] == single_img.shape[-2:]
            print("cum_roi_mask.shape ->", cum_roi_mask.shape)
        else:
            cum_roi_mask = None

        crop_size = (self.input_img_dict["size_raw_image"], self.input_img_dict["size_raw_image"]) if crop_size is None else crop_size
        stride = (int(crop_size[0]//4), int(crop_size[1]//4)) if stride is None else stride
        n_objects_max_per_patch = self.input_img_dict["n_objects_max"] if n_objects_max_per_patch is \
                                                                          None else n_objects_max_per_patch
        prob_corr_factor = getattr(self, "prob_corr_factor", 0.0) if prob_corr_factor is None else prob_corr_factor
        overlap_threshold = self.nms_dict["overlap_threshold"] if overlap_threshold is None else overlap_threshold

        assert crop_size[0] % stride[0] == 0, "crop and stride size are NOT compatible"
        assert crop_size[1] % stride[1] == 0, "crop and stride size are NOT compatible"
        assert len(single_img.shape) == 3  # ch, w, h

        with torch.no_grad():

            w_img, h_img = single_img.shape[-2:]
            n_prediction = (crop_size[0]//stride[0]) * (crop_size[1]//stride[1])
            print(f'Each pixel will be segmented {n_prediction} times')

            pad_w = crop_size[0] - stride[0]
            pad_h = crop_size[1] - stride[1]

            # Build a list with the locations of the corner of the images
            location_of_corner = []
            for i in range(-pad_w, w_img, stride[0]):
                for j in range(-pad_h, h_img, stride[1]):
                    location_of_corner.append([i, j])

            # Identify the starting points
            ij_tmp = torch.tensor(location_of_corner, device=single_img.device, dtype=torch.long)  # shape: N, 2
            x3 = (ij_tmp[..., 0] + crop_size[0]).clamp(max=w_img)
            y3 = (ij_tmp[..., 1] + crop_size[1]).clamp(max=h_img)
            x1 = ij_tmp[..., 0]
            y1 = ij_tmp[..., 1]
            x1_gt_0 = (x1 > 0)
            y1_gt_0 = (y1 > 0)
            x1.clamp_(min=0)
            y1.clamp_(min=0)
            dx = x3 - x1
            dy = y3 - y1
            start_x = (crop_size[0] - dx) * (x1 == 0)
            start_y = (crop_size[0] - dy) * (y1 == 0)

            if cum_roi_mask is not None:
                # Exclude stuff if outside the roi_mask
                integral = cum_roi_mask[0, x3-1, y3-1] - \
                           cum_roi_mask[0, x1-1, y3-1] * x1_gt_0 - \
                           cum_roi_mask[0, x3-1, y1-1] * y1_gt_0 + \
                           cum_roi_mask[0, x1-1, y1-1] * x1_gt_0 * y1_gt_0
                fraction = integral.float() / (crop_size[0] * crop_size[1])
                filter = fraction > 0.01  # if there is more than 1% ROI the patch will be processed.

                dx = dx[filter]
                dy = dy[filter]
                start_x = start_x[filter]
                start_y = start_y[filter]
                end_x = start_x + dx
                end_y = start_y + dy
                x1 = x1[filter]
                x3 = x3[filter]
                y1 = y1[filter]
                y3 = y3[filter]

            print(f'I am going to process {start_x.shape[0]} patches')
            if not (start_x.shape[0] >= 1):
                raise Exception("No patches will be anlyzed. Something went wrong!")

            # Prepare storage
            n_instances_tot = 0
            big_integer_mask, big_fg_prob = None, None
            index_max = w_img * h_img

            # TODO: use roi mask to make a smaller single_index_matrix......
            single_index_matrix = torch.arange(index_max,
                                               dtype=torch.long,
                                               device=single_img.device).view(w_img, h_img)



            big_csr_similarity = scipy.sparse.csr_matrix((index_max, index_max), dtype=float)

            # split the list in chunks of batch_size
            list_k_start = range(0, start_x.shape[0], batch_size)
            for n, k_start in enumerate(list_k_start):
                k_end = min(k_start + batch_size, start_x.shape[0])

                # Prepare batch of images
                batch_imgs = torch.empty((k_end-k_start, single_img.shape[-3], crop_size[0], crop_size[1]),
                                         device=single_img.device,
                                         dtype=single_img.dtype)
                batch_of_index = torch.empty((k_end-k_start, 1, crop_size[0], crop_size[1]),
                                             device=single_index_matrix.device,
                                             dtype=single_index_matrix.dtype)
                print(k_start, k_end)
                for k in range(k_start, k_end):
                    print(k)
                    batch_imgs[k, :, start_x[k]:end_x[k],
                                     start_y[k]:end_y[k]] = single_img[:, x1[k]:x3[k], y1[k]:y3[k]]

                    batch_of_index[k, 0, start_x[k]:end_x[k],
                                         start_y[k]:end_y[k]] = single_index_matrix[x1[k]:x3[k], y1[k]:y3[k]]

                # print progress
                if (n % 100 == 0) or (n == len(list_k_start)-1):
                    print(f'{n} out of {len(list_k_start)-1} -> batch_of_imgs.shape = {batch_imgs.shape}')

                # Segment the batch (and move to CPU or GPU depending where the model lives)
                segmentation = self.segment(batch_imgs.to(self.sigma_fg.device),
                                            n_objects_max=n_objects_max_per_patch,
                                            prob_corr_factor=prob_corr_factor,
                                            overlap_threshold=overlap_threshold,
                                            noisy_sampling=True,
                                            draw_boxes=False,
                                            batch_of_index=batch_of_index,
                                            radius_nn=radius_nn)
                print(segmentation._fields)
                assert 1==2

#### big_csr_similarity += segmentation.similarity.to_sparse_similarity(index_matrix=batch_of_index,
####                                                                                   index_max=index_max).csr_matrix
####
####                if big_fg_prob is None:
####                    # Probability and integer mask are dense tensor
####                    big_fg_prob = torch.zeros((w_img, h_img),
####                                              device=segmentation.fg_prob.device,
####                                              dtype=segmentation.fg_prob.dtype)
####                    big_integer_mask = torch.zeros((w_img, h_img),
####                                                   device=segmentation.integer_mask.device,
####                                                   dtype=segmentation.integer_mask.dtype)
####
####                for k in range(len(ij_list)):
####
####                    big_fg_prob[x1[k]:x3[k],
####                                y1[k]:y3[k]] += segmentation.fg_prob[k, 0,
####                                                                     start_x[k]:start_x[k]+dx[k],
####                                                                     start_y[k]:start_y[k]+dy[k]]
####
####                    # Find a set of not-overlapping tiles to obtain a sample segmentation (without graph clustering)
####                    if (x1[k] % crop_size[0] == 0) and (y1[k] % crop_size[1] == 0):
####                        n_instances = torch.max(segmentation.integer_mask[k])
####                        shifted_integer_mask = (segmentation.integer_mask[k] > 0) * \
####                                               (segmentation.integer_mask[k] + n_instances_tot)
####                        n_instances_tot += n_instances
####                        big_integer_mask[x1[k]:x3[k],
####                                         y1[k]:y3[k]] = shifted_integer_mask[0,
####                                                                             start_x[k]:start_x[k]+dx[k],
####                                                                             start_y[k]:start_y[k]+dy[k]]
####            # Normalize the similarity
####            big_csr_similarity /= n_prediction
####
####            # Remove possible missing values in the big_integer_mask
####            original_type = big_integer_mask.dtype
####            bin_count = torch.bincount(big_integer_mask.flatten())
####            old2new_label = torch.cumsum(bin_count > 0, dim=0) - 1  # conversion old -> new
####            integer_mask_no_gaps = old2new_label[big_integer_mask.long()].to(original_type)
#####
####            return Segmentation(raw_image=single_img[None],
####                                fg_prob=big_fg_prob[None, None],
####                                integer_mask=integer_mask_no_gaps[None, None],
####                                bounding_boxes=None,
####                                similarity=SparseSimilarity(csr_matrix=big_csr_similarity,
####                                                            index_matrix=single_index_matrix))
####

    # this is the generic function which has all the options unspecified
    def process_batch_imgs(self,
                           imgs_in: torch.tensor,
                           generate_synthetic_data: bool,
                           topk_only: bool,
                           draw_image: bool,
                           draw_bg: bool,
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
                                       reg_av=regularizations.cost_total)

        all_metrics = Metric_and_Reg.from_merge(metrics=metrics, regularizations=regularizations)

        with torch.no_grad():
            if draw_image:
                imgs_rec = draw_img(prob=results.prob,
                                    bounding_box=results.bounding_box,
                                    big_mask=results.big_mask,
                                    big_img=results.big_img,
                                    big_bg=results.big_bg,
                                    draw_bg=draw_bg,
                                    draw_boxes=draw_boxes)
            else:
                imgs_rec = torch.zeros_like(imgs_in)

        return Output(metrics=all_metrics, inference=results, imgs=imgs_rec)

    def forward(self,
                imgs_in: torch.tensor,
                draw_image: bool = False,
                draw_bg: bool = False,
                draw_boxes: bool = False,
                verbose: bool = False):

        return self.process_batch_imgs(imgs_in=imgs_in,
                                       generate_synthetic_data=False,
                                       topk_only=False,
                                       draw_image=draw_image,
                                       draw_bg=draw_bg,
                                       draw_boxes=draw_boxes,
                                       verbose=verbose,
                                       noisy_sampling=True,  # True if self.training else False,
                                       prob_corr_factor=getattr(self, "prob_corr_factor", 0.0),
                                       overlap_threshold=self.nms_dict.get("overlap_threshold", 0.3),
                                       n_objects_max=self.input_img_dict["n_objects_max"],
                                       bg_is_zero=self.input_img_dict.get("bg_is_zero", True),
                                       bg_resolution=self.input_img_dict.get("background_resolution_before_upsampling",
                                                                             (2, 2)))

    def generate(self,
                 imgs_in: torch.tensor,
                 draw_bg: bool = False,
                 draw_boxes: bool = False,
                 verbose: bool = False):

        with torch.no_grad():
            
            return self.process_batch_imgs(imgs_in=torch.zeros_like(imgs_in),
                                           generate_synthetic_data=True,
                                           topk_only=False,
                                           draw_image=True,
                                           draw_bg=draw_bg,
                                           draw_boxes=draw_boxes,
                                           verbose=verbose,
                                           noisy_sampling=True,
                                           prob_corr_factor=0.0,
                                           overlap_threshold=self.nms_dict.get("overlap_threshold", 0.3),
                                           n_objects_max=self.input_img_dict["n_objects_max"],
                                           bg_is_zero=True,
                                           bg_resolution=(2, 2))
