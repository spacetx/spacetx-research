from MODULES.utilities import *
from MODULES.utilities_ml import Moving_Average_Calculator
from MODULES.utilities_visualization import draw_bounding_boxes, draw_img
from MODULES.vae_parts import *
from MODULES.namedtuple import *
from typing import Optional


def create_ckpt(model: torch.nn.Module,
                optimizer: Optional[torch.optim.Optimizer] = None,
                history_dict: Optional[dict] = None,
                hyperparams_dict: Optional[dict] = None,
                epoch: Optional[int] = None) -> dict:\

    all_member_var = model.__dict__
    member_var_to_save = {}
    for k, v in all_member_var.items():
        if not k.startswith("_") and k != 'training':
            member_var_to_save[k] = v

    ckpt = {'model_member_var': member_var_to_save,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None if (optimizer is None) else optimizer.state_dict(),
            'history_dict': history_dict,
            'hyperparam_dict': hyperparams_dict,
            'epoch': epoch}

    return ckpt


def ckpt2file(ckpt: dict, path: str):
    torch.save(ckpt, path)


def file2ckpt(path: str, device: Optional[str] = None):
    """ wrapper around torch.load """
    if device is None:
        ckpt = torch.load(path)
    elif device == 'cuda':
        ckpt = torch.load(path, map_location="cuda:0")
    elif device == 'cpu':
        ckpt = torch.load(path, map_location=torch.device('cpu'))
    else:
        raise Exception("device is not recognized")
    return ckpt


def load_from_ckpt(ckpt,
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
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
    geco_params, similarity_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if name.startswith("geco"):
            geco_params.append(param)
        elif name.startswith("similarity"):
            similarity_params.append(param)
        else:
            other_params.append(param)

    if dict_params_optimizer["type"] == "adam":
        optimizer = torch.optim.Adam([{'params': geco_params, 'lr': dict_params_optimizer["base_lr_geco"],
                                       'betas': dict_params_optimizer["betas_geco"]},
                                      {'params': similarity_params, 'lr': dict_params_optimizer["base_lr_similarity"],
                                       'betas': dict_params_optimizer["betas_similarity"]},
                                      {'params': other_params, 'lr': dict_params_optimizer["base_lr"],
                                       'betas': dict_params_optimizer["betas"]}],
                                     eps=dict_params_optimizer["eps"],
                                     weight_decay=dict_params_optimizer["weight_decay"])
        
    elif dict_params_optimizer["type"] == "SGD":
        optimizer = torch.optim.SGD([{'params': geco_params, 'lr': dict_params_optimizer["base_lr_geco"]},
                                     {'params': similarity_params, 'lr': dict_params_optimizer["base_lr_similarity"]},
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
        # self.ma_calculator = Moving_Average_Calculator(beta=0.999)  # i.e. average over the last 100 mini-batches

        # Raw image parameters
        self.dict_soft_constraints = params["soft_constraint"]
        self.nms_dict = params["nms"]
        self.sigma_fg = torch.nn.Parameter(data=torch.tensor(params["GECO_loss"]["fg_std"],
                                                             dtype=torch.float)[..., None, None],  # singleton w,h
                                           requires_grad=False)
        self.sigma_bg = torch.nn.Parameter(data=torch.tensor(params["GECO_loss"]["bg_std"],
                                                             dtype=torch.float)[..., None, None],  # singleton w,h
                                           requires_grad=False)

        self.geco_dict = params["GECO_loss"]
        self.input_img_dict = params["input_image"]

        self.geco_sparsity_factor = torch.nn.Parameter(data=torch.tensor(self.geco_dict["factor_sparsity_range"][1],
                                                                         dtype=torch.float), requires_grad=True)
        self.geco_balance_factor = torch.nn.Parameter(data=torch.tensor(self.geco_dict["factor_balance_range"][1],
                                                                        dtype=torch.float), requires_grad=True)

        # Put everything on the cude if cuda available
        if torch.cuda.is_available():
            self.cuda()

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
        # print(inference.big_mask_NON_interacting.shape)  # n_box_few, batch, 1, w_big, h_big
        # print(inference.logit_few.shape)                 # n_box_few, batch
        # print(inference.sample_c.shape)                  # n_box_few, batch

        x = inference.sample_c[..., None, None, None] * inference.big_mask_NON_interacting
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
        cost_volume_minibatch = (cost_volume_absolute * inference.sample_c).sum(dim=-2).mean()  # sum boxes, mean batch_size
        return RegMiniBatch(reg_overlap=cost_overlap.mean(),
                            reg_area_obj=cost_volume_minibatch)

    @staticmethod
    def NLL_MSE(output: torch.tensor, target: torch.tensor, sigma: torch.tensor) -> torch.Tensor:
        return ((output - target) / sigma).pow(2)

    def compute_metrics(self,
                        imgs_in: torch.Tensor,
                        inference: Inference,
                        regularizations: RegMiniBatch) -> MetricMiniBatch:

        # Preparation
        batch_size, ch, w, h = imgs_in.shape
        # p_times_area_map = inference.p_map * inference.area_map
        mixing_k = inference.big_mask * inference.sample_c[..., None, None, None]
        mixing_fg = torch.sum(mixing_k, dim=-5)  # sum over boxes
        mixing_bg = torch.ones_like(mixing_fg) - mixing_fg
        fg_fraction = torch.mean(mixing_fg)  # mean over batch_size
        assert len(mixing_fg.shape) == 4  # batch, ch=1, w, h

        # 0. Get the regularization

        # 1. Observation model
        # if the observation_std is fixed then normalization 1.0/sqrt(2*pi*sigma^2) is irrelevant.
        # We are better off using MeanSquareError metric
        mse_k = CompositionalVae.NLL_MSE(output=inference.big_img, target=imgs_in, sigma=self.sigma_fg)
        mse_bg = CompositionalVae.NLL_MSE(output=inference.big_bg, target=imgs_in,
                                          sigma=self.sigma_bg)  # batch_size, ch, w, h
        mse_av = (torch.sum(mixing_k * mse_k, dim=-5) + mixing_bg * mse_bg).mean()  # mean over batch_size, ch, w, h

        # 2. Sparsity should encourage:
        # 1. small probabilities
        # 2. tight bounding boxes
        # 3. tight masks
        # The three terms take care of all these requirement.
        n_boxes = inference.sample_c.shape[-2]
        area_boxes = inference.sample_bb.bw * inference.sample_bb.bh
        sparsity_mask = torch.sum(mixing_fg) / torch.numel(mixing_fg)  # strictly smaller than 1
        sparsity_box = torch.sum(area_boxes * inference.sample_c) / torch.numel(mixing_fg)
        sparsity_prob = torch.sum(inference.sample_c_map) / (batch_size * n_boxes)
        sparsity_av = sparsity_mask + sparsity_box + sparsity_prob

        # 3. compute KL
        # TODO: Can add KL background and make sure that all KL have the same order by adding self-balancing hyperparameters.
        kl_zinstance = torch.sum(inference.kl_zinstance) / batch_size
        kl_zwhere = torch.sum(inference.kl_zwhere) / batch_size
        kl_logit = torch.sum(inference.kl_logit) / batch_size
        kl_av = kl_zinstance + kl_zwhere + kl_logit

        # 6. Note that I clamp in_place
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
        reg_av = regularizations.total()
        loss_vae = f_sparsity * sparsity_av + \
                   f_balance * (mse_av + reg_av) + \
                   one_minus_f_balance * kl_av

        # GECO BUSINESS
        if self.geco_dict["is_active"]:
            with torch.no_grad():
                # If fg_fraction_av > max(target) -> tmp1 > 0 -> delta_1 < 0 -> too much fg -> increase sparsity
                # If fg_fraction_av < min(target) -> tmp2 > 0 -> delta_1 > 0 -> too little fg -> decrease sparsity
                tmp1 = (fg_fraction - max(self.geco_dict["target_fg_fraction"])).clamp(min=0)
                tmp2 = (min(self.geco_dict["target_fg_fraction"]) - fg_fraction).clamp(min=0)
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
                               mse_tot=mse_av.detach().item(),
                               reg_tot=reg_av.detach().item(),
                               kl_tot=kl_av.detach().item(),
                               sparsity_tot=sparsity_av.detach().item(),

                               kl_instance=kl_zinstance.detach().item(),
                               kl_where=kl_zwhere.detach().item(),
                               kl_logit=kl_logit.detach().item(),
                               sparsity_mask=sparsity_mask.detach().item(),
                               sparsity_box=sparsity_box.detach().item(),
                               sparsity_prob=sparsity_prob.detach().item(),
                               reg_overlap=regularizations.reg_overlap.detach().item(),
                               reg_area_obj=regularizations.reg_area_obj.detach().item(),

                               fg_fraction=fg_fraction.detach().item(),
                               geco_sparsity=f_sparsity.detach().item(),
                               geco_balance=f_balance.detach().item(),
                               delta_1=delta_1.detach().item(),
                               delta_2=delta_2.detach().item(),

                               similarity_sigma2=inference.similarity_sigma2.detach().cpu().numpy(),
                               similarity_weights=inference.similarity_weights.detach().cpu().numpy())

    @staticmethod
    def compute_sparse_similarity_matrix(mixing_k: torch.tensor,
                                         batch_of_index: torch.tensor,
                                         max_index: int,
                                         radius_nn: int,
                                         min_threshold: float = 0.01) -> torch.sparse.FloatTensor:
        """ Compute the similarity between two pixels by computing the product of mixing_k
            describing the probability that each pixel belong to a given foreground instance
            If the similarity is less than min_threshold the value is not recorded (i.e. effectively zero)
            to save memory.

            INPUT: mixing_k of shape --> n_boxes, batch_shape, 1, w, h
            OUTPUT: sparse tensor fo size (max_index, max_index)
        """
        with torch.no_grad():
            # start_time = time.time()
            n_boxes, batch_shape, ch_in, w, h = mixing_k.shape
            assert ch_in == 1
            assert (batch_shape, 1, w, h) == batch_of_index.shape

            # Pad width and height with zero before rolling to avoid spurious connections due to PBC
            pad = radius_nn + 1
            pad_mixing_k = F.pad(mixing_k, pad=[pad, pad, pad, pad], mode="constant", value=0.0)
            pad_index = F.pad(batch_of_index, pad=[pad, pad, pad, pad], mode="constant", value=-1)
            row = batch_of_index[:, 0]  # shape: batch, w, h
            row_ge_0 = (row >= 0)

            sparse_similarity = torch.sparse.FloatTensor(max_index, max_index).to(mixing_k.device)
            for pad_mixing_k_shifted, pad_index_shifted in roller_2d(a=pad_mixing_k,
                                                                     b=pad_index,
                                                                     radius=radius_nn):
                v = (pad_mixing_k * pad_mixing_k_shifted).sum(dim=-5)[:, 0, pad:(pad + w),
                    pad:(pad + h)]  # shape: batch, w, h
                col = pad_index_shifted[:, 0, pad:(pad + w), pad:(pad + h)]  # shape: batch, w, h

                mask = (v > min_threshold) * (col >= 0) * row_ge_0

                index_tensor = torch.stack((row[mask], col[mask]), dim=0)
                tmp_similarity = torch.sparse.FloatTensor(index_tensor, v[mask],
                                                          torch.Size([max_index, max_index]))
                sparse_similarity.add_(tmp_similarity)
                sparse_similarity = sparse_similarity.coalesce()

            # print("similarity time", time.time()-start_time)
            return sparse_similarity

    def segment(self, batch_imgs: torch.tensor,
                n_objects_max: Optional[int] = None,
                prob_corr_factor: Optional[float] = None,
                overlap_threshold: Optional[float] = None,
                noisy_sampling: bool = True,
                draw_boxes: bool = False,
                batch_of_index: Optional[torch.tensor] = None,
                max_index: Optional[int] = None,
                radius_nn: int = 5) -> Segmentation:
        """ Segment the batch of images """

        # start_time = time.time()
        n_objects_max = self.input_img_dict["n_objects_max"] if n_objects_max is None else n_objects_max
        prob_corr_factor = getattr(self, "prob_corr_factor", 0.0) if prob_corr_factor is None else prob_corr_factor
        overlap_threshold = self.nms_dict["overlap_threshold"] if overlap_threshold is None else overlap_threshold

        with torch.no_grad():
            inference: Inference = self.inference_and_generator(imgs_in=batch_imgs,
                                                                generate_synthetic_data=False,
                                                                prob_corr_factor=prob_corr_factor,
                                                                overlap_threshold=overlap_threshold,
                                                                n_objects_max=n_objects_max,
                                                                topk_only=False,
                                                                noisy_sampling=noisy_sampling,
                                                                bg_is_zero=True,
                                                                bg_resolution=(1, 1))

            mixing_k = inference.big_mask * inference.sample_c[..., None, None, None]

            # Now compute fg_prob, integer_segmentation_mask, similarity
            most_likely_mixing, index = torch.max(mixing_k, dim=-5, keepdim=True)  # 1, batch_size, 1, w, h
            integer_mask = ((most_likely_mixing > 0.5) * (index + 1)).squeeze(-5).to(
                dtype=torch.int32)  # bg = 0 fg = 1,2,3,...

            fg_prob = torch.sum(mixing_k, dim=-5)  # sum over instances

            bounding_boxes = draw_bounding_boxes(c=inference.sample_c,
                                                 bounding_box=inference.sample_bb,
                                                 width=integer_mask.shape[-2],
                                                 height=integer_mask.shape[-1]) if draw_boxes else None

            # print("inference time", time.time()-start_time)

            if batch_of_index is None:
                return Segmentation(raw_image=batch_imgs,
                                    fg_prob=fg_prob,
                                    integer_mask=integer_mask,
                                    bounding_boxes=bounding_boxes,
                                    similarity=None)

            else:
                max_index = torch.max(batch_of_index) if max_index is None else max_index
                similarity_matrix = CompositionalVae.compute_sparse_similarity_matrix(mixing_k=mixing_k,
                                                                                      batch_of_index=batch_of_index,
                                                                                      max_index=max_index,
                                                                                      radius_nn=radius_nn,
                                                                                      min_threshold=0.1)
                return Segmentation(raw_image=batch_imgs,
                                    fg_prob=fg_prob,
                                    integer_mask=integer_mask,
                                    bounding_boxes=bounding_boxes,
                                    similarity=SparseSimilarity(sparse_matrix=similarity_matrix,
                                                                index_matrix=None))

    def segment_with_tiling(self,
                            single_img: torch.Tensor,
                            roi_mask: Optional[torch.Tensor],
                            crop_size: Optional[tuple] = None,
                            stride: Optional[tuple] = None,
                            n_objects_max_per_patch: Optional[int] = None,
                            prob_corr_factor: Optional[float] = None,
                            overlap_threshold: Optional[float] = None,
                            radius_nn: int = 5,
                            batch_size: int = 32) -> Segmentation:
        """ Uses a sliding window approach to collect a co_objectiveness information
            about the pixels of a large image.

            On CPU, pad the image with zeros (this lead to duplication of the data).
            Select the slices and then copy to GPU
        """
        assert len(single_img.shape) == 3
        assert roi_mask is None or len(roi_mask.shape) == 3

        crop_size = (self.input_img_dict["size_raw_image"],
                     self.input_img_dict["size_raw_image"]) if crop_size is None else crop_size
        stride = (int(crop_size[0] // 4), int(crop_size[1] // 4)) if stride is None else stride
        n_objects_max_per_patch = self.input_img_dict["n_objects_max"] if n_objects_max_per_patch is None \
            else n_objects_max_per_patch
        prob_corr_factor = getattr(self, "prob_corr_factor", 0.0) if prob_corr_factor is None else prob_corr_factor
        overlap_threshold = self.nms_dict["overlap_threshold"] if overlap_threshold is None else overlap_threshold

        assert crop_size[0] % stride[0] == 0, "crop and stride size are NOT compatible"
        assert crop_size[1] % stride[1] == 0, "crop and stride size are NOT compatible"
        assert len(single_img.shape) == 3  # ch, w, h

        with torch.no_grad():

            w_img, h_img = single_img.shape[-2:]
            n_prediction = (crop_size[0] // stride[0]) * (crop_size[1] // stride[1])
            print(f'Each pixel will be segmented {n_prediction} times')

            pad_w = crop_size[0] - stride[0]
            pad_h = crop_size[1] - stride[1]
            pad_list = [pad_w, crop_size[0], pad_h, crop_size[1]]

            # This is duplicating the single_img on the CPU
            # Note: unsqueeze, pad, suqeeze
            try:
                img_padded = F.pad(single_img.cpu().unsqueeze(0),
                                   pad=pad_list, mode='reflect')  # 1, ch_in, w_pad, h_pad
            except RuntimeError:
                img_padded = F.pad(single_img.cpu().unsqueeze(0),
                                   pad=pad_list, mode='constant', value=0)  # 1, ch_in, w_pad, h_pad
            w_paddded, h_padded = img_padded.shape[-2:]

            # This is creating the index matrix on the cpu
            max_index = w_img * h_img
            index_matrix_padded = F.pad(torch.arange(max_index,
                                                     dtype=torch.long,
                                                     device=torch.device('cpu')).view(1, 1, w_img, h_img),
                                        pad=pad_list, mode='constant', value=-1)  # 1, 1, w_pad, h_pad

            assert index_matrix_padded.shape[-2:] == img_padded.shape[-2:]
            assert index_matrix_padded.shape[0] == img_padded.shape[0]
            assert len(index_matrix_padded.shape) == len(img_padded.shape)

            # Build a list with the locations of the corner of the images
            location_of_corner = []
            for i in range(0, w_img + pad_w, stride[0]):
                for j in range(0, h_img + pad_h, stride[1]):
                    location_of_corner.append([i, j])

            ij_tmp = torch.tensor(location_of_corner, device=torch.device('cpu'), dtype=torch.long)  # shape: N, 2
            x1 = ij_tmp[..., 0]
            y1 = ij_tmp[..., 1]
            del ij_tmp

            if roi_mask is not None:
                assert roi_mask.shape[-2:] == single_img.shape[-2:]
                cum_roi_mask = F.pad(torch.cumsum(torch.cumsum(roi_mask, dim=-1), dim=-2),
                                     pad=pad_list, mode='constant', value=0)
                assert cum_roi_mask.shape[-2:] == img_padded.shape[-2:]

                # Exclude stuff if outside the roi_mask
                integral = cum_roi_mask[0, x1 + crop_size[0] - 1, y1 + crop_size[1] - 1] - \
                           cum_roi_mask[0, x1 - 1, y1 + crop_size[1] - 1] * (x1 > 0) - \
                           cum_roi_mask[0, x1 + crop_size[0] - 1, y1 - 1] * (y1 > 0) + \
                           cum_roi_mask[0, x1 - 1, y1 - 1] * (x1 > 0) * (y1 > 0)
                fraction = integral.float() / (crop_size[0] * crop_size[1])
                mask = fraction > 0.01  # if there is more than 1% ROI the patch will be processed.
                x1 = x1[mask]
                y1 = y1[mask]
                del cum_roi_mask
                del mask

            print(f'I am going to process {x1.shape[0]} patches')
            if not (x1.shape[0] >= 1):
                raise Exception("No patches will be analyzed. Something went wrong!")

            # split the list in chunks of batch_size
            index = torch.arange(0, x1.shape[0], dtype=torch.long, device=torch.device('cpu'))
            n_list_of_list = [index[n:n + batch_size] for n in range(0, index.shape[0], batch_size)]
            n_instances_tot = 0
            need_initialization = True
            for n_batches, n_list in enumerate(n_list_of_list):

                batch_imgs = torch.cat([img_padded[...,
                                          x1[n]:x1[n] + crop_size[0],
                                          y1[n]:y1[n] + crop_size[1]] for n in n_list], dim=-4)

                batch_index = torch.cat([index_matrix_padded[...,
                                           x1[n]:x1[n] + crop_size[0],
                                           y1[n]:y1[n] + crop_size[1]] for n in n_list], dim=-4)

                # print progress
                if (n_batches % 10 == 0) or (n_batches == len(n_list_of_list) - 1):
                    print(f'{n_batches} out of {len(n_list_of_list) - 1} -> batch_of_imgs.shape = {batch_imgs.shape}')

                segmentation = self.segment(batch_imgs=batch_imgs.to(self.sigma_fg.device),
                                            n_objects_max=n_objects_max_per_patch,
                                            prob_corr_factor=prob_corr_factor,
                                            overlap_threshold=overlap_threshold,
                                            noisy_sampling=True,
                                            draw_boxes=False,
                                            batch_of_index=batch_index.to(self.sigma_fg.device),
                                            max_index=max_index,
                                            radius_nn=radius_nn)
                # print("segmentation time", time.time()-start_time)

                # Initialize only the fist time
                if need_initialization:
                    # Probability and integer mask are dense tensor
                    big_fg_prob = torch.zeros((w_paddded, h_padded),
                                              device=torch.device('cpu'),
                                              dtype=segmentation.fg_prob.dtype)
                    big_integer_mask = torch.zeros((w_paddded, h_padded),
                                                   device=torch.device('cpu'),
                                                   dtype=segmentation.integer_mask.dtype)
                    # Similarity is a sparse tensor
                    sparse_similarity_matrix = torch.sparse.FloatTensor(max_index, max_index).cpu()
                    need_initialization = False

                # Unpack the data from batch
                sparse_similarity_matrix.add_(segmentation.similarity.sparse_matrix.cpu())
                sparse_similarity_matrix = sparse_similarity_matrix.coalesce()
                fg_prob = segmentation.fg_prob.cpu()
                integer_mask = segmentation.integer_mask.cpu()

                for k, n in enumerate(n_list):
                    big_fg_prob[x1[n]:x1[n] + crop_size[0], y1[n]:y1[n] + crop_size[1]] += fg_prob[k, 0]

                    # Find a set of not-overlapping tiles to obtain a sample segmentation (without graph clustering)
                    if ((x1[n] - pad_w) % crop_size[0] == 0) and ((y1[n] - pad_h) % crop_size[1] == 0):
                        n_instances = torch.max(integer_mask[k])
                        shifted_integer_mask = (integer_mask[k] > 0) * \
                                               (integer_mask[k] + n_instances_tot)
                        n_instances_tot += n_instances
                        big_integer_mask[x1[n]:x1[n] +
                                               crop_size[0], y1[n]:y1[n] + crop_size[1]] = shifted_integer_mask[0]

            # End of loop over batches
            sparse_similarity_matrix.div_(n_prediction)
            big_fg_prob.div_(n_prediction)

            return Segmentation(raw_image=single_img[None],
                                fg_prob=big_fg_prob[None, None, pad_w:pad_w + w_img, pad_h:pad_h + h_img],
                                integer_mask=big_integer_mask[None, None, pad_w:pad_w + w_img, pad_h:pad_h + h_img],
                                bounding_boxes=None,
                                similarity=SparseSimilarity(sparse_matrix=sparse_similarity_matrix,
                                                            index_matrix=index_matrix_padded[0, 0, pad_w:pad_w + w_img,
                                                                         pad_h:pad_h + h_img]))

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

        inference: Inference = self.inference_and_generator(imgs_in=imgs_in,
                                                            generate_synthetic_data=generate_synthetic_data,
                                                            prob_corr_factor=prob_corr_factor,
                                                            overlap_threshold=overlap_threshold,
                                                            n_objects_max=n_objects_max,
                                                            topk_only=topk_only,
                                                            noisy_sampling=noisy_sampling,
                                                            bg_is_zero=bg_is_zero,
                                                            bg_resolution=bg_resolution)

        regularizations: RegMiniBatch = self.compute_regularizations(inference=inference,
                                                                     verbose=verbose)

        metrics: MetricMiniBatch = self.compute_metrics(imgs_in=imgs_in,
                                                        inference=inference,
                                                        regularizations=regularizations)

        with torch.no_grad():
            if draw_image:
                imgs_rec = draw_img(c=inference.sample_c,
                                    bounding_box=inference.sample_bb,
                                    big_mask=inference.big_mask,
                                    big_img=inference.big_img,
                                    big_bg=inference.big_bg,
                                    draw_bg=draw_bg,
                                    draw_boxes=draw_boxes)
            else:
                imgs_rec = torch.zeros_like(imgs_in)

        return Output(metrics=metrics, inference=inference, imgs=imgs_rec)

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
