from MODULES.utilities import *
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

        # Raw image parameters
        self.dict_soft_constraints = params["soft_constraint"]
        self.nms_dict = params["nms"]
        self.sigma_fg = torch.tensor(params["GECO_loss"]["fg_std"], dtype=torch.float)[..., None, None]  # singleton w,h
        self.sigma_bg = torch.tensor(params["GECO_loss"]["bg_std"], dtype=torch.float)[..., None, None]  # singleton w,h

        self.geco_dict = params["GECO_loss"]
        self.input_img_dict = params["input_image"]

        # Initialize the lambda_eff = e^B-e^A
        one = torch.ones(1, dtype=torch.float)
        zero = torch.zeros(1, dtype=torch.float)
        self.geco_log_fgfraction_A = torch.nn.Parameter(data=zero, requires_grad=True)
        self.geco_log_fgfraction_B = torch.nn.Parameter(data=one, requires_grad=True)
        self.geco_log_ncell_A = torch.nn.Parameter(data=zero, requires_grad=True)
        self.geco_log_ncell_B = torch.nn.Parameter(data=one, requires_grad=True)
        self.geco_log_mse_A = torch.nn.Parameter(data=zero, requires_grad=True)
        self.geco_log_mse_B = torch.nn.Parameter(data=2.0*one, requires_grad=True)
        self.running_avarage_kl_logit = torch.nn.Parameter(data=4*torch.ones(1, dtype=torch.float), requires_grad=True)

        self.log_mse_max = numpy.log(self.geco_dict["geco_lambda_mse_max"])
        self.log_fg_max = numpy.log(self.geco_dict["geco_lambda_fgfraction_max"])
        self.log_ncell_max = numpy.log(self.geco_dict["geco_lambda_ncell_max"])

        # Put everything on the cude if cuda available
        if torch.cuda.is_available():
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

        # 1. Mixing probability should become certain.
        # I want to minimize the entropy: - sum_k pi_k log(pi_k)
        # Equivalently I can minimize overlap: sum_k pi_k * (1 - pi_k)
        # Both are minimized if pi_k = 0,1
        overlap = torch.sum(inference.mixing * (torch.ones_like(inference.mixing) - inference.mixing), dim=-5)  # sum boxes
        cost_overlap = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                    var_name="overlap",
                                                    var_value=overlap,
                                                    verbose=verbose,
                                                    chosen=chosen).sum(dim=(-1, -2, -3))  # sum over ch, w, h

        # Mask should have a min and max volume
        volume_mask_absolute = inference.mixing.sum(dim=(-1, -2, -3))  # sum over ch,w,h
        cost_volume_absolute = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                            var_name="mask_volume_absolute",
                                                            var_value=volume_mask_absolute,
                                                            verbose=verbose,
                                                            chosen=chosen)
        cost_volume_minibatch = (cost_volume_absolute * inference.sample_c.detach()).sum(dim=-2)  # sum boxes

        # Compute bounding boxes overlap matrix
        x1 = inference.sample_bb.bx - 0.5 * inference.sample_bb.bw  # boxes_few, batch_size
        x3 = inference.sample_bb.bx + 0.5 * inference.sample_bb.bw  # boxes_few, batch_size
        y1 = inference.sample_bb.bx - 0.5 * inference.sample_bb.bh  # boxes_few, batch_size
        y3 = inference.sample_bb.bx + 0.5 * inference.sample_bb.bh  # boxes_few, batch_size
        xi1 = torch.max(x1.unsqueeze(0), x1.unsqueeze(1))  # boxes_few, boxes_few, batch_size
        yi1 = torch.max(y1.unsqueeze(0), y1.unsqueeze(1))  # boxes_few, boxes_few, batch_size
        xi3 = torch.min(x3.unsqueeze(0), x3.unsqueeze(1))  # boxes_few, boxes_few, batch_size
        yi3 = torch.min(y3.unsqueeze(0), y3.unsqueeze(1))  # boxes_few, boxes_few, batch_size
        intersection_area = torch.clamp(xi3 - xi1, min=0) * torch.clamp(yi3 - yi1, min=0)  # boxes_few, boxes_few, batch_size

        # set diagonal and elements corresponding to off boxes to zero
        c_detached = inference.sample_c.detach()  # boxes_few, batch_size
        diag = torch.eye(c_detached.shape[0],
                         dtype=c_detached.dtype,
                         device=c_detached.device).unsqueeze(-1)  # boxes_few, boxes_few, 1
        box_overlap = torch.sum(intersection_area * c_detached.unsqueeze(0) *
                                c_detached.unsqueeze(1) * (1-diag), dim=(0, 1))

        cost_box_overlap = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                        var_name="box_overlap",
                                                        var_value=box_overlap,
                                                        verbose=verbose,
                                                        chosen=chosen).mean()

        # Compute the ideal Bounding boxes
        with torch.no_grad():
            n_width, n_height = inference.mixing.shape[-2:]
            ix_grid = torch.arange(start=0,
                                   end=n_width,
                                   dtype=torch.long,
                                   device=inference.mixing.device).unsqueeze(-1)  # n_width, 1
            iy_grid = torch.arange(start=0,
                                   end=n_height,
                                   dtype=torch.long,
                                   device=inference.mixing.device).unsqueeze(-2)  # 1, n_height

            mask = (inference.mixing > 0.5).long()  # shape: n_box_few, batch_size, 1, width, height
            # compute ideal x1,x3,y1,y3 of shape: n_box_few, batch_size
            buffer_size = 2
            ideal_x3 = torch.max(torch.flatten(mask * ix_grid, start_dim=-3), dim=-1)[0]
            ideal_y3 = torch.max(torch.flatten(mask * iy_grid, start_dim=-3), dim=-1)[0]
            ideal_x1 = n_width - torch.max(torch.flatten(mask * (n_width - ix_grid), start_dim=-3), dim=-1)[0]
            ideal_y1 = n_height - torch.max(torch.flatten(mask * (n_height - iy_grid), start_dim=-3), dim=-1)[0]
            ideal_x1 = (ideal_x1 - buffer_size).clamp(min=0, max=n_width)
            ideal_y1 = (ideal_y1 - buffer_size).clamp(min=0, max=n_height)
            ideal_x3 = (ideal_x3 + buffer_size).clamp(min=0, max=n_width)
            ideal_y3 = (ideal_y3 + buffer_size).clamp(min=0, max=n_height)

            # assuming that bx and bw are fixed. What should bw and bh be?
            size_obj_min = self.input_img_dict["size_object_min"]
            size_obj_max = self.input_img_dict["size_object_max"]
            bw_target = torch.max(ideal_x3 - inference.sample_bb.bx,
                                  inference.sample_bb.bx - ideal_x1).clamp(min=size_obj_min, max=size_obj_max)
            bh_target = torch.max(ideal_y3 - inference.sample_bb.by,
                                  inference.sample_bb.by - ideal_y1).clamp(min=size_obj_min, max=size_obj_max)

            # compute the cost
            dw_cost = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                   var_name="bounding_boxes_regression",
                                                   var_value=bw_target - inference.sample_bb.bw,
                                                   verbose=verbose,
                                                   chosen=chosen)
            dh_cost = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                                   var_name="bounding_boxes_regression",
                                                   var_value=bh_target - inference.sample_bb.bh,
                                                   verbose=verbose,
                                                   chosen=chosen)
            cost_bounding_box = (inference.sample_c.detach() * (dw_cost + dh_cost)).sum(dim=-2)  #sum over boxes

            # print("bw ->", bw_target[:, 0], inference.sample_bb.bw[:, 0])
            # print("bh ->", bh_target[:, 0], inference.sample_bb.bh[:, 0])
            # print("c  ->", inference.sample_c[:, 0])
            # print("DEBUG", cost_bounding_box.mean())

        return RegMiniBatch(reg_overlap=cost_overlap.mean(),            # mean over batch_size
                            reg_bb_regression=cost_bounding_box.mean(),  # mean over batch_size
                            reg_box_overlap=cost_box_overlap.mean(),  # mean over batch size
                            reg_area_obj=cost_volume_minibatch.mean())  # mean over batch_size

    @staticmethod
    def NLL_MSE(output: torch.tensor, target: torch.tensor, sigma: torch.tensor) -> torch.Tensor:
        return ((output - target) / sigma).pow(2)

    def compute_metrics(self,
                        imgs_in: torch.Tensor,
                        inference: Inference,
                        regularizations: RegMiniBatch) -> MetricMiniBatch:

        # Preparation
        n_box_few, batch_size = inference.sample_c.shape
        one = torch.ones(1, dtype=imgs_in.dtype, device=imgs_in.device)

        # 1. Observation model
        mixing_fg = torch.sum(inference.mixing, dim=-5)  # sum over boxes
        mixing_bg = one - mixing_fg
        mse = CompositionalVae.NLL_MSE(output=inference.big_img,
                                       target=imgs_in,
                                       sigma=self.sigma_fg)  # boxes, batch_size, ch, w, h
        mse_bg = CompositionalVae.NLL_MSE(output=inference.big_bg,
                                          target=imgs_in,
                                          sigma=self.sigma_bg)  # batch_size, ch, w, h
        mse_av = ((inference.mixing * mse).sum(dim=-5) + mixing_bg * mse_bg).mean()  # mean over batch_size, ch, w, h

        # 2. compute KL
        # Note that I compute the mean over batch, latent_dimensions and n_object.
        # This means that latent_dim can effectively control the complexity of the reconstruction,
        # i.e. more latent more capacity.
        kl_zbg = torch.mean(inference.kl_zbg)              # mean over: batch, latent_dim
        c_masked = inference.sample_c.detach().unsqueeze(-1)  # shape: n_boxes, batch, 1
        kl_zinstance = torch.mean(inference.kl_zinstance * c_masked) * n_box_few  # mean over: n_boxes, batch, latent_dim
        kl_zwhere = torch.mean(inference.kl_zwhere * c_masked) * n_box_few        # mean over: n_boxes, batch, latent_dim
        kl_logit = torch.mean(inference.kl_logit)  # mean over: batch
        kl_av = kl_zbg + kl_zinstance + kl_zwhere + \
                torch.exp(-self.running_avarage_kl_logit) * kl_logit + \
                self.running_avarage_kl_logit - self.running_avarage_kl_logit.detach()

        # GECO
        # 1. clamp_in_place
        self.geco_log_fgfraction_A.data.clamp_(max=self.log_fg_max, min=-10.0)
        self.geco_log_fgfraction_B.data.clamp_(max=self.log_fg_max, min=-10.0)
        self.geco_log_ncell_A.data.clamp_(max=self.log_ncell_max, min=-10.)
        self.geco_log_ncell_B.data.clamp_(max=self.log_ncell_max, min=-10.)
        self.geco_log_mse_A.data.clamp_(max=self.log_mse_max, min=-10.0)
        self.geco_log_mse_B.data.clamp_(max=self.log_mse_max, min=-10.0)

        # Get both the log_lambda and lambda_detached
        log_fg_A = self.geco_log_fgfraction_A
        log_fg_B = self.geco_log_fgfraction_B
        log_mse_A = self.geco_log_mse_A
        log_mse_B = self.geco_log_mse_B
        log_ncell_A = self.geco_log_ncell_A
        log_ncell_B = self.geco_log_ncell_B

        lambda_fg_A = self.geco_log_fgfraction_A.exp().detach()
        lambda_fg_B = self.geco_log_fgfraction_B.exp().detach()
        lambda_ncell_A = self.geco_log_ncell_A.exp().detach()
        lambda_ncell_B = self.geco_log_ncell_B.exp().detach()
        lambda_mse_A = self.geco_log_mse_A.exp().detach()
        lambda_mse_B = self.geco_log_mse_B.exp().detach()

        # 3. Compute constraint C where C<0
        fgfraction_av = torch.mean(mixing_fg)
        ncell_av = torch.sum(inference.sample_c_map_after_nms) / batch_size
        C_fg_A = self.geco_dict["target_fgfraction"][0] - fgfraction_av
        C_fg_B = fgfraction_av - self.geco_dict["target_fgfraction"][1]
        C_ncell_A = self.geco_dict["target_ncell"][0] - ncell_av
        C_ncell_B = ncell_av - self.geco_dict["target_ncell"][1]
        C_mse_A = self.geco_dict["target_mse"][0] - mse_av
        C_mse_B = mse_av - self.geco_dict["target_mse"][1]

        # loss_geco = - log_lambda * C where C<0
        loss_geco = - log_fg_A * C_fg_A.detach() - log_fg_B * C_fg_B.detach() \
                    - log_mse_A * C_mse_A.detach() - log_mse_B * C_mse_B.detach() \
                    - log_ncell_A * C_ncell_A.detach() - log_ncell_B * C_ncell_B.detach()

        # print("log_fg_A, log_fg_B, lambda_fg, fgfraction", log_fg_A, log_fg_B, lambda_fg_B - lambda_fg_A, fgfraction_av)

        # loss_vae = lambda * C
        reg_av = regularizations.total()
        loss_vae = (lambda_fg_B - lambda_fg_A) * fgfraction_av + \
                   (lambda_ncell_B - lambda_ncell_A) * torch.mean(inference.prob_map) + \
                   (lambda_mse_B - lambda_mse_A) * (mse_av + reg_av) + kl_av

        # add everything you want as long as there is one loss
        return MetricMiniBatch(loss=loss_vae + loss_geco,
                               mse_av=mse_av.detach().item(),
                               reg_av=reg_av.detach().item(),
                               kl_av=kl_av.detach().item(),
                               ncell_av=ncell_av.detach().item(),
                               fgfraction_av=fgfraction_av.detach().item(),

                               kl_zbg=kl_zbg.detach().item(),
                               kl_instance=kl_zinstance.detach().item(),
                               kl_where=kl_zwhere.detach().item(),
                               kl_logit=kl_logit.detach().item(),

                               lambda_fg=(lambda_fg_B-lambda_fg_A).detach().item(),
                               lambda_ncell=(lambda_ncell_B - lambda_ncell_A).detach().item(),
                               lambda_mse=(lambda_mse_B - lambda_mse_A).detach().item(),

                               count_prediction=torch.sum(inference.sample_c, dim=0).detach().cpu().numpy(),
                               wrong_examples=-1*numpy.ones(1),
                               accuracy=-1.0,

                               similarity_l=inference.similarity_l.detach().item(),
                               similarity_w=inference.similarity_w.detach().item(),
                               lambda_logit=self.running_avarage_kl_logit.detach().item())

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
                topk_only: bool = False,
                draw_boxes: bool = False,
                batch_of_index: Optional[torch.tensor] = None,
                max_index: Optional[int] = None,
                radius_nn: int = 5) -> Segmentation:
        """ Segment the batch of images """

        # start_time = time.time()
        n_objects_max = self.input_img_dict["n_objects_max"] if n_objects_max is None else n_objects_max
        prob_corr_factor = getattr(self, "prob_corr_factor", 0.0) if prob_corr_factor is None else prob_corr_factor
        overlap_threshold = self.nms_dict["overlap_threshold_test"] if overlap_threshold is None else overlap_threshold

        with torch.no_grad():
            inference: Inference = self.inference_and_generator(imgs_in=batch_imgs,
                                                                generate_synthetic_data=False,
                                                                prob_corr_factor=prob_corr_factor,
                                                                overlap_threshold=overlap_threshold,
                                                                n_objects_max=n_objects_max,
                                                                topk_only=topk_only,
                                                                noisy_sampling=noisy_sampling)

            # Now compute fg_prob, integer_segmentation_mask, similarity
            most_likely_mixing, index = torch.max(inference.mixing, dim=-5, keepdim=True)  # 1, batch_size, 1, w, h
            integer_mask = ((most_likely_mixing > 0.5) * (index + 1)).squeeze(-5).to(dtype=torch.int32)  # bg = 0 fg = 1,2,3,...
            fg_prob = torch.sum(inference.mixing, dim=-5)  # sum over instances

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
                similarity_matrix = CompositionalVae.compute_sparse_similarity_matrix(mixing_k=inference.mixing,
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
                            topk_only: bool = False,
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

                # pad before computing the cumsum
                roi_mask_padded = F.pad(roi_mask, pad=pad_list, mode='constant', value=0)
                cum_roi_mask = torch.cumsum(torch.cumsum(roi_mask_padded, dim=-1), dim=-2)
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
                                            topk_only=topk_only,
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

    # this is the fully generic function which has all the options unspecified
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
                           n_objects_max: int) -> Output:
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
                                                            noisy_sampling=noisy_sampling)

        regularizations: RegMiniBatch = self.compute_regularizations(inference=inference,
                                                                     verbose=verbose)

        metrics: MetricMiniBatch = self.compute_metrics(imgs_in=imgs_in,
                                                        inference=inference,
                                                        regularizations=regularizations)

        with torch.no_grad():
            if draw_image:
                imgs_rec = draw_img(c=inference.sample_c,
                                    bounding_box=inference.sample_bb,
                                    mixing_k=inference.mixing,
                                    big_img=inference.big_img,
                                    big_bg=inference.big_bg,
                                    draw_bg=draw_bg,
                                    draw_boxes=draw_boxes)
            else:
                imgs_rec = torch.zeros_like(imgs_in)

        return Output(metrics=metrics, inference=inference, imgs=imgs_rec)

    def forward(self,
                imgs_in: torch.tensor,
                overlap_threshold: float,
                noisy_sampling: bool = True,
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
                                       noisy_sampling=noisy_sampling,
                                       prob_corr_factor=getattr(self, "prob_corr_factor", 0.0),
                                       overlap_threshold=overlap_threshold,
                                       n_objects_max=self.input_img_dict["n_objects_max"])

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
                                           overlap_threshold=-1.0,
                                           n_objects_max=self.input_img_dict["n_objects_max"])
