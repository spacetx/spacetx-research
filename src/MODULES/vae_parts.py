import torch
import torch.nn.functional as F
import numpy
from typing import Optional

from MODULES.cropper_uncropper import Uncropper, Cropper
from MODULES.unet_model import UNet
from MODULES.encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, DecoderBackground
from MODULES.utilities import convert_to_box_list, invert_convert_to_box_list
from MODULES.utilities import compute_ranking, compute_average_in_box
from MODULES.utilities_ml import sample_and_kl_diagonal_normal, sample_c_map
from MODULES.utilities_ml import compute_kl_DPP, compute_kl_Bernoulli, SimilarityKernel
from MODULES.namedtuple import Inference, NMSoutput, BB, UNEToutput, ZZ, DIST, RegMiniBatch, MetricMiniBatch
from MODULES.non_max_suppression import NonMaxSuppression


def tmaps_to_bb(tmaps, width_raw_image: int, height_raw_image: int, min_box_size: float, max_box_size: float):
    tx_map, ty_map, tw_map, th_map = torch.split(tmaps, 1, dim=-3)
    n_width, n_height = tx_map.shape[-2:]
    ix_array = torch.arange(start=0, end=n_width, dtype=tx_map.dtype, device=tx_map.device)
    iy_array = torch.arange(start=0, end=n_height, dtype=tx_map.dtype, device=tx_map.device)
    ix_grid, iy_grid = torch.meshgrid([ix_array, iy_array])

    bx_map: torch.Tensor = width_raw_image * (ix_grid + tx_map) / n_width
    by_map: torch.Tensor = height_raw_image * (iy_grid + ty_map) / n_height
    bw_map: torch.Tensor = min_box_size + (max_box_size - min_box_size) * tw_map
    bh_map: torch.Tensor = min_box_size + (max_box_size - min_box_size) * th_map
    return BB(bx=convert_to_box_list(bx_map).squeeze(-1),
              by=convert_to_box_list(by_map).squeeze(-1),
              bw=convert_to_box_list(bw_map).squeeze(-1),
              bh=convert_to_box_list(bh_map).squeeze(-1))


def sample_from_constraints_dict(dict_soft_constraints: dict,
                                 var_name: str,
                                 var_value: torch.Tensor,
                                 verbose: bool = False,
                                 chosen: Optional[int] = None) -> torch.Tensor:

    cost = torch.zeros_like(var_value)
    var_constraint_params = dict_soft_constraints[var_name]

    if 'lower_bound_value' in var_constraint_params:
        left = var_constraint_params['lower_bound_value']
        width_low = var_constraint_params['lower_bound_width']
        exponent_low = var_constraint_params['lower_bound_exponent']
        strength_low = var_constraint_params['lower_bound_strength']
        activity_low = torch.clamp(left + width_low - var_value, min=0.) / width_low
        cost += strength_low * activity_low.pow(exponent_low)

    if 'upper_bound_value' in var_constraint_params:
        right = var_constraint_params['upper_bound_value']
        width_up = var_constraint_params['upper_bound_width']
        exponent_up = var_constraint_params['upper_bound_exponent']
        strength_up = var_constraint_params['upper_bound_strength']
        activity_up = torch.clamp(var_value - right + width_up, min=0.) / width_up
        cost += strength_up * activity_up.pow(exponent_up)

    if 'strength' in var_constraint_params:
        strength = var_constraint_params['strength']
        exponent = var_constraint_params['exponent']
        cost += strength * var_value.pow(exponent)

    if verbose:
        if chosen is None:
            print("constraint name ->", var_name)
            print("input value ->", var_value)
            print("cost ->", cost)
        else:
            print("constraint name ->", var_name)
            print("input value ->", var_value[..., chosen])
            print("cost ->", cost[..., chosen])

    return cost


def from_w_to_pi(weight: torch.Tensor, dim: int):
    """ Compute the interacting and non-interacting mixing probabilities
        Make sure that when summing over dim=dim the mask sum to zero or one
        mask_j = fg_mask * partitioning_j
        where fg_mask = tanh ( sum_i w_i) and partitioning_j = w_j / (sum_i w_i)
    """
    assert len(weight.shape) == 5
    sum_weight = torch.sum(weight, dim=dim, keepdim=True)
    fg_mask = torch.tanh(sum_weight)
    partitioning = weight / torch.clamp(sum_weight, min=1E-6)
    return fg_mask * partitioning


class Inference_and_Generation(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        # variables
        self.size_max: int = params["input_image"]["size_object_max"]
        self.size_min: int = params["input_image"]["size_object_min"]
        self.cropped_size: int = params["architecture"]["cropped_size"]

        # modules
        self.similarity_kernel_dpp = SimilarityKernel(n_kernels=params["DPP"]["n_kernels"])
        self.unet: UNet = UNet(params)

        # Decoders
        self.decoder_zbg: DecoderBackground = DecoderBackground(dim_z=params["architecture"]["dim_zbg"],
                                                                ch_out=params["input_image"]["ch_in"])

        self.decoder_zwhere: Decoder1by1Linear = Decoder1by1Linear(dim_z=params["architecture"]["dim_zwhere"],
                                                                   ch_out=4)

        self.decoder_logit: Decoder1by1Linear = Decoder1by1Linear(dim_z=params["architecture"]["dim_logit"],
                                                                  ch_out=1)

        self.decoder_zinstance: DecoderConv = DecoderConv(size=params["architecture"]["cropped_size"],
                                                          dim_z=params["architecture"]["dim_zinstance"],
                                                          ch_out=params["input_image"]["ch_in"] + 1)

        # Encoders
        self.encoder_zinstance: EncoderConv = EncoderConv(size=params["architecture"]["cropped_size"],
                                                          ch_in=params["architecture"]["n_ch_output_features"],
                                                          dim_z=params["architecture"]["dim_zinstance"])

        # Raw image parameters
        self.dict_soft_constraints = params["soft_constraint"]
        self.nms_dict = params["nms"]
        self.sigma_fg = torch.nn.Parameter(data=torch.tensor(params["GECO_loss"]["fg_std"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)
        self.sigma_bg = torch.nn.Parameter(data=torch.tensor(params["GECO_loss"]["bg_std"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)

        self.geco_dict = params["GECO_loss"]
        self.input_img_dict = params["input_image"]

        self.geco_fgfraction = torch.nn.Parameter(data=torch.tensor(self.geco_dict["geco_fgfraction_range"][1],
                                                                    dtype=torch.float), requires_grad=True)
        self.geco_ncell = torch.nn.Parameter(data=torch.tensor(self.geco_dict["geco_ncell_range"][1],
                                                               dtype=torch.float), requires_grad=True)
        self.geco_mse = torch.nn.Parameter(data=torch.tensor(self.geco_dict["geco_mse_range"][1],
                                                             dtype=torch.float), requires_grad=True)

        self.running_avarage_kl_logit = torch.nn.Parameter(data=4 * torch.ones(1, dtype=torch.float),
                                                           requires_grad=True)

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
        overlap = torch.sum(inference.mixing * (torch.ones_like(inference.mixing) - inference.mixing),
                            dim=-5)  # sum boxes
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
            buffer_size = 3
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
                                               var_value=torch.abs(bw_target - inference.sample_bb.bw),
                                               verbose=verbose,
                                               chosen=chosen)
        dh_cost = sample_from_constraints_dict(dict_soft_constraints=self.dict_soft_constraints,
                                               var_name="bounding_boxes_regression",
                                               var_value=torch.abs(bh_target - inference.sample_bb.bh),
                                               verbose=verbose,
                                               chosen=chosen)
        cost_bounding_box = (inference.sample_c.detach() * (dw_cost + dh_cost)).sum(dim=-2)  # sum over boxes

        # print("bw ->", bw_target[:, 0], inference.sample_bb.bw[:, 0])
        # print("bh ->", bh_target[:, 0], inference.sample_bb.bh[:, 0])
        # print("c  ->", inference.sample_c[:, 0])
        # print("DEBUG", cost_bounding_box.mean())

        return RegMiniBatch(reg_overlap=cost_overlap.mean(),  # mean over batch_size
                            reg_bbox_regression=cost_bounding_box.mean(),  # mean over batch size
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

        # 1. Observation model
        mixing_fg = torch.sum(inference.mixing, dim=-5)  # sum over boxes
        mixing_bg = torch.ones_like(mixing_fg) - mixing_fg
        mse = Inference_and_Generation.NLL_MSE(output=inference.big_img,
                                               target=imgs_in,
                                               sigma=self.sigma_fg)  # boxes, batch_size, ch, w, h
        mse_bg = Inference_and_Generation.NLL_MSE(output=inference.big_bg,
                                                  target=imgs_in,
                                                  sigma=self.sigma_bg)  # batch_size, ch, w, h
        mse_av = ((inference.mixing * mse).sum(dim=-5) + mixing_bg * mse_bg).mean()  # mean over batch_size, ch, w, h

        # TODO: put htis stuff inside torch.no_grad()
        with torch.no_grad():
            g_mse = (min(self.geco_dict["target_mse"]) - mse_av).clamp(min=0) + \
                    (max(self.geco_dict["target_mse"]) - mse_av).clamp(max=0)

        # 2. Sparsity should encourage:
        # 1. few object
        # 2. tight bounding boxes
        # 3. tight masks
        # The three terms take care of all these requirement.
        # Note:
        # 1) All the terms contain c=Bernoulli(p). It is actually the same b/c during back prop c=p
        # 2) fg_fraction is based on the selected quantities
        # 3) sparsity n_cell is based on c_map so that the entire matrix becomes sparse.
        with torch.no_grad():
            x_sparsity_av = torch.mean(mixing_fg)
            x_sparsity_max = max(self.geco_dict["target_fgfraction"])
            x_sparsity_min = min(self.geco_dict["target_fgfraction"])
            # g_sparsity = (x_sparsity_min - x_sparsity_av).clamp(min=0) + \
            #              (x_sparsity_max - x_sparsity_av).clamp(max=0)
            g_sparsity = torch.min(x_sparsity_av - x_sparsity_min,
                                   x_sparsity_max - x_sparsity_av)  # positive if in range
        c_times_area_few = inference.sample_c * inference.sample_bb.bw * inference.sample_bb.bh
        x_sparsity = 0.5 * (torch.sum(mixing_fg) + torch.sum(c_times_area_few)) / torch.numel(mixing_fg)
        f_sparsity = x_sparsity * torch.sign(x_sparsity_av - x_sparsity_min).detach()

        with torch.no_grad():
            x_cell_av = torch.sum(inference.sample_c_map_after_nms) / batch_size
            x_cell_max = max(self.geco_dict["target_ncell"])
            x_cell_min = min(self.geco_dict["target_ncell"])
            # g_cell = ((x_cell_min - x_cell_av).clamp(min=0) + (x_cell_max - x_cell_av).clamp(max=0)) / n_box_few
            g_cell = torch.min(x_cell_av - x_cell_min,
                               x_cell_max - x_cell_av) / n_box_few  # positive if in range, negative otherwise
        x_cell = torch.sum(inference.sample_c_map_before_nms) / (batch_size * n_box_few)
        f_cell = x_cell * torch.sign(x_cell_av - x_cell_min).detach()

        # 3. compute KL
        # Note that I compute the mean over batch, latent_dimensions and n_object.
        # This means that latent_dim can effectively control the complexity of the reconstruction,
        # i.e. more latent more capacity.
        kl_zbg = torch.mean(inference.kl_zbg)  # mean over: batch, latent_dim
        kl_zinstance = torch.mean(inference.kl_zinstance)  # mean over: n_boxes, batch, latent_dim
        kl_zwhere = torch.mean(inference.kl_zwhere)  # mean over: n_boxes, batch, latent_dim
        kl_logit = torch.mean(inference.kl_logit)  # mean over: batch

        kl_av = kl_zbg + kl_zinstance + kl_zwhere + \
                torch.exp(-self.running_avarage_kl_logit) * kl_logit + \
                self.running_avarage_kl_logit - self.running_avarage_kl_logit.detach()

        # 6. Note that I clamp in_place
        geco_mse_detached = self.geco_mse.data.clamp_(min=min(self.geco_dict["geco_mse_range"]),
                                                      max=max(self.geco_dict["geco_mse_range"])).detach()
        geco_ncell_detached = self.geco_ncell.data.clamp_(min=min(self.geco_dict["geco_ncell_range"]),
                                                          max=max(self.geco_dict["geco_ncell_range"])).detach()
        geco_fgfraction_detached = self.geco_fgfraction.data.clamp_(min=min(self.geco_dict["geco_fgfraction_range"]),
                                                                    max=max(self.geco_dict[
                                                                                "geco_fgfraction_range"])).detach()
        one_minus_geco_mse_detached = torch.ones_like(geco_mse_detached) - geco_mse_detached

        reg_av = regularizations.total()
        sparsity_av = geco_fgfraction_detached * f_sparsity + geco_ncell_detached * f_cell
        loss_vae = sparsity_av + geco_mse_detached * (mse_av + reg_av) + one_minus_geco_mse_detached * kl_av
        loss_geco = self.geco_fgfraction * g_sparsity.detach() + \
                    self.geco_ncell * g_cell.detach() + \
                    self.geco_mse * g_mse.detach()
        loss = loss_vae + loss_geco - loss_geco.detach()

        # add everything you want as long as there is one loss
        return MetricMiniBatch(loss=loss,
                               mse_tot=mse_av.detach().item(),
                               reg_tot=reg_av.detach().item(),
                               kl_tot=kl_av.detach().item(),
                               sparsity_tot=sparsity_av.detach().item(),

                               kl_zbg=kl_zbg.detach().item(),
                               kl_instance=kl_zinstance.detach().item(),
                               kl_where=kl_zwhere.detach().item(),
                               kl_logit=kl_logit.detach().item(),

                               reg_overlap=regularizations.reg_overlap.detach().item(),
                               reg_area_obj=regularizations.reg_area_obj.detach().item(),

                               lambda_sparsity=self.geco_fgfraction.data.detach().item(),
                               lambda_cell=self.geco_ncell.data.detach().item(),
                               lambda_mse=self.geco_mse.data.detach().item(),
                               f_sparsity=f_sparsity.detach().item(),
                               g_sparsity=g_sparsity.detach().item(),
                               f_cell=f_cell.detach().item(),
                               g_cell=g_cell.detach().item(),
                               f_mse=mse_av.detach().item(),
                               g_mse=g_mse.detach().item(),
                               fg_fraction_av=x_sparsity_av.detach().item(),
                               n_cell_av=x_cell_av.detach().item(),

                               count_prediction=torch.sum(inference.sample_c, dim=0).detach().cpu().numpy(),
                               wrong_examples=-1 * numpy.ones(1),
                               accuracy=-1.0,

                               similarity_l=inference.similarity_l.detach().item(),
                               similarity_w=inference.similarity_w.detach().item(),
                               lambda_logit=self.running_avarage_kl_logit.detach().item())

    def compute_inference(self, imgs_in: torch.Tensor,
                          generate_synthetic_data: bool,
                          prob_corr_factor: float,
                          overlap_threshold: float,
                          n_objects_max: int,
                          topk_only: bool,
                          noisy_sampling: bool,
                          quantize_prob: bool,
                          quantize_prob_value: float) -> Inference:

        # 0. preparation
        batch_size, ch_raw_image, width_raw_image, height_raw_image = imgs_in.shape

        # ---------------------------#
        # 1. UNET
        # ---------------------------#
        unet_output: UNEToutput = self.unet.forward(imgs_in, verbose=False)

        # background
        zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zbg.mu,
                                                  posterior_std=unet_output.zbg.std,
                                                  prior_mu=torch.zeros_like(unet_output.zbg.mu),
                                                  prior_std=torch.ones_like(unet_output.zbg.std),
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data)

        big_bg = torch.sigmoid(self.decoder_zbg(z=zbg.sample,
                                                high_resolution=(imgs_in.shape[-2], imgs_in.shape[-1])))

        # bounbding boxes
        zwhere_map: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zwhere.mu,
                                                         posterior_std=unet_output.zwhere.std,
                                                         prior_mu=torch.zeros_like(unet_output.zwhere.mu),
                                                         prior_std=torch.ones_like(unet_output.zwhere.std),
                                                         noisy_sampling=noisy_sampling,
                                                         sample_from_prior=generate_synthetic_data)

        bounding_box_all: BB = tmaps_to_bb(tmaps=torch.sigmoid(self.decoder_zwhere(zwhere_map.sample)),
                                           width_raw_image=width_raw_image,
                                           height_raw_image=height_raw_image,
                                           min_box_size=self.size_min,
                                           max_box_size=self.size_max)

        with torch.no_grad():

            # Correct probability if necessary
            if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):
                av_intensity = compute_average_in_box((imgs_in - big_bg).abs(), bounding_box_all)
                assert len(av_intensity.shape) == 2
                n_boxes_all, batch_size = av_intensity.shape
                ranking = compute_ranking(av_intensity)  # n_boxes_all, batch. It is in [0,n_box_all-1]
                tmp = (ranking + 1).float() / n_boxes_all  # less or equal to 1
                q_approx = tmp.pow(10)  # suppress most probabilities but keep few close to 1.
                p_map_delta = invert_convert_to_box_list(q_approx.unsqueeze(-1),
                                                         original_width=unet_output.logit.shape[-2],
                                                         original_height=unet_output.logit.shape[-1])

        # Now I have p, log(p), log(1-p)
        if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):
            p_map = ((1 - prob_corr_factor) * torch.sigmoid(unet_output.logit) +
                     prob_corr_factor * p_map_delta).clamp(min=1E-4, max=1-1E-4)
            log_p_map = torch.log(p_map)
            log_one_minus_p_map = torch.log1p(-p_map)
        else:
            p_map = torch.sigmoid(unet_output.logit)
            log_p_map = F.logsigmoid(unet_output.logit)
            log_one_minus_p_map = F.logsigmoid(-unet_output.logit)

        # Sample the probability map from prior or posterior
        similarity_kernel = self.similarity_kernel_dpp.forward(n_width=unet_output.logit.shape[-2],
                                                               n_height=unet_output.logit.shape[-1])
        if quantize_prob:
            # print("I am quantizing the probability")
            # print(p_map[0,0].sum(), (p_map > quantize_prob_value).float()[0,0].sum())
            c_map_before_nms = sample_c_map(p_map=(p_map > quantize_prob_value).float(),
                                            similarity_kernel=similarity_kernel,
                                            noisy_sampling=noisy_sampling,
                                            sample_from_prior=generate_synthetic_data)
        else:
            c_map_before_nms = sample_c_map(p_map=p_map,
                                            similarity_kernel=similarity_kernel,
                                            noisy_sampling=noisy_sampling,
                                            sample_from_prior=generate_synthetic_data)

        # NMS + top-K operation
        with torch.no_grad():
            score = convert_to_box_list(c_map_before_nms+p_map).squeeze(-1)  # shape: n_box_all, batch_size
            combined_topk_only = topk_only or generate_synthetic_data  # if generating from DPP do not do NMS
            nms_output: NMSoutput = NonMaxSuppression.compute_mask_and_index(score_nb=score,
                                                                             bounding_box_nb=bounding_box_all,
                                                                             iom_threshold=overlap_threshold,
                                                                             k_objects_max=n_objects_max,
                                                                             topk_only=combined_topk_only)
            # Mask with all zero except 1s where the box where selected
            mask = torch.zeros_like(score).scatter(dim=0,
                                                   index=nms_output.index_top_k,
                                                   src=torch.ones_like(score))  # shape: n_box_all, batch_size
            mask_map = invert_convert_to_box_list(mask.unsqueeze(-1),
                                                  original_width=c_map_before_nms.shape[-2],
                                                  original_height=c_map_before_nms.shape[-1])  # shape: batch_size, 1, w, h

        # TODO: check if I can use the c_map_after_nms in both places.....
        kl_logit_prior = compute_kl_DPP(c_map=(c_map_before_nms * mask_map).detach(),
                                        similarity_kernel=similarity_kernel)
        kl_logit_posterior = compute_kl_Bernoulli(c_map=(c_map_before_nms * mask_map).detach(),
                                                  log_p_map=log_p_map,
                                                  log_one_minus_p_map=log_one_minus_p_map)
        kl_logit = kl_logit_posterior - kl_logit_prior  # this will make adjust DPP and keep entropy of posterior

        c_few = torch.gather(convert_to_box_list(c_map_before_nms).squeeze(-1), dim=0, index=nms_output.index_top_k)

        bounding_box_few: BB = BB(bx=torch.gather(bounding_box_all.bx, dim=0, index=nms_output.index_top_k),
                                  by=torch.gather(bounding_box_all.by, dim=0, index=nms_output.index_top_k),
                                  bw=torch.gather(bounding_box_all.bw, dim=0, index=nms_output.index_top_k),
                                  bh=torch.gather(bounding_box_all.bh, dim=0, index=nms_output.index_top_k))

        zwhere_sample_all = convert_to_box_list(zwhere_map.sample)  # shape: nbox_all, batch_size, ch
        zwhere_kl_all = convert_to_box_list(zwhere_map.kl)          # shape: nbox_all, batch_size, ch
        new_index = nms_output.index_top_k.unsqueeze(-1).expand(-1, -1, zwhere_kl_all.shape[-1])  # shape: nbox_few, batch_size, ch
        zwhere_kl_few = torch.gather(zwhere_kl_all, dim=0, index=new_index)  # shape (nbox_few, batch_size, ch)
        zwhere_sample_few = torch.gather(zwhere_sample_all, dim=0, index=new_index)

        # ------------------------------------------------------------------#
        # 5. Crop the unet_features according to the selected boxes
        # ------------------------------------------------------------------#
        n_boxes, batch_size = bounding_box_few.bx.shape
        unet_features_expanded = unet_output.features.unsqueeze(0).expand(n_boxes, batch_size, -1, -1, -1)
        cropped_feature_map: torch.Tensor = Cropper.crop(bounding_box=bounding_box_few,
                                                         big_stuff=unet_features_expanded,
                                                         width_small=self.cropped_size,
                                                         height_small=self.cropped_size)

        # ------------------------------------------------------------------#
        # 6. Encode, sample z and decode to big images and big weights
        # ------------------------------------------------------------------#
        zinstance_posterior: ZZ = self.encoder_zinstance.forward(cropped_feature_map)
        zinstance_few: DIST = sample_and_kl_diagonal_normal(posterior_mu=zinstance_posterior.mu,
                                                            posterior_std=zinstance_posterior.std,
                                                            prior_mu=torch.zeros_like(zinstance_posterior.mu),
                                                            prior_std=torch.ones_like(zinstance_posterior.std),
                                                            noisy_sampling=noisy_sampling,
                                                            sample_from_prior=generate_synthetic_data)

        small_stuff = torch.sigmoid(self.decoder_zinstance.forward(zinstance_few.sample))  # stuff between 0 and 1
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_few,
                                     small_stuff=small_stuff,
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: n_box, batch, ch, w, h
        big_mask, big_img = torch.split(big_stuff, split_size_or_sections=(1, big_stuff.shape[-3]-1), dim=-3)
        big_mask_times_c = big_mask * c_few[..., None, None, None]  # this is strictly smaller than 1
        mixing = big_mask_times_c / big_mask_times_c.sum(dim=-5).clamp_(min=1.0)  # softplus-like function
        similarity_l, similarity_w = self.similarity_kernel_dpp.get_l_w()

        return Inference(prob_map=p_map,
                         big_bg=big_bg,
                         mixing=mixing,
                         big_img=big_img,
                         # the sample of the 4 latent variables
                         sample_c_map_before_nms=c_map_before_nms,
                         sample_c_map_after_nms=(c_map_before_nms * mask_map),
                         sample_c=c_few,
                         sample_bb=bounding_box_few,
                         sample_zwhere=zwhere_sample_few,
                         sample_zinstance=zinstance_few.sample,
                         sample_zbg=zbg.sample,
                         # the kl of the 4 latent variables
                         kl_logit=kl_logit,
                         kl_zwhere=zwhere_kl_few,
                         kl_zinstance=zinstance_few.kl,
                         kl_zbg=zbg.kl,
                         # similarity kernels
                         similarity_l=similarity_l,
                         similarity_w=similarity_w)

    def forward(self, imgs_in: torch.Tensor,
                generate_synthetic_data: bool,
                prob_corr_factor: float,
                overlap_threshold: float,
                n_objects_max: int,
                topk_only: bool,
                noisy_sampling: bool,
                quantize_prob: bool,
                quantize_prob_value: float) -> (Inference, MetricMiniBatch):

        inference: Inference = self.compute_inference(imgs_in=imgs_in,
                                                      generate_synthetic_data=generate_synthetic_data,
                                                      prob_corr_factor=prob_corr_factor,
                                                      overlap_threshold=overlap_threshold,
                                                      n_objects_max=n_objects_max,
                                                      topk_only=topk_only,
                                                      noisy_sampling=noisy_sampling,
                                                      quantize_prob=quantize_prob,
                                                      quantize_prob_value=quantize_prob_value)

        regularizations: RegMiniBatch = self.compute_regularizations(inference=inference, verbose=False)

        metric: MetricMiniBatch = self.compute_metrics(imgs_in=imgs_in,
                                                       inference=inference,
                                                       regularizations=regularizations)

        return inference, metric
