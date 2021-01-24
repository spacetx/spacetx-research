import torch
import torch.nn.functional as F

from MODULES.cropper_uncropper import Uncropper, Cropper
from MODULES.unet_model import UNet
from MODULES.encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, DecoderBackground
from MODULES.utilities import tmaps_to_bb, convert_to_box_list, invert_convert_to_box_list
from MODULES.utilities import compute_ranking, compute_average_in_box
from MODULES.utilities_ml import sample_and_kl_diagonal_normal, sample_c_map
from MODULES.utilities_ml import compute_kl_DPP, compute_kl_Bernoulli, SimilarityKernel
from MODULES.namedtuple import Inference, NMSoutput, BB, UNEToutput, ZZ, DIST
from MODULES.non_max_suppression import NonMaxSuppression


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

    def forward(self, imgs_in: torch.Tensor,
                generate_synthetic_data: bool,
                prob_corr_factor: float,
                overlap_threshold: float,
                n_objects_max: int,
                topk_only: bool,
                noisy_sampling: bool) -> Inference:

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
        c_map_before_nms = sample_c_map(p_map=p_map,
                                        similarity_kernel=similarity_kernel,
                                        noisy_sampling=noisy_sampling,
                                        sample_from_prior=generate_synthetic_data)

        # NMS + top-K operation
        with torch.no_grad():
            score = convert_to_box_list(c_map_before_nms+p_map).squeeze(-1)  # shape: n_box_all, batch_size
            combined_topk_only = topk_only or generate_synthetic_data  # if generating from DPP do not do NMS
            nms_output: NMSoutput = NonMaxSuppression.compute_mask_and_index(score=score,
                                                                             bounding_box=bounding_box_all,
                                                                             overlap_threshold=overlap_threshold,
                                                                             n_objects_max=n_objects_max,
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

        # Compute the overlap
        # A = (x1+x2+x3)^2 = x1^2 + x2^2 + x3^2 + 2 x1*x2 + 2 x1*x3 + 2 x2*x3
        # Therefore sum_{i \ne j} x_i x_j = x1*x2 + x1*x3 + x2*x3 = 0.5 * [(sum xi)^2 - (sum xi^2)]
        sum_x = big_mask_times_c.sum(dim=-5)  # sum over boxes first
        sum_x2 = big_mask_times_c.pow(2).sum(dim=-5)  # square first and sum over boxes later
        mask_overlap = 0.5 * (sum_x.pow(2) - sum_x2).clamp(min=0)

        return Inference(prob_map=p_map,
                         big_bg=big_bg,
                         mixing=mixing,
                         big_img=big_img,
                         mask_overlap=mask_overlap,
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
