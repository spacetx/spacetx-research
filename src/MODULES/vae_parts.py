import torch
import torch.nn.functional as F

from MODULES.cropper_uncropper import Uncropper, Cropper
from MODULES.unet_model import UNet
from MODULES.encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, DecoderBackground
from MODULES.utilities import tmaps_to_bb, convert_to_box_list, invert_convert_to_box_list, compute_prob_correction
from MODULES.utilities_ml import sample_and_kl_diagonal_normal, sample_and_kl_prob, SimilarityKernel
from MODULES.namedtuple import Inference, BB, UNEToutput, ZZ, DIST
# from MODULES.non_max_suppression import NonMaxSuppression


def from_weights_to_masks(weight: torch.Tensor, dim: int):
    """ Make sure that when summing over dim=dim the mask sum to zero or one
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
                noisy_sampling: bool,
                bg_is_zero: bool,
                bg_resolution: tuple) -> Inference:

        # 0. preparation
        batch_size, ch_raw_image, width_raw_image, height_raw_image = imgs_in.shape

        # ---------------------------#
        # 1. UNET
        # ---------------------------#
        unet_output: UNEToutput = self.unet.forward(imgs_in, verbose=False)

        zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zbg.mu,
                                                  posterior_std=unet_output.zbg.std,
                                                  prior_mu=torch.zeros_like(unet_output.zbg.mu),
                                                  prior_std=torch.ones_like(unet_output.zbg.std),
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data)
        if bg_is_zero:
            big_bg = torch.zeros_like(imgs_in)
        else:
            big_bg = torch.sigmoid(self.decoder_zbg(z=zbg.sample,
                                                    high_resolution=(imgs_in.shape[-2], imgs_in.shape[-1])))

        # correct prob if necessary:
        with torch.no_grad():
            if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):
                bounding_box_no_noise: BB = tmaps_to_bb(tmaps=torch.sigmoid(self.decoder_zwhere(unet_output.zwhere.mu)),
                                                        width_raw_image=width_raw_image,
                                                        height_raw_image=height_raw_image,
                                                        min_box_size=self.size_min,
                                                        max_box_size=self.size_max)
                delta_p = compute_prob_correction(images=imgs_in,
                                                  background=big_bg,
                                                  bounding_box=bounding_box_no_noise)
                logit_uncorrected = convert_to_box_list(unet_output.logit.mu).squeeze(-1)
                p_uncorrected = torch.sigmoid(logit_uncorrected)
                p_corrected = ((1 - prob_corr_factor) * p_uncorrected + prob_corr_factor * delta_p)
                delta_logit = torch.logit(p_corrected, eps=0.0001) - logit_uncorrected
                delta_logit_map = invert_convert_to_box_list(delta_logit.unsqueeze(-1),
                                                             original_width=unet_output.logit.mu.shape[-2],
                                                             original_height=unet_output.logit.mu.shape[-2])
            else:
                delta_logit_map = torch.zeros_like(unet_output.logit.mu)
        logit_map = unet_output.logit.mu + delta_logit_map

        # similarity kernel is necessary to compute log_prob(c)
        similarity_kernel = self.similarity_kernel_dpp.forward(n_width=logit_map.shape[-2],
                                                               n_height=logit_map.shape[-1])

        c_distribution: DIST = sample_and_kl_prob(logit_map=logit_map,
                                                  similarity_kernel=similarity_kernel,
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data)

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

        area_all = bounding_box_all.bw * bounding_box_all.bh
        area_map = invert_convert_to_box_list(area_all.unsqueeze(-1),
                                              original_width=unet_output.logit.mu.shape[-2],
                                              original_height=unet_output.logit.mu.shape[-1])

        prob_map = torch.sigmoid(c_distribution.sample)
        c_all = convert_to_box_list(c_distribution.sample).squeeze(-1)
        with torch.no_grad():
            k = min(n_objects_max,  c_all.shape[0])
            index_top_k = torch.topk(c_all, k=k, dim=-2, largest=True, sorted=True)[1]

        c_few = torch.gather(c_all, dim=0, index=index_top_k)
        prob_few = torch.gather(convert_to_box_list(prob_map).squeeze(-1), dim=0, index=index_top_k)
        bounding_box_few: BB = BB(bx=torch.gather(bounding_box_all.bx, dim=0, index=index_top_k),
                                  by=torch.gather(bounding_box_all.by, dim=0, index=index_top_k),
                                  bw=torch.gather(bounding_box_all.bw, dim=0, index=index_top_k),
                                  bh=torch.gather(bounding_box_all.bh, dim=0, index=index_top_k))

        zwhere_kl_all = convert_to_box_list(zwhere_map.kl)  # shape: nbox_all, batch_size, ch
        zwhere_sample_all = convert_to_box_list(zwhere_map.sample)  # shape: nbox_all, batch_size, ch
        new_index = index_top_k.unsqueeze(-1).expand(-1, -1, zwhere_kl_all.shape[-1])  # shape: nbox_few, batch_size, ch
        zwhere_sample_few = torch.gather(zwhere_sample_all, dim=0, index=new_index)  # shape: nbox_few, batch_size, ch
        zwhere_kl_few = torch.gather(zwhere_kl_all, dim=0, index=new_index)  # shape: nbox_few, batch_size, ch)

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

        small_stuff_raw = self.decoder_zinstance.forward(zinstance_few.sample)
        # Apply softplus to first channel (i.e. mask channel) and sigmoid to all others (i.e. img channels)
        small_stuff = torch.cat((F.softplus(small_stuff_raw[..., :1, :, :]),
                                 torch.sigmoid(small_stuff_raw[..., 1:, :, :])), dim=-3)
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_few,
                                     small_stuff=small_stuff*c_few[..., None, None, None],  # singleton for: ch, w, h
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: n_box, batch, ch, w, h
        ch_size = big_stuff.shape[-3]
        big_weight, big_img = torch.split(big_stuff, split_size_or_sections=(1, ch_size-1), dim=-3)

        # -----------------------
        # 7. From weight to masks
        # ------------------------
        big_mask = from_weights_to_masks(weight=big_weight, dim=-5)
        big_mask_NON_interacting = torch.tanh(big_weight)

        # 8. Return the inferred quantities
        similarity_l, similarity_w = self.similarity_kernel_dpp.get_l_w()
        return Inference(area_map=area_map,
                         prob_map=prob_map,
                         prob_few=prob_few,
                         c_few=c_few,
                         bb_few=bounding_box_few,
                         big_bg=big_bg,
                         big_mask=big_mask,
                         big_mask_NON_interacting=big_mask_NON_interacting,
                         big_img=big_img,
                         # the sample of the 4 latent variables
                         sample_c_map=c_distribution.sample,
                         sample_zwhere=zwhere_sample_few,
                         sample_zinstance=zinstance_few.sample,
                         sample_zbg=zbg.sample,
                         # the kl of the 4 latent variables
                         kl_logit=c_distribution.kl,
                         kl_zwhere=zwhere_kl_few,
                         kl_zinstance=zinstance_few.kl,
                         kl_zbg=zbg.kl,
                         # similarity kernels
                         similarity_l=similarity_l,
                         similarity_w=similarity_w)
