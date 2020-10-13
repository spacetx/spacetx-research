import torch
import torch.nn.functional as F

from MODULES.cropper_uncropper import Uncropper, Cropper
from MODULES.unet_model import UNet
from MODULES.encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, EncoderConvLeaky, DecoderConvLeaky
from MODULES.utilities import tmaps_to_bb, convert_to_box_list
from MODULES.utilities_ml import sample_and_kl_diagonal_normal, sample_and_kl_prob, SimilarityKernel
from MODULES.namedtuple import Inference, NMSoutput, BB, UNEToutput, ZZ, DIST


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


def downsample_and_upsample(x: torch.Tensor, low_resolution: tuple, high_resolution: tuple):
    low_res_x = F.adaptive_avg_pool2d(x, output_size=low_resolution)
    # low_res_x = F.adaptive_max_pool2d(x, output_size=low_resolution)
    high_res_x = F.interpolate(low_res_x, size=high_resolution, mode='bilinear', align_corners=True)
    return high_res_x


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
        self.decoder_zwhere: Decoder1by1Linear = Decoder1by1Linear(dim_z=params["architecture"]["dim_zwhere"],
                                                                   ch_out=4,
                                                                   groups=params["architecture"]["dim_zwhere"])

        self.decoder_logit: Decoder1by1Linear = Decoder1by1Linear(dim_z=params["architecture"]["dim_logit"],
                                                                  ch_out=1,
                                                                  groups=1)

        leaky = False
        if leaky:
            self.decoder_zinstance: DecoderConvLeaky = DecoderConvLeaky(size=params["architecture"]["cropped_size"],
                                                                        dim_z=params["architecture"]["dim_zinstance"],
                                                                        ch_out=params["input_image"]["ch_in"] + 1)

            # encoder z_mask (takes the unet_features)
            self.encoder_zinstance: EncoderConvLeaky = EncoderConvLeaky(size=params["architecture"]["cropped_size"],
                                                                        ch_in=params["architecture"]["n_ch_output_features"],
                                                                        dim_z=params["architecture"]["dim_zinstance"])
        else:
            self.decoder_zinstance: DecoderConv = DecoderConv(size=params["architecture"]["cropped_size"],
                                                              dim_z=params["architecture"]["dim_zinstance"],
                                                              ch_out=params["input_image"]["ch_in"] + 1)

            # encoder z_mask (takes the unet_features)
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
        if bg_is_zero:
            big_bg = torch.zeros_like(imgs_in)
        else:
            # I could also sample
            # bg = unet_output.zbg.mu + eps * unet_output.zbg.std
            bg_map = downsample_and_upsample(unet_output.zbg.mu,
                                             low_resolution=bg_resolution,
                                             high_resolution=(imgs_in.shape[-2], imgs_in.shape[-1]))
            big_bg = torch.sigmoid(bg_map)

        with torch.no_grad():
            bounding_box_no_noise: BB = tmaps_to_bb(tmaps=torch.sigmoid(self.decoder_zwhere(unet_output.zwhere.mu)),
                                                    width_raw_image=width_raw_image,
                                                    height_raw_image=height_raw_image,
                                                    min_box_size=self.size_min,
                                                    max_box_size=self.size_max)

        similarity_kernel = self.similarity_kernel_dpp.forward(n_width=unet_output.logit.mu.shape[-2],
                                                               n_height=unet_output.logit.mu.shape[-1])

        c_map: DIST
        nms_output: NMSoutput
        c_map, nms_output = sample_and_kl_prob(logit_map=unet_output.logit.mu,
                                               similarity_kernel=similarity_kernel,
                                               images=imgs_in,
                                               background=big_bg,
                                               bounding_box_no_noise=bounding_box_no_noise,
                                               prob_corr_factor=prob_corr_factor,
                                               overlap_threshold=overlap_threshold,
                                               n_objects_max=n_objects_max,
                                               topk_only=topk_only,
                                               noisy_sampling=noisy_sampling,
                                               sample_from_prior=generate_synthetic_data)
        c_few = torch.gather(convert_to_box_list(c_map.sample).squeeze(-1), dim=0, index=nms_output.index_top_k)
        logit_few = torch.gather(convert_to_box_list(unet_output.logit.mu).squeeze(-1), dim=0,
                                 index=nms_output.index_top_k)
        print("AFTER NMS")
        print(c_few.shape)
        print(logit_few.shape)
        print(c_map.sample.shape)
        print(c_map.kl.shape)
        print(nms_output.nms_mask.shape)
        print(nms_output.index_top_k.shape)

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

        bounding_box_few: BB = BB(bx=torch.gather(bounding_box_all.bx, dim=0, index=nms_output.index_top_k),
                                  by=torch.gather(bounding_box_all.by, dim=0, index=nms_output.index_top_k),
                                  bw=torch.gather(bounding_box_all.bw, dim=0, index=nms_output.index_top_k),
                                  bh=torch.gather(bounding_box_all.bh, dim=0, index=nms_output.index_top_k))

        zwhere_kl_all = convert_to_box_list(zwhere_map.kl)  # shape: nbox_all, batch_size, ch
        new_index = nms_output.index_top_k.unsqueeze(-1).expand(-1, -1, zwhere_kl_all.shape[-1])  # shape: nbox_few, batch_size, ch
        zwhere_kl_few = torch.gather(zwhere_kl_all, dim=0, index=new_index)  # shape (nbox_few, batch_size, ch)

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
                                     small_stuff=small_stuff,
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
        return Inference(logit_map=unet_output.logit.mu,
                         logit_few=logit_few,
                         big_bg=big_bg,
                         big_mask=big_mask,
                         big_mask_NON_interacting=big_mask_NON_interacting,
                         big_img=big_img,
                         # the sample of the 3 latent variables
                         sample_c_map=c_map.sample,
                         sample_c=c_few,
                         sample_bb=bounding_box_few,
                         sample_zinstance=zinstance_few.sample,
                         # the kl of the 3 latent variables
                         kl_logit=c_map.kl,
                         kl_zwhere=zwhere_kl_few,
                         kl_zinstance=zinstance_few.kl)
