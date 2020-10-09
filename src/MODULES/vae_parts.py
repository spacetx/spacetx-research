import torch
import torch.nn.functional as F
from cropper_uncropper import Uncropper, Cropper
from non_max_suppression import NonMaxSuppression
from unet_model import UNet
from encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, EncoderConvLeaky, DecoderConvLeaky
from utilities import compute_average_in_box, compute_ranking
from utilities_ml import sample_and_kl_diagonal_normal, SimilarityKernel, FiniteDPP #, sample_and_kl_multivariate_normal
from namedtuple import Inference, BB, NMSoutput, UNEToutput, ZZ, DIST


class PassBernoulli(torch.autograd.Function):
    """ Forward is c=Bernoulli(p). Backward is identity"""
    @staticmethod
    def forward(ctx, p):
        c = torch.rand_like(p) < p
        return c
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def pass_bernoulli(prob):
    return PassBernoulli.apply(prob)


class PassMask(torch.autograd.Function):
    """ Forward is masking, Backward is identity"""
    @staticmethod
    def forward(ctx, c, nms_mask):
        return c * nms_mask
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def pass_mask(c, mask):
    return PassMask.apply(c, mask)


def convert_to_box_list(x: torch.Tensor) -> torch.Tensor:
    """ takes input of shape: (batch, ch, width, height)
        and returns output of shape: (n_list, batch, ch)
        where n_list = width x height
    """
    assert len(x.shape) == 4
    batch_size, ch, width, height = x.shape
    return x.permute(2, 3, 0, 1).view(width*height, batch_size, ch)


def invert_convert_to_box_list(x: torch.Tensor, original_width: int, original_height: int) -> torch.Tensor:
    """ takes input of shape: (width x height, batch, ch)
        and return shape: (batch, ch, width, height)
    """
    assert len(x.shape) == 3
    n_list, batch_size, ch = x.shape
    assert n_list == original_width * original_height
    return x.permute(1, 2, 0).view(batch_size, ch, original_width, original_height)


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
        self.similarity_kernel_dpp = SimilarityKernel(n_kernels=params["similarity"]["n_kernels"])
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

        # Correction factor
        if not generate_synthetic_data:

            # Bounding_box_no_noise are used to NMS and prob_correction_factor
            with torch.no_grad():
                bounding_box_no_noise: BB = tmaps_to_bb(self.decoder_zwhere(unet_output.zwhere.mu))

            if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):
                with torch.no_grad():
                    av_intensity: torch.Tensor = compute_average_in_box((imgs_in - big_bg).abs(), bounding_box_no_noise)
                    # av_intensity: torch.Tensor = compute_average_in_box((imgs_in - big_bg).pow(2), bounding_box_no_noise)
                    assert len(av_intensity.shape) == 2
                    n_boxes_all, batch_size = av_intensity.shape
                    ranking = compute_ranking(av_intensity)  # n_boxes_all, batch. It is in [0,n_box_all-1]
                    tmp = ((ranking + 1).float() / (n_boxes_all + 1))
                    q_approx = tmp.pow(10)

                q_uncorrected = convert_to_box_list(torch.sigmoid(unet_output.logit)).unsqueeze(-1)
                q = ((1 - prob_corr_factor) * q_uncorrected + prob_corr_factor * q_approx).clamp(min=1E-4, max=1-1E-4)
                log_q = torch.log(q)
                log_one_minus_q = torch.log1p(-q)
            else:
                logit_reshaped = convert_to_box_list(unet_output.logit).unsqueeze(-1)
                q = torch.sigmoid(logit_reshaped)
                log_q = F.logsigmoid(logit_reshaped)
                log_one_minus_q = F.logsigmoid(-logit_reshaped)

            # sample, NMS, log_prob
            c = pass_bernoulli(prob=q)  # forward: c=Bernoulli(prob), Backward=Identity
            with torch.no_grad():
                nms_output: NMSoutput = NonMaxSuppression.compute_mask_and_index(prob=q,
                                                                                 bounding_box=bounding_box_no_noise,
                                                                                 overlap_threshold=overlap_threshold,
                                                                                 n_objects_max=n_objects_max,
                                                                                 topk_only=topk_only,
                                                                                 active=c) # only if active apply NMS

            print("nms_output.nms_mask.shape", nms_output.nms_mask.shape)
            c_masked = pass_mask(c, nms_output.nms_mask)  # I might suppress some of the c
            print("c_masked.shape", c_masked.shape)
            w_logit, h_logit = unet_output.logit.shape[-2:]
            log_prob_prior = FiniteDPP(L=self.similarity_kernel_dpp.forward(n_width=w_logit,
                                                                            n_height=h_logit)).log_prob(c_masked)
            log_prob_posterior = (c_masked * log_q + ~c_masked * log_one_minus_q).sum(dim=0)
            print("log_prob_prior.shape", log_prob_prior.shape)
            print("log_prob_posterior.shape", log_prob_posterior.shape)
            kl_prob = log_prob_posterior - log_prob_prior

        else:
            FROM HERE
            # Here I am generating synthetic data



        # SAMPLE ZWHERE OF SELECTED BOXES
        # COMPOUTE KL_ZWHERE OF SELECTED BOXES

        with torch.no_grad():
            bounding_box_all: BB = tmaps_to_bb(self.decoder_zwhere(unet_output.zwhere.mu))



    # ---------------------------#
    # 2. ZWHERE to BoundingBoxes
    # ---------------------------#
        zwhere_map: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zwhere.mu,
                                                         posterior_std=unet_output.zwhere.std,
                                                         prior_mu=torch.zeros_like(unet_output.zwhere.mu),
                                                         prior_std=torch.ones_like(unet_output.zwhere.std),
                                                         noisy_sampling=noisy_sampling,
                                                         sample_from_prior=generate_synthetic_data)

        prob_few: torch.Tensor = torch.gather((prob_all * nms_output.nms_mask), dim=0, index=nms_output.index_top_k)

        bounding_box_few: BB = BB(bx=torch.gather(bounding_box_all.bx, dim=0, index=nms_output.index_top_k),
                                  by=torch.gather(bounding_box_all.by, dim=0, index=nms_output.index_top_k),
                                  bw=torch.gather(bounding_box_all.bw, dim=0, index=nms_output.index_top_k),
                                  bh=torch.gather(bounding_box_all.bh, dim=0, index=nms_output.index_top_k))

        zwhere_kl_all = convert_to_box_list(zwhere_map.kl)  # shape (nbox_all, batch_size, ch)
        new_index = nms_output.index_top_k.unsqueeze(-1).expand(-1, -1,
                                                                zwhere_kl_all.shape[-1])  # (nbox_few, batch_size, ch)
        zwhere_kl_few = torch.gather(zwhere_kl_all, dim=0, index=new_index)  # shape (nbox_few, batch_size, ch)

        # ------------------------------------------------------------------#
        # 5. Crop the unet_features according to the selected boxes
        # ------------------------------------------------------------------#
        n_boxes, batch_size = bounding_box_few.bx.shape
        # print(unet_output.features.shape)
        # print(imgs_in.shape)
        # append the raw image in the channel dimension. 
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
        return Inference(length_scale_GP=length_scale_GP,
                         p_map=p_map_cor,
                         area_map=area_map,
                         big_bg=big_bg,
                         big_mask=big_mask,
                         big_mask_NON_interacting=big_mask_NON_interacting,
                         big_img=big_img,
                         # the sample of the 3 latent variables
                         sample_prob=prob_few,
                         sample_bb=bounding_box_few,
                         sample_zinstance=zinstance_few.sample,
                         # the kl of the 3 latent variables
                         kl_logit_map=logit_map.kl,
                         kl_zwhere_map=zwhere_map.kl,
                         kl_zwhere=zwhere_kl_few,
                         kl_zinstance=zinstance_few.kl)
