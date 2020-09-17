import torch
import torch.nn.functional as F
from .cropper_uncropper import Uncropper, Cropper
from .non_max_suppression import NonMaxSuppression
from .unet_model import UNet
from .encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, EncoderConvLeaky, DecoderConvLeaky
from .utilities import compute_average_in_box, compute_ranking
from .utilities import sample_and_kl_diagonal_normal, sample_and_kl_multivariate_normal
from .utilities import downsample_and_upsample
from .namedtuple import Inference, BB, NMSoutput, UNEToutput, ZZ, DIST


def squared_exp_kernel(locations: torch.Tensor, length_scale: float, eps: float = 1E-6):
    """ Input:
        locations.shape = (*, n, D)
        Output:
        C.shape = (*, n, n)
        where batched_dimension * might or might not be present
        It is necessary to add small positive shift on the diagonal otherwise Cholesky decomposition fails
    """
    loc = locations/length_scale  # *, n, D
    loc1 = loc.unsqueeze(-2)  # *,n,1,D
    loc2 = loc.unsqueeze(-3)  # *,1,n,D
    scaled_d2 = (loc1-loc2).pow(2).sum(dim=-1)  # *,n,n
    cov = torch.exp(-0.5 * scaled_d2)  # *,n,n . This covariance is between (0,1)

    # add STRICTLY POSITIVE shift on main diagonal to prevent matrix from becoming close to singular
    shift = eps * torch.eye(loc.shape[-2], dtype=cov.dtype, device=cov.device, requires_grad=False)  # n,n
    return (cov + shift).clamp(min=0)


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
    high_res_x = F.interpolate(low_res_x, size=high_resolution, mode='bilinear', align_corners=True)
    return high_res_x





class Inference_and_Generation(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        # variables
        self.size_max: int = params["input_image"]["size_object_max"]
        self.size_min: int = params["input_image"]["size_object_min"]
        self.cropped_size: int = params["architecture"]["cropped_size"]
        self.prior_L_cov = None

        self.length_scale_GP_raw = torch.nn.Parameter(data=torch.tensor(params["input_image"]["length_scale_GP"]),
                                                      requires_grad=True)

        # modules
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

        # ---------------------------#
        # 2. ZWHERE to BoundingBoxes
        # ---------------------------#
        zwhere_map: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zwhere.mu,
                                                         posterior_std=unet_output.zwhere.std,
                                                         prior_mu=torch.zeros_like(unet_output.zwhere.mu),
                                                         prior_std=torch.ones_like(unet_output.zwhere.std),
                                                         noisy_sampling=noisy_sampling,
                                                         sample_from_prior=generate_synthetic_data)

        tx_map, ty_map, tw_map, th_map = torch.split(torch.sigmoid(self.decoder_zwhere(zwhere_map.sample)), 1, dim=-3)

        # TODO make this a function and its inverse so that it is easier to add supervised loss
        #  tbb_2_bb(tx,ty,tw,th,raw_img_size,min_box_size,max_box_size) -> bx_map,by_map,bw_map,bh_map
        #  bb_2_tbb(bx,by,bw,bh,raw_img_size,min_box_size,max_box_size) -> tx_map,ty_map,tw_map,th_map
        with torch.no_grad():
            n_width, n_height = tx_map.shape[-2:]
            ix_array = torch.arange(start=0, end=n_width, dtype=tx_map.dtype, device=tx_map.device)
            iy_array = torch.arange(start=0, end=n_height, dtype=tx_map.dtype, device=tx_map.device)
            ix_grid, iy_grid = torch.meshgrid([ix_array, iy_array])
            pmap_points = torch.stack((ix_grid, iy_grid), dim=-1)  # n_width, n_height, 2

        bx_map: torch.Tensor = width_raw_image * (ix_grid + tx_map) / n_width
        by_map: torch.Tensor = height_raw_image * (iy_grid + ty_map) / n_height
        bw_map: torch.Tensor = self.size_min + (self.size_max - self.size_min) * tw_map
        bh_map: torch.Tensor = self.size_min + (self.size_max - self.size_min) * th_map
        area_map: torch.Tensor = bw_map * bh_map

        bounding_box_all: BB = BB(bx=convert_to_box_list(bx_map).squeeze(-1),
                                  by=convert_to_box_list(by_map).squeeze(-1),
                                  bw=convert_to_box_list(bw_map).squeeze(-1),
                                  bh=convert_to_box_list(bh_map).squeeze(-1))

        # ---------------------------#
        # 3. LOGIT to Probabilities #
        # ---------------------------#

        # Diagonalize the covariance matrix at each iteration since it depends on the tunable parameter length_scale_GP
        with torch.no_grad():
            scale_factor_x = float(imgs_in.shape[-2]) / unet_output.logit.mu.shape[-2]
            scale_factor_y = float(imgs_in.shape[-1]) / unet_output.logit.mu.shape[-1]
            scale_factor = torch.tensor([scale_factor_x, scale_factor_y], device=pmap_points.device).view(1, 1, 2)
            locations = (pmap_points * scale_factor).view(-1, 2).requires_grad_(False)

        length_scale_GP = F.softplus(self.length_scale_GP_raw)
        prior_covariance = squared_exp_kernel(locations=locations,
                                              length_scale=length_scale_GP,
                                              eps=1E-3)
        prior_L_cov = torch.cholesky(prior_covariance)

        posterior_mu = torch.flatten(unet_output.logit.mu, start_dim=1)  # batch_size, nx*ny
        posterior_L_cov = torch.diag_embed(torch.flatten(unet_output.logit.std, start_dim=1), dim1=-2, dim2=-1)

        logit_map: DIST = sample_and_kl_multivariate_normal(posterior_mu=posterior_mu,
                                                            posterior_L_cov=posterior_L_cov,
                                                            prior_mu=torch.zeros_like(posterior_mu, requires_grad=False),
                                                            prior_L_cov=prior_L_cov,
                                                            noisy_sampling=noisy_sampling,
                                                            sample_from_prior=generate_synthetic_data)

        p_map = torch.sigmoid(self.decoder_logit(logit_map.sample.view_as(unet_output.logit.mu)))

        # Correct the probability if necessary
        with torch.no_grad():

            # 1. correction if necessary
            if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0) and not generate_synthetic_data:

                # probability correction if necessary
                av_intensity: torch.Tensor = compute_average_in_box((imgs_in - big_bg).abs(), bounding_box_all)
                # av_intensity: torch.Tensor = compute_average_in_box((imgs_in - big_bg).pow(2), bounding_box_all)
                assert len(av_intensity.shape) == 2
                n_boxes_all, batch_size = av_intensity.shape
                ranking: torch.Tensor = compute_ranking(av_intensity)  # n_boxes_all, batch. It is in [0,n_box_all-1]
                tmp: torch.Tensor = ((ranking + 1).float() / (n_boxes_all + 1))
                tmp_2: torch.Tensor = invert_convert_to_box_list(tmp.unsqueeze(-1),
                                                                 original_width=p_map.shape[-2],
                                                                 original_height=p_map.shape[-1])
                p_approx: torch.Tensor = tmp_2.pow(10)
            else:
                prob_corr_factor = 0
                p_approx: torch.Tensor = torch.zeros_like(p_map)

        p_map_cor: torch.Tensor = (1 - prob_corr_factor) * p_map + prob_corr_factor * p_approx
        prob_all: torch.Tensor = convert_to_box_list(p_map_cor).squeeze(-1)
        assert bounding_box_all.bx.shape == prob_all.shape  # n_box_all, batch_size

        # ------------------------------------------------------------------#
        # 4. NMS and TOP-K to select few probabilities and bounding boxes
        # ------------------------------------------------------------------#
        with torch.no_grad():
            nms_output: NMSoutput = NonMaxSuppression.compute_mask_and_index(prob=prob_all,
                                                                             bounding_box=bounding_box_all,
                                                                             overlap_threshold=overlap_threshold,
                                                                             n_objects_max=n_objects_max,
                                                                             topk_only=topk_only)

        prob_few: torch.Tensor = torch.gather((prob_all * nms_output.nms_mask), dim=0, index=nms_output.index_top_k)
        bounding_box_few: BB = BB(bx=torch.gather(bounding_box_all.bx, dim=0, index=nms_output.index_top_k),
                                  by=torch.gather(bounding_box_all.by, dim=0, index=nms_output.index_top_k),
                                  bw=torch.gather(bounding_box_all.bw, dim=0, index=nms_output.index_top_k),
                                  bh=torch.gather(bounding_box_all.bh, dim=0, index=nms_output.index_top_k))

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
        return Inference(length_scale_GP=length_scale_GP.data.detach(),
                         p_map=p_map_cor,
                         area_map=area_map,
                         big_bg=big_bg,
                         big_mask=big_mask,
                         big_mask_NON_interacting=big_mask_NON_interacting,
                         big_img=big_img,
                         # the sample of the 3 latent variables
                         prob=prob_few,
                         bounding_box=bounding_box_few,
                         zinstance_each_obj=zinstance_few.sample,
                         # the kl of the 3 latent variables
                         kl_logit_map=logit_map.kl,
                         kl_zwhere_map=zwhere_map.kl,
                         kl_zinstance_each_obj=zinstance_few.kl)
