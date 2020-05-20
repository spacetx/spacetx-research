import torch
import torch.nn.functional as F
from .cropper_uncropper import Uncropper, Cropper
from .non_max_suppression import NonMaxSuppression
from .unet_model import UNet
from .encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear # DecoderZwhere # Encoder1by1, Decoder1by1
from .utilities import compute_average_intensity_in_box, compute_ranking
from .utilities import sample_and_kl_diagonal_normal, sample_and_kl_multivariate_normal
from .utilities import downsample_and_upsample
from .namedtuple import Inference, BB, NMSoutput, UNEToutput, ZZ, DIST


def squared_exp_kernel(points1: torch.Tensor, points2: torch.Tensor, length_scale: float, eps: float = 1E-6):
    """ Input:
        points1.shape = (*, n, D)
        points2.shape = (*, m, D)
        Output: 
        C.shape = (*, n, m)
        where batched_dimension * might or might not be present
        It is necessary to add small positive shift on the diagonal otherwise Cholesky decomposition fails
    """
    dim1 = points1.shape[-1]
    dim2 = points2.shape[-1]
    assert dim1 == dim2
    points1 = points1.unsqueeze(-2)/length_scale  # *, n, 1, D
    points2 = points2.unsqueeze(-3)/length_scale  # *, 1, m, D
    scaled_d2 = (points1-points2).pow(2).sum(dim=-1)  # *, n, m
    cov = torch.exp(-0.5 * scaled_d2)  # *, n, m . This covariance is between (0,1)

    # add STRICTLY POSITIVE noise on the diagonal to prevent matrix from becoming close to singular
    diag_shift = eps * torch.ones_like(cov[..., 0])  # *, n
    return (cov + torch.diag_embed(diag_shift, dim1=-2, dim2=-1)).clamp(min=0)


def flatten_by_batch(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 4
    batch_size = x.shape[0]
    return x.view(batch_size, -1)


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
    low_res_x = F.interpolate(x, size=low_resolution, mode='bilinear', align_corners=True)
    high_res_x = F.interpolate(low_res_x, size=high_resolution, mode='bilinear', align_corners=True)
    return high_res_x


class Moving_Average_Calculator:
    """ beta is the factor multiplying the moving average.
        Approximately we average the last 1/(1-beta) points.
        For example:
        beta = 0.9 -> 10 points
        beta = 0.99 -> 100 points
        The larger beta the longer the time average.

        Usage:
        MA = Moving_Average_Calculator(beta = 0.99)
        input_dict = { "x" : 100+i+np.random.randn(),
                   "y" : 50+i+np.random.randn()}
        MA(input_dict)
    """

    def __init__(self, beta):
        super().__init__()
        self._bias = None
        self._steps = 0
        self._beta = beta
        self._dict_accumulate = {}
        self._dict_MA = {}
        # print("initialization empty. Step ->", self._steps)

    def accumulate(self, input_dict):
        self._steps += 1
        self._bias = 1 - self._beta ** self._steps
        # print("Mopving_Average_Calculator step", self._steps)

        for key, value in input_dict.items():
            try:
                tmp = self._beta * self._dict_accumulate[key] + (1 - self._beta) * value
                self._dict_accumulate[key] = tmp
            except KeyError:
                self._dict_accumulate[key] = (1 - self._beta) * value
            self._dict_MA[key] = self._dict_accumulate[key] / self._bias
        return self._dict_MA


class Inference_and_Generation(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        # variables
        self.size_max: int = params["input_image"]["size_object_max"]
        self.size_min: int = params["input_image"]["size_object_min"]
        self.cropped_size: int = params["architecture"]["cropped_size"]
        self.lenght_scale_prior: float = params["input_image"]["sigma_squared_exp_kernel_porb_map"]
        self.prior_L_cov = None

        # modules
        self.unet: UNet = UNet(params)

        # encoder z_what (takes the raw image)
        self.encoder_zwhat: EncoderConv = EncoderConv(size=params["architecture"]["cropped_size"],
                                                      ch_in=params["input_image"]["ch_in"],
                                                      dim_z=params["architecture"]["dim_zwhat"])

        # encoder z_mask (takes the unet_features)
        self.encoder_zmask: EncoderConv = EncoderConv(size=params["architecture"]["cropped_size"],
                                                      ch_in=params["architecture"]["n_ch_output_features"],
                                                      dim_z=params["architecture"]["dim_zmask"])

        # Decoders
        self.decoder_zwhere: Decoder1by1Linear = Decoder1by1Linear(dim_z=params["architecture"]["dim_zwhere"],
                                                                   ch_out=4)

        self.decoder_logit: Decoder1by1Linear = Decoder1by1Linear(dim_z=params["architecture"]["dim_logit"],
                                                                  ch_out=1)

        self.decoder_mask: DecoderConv = DecoderConv(size=params["architecture"]["cropped_size"],
                                                     dim_z=params["architecture"]["dim_zmask"],
                                                     ch_out=1)
        self.decoder_imgs: DecoderConv = DecoderConv(size=params["architecture"]["cropped_size"],
                                                     dim_z=params["architecture"]["dim_zwhat"],
                                                     ch_out=params["input_image"]["ch_in"])

    def forward(self, imgs_in: torch.Tensor,
                generate_synthetic_data: bool,
                prob_corr_factor: float,
                overlap_threshold: float,
                score_threshold: float,
                randomize_nms_factor: float,
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
            bg_mu = torch.zeros_like(imgs_in)
        else:
            bg_mu = downsample_and_upsample(unet_output.bg_mu,
                                            low_resolution=bg_resolution,
                                            high_resolution=(imgs_in.shape[-2], imgs_in.shape[-1]))

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
        posterior_mu = flatten_by_batch(unet_output.logit.mu)
        posterior_L_cov = torch.diag_embed(flatten_by_batch(unet_output.logit.std), dim1=-2, dim2=-1)

        # Diagonalize the covaraince matrix if necessary
        if self.prior_L_cov is None or (self.prior_L_cov.shape[-1] != posterior_mu.shape[-1]):

            length_scale = self.lenght_scale_prior * float(unet_output.logit.mu.shape[-1]) / imgs_in.shape[-1]
            prior_covariance = squared_exp_kernel(points1=pmap_points.view(-1, 2),
                                                  points2=pmap_points.view(-1, 2),
                                                  length_scale=length_scale,
                                                  eps=1E-3)
            self.prior_L_cov = torch.cholesky(prior_covariance)

        logit_map: DIST = sample_and_kl_multivariate_normal(posterior_mu=posterior_mu,
                                                            posterior_L_cov=posterior_L_cov,
                                                            prior_mu=torch.zeros_like(posterior_mu).detach(),
                                                            prior_L_cov=self.prior_L_cov.detach(),
                                                            noisy_sampling=noisy_sampling,
                                                            sample_from_prior=generate_synthetic_data)

        p_map = torch.sigmoid(self.decoder_logit(logit_map.sample.view_as(unet_output.logit.mu)))

        # Add probability correction if necessary
        if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0) and not generate_synthetic_data:
            with torch.no_grad():
                av_intensity: torch.Tensor = compute_average_intensity_in_box(torch.abs(imgs_in - bg_mu),
                                                                              bounding_box_all)
                assert len(av_intensity.shape) == 2
                n_boxes_all, batch_size = av_intensity.shape
                ranking: torch.Tensor = compute_ranking(av_intensity)  # n_boxes_all, batch. It is in [0,n_box_all-1]
                tmp: torch.Tensor = ((ranking + 1).float() / (n_boxes_all + 1))
                tmp_2: torch.Tensor = invert_convert_to_box_list(tmp.unsqueeze(-1),
                                                                 original_width=p_map.shape[-2],
                                                                 original_height=p_map.shape[-1])
                p_approx: torch.Tensor = tmp_2.pow(10)
            # weighted average of the prob by the inference network and prob by correction
            p_map_cor: torch.Tensor = (1 - prob_corr_factor) * p_map + prob_corr_factor * p_approx
        else:
            p_map_cor: torch.Tensor = p_map

        prob_all: torch.Tensor = convert_to_box_list(p_map_cor).squeeze(-1)
        assert bounding_box_all.bx.shape == prob_all.shape  # n_box_all, batch_size

        # ------------------------------------------------------------------#
        # 4. NMS and TOP-K to select few probabilities and bounding boxes
        # ------------------------------------------------------------------#
        with torch.no_grad():
            nms_output: NMSoutput = NonMaxSuppression.compute_mask_and_index(prob=prob_all,
                                                                             bounding_box=bounding_box_all,
                                                                             score_threshold=score_threshold,
                                                                             overlap_threshold=overlap_threshold,
                                                                             randomize_nms_factor=randomize_nms_factor,
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
        unet_features_expanded = unet_output.features.unsqueeze(0).expand(n_boxes, batch_size, -1, -1, -1)
        cropped_feature_map: torch.Tensor = Cropper.crop(bounding_box=bounding_box_few,
                                                         big_stuff=unet_features_expanded,
                                                         width_small=self.cropped_size,
                                                         height_small=self.cropped_size)

        # ------------------------------------------------------------------#
        # 6. Encode, sample zmask and decode to BIG MASKS
        # ------------------------------------------------------------------#
        zmask: ZZ = self.encoder_zmask.forward(cropped_feature_map)
        zmask_few: DIST = sample_and_kl_diagonal_normal(posterior_mu=zmask.mu,
                                                        posterior_std=zmask.std,
                                                        prior_mu=torch.zeros_like(zmask.mu),
                                                        prior_std=torch.ones_like(zmask.std),
                                                        noisy_sampling=noisy_sampling,
                                                        sample_from_prior=generate_synthetic_data)

        # multiply by prob? prob_detached?
        small_weight = F.softplus(self.decoder_mask.forward(zmask_few.sample))
        big_weight = Uncropper.uncrop(bounding_box=bounding_box_few,
                                      small_stuff=small_weight,
                                      width_big=width_raw_image,
                                      height_big=height_raw_image)
        # big_mask = from_weights_to_masks(weight=prob_times_big_weight, dim=-5)
        big_mask = from_weights_to_masks(weight=big_weight, dim=-5)

        # ------------------------------------------------------------------#
        # 7. mask the raw image and crop it
        # should mask be detached here?
        # ------------------------------------------------------------------#
        cropped_img = Cropper.crop(bounding_box=bounding_box_few,
                                   big_stuff=imgs_in * big_mask,  # should mask be detached here?
                                   width_small=self.cropped_size,
                                   height_small=self.cropped_size)

        # ------------------------------------------------#
        # 8. Encode, sample z_what and decode to BIG IMGS
        # ------------------------------------------------#
        zwhat: ZZ = self.encoder_zwhat.forward(cropped_img)
        zwhat_few: DIST = sample_and_kl_diagonal_normal(posterior_mu=zwhat.mu,
                                                        posterior_std=zwhat.std,
                                                        prior_mu=torch.zeros_like(zwhat.mu),
                                                        prior_std=torch.ones_like(zwhat.std),
                                                        noisy_sampling=noisy_sampling,
                                                        sample_from_prior=generate_synthetic_data)
        small_img = torch.sigmoid(self.decoder_imgs.forward(zwhat_few.sample))
        big_img: torch.Tensor = Uncropper.uncrop(bounding_box=bounding_box_few,
                                                 small_stuff=small_img,
                                                 width_big=width_raw_image,
                                                 height_big=height_raw_image)

        # 9. Return the inferred quantities
        return Inference(bg_mu=bg_mu,
                         p_map=p_map_cor,
                         area_map=area_map,
                         big_mask=big_mask,
                         big_img=big_img,
                         prob=prob_few,
                         bounding_box=bounding_box_few,
                         kl_logit_map=logit_map.kl,
                         kl_zwhere_map=zwhere_map.kl,
                         kl_zwhat_each_obj=zwhat_few.kl,
                         kl_zmask_each_obj=zmask_few.kl)
