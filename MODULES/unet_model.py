import torch
from .unet_parts import DownBlock, DoubleConvolutionBlock, UpBlock
from .encoders_decoders import Encoder1by1, MLP_1by1, PredictBackground
import numpy as np
from collections import deque
from .namedtuple import UNEToutput


class UNet(torch.nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        # Parameters UNet
        self.n_max_pool = params["architecture"]["n_max_pool"]
        self.level_zwhere_and_logit_output = params["architecture"]["level_zwhere_and_logit_output"]
        self.level_background_output = params["architecture"]["level_background_output"]
        self.n_ch_output_features = params["architecture"]["n_ch_output_features"]
        self.ch_after_first_two_conv = params["architecture"]["n_ch_after_first_two_conv"]
        self.dim_zwhere = params["architecture"]["dim_zwhere"]
        self.dim_logit = params["architecture"]["dim_logit"]
        self.ch_raw_image = params["input_image"]["ch_in"]

        # Initializations
        ch = self.ch_after_first_two_conv
        j = 1
        self.j_list = [j]
        self.ch_list = [ch]

        # Down path to center
        self.down_path = torch.nn.ModuleList([DoubleConvolutionBlock(self.ch_raw_image, self.ch_list[-1])])
        for i in range(0, self.n_max_pool):
            j = j * 2
            ch = ch * 2
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.down_path.append(DownBlock(self.ch_list[-2], self.ch_list[-1]))

        # Up path
        self.up_path = torch.nn.ModuleList()
        for i in range(0, self.n_max_pool):
            j = int(j / 2)
            ch = int(ch / 2)
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.up_path.append(UpBlock(self.ch_list[-2], self.ch_list[-1]))

        # Compute s_p_k
        self.s_p_k = list()
        for module in self.down_path:
            self.s_p_k = module.__add_to_spk_list__(self.s_p_k)
        for module in self.up_path:
            self.s_p_k = module.__add_to_spk_list__(self.s_p_k)

        # Prediction maps
        self.pred_features = torch.nn.Sequential(torch.nn.ReLU(),
                                                 MLP_1by1(ch_in=self.ch_list[-1],
                                                          ch_out=self.n_ch_output_features,
                                                          ch_hidden=-1))

        self.encode_zwhere = Encoder1by1(ch_in=self.ch_list[-self.level_zwhere_and_logit_output - 1],
                                         dim_z=self.dim_zwhere)

        self.encode_logit = Encoder1by1(ch_in=self.ch_list[-self.level_zwhere_and_logit_output - 1],
                                        dim_z=self.dim_logit)

        # I don't need all the channels to predict the background. Few channels are enough
        self.ch_in_bg = min(5, self.ch_list[-self.level_background_output - 1])
        self.pred_background = torch.nn.Sequential(torch.nn.ReLU(),
                                                   PredictBackground(ch_in=self.ch_in_bg,
                                                                     ch_out=self.ch_raw_image,
                                                                     ch_hidden=-1))

    def forward(self, x: torch.Tensor, verbose: bool):
        input_w, input_h = x.shape[-2:]
        if verbose:
            print("INPUT ---> shape ", x.shape)

        # Down path and save the tensor which will need to be concatenated
        to_be_concatenated = deque()
        for i, down in enumerate(self.down_path):
            x = down(x, verbose)
            if verbose:
                print("down   ", i, " shape ", x.shape)
            if i < self.n_max_pool:
                to_be_concatenated.append(x)
                if verbose:
                    print("appended")

        # During up path I need to concatenate with the tensor obtained during the down path
        # If distance is < self.n_prediction_maps I need to export a prediction map
        zwhere, logit, zbg = None, None, None
        for i, up in enumerate(self.up_path):
            dist_to_end_of_net = self.n_max_pool - i
            if dist_to_end_of_net == self.level_zwhere_and_logit_output:
                zwhere = self.encode_zwhere(x)
                logit = self.encode_logit(x)
            if dist_to_end_of_net == self.level_background_output:
                zbg = self.pred_background(x[..., :self.ch_in_bg, :, :])  # only few channels needed for predicting bg

            x = up(to_be_concatenated.pop(), x, verbose)
            if verbose:
                print("up     ", i, " shape ", x.shape)

        # always add a pred_map to the rightmost layer (which had distance 0 from the end of the net)
        features = self.pred_features(x)

        return UNEToutput(zwhere=zwhere,
                          logit=logit,
                          zbg=zbg,
                          features=features)

    def show_grid(self, ref_image):

        assert len(ref_image.shape) == 4
        batch, ch, w_raw, h_raw = ref_image.shape

        nj = len(self.j_list)
        check_board = ref_image.new_zeros((nj, 1, 1, w_raw, h_raw))  # for each batch and channel the same check_board
        counter_w = torch.arange(w_raw)
        counter_h = torch.arange(h_raw)

        for k in range(nj):
            j = self.j_list[k]
            index_w = 1 + ((counter_w / j) % 2)  # either 1 or 2
            dx = index_w.float().view(w_raw, 1)
            index_h = 1 + ((counter_h / j) % 2)  # either 1 or 2
            dy = index_h.float().view(1, h_raw)
            check_board[k, 0, 0, :, :] = 0.25 * (dy * dx)  # dx*dy=1,2,4 multiply by 0.25 to have (0,1)

        assert check_board.shape == (nj, 1, 1, w_raw, h_raw)

        # I need to sum:
        # check_board of shape: --> levels, 1,      1, w_raw, h_raw
        # ref_image of shape ----->         batch, ch, w_raw, h_raw
        return ref_image + check_board

    def describe_receptive_field(self, image):
        """ Show the value of ch_w_h_j_rf_loc as the tensor moves thorugh the net.
            Here:
            a. w,h are the width and height
            b. j is grid spacing
            c. rf is the maximum theoretical receptive field
            d. wloc,hloc are the location of the center of the first cell
        """
        w, h = image.shape[-2:]
        j = 1
        rf = 1
        w_loc = 0.5
        h_loc = 0.5
        current_layer = (w, h, j, rf, w_loc, h_loc)
        i = -1
        for i in range(0, len(self.s_p_k)):
            print("At layer l= ", i, " we have w_h_j_rf_wloc_hloc= ", current_layer)
            current_layer = self.out_from_in(self.s_p_k[i], current_layer)
        print("At layer l= ", i + 1, " we have w_h_j_rf_wloc_hloc= ", current_layer)

    @staticmethod
    def out_from_in(s_p_k, layer_in):
        w_in, h_in, j_in, rf_in, wloc_in, hloc_in = layer_in
        s = s_p_k[0]
        p = s_p_k[1]
        k = s_p_k[2]

        w_out = np.floor((w_in - k + 2 * p) / s) + 1
        h_out = np.floor((h_in - k + 2 * p) / s) + 1

        pad_w = ((w_out - 1) * s - w_in + k) / 2
        pad_h = ((h_out - 1) * s - h_in + k) / 2

        j_out = j_in * s
        rf_out = rf_in + (k - 1) * j_in
        wloc_out = wloc_in + ((k - 1) / 2 - pad_w) * j_in
        hloc_out = hloc_in + ((k - 1) / 2 - pad_h) * j_in
        return int(w_out), int(h_out), j_out, int(rf_out), wloc_out, hloc_out
