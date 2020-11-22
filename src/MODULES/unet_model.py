import torch
from MODULES.unet_parts import DownBlock, DoubleConvolutionBlock, UpBlock
from MODULES.encoders_decoders import EncoderWhere, EncoderBackground, MLP_1by1
from collections import deque
from MODULES.namedtuple import UNEToutput


class UNet(torch.nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        # Parameters UNet
        self.n_max_pool = params["architecture"]["n_max_pool"]
        self.level_zwhere_and_logit_output = params["architecture"]["level_zwhere_and_logit_output"]
        self.level_background_output = params["architecture"]["level_background_output"]
        self.n_ch_output_features = params["architecture"]["n_ch_output_features"]
        self.ch_after_first_two_conv = params["architecture"]["n_ch_after_first_two_conv"]
        self.dim_zbg = params["architecture"]["dim_zbg"]
        self.dim_zwhere = params["architecture"]["dim_zwhere"]
        self.ch_raw_image = params["input_image"]["ch_in"]
        self.concatenate_raw_image_to_fmap = params["architecture"]["concatenate_raw_image_to_fmap"]
        self.crop_raw_image = params["architecture"]["crop_raw_image"]
        if self.crop_raw_image:
            assert self.ch_raw_image == self.n_ch_output_features

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
            j = int(j // 2)
            ch = int(ch // 2)
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.up_path.append(UpBlock(self.ch_list[-2], self.ch_list[-1]))

        # Prediction maps
        if not self.crop_raw_image:
            ch_out_fmap = self.n_ch_output_features - \
                          self.ch_raw_image if self.concatenate_raw_image_to_fmap else self.n_ch_output_features
            self.predict_features = MLP_1by1(ch_in=self.ch_list[-1],
                                             ch_out=ch_out_fmap,
                                             ch_hidden=-1)  # this means there is NO hidden layer
        else:
            self.predict_features = torch.nn.Identity()

        # Predict the logit
        self.ch_in_logit = self.ch_list[-self.level_zwhere_and_logit_output - 1]
        self.predict_logit = MLP_1by1(ch_in=self.ch_in_logit,
                                      ch_out=1,
                                      ch_hidden=self.ch_in_logit//2)

        # Encode zwhere in mu and std
        self.ch_in_zwhere = self.ch_list[-self.level_zwhere_and_logit_output - 1]
        self.encode_zwhere = EncoderWhere(ch_in=self.ch_in_zwhere,
                                          dim_z=self.dim_zwhere,
                                          ch_hidden=self.ch_in_zwhere//2)

        # Encode zbg in mu and std
        self.ch_in_bg = self.ch_list[-self.level_background_output - 1]
        self.encode_background = EncoderBackground(ch_in=self.ch_in_bg,
                                                   dim_z=self.dim_zbg)

    def forward(self, x: torch.Tensor, verbose: bool):
        # input_w, input_h = x.shape[-2:]
        if verbose:
            print("INPUT ---> shape ", x.shape)

        # Down path and save the tensor which will need to be concatenated
        raw_image = x
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
                logit = self.predict_logit(x)
            if dist_to_end_of_net == self.level_background_output:
                zbg = self.encode_background(x)  # only few channels needed for predicting bg

            x = up(to_be_concatenated.pop(), x, verbose)
            if verbose:
                print("up     ", i, " shape ", x.shape)

        # always add a pred_map to the rightmost layer (which had distance 0 from the end of the net)
        if self.crop_raw_image:
            features = raw_image
        elif self.concatenate_raw_image_to_fmap:
            features = torch.cat((self.predict_features(x), raw_image), dim=-3)  # Here I am concatenating the raw image
        else:
            features = self.predict_features(x)

        return UNEToutput(zwhere=zwhere,
                          logit=logit,
                          zbg=zbg,
                          features=features)

    def show_grid(self, ref_image):
        """ overimpose a grid the size of the corresponding resolution of each unet layer """

        assert len(ref_image.shape) == 4
        batch, ch, w_raw, h_raw = ref_image.shape

        nj = len(self.j_list)
        check_board = ref_image.new_zeros((nj, 1, 1, w_raw, h_raw))  # for each batch and channel the same check_board
        counter_w = torch.arange(w_raw)
        counter_h = torch.arange(h_raw)

        for k in range(nj):
            j = self.j_list[k]
            index_w = 1 + ((counter_w // j) % 2)  # either 1 or 2
            dx = index_w.float().view(w_raw, 1)
            index_h = 1 + ((counter_h // j) % 2)  # either 1 or 2
            dy = index_h.float().view(1, h_raw)
            check_board[k, 0, 0, :, :] = 0.25 * (dy * dx)  # dx*dy=1,2,4 multiply by 0.25 to have (0,1)

        assert check_board.shape == (nj, 1, 1, w_raw, h_raw)

        # I need to sum:
        # check_board of shape: --> levels, 1,      1, w_raw, h_raw
        # ref_image of shape ----->         batch, ch, w_raw, h_raw
        return ref_image + check_board

