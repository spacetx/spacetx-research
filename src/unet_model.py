import torch
from collections import deque
from .unet_parts import DownBlock, DoubleConvolutionBlock, UpBlock
from .encoders_decoders import EncoderWhereLogit, EncoderBackground, MLP_1by1
from .namedtuple import UNEToutput


class PreProcessor(torch.nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.ch_raw_image = params["input_image"]["ch_in"]
        self.n_ch_after_preprocessing = params["architecture"]["n_ch_after_preprocessing"]
        self.preprocessor = DoubleConvolutionBlock(self.ch_raw_image, self.n_ch_after_preprocessing)

    def forward(self, x: torch.Tensor, verbose: bool) -> torch.Tensor:
        y = self.preprocessor.forward(x)
        if verbose:
            print("preprocessor input_shape -->", x.shape)
            print("preprocessor output_shape ->", y.shape)
        return y


class UNet(torch.nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        # Parameters UNet
        self.n_max_pool = params["architecture"]["n_max_pool_unet"]
        self.level_zwhere_and_logit_output = params["architecture"]["level_zwherelogit_unet"]
        self.n_ch_output_features = params["architecture"]["n_ch_output_features"]
        self.n_ch_after_preprocessing = params["architecture"]["n_ch_after_preprocessing"]
        self.dim_zwhere = params["architecture"]["dim_zwhere"]

        # Initializations
        ch = self.n_ch_after_preprocessing
        j = 1
        down_j_list = [j]
        down_ch_list = [ch]

        # Down path to center
        self.down_path = torch.nn.ModuleList()
        for i in range(0, self.n_max_pool):
            j = j * 2
            ch = ch * 2
            down_ch_list.append(ch)
            down_j_list.append(j)
            self.down_path.append(DownBlock(down_ch_list[-2], down_ch_list[-1]))

        print("down j_list  ->", down_j_list)
        print("down ch_list ->", down_ch_list)

        up_j_list = down_j_list[::-1]
        up_ch_list = down_ch_list[::-1]
        up_ch_list[-1] = self.n_ch_output_features
        print("up j_list  ->", up_j_list)
        print("up ch_list ->", up_ch_list)





        # Up path
        self.up_path = torch.nn.ModuleList()
        for i in range(0, self.n_max_pool):
            j = int(j // 2)
            ch = int(ch // 2)
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.up_path.append(UpBlock(self.ch_list[-2], self.ch_list[-1]))
        print("up j_list -->", self.j_list)
        print("up ch_list ->", self.ch_list)


        # Prediction maps
        self.predict_features = MLP_1by1(ch_in=self.ch_list[-1],
                                         ch_out=self.n_ch_output_features,
                                         ch_hidden=-1)  # this means there is NO hidden layer

        # Encode zwhere and logit in mu and std
        self.ch_in_zwhere = self.ch_list[-self.level_zwhere_and_logit_output - 1]
        self.encode_zwherelogit = EncoderWhereLogit(ch_in=self.ch_in_zwhere,
                                                    dim_z=self.dim_zwhere,
                                                    dim_logit=1,
                                                    ch_hidden=int((self.ch_in_zwhere+self.dim_zwhere)//2))

        # Encode zbg in mu and std
        self.ch_in_bg = self.ch_list[-self.level_background_output - 1]
        self.encode_background = EncoderBackground(ch_in=self.ch_in_bg,
                                                   dim_z=self.dim_zbg)

    def forward(self, x: torch.Tensor, verbose: bool):
        # input_w, input_h = x.shape[-2:]
        if verbose:
            print("INPUT ---> shape ", x.shape)

        # Down path and save the tensor which will need to be concatenated
        to_be_concatenated = deque()
        for i, down in enumerate(self.down_path):
            to_be_concatenated.append(x)  # save before applying max_pool
            x = down(x, verbose)
            if verbose:
                print("down   ", i, " shape ", x.shape)

        # At the bottom of Unet extract the background
        zbg = self.encode_background(x)


        # During up path I need to concatenate with the tensor obtained during the down path
        # If distance is < self.n_prediction_maps I need to export a prediction map
        zwhere, logit = None, None, None
        for i, up in enumerate(self.up_path):
            dist_to_end_of_net = self.n_max_pool - i
            if dist_to_end_of_net == self.level_zwhere_and_logit_output:
                zwhere, logit = self.encode_zwherelogit(x)
                if verbose:
                    print("extracting zwhere and logit", x.shape)

            x = up(to_be_concatenated.pop(), x, verbose)
            if verbose:
                print("up     ", i, " shape ", x.shape)

        # At the top of unet extract the features
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

