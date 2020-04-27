import torch
from .encoders_decoders import EncoderConv
from .cropper_uncropper import Uncropper, Cropper
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import collections

BB = collections.namedtuple("bounding_box", "bx by bw bh")


class CenterCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, width, height):
        w_current, h_current = x.shape[-2:]
        if (w_current < width) or (h_current < height):
            raise Exception
        else:
            dw1 = int(0.5*(w_current - width))
            dw2 = w_current - width - dw1
            assert dw1 == dw2

            dh1 = int(0.5*(h_current - height))
            dh2 = h_current - height - dh1
            assert dh1 == dh2

            #print(dw1, dw2, width, w_current)
            #print(dh1, dh2, height, h_current)

            return x[..., dw1:-dw2, dh1:-dh2]


class ComputeKQV(torch.nn.Module):

    FILTER_SIZE = 5

    def __init__(self, params, ch_in=None):
        super().__init__()
        self.ch_in = ch_in
        self.ch_v = params["interaction"]["ch_v"]
        self.ch_k_and_q = params["interaction"]["ch_k_and_q"]
        self.ch_out = 2 * self.ch_k_and_q + self.ch_v
        self.ch_hidden = int(0.5*(self.ch_in+self.ch_out))
        self.shallow_cnn = nn.Sequential(
            nn.Conv2d(self.ch_in, self.ch_hidden, self.FILTER_SIZE, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ch_hidden, self.ch_hidden, self.FILTER_SIZE, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ch_hidden, self.ch_out, self.FILTER_SIZE, bias=True),
        )

    def forward(self, x):
        return self.shallow_cnn(x)


class Interaction(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.cropper = Cropper()
        self.uncropper = Uncropper()
        self.cnn_kqv = ComputeKQV(params, ch_in=params["unet"]["n_ch_output_features"] + 3)
        self.center_crop = CenterCrop()
        self.cropped_width = params["global"]["cropped_width"]
        self.cropped_height = params["global"]["cropped_width"]
        self.ch_v = params["interaction"]["ch_v"]
        self.ch_k_and_q = params["interaction"]["ch_k_and_q"]
        self.encoder_dz = EncoderConv(params,
                                      ch_in=params["interaction"]["ch_v"],
                                      dim_z=params["global"]["dim_zmask"])

    def forward(self, small_weight=None,
                bounding_box=None,
                unet_features_expanded=None,
                prob=None,
                width_raw_image=None,
                height_raw_image=None):

        n_boxes, batch_size = bounding_box.bx.shape
        small_bb = torch.ones_like(small_weight)
        pad = int(0.5 * self.cropped_height)
        small_weight_padded = F.pad(small_weight, (pad, pad, pad, pad), mode='constant', value=0)
        small_bb_padded = F.pad(small_bb, (pad, pad, pad, pad), mode='constant', value=0)
        bounding_box_few_twice_the_size = BB(bx=bounding_box.bx,
                                             by=bounding_box.by,
                                             bw=2 * bounding_box.bw,
                                             bh=2 * bounding_box.bh)
        cropped_feature_map_twice_the_size = self.cropper(bounding_box=bounding_box_few_twice_the_size,
                                                          big_stuff=unet_features_expanded,
                                                          width_small=2 * self.cropped_width,
                                                          height_small=2 * self.cropped_height)
        y = torch.cat((small_weight_padded,
                       small_bb_padded,
                       small_bb_padded * prob[..., None, None, None],
                       cropped_feature_map_twice_the_size), dim=-3)
        independent_dim = list(y.shape[:-3])
        dependent_dim = list(y.shape[-3:])
        #print(y.shape)
        kqv_tmp = self.cnn_kqv(y.view([-1] + dependent_dim))
        new_dependent_dim = list(kqv_tmp.shape[-3:])
        kqv = kqv_tmp.view(independent_dim + new_dependent_dim)
        #print(kqv.shape)
        kqv_cropped = self.center_crop(kqv, width=self.cropped_width, height=self.cropped_height)
        kqv_big = self.uncropper.forward(bounding_box=bounding_box,
                                         small_stuff=kqv_cropped,
                                         width_big=width_raw_image,
                                         height_big=height_raw_image)
        k_big, q_big, v_big = torch.split(kqv_big, (self.ch_k_and_q, self.ch_k_and_q, self.ch_v), dim=-3)
        delta_big = torch.zeros_like(v_big)

        for i in range(n_boxes):
            w_tmp = F.softmax(torch.sum(q_big[i] * k_big, dim=-3, keepdim=True) / np.sqrt(self.ch_k_and_q), dim=-5)
            delta_big[i] = torch.sum(w_tmp * v_big, dim=-5)

        delta_small = self.cropper(bounding_box=bounding_box,
                                   big_stuff=delta_big,
                                   width_small=self.cropped_width,
                                   height_small=self.cropped_height)
        delta_z = self.encoder_dz(delta_small)[0]

        return delta_z

#####
#####    def __init__(self, params):
#####        super().__init__()
#####
#####        self.cropper = Cropper(params)
#####        self.uncropper = Uncropper()
#####        self.ch_v = params["interaction"]["ch_v"]
#####        self.dim_z = params["global"]["dim_zmask"]
#####
#####        self.decoder_z_to_qkv = DecoderConv(params,
#####                                            dim_z=self.dim_z+1,
#####                                            ch_out=2+self.ch_v)
#####
#####        self.encoder_delta_to_z = EncoderConv(params,
#####                                              ch_in=2*self.ch_v,
#####                                              dim_z=self.dim_z)
#####        self.combine_network = torch.Linear(in_features=2*self.dim_z, out_features=self.dim_z, bias=True)
#####
#####        def forward(self, zmask=None, bounding_box=None, prob=None, width_big=None, height_big=None):
#####
#####        # v_small_softplus = F.softplus(v_small)
#####        # qkv_big = self.uncropper.forward(bounding_box=bounding_box_few,
#####        # small_stuff = torch.cat((q_small, k_small, v_small_softplus), dim=-3),
#####        # width_big = width_raw_image,
#####        # height_big = height_raw_image)
#####        # q_big, k_big, v_big = torch.split(qkv_big, (1, 1, 1), dim=-3)
#####
#####        # From each mask subtract the contribution coming from the other objects
#####        # big_weight = torch.zeros_like(v_big)
#####        # for i in range(n_boxes):
#####        #    w_tmp = F.softmax(q_big[i] * k_big, dim=-5)  # are strictly positive and sum to one
#####        # big_weight[i] = ((1.0 + w_tmp[i]) * v_big[i] - torch.sum(w_tmp * v_big, dim=-5)).clamp(
#####        #    min=0)  # big_weight>0 with zero background
#####
#####        # Note that I do NOT multiply by probability
#####        sum_weight = torch.sum(big_weight, dim=-5)
#####        fg_mask = torch.tanh(sum_weight)
#####        big_mask = fg_mask * big_weight / torch.clamp(sum_weight, min=1E-6)
#####        # 8. Compute the assignment probabilities and the instance masks
#####        # At the end I wnat:
#####        # a) q,k arbitrary with zero background.
#####        # b) v > 0 with zero background
#####        q_small, k_small, v_small = torch.split(self.decoder_masks.forward(zmask_few), (1, 1, 1), dim=-3)
#####        v_small_softplus = F.softplus(v_small)
#####
#####        q_big, k_big, v_big = torch.split(qkv_big, (1, 1, 1), dim=-3)
#####
#####        # From each mask subtract the contribution coming from the other objects
#####        big_weight = torch.zeros_like(v_big)
#####        for i in range(n_boxes):
#####            w_tmp = F.softmax(q_big[i] * k_big, dim=-5)  # are strictly positive and sum to one
#####            big_weight[i] = ((1.0 + w_tmp[i]) * v_big[i] - torch.sum(w_tmp * v_big, dim=-5)).clamp(
#####                min=0)  # big_weight>0 with zero background
#####
#####
#####
#####        # 0. preparation
#####        n_box, batch_size = zmask.shape[:2]
#####
#####        # 1. each latent code is decoded into a triplet of: key, query, value
#####        p=prob[:,:, None].detach()  # add singleton for channel
#####        qkv_small = self.decoder_z_to_qkv(torch.cat((zmask,p), dim=-1))
#####        qkv_big = self.uncropper.forward(bounding_box=bounding_box,
#####                                         small_stuff=qkv_small,
#####                                         width_big=width_big,
#####                                         height_big=height_big)
#####        q_big, k_big, v_big = torch.split(qkv_big, (1, 1, self.ch_v), dim=-3)  #split along the channel dimension
#####        q_small, k_small, v_small = torch.split(qkv_small, (1, 1, self.ch_v), dim=-3)  #split along the channel dimension
#####
#####        # 2. compute the interaction: delta_i = sum_j sigmoid(q_i * k_j) * v_j
#####        delta_big = torch.zeros_like(v_big)
#####        print("k_big.shape", k_big.shape)
#####
#####        assert 1==2
#####        for i in range(n_box):
#####            delta_big[i] = torch.sum( F.softmax(q_big[i] * k_big, dim=-5) * v_big, dim=-5)  # sum over box dimneison
#####
#####            #with torch.no_grad():
#####            #    eliminate_self_interaction = torch.ones_like(tmp_big)
#####            #    eliminate_self_interaction[i] = 0.0
#####            #delta_big[i] = torch.sum(tmp_big * eliminate_self_interaction, dim=-5)  # sum over box dimneison
#####
#####
#####        # 3. convert delta_i into z_i
#####        delta_small = self.cropper(bounding_box=bounding_box, big_stuff=delta_big)
#####        tmp_small = torch.cat((v_small, delta_small), dim=-3) # cat along the channel dimension
#####        delta_z = self.encoder_delta_to_z(tmp_small)[0]  # takes only the z_mu not the z_std
#####        assert delta_z.shape == zmask.shape
#####
#####
#####        #return self.combine_network(torch.cat((zmask,delta_z),dim=-1))
#####        #return delta_z
#####        return zmask + 0.01*delta_z
