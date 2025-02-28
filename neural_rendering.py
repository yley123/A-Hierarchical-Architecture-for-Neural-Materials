import matplotlib.pyplot as plt
import torch
import numpy as np
import utils
import dataset.generator_mitsuba
import os
import path_config
from torch import nn
from d2l import torch as d2l

import error_metric

import dataset.dataset_reader
import dataset.generator_mitsuba
import dataset.rays
import argparse
from tqdm import tqdm
import aux_info
import visualizer
import vis2
import vis3
import vis32
import generatevedio
import gene3dsample
import experiments
import traceback
import renderer
import common
import calculateloss
import calculateloss1

from mipmaptexture import MipmapTexture

import neurtext
from positionembeding import Embedding




class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FullyConnected1(torch.nn.Module):
    def __init__(self, num_in, num_out=3):
        super(FullyConnected1, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        print("really use the fully1")

        # self.func = torch.nn.Sequential(  # define the convolution layer
        #     torch.nn.Conv2d(num_in, 256, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(256, 256, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(256, 128, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(128, 64, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(64, 32, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 16, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(16, 8, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(8, self.num_out, 1),
        # )
        self.func = torch.nn.Sequential(  # define the convolution layer
            torch.nn.Conv2d(num_in, 25, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(25, 25, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(25, 25, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(25, self.num_out, 1),
        )

    def forward(self, x):
        return self.func(x)

    def count_weights_and_biases(self):
        num_weights = sum(p.numel() for p in self.parameters() if p.requires_grad and len(p.shape) > 1)
        num_biases = sum(p.numel() for p in self.parameters() if p.requires_grad and len(p.shape) == 1)
        return {"weights": num_weights, "biases": num_biases}


class FullyConnected2(torch.nn.Module):
    def __init__(self, num_in, num_out=3,pe=False,r=False,rpe=False):
        super(FullyConnected2, self).__init__()
        print("now is fully2")

        self.num_in = num_in
        # if pe!=0:
        #     print("pe mode, fully2 on")
        #     self.num_in=self.num_in+92
        #     if pe==2:
        #         self.num_in = self.num_in + 112

        if r:
            print("pe mode, rough on")
            self.num_in = self.num_in + 1
        if rpe:
            print("rpe mode, rough pe on")
            self.num_in = self.num_in + 21
        self.num_out = num_out

        self.func = torch.nn.Sequential(  # define the convolution layer
            torch.nn.Conv2d(self.num_in, 125, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(125, 125, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(125, 125, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(125, 125, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(125, 125, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(125, self.num_out, 1),
        )
        # self.func = torch.nn.Sequential(  # define the convolution layer
        #     torch.nn.Conv2d(num_in, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, self.num_out, 1),
        # )

    def forward(self, x):
        return self.func(x)


class FullyConnected6(torch.nn.Module):
    def __init__(self, num_in, num_out=3,pe=False,r=False,rpe=False):
        super(FullyConnected6, self).__init__()
        print("now is fully2")

        self.num_in = num_in
        # if pe!=0:
        #     print("pe mode, fully2 on")
        #     self.num_in=self.num_in+92
        #     if pe==2:
        #         self.num_in = self.num_in + 112

        if r:
            print("pe mode, rough on")
            self.num_in = self.num_in + 1
        if rpe:
            print("rpe mode, rough pe on")
            self.num_in = self.num_in + 21
        self.num_out = num_out

        self.func = torch.nn.Sequential(  # define the convolution layer
            torch.nn.Conv2d(self.num_in, 80, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 80, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 80, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 80, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, self.num_out, 1),
        )
        # self.func = torch.nn.Sequential(  # define the convolution layer
        #     torch.nn.Conv2d(num_in, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, self.num_out, 1),
        # )

    def forward(self, x):
        return self.func(x)
        
        
        
class FullyConnected5(torch.nn.Module):
    def __init__(self, num_in, num_out=3,pe=False,r=False,rpe=False):
        super(FullyConnected2, self).__init__()
        print("now is fully5")

        self.num_in = num_in
        # if pe!=0:
        #     print("pe mode, fully2 on")
        #     self.num_in=self.num_in+92
        #     if pe==2:
        #         self.num_in = self.num_in + 112

        if r:
            print("pe mode, rough on")
            self.num_in = self.num_in + 1
        if rpe:
            print("rpe mode, rough pe on")
            self.num_in = self.num_in + 21
        self.num_out = num_out

        self.func = torch.nn.Sequential(  # define the convolution layer
            torch.nn.Conv2d(self.num_in, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, self.num_out, 1),
        )
        # self.func = torch.nn.Sequential(  # define the convolution layer
        #     torch.nn.Conv2d(num_in, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, self.num_out, 1),
        # )

    def forward(self, x):
        return self.func(x)
        
        
class FullyConnected7(torch.nn.Module):
    def __init__(self, num_in, num_out=3,pe=False,r=False,rpe=False):
        super(FullyConnected2, self).__init__()
        print("now is fully7")

        self.num_in = num_in
        # if pe!=0:
        #     print("pe mode, fully2 on")
        #     self.num_in=self.num_in+92
        #     if pe==2:
        #         self.num_in = self.num_in + 112

        if r:
            print("pe mode, rough on")
            self.num_in = self.num_in + 1
        if rpe:
            print("rpe mode, rough pe on")
            self.num_in = self.num_in + 21
        self.num_out = num_out

        self.func = torch.nn.Sequential(  # define the convolution layer
            torch.nn.Conv2d(self.num_in, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, 500, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(500, self.num_out, 1),
        )
        # self.func = torch.nn.Sequential(  # define the convolution layer
        #     torch.nn.Conv2d(num_in, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, self.num_out, 1),
        # )

    def forward(self, x):
        return self.func(x)

class FullyConnected4(torch.nn.Module):
    def __init__(self, num_in, num_out=3,pe=False,r=False,rpe=False):
        super(FullyConnected4, self).__init__()
        print("now is fully4")

        self.num_in = num_in
        # if pe!=0:
        #     print("pe mode, fully2 on")
        #     self.num_in=self.num_in+92
        #     if pe==2:
        #         self.num_in = self.num_in + 112

        if r:
            print("pe mode, rough on")
            self.num_in = self.num_in + 1
        if rpe:
            print("rpe mode, rough pe on")
            self.num_in = self.num_in + 21
        self.num_out = num_out

        self.func = torch.nn.Sequential(  # define the convolution layer
            torch.nn.Conv2d(num_in, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, self.num_out, 1),
        )
        # self.func = torch.nn.Sequential(  # define the convolution layer
        #     torch.nn.Conv2d(num_in, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, self.num_out, 1),
        # )

    def forward(self, x):
        return self.func(x)
from network1 import Residual,Inception

class FullyConnected3(torch.nn.Module):
    def __init__(self, num_in, num_out=3,pe=False,r=False,rpe=False):
        super(FullyConnected3, self).__init__()

        self.num_in = num_in
        print("now is fully3 incepetion")


        if r:
            print("pe mode, rough on")
            self.num_in = self.num_in + 1
        if rpe:
            print("rpe mode, rough pe on")
            self.num_in = self.num_in + 21
        self.num_out = num_out

        self.func = torch.nn.Sequential(  # define the convolution layer
            torch.nn.Conv2d(self.num_in, 25, 1),
            Inception(25, 7, (12, 12), (3, 3), 3),
            Inception(25, 7, (12, 12), (3, 3), 3),
            torch.nn.Conv2d(25, self.num_out, 1),
        )
        # self.func = torch.nn.Sequential(  # define the convolution layer
        #     torch.nn.Conv2d(num_in, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, 25, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(25, self.num_out, 1),
        # )

    def forward(self, x):
        return self.func(x)

    def count_weights_and_biases(self):
        num_weights = sum(p.numel() for p in self.parameters() if p.requires_grad and len(p.shape) > 1)
        num_biases = sum(p.numel() for p in self.parameters() if p.requires_grad and len(p.shape) == 1)
        return {"weights": num_weights, "biases": num_biases}


class FullyConnectedFlexible(torch.nn.Module):
    def __init__(self, num_in, num_out=3, num_conv=None, conv_ch=None):
        super(FullyConnectedFlexible, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.num_conv = num_conv
        self.conv_ch = conv_ch

        layers = []

        for idx in range(num_conv):
            if idx != 0:
                ch_in = conv_ch
            else:
                ch_in = num_in

            if idx != num_conv - 1:
                ch_out = conv_ch
            else:
                ch_out = num_out

            layers.append(torch.nn.Conv2d(ch_in, ch_out, 1))
            if idx != num_conv - 1:
                layers.append(torch.nn.ReLU())

        self.func = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.func(x)


class DataItem:
    def __init__(self, data):
        self.data = data

    def get_pair(self):
        res = 512
        x = np.linspace(-1, 1, res)
        y = np.linspace(-1, 1, res)
        assert False
        xv, yv = np.meshgrid(x, y)
        location = np.stack([xv, yv], 0)
        location = torch.Tensor(location).float()
        location = location.unsqueeze(0)

        ground = self.data
        ground = torch.Tensor(ground).float()
        ground = ground.permute([2, 0, 1])
        ground = ground.unsqueeze(0)

        input = torch.zeros_like(ground)

        return input, ground, location


class DataGiver:
    def __init__(self):
        mr = dataset.generator_mitsuba.MitsubaRenderer()
        scene_info = mr.create_scene()
        self.buffer = mr.render(scene_info)

    def take(self):
        return DataItem(self.buffer)


class EvalOutput:
    def __init__(self):
        self.level_id = None
        self.neural_offset = None
        self.neural_offset_actual = None

        self.neural_offset_texture = None
        self.shadow_neural_offset = None
        self.neural_texture = None
        self.valid_pixels = None

        self.shadows_mask = None

        self.probability = None


def get_padding_name(tiled):
    if tiled:
        return "repeat"
    else:
        return "border"


class NeuralMaterialSavable:
    class NeuralOffset:
        def __init__(self, nms, num_ch, base_resolution, tiled, min_sigma, scale_down):
            self.number_mip_maps_levels = 1
            self.neural_offsets = []

            self.min_sigma = min_sigma
            self.scale_down = scale_down

            for level in range(self.number_mip_maps_levels):
                resolution = np.array(list(base_resolution))
                resolution = resolution / 2 ** level / self.scale_down
                resolution = [num_ch] + list(resolution.astype(int))

                neural_offset = torch.zeros(resolution, **nms.live.get_type_device(), requires_grad=True)
                self.neural_offsets.append(neural_offset)

            self.tiled = tiled
            print("resolution",base_resolution)

        def fuse_blur(self):
            with torch.no_grad():
                for idx in range(self.number_mip_maps_levels):
                    text = self.neural_offsets[idx]
                    text = utils.tensor.blur(1.03, text.unsqueeze(0), True)[0, :, :, :]
                    self.neural_offsets[idx][...] = text[...]

        def get_params(self):
            return [x for x in self.neural_offsets]

        def set_learning(self, state):
            for neural_offsets in self.neural_offsets:
                neural_offsets.requires_grad = state

        def get_numpy_array(self, sigma):
            sigma = self.get_sigma(sigma)

            neural_offset = self.neural_offsets[0].unsqueeze(0)
            neural_offset = utils.tensor.blur(sigma, neural_offset, True)
            neural_offset = neural_offset[0, :, :, :]
            neural_offset = neural_offset.permute(1, 2, 0)
            neural_offset = neural_offset.data.cpu().numpy()
            return neural_offset

        def get_sigma(self, sigma):
            if not hasattr(self, "min_sigma"):
                self.min_sigma = 2

            sigma = max(sigma, self.min_sigma)

            return sigma

        def get(self, location, sigma):
            num_batch = location.shape[0]

            sigma = self.get_sigma(sigma)

            total_neural_offset = 0

            for idx in range(self.number_mip_maps_levels):
                neural_offset = self.neural_offsets[idx].unsqueeze(0)

                neural_offset = utils.tensor.blur(sigma / 2 ** idx, neural_offset, True)
                neural_offset = neural_offset.repeat([num_batch, 1, 1, 1])

                neural_offset = utils.tensor.grid_sample(
                    neural_offset,
                    location,
                    'bilinear',
                    padding_mode=get_padding_name(self.tiled)
                )

                total_neural_offset = total_neural_offset + neural_offset

            return total_neural_offset

        def calculate_offset(self, dir, neural_depth):

            double_val = neural_depth.shape[1] > 1
            # double_val = False

            direct = False
            no_angle = False
            if double_val:
                dir_orth = torch.zeros_like(dir)
                dir_orth[:, 0:1, :, :] = dir[:, 1:2, :, :]
                dir_orth[:, 1:2, :, :] = -dir[:, 0:1, :, :]

            if no_angle:
                return neural_depth[:, 0:2, :, :]

            dir_z = utils.la4.get_3rd_axis(dir)

            dir_z = torch.clamp(dir_z, min=0.6)

            shift_val = dir / dir_z
            cur_location = shift_val * neural_depth[:, 0:(2 if direct else 1), :, :]

            if double_val:
                cur_location = cur_location + dir_orth / dir_z * neural_depth[:, 1:2, :, :]

            return cur_location

        def get_neural_texture_vis(self, index, mipmap_level_id):
            return neurtext.get_neural_texture_vis(self.neural_offsets[0], index)

    def __init__(self, live):

        self.live = live
        self.init(live)

    def set_base_radius(self, base_radius):
        self.base_radius = base_radius
        self.neural_textures.base_radius = base_radius

        if self.use_shadow_offsets:
            self.shadow_textures.base_radius = base_radius

    def get_num_mipmap(self):
        return self.number_mip_maps_levels

    def init(self, live):

        if live.dataset is None:
            return

        if self.live.args.experiment is not None:
            experiment = experiments.get_experiment(self.live.args.experiment)()

        self.iter_count = 0
        self.max_iter = 30000
        self.base_radius = None
        self.losssaver = []

        self.pyramid_sync_point = experiment.pyramid_sync_point

        self.cosine_factor_out = experiment.cosine_factor_out

        self.neural_text_num_ch = experiment.neural_text_num_ch
        self.use_shadow_offsets = experiment.use_shadow_offsets
        self.use_offset = experiment.use_offset

        self.use_offset_2 = experiment.use_offset_2
        self.offset_2_num_ch = experiment.offset_2_num_ch

        self.no_sigma = experiment.no_sigma

        self.nom_min_sigma = experiment.nom_min_sigma
        self.nom_scale_down = experiment.nom_scale_down

        self.sigma_start_radius = experiment.sigma_start_radius

        self.sigme_etime = experiment.sigma_1_time / np.log(self.sigma_start_radius)
        self.sigma_1_time = experiment.sigma_1_time

        self.shadow_mult = experiment.shadow_mult

        self.main_angletf = experiment.main_angletf

        self.custom_net = experiment.custom_net
        self.custom_net_num_conv = experiment.custom_net_num_conv
        self.custom_net_conv_ch = experiment.custom_net_conv_ch

        self.ims_t2_ch = 0
        # assert False

        self.loss_name = self.live.args.loss

        self.input_num_ch = self.main_angletf.num_channels
        self.num_ch_total = self.neural_text_num_ch + self.input_num_ch

        self.offset_network_num_aux = 0  # 7

        self.num_ch_total =self.num_ch_total

        if self.use_offset:
            self.num_ch_total += self.offset_network_num_aux

        self.offset_num_ch = experiment.offset_num_ch

        self.offset_total_num_ch = self.offset_num_ch + 2
        self.offset_2_total_num_ch = self.offset_2_num_ch + 2

        if self.use_shadow_offsets:
            self.shadow_text_num_ch = experiment.shadow_text_num_ch

            self.num_ch_total += self.shadow_text_num_ch

            self.shadow_offset_num_ch = experiment.shadow_offset_num_ch
            self.shadow_offset_total_num_ch = self.shadow_offset_num_ch + 2

        self.neural_textures = None
        self.neural_offsets = None

        self.resolution = None
        self.tiled = True

        if live.dataset is not None:
            resolution = live.dataset.resolution()
            self.resolution = resolution
            resolution = (list(self.resolution))

            if live.args.levels is None:
                self.number_mip_maps_levels = int(np.log2(max(np.array(resolution))))
            else:
                self.number_mip_maps_levels = live.args.levels

            self.neural_textures = MipmapTexture(self, self.number_mip_maps_levels,
                                                 self.neural_text_num_ch + self.ims_t2_ch,
                                                 resolution, self.base_radius, self.tiled)
            self.neural_offsets = NeuralMaterialSavable.NeuralOffset(self, self.offset_num_ch, resolution, self.tiled,
                                                                     min_sigma=self.nom_min_sigma,
                                                                     scale_down=self.nom_scale_down)
            if self.live.args.alpha:
                alpha=1
            else:
                alpha=0

            if self.live.args.pe != 0:
                print("pe mode")
                self.num_ch_total = self.num_ch_total + 92
                if self.live.args.pe == 2:
                    print("pe mode 2")
                    self.num_ch_total = self.num_ch_total + 112
                if self.live.args.pe == 3:
                    print("pe mode 3")
                    self.num_ch_total = self.num_ch_total + 312
                if self.live.args.xyzoff:
                    self.num_ch_total = self.num_ch_total -1

            if not self.custom_net:
                if self.live.args.net==3:
                    print("use net 3")
                    self.network = FullyConnected3(self.num_ch_total, alpha+3 + self.shadow_mult,self.live.args.pe,self.live.args.r,self.live.args.rpe)
                elif self.live.args.net==4:
                    print("use net 4")
                    self.network = FullyConnected4(self.num_ch_total, alpha + 3 + self.shadow_mult, self.live.args.pe,
                                                   self.live.args.r, self.live.args.rpe)

                elif self.live.args.net==1:
                    print("pe mode off, use fully 1")
                    self.network = FullyConnected1(self.num_ch_total, alpha+3 + self.shadow_mult)
                elif self.live.args.net==2:
                    print("pe mode on, use fully 2")

                    self.network = FullyConnected2(self.num_ch_total, alpha+3 + self.shadow_mult,self.live.args.pe,self.live.args.r,self.live.args.rpe)
                elif self.live.args.net==5:
                    print("pe mode on, use fully 5")

                    self.network = FullyConnected5(self.num_ch_total, alpha+3 + self.shadow_mult,self.live.args.pe,self.live.args.r,self.live.args.rpe)
                elif self.live.args.net==6:
                    print("pe mode on, use fully 6")

                    self.network = FullyConnected6(self.num_ch_total, alpha+3 + self.shadow_mult,self.live.args.pe,self.live.args.r,self.live.args.rpe)
                elif self.live.args.net==7:
                    print("pe mode on, use fully 7")

                    self.network = FullyConnected6(self.num_ch_total, alpha+3 + self.shadow_mult,self.live.args.pe,self.live.args.r,self.live.args.rpe)
            else:
                self.network = FullyConnectedFlexible(self.num_ch_total, alpha+3 + self.shadow_mult,
                                                      self.custom_net_num_conv, self.custom_net_conv_ch)
            #print(self.network.count_weights_and_biases())
            num_gpus=self.live.args.gpu
            devices = [d2l.try_gpu(i) for i in range(num_gpus)]
            #print(devices)
            self.network = nn.DataParallel(self.network, device_ids=devices)
            self.network = live.to_device(self.network)

            self.offset_network_num_dir = 1

            if not self.custom_net:
                self.offset_network = FullyConnected1(self.offset_total_num_ch,
                                                      self.offset_network_num_dir + self.offset_network_num_aux)
            else:
                self.offset_network = FullyConnectedFlexible(self.offset_total_num_ch,
                                                             self.offset_network_num_dir + self.offset_network_num_aux,
                                                             self.custom_net_num_conv, self.custom_net_conv_ch)

            self.offset_network = live.to_device(self.offset_network)

            if self.use_shadow_offsets:
                self.shadow_textures = MipmapTexture(self, self.number_mip_maps_levels,
                                                     self.shadow_text_num_ch,
                                                     resolution, self.base_radius, self.tiled)

                self.shadow_neural_offsets = NeuralMaterialSavable.NeuralOffset(self, self.shadow_offset_num_ch,
                                                                                resolution,
                                                                                self.tiled,
                                                                                min_sigma=self.nom_min_sigma,
                                                                                scale_down=self.nom_scale_down)

                self.shadow_offset_network = FullyConnected1(self.shadow_offset_total_num_ch, 1)
                self.shadow_offset_network = live.to_device(self.shadow_offset_network)

            if self.use_offset_2:
                self.neural_offsets_2 = NeuralMaterialSavable.NeuralOffset(self, self.offset_2_num_ch, resolution,
                                                                           self.tiled)

                self.offset_network_2 = FullyConnected1(self.offset_2_total_num_ch, 1)
                self.offset_network_2 = live.to_device(self.offset_network)

            optimizer_params = []
            optimizer_params += self.neural_textures.get_params()
            optimizer_params += list(self.network.parameters())
            optimizer_params += list(self.offset_network.parameters())
            optimizer_params += self.neural_offsets.get_params()

            if self.use_shadow_offsets:
                optimizer_params += self.shadow_textures.get_params()
                optimizer_params += list(self.shadow_offset_network.parameters())
                optimizer_params += self.shadow_neural_offsets.get_params()

            if self.use_offset_2:
                optimizer_params += self.neural_offsets_2.get_params()
                optimizer_params += list(self.offset_network_2.parameters())
            if self.live.args.pe or self.live.args.rpe:
                experiment.learning_rate = experiment.learning_rate / 2
            print("learning_rate",experiment.learning_rate)

            self.optimizer = torch.optim.Adam(optimizer_params, lr=experiment.learning_rate)

    def save(self, path):
        common.save(self, path)

    def load(self, path):
        common.load(self, path)

    def get_neural_texture_vis(self, text_type, index, mipmap_level_id):
        if text_type == 0:
            return self.neural_textures.get_neural_texture_vis(index, mipmap_level_id)
        if text_type == 1:
            return self.neural_offsets.get_neural_texture_vis(index, mipmap_level_id)
        if text_type == 2:
            return self.shadow_textures.get_neural_texture_vis(index, mipmap_level_id)

    def get_sigma(self):
        if self.no_sigma:
            return .1

        iter_count = self.iter_count
        # if iter_count < 4000*4:
        #     iter_count = iter_count% 4000

        # sigma = 8 * 2 ** (-iter_count/(1200))
        sigma = self.sigma_start_radius * np.exp(-iter_count / self.sigme_etime)

        sigma = np.clip(sigma, 0.1, float("inf"))
        # sigma = np.clip(sigma, 1.5, float("inf"))
        return sigma

    def dump_buffers(self, eval_output):
        return
        self.stats()

        path = os.path.join(path_config.path_images, "dump")

        if eval_output.neural_texture is not None:
            neural_texture = eval_output.neural_texture * .5 + .5

            neural_texture = neural_texture * eval_output.valid_pixels

            utils.tensor.save_png(neural_texture, os.path.join(path, "neural_texture.png"))

        if eval_output.neural_offset_texture is not None:
            neural_offset_texture = eval_output.neural_offset_texture * .5 + .5

            neural_offset_texture = neural_offset_texture * eval_output.valid_pixels

            utils.tensor.save_png(neural_offset_texture, os.path.join(path, "neural_offset_texture.png"))

        if eval_output.neural_offset_actual is not None:
            neural_offset_actual = eval_output.neural_offset_actual * 4 + .5

            neural_offset_actual = neural_offset_actual.repeat([1, 3, 1, 1])
            neural_offset_actual = neural_offset_actual[:, :3, :, :]
            neural_offset_actual[:, 2:, :, :] = 0

            neural_offset_actual = neural_offset_actual * eval_output.valid_pixels

            utils.tensor.save_png(neural_offset_actual, os.path.join(path, "neural_offset.png"))

        if eval_output.level_id is not None:
            level_id = eval_output.level_id
            level_id = level_id / 5

            level_id = level_id * eval_output.valid_pixels

            utils.tensor.save_png(level_id, os.path.join(path, "lod.png"))

        print("Saving")
        import time
        time.sleep(.1)

        exit()

    def get_pixels_weight(self, camera_query_radius, max_level, training_levels):
        with torch.no_grad():
            if self.base_radius > 0:
                i_camera_query_radius = camera_query_radius / self.base_radius
                i_camera_query_radius = i_camera_query_radius.clamp(1)

                prob2 = 1
                # prob2 = .1 \
                #         + .8 *(1 / i_camera_query_radius / i_camera_query_radius)/np.log(4.)
                # assert False

                # print("Mean prob", prob2.mean())

                i_biggest_radius = 2 ** (max_level - 1)

                mask = i_camera_query_radius <= i_biggest_radius
                # mask =   i_camera_query_radius <= 1.2

                if False:
                    num_exec = 0
                    for training_level in training_levels:
                        cur_mask = i_camera_query_radius <= 2 ** (training_level - 1)
                        num_exec = num_exec + cur_mask

                    prob_inv = num_exec / num_exec.float().mean()
                prob_inv = 1

                return mask * prob_inv / prob2
            else:
                return 1

    def stats(self):

        def get_num_params(x):
            return sum(p.numel() for p in x.parameters())

        print("Number of params in network: ", get_num_params(self.network))
        if self.use_offset:
            print("Number of params in neural offset: ", get_num_params(self.offset_network))

        if self.use_shadow_offsets:
            print("Number of params in shadow neural offset: ", get_num_params(self.shadow_offset_network))


    def evaluate(self, input, location, rough=None,query_radius=None, level_id=None, mimpap_type=0, camera_dir=None,
                 training=False, max_level=-1, valid_pixels=None,pe=True):

        # if not training:
        #     self.stats()

        if max_level == -1:
            max_level = self.number_mip_maps_levels

        eval_output = EvalOutput()
        #print(input.type)

        light_dir = input[:, 2:4, :, :]
        #print(light_dir.shape)

        query_radius = query_radius

        if query_radius is not None:
            eval_output.level_id = torch.log2(query_radius / self.neural_textures.base_radius)

        eval_output.valid_pixels = valid_pixels

        # utils.tensor.displays(light_dir)
        # utils.tensor.displays(location*.5+.5)
        # utils.tensor.displays(camera_dir*.5+.5)
        # utils.tensor.displays(camera_dir*.5+.5)

        # a = camera_dir[:,0,:,:]
        # b = camera_dir[:,1,:,:]
        #
        #
        # camera_dir[:, 0, :, :] = a
        # camera_dir[:, 1, :, :] = -b
        #
        # light_dir[:, 0, :,:] = light_dir[:, 0, :,:]
        # light_dir[:,1, :,:] = light_dir[:, 1, :,:]

        # if training:
        #     light_add = torch.randn_like(light_dir)*.33 * (self.iter_count/4000)
        #     light_dir = light_dir + light_add
        #     light_dir = torch.randn_like(light_dir)*0
        #
        #     camera_dir = camera_dir*0

        # camera_dir[:, 1, :, :] = -camera_dir[:, 1, :, :]

        light_dir_z = utils.la4.get_3rd_axis(light_dir)
        #print("lz", light_dir_z.shape)

        num_batch = location.shape[0]

        if isinstance(camera_dir, list):
            ones = torch.ones([num_batch, 1] + list(self.resolution), **utils.tensor.Type(location).same_type())
            camera_dir_x = ones * camera_dir[0]
            camera_dir_y = ones * camera_dir[1]
            camera_dir = torch.cat([camera_dir_x, camera_dir_y], 1)
        #print("lo1", location.shape)

        #print("lo", location.shape)
        if self.live.args.pe and pe:
            #print(camera_dir.shape)
            camera_dir_x1 = camera_dir[:, 0,  None,:, :]
            #print("cax", camera_dir_x1.shape)
            camera_dir_y1 = camera_dir[:, 1, None, :, :]
            camera_dir1 = torch.cat([camera_dir_x1, camera_dir_y1], 1)
            #print(camera_dir1.shape)
            camera_dir_z1=utils.la4.get_3rd_axis(camera_dir1)
            cd3d=torch.cat([camera_dir_x1, camera_dir_y1,camera_dir_z1], 1)
            #print("111",cd3d.shape)
            ld3d=torch.cat([light_dir,light_dir_z], 1)
            if self.live.args.pe==1:
                poEb=Embedding(2, 10)
                dirEb = Embedding(3, 4)
                pooo = Embedding(3, 10)
            if self.live.args.pe==2:
                poEb = Embedding(2, 20)
                dirEb = Embedding(3, 10)
                pooo=Embedding(3, 13)
            if self.live.args.pe==3:
                poEb = Embedding(2, 40)
                dirEb = Embedding(3, 20)
                pooo = Embedding(3, 10)
            #poEb=Embedding(2, 20)
            #dirEb = Embedding(3, 10)
            cameradhf=dirEb(cd3d)
            #print("333", cameradhf.shape)
            lightdhf=dirEb(ld3d)
            #print("111", lightdhf.shape)
            locationhd=poEb(location)
            #print("222",locationhd.shape)
        location = location.permute(0, 2, 3, 1)
        #print("location",locationhd)

        sigma = self.get_sigma()

        if training and self.iter_count > self.sigma_1_time:
            self.neural_offsets.set_learning(False)

            if self.use_shadow_offsets:
                self.shadow_neural_offsets.set_learning(False)

        if training:
            if self.iter_count < 3000:
                if self.iter_count % 100 == 0:

                    self.neural_textures.fuse_blur()

                    # if self.use_offset: #TODO is this correct?

                    if self.use_shadow_offsets:
                        self.shadow_textures.fuse_blur()
                        self.shadow_neural_offsets.fuse_blur()

        if training and self.iter_count == self.pyramid_sync_point:
            self.neural_textures.recreate_pyramid()

            if self.use_shadow_offsets:
                self.shadow_textures.recreate_pyramid()

        neural_offset_aux = None
        if self.use_offset:
            neural_offset = self.neural_offsets.get(location, sigma)
            eval_output.neural_offset_texture = neural_offset
            #print(neural_offset.type)
            #print(camera_dir.type)

            neural_offset_aux = self.offset_network(torch.cat([neural_offset, camera_dir], dim=-3), )
            eval_output.camera_dir = camera_dir

            # eval_output.neural_offset_texture_e0 = torch.nn.Sequential(self.offset_network.func[:2])(torch.cat([neural_offset, camera_dir], dim=-3))
            # eval_output.neural_offset_texture_e0 = torch.nn.Sequential(self.offset_network.func[:])(
            #     torch.cat([neural_offset, camera_dir], dim=-3))

            neural_offset = neural_offset_aux[:, 0:self.offset_network_num_dir, :, :] * .1
            neural_offset_aux = neural_offset_aux[:, self.offset_network_num_dir:, :, :]

            eval_output.neural_offset = neural_offset.permute(0, 2, 3, 1)
            neural_offset = self.neural_offsets.calculate_offset(camera_dir, neural_offset)

            if True:
                eval_output.neural_offset_actual = neural_offset.detach()

            neural_offset = neural_offset.permute(0, 2, 3, 1)

            if self.use_offset_2:
                neural_offset_2 = self.neural_offsets_2.get(location + neural_offset, sigma)

                neural_offset_2 = self.offset_network_2(torch.cat([neural_offset_2, camera_dir], dim=-3), ) * .1

                eval_output.neural_offset_2 = neural_offset_2.permute(0, 2, 3, 1) + eval_output.neural_offset
                neural_offset_2 = self.neural_offsets_2.calculate_offset(camera_dir, neural_offset_2)
                neural_offset_2 = neural_offset_2.permute(0, 2, 3, 1)

                neural_offset = neural_offset + neural_offset_2
                del neural_offset_2

        else:
            neural_offset = 0

        if not self.live.args.xyzoff:
            intersection_location = location + neural_offset
        else:
            offset3d = torch.sum(neural_offset, dim=-1, keepdim=True)
            #print(offset3d.shape)
            #print(location.shape)
            intersection_location = torch.cat([location, offset3d], dim=-1)
            #print(intersection_location.shape)
            locationhd = pooo(intersection_location.permute(0, 3, 1,2))
            #print(locationhd.shape)
        #print("intersection",intersection_location.shape)
        #print("location2",location)
        #os.system("pause")

        del neural_offset
        del location

        t2_lookup = self.neural_textures.evaluate_at_points(mimpap_type, intersection_location, query_radius, sigma,
                                                            level_id, eval_output,
                                                            max_level=max_level)

        total_lookup_texture = t2_lookup[:, :self.neural_text_num_ch, :, :]

        eval_output.neural_texture = total_lookup_texture
        #print("shape???????????????",total_lookup_texture.shape)

        if False:
            self.dump_buffers(eval_output)
            
        if self.live.args.pe and pe:
            ang_input = torch.cat([cameradhf, lightdhf,locationhd], dim=1)
            if self.live.args.r or self.live.args.rpe:
                if self.live.args.rpe:
                    #print("rpe mode")
                    roEb = Embedding(1, 10)
                    rougheb=roEb(rough)
                    combined_input = [ang_input, rougheb, total_lookup_texture]
                else:
                    #print("r mode")
                #print(rougheb.shape)
                #print(ang_input.shape)
                    combined_input = [ang_input, rough, total_lookup_texture]
            else:
                #print(rough.shape)
                #print(rough)
                #print(ang_input.shape)
                #print("normal mode")
                combined_input = [ang_input, total_lookup_texture]
        else:
            ang_input = self.main_angletf(camera_dir, light_dir)
            if self.live.args.r or self.live.args.rpe:
                combined_input = [ang_input.convert(), rough, total_lookup_texture]
            else:
                combined_input = [ang_input.convert(), total_lookup_texture]

        #combined_input = [ang_input.convert(), rough,total_lookup_texture]

        #print(dir(combined_input))
        if neural_offset_aux is not None and neural_offset_aux.shape[1] > 0:
            combined_input.append(neural_offset_aux)

        if self.use_shadow_offsets:
            # detachrd intersection
            shadow_neural_offset = self.shadow_neural_offsets.get(intersection_location.detach(), sigma)
            shadow_neural_offset = self.shadow_offset_network(
                torch.cat([shadow_neural_offset, light_dir], dim=-3), ) * .1

            eval_output.shadow_neural_offset = shadow_neural_offset.permute(0, 2, 3, 1)
            shadow_neural_offset = self.shadow_neural_offsets.calculate_offset(light_dir, shadow_neural_offset)
            shadow_neural_offset = shadow_neural_offset.permute(0, 2, 3, 1)

            shadow_texture = self.shadow_textures.evaluate_at_points(mimpap_type,
                                                                     intersection_location + shadow_neural_offset,
                                                                     query_radius, sigma, level_id, eval_output,
                                                                     max_level=max_level)

            combined_input.append(shadow_texture)

        combined_input = torch.cat(combined_input, dim=1)


        result = self.network(combined_input)
        #print("result",result)
        if self.live.args.alpha:
            resultalpha = result[:, 3:4, :, :]



        result_aux = result[:, 3:, :, :]

        result = result[:, :3, :, :]
        #print(result.shape)
        if True:
            result = utils.tensor.yuv_to_rgb(result)

        if True:
            result = torch.exp(result) - 1

        if self.shadow_mult:
            shadow_mask = (.1 + torch.sigmoid(result))
            #print(shadow_mask.shape)
            eval_output.shadows_mask = shadow_mask


            result = result * shadow_mask

        if not training:
            result = torch.clamp(result, 0)

        # result = result*0+.1
        if self.live.args.alpha:

            result=torch.cat([result,resultalpha],dim=-3)

        return result, eval_output

    def generate_locations(self):
        height, width = self.resolution

        location = utils.tensor.generate_grid(height, width)
        location = self.live.to_device(location)

        return location

    def generate_uniform_locations(self, y, x):
        height, width = self.resolution

        one = torch.ones([1, 1, height, width]).float()
        y = y * one
        x = x * one

        location = torch.cat([x, y], 1)

        location = self.live.to_device(location)
        return location

    def generate_light(self):
        height, width = self.resolution

        location = utils.tensor.generate_grid(height, width)

        location = self.live.to_device(location)

        dir_x = location[:, 0:1, :, :]
        dir_y = location[:, 1:2, :, :]

        return dir_x, dir_y

    def generate_input(self, input_info: aux_info.InputInfo, use_repeat=False):
        input_info.to_torch(self.resolution, use_repeat)
        #print("self.resolution",self.resolution)
        #print("input_info.camera_dir_x.shape11",input_info.camera_dir_x.shape)

        result = torch.cat([input_info.camera_dir_x, input_info.camera_dir_y,
                            input_info.light_dir_x, input_info.light_dir_y,
                            ], dim=-3)
        #print("result.shape11", result.shape)

        result = self.live.to_device(result)

        mipmap_level_id = self.live.to_device(input_info.mipmap_level_id)

        return result, mipmap_level_id

    def generate_input_ex(self, input_info: aux_info.InputInfo):
        input_info.to_torch(self.resolution)

        camera_dir = torch.cat([input_info.camera_dir_x, input_info.camera_dir_y,
                                ], dim=-3)

        light_dir = torch.cat([
            input_info.light_dir_x, input_info.light_dir_y,
        ], dim=-3)

        light_dir = self.live.to_device(light_dir)
        camera_dir = self.live.to_device(camera_dir)

        mipmap_level_id = self.live.to_device(input_info.mipmap_level_id)

        return camera_dir, light_dir, mipmap_level_id


import common_renderer


class NeuralMaterialLive(common_renderer.CommonRenderer):
    def __init__(self, args):
        self.args = args
        self.model2=args

        self.device = torch.device("cuda:0")

        self.save_every = 100

        self.batch_size = self.args.batch

        if self.args.dataset is not None:
            self.datalo = self.args.dataset[0]
            dataset_paths = [path_config.get_value("path_dataset", dataset_path) for dataset_path in self.args.dataset]
            self.dataset = dataset.dataset_reader.DatasetMulti(dataset_paths, cosine_mult=self.args.cosine_mult,
                                                               boost=self.args.boost)

            self.dataloader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True,
                                                          num_workers=self.args.workers, drop_last=True,)
        else:
            self.dataset = None
            self.dataloader = None

        #print("dsdsdsd",dir(self.dataset.dataset.ground_color))
        #a=self.dataset.__getitem__(1)
        #print(a[0])
        #input, ground, location, camera_query_radius, base_radius, rough =self.dataset.__getitem__(1)

        self.path_save = os.path.join(path_config.path_models, self.args.outm)
        self.vars = NeuralMaterialSavable(self)

        if self.args.inm:
            path_load = path_config.get_value("path_models", self.args.inm)

            self.vars.load(path_load)

        if self.args.inm2:
            self.model2 = NeuralMaterialSavable(self)
            path_load = path_config.get_value("path_models", self.args.inm2)

            self.model2.load(path_load)
            #self.model2.live.args.pe =False
            #print(self.args)
        else:
            self.model2=None

        if self.args.max_iter is not None:
            self.vars.max_iter = self.args.max_iter

    def get_type_device(self):
        return {"dtype": torch.float32, "device": self.device}

    def to_device(self, *xs):
        return utils.tensor.to_device(self.device, *xs)

    def zero_grad(self):
        self.vars.optimizer.zero_grad()

    def get_num_mipmap(self):
        return self.neural_textures.get_num_mipmap()

    def train_step(self, data,idx,maxidx):
        #print(len(data))

        if len(data)==6:
            input, ground, location, camera_query_radius, base_radius,rough = data
            #print(input.shape)
            input, ground, location, camera_query_radius ,rough= self.to_device(input, ground, location, camera_query_radius,rough)
        else:
            input, ground, location, camera_query_radius, base_radius = data
            #print(input.shape)
            input, ground, location, camera_query_radius = self.to_device(input, ground, location,
                                                                                 camera_query_radius)
            rough=None
        #print(input.shape)
        #asdf=ground.data.cpu().numpy()
        #asdf=np.float32(asdf)
        #asdf=asdf[1,1,:]
        #print(asdf.shape)
        #asdv=asdf[(2, 1, 0)]

        #print(asdf.shape)
        #plt.imshow(asdf)
        #plt.show()
        #pausedsda
        camera_dir = input[:, :2, :, :]
        light_dir = input[:, 2:4, :, :]

        if self.vars.base_radius is None:
            # self.vars.set_base_radius(base_radius[0])
            self.vars.set_base_radius(1)

        if True:

            training_levels = [5, 5, 5, 7, self.vars.number_mip_maps_levels]
            # training_levels = [1,1,1,1,self.vars.number_mip_maps_levels]

            max_level = np.random.choice(training_levels)  # randomly generate number in training levels
            max_level = min(max_level, self.vars.number_mip_maps_levels)

            result, eval_output = self.vars.evaluate(input, location, rough=rough,query_radius=camera_query_radius,
                                                     camera_dir=camera_dir,
                                                     training=True, max_level=max_level)
            #print("result",result)

            loss_weight = self.vars.get_pixels_weight(camera_query_radius, max_level, training_levels)

            def nohigh(x):
                x = x.clamp(-0.1)
                return torch.log1p(x)

            def nsqrt(x):
                x = x.clamp(0.0000001)
                return torch.sqrt(x)
            def kaifang(x,n):
                x = x.clamp(0.0000001)
                return torch.pow(x,n)
            def sanatize(x):
                x=torch.where(torch.isnan(x),torch.full_like(x, 0), x)
                x=torch.where(torch.isinf(x),torch.full_like(x, 1),x)
                x = x.clamp(-0.1,2)
                return x
                #return np.nan_to_num(x, nan=0, posinf=0, neginf=0)

            ground=sanatize(ground)

            if self.vars.loss_name == "comb1":
                total_rad_loss = utils.tensor.mse_loss(loss_weight, nohigh(result), nohigh(ground))
                if False:
                    total_rad_loss = total_rad_loss - result.clamp(max=0).mean()
                else:
                    total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * .01
            elif self.vars.loss_name == "comb2":
                #print(ground-result)
                #print("1", ground, "2", result)
                total_rad_loss = utils.tensor.mse_loss(loss_weight, nohigh(result), nohigh(ground))
                #print("total_rad_loss1",total_rad_loss)
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * .1
                #print("total_rad_loss2", total_rad_loss)

            elif self.vars.loss_name == "l1":
                #print("1",ground,"2",result)
                total_rad_loss = utils.tensor.l1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "l2":
                total_rad_loss = utils.tensor.l2_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "rl1":
                mean_ground = ground.mean(1, keepdims=True) + .1
                total_rad_loss = utils.tensor.l1_loss(loss_weight / (mean_ground), result, ground)
            elif self.vars.loss_name == "rmse":
                new_g = nohigh(ground)
                mean_ground = new_g.mean(1, keepdims=True) + .1
                total_rad_loss = utils.tensor.mse_loss(loss_weight / (mean_ground) / (mean_ground), nohigh(result),
                                                       new_g)
                if True:
                    total_rad_loss = total_rad_loss - result.clamp(max=0).mean()
            elif self.vars.loss_name == "ssim":
                total_rad_loss=utils.tensor.ssim_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "msssiml1":
                total_rad_loss=utils.tensor.msssiml1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "msssim":
                total_rad_loss=utils.tensor.msssim_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "featureloss":
                total_rad_loss = utils.tensor.featureloss(loss_weight, result, ground)
            elif self.vars.loss_name == "mse":
                total_rad_loss = utils.tensor.mse_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "mtl":
                if idx<maxidx/2:
                    total_rad_loss = utils.tensor.l1_loss(loss_weight, result, ground)
                elif idx<maxidx*3/4:
                    total_rad_loss=utils.tensor.msssiml1_loss(loss_weight, result, ground)
                else:
                    total_rad_loss = utils.tensor.featureloss(loss_weight, result, ground)
            elif self.vars.loss_name == "gamma":
                total_rad_loss = utils.tensor.gamma_loss(loss_weight, result, ground)

            elif self.vars.loss_name == "gammal1":
                total_rad_loss = utils.tensor.gamma_lossl1(loss_weight, result, ground)

            elif self.vars.loss_name == "devide":
                total_rad_loss = utils.tensor.devide_loss(loss_weight, result, ground)

            elif self.vars.loss_name == "gammamse":
                total_rad_loss = utils.tensor.gamma_lossmse(loss_weight, result, ground)
            elif self.vars.loss_name == "weight":
                total_rad_loss = utils.tensor.weight_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "weightl1":
                total_rad_loss = utils.tensor.weight_loss(loss_weight, result, ground)
                #print("print(total_rad_loss)1",total_rad_loss.data)
                total_rad_loss=total_rad_loss+utils.tensor.l1_loss(loss_weight, result, ground)
                #print("print(total_rad_loss)2",total_rad_loss.data)
            elif self.vars.loss_name == "gamma1":
                total_rad_loss = utils.tensor.gamma1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "gamma2":
                total_rad_loss = utils.tensor.gamma2_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "gammah":
                total_rad_loss = utils.tensor.gammah_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "gammahl1":
                total_rad_loss = utils.tensor.gammah_lossl1(loss_weight, result, ground)
            elif self.vars.loss_name == "gamma1h":
                total_rad_loss = utils.tensor.gamma1h_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "gammahmse":
                total_rad_loss = utils.tensor.gammah_lossmse(loss_weight, result, ground)
            elif self.vars.loss_name == "devide1":
                total_rad_loss = utils.tensor.devide1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "devide5":
                total_rad_loss = utils.tensor.devide5_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "devideal1":
                total_rad_loss = utils.tensor.devide_loss(loss_weight, result, ground)
                total_rad_loss=total_rad_loss+utils.tensor.l1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "devideagl1":
                total_rad_loss = utils.tensor.devide_loss(loss_weight, result, ground)
                total_rad_loss = total_rad_loss +  utils.tensor.gamma_lossl1(loss_weight, result, ground)
            elif self.vars.loss_name == "devideg":
                total_rad_loss = utils.tensor.devideg_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "devideh":
                total_rad_loss = utils.tensor.devideh_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "edgeloss":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground)
            elif self.vars.loss_name == "edgelossl1":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground)
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "edgeloss1l19":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground)*0.1
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)*0.9
            elif self.vars.loss_name == "edgeloss9l11":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground)*0.9
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)*0.1
            elif self.vars.loss_name == "edgeloss5l15":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground)*0.5
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)*0.5
            elif self.vars.loss_name == "edgel1999":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground) * 0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.999
            elif self.vars.loss_name == "edgel199":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground) * 0.01
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.99
            elif self.vars.loss_name == "edge1l199":
                total_rad_loss = utils.tensor.f1edgeloss(loss_weight, result, ground) * 0.01
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.99
            elif self.vars.loss_name == "edge2l199":
                total_rad_loss = utils.tensor.f2edgeloss(loss_weight, result, ground) * 0.01
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.99
            elif self.vars.loss_name == "edgel19999":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground) * 0.0001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.9999
            elif self.vars.loss_name == "edge1l1999":
                total_rad_loss = utils.tensor.f1edgeloss(loss_weight, result, ground) * 0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.999
            elif self.vars.loss_name == "edge2l1999":
                total_rad_loss = utils.tensor.f2edgeloss(loss_weight, result, ground) * 0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.999
            elif self.vars.loss_name == "edgelogl1999":
                total_rad_loss = utils.tensor.fedgeloss(loss_weight, result, ground) * 0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, nohigh(result), nohigh(ground)) * 0.999
            elif self.vars.loss_name == "edge1logl1999":
                total_rad_loss = utils.tensor.f1edgeloss(loss_weight, result, ground) * 0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, nohigh(result), nohigh(ground)) * 0.999
            elif self.vars.loss_name == "edge2logl1999":
                total_rad_loss = utils.tensor.f2edgeloss(loss_weight, result, ground) * 0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, nohigh(result), nohigh(ground)) * 0.999
            elif self.vars.loss_name == "3edgelogl1999":
                total_rad_loss = utils.tensor.f3edgeloss(loss_weight, result, ground) * 0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, nohigh(result), nohigh(ground)) * 0.999
            elif self.vars.loss_name == "3edgel1999":
                total_rad_loss = utils.tensor.f3edgeloss(loss_weight, result, ground) * 0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.999
            elif self.vars.loss_name == "gradloss":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)
            elif self.vars.loss_name == "gradlossl1":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)       
            elif self.vars.loss_name == "gradlossl11":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)*0.1
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "gradlossl12":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)*0.01
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "gradlossl13":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)*0.001
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "gradsqlossl1":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, nsqrt(result),nsqrt(ground))
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight,  nsqrt(result),nsqrt(ground))
            elif self.vars.loss_name == "sqrtl1":
                total_rad_loss = utils.tensor.l1_loss(loss_weight, nsqrt(result),nsqrt(ground))
            elif self.vars.loss_name == "stagegl1":
                if idx < maxidx / 4:
                    total_rad_loss = utils.tensor.l1_loss(loss_weight, result, ground)
                else:
                    total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)
                    total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "stage1":
                if idx < maxidx / 4:
                    total_rad_loss = utils.tensor.l1_loss(loss_weight, kaifang(result,0.25), kaifang(ground,0.25))
                elif idx < maxidx * 3 / 4:
                    total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)
                    total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)
                else:
                    total_rad_loss = utils.tensor.fgradloss(loss_weight, kaifang(result, 0.5), kaifang(ground, 0.5))
                    total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, kaifang(result, 0.5),
                                                                           kaifang(ground, 0.5))

            elif self.vars.loss_name == "stage2":
                if idx < maxidx / 2:
                    total_rad_loss = utils.tensor.l1_loss(loss_weight, kaifang(result,0.25), kaifang(ground,0.25))
                else:
                    total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)
                    total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground)
            elif self.vars.loss_name == "gradlossl14":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, result, ground)
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, result, ground) * 0.1
            elif self.vars.loss_name == "4maploss":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, kaifang(result, 0.25), kaifang(ground, 0.25))
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, kaifang(result, 0.25), kaifang(ground, 0.25))
            elif self.vars.loss_name == "stage3":
                if idx < maxidx / 4:
                    total_rad_loss = utils.tensor.l1_loss(loss_weight, kaifang(result, 0.25), kaifang(ground, 0.25))
                else:
                    total_rad_loss = utils.tensor.fgradloss(loss_weight, kaifang(result, 0.25), kaifang(ground, 0.25))
                    total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, kaifang(result, 0.25), kaifang(ground, 0.25))
            elif self.vars.loss_name == "8maploss":
                total_rad_loss = utils.tensor.fgradloss(loss_weight, kaifang(result, 0.125), kaifang(ground, 0.125))
                total_rad_loss = total_rad_loss + utils.tensor.l1_loss(loss_weight, kaifang(result, 0.125), kaifang(ground, 0.125))












            else:
                assert False

            total_loss = total_rad_loss
            #print(total_loss)

            self.zero_grad()
            total_loss.backward()
            self.vars.optimizer.step()

        return result, eval_output

    def train(self):
        torch.autograd.set_detect_anomaly(self.args.debug)
        #torch.autograd.set_detect_anomaly(True)

        go = True

        # pbar_every = 100
        # make_pbar = lambda: tqdm(total=pbar_every, desc="{}".format(self.vars.iter_count),
        #                          bar_format='{l_bar}{bar}[{elapsed}<{remaining}, {rate_fmt}]')
        #
        # pbar = make_pbar()
        # pbar_count = 0

        while go:
            for data in self.dataloader:
                result, eval_output = self.train_step(data,self.vars.iter_count,self.vars.max_iter)

                if self.vars.iter_count % self.save_every == 0:
                    self.vars.save(self.path_save)

                    if self.vars.iter_count % (self.save_every*100/self.args.batch) == 0:
                        loss = self.calculateloss()
                        
                        self.vars.losssaver.append(loss)
                        
                    # utils.tensor.displays(result)
                    # if eval_output.neural_offset is not None:
                    #     utils.tensor.displays(eval_output.neural_offset + .5)

                self.vars.iter_count += 1



                #if self.vars.iter_count % 1 == 0:
                ##    timer, num_epochs = d2l.Timer(), 10
                #    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
                #    animator.add(self.vars.iter_count / 1 + 1, (d2l.evaluate_accuracy_gpu(self.vars.network, data),))
                # print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
                #      f'在{str(devices)}')
                # print(result)

                if self.vars.iter_count >= self.vars.max_iter:
                    print("loss", self.vars.losssaver)
                    losssss = self.calculateloss1()
                    go = False
                    break

        #         pbar.update(1)
        #         pbar_count += 1
        #         if pbar_count == pbar_every:
        #             pbar.close()
        #             pbar = make_pbar()
        #             pbar_count = 0
        #
        # pbar.close()
        # print("loss",self.vars.losssaver)

        self.vars.save(self.path_save)

        print("Training is done")

    def cpp_evaluate(self, datainput):
        print("Boo")

        num_batch, _ = datainput.shape

        datainput = torch.Tensor(datainput).float()
        datainput = datainput.unsqueeze(-1).unsqueeze(-1)
        datainput = self.to_device(datainput)

        camera_dir = datainput[:, 0:2, :, :]
        light_dir = datainput[:, 2:4, :, :]
        location = datainput[:, 4:6, :, :]

        camera_query_radius = 0

        input = torch.cat([camera_dir, light_dir], dim=1)

        result, neural_offset = self.vars.evaluate(input, location, query_radius=camera_query_radius,
                                                   camera_dir=camera_dir)

        result = result[:, :, 0, 0]
        result = result.data.cpu().numpy()
        result = np.ascontiguousarray(result)

        return result

    def cpp_evaluate3(self, datainput):
        # print("Evaluating")
        datainput = torch.Tensor(datainput).float()
        datainput = datainput[0, ...].unsqueeze(-1).unsqueeze(-1)

        datainput = datainput.permute(3, 1, 2, 0)
        print(datainput.shape)

        datainput = self.to_device(datainput)

        try:
            result, eval_output = self.brdf_eval(datainput)
        except Exception as e:

            print(traceback.format_exc())
            print(e)
            raise e

        result = result.permute(3, 1, 2, 0)
        result = result[:, :, 0, 0]

        result = result.unsqueeze(0)

        result = result.data.cpu().numpy()
        result = np.ascontiguousarray(result)

        return result

    def vis(self):
        visualizer.main(self.vars,self.model2)
    def vis2(self):
        vis2.main(self.vars, self.model2, self.datalo)
    def vis3(self):
        vis3.main(self.vars, self.model2, self.datalo)
    def vis4(self):
        vis32.main(self.vars, self.model2, self.datalo)
    def render3(self):
        gene3dsample.main(self.vars)
    def calculateloss(self):
        return calculateloss.main(self.vars, self.model2, self.datalo)
    def calculateloss1(self):
        return calculateloss1.main(self.vars, self.model2, self.datalo)
    def generatevedio(self):
        generatevedio.main(self.vars, self.model2)

    def export(self):
        import storage_model

        export_model = storage_model.StorageModel()

        decleration_code = []
        init_code = []
        run_code = []

        vector_names = ["vector1", "vector2"]

        def to_numpy(x):
            return x.data.cpu().numpy().astype(np.float32)

        def export_CNN_as_linear(model, prefix):
            idx_count = -1
            for idx, submodule in enumerate(model):
                if isinstance(submodule, torch.nn.ReLU):
                    continue
                elif isinstance(submodule, torch.nn.Conv2d):
                    idx_count += 1

                    module_name = prefix + ".mpl" + str(idx_count)

                    module_name_legal = module_name.replace(".", "_")

                    has_relu = idx + 1 < len(model) and isinstance(model[idx + 1], torch.nn.ReLU)

                    has_bias = submodule.bias is not None

                    if has_bias:
                        bias = to_numpy(submodule.bias)
                        bias = bias.reshape(-1)
                    else:
                        bias = None

                    weight = to_numpy(submodule.weight)
                    weight = weight[:, :, 0, 0]

                    num_in = weight.shape[1]
                    num_out = weight.shape[0]

                    export_model.add_linear_module(module_name, weight, bias, has_relu)
                    print("Addeding " + module_name)

                    decleration_code.append(
                        "ActiveLinearModule<{num_in},{num_out},{has_bias},{has_relu}> {module_name_legal};".format(
                            module_name_legal=module_name_legal, num_in=num_in, num_out=num_out, has_bias=has_bias,
                            has_relu=has_relu
                        ))

                    init_code.append('{module_name_legal}.load_from_storage_model("{module_name}");'.format(
                        module_name=module_name, module_name_legal=module_name_legal
                    ))

                    run_code.append('{module_name_legal}.forward({vec1}, {vec2});'.format(
                        module_name_legal=module_name_legal, vec1=vector_names[0], vec2=vector_names[1]
                    ))
                    vector_names[0], vector_names[1] = vector_names[1], vector_names[0]

                    print("BB")

        export_CNN_as_linear(self.vars.network.func, "main")
        run_code.append("//=====================")
        export_CNN_as_linear(self.vars.offset_network.func, "offset_network")

        decleration_code = "\n".join(decleration_code)
        init_code = "\n".join(init_code)
        run_code = "\n".join(run_code)

        code = decleration_code + "\n" * 4 + init_code + "\n" * 4 + run_code
        print(code)

        sigma = self.vars.get_sigma()
        export_model.set_nerual_mipmap_texture(self.vars.neural_textures.get_numpy_array(sigma))
        export_model.set_nerual_offset_texture(self.vars.neural_offsets.get_numpy_array(sigma))

        export_path = path_config.get_value("path_export", self.args.export)
        print(export_path)
        export_model.save(export_path)

    def get_vars(self):
        return self.vars

    def eval(self):
        pass


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inm', help='input model path', type=str, default=None)
    parser.add_argument('--inm2', help='input model path', type=str, default=None)
    parser.add_argument('--outm', help='output model path', type=str, default="default.pth")
    parser.add_argument('--vis', help='visualize', action='store_true')
    parser.add_argument('--vis2', help='visualize', action='store_true')
    parser.add_argument('--vis3', help='visualize', action='store_true')
    parser.add_argument('--vis4', help='visualize', action='store_true')
    parser.add_argument('--caloss', help='visualize', action='store_true')
    parser.add_argument('--render3', help='visualize', action='store_true')
    parser.add_argument('--gv', help='visualize', action='store_true')
    parser.add_argument('--stats', help='Print stats', action='store_true')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--export', help='output model path', type=str, default=None)

    parser.add_argument('--use_offset', action='store_true')

    parser.add_argument('--max_iter', default=30000, type=int)

    parser.add_argument('--batch', default=4, type=int)

    parser.add_argument('--t_pg', action='store_true')
    parser.add_argument('--pe', default=0, type=int)
    parser.add_argument('--rpe', action='store_true')
    parser.add_argument('--r', action='store_true')
    parser.add_argument('--alpha', action='store_true')
    parser.add_argument('--net', default=2, type=int)


    parser.add_argument('--dataset', help='dataset path', type=str, default=None, nargs='+')

    parser.add_argument('--loss', default='comb1', type=str)

    parser.add_argument('--levels', help='out name', type=int, default=None)
    parser.add_argument('--boost', help='Color boost', type=float, default=1.0)
    parser.add_argument('--vtype', help='Color boost', type=int, default=1)
    parser.add_argument('--cut3', help='Color boost', action='store_true')
    parser.add_argument('--xyzoff', help='Color boost', action='store_true')

    parser.add_argument('--workers', help='out name', type=int, default=8)
    parser.add_argument('--gpu', help='out name', type=int, default=1)
    parser.add_argument('--cosine_mult', help='Color boost', action='store_true')
    parser.add_argument('--m', help='out name', type=int, default=8)

    common_renderer.add_param_CommonRenderer(parser)

    parser.add_argument('--experiment', help='Experiments', type=str, default="StandardRawLongShadowMaskOnly",
                        choices=experiments.get_experiments_list())

    args = parser.parse_args()


    neural_material = NeuralMaterialLive(args)

    if args.vis:
        neural_material.vis()
    elif args.vis2:
        neural_material.vis2()
    elif args.vis3:
        neural_material.vis3()
    elif args.vis4:
        neural_material.vis4()

    elif args.render3:
        neural_material.render3()
    elif args.caloss:
        neural_material.calculateloss()
    elif args.gv:
        neural_material.generatevedio()
    elif args.export:
        neural_material.export()
    elif args.render:
        neural_material.render(args=args)
    elif args.ren_seq:
        neural_material.render_sequence()
    else:
        neural_material.train()


if __name__ == "__main__":
    main()
