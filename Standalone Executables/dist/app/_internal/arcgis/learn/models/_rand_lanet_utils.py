# MIT License

# Copyright (c) 2020 Tsunghan Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Based on https://github.com/tsunghan-wu/RandLA-Net-pytorch


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from fastai.torch_core import data_collate
import types
from functools import partial
from ._pointcnn_utils import get_indices

try:
    from .._utils.nearest_neighbors import knn_batch as knn_search
except Exception:
    raise Exception(
        f"The arcgis package was not installed, correctly(knn). Use deep learning essentials metapackage from https://github.com/Esri/deep-learning-frameworks"
    )
from functools import partial
import random

knn_search = partial(knn_search, omp=True)


def input_dict(input_list, cfg, is_sqn=False):
    num_layers = cfg["num_layers"]
    inputs = {}
    inputs["xyz"] = []
    if is_sqn:
        # add original points
        inputs["xyz"].append(input_list[3 * num_layers].float())
    for tmp in input_list[:num_layers]:
        inputs["xyz"].append(tmp.float())
    inputs["neigh_idx"] = []
    for tmp in input_list[num_layers : 2 * num_layers]:
        inputs["neigh_idx"].append(torch.from_numpy(tmp).long())
    inputs["sub_idx"] = []
    for tmp in input_list[2 * num_layers : 3 * num_layers]:
        inputs["sub_idx"].append(torch.from_numpy(tmp).long())
    if is_sqn:
        inputs["features"] = input_list[3 * num_layers + 1].transpose(1, 2).float()
    else:
        inputs["interp_idx"] = []
        for tmp in input_list[3 * num_layers : 4 * num_layers]:
            inputs["interp_idx"].append(torch.from_numpy(tmp).long())
        inputs["features"] = input_list[4 * num_layers].transpose(1, 2).float()

    return inputs


def batch_preprocess_dict(batch_pc, cfg, is_sqn=False):
    features = batch_pc
    input_points = []
    input_neighbors = []
    input_pools = []
    input_up_samples = []
    # need to handule points with extra fetures
    batch_pc = batch_pc[:, :, :3]  # take x,y,z only
    min_layer_point = 512
    for i in range(cfg["num_layers"]):
        layer_num_point = batch_pc.shape[1] // cfg["sub_sampling_ratio"][i]
        layer_num_point = max(layer_num_point, min_layer_point // (2**i))
        neighbour_idx = knn_search(batch_pc, batch_pc, cfg["k_n"])
        sub_points = batch_pc[:, :layer_num_point, :]
        pool_i = neighbour_idx[:, :layer_num_point, :]
        if is_sqn:
            input_points.append(sub_points)
        else:
            up_i = knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_up_samples.append(up_i)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        batch_pc = sub_points

    input_list = input_points + input_neighbors + input_pools

    if is_sqn:
        # add original points
        input_list += [features[:, :, :3]]
    else:
        input_list += input_up_samples

    input_list += [features]

    return input_dict(input_list, cfg, is_sqn)


def transform_data(input, target, sample_point_num, cfg, transform_fn=False, **kwargs):
    (
        input,
        point_nums,
    ) = input  ## (batch, total_points, num_features), (batch,)
    batch, _, num_features = input.shape

    ## get indices to sample
    indices = torch.tensor(get_indices(batch, sample_point_num, point_nums.long())).to(
        input.device
    )

    ## Get indices in the correct shape to be used for indexing
    indices = indices.view(-1, 2).long()

    ##  sample points from all the input and output points
    input = (
        input[indices[:, 0], indices[:, 1]]
        .view(batch, sample_point_num, num_features)
        .contiguous()
    )  ## batch, sample_point_num, num_features
    if target is not None:
        target = (
            target[indices[:, 0], indices[:, 1]]
            .view(batch, sample_point_num)
            .contiguous()
        )  ## batch, sample_point_num
        if transform_fn:
            if random.random() > 0.5:
                input[:, :, :3] = transform_fn._transform_tool(input)

    return batch_preprocess_dict(input, cfg, **kwargs), target


def prepare_data_dict(data, sample_point_num, cfg, **kwargs):
    def collate_fn(self, batch, sample_point_num, cfg, transform_fn=False, **kwargs):
        batch = data_collate(batch)
        return transform_data(
            batch[0], batch[1], sample_point_num, cfg, transform_fn, **kwargs
        )

    collate_fn_train = partial(
        collate_fn,
        sample_point_num=sample_point_num,
        cfg=cfg,
        transform_fn=data.transform_fn,
        **kwargs,
    )
    collate_fn_val = partial(
        collate_fn, sample_point_num=sample_point_num, cfg=cfg, **kwargs
    )
    data.train_dl.dl.collate_fn = types.MethodType(collate_fn_train, data.train_dl.dl)
    data.valid_dl.dl.collate_fn = types.MethodType(collate_fn_val, data.valid_dl.dl)

    return data


class RandLANetSeg(nn.Module):
    def __init__(self, config, in_chanls):
        super().__init__()
        self.config = config

        self.fc0 = Conv1d(in_chanls, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config["num_layers"]):
            d_out = self.config["out_channels"][i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        num_of_layers = self.config["num_layers"] - 1
        for j in range(self.config["num_layers"]):
            if j < num_of_layers:
                d_in = d_out + 2 * self.config["out_channels"][-j - 2]
                d_out = 2 * self.config["out_channels"][-j - 2]
            else:
                d_in = 4 * self.config["out_channels"][0]
                d_out = 2 * self.config["out_channels"][0]
            self.decoder_blocks.append(Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.fc1 = Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = Conv2d(
            32,
            self.config["num_classes"],
            kernel_size=(1, 1),
            bn=False,
            activation=None,
        )

    def forward(self, end_points):
        # transform input for infrencing
        if not isinstance(end_points, dict):
            device = end_points.device
            end_points = batch_preprocess_dict(end_points.cpu(), self.config)
            for key in end_points:
                if type(end_points[key]) is list:
                    for i in range(len(end_points[key])):
                        end_points[key][i] = end_points[key][i].to(device)
                else:
                    end_points[key] = end_points[key].to(device)

        features = end_points["features"]  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config["num_layers"]):
            f_encoder_i = self.dilated_res_blocks[i](
                features, end_points["xyz"][i], end_points["neigh_idx"][i]
            )

            f_sampled_i = self.random_sample(f_encoder_i, end_points["sub_idx"][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config["num_layers"]):
            f_interp_i = self.nearest_interpolation(
                features, end_points["interp_idx"][-j - 1]
            )
            f_decoder_i = self.decoder_blocks[j](
                torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1)
            )

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        return f_out.permute(0, 2, 1)

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        )
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[
            0
        ]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        )
        interpolated_features = interpolated_features.unsqueeze(
            3
        )  # batch*channel*npoints*1
        return interpolated_features


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = Conv2d(
            d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None
        )
        self.shortcut = Conv2d(
            d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None
        )

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)

        self.mlp2 = Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(
            feature.squeeze(-1).permute((0, 2, 1)), neigh_idx
        )
        f_neighbours = f_neighbours.permute(
            (0, 3, 1, 2)
        )  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(
            f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx
        )
        f_neighbours = f_neighbours.permute(
            (0, 3, 1, 2)
        )  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(
            1, 1, neigh_idx.shape[-1], 1
        )  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        # batch*npoint*nsamples*1
        relative_dis = torch.sqrt(
            torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True)
        )
        # batch*npoint*nsamples*10
        relative_feature = torch.cat(
            [relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1
        )
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(
            pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2])
        )
        features = features.reshape(
            batch_size, num_points, neighbor_idx.shape[-1], d
        )  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args: List[int],
        *,
        bn: bool = False,
        activation=nn.ReLU(inplace=True),
        preact: bool = False,
        first: bool = False,
        name: str = "",
        instance_norm: bool = False,
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0))
                    else None,
                    preact=preact,
                    instance_norm=instance_norm,
                ),
            )


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        activation,
        bn,
        init,
        conv=None,
        batch_norm=None,
        bias=True,
        preact=False,
        name="",
        instance_norm=False,
        instance_norm_func=None,
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(
                    out_size, affine=False, track_running_stats=False
                )
            else:
                in_unit = instance_norm_func(
                    in_size, affine=False, track_running_stats=False
                )

        if preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

            if not bn and instance_norm:
                self.add_module(name + "in", in_unit)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

            if not bn and instance_norm:
                self.add_module(name + "in", in_unit)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
        instance_norm=False,
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d,
        )


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
        instance_norm=False,
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d,
        )


class FC(nn.Sequential):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        activation=nn.ReLU(inplace=True),
        bn: bool = False,
        init=None,
        preact: bool = False,
        name: str = "",
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "fc", fc)

        if not preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + "activation", activation)


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model).__name__)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
