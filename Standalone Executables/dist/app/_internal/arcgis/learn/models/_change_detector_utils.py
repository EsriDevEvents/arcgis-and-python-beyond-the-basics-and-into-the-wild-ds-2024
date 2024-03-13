# Code borrowed from https://github.com/justchenhao/STANet
# BSD 2-Clause License

# Copyright (c) 2020, justchenhao
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
from fastai.basic_train import Learner
from fastai.torch_core import split_model_idx, flatten_model
from .._utils.segmentation_loss_functions import dice
from fastai.basic_train import LearnerCallback
import numpy as np
from torch import optim
from functools import partial

AdamW = partial(optim.Adam, betas=(0.5, 0.99))


class F_mynet3(nn.Module):
    def __init__(self, backbone="resnet18", in_c=3, f_c=64, output_stride=8, **kwargs):
        self.in_c = in_c
        super(F_mynet3, self).__init__()
        self.module = mynet3(
            backbone=backbone,
            output_stride=output_stride,
            f_c=f_c,
            in_c=self.in_c,
            **kwargs
        )

    def forward(self, input):
        return self.module(input)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def ResNet34(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 64
    """
    # print(in_c)
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls["resnet34"])
    return model


def ResNet18(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 256, 128, 64, 64
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls["resnet18"])
    return model


def ResNet50(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    2048, 256
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls["resnet50"])
    return model


def ResNet101(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls["resnet101"])
    return model


def ResNet152(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    model = ResNet(Bottleneck, [3, 4, 36, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls["resnet152"])
    return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn1 = BatchNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, output_stride, BatchNorm, pretrained=True, in_c=3
    ):
        self.inplanes = 64
        self.in_c = in_c
        # print('in_c: ',self.in_c)
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 4:
            strides = [1, 1, 1, 1]
            dilations = [1, 2, 4, 8]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(
            self.in_c, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=strides[0],
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=strides[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=strides[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.layer4 = self._make_MG_unit(
            block,
            512,
            blocks=blocks,
            stride=strides[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation, downsample, BatchNorm)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm)
            )

        return nn.Sequential(*layers)

    def _make_MG_unit(
        self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation=blocks[0] * dilation,
                downsample=downsample,
                BatchNorm=BatchNorm,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=blocks[i] * dilation,
                    BatchNorm=BatchNorm,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # | 4
        x = self.layer1(x)  # | 4
        low_level_feat2 = x  # | 4
        x = self.layer2(x)  # | 8
        low_level_feat3 = x
        x = self.layer3(x)  # | 16
        low_level_feat4 = x
        x = self.layer4(x)  # | 32
        return x, low_level_feat2, low_level_feat3, low_level_feat4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def build_backbone(backbone, output_stride, BatchNorm, in_c=3):
    if backbone == "resnet152":
        return ResNet152(output_stride, BatchNorm, in_c=in_c)
    if backbone == "resnet101":
        return ResNet101(output_stride, BatchNorm, in_c=in_c)
    if backbone == "resnet50":
        return ResNet50(output_stride, BatchNorm, in_c=in_c)
    elif backbone == "resnet34":
        return ResNet34(output_stride, BatchNorm, in_c=in_c)
    elif backbone == "resnet18":
        return ResNet18(output_stride, BatchNorm, in_c=in_c)
    else:
        raise NotImplementedError


def define_F(in_c, f_c, type="unet", backbone="resnet18", **kwargs):
    if type == "mynet3":
        # print("using mynet3 backbone")
        return F_mynet3(
            backbone=backbone, in_c=in_c, f_c=f_c, output_stride=32, **kwargs
        )
    else:
        NotImplementedError("no such F type!")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CDSA(nn.Module):
    """self attention module for change detection"""

    def __init__(self, in_c, ds=1, mode="BAM"):
        super(CDSA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        # print('ds: ', self.ds)
        self.mode = mode
        if self.mode == "BAM":
            self.Self_Att = BAM(self.in_C, ds=self.ds)
        elif self.mode == "PAM":
            self.Self_Att = PAM(
                in_channels=self.in_C,
                out_channels=self.in_C,
                sizes=[1, 2, 4, 8],
                ds=self.ds,
            )
        self.apply(weights_init)

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = torch.cat((x1, x2), 3)
        x = self.Self_Att(x)
        return x[:, :, :, 0:height], x[:, :, :, height:]


class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, fc, backbone, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        if backbone in ["resnet18", "resnet34"]:
            self.dr2 = DR(64, 96)
            self.dr3 = DR(128, 96)
            self.dr4 = DR(256, 96)
            self.dr5 = DR(512, 96)
        else:
            self.dr2 = DR(256, 96)
            self.dr3 = DR(512, 96)
            self.dr4 = DR(1024, 96)
            self.dr5 = DR(2048, 96)
        self.last_conv = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.fc),
            nn.ReLU(),
        )

        self._init_weight()

    def forward(self, x, low_level_feat2, low_level_feat3, low_level_feat4):
        # x1 = self.dr1(low_level_feat1)
        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)
        x = F.interpolate(x, size=x2.size()[2:], mode="bilinear", align_corners=True)
        # x2 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode="bilinear", align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(fc, backbone, BatchNorm):
    return Decoder(fc, backbone, BatchNorm)


from ._arcgis_model import _change_tail


class mynet3(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        output_stride=16,
        f_c=64,
        freeze_bn=False,
        in_c=3,
        data=None,
    ):
        super(mynet3, self).__init__()
        # print('arch: mynet3')
        BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)
        if data is not None:
            if data._is_multispectral:
                self.backbone = _change_tail(self.backbone, data)
        self.decoder = build_decoder(f_c, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, f2, f3, f4 = self.backbone(input)
        x = self.decoder(x, f2, f3, f4)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class _PAMBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input/Output:
        N * C  *  H  *  (2*W)
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to partition the input feature maps
        ds                : downsampling scale
    """

    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.key_channels),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.key_channels),
        )
        self.f_value = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.value_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)
        # input shape: b,c,h,2w
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) // 2

        local_y = []
        local_x = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h - (h % step_h)
                if j == (self.scale - 1):
                    end_y = w - (w % step_h)
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)  # B*N*H*W*2
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)  # B*N*H*W*2
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)  # B*N*H*W*2

        local_block_cnt = 2 * self.scale * self.scale

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(
                batch_size_new, self.value_channels, -1
            )

            query_local = query_local.contiguous().view(
                batch_size_new, self.key_channels, -1
            )
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(
                batch_size_new, self.key_channels, -1
            )

            sim_map = torch.bmm(query_local, key_local)  # batch matrix multiplication
            sim_map = (self.key_channels**-0.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            # context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(
                batch_size_new, self.value_channels, h_local, w_local, 2
            )
            return context_local

        #  Parallel Computing to speed up
        #  reshape value_local, q, k
        v_list = [
            value[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
            for i in range(0, local_block_cnt, 2)
        ]
        v_locals = torch.cat(v_list, dim=0)
        q_list = [
            query[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
            for i in range(0, local_block_cnt, 2)
        ]
        q_locals = torch.cat(q_list, dim=0)
        k_list = [
            key[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
            for i in range(0, local_block_cnt, 2)
        ]
        k_locals = torch.cat(k_list, dim=0)
        # print(v_locals.shape)
        context_locals = func(v_locals, q_locals, k_locals)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size * (j + i * self.scale)
                right = batch_size * (j + i * self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)

        if self.ds != 1:
            context = F.interpolate(context, [h * self.ds, 2 * w * self.ds])

        return context


class PAMBlock(_PAMBlock):
    def __init__(
        self, in_channels, key_channels=None, value_channels=None, scale=1, ds=1
    ):
        if key_channels == None:
            key_channels = in_channels // 8
        if value_channels == None:
            value_channels = in_channels
        super(PAMBlock, self).__init__(
            in_channels, key_channels, value_channels, scale, ds
        )


class PAM(nn.Module):
    """
    PAM module
    """

    def __init__(self, in_channels, out_channels, sizes=([1]), ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds  # output stride
        self.value_channels = out_channels
        self.key_channels = out_channels // 8

        self.stages = nn.ModuleList(
            [
                self._make_stage(
                    in_channels, self.key_channels, self.value_channels, size, self.ds
                )
                for size in sizes
            ]
        )
        self.conv_bn = nn.Sequential(
            nn.Conv2d(
                in_channels * self.group,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
        )

    def _make_stage(self, in_channels, key_channels, value_channels, size, ds):
        return PAMBlock(in_channels, key_channels, value_channels, size, ds)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]

        #  concat
        context = []
        for i in range(0, len(priors)):
            context += [
                F.interpolate(
                    priors[i],
                    size=feats.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            ]

        output = self.conv_bn(torch.cat(context, 1))

        return output


class BAM(nn.Module):
    """Basic self-attention module"""

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        # print('ds: ',ds)
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-0.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input

        return out


class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # Not getting used.
        label[label == 255] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_num = torch.sum((label == 1).float()) + 0.0001
        neg_num = torch.sum((label == -1).float()) + 0.0001

        loss_1 = torch.sum((1 + label) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = (
            torch.sum(
                (1 - label)
                / 2
                * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
            )
            / neg_num
        )
        loss = loss_1 + loss_2
        return loss


class CD_Model(nn.Module):
    def __init__(self, netF, netA):
        super().__init__()
        self.netF = netF
        self.netA = netA

    def forward(self, inp1, inp2):
        feat_A = self.netF(inp1)
        feat_B = self.netF(inp2)
        # print(feat_A.shape, feat_B.shape)
        feat_A, feat_B = self.netA(feat_A, feat_B)

        dist = F.pairwise_distance(feat_A, feat_B, keepdim=True)

        dist = F.interpolate(
            dist, size=inp1.shape[2:], mode="bilinear", align_corners=True
        )

        return dist


def split_layer_groups(learn):
    layers = flatten_model(learn.model)
    for idx, l in enumerate(layers):
        if "decoder" in layers:
            break
    learn.layer_groups = split_model_idx(learn.model, [idx])


class METRICS:
    "Datastore"
    precision = None
    recall = None
    f1 = None
    pass


from .._utils.classified_tiles import post_process_CD, calculate_metrics


def compute_datastore(preds, targets):
    ## hardcoding 2 for binary CD
    preds, targets = post_process_CD(preds, targets, 2)
    precision, recall, f1 = calculate_metrics(preds, targets, "per_class")
    # print(precision.shape, recall.shape, f1.shape)
    METRICS.precision = precision.mean(0)[1]
    METRICS.recall = recall.mean(0)[1]
    METRICS.f1 = f1.mean(0)[1]


def precision(preds, targets):
    if METRICS.precision is None:
        compute_datastore(preds, targets)
    return METRICS.precision


def recall(preds, targets):
    if METRICS.recall is None:
        compute_datastore(preds, targets)
    return METRICS.recall


def f1(preds, targets):
    if METRICS.f1 is None:
        compute_datastore(preds, targets)
    return METRICS.f1


def get_scores(confusion_matrix):
    """
    Returns score about:
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
    For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
    :return:
    """
    hist = confusion_matrix
    num_classes = len(confusion_matrix)
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    acc_cls_ = tp / (sum_a1 + np.finfo(np.float32).eps)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    # ---------------------------------------------------------------------- #
    # 2. Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    cls_iu = dict(zip(range(num_classes), iu))

    # F1 score
    F1 = 2 * acc_cls_ * precision / (acc_cls_ + precision + np.finfo(np.float32).eps)

    scores = {"Overall_Acc": acc, "Mean_IoU": mean_iu}
    scores.update(cls_iu)
    scores.update({"precision_1": precision[1], "recall_1": acc_cls_[1], "F1_1": F1[1]})
    return scores


def __fast_hist(label_gt, label_pred, num_classes):
    """
    Collect values for Confusion Matrix
    For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
    :param label_gt: <np.array> ground-truth
    :param label_pred: <np.array> prediction
    :return: <np.ndarray> values for confusion matrix
    """
    mask = (label_gt >= 0) & (label_gt < num_classes)
    hist = np.bincount(
        num_classes * label_gt[mask].astype(int) + label_pred[mask],
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist


# def precision_recall_score()


def calculate_author_metrics(self):
    n_classes = len(self._data.classes)
    dl = self._data.valid_dl
    #
    confusion_matrix = np.zeros((n_classes, n_classes))
    for x, y in dl:
        with torch.no_grad():
            predictions = self.learn.model(*x).detach().to("cpu")
        predictions, y = post_process_CD(predictions, y, n_classes=n_classes)

        # print(y.shape, predictions.shape)
        predictions = predictions.argmax(1).numpy().flatten()
        y = y.cpu().numpy().flatten()
        # print(y.shape, predictions.shape)
        confusion_matrix += __fast_hist(y, predictions, n_classes)

    return get_scores(confusion_matrix)


def get_learner(data, backbone, self_attention_type="PAM"):
    netF = define_F(
        in_c=3,
        f_c=64,
        type="mynet3",
        backbone=backbone,
        data=data if data._is_multispectral else None,
    )
    netA = CDSA(in_c=64, ds=1, mode=self_attention_type)
    lossf = BCL()
    learn = Learner(
        data=data,
        model=CD_Model(netF, netA),
        loss_func=lossf,
        metrics=[precision, recall, f1],
        opt_func=AdamW,
    )
    learn.callbacks.append(ResetMetricCallback(learn))
    split_layer_groups(learn)
    learn.freeze()
    return learn


class ResetMetricCallback(LearnerCallback):
    def on_batch_end(self, **kwargs):
        METRICS.precision = None
        METRICS.recall = None
        METRICS.f1 = None
