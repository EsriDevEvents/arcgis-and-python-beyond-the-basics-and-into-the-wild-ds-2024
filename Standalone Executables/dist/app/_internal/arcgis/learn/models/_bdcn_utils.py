# MIT License

# Copyright (c) 2018 XuanyiLi

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

# Based on https://github.com/pkuCactus/BDCN

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from fastai.callbacks.hooks import hook_outputs
from fastai.callbacks.hooks import model_sizes
from fastai.vision import flatten_model
from collections import Counter
from ._timm_utils import get_backbone
from ._hed_utils import modify_layers, get_hooks


class _MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super().__init__()
        self.rate = rate
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate * 1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate * 2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate * 3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv_down = nn.Conv2d(32, 21, (1, 1), stride=1)

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = self.conv_down(o + o1 + o2 + o3)
        return out


class _IDblock(nn.Module):
    def __init__(self, layer_shape):
        super().__init__()
        self.sconvs = nn.ModuleList([])

        for i in range(layer_shape[2]):
            self.sconvs.append(_MSBlock(layer_shape[0], 4))

        self.score_dsn1 = nn.Conv2d(21, 1, 1, stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)

    def forward(self, x_input):
        act_sum = self.sconvs[0](x_input[0])
        for i in range(1, len(x_input)):
            act_sum += self.sconvs[i](x_input[i])
        s1 = self.score_dsn1(act_sum)
        s2 = self.score_dsn1_1(act_sum)

        return s1, s2


def get_bdcn_hooks(backbone_fn, backbone, chip_size):
    if "timm" in backbone_fn.__module__:
        hooks = get_hooks(backbone, chip_size)
    else:
        hookable_modules = flatten_model(backbone)
        hooks = [
            hookable_modules[i]
            for i, module in enumerate(hookable_modules)
            if isinstance(module, nn.ReLU)
        ]

    return hooks


class _BDCNModel(nn.Module):
    def __init__(self, backbone_fn, chip_size=224, pretrained=True):
        super().__init__()
        self.backbone = get_backbone(backbone_fn, pretrained)
        if len(self.backbone) < 2:
            self.backbone = self.backbone[0]
        modify_layers(self.backbone, backbone_fn)
        hooks = get_bdcn_hooks(backbone_fn, self.backbone, chip_size)
        self.hook = hook_outputs(hooks)
        model_sizes(self.backbone, size=(chip_size, chip_size))
        layer_shape = [(k.stored.shape[1], k.stored.shape[2]) for k in self.hook]
        self.block_shape = []
        for k, v in Counter(layer_shape).items():
            self.block_shape.append((k[0], k[1], v))
        self.block_shape.sort(key=lambda size: size[1], reverse=True)

        self.idb1 = _IDblock(self.block_shape[0])
        self.idb2 = _IDblock(self.block_shape[1])
        self.idb3 = _IDblock(self.block_shape[2])
        self.idb4 = _IDblock(self.block_shape[3])
        self.idb5 = _IDblock(self.block_shape[4])

        self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        self.fuse = nn.Conv2d(10, 1, 1, stride=1)

    def forward(self, x):
        self.backbone(x)
        features = self.hook.stored
        num_to = self.block_shape[0][-1]
        s1, s11 = self.idb1(features[:num_to])

        num_from = num_to
        num_to += self.block_shape[1][-1]
        s2, s21 = self.idb2(features[num_from:num_to])
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2(s21)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)

        num_from = num_to
        num_to += self.block_shape[2][-1]
        s3, s31 = self.idb3(features[num_from:num_to])
        s3 = self.upsample_4(s3)
        s31 = self.upsample_4(s31)
        s3 = crop(s3, x, 2, 2)
        s31 = crop(s31, x, 2, 2)

        num_from = num_to
        num_to += self.block_shape[3][-1]
        s4, s41 = self.idb4(features[num_from:num_to])
        s4 = self.upsample_8(s4)
        s41 = self.upsample_8(s41)
        s4 = crop(s4, x, 4, 4)
        s41 = crop(s41, x, 4, 4)

        num_from = num_to
        num_to += self.block_shape[4][-1]
        s5, s51 = self.idb5(features[num_from:num_to])
        s5 = self.upsample_8_5(s5)
        s51 = self.upsample_8_5(s51)
        s5 = crop(s5, x, 0, 0)
        s51 = crop(s51, x, 0, 0)

        if s1.size(2) != s2.size(2):
            s1 = F.interpolate(
                s1, (x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False
            )
            s11 = F.interpolate(
                s11, (x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False
            )

        o1, o2, o3, o4, o5 = (
            s1.detach(),
            s2.detach(),
            s3.detach(),
            s4.detach(),
            s5.detach(),
        )
        o11, o21, o31, o41, o51 = (
            s11.detach(),
            s21.detach(),
            s31.detach(),
            s41.detach(),
            s51.detach(),
        )
        p1_1 = s1
        p2_1 = s2 + o1
        p3_1 = s3 + o2 + o1
        p4_1 = s4 + o3 + o2 + o1
        p5_1 = s5 + o4 + o3 + o2 + o1
        p1_2 = s11 + o21 + o31 + o41 + o51
        p2_2 = s21 + o31 + o41 + o51
        p3_2 = s31 + o41 + o51
        p4_2 = s41 + o51
        p5_2 = s51

        fuse = self.fuse(
            torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1)
        )
        results = [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse]
        results = [torch.sigmoid(r) for r in results]

        return results


def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert h2 <= h1 and w2 <= w1
    data = data1[:, :, crop_h : crop_h + h2, crop_w : crop_w + w2]
    return data


def cross_entropy_loss2d(inputs, targets, balance=1.1):
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1.0 / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights).to(targets.device)

    loss = nn.BCELoss(weights, reduction="sum")(inputs, targets.type(torch.float))
    return loss


def bdcn_loss(out, labels):
    loss = 0
    for k in range(10):
        loss += (
            0.5 * cross_entropy_loss2d(out[k], labels) / labels.shape[0]
        )  # devide by batch size
    loss += 1.1 * cross_entropy_loss2d(out[-1], labels) / labels.shape[0]
    return loss
