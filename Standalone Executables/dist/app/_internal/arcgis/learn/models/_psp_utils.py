# MIT License

# Copyright (c) 2019 Hengshuang Zhao

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

# Based on https://github.com/hszhao/semseg

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

import math
from fastai.callbacks.hooks import hook_outputs, hook_output
from fastai.callbacks.hooks import model_sizes
from fastai.vision import flatten_model
from fastai.vision.models import unet
from fastai.basic_train import Learner
from fastai.vision import to_device
from ._PointRend import PointRendSemSegHead
from ._arcgis_model import _set_ddp_multigpu, _isnotebook
from ._timm_utils import get_backbone
from ._deeplab_utils import (
    get_dilation_index,
    get_hooks,
    get_last_module,
    add_dilation,
)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _PyramidPoolingModule(nn.Module):
    """
    Creates the pyramid pooling module as in https://arxiv.org/abs/1612.01105
    Takes a feature map from the backbone and pools it at different scales
    according to the given pyramid sizes and upsamples it to original feature
    map size and concatenates it with the feature map.
    Code from https://github.com/hszhao/semseg.
    """

    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []

        ## Creating modules for different pyramid sizes
        for s in setting:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim, momentum=0.95),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            ## Pass through the module which reduces its spatial size and then upsamples it.
            out.append(
                F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True)
            )
        out = torch.cat(out, 1)
        return out


def _pspnet_unet(
    num_classes, backbone_fn, chip_size=224, pyramid_sizes=(1, 2, 3, 6), pretrained=True
):
    """
    Function which returns PPM module attached to backbone which is then used to form the Unet.
    """
    backbone = get_backbone(backbone_fn, pretrained)
    backbone_name = backbone_fn.__name__
    modify_dilation_index, _ = get_dilation_index(backbone_name)
    hookable_modules = get_last_module(backbone)
    add_dilation(backbone_fn, hookable_modules, modify_dilation_index)

    ## returns the size of various activations
    feature_sizes = model_sizes(backbone, size=(chip_size, chip_size))

    ## Get number of channels in the last layer
    num_channels = feature_sizes[-1][1]

    penultimate_channels = num_channels / len(pyramid_sizes)
    ppm = _PyramidPoolingModule(num_channels, int(penultimate_channels), pyramid_sizes)

    in_final = int(penultimate_channels) * len(pyramid_sizes) + num_channels

    # Reduce channel size after pyramid pooling module to avoid CUDA OOM error.
    final_conv = nn.Conv2d(
        in_channels=in_final, out_channels=512, kernel_size=3, padding=1
    )

    ## To make Dynamic Unet work as it expects a backbone which can be indexed.
    if "densenet" in backbone_name or "vgg" in backbone_name:
        backbone = backbone[0]
    layers = [*backbone, ppm, final_conv]
    return nn.Sequential(*layers)


class AuxPSUnet(nn.Module):
    """
    Adds auxillary loss to PSUnet.
    """

    def __init__(self, model, chip_size, num_classes):
        super(AuxPSUnet, self).__init__()
        self.model = model

        for idx, i in enumerate(flatten_model(self.model)):
            if hasattr(i, "dilation"):
                dilation = i.dilation
                dilation = dilation[0] if isinstance(dilation, tuple) else dilation
                if dilation > 1:
                    break

        self.hook = hook_output(flatten_model(model)[idx - 1])

        ## returns the size of various activations
        model_sizes(self.model, size=(chip_size, chip_size))

        ## Geting the stored parameters inside of the hook
        aux_in_channels = self.hook.stored.shape[1]
        del self.hook.stored
        self.aux_logits = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.model(x)
        out = F.interpolate(out, x.shape[2:], mode="bilinear", align_corners=True)
        if self.training:
            aux_l = self.aux_logits(self.hook.stored)
            ## Remove hook to free up memory
            self.hook.remove()
            return (
                out,
                F.interpolate(aux_l, x.shape[2:], mode="bilinear", align_corners=True),
            )
        else:
            return out


def _add_auxillary_branch_to_psunet(model, chip_size, num_classes):
    return AuxPSUnet(model, chip_size, num_classes)


class PSPUnet(nn.Module):
    """
    Keep model output and input size same.
    """

    def __init__(self, model):
        super(PSPUnet, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        out = F.interpolate(out, x.shape[2:], mode="bilinear", align_corners=True)
        return out


class PSPNet(nn.Module):
    """
    Vanilla PSPNet
    """

    def __init__(
        self,
        num_classes,
        backbone_fn,
        chip_size=224,
        pyramid_sizes=(1, 2, 3, 6),
        pretrained=True,
        pointrend=False,
        keep_dilation=False,
    ):
        super(PSPNet, self).__init__()
        self.pointrend = pointrend
        self.backbone = get_backbone(backbone_fn, pretrained)
        backbone_name = backbone_fn.__name__
        modify_dilation_index, self.vgg = get_dilation_index(
            backbone_name, pointrend, keep_dilation
        )
        hookable_modules = get_last_module(self.backbone)
        add_dilation(backbone_fn, hookable_modules, modify_dilation_index)
        hooks = get_hooks(backbone_name, hookable_modules)

        ## Hook at the index where we need to get the auxillary logits out along with Fine-grained features
        self.hook = hook_outputs(hooks)

        ## returns the size of various activations
        feature_sizes = model_sizes(self.backbone, size=(chip_size, chip_size))

        if not self.vgg:
            ## Geting the number of channel persent in stored activation inside of the hook
            aux_in_channels = self.hook[0].stored.shape[1]
            ## Get number of channels in the last layer
            num_channels = feature_sizes[-1][1]
        else:
            aux_in_channels = self.hook[-2].stored.shape[1]
            num_channels = self.hook[-1].stored.shape[1]

        penultimate_channels = num_channels / len(pyramid_sizes)
        self.ppm = _PyramidPoolingModule(
            num_channels, int(penultimate_channels), pyramid_sizes
        )

        self.final = nn.Sequential(
            ## To handle case when the length of pyramid_sizes is odd
            nn.Conv2d(
                int(penultimate_channels) * len(pyramid_sizes) + num_channels,
                math.ceil(penultimate_channels),
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(math.ceil(penultimate_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(math.ceil(penultimate_channels), num_classes, kernel_size=1),
        )

        self.aux_logits = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

        if self.pointrend:
            if self.vgg:
                point_num_channels = (
                    self.hook[-3].stored.shape[1] + self.hook[-4].stored.shape[1]
                )
                stride = chip_size / self.hook[-1].stored.shape[2]
            else:
                point_num_channels = self.hook[1].stored.shape[1]
                stride = chip_size / feature_sizes[-1][2]

            subdivision_steps = math.ceil(math.log(stride, 2))
            self.pointrend_head = PointRendSemSegHead(
                num_classes,
                point_num_channels,
                train_num_points=(chip_size / stride) ** 2,
                subdivision_num_points=(chip_size / (stride / 2)) ** 2,
                subdivision_steps=subdivision_steps,
            )

        initialize_weights(self.aux_logits)
        initialize_weights(self.ppm, self.final)

    def forward(self, x):
        x_size = x.size()
        x = self.backbone(x)
        features = self.hook.stored

        if self.vgg:
            x = self.ppm(features[-1])
            x = self.final(x)
        else:
            x = self.ppm(x)
            x = self.final(x)

        if self.pointrend:
            if self.vgg:
                pointrend_out = self.pointrend_head(x, [features[-4], features[-3]])
            else:
                pointrend_out = self.pointrend_head(x, [features[1]])

        result = F.interpolate(x, x_size[2:], mode="bilinear", align_corners=True)

        if self.training:
            if self.vgg:
                x = self.aux_logits(features[-2])
            else:
                x = self.aux_logits(features[0])

            x = F.interpolate(x, x_size[2:], mode="bilinear", align_corners=True)

            if self.pointrend:
                return result, x, pointrend_out
            else:
                return result, x

        else:
            if self.pointrend:
                if pointrend_out.shape[-1] != x_size[-1]:
                    pointrend_out = F.interpolate(
                        pointrend_out, x_size[2:], mode="bilinear", align_corners=True
                    )
                return pointrend_out
            else:
                return result


class DummyDistributed:
    "Dummy class to create a Learner since learner is created from fuction not a class. It will be used in case of multigpu training."

    def __getitem__(self, item):
        return eval("self." + item)


def _pspnet_learner(
    data,
    backbone,
    chip_size=224,
    pyramid_sizes=(1, 2, 3, 6),
    pretrained=True,
    pointrend=False,
    keep_dilation=False,
    **kwargs
):
    "Build psp_net learner from `data` and `arch`."
    model = to_device(
        PSPNet(
            data.c,
            backbone,
            chip_size,
            pyramid_sizes,
            pretrained,
            pointrend,
            keep_dilation,
        ),
        data.device,
    )
    if not _isnotebook():
        distributed_prep = DummyDistributed()
        _set_ddp_multigpu(distributed_prep)
        if distributed_prep._multigpu_training:
            learn = Learner(data, model, **kwargs).to_distributed(
                distributed_prep._rank_distributed
            )
            learn._map_location_multi_gpu = {
                "cuda:%d" % 0: "cuda:%d" % distributed_prep._rank_distributed
            }
        else:
            learn = Learner(data, model, **kwargs)
    else:
        learn = Learner(data, model, **kwargs)
    return learn


def _pspnet_learner_with_unet(
    data,
    backbone,
    chip_size=224,
    pyramid_sizes=(1, 2, 3, 6),
    pretrained=True,
    unet_aux_loss=False,
    vggv2=True,
    **kwargs
):
    "Build psunet learner from `data` and `arch`."
    model = unet.DynamicUnet(
        encoder=_pspnet_unet(data.c, backbone, chip_size, pyramid_sizes, pretrained),
        n_classes=data.c,
        last_cross=False,
    )

    if unet_aux_loss:
        model = _add_auxillary_branch_to_psunet(model, chip_size, data.c)
    elif vggv2:
        model = PSPUnet(model)
    if not _isnotebook():
        distributed_prep = DummyDistributed()
        _set_ddp_multigpu(distributed_prep)
        if distributed_prep._multigpu_training:
            learn = Learner(data, model, **kwargs).to_distributed(
                distributed_prep._rank_distributed
            )
            learn._map_location_multi_gpu = {
                "cuda:%d" % 0: "cuda:%d" % distributed_prep._rank_distributed
            }
        else:
            learn = Learner(data, model, **kwargs)
    else:
        learn = Learner(data, model, **kwargs)
    return learn


def isin(target, keep_indices):
    # import pdb; pdb.set_trace();
    old_shape = target.shape
    mask = torch.cat(
        [(target.view(-1) == k)[:, None] for k in keep_indices], dim=1
    ).any(1)
    mask = mask.view(old_shape).contiguous()
    return mask


def accuracy(input, target, ignore_mapped_class=[]):
    """Computes per pixel accuracy."""
    if isinstance(input, tuple):  # while training
        input = input[0]
    if ignore_mapped_class == []:
        target = target.squeeze(1)
        return (input.argmax(dim=1) == target).float().mean()
    else:
        target = target.squeeze(1)
        _, total_classes, _, _ = input.shape
        keep_indices = [i for i in range(total_classes) if i not in ignore_mapped_class]
        for k in ignore_mapped_class:
            input[:, k] = input.min() - 1
        targ_mask = isin(target, keep_indices)
        return (input.argmax(dim=1)[targ_mask] == target[targ_mask]).float().mean()
