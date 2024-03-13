"""
The code is borrowed from:
-------------------------------------------------------------------------------
https://github.com/anilbatra2185/road_connectivity/blob/master/model/linknet.py
"""

import math
from typing import Tuple

import numpy as np
import torch

# from arcgis.learn._utils.segmentation_loss_functions import mIoULoss
from arcgis.learn.models._deeplab_utils import mask_iou
from fastai.callbacks.hooks import hook_outputs, model_sizes
from fastprogress.fastprogress import progress_bar
from torch import nn


class DecoderBlock(nn.Module):
    """
    Decoder Block to perform Transposed Convolutions
    @param in_channels: Number of input channels
    @param out_channels: Number of output channels
    @param group: grouping of channels used in convolution
                Defaut is set to 1 i.e. vanilla convolution
    """

    def __init__(self, in_channels, out_channels, hook=None, group=1):
        super(DecoderBlock, self).__init__()

        self.hook = hook
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, groups=group)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=group,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1, groups=group)
        self.norm3 = nn.BatchNorm2d(out_channels)
        # self.up    = nn.Upsample(upsample_size)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        if self.hook is not None:
            self.up = nn.Upsample(self.hook.stored.shape[2])
            x = self.up(x)
        x = self.relu3(x)
        if self.hook:
            x = x + self.hook.stored
        return x


class LinkNetModel(nn.Module):
    """
    Implements LinkNet model.
    Reference: https://arxiv.org/pdf/1707.03718.pdf
    """

    def __init__(self, encoder, n_classes, chip_size, n_bands=3):
        super(LinkNetModel, self).__init__()

        self.chip_size = chip_size
        # Fetch the sizes of various activation layers of the backbone
        sfs_sizes = model_sizes(encoder, size=self.chip_size)
        filters = [x[1] for x in sfs_sizes[-4:]]

        # Encoder
        self.encoder = encoder
        self.encoder_outputs = hook_outputs(self.encoder)

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2], self.encoder_outputs[-2])
        self.decoder3 = DecoderBlock(filters[2], filters[1], self.encoder_outputs[-3])
        self.decoder2 = DecoderBlock(filters[1], filters[0], self.encoder_outputs[-4])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, n_classes, 2, padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in [self.finaldeconv1, self.finalconv2]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]

        e = self.encoder(x)
        d4 = self.decoder4(e)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5[:, :, :rows, :cols]


class LinkNetMultiTaskModel(nn.Module):
    """
    Implements LinkNet MultiTask model.
    Reference: https://arxiv.org/pdf/1707.03718.pdf
    """

    def __init__(
        self, encoder, task1_classes, task2_classes, chip_size, n_bands=3, is_timm=False
    ):
        super(LinkNetMultiTaskModel, self).__init__()

        self.chip_size = chip_size
        self._is_timm = is_timm
        self.encoder = encoder

        if self._is_timm:
            from ._hed_utils import get_hooks

            if len(self.encoder) < 2:
                self.encoder = self.encoder[0]
            hooks = get_hooks(self.encoder, chip_size[0])
            self.encoder_outputs = hook_outputs(hooks)[1:]
            model_sizes(self.encoder, size=chip_size)
            filters = [k.stored.shape[1] for k in self.encoder_outputs]
        else:
            # Fetch the sizes of various activation layers of the backbone
            sfs_sizes = model_sizes(encoder, size=self.chip_size)
            filters = [x[1] for x in sfs_sizes[-4:]]
            self.encoder_outputs = hook_outputs(self.encoder)

        # Decoder for Road Segmentation
        self.decoder4 = DecoderBlock(filters[3], filters[2], self.encoder_outputs[-2])
        self.decoder3 = DecoderBlock(filters[2], filters[1], self.encoder_outputs[-3])
        self.decoder2 = DecoderBlock(filters[1], filters[0], self.encoder_outputs[-4])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier for Road Segmentation
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

        # Decoder for Road Orientation
        self.o_decoder4 = DecoderBlock(filters[3], filters[2], self.encoder_outputs[-2])
        self.o_decoder3 = DecoderBlock(filters[2], filters[1], self.encoder_outputs[-3])
        self.o_decoder2 = DecoderBlock(filters[1], filters[0], self.encoder_outputs[-4])
        self.o_decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier for Road Orientation
        self.o_finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.o_finalrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.o_finalconv2 = nn.Conv2d(32, 32, 3)
        self.o_finalrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.o_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

        for m in [
            self.finaldeconv1,
            self.finalconv2,
            self.o_finaldeconv1,
            self.o_finalconv2,
        ]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]

        if self._is_timm:
            if x.shape[0] < 2 and self.training:
                e = self.encoder(torch.cat((x, x)))
            else:
                e = self.encoder(x)
            e = self.encoder_outputs[-1].stored
        else:
            e = self.encoder(x)
        # Decoder - Road Segmentation
        d4 = self.decoder4(e)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)

        # Final Classification - Road Segmentation
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        # Decoder - Road Orientation
        o_d4 = self.o_decoder4(e)
        o_d3 = self.o_decoder3(o_d4)
        o_d2 = self.o_decoder2(o_d3)
        o_d1 = self.o_decoder1(o_d2)

        # Final Classification - Road Segmentation
        o_f1 = self.o_finaldeconv1(o_d1)
        o_f2 = self.o_finalrelu1(o_f1)
        o_f3 = self.o_finalconv2(o_f2)
        o_f4 = self.o_finalrelu2(o_f3)
        o_f5 = self.o_finalconv3(o_f4)

        if self._is_timm and x.shape[0] < 2 and self.training:
            f5 = f5[0][None]
            o_f5 = o_f5[0][None]

        return f5[:, :, :rows, :cols], o_f5[:, :, :rows, :cols]


class road_orient_loss(nn.Module):
    """
    Custom PyTorch Loss Module to compute loss for backpropagation of multi-task
    outputs. It compose of Mean-IoU and CrossEntropy losses.
    In current version it only supports Mean-IoU for Road Segmentation and
    Cross-Entropy for Road Orientations.
    """

    def __init__(self, n_classes: int = 2, loss_weights: Tuple = (1.0, 1.0)):
        super().__init__()
        self.n_classes = n_classes
        self.loss_weights = loss_weights if loss_weights else (1.0, 1.0)
        self.road_loss = nn.CrossEntropyLoss()  # mIoULoss(self.n_classes)
        self.orient_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, *target):
        if isinstance(predictions, (list, tuple)):
            if len(target) > 1:
                seg_target, orient_target = target
                return self.loss_weights[0] * self.road_loss(
                    predictions[0], seg_target.squeeze(1).long()
                ) + self.loss_weights[1] * self.orient_loss(
                    predictions[1], orient_target.squeeze(1).long()
                )
            else:
                seg_target = target[0]
                return self.loss_weights[0] * self.road_loss(
                    predictions[0], seg_target.squeeze(1).long()
                )
        else:
            return self.road_loss(predictions, target[0].squeeze(1).long())


def isin(target, keep_indices):
    old_shape = target.shape
    mask = torch.cat(
        [(target.view(-1) == k)[:, None] for k in keep_indices], dim=1
    ).any(1)
    mask = mask.view(old_shape).contiguous()
    return mask


def accuracy(input, *target, ignore_mapped_class=[]):
    """
    Compute pixel based accuracy
    """
    if isinstance(input, tuple):  # while training
        input = input[0]
    if isinstance(target, tuple):
        target = target[0]

    target = target.squeeze(1).cpu().long()
    input = input.cpu()
    if ignore_mapped_class == []:
        return (input.argmax(dim=1) == target).float().mean()
    else:
        _, total_classes, _, _ = input.shape
        keep_indices = [i for i in range(total_classes) if i not in ignore_mapped_class]
        for k in ignore_mapped_class:
            input[:, k] = input.min() - 1
        targ_mask = isin(target, keep_indices)
        return (input.argmax(dim=1)[targ_mask] == target[targ_mask]).float().mean()


def miou(prediction, *target, ignore_mapped_class=[], smooth=1e-8):
    """
    Compute pixel based Mean IoU
    """
    if isinstance(prediction, tuple):  # while training
        prediction = prediction[0]
    if isinstance(target, tuple):
        target = target[0]

    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

    target = target.squeeze(1).long()
    batch_size = prediction.size(0)
    classes = prediction.size(1)
    prediction = prediction.argmax(dim=1)

    histogram = fast_hist(
        prediction.view(batch_size, -1).cpu().numpy(),
        target.view(batch_size, -1).cpu().numpy(),
        classes,
    )
    iou_per_class = np.diag(histogram) / (
        histogram.sum(1) + histogram.sum(0) - np.diag(histogram)
    )

    if ignore_mapped_class == []:
        return torch.tensor(np.nanmean(iou_per_class))
    else:
        sum = 0.0
        mean_classes = 0
        for k in classes:
            if k not in ignore_mapped_class:
                mean_classes += 1
                sum += iou_per_class[k]
        if mean_classes == 0:
            return 0
        return torch.tensor(sum / mean_classes)


def compute_miou(model, dl, mean, num_classes, show_progress, ignore_mapped_class=[]):
    ious = []
    model.learn.model.eval()

    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

    with torch.no_grad():
        for input, target in progress_bar(dl, display=show_progress):
            pred = model.learn.model(input)
            if isinstance(pred, tuple):  # while training
                pred = pred[0]
            if isinstance(target, tuple):
                target = target[0]
            target = target[0].squeeze(1).long()
            if ignore_mapped_class != []:
                for k in ignore_mapped_class:
                    pred[:, k] = pred.min() - 1
                pred = pred.argmax(dim=1)
            else:
                pred = pred.argmax(dim=1)
            mask1 = []
            mask2 = []
            for i in range(pred.shape[0]):
                mask1.append(
                    pred[i].to(model._device)
                    == num_classes[:, None, None].to(model._device)
                )
                mask2.append(
                    target[i].to(model._device)
                    == num_classes[:, None, None].to(model._device)
                )
            mask1 = torch.stack(mask1)
            mask2 = torch.stack(mask2)
            iou = mask_iou(mask1, mask2)
            ious.append(iou.tolist())

        return np.mean(ious, 0)
