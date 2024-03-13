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

# Based on https://github.com/meteorshowers/hed-pytorch/

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from fastai.callbacks.hooks import hook_outputs
from fastai.callbacks.hooks import model_sizes
from fastai.vision.models.unet import _get_sfs_idxs
from fastprogress.fastprogress import progress_bar
from fastai.vision import flatten_model
from ._timm_utils import get_backbone
from fastai.basic_train import LearnerCallback
from torch.nn.parallel import DistributedDataParallel


def modify_layers(backbone, backbone_fn):
    hookable_modules = flatten_model(backbone)
    for i, module in enumerate(hookable_modules):
        if (
            isinstance(module, nn.MaxPool2d)
            and not "xception" in backbone_fn.__module__
        ):
            module.ceil_mode = True
            module.kernel_size = 2
        elif isinstance(module, nn.Conv2d) and i == 0:
            module.stride = (1, 1)


def get_hooks(backbone, chip_size):
    try:
        hookable_modules = flatten_model(backbone)
        feature_sizes = model_sizes(
            nn.Sequential(*hookable_modules), size=(chip_size, chip_size)
        )
        f_change_idxs = _get_sfs_idxs(feature_sizes)
        hooks = [hookable_modules[i] for i in f_change_idxs + [-1]][:5]
        return hooks
    except:
        hookable_modules = list(backbone.children())
        feature_sizes = model_sizes(
            nn.Sequential(*hookable_modules), size=(chip_size, chip_size)
        )
        f_change_idxs = _get_sfs_idxs(feature_sizes)
        if len(f_change_idxs) > 1 and f_change_idxs[0] == f_change_idxs[1]:
            del f_change_idxs[0]
        if len(f_change_idxs) > 3:
            hooks = [hookable_modules[i] for i in f_change_idxs + [-1]][:5]
            return hooks

        def get_len(m):
            try:
                return len(m)
            except:
                return 0

        module_len = [get_len(m) for m in hookable_modules]
        hooks = []
        idx = 0
        while len(hooks) < 5 and len(hookable_modules) > idx:
            if module_len[idx] == 0:
                try:
                    if module_len[idx + 1] > 0:
                        hooks.append(hookable_modules[idx])
                except:
                    hooks.append(hookable_modules[idx])
            else:
                feature_sizes = model_sizes(
                    nn.Sequential(*hookable_modules[idx]), size=(chip_size, chip_size)
                )
                if len(feature_sizes) < 2:
                    hooks.append(hookable_modules[idx])
                else:
                    f_change_idxs = _get_sfs_idxs(feature_sizes)
                    if len(f_change_idxs) > 1 and f_change_idxs[0] == f_change_idxs[1]:
                        del f_change_idxs[0]
                    if f_change_idxs == []:
                        hooks.append(hookable_modules[idx])
                    int_idx = 0
                    while len(hooks) < 5 and len(f_change_idxs) > int_idx:
                        hooks.append(hookable_modules[idx][f_change_idxs[int_idx]])
                        int_idx += 1
            idx += 1
        if len(hooks) < 5:
            hooks.append(hookable_modules[idx - 1][-1])
        return hooks


class _HEDModel(nn.Module):
    def __init__(self, backbone_fn, chip_size=224, pretrained=True):
        super().__init__()
        self.backbone = get_backbone(backbone_fn, pretrained)
        modify_layers(self.backbone, backbone_fn)
        if len(self.backbone) < 2:
            self.backbone = self.backbone[0]
        hooks = get_hooks(self.backbone, chip_size)
        self.hook = hook_outputs(hooks)
        model_sizes(self.backbone, size=(chip_size, chip_size))
        layer_num_channels = [k.stored.shape[1] for k in self.hook]

        self.score_dsn1 = nn.Conv2d(layer_num_channels[0], 1, 1)
        self.score_dsn2 = nn.Conv2d(layer_num_channels[1], 1, 1)
        self.score_dsn3 = nn.Conv2d(layer_num_channels[2], 1, 1)
        self.score_dsn4 = nn.Conv2d(layer_num_channels[3], 1, 1)
        self.score_dsn5 = nn.Conv2d(layer_num_channels[4], 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]
        x = self.backbone(x)
        features = self.hook.stored

        so1 = self.score_dsn1(features[0])
        so2 = self.score_dsn2(features[1])
        so3 = self.score_dsn3(features[2])
        so4 = self.score_dsn4(features[3])
        so5 = self.score_dsn5(features[4])

        weight_deconv2 = make_bilinear_weights(4, 1).to(x.device)
        weight_deconv3 = make_bilinear_weights(8, 1).to(x.device)
        weight_deconv4 = make_bilinear_weights(16, 1).to(x.device)
        weight_deconv5 = make_bilinear_weights(32, 1).to(x.device)

        upsample2 = torch.nn.functional.conv_transpose2d(so2, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5, weight_deconv5, stride=16)

        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        if so1.size(2) != so2.size(2):
            so1 = F.interpolate(
                so1, (img_H, img_W), mode="bilinear", align_corners=False
            )

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)

        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]

        return results


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.0))
    y1 = int(round((h - th) / 2.0))
    return variable[:, :, y1 : y1 + th, x1 : x1 + tw]


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def cross_entropy_loss(prediction, label):
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask
    )  # , reduce=False)
    return torch.sum(cost)


def hed_loss(out, labels):
    loss = 0.0
    for o in out:
        loss = loss + cross_entropy_loss(o, labels)
    return loss


def accuracy(input, target):
    if isinstance(input, tuple):  # while training
        input = input[0]
    target = target.byte().squeeze(1)
    input = input[-1]
    input = (input >= 0.5).byte().squeeze(1)
    return (input == target).float().mean()


def get_true_positive(mask1, mask2, buffer):
    tp = 0
    indices = np.where(mask1 == 1)
    for ind in range(len(indices[0])):
        tp += np.any(
            mask2[
                max(indices[0][ind] - buffer, 0) : indices[0][ind] + buffer + 1,
                max(indices[1][ind] - buffer, 0) : indices[1][ind] + buffer + 1,
            ]
        ).astype(np.int)
    return tp


def get_confusion_metric(gt, pred, buffer):
    from skimage.morphology import skeletonize, binary_dilation

    tp, predicted_tp, actual_tp = 0, 0, 0
    for i in range(gt.shape[0]):
        gt_mask = skeletonize(binary_dilation(gt[i]))
        pred_mask = skeletonize(binary_dilation(pred[i]))
        tp += get_true_positive(gt_mask, pred_mask, buffer)
        predicted_tp += len(np.where(pred_mask == 1)[0])
        actual_tp += len(np.where(gt_mask == 1)[0])

    return tp, predicted_tp, actual_tp


def f1_score(pred, gt):
    device = gt.device
    gt = gt.byte().squeeze(1).cpu().numpy()
    pred = (pred[-1] >= 0.5).byte().squeeze(1).cpu().numpy()
    tp, predicted_tp, actual_tp = get_confusion_metric(gt, pred, 3)
    precision = tp / (predicted_tp + 1e-12)
    recall = tp / (actual_tp + 1e-12)
    f1score = 2 * precision * recall / (precision + recall + 1e-12)
    return torch.tensor(f1score).to(device)


def accuracies(model, dl, detect_thresh=0.5, buffer=3, show_progress=True):
    precision, recall, f1score = [], [], []
    model.learn.model.eval()
    acc = {}
    with torch.no_grad():
        for input, gt in progress_bar(dl, display=show_progress):
            predictions = model.learn.model(input)
            gt = gt.byte().squeeze(1).cpu().numpy()
            pred = (predictions[-1] >= detect_thresh).byte().squeeze(1).cpu().numpy()
            tp, predicted_tp, actual_tp = get_confusion_metric(gt, pred, buffer)
            prec = tp / (predicted_tp + 1e-12)
            rec = tp / (actual_tp + 1e-12)
            precision.append(prec)
            recall.append(rec)
            f1score.append(2 * prec * rec / (prec + rec + 1e-12))
    acc["Precision"] = np.mean(precision)
    acc["Recall"] = np.mean(recall)
    acc["F1 Score"] = np.mean(f1score)

    return acc


class DDPCallback(LearnerCallback):
    def __init__(self, learn, cuda_id):
        super().__init__(learn)
        self.cuda_id = cuda_id

    def on_train_begin(self, **kwargs):
        self.learn.model = DistributedDataParallel(
            self.learn.model.module,
            device_ids=[self.cuda_id],
            output_device=self.cuda_id,
            find_unused_parameters=True,
        )
