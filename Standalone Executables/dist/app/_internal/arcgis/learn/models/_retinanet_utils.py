"""
Apache License, Version 2.0 Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.
"License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.
"Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.
"Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.
"You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.
"Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.
"Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.
"Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).
"Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.
"Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
"Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.

2. Grant of Copyright License.
Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.

3. Grant of Patent License.
Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

4. Redistribution.
You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:
You must give any other recipients of the Work or Derivative Works a copy of this License; and You must cause any modified files to carry prominent notices stating that You changed the files; and You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License. You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

5. Submission of Contributions.
Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.

6. Trademarks.
This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty.
Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

8. Limitation of Liability.
In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability.
While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.

"""

import torch
from torch import nn, LongTensor, Tensor
import torch.nn.functional as F
from fastai.vision.image import ImageBBox
from fastai.vision.data import ObjectCategoryList, ObjectItemList
from fastai.vision.learner import create_body
import numpy as np
from fastai.callbacks.hooks import model_sizes, hook_outputs
from fastai.layers import conv2d, conv_layer
from fastai.core import ifnone, is_tuple, range_of
import math
import matplotlib.pyplot as plt
import warnings
import logging
from typing import Tuple, List
from fastai.basic_train import Callback
from fastai.torch_core import add_metrics
from ._timm_utils import _get_feature_size

from fastprogress.fastprogress import progress_bar


class LateralUpsampleMerge(nn.Module):
    def __init__(self, ch, ch_lat, hook):
        super().__init__()
        self.hook = hook
        self.conv_lat = conv2d(ch_lat, ch, ks=1, bias=True)

    def forward(self, x):
        scale_factor = 2
        # To catch warning from tensorboard
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            size_to_interpolate = tuple(
                [int(a) * scale_factor for a in [x.shape[2], x.shape[3]]]
            )
        # Interpolate the Lateral layer to match the size of the Upsample layer before merging them
        return F.interpolate(
            self.conv_lat(self.hook.stored), size=size_to_interpolate
        ) + F.interpolate(x, scale_factor=scale_factor)


class RetinaNetModel(nn.Module):
    "Implements RetinaNet from https://arxiv.org/abs/1708.02002"

    def __init__(
        self,
        backbone,
        backbone_pretrained,
        backbone_cut,
        n_classes,
        final_bias=0.0,
        chs=256,
        n_anchors=9,
        flatten=True,
        chip_size=(256, 256),
        n_bands=3,
    ):
        # chs - channels for top down layers in FPN

        super().__init__()
        self.n_classes, self.flatten = n_classes, flatten
        self.chip_size = chip_size
        encoder = create_body(backbone, backbone_pretrained, backbone_cut)

        # Fetch the sizes of various activation layers of the backbone
        sfs_szs = _get_feature_size(
            backbone,
            cut=backbone_cut,
            chip_size=self.chip_size,
        )

        hooks = hook_outputs(nn.Sequential(*encoder.children()))

        self.encoder = encoder
        self.c5top5 = conv2d(sfs_szs[-1][1], chs, ks=1, bias=True)
        self.c5top6 = conv2d(sfs_szs[-1][1], chs, stride=2, bias=True)
        self.p6top7 = nn.Sequential(nn.ReLU(), conv2d(chs, chs, stride=2, bias=True))
        self.merges = nn.ModuleList(
            [
                LateralUpsampleMerge(chs, szs[1], hook)
                for szs, hook in zip(sfs_szs[-2:-4:-1], hooks[-2:-4:-1])
            ]
        )
        self.smoothers = nn.ModuleList(
            [conv2d(chs, chs, 3, bias=True) for _ in range(3)]
        )
        self.classifier = self._head_subnet(n_classes, n_anchors, final_bias, chs=chs)
        self.box_regressor = self._head_subnet(4, n_anchors, 0.0, chs=chs)

        # Create a dummy x to be passed through the model and fetch the sizes
        x_dummy = torch.rand(2, n_bands, self.chip_size[0], self.chip_size[1])
        p_states = self._create_p_states(x_dummy)
        self.sizes = [[p.size(2), p.size(3)] for p in p_states]

    def _head_subnet(self, n_classes, n_anchors, final_bias=0.0, n_conv=4, chs=256):
        layers = [conv_layer(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        layers[-1].weight.data.fill_(0)
        return nn.Sequential(*layers)

    def _apply_transpose(self, func, p_states, n_classes):
        if not self.flatten:
            sizes = [[p.size(0), p.size(2), p.size(3)] for p in p_states]
            return [
                func(p).permute(0, 2, 3, 1).view(*sz, -1, n_classes)
                for p, sz in zip(p_states, sizes)
            ]
        else:
            return torch.cat(
                [
                    func(p)
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .view(p.size(0), -1, n_classes)
                    for p in p_states
                ],
                1,
            )

    def _create_p_states(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5top5(c5.clone()), self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[-1]))
        for merge in self.merges:
            p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        return p_states

    def forward(self, x):
        p_states = self._create_p_states(x)
        return [
            self._apply_transpose(self.classifier, p_states, self.n_classes),
            self._apply_transpose(self.box_regressor, p_states, 4),
        ]


#########################
## Functions for Anchors
#########################


@torch.jit.script
def create_grid(size: Tuple[int, int]):
    "Create a grid of a given `size`."
    # print(size)
    out_size = size if isinstance(size, tuple) else (size, size)  # TODO: here
    # grid = torch.FloatTensor(H, W, 2)#here
    H = int(out_size[0])
    W = int(out_size[1])
    # print(type(H),H,out_size)

    grid = torch.empty((H, W, 2)).float()
    linear_points = (
        torch.linspace(-1 + 1 / W, 1 - 1 / W, W) if W > 1 else torch.tensor([0.0])
    )
    grid[:, :, 1] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, 0])
    linear_points = (
        torch.linspace(-1 + 1 / H, 1 - 1 / H, H) if H > 1 else torch.tensor([0.0])
    )
    grid[:, :, 0] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1, 2)


def show_anchors(ancs, size):
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xticks(np.linspace(-1, 1, size[1] + 1))
    ax.set_yticks(np.linspace(-1, 1, size[0] + 1))
    ax.grid()
    ax.scatter(ancs[:, 1], ancs[:, 0])  # y is first
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)  # -1 is top, 1 is bottom
    for i, (x, y) in enumerate(zip(ancs[:, 1], ancs[:, 0])):
        ax.annotate(i, xy=(x, y))


def get_anchors(anchors: List[Tensor], flatten: bool):
    if torch.jit.is_scripting():
        return torch.cat([anc.view(-1, 4) for anc in anchors], 0)
    else:
        return (
            torch.cat([anc.view(-1, 4) for anc in anchors], 0) if flatten else anchors
        )


@torch.jit.script
def create_anchors(
    sizes: List[List[int]],
    ratios: List[float],
    scales: List[float],
    flatten: bool = True,
):
    "Create anchor of `sizes`, `ratios` and `scales`."
    aspects = [
        [[s * math.sqrt(r), s * math.sqrt(1 / r)] for s in scales] for r in ratios
    ]
    aspects = torch.tensor(aspects).view(-1, 2)
    anchors = []
    for idx, size in enumerate(sizes):
        # 4 here to have the anchors overlap.
        h = size[0]
        w = size[1]
        sized_aspects = 4 * (aspects * torch.tensor([2 / h, 2 / w])).unsqueeze(0)
        base_grid = create_grid((h, w)).unsqueeze(1)  # TODO
        n, a = base_grid.size(0), aspects.size(0)
        ancs = torch.cat([base_grid.expand(n, a, 2), sized_aspects.expand(n, a, 2)], 2)
        anchors.append(ancs.view(h, w, a, 4))
    return get_anchors(anchors, flatten)


def activ_to_bbox(acts, anchors, flatten: bool = True):
    "Extrapolate bounding boxes on anchors from the model activations."
    if torch.jit.is_scripting():
        with torch.no_grad():
            # acts.mul_(acts.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
            acts.mul_(
                torch.tensor(
                    [[0.1, 0.1, 0.2, 0.2]], dtype=acts.dtype, device=acts.device
                )
            )
            centers = anchors[..., 2:] * acts[..., :2] + anchors[..., :2]
            sizes = anchors[..., 2:] * torch.exp(acts[..., :2])
        return torch.cat([centers, sizes], -1)
    else:
        if flatten:
            with torch.no_grad():
                acts.mul_(acts.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
                centers = anchors[..., 2:] * acts[..., :2] + anchors[..., :2]
                sizes = anchors[..., 2:] * torch.exp(acts[..., :2])
            return torch.cat([centers, sizes], -1)
        else:
            return [activ_to_bbox(act, anc) for act, anc in zip(acts, anchors)]


def cthw2tlbr(boxes):
    "Convert center/size format `boxes` to top/left bottom/right corners."
    top_left = boxes[:, :2] - boxes[:, 2:] / 2
    bot_right = boxes[:, :2] + boxes[:, 2:] / 2
    return torch.cat([top_left, bot_right], 1)


def intersection(anchors, targets):
    "Compute the sizes of the intersections of `anchors` by `targets`."
    ancs, tgts = cthw2tlbr(anchors), cthw2tlbr(targets)
    a, t = ancs.size(0), tgts.size(0)
    ancs, tgts = ancs.unsqueeze(1).expand(a, t, 4), tgts.unsqueeze(0).expand(a, t, 4)
    top_left_i = torch.max(ancs[..., :2], tgts[..., :2])
    bot_right_i = torch.min(ancs[..., 2:], tgts[..., 2:])
    sizes = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[..., 0] * sizes[..., 1]


def IoU_values(anchors, targets):
    "Compute the IoU values of `anchors` by `targets`."
    inter = intersection(anchors, targets)
    anc_sz, tgt_sz = anchors[:, 2] * anchors[:, 3], targets[:, 2] * targets[:, 3]
    union = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter
    return inter / (union + 1e-8)


def match_anchors(anchors, targets, match_thr=0.5, bkg_thr=0.4):
    "Match `anchors` to targets. -1 is match to background, -2 is ignore."
    ious = IoU_values(anchors, targets)
    matches = anchors.new(anchors.size(0)).zero_().long() - 2
    vals, idxs = torch.max(ious, 1)
    matches[vals < bkg_thr] = -1
    matches[vals > match_thr] = idxs[vals > match_thr]
    return matches


def tlbr2cthw(boxes):
    "Convert top/left bottom/right format `boxes` to center/size corners."
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([center, sizes], 1)


def bbox_to_activ(bboxes, anchors, flatten=True):
    "Return the target of the model on `anchors` for the `bboxes`."
    if flatten:
        t_centers = (bboxes[..., :2] - anchors[..., :2]) / anchors[..., 2:]
        t_sizes = torch.log(bboxes[..., 2:] / anchors[..., 2:] + 1e-8)
        return torch.cat([t_centers, t_sizes], -1).div_(
            bboxes.new_tensor([[0.1, 0.1, 0.2, 0.2]])
        )
    else:
        return [activ_to_bbox(act, anc) for act, anc in zip(acts, anchors)]


def encode_class(idxs, n_classes):
    target = idxs.new_zeros(len(idxs), n_classes).float()
    mask = idxs != 0
    i1s = LongTensor(list(range(len(idxs))))
    target[i1s[mask], idxs[mask] - 1] = 1
    return target


###############
# Focal Loss
###############


class RetinaNetFocalLoss(nn.Module):
    def __init__(
        self,
        sizes,
        scales,
        ratios,
        device,
        gamma=2.0,
        alpha=0.25,
        pad_idx=0,
        reg_loss=F.smooth_l1_loss,
    ):
        super().__init__()
        self.gamma, self.alpha, self.pad_idx, self.reg_loss = (
            gamma,
            alpha,
            pad_idx,
            reg_loss,
        )
        self.sizes = sizes
        self.scales = scales
        self.ratios = ratios
        self._device = device
        self._create_anchors(self.sizes, self._device)

    def _change_anchors(self, sizes) -> bool:
        if not hasattr(self, "sizes"):
            return True
        for sz1, sz2 in zip(self.sizes, sizes):
            if sz1[0] != sz2[0] or sz1[1] != sz2[1]:
                return True
        return False

    def _create_anchors(self, sizes, device: torch.device):
        self.anchors = create_anchors(sizes, self.ratios, self.scales).to(device)

    def _unpad(self, bbox_tgt, clas_tgt):
        non_zero = torch.nonzero(clas_tgt - self.pad_idx)
        i = bbox_tgt.shape[0] if non_zero.nelement() == 0 else torch.min(non_zero)
        return tlbr2cthw(bbox_tgt[i:]), clas_tgt[i:] - 1 + self.pad_idx

    def _focal_loss(self, clas_pred, clas_tgt):
        encoded_tgt = encode_class(clas_tgt, clas_pred.size(1))
        ps = torch.sigmoid(clas_pred)
        weights = encoded_tgt * (1 - ps) + (1 - encoded_tgt) * ps
        alphas = (1 - encoded_tgt) * self.alpha + encoded_tgt * (1 - self.alpha)
        weights.pow_(self.gamma).mul_(alphas)
        clas_loss = F.binary_cross_entropy_with_logits(
            clas_pred, encoded_tgt, weights.detach(), reduction="sum"
        )
        return clas_loss

    def _one_loss(self, clas_pred, bbox_pred, clas_tgt, bbox_tgt):
        bbox_tgt, clas_tgt = self._unpad(bbox_tgt, clas_tgt)
        try:
            matches = match_anchors(self.anchors, bbox_tgt)
        except:
            return torch.tensor(0.0, requires_grad=True).to(self._device)

        bbox_mask = matches >= 0
        if bbox_mask.sum() != 0:
            bbox_pred = bbox_pred[bbox_mask]
            bbox_tgt = bbox_tgt[matches[bbox_mask]]
            bb_loss = self.reg_loss(
                bbox_pred, bbox_to_activ(bbox_tgt, self.anchors[bbox_mask])
            )
        else:
            bb_loss = torch.tensor(0.0).to(self._device)
        matches.add_(1)
        clas_tgt = clas_tgt + 1
        clas_mask = matches >= 0
        clas_pred = clas_pred[clas_mask]
        clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
        clas_tgt = clas_tgt[matches[clas_mask]]
        final_lloss = bb_loss + self._focal_loss(clas_pred, clas_tgt) / torch.clamp(
            bbox_mask.sum(), min=1.0
        )
        return final_lloss

    def forward(self, output, bbox_tgts, clas_tgts):
        clas_preds, bbox_preds = output
        return sum(
            [
                self._one_loss(cp, bp, ct, bt)
                for (cp, bp, ct, bt) in zip(
                    clas_preds, bbox_preds, clas_tgts, bbox_tgts
                )
            ]
        ) / clas_tgts.size(0)


######################
## INFERENCE functions
######################


def nms(boxes, scores, thresh: float = 0.2):
    idx_sort = scores.argsort(descending=True)
    boxes, scores = boxes[idx_sort], scores[idx_sort]
    to_keep, indexes = [], torch.tensor(range_of(scores)).long()
    while len(scores) > 0:
        to_keep.append(idx_sort[indexes[0]])
        iou_vals = IoU_values(boxes, boxes[:1]).squeeze()
        mask_keep = iou_vals <= thresh
        if len(mask_keep.nonzero()) == 0:
            break
        boxes, scores, indexes = boxes[mask_keep], scores[mask_keep], indexes[mask_keep]
    to_keep = [int(idx.item()) for idx in to_keep]
    return torch.tensor(to_keep).long()


def process_output(output, detect_thresh=0.25, crit=None):
    clas_pred, bbox_pred, sizes = output[0], output[1], crit.sizes
    anchors = create_anchors(sizes, crit.ratios, crit.scales).to(clas_pred.device)
    bbox_pred = activ_to_bbox(bbox_pred, anchors)
    clas_pred = torch.sigmoid(clas_pred)
    detect_mask = clas_pred.max(1)[0] > detect_thresh
    bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]
    bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))

    # Handling the case when the there are no predictions on an image
    if clas_pred.shape[0] == 0:
        scores = clas_pred.squeeze()
        preds = torch.zeros(clas_pred.shape).long().squeeze()
    else:
        scores, preds = clas_pred.max(1)

    return bbox_pred, scores, preds


def get_predictions(output, detect_thresh=0.2, crit=None, nms_overlap=0.1):
    bbox_pred, scores, preds = process_output(output, detect_thresh, crit=crit)

    # Filter out the predicted boxes with size zero
    mask_keep = (bbox_pred[:, 2] * bbox_pred[:, 3]) != 0
    bbox_pred, preds, scores = bbox_pred[mask_keep], preds[mask_keep], scores[mask_keep]

    # Apply nms
    to_keep = nms(bbox_pred, scores, thresh=nms_overlap)
    bbox_pred, preds, scores = (
        bbox_pred[to_keep].cpu(),
        preds[to_keep].cpu(),
        scores[to_keep].cpu(),
    )

    # Convert the bbox predictions to TL-BR to be passed to ImageBBox Class in fastai through reconstruct
    bbox_pred = cthw2tlbr(bbox_pred)
    # Add 1 to class predictions to account for prepending of background as a class
    preds += 1

    return bbox_pred, preds, scores


#########################
## mAP functions
#########################


def unpad(tgt_bbox, tgt_clas, pad_idx=0):
    i = torch.min(torch.nonzero(tgt_clas - pad_idx))
    return tlbr2cthw(tgt_bbox[i:]), tgt_clas[i:] - 1 + pad_idx


def _get_y(bbox, clas):
    "Unpads the targets - undoes the earlier addition of sparse data to make a batch consistent"
    try:
        bbox = bbox.view(-1, 4)  # /sz
    except Exception:
        bbox = torch.zeros(size=[0, 4])
    bb_keep = ((bbox[:, 2] - bbox[:, 0]) > 0).nonzero()[:, 0]
    return bbox[bb_keep], clas[bb_keep]


class AveragePrecision(Callback):
    def __init__(self, model, n_classes):
        self.model = model
        self.n_classes = n_classes

    def on_epoch_begin(self, **kwargs):
        self.tps, self.clas, self.p_scores = [], [], []
        self.classes, self.n_gts = (
            LongTensor(range(self.n_classes)),
            torch.zeros(self.n_classes).long(),
        )

    def on_batch_end(self, last_output, last_target, **kwargs):
        tps, p_scores, clas, self.n_gts = compute_cm(
            self.model, last_output, last_target, self.n_gts, self.classes
        )
        self.tps.extend(tps)
        self.p_scores.extend(p_scores)
        self.clas.extend(clas)

    def on_epoch_end(self, last_metrics, **kwargs):
        aps = compute_ap_score(
            self.tps, self.p_scores, self.clas, self.n_gts, self.n_classes
        )
        aps = torch.mean(torch.tensor(aps))
        return add_metrics(last_metrics, aps)


def compute_class_AP(
    model, dl, n_classes, show_progress, iou_thresh=0.1, detect_thresh=0.5, num_keep=100
):
    tps, clas, p_scores = [], [], []
    classes, n_gts = LongTensor(range(n_classes)), torch.zeros(n_classes).long()
    model.learn.model.eval()

    with torch.no_grad():
        for input, target in progress_bar(dl, display=show_progress):
            # input - 4(batch-size),3,256,256
            # target - 2(regression,classification), 4(batch-size), 3/4/2(max no of detections in the batch), 4/1(bbox,class)
            output = model.learn.pred_batch(batch=(input, target))

            tps1, p_scores1, clas1, n_gts = compute_cm(
                model, output, target, n_gts, classes, iou_thresh, detect_thresh
            )
            tps.extend(tps1)
            p_scores.extend(p_scores1)
            clas.extend(clas1)

        aps = compute_ap_score(tps, p_scores, clas, n_gts, n_classes)
        return aps


def compute_cm(
    model, output, target, n_gts, classes, iou_thresh=0.1, detect_thresh=0.5
):
    tps, clas, p_scores = [], [], []
    for i in range(target[0].size(0)):  # range batch-size
        # output[0] - classpreds, output[1] - bbox preds
        op = model._data.y.analyze_pred(
            (output[0][i], output[1][i]),
            model=model,
            thresh=detect_thresh,
            nms_overlap=iou_thresh,
            ret_scores=True,
            device=model._device,
        )
        # op - bbox preds, class preds, scores

        # Unpad the targets
        tgt_bbox, tgt_clas = _get_y(target[0][i], target[1][i])

        try:
            bbox_pred, preds, scores = op
            if len(bbox_pred) != 0 and len(tgt_bbox) != 0:
                bbox_pred = bbox_pred.to(model._device)
                preds = preds.to(model._device)
                tgt_bbox = tgt_bbox.to(model._device)

                # Convert the bbox coordinates to center-height-width(cthw) before calculating Intersection Over Union
                ious = IoU_values(tlbr2cthw(bbox_pred), tlbr2cthw(tgt_bbox))
                max_iou, matches = ious.max(1)
                detected = []

                for i in range(len(preds)):
                    if (
                        max_iou[i] >= iou_thresh
                        and matches[i] not in detected
                        and tgt_clas[matches[i]] == preds[i]
                    ):
                        detected.append(matches[i])
                        tps.append(1)
                    else:
                        tps.append(0)
                clas.append(preds.cpu())
                p_scores.append(scores.cpu())
        except:
            pass
        n_gts += ((tgt_clas.cpu()[:, None] - 1) == classes[None, :]).sum(0)

    return tps, p_scores, clas, n_gts


def compute_ap_score(tps, p_scores, clas, n_gts, n_classes):
    # If no true positives are found return an average precision score of 0.
    if len(tps) == 0:
        return [0.0 for cls in range(1, n_classes + 1)]

    tps, p_scores, clas = torch.tensor(tps), torch.cat(p_scores, 0), torch.cat(clas, 0)
    fps = 1 - tps
    idx = p_scores.argsort(descending=True)
    tps, fps, clas = tps[idx], fps[idx], clas[idx]
    aps = []

    for cls in range(1, n_classes + 1):
        tps_cls, fps_cls = (
            tps[clas == cls].float().cumsum(0),
            fps[clas == cls].float().cumsum(0),
        )
        if tps_cls.numel() != 0 and tps_cls[-1] != 0:
            precision = tps_cls / (tps_cls + fps_cls + 1e-8)
            recall = tps_cls / (n_gts[cls - 1] + 1e-8)
            aps.append(compute_ap(precision, recall))
        else:
            aps.append(0.0)
    return aps


def compute_ap(precision, recall):
    "Compute the average precision for `precision` and `recall` curve."
    recall = np.concatenate(([0.0], list(recall), [1.0]))
    precision = np.concatenate(([0.0], list(precision), [0.0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return ap


try:

    @torch.jit.script
    def _process_output_jit(
        output: Tuple[Tensor, Tensor], detect_thresh: float, crit_vals: List[Tensor]
    ):
        clas_pred = output[0].clone().detach()
        bbox_pred = output[1].clone().detach()
        sizes: List[List[int]] = crit_vals[0].clone().detach().tolist()
        ratios: List[float] = crit_vals[1].clone().detach().tolist()
        scales: List[float] = crit_vals[2].clone().detach().tolist()
        anchors = create_anchors(sizes, ratios, scales).to(clas_pred.device)
        bbox_pred = activ_to_bbox(bbox_pred, anchors)
        clas_pred = torch.sigmoid(clas_pred)
        detect_mask = clas_pred.max(1)[0] > detect_thresh
        bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]
        bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))
        # Handling the case when the there are no predictions on an image
        if clas_pred.shape[0] == 0:
            scores = clas_pred.squeeze()
            preds = torch.zeros(clas_pred.shape).long().squeeze()
        else:
            scores, preds = clas_pred.max(1)

        return bbox_pred, scores, preds

    @torch.jit.script
    def _get_predictions_jit(
        output: Tuple[Tensor, Tensor],
        detect_thresh: float,
        nms_overlap: float,
        crit_vals: List[Tensor],
    ):
        bbox_pred, scores, preds = _process_output_jit(output, detect_thresh, crit_vals)
        device = output[0].device

        # Filter out the predicted boxes with size zero
        mask_keep = (bbox_pred[:, 2] * bbox_pred[:, 3]) != 0
        bbox_pred, preds, scores = (
            bbox_pred[mask_keep],
            preds[mask_keep],
            scores[mask_keep],
        )

        # Apply nms
        to_keep = nms(bbox_pred, scores, thresh=nms_overlap)

        bbox_pred, preds, scores = (
            bbox_pred[to_keep].to(device),
            preds[to_keep].to(device),
            scores[to_keep].to(device),
        )

        # Convert the bbox predictions to TL-BR to be passed to ImageBBox Class in fastai through reconstruct
        bbox_pred = cthw2tlbr(bbox_pred)
        # Add 1 to class predictions to account for prepending of background as a class
        preds += 1

        return bbox_pred, preds, scores

    @torch.jit.script
    def _analyze_pred_jit(
        pred: Tuple[Tensor, Tensor],
        thresh: float,
        nms_overlap: float,
        crit_vals: List[Tensor],
    ):
        return _get_predictions_jit(
            pred, detect_thresh=thresh, nms_overlap=nms_overlap, crit_vals=crit_vals
        )

    def _post_process(bboxes):
        # print(bboxes)
        processed_bboxes = []
        for bbox in bboxes:
            bbox = (bbox + 1) / 2
            output = torch.clone(bbox)
            bbox[0] = output[1]
            bbox[1] = output[0]
            bbox[2] = output[3]
            bbox[3] = output[2]
            processed_bboxes.append(bbox)

        out_bboxes = torch.stack([bbox for bbox in processed_bboxes])
        return out_bboxes

    def _reconstruct_jit(t: Tuple[Tensor, Tensor, Tensor]):  # TODO
        """Function to take post-processed output of model and return ImageBBox."""

        bboxes, labels, scores = t
        if not len((labels).nonzero()) == 0:
            i = (labels).nonzero().min()
            bboxes, labels, scores = bboxes[i:], labels[i:], scores[i:]

        return bboxes, labels, scores

    @torch.jit.script
    def _process_bboxes_jit(
        output: List[Tensor], crit_vals: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        pred_bboxes = []
        pred_labels = []
        pred_scores = []

        batch = 0
        for chip_idx, (clas) in enumerate(output[0]):
            bboxes = output[1]
            bbox = bboxes[chip_idx].clone().detach()

            pp_output = _analyze_pred_jit(
                pred=(clas, bbox), thresh=0.1, nms_overlap=0.1, crit_vals=crit_vals
            )

            if pp_output is None:
                continue

            t = list(pp_output)
            if len(t[0]) == 0:
                continue

            output_final = _reconstruct_jit(pp_output)

            if not output_final[0] is None:
                pred_bboxes.append(_post_process(output_final[0]))
                pred_labels.append(output_final[1])
                pred_scores.append(output_final[2])
                batch += 1

        if not len(pred_bboxes) == 0:
            pred_bboxes_final = torch.stack([bbox for bbox in pred_bboxes])
            pred_labels_final = torch.stack([label for label in pred_labels])
            pred_scores_final = torch.stack([score for score in pred_scores])
            return pred_bboxes_final, pred_labels_final, pred_scores_final
        else:
            dummy = torch.empty((batch, 0, 0, 0)).float()
            return dummy, dummy, dummy

except Exception as e:
    import traceback

    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
