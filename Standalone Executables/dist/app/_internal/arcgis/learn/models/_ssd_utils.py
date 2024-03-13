import torch
from torch import nn, LongTensor
import torch.nn.functional as F
from fastai.core import split_kwargs_by_func
from fastprogress.fastprogress import progress_bar
from fastai.basic_train import Callback
from fastai.torch_core import add_metrics
from .._utils.pointcloud_od import confusion_matrix3d

import numpy as np
import random
import math


def conv_params(in_size, out_size):
    filters = [3, 2, 5, 4]
    strides = [1, 2, 3]  # max_stride = 3
    pads = [0, 1, 2, 3]  # max pad

    if out_size == 1:
        return 1, 0, in_size

    for filter_size in filters:
        for pad in pads:
            for stride in strides:
                if (out_size - 1) * stride == (in_size - filter_size) + 2 * pad:
                    return stride, pad, filter_size
    return None, None, None


def conv_paramsv2(in_size, out_size):
    filters = [3]
    strides = [1, 2]  # max_stride = 2
    pads = [0, 1]  # max pad

    if out_size == 1:
        return 1, 0, in_size

    for filter_size in filters:
        for pad in pads:
            for stride in strides:
                if (((in_size - filter_size) + 2 * pad) // stride) + 1 == out_size:
                    return stride, pad, filter_size
    return None, None, None


class StdConv(nn.Module):
    def __init__(self, nin, nout, filter_size=3, stride=2, padding=1, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, filter_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.bn(F.relu(self.conv(x))))


class StdConvv2(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        upsample_size=0,
        upsample=False,
        filter_size=3,
        stride=2,
        padding=1,
        drop=0.1,
    ):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(nin, nout, filter_size, stride=stride, padding=padding)
        self.up = nn.Upsample(upsample_size)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.upsample == True:
            return self.drop(self.bn(F.relu(self.conv(self.up(x)))))
        return self.drop(self.bn(F.relu(self.conv(x))))


def flatten_conv(x, k):
    bs, nf, gx, gy = x.size()
    x = x.permute(0, 2, 3, 1).contiguous()
    return x.view(bs, -1, nf // k)


class OutConv(nn.Module):
    def __init__(self, k, nin, num_classes, bias):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(nin, (num_classes) * k, 3, padding=1)
        self.oconv2 = nn.Conv2d(nin, 4 * k, 3, padding=1)
        self.oconv1.bias.data.zero_().add_(bias)

    def forward(self, x):
        return [
            flatten_conv(self.oconv1(x), self.k),
            flatten_conv(self.oconv2(x), self.k),
        ]


class SSDHead(nn.Module):
    def __init__(
        self,
        grids,
        anchors_per_cell,
        num_classes,
        num_features=7,
        drop=0.3,
        bias=-4.0,
        num_channels=512,
    ):
        super().__init__()
        self.drop = nn.Dropout(drop)

        self.sconvs = nn.ModuleList([])
        self.oconvs = nn.ModuleList([])

        self.anc_grids = grids

        self._k = anchors_per_cell

        self.sconvs.append(StdConv(num_channels, 256, stride=1, drop=drop))

        for i in range(len(grids)):
            if i == 0:
                stride, pad, filter_size = conv_params(num_features, grids[i])
            else:
                stride, pad, filter_size = conv_params(grids[i - 1], grids[i])

            if stride is None:
                print(grids[i - 1], " --> ", grids[i])
                raise Exception("cannot create model for specified grids.")

            self.sconvs.append(
                StdConv(256, 256, filter_size, stride=stride, padding=pad, drop=drop)
            )
            self.oconvs.append(
                OutConv(self._k, 256, num_classes=num_classes, bias=bias)
            )

    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconvs[0](x)
        out_classes = []
        out_bboxes = []
        for sconv, oconv in zip(self.sconvs[1:], self.oconvs):
            x = sconv(x)
            out_class, out_bbox = oconv(x)
            out_classes.append(out_class)
            out_bboxes.append(out_bbox)

        return [torch.cat(out_classes, dim=1), torch.cat(out_bboxes, dim=1)]


class SSDHeadv2(nn.Module):
    def __init__(
        self,
        grids,
        anchors_per_cell,
        num_classes,
        num_features=7,
        drop=0.3,
        bias=-4.0,
        num_channels=512,
    ):
        super().__init__()
        self.drop = nn.Dropout(drop)

        self.sconvs = nn.ModuleList([])
        self.oconvs = nn.ModuleList([])

        self.anc_grids = grids

        self._k = anchors_per_cell

        self.sconvs.append(StdConvv2(num_channels, 256, stride=1, drop=drop))

        for i in range(len(grids)):
            upsample = False

            if i == 0 and num_features >= grids[i]:
                stride, pad, filter_size = conv_paramsv2(num_features, grids[i])
                if stride is None:
                    upsample = True
                    stride, pad, filter_size = 1, 1, 3
            elif i == 0 and num_features < grids[i]:
                upsample = True
                stride, pad, filter_size = 1, 1, 3

            elif i != 0 and grids[i - 1] > grids[i]:
                stride, pad, filter_size = conv_paramsv2(grids[i - 1], grids[i])
                if stride is None:
                    upsample = True
                    stride, pad, filter_size = 1, 1, 3
            else:
                upsample = True
                stride, pad, filter_size = 1, 1, 3

            self.sconvs.append(
                StdConvv2(
                    256,
                    256,
                    grids[i],
                    upsample,
                    filter_size,
                    stride=stride,
                    padding=pad,
                    drop=drop,
                )
            )
            self.oconvs.append(
                OutConv(self._k, 256, num_classes=num_classes, bias=bias)
            )

    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconvs[0](x)
        out_classes = []
        out_bboxes = []
        for sconv, oconv in zip(self.sconvs[1:], self.oconvs):
            x = sconv(x)
            out_class, out_bbox = oconv(x)
            out_classes.append(out_class)
            out_bboxes.append(out_bbox)

        return [torch.cat(out_classes, dim=1), torch.cat(out_bboxes, dim=1)]


def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]


class BCE_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, targ):
        t = one_hot_embedding(targ, self.num_classes)
        t = torch.Tensor(t[:, 1:].contiguous()).to(pred.device)
        x = pred[:, 1:]
        w = self.get_weight(x, t)
        return F.binary_cross_entropy_with_logits(x, t, w, reduction="sum") / (
            self.num_classes - 1
        )

    def get_weight(self, x, t):
        return None


class FocalLoss(BCE_Loss):
    def get_weight(self, x, t):
        alpha, gamma = 0.25, 1
        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        w = w * (1 - pt).pow(gamma)
        return w.detach()


def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def postprocess(
    pred, model=None, thresh=0.5, nms_overlap=0.1, ret_scores=True, device=None
):
    """
    It works on a single activation, does not support batch.
    """
    if device is None:
        device = torch.device("cpu")
    b_clas, b_bb = pred
    a_ic = model._actn_to_bb(
        b_bb.to(device), model._anchors.to(device), model._grid_sizes.to(device)
    )
    conf_scores, clas_ids = b_clas[:, 1:].max(1)
    conf_scores = b_clas.t().sigmoid().to(device)

    out1, bbox_list, class_list = [], [], []

    for cl in range(1, len(conf_scores)):
        c_mask = conf_scores[cl] > thresh
        if c_mask.sum() == 0:
            continue
        scores = conf_scores[cl][c_mask]
        l_mask = c_mask.unsqueeze(1)
        l_mask = l_mask.expand_as(a_ic)
        boxes = a_ic[l_mask].view(-1, 4)  # boxes are now in range[ 0, 1]
        boxes = (boxes - 0.5) * 2.0  # putting boxes in range[-1, 1]
        ids, count = nms(
            boxes.data, scores, nms_overlap, 50
        )  # FIX- NMS overlap hardcoded
        ids = ids[:count]
        out1.append(scores[ids])
        bbox_list.append(boxes.data[ids])
        class_list.append(torch.tensor([cl] * count))

    if len(bbox_list) == 0:
        return None

    if ret_scores:
        return (
            torch.cat(bbox_list, dim=0).to(device),
            torch.cat(class_list, dim=0).to(device),
            torch.cat(out1, dim=0).to(device),
        )
    else:
        return torch.cat(bbox_list, dim=0), torch.cat(class_list, dim=0)


class AveragePrecision(Callback):
    def __init__(self, model, n_classes, mode_3d=False):
        self.model = model
        self.n_classes = n_classes
        self.mode_3d = mode_3d

    def on_epoch_begin(self, **kwargs):
        self.tps, self.clas, self.p_scores = [], [], []
        self.classes, self.n_gts = (
            LongTensor(range(self.n_classes)),
            torch.zeros(self.n_classes).long(),
        )

    def on_batch_end(self, last_output, last_target, **kwargs):
        if (
            getattr(self.model, "_is_fasterrcnn", False)
            or "MMDetection" in self.model.__str__()
        ):
            last_output = last_output[0]

        if self.mode_3d:
            tps, p_scores, clas, self.n_gts = confusion_matrix3d(
                last_output, last_target, self.n_gts, self.classes
            )
        else:
            tps, p_scores, clas, self.n_gts = compute_cm(
                self.model, last_output, last_target, self.n_gts, self.classes
            )
        self.tps.extend(tps)
        self.p_scores.extend(p_scores)
        self.clas.extend(clas)

    def on_epoch_end(self, last_metrics, **kwargs):
        aps = compute_ap_score(
            self.tps, self.p_scores, self.clas, self.n_gts, self.n_classes, self.mode_3d
        )
        aps = torch.mean(torch.tensor(aps))
        return add_metrics(last_metrics, aps)


def compute_class_AP(
    model,
    dl,
    n_classes,
    show_progress,
    iou_thresh=0.5,
    detect_thresh=0.35,
    num_keep=100,
    **kwargs
):
    tps, clas, p_scores = [], [], []
    if getattr(model, "_is_model_extension", False):
        transform_kwargs, kwargs = split_kwargs_by_func(
            kwargs, model._model_conf.transform_input
        )
    classes, n_gts = LongTensor(range(n_classes)), torch.zeros(n_classes).long()
    with torch.no_grad():
        for input, target in progress_bar(dl, display=show_progress):
            if getattr(model, "_is_model_extension", False):
                try:
                    if model._is_multispectral:
                        output = model.learn.model.eval()(
                            model._model_conf.transform_input_multispectral(
                                input, **transform_kwargs
                            )
                        )
                    else:
                        output = model.learn.model.eval()(
                            model._model_conf.transform_input(input, **transform_kwargs)
                        )
                except Exception as e:
                    if getattr(model, "_is_fasterrcnn", False):
                        output = []
                        for _ in range(input.shape[0]):
                            res = {}
                            res["boxes"] = torch.empty(0, 4)
                            res["scores"] = torch.tensor([])
                            res["labels"] = torch.tensor([])
                            output.append(res)
                    else:
                        raise e
            else:
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
    model, output, target, n_gts, classes, iou_thresh=0.1, detect_thresh=0.2
):
    tps, clas, p_scores = [], [], []

    if getattr(model, "_is_model_extension", False):
        analyzed_pred_out = model._analyze_pred(
            output,
            thresh=detect_thresh,
            nms_overlap=iou_thresh,
            ret_scores=True,
            device=model._device,
        )

    for i in range(target[0].size(0)):
        if getattr(model, "_is_model_extension", False):
            op = analyzed_pred_out[i]
        else:
            op = model._data.y.analyze_pred(
                (output[0][i], output[1][i]),
                model=model,
                thresh=detect_thresh,
                nms_overlap=iou_thresh,
                ret_scores=True,
                device=model._device,
            )
        tgt_bbox, tgt_clas = model._get_y(target[0][i], target[1][i])

        try:
            bbox_pred, preds, scores = op
            if len(bbox_pred) != 0 and len(tgt_bbox) != 0:
                ious = model._jaccard(bbox_pred, tgt_bbox)
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
        except Exception as e:
            pass
        n_gts += ((tgt_clas.cpu()[:, None] - 1) == classes[None, :]).sum(0)

    return tps, p_scores, clas, n_gts


def compute_ap_score(tps, p_scores, clas, n_gts, n_classes, mode_3d=False):
    # If no true positives are found return an average precision score of 0.
    if len(tps) == 0:
        return [0.0 for _ in range(n_classes)]

    tps, p_scores, clas = torch.tensor(tps), torch.cat(p_scores, 0), torch.cat(clas, 0)
    fps = 1 - tps
    idx = p_scores.argsort(descending=True)
    tps, fps, clas = tps[idx], fps[idx], clas[idx]
    aps = []
    for n_gts_idx, cls in enumerate(range(1, n_classes + 1)):
        if mode_3d:
            cls -= 1
        tps_cls, fps_cls = (
            tps[clas == cls].float().cumsum(0),
            fps[clas == cls].float().cumsum(0),
        )
        if tps_cls.numel() != 0 and tps_cls[-1] != 0:
            precision = tps_cls / (tps_cls + fps_cls + 1e-8)
            recall = tps_cls / (n_gts[n_gts_idx] + 1e-8)
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


def iou(ann, centroids):
    similarities = []

    for centroid in centroids:
        inter = np.prod(np.minimum(ann, centroid))
        union = np.prod(ann) + np.prod(centroid) - inter
        similarities.append(inter / union)

    return np.array(similarities)


def avg_iou(bboxes, centroids):
    sum = 0.0

    for bbox in bboxes:
        sum += max(iou(bbox, centroids))

    return sum / bboxes.shape[0]


def kmeans(bboxes, num_anchor):
    # Method based on https://github.com/experiencor/keras-yolo3

    num_points, dim = bboxes.shape
    prev_centroids = np.ones(num_points) * (-1)
    indices = [random.randrange(num_points) for i in range(num_anchor)]
    centroids = bboxes[indices]

    while True:
        distances = []
        for bbox in bboxes:
            d = 1 - iou(bbox, centroids)
            distances.append(d)
        distances = np.array(distances)
        cur_centroids = np.argmin(distances, axis=1)

        if (prev_centroids == cur_centroids).all():
            return centroids

        centroid_sums = np.zeros(
            (num_points, dim), float
        )  # num_points needs to be num_anchors
        for i in range(num_points):
            centroid_sums[cur_centroids[i]] += bboxes[i]
        for i in range(num_anchor):
            centroids[i] = centroid_sums[i] / (np.sum(cur_centroids == i) + 1e-6)

        prev_centroids = cur_centroids.copy()
