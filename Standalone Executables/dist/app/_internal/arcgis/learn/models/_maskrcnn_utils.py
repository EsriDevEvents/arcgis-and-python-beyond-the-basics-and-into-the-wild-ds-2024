from fastai.vision import Image
from fastai.vision.data import SegmentationProcessor, ImageList
from fastai.layers import CrossEntropyFlat
from fastai.basic_train import LearnerCallback
from fastai.torch_core import add_metrics
from torch.jit.annotations import List, Dict
from torchvision.models.detection.roi_heads import (
    fastrcnn_loss,
    maskrcnn_loss,
    maskrcnn_inference,
)
from torchvision.models.detection.transform import (
    resize_boxes,
    paste_masks_in_image,
)
import torch
from torch.nn.parallel import DistributedDataParallel
import warnings
import numpy as np
import matplotlib.pyplot as plt
from .._utils.common import ArcGISMSImage
from typing import Callable
import warnings
from .._utils.utils import check_imbalance
from fastprogress.fastprogress import progress_bar
from torchvision.ops import boxes as box_ops
from fastai.vision.transform import dihedral_affine
from fastai.vision import ImageBBox


def forward_roi(self, features, proposals, image_shapes, targets=None):
    """
    Arguments:
        features (List[Tensor])
        proposals (List[Tensor[N, 4]])
        image_shapes (List[Tuple[H, W]])
        targets (List[Dict])
    """

    train_val = getattr(self, "train_val", False)

    if targets is not None:
        for t in targets:
            floating_point_types = (torch.float, torch.double, torch.half)
            assert (
                t["boxes"].dtype in floating_point_types
            ), "target boxes must of float type"
            assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

    losses = {}
    result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])

    if self.training:
        if train_val:
            original_prpsl = [p.clone() for p in proposals]
        (
            proposals,
            matched_idxs,
            labels,
            regression_targets,
        ) = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets
        )

        # during training, only focus on positive boxes
        num_images = len(proposals)
        mask_proposals = []
        pos_matched_idxs = []
        for img_id in range(num_images):
            pos = torch.where(labels[img_id] > 0)[0]
            mask_proposals.append(proposals[img_id][pos])
            pos_matched_idxs.append(matched_idxs[img_id][pos])

        mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)

        gt_masks = [t["masks"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        rcnn_loss_mask = maskrcnn_loss(
            mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
        )

        losses = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
            "loss_mask": rcnn_loss_mask,
        }

    if not self.training or train_val:
        if train_val:
            proposals = original_prpsl

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        boxes, scores, labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        mask_proposals = [p["boxes"] for p in result]
        mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)
        labels = [r["labels"] for r in result]
        masks_probs = maskrcnn_inference(mask_logits, labels)
        for mask_prob, r in zip(masks_probs, result):
            r["masks"] = mask_prob

    return result, losses


def postprocess_transform(self, result, image_shapes, original_image_sizes):
    train_val = getattr(self, "train_val", False)

    if not self.training or train_val:
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(result, image_shapes, original_image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks

    elif self.training:
        return result

    return result


def post_nms_top_n(self):
    train_val = getattr(self, "train_val", False)

    if train_val:
        return self._post_nms_top_n["testing"]
    elif self.training:
        return self._post_nms_top_n["training"]
    return self._post_nms_top_n["testing"]


def pre_nms_top_n(self):
    train_val = getattr(self, "train_val", False)

    if train_val:
        self._pre_nms_top_n["testing"]
    elif self.training:
        return self._pre_nms_top_n["training"]
    return self._pre_nms_top_n["testing"]


def eager_outputs_modified(self, losses, detections):
    train_val = getattr(self, "train_val", False)

    if train_val:
        return detections, losses
    elif self.training:
        return losses
    return detections


class ArcGISImageSegment(Image):
    "Support applying transforms to segmentation masks data in `px`."

    def __init__(self, x, cmap=None, norm=None):
        super(ArcGISImageSegment, self).__init__(x)
        self.cmap = cmap
        self.mplnorm = norm
        self.type = np.unique(x)

    def lighting(self, func, *args, **kwargs):
        return self

    def refresh(self):
        self.sample_kwargs["mode"] = "nearest"
        return super().refresh()

    @property
    def data(self):
        "Return this image pixels as a `LongTensor`."
        return self.px.long()

    def show(
        self,
        ax=None,
        figsize=(3, 3),
        title=None,
        hide_axis=True,
        cmap="tab20",
        alpha=0.5,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        masks = self.data[0].numpy()
        for i in range(1, self.data.shape[0]):
            max_unique = np.max(np.unique(masks))
            mask = np.where(self.data[i] > 0, self.data[i] + max_unique, self.data[i])
            masks += mask
        ax.imshow(masks, cmap=cmap, alpha=alpha, **kwargs)
        if hide_axis:
            ax.axis("off")
        if title:
            ax.set_title(title)


def is_no_color(color_mapping):
    if isinstance(color_mapping, dict):
        color_mapping = list(color_mapping.values())
    return (np.array(color_mapping) == [-1.0, -1.0, -1.0]).any()


class ArcGISSegmentationLabelList(ImageList):
    "`ItemList` for segmentation masks."
    _processor = SegmentationProcessor

    def __init__(
        self,
        items,
        chip_size,
        classes=None,
        class_mapping=None,
        color_mapping=None,
        index_dir=None,
        **kwargs,
    ):
        super().__init__(items, **kwargs)
        self.class_mapping = class_mapping
        self.color_mapping = color_mapping
        self.copy_new.append("classes")
        self.classes, self.loss_func = classes, CrossEntropyFlat(axis=1)
        self.chip_size = chip_size
        self.inverse_class_mapping = {}
        self.index_dir = index_dir
        for k, v in self.class_mapping.items():
            self.inverse_class_mapping[v] = k
        if is_no_color(list(color_mapping.values())):
            self.cmap = "tab20"  ## compute cmap from palette
            import matplotlib as mpl

            bounds = list(color_mapping.keys())
            if (
                len(bounds) < 3
            ):  # Two handle two classes i am adding one number to the classes which is not already in bounds
                bounds = bounds + [max(bounds) + 1]
            self.mplnorm = mpl.colors.BoundaryNorm(bounds, len(bounds))
        else:
            import matplotlib as mpl

            bounds = list(color_mapping.keys())
            if (
                len(bounds) < 3
            ):  # Two handle two classes i am adding one number to the classes which is not already in bounds
                bounds = bounds + [max(bounds) + 1]
            self.cmap = mpl.colors.ListedColormap(
                np.array(list(color_mapping.values())) / 255
            )
            self.mplnorm = mpl.colors.BoundaryNorm(bounds, self.cmap.N)

        if len(color_mapping.keys()) == 1:
            self.cmap = "tab20"
            self.mplnorm = None

    def open(self, fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
            if len(fn) != 0:
                img_shape = ArcGISMSImage.read_image(fn[0]).shape
            else:
                labeled_mask = torch.zeros(
                    (len(self.class_mapping), self.chip_size, self.chip_size)
                )
                return ArcGISImageSegment(
                    labeled_mask, cmap=self.cmap, norm=self.mplnorm
                )
            k = 0

            labeled_mask = np.zeros((1, img_shape[0], img_shape[1]))

            for j in range(len(self.class_mapping)):
                if k < len(fn):
                    lbl_name = int(
                        self.index_dir[self.inverse_class_mapping[fn[k].parent.name]]
                    )
                else:
                    lbl_name = len(self.class_mapping) + 2
                if lbl_name == j + 1:
                    img = ArcGISMSImage.read_image(fn[k])
                    k = k + 1
                    if len(img.shape) == 3:
                        img = img.transpose(2, 0, 1)
                        img_mask = img[0]
                        for i in range(1, img.shape[0]):
                            max_unique = np.max(np.unique(img_mask))
                            img_i = np.where(img[i] > 0, img[i] + max_unique, img[i])
                            img_mask += img_i
                        img_mask = np.expand_dims(img_mask, axis=0)
                    else:
                        img_mask = np.expand_dims(img, axis=0)
                else:
                    img_mask = np.zeros((1, img_shape[0], img_shape[1]))
                labeled_mask = np.append(labeled_mask, img_mask, axis=0)
            labeled_mask = labeled_mask[1:, :, :]
            labeled_mask = torch.Tensor(list(labeled_mask))
        return ArcGISImageSegment(labeled_mask, cmap=self.cmap, norm=self.mplnorm)

    def reconstruct(self, t):
        return ArcGISImageSegment(t, cmap=self.cmap, norm=self.mplnorm)


class ArcGISInstanceSegmentationItemList(ImageList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls, _square_show_res = ArcGISSegmentationLabelList, False
    _div = None
    _imagery_type = None

    def open(self, fn):
        return ArcGISMSImage.open(fn, div=self._div, imagery_type=self._imagery_type)

    def check_class_imbalance(
        self, func: Callable, stratify=False, class_imbalance_pct=0.01
    ):
        try:
            labelval = [(func(o)) for o in self.items]
            total_sample = np.concatenate(labelval)
            unique_sample = set(total_sample)

            check_imbalance(total_sample, unique_sample, class_imbalance_pct, stratify)
        except Exception as e:
            warnings.warn(f"Unable to check for class imbalance [reason : {e}]")

        return self

    def label_list_from_func(self, func: Callable):
        "Apply `func` to every input to get its label."
        import pandas as pd

        self._list_of_labels = ["_".join(func(o)) for o in self.items]
        self._idx_label_tuple_list = [
            (i, label) for i, label in enumerate(self._list_of_labels)
        ]
        label_series = pd.Series(self._list_of_labels)
        single_instance_labels = list(
            label_series.value_counts()[label_series.value_counts() == 1].index
        )
        self._label_idx_mapping = {
            label: i for i, label in enumerate(self._list_of_labels)
        }
        for (
            label
        ) in single_instance_labels:  # adding duplicate instance of unique labels
            self._idx_label_tuple_list.append((self._label_idx_mapping[label], label))
        return self

    def stratified_split_by_pct(self, valid_pct: float = 0.2, seed: int = None):
        try:
            "Split the items in a stratified manner by putting `valid_pct` in the validation set, optional `seed` can be passed."
            from sklearn.model_selection import train_test_split
            import random, math

            if valid_pct == 0.0:
                return self.split_none()
            if seed is not None:
                np.random.seed(seed)
            if (
                len(set(self._list_of_labels)) > len(self._list_of_labels) * valid_pct
            ):  # if validation samples length is less than unique labels
                classes = len(set(self._list_of_labels))
                xlen = len(self._list_of_labels)
                sample_shortage = math.ceil((classes - xlen * valid_pct) / valid_pct)
                extra_samples = random.choices(
                    self._idx_label_tuple_list, k=sample_shortage
                )
                self._idx_label_tuple_list.extend(extra_samples)
            X, y = [], []
            for index, label in self._idx_label_tuple_list:
                X.append(index)
                y.append(label)
            train_idx, val_idx, _, _ = train_test_split(
                X, y, test_size=valid_pct, random_state=seed, stratify=y
            )
            return self.split_by_idxs(train_idx, val_idx)
        except Exception as e:
            warnings.warn(
                f"Unable to perform stratified splitting [reason : {e}], falling back to random split"
            )
            return self.split_by_rand_pct(valid_pct=valid_pct, seed=seed)


class ArcGISInstanceSegmentationMSItemList(ArcGISInstanceSegmentationItemList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls, _square_show_res = ArcGISSegmentationLabelList, False

    def open(self, fn):
        return ArcGISMSImage.open_gdal(fn)


def mask_rcnn_loss(loss_value, *args):
    if isinstance(loss_value, tuple):
        loss_value = loss_value[1]

    final_loss = 0.0
    for i in loss_value.values():
        i[torch.isnan(i)] = 0.0
        i[torch.isinf(i)] = 0.0
        final_loss += i

    return final_loss


def mask_to_dict(last_target, device):
    target_list = []

    for i in range(len(last_target)):
        boxes = []
        masks = np.zeros((1, last_target[i].shape[1], last_target[i].shape[2]))
        labels = []
        for j in range(last_target[i].shape[0]):
            mask = np.array(last_target[i].data[j].cpu())
            obj_ids = np.unique(mask)

            if len(obj_ids) == 1:
                continue

            obj_ids = obj_ids[1:]
            mask_j = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)

            for k in range(num_objs):
                pos = np.where(mask_j[k])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmax - xmin == 0:
                    xmax += 1
                if ymax - ymin == 0:
                    ymax += 1
                boxes.append([xmin, ymin, xmax, ymax])

            masks = np.append(masks, mask_j, axis=0)
            labels_j = torch.ones((num_objs,), dtype=torch.int64)
            labels_j = labels_j * (j + 1)
            labels.append(labels_j)

        if masks.shape[0] == 1:  # if no object in image
            masks[0, 50:51, 50:51] = 1
            labels = torch.tensor([0])
            boxes = torch.tensor([[50.0, 50.0, 51.0, 51.0]])
        else:
            labels = torch.cat(labels)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = masks[1:, :, :]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        target = {}
        target["boxes"] = boxes.to(device)
        target["labels"] = labels.to(device)
        target["masks"] = masks.to(device)
        target_list.append(target)

    return target_list


class train_callback(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Handle new batch `xb`,`yb` in `train` or validation."
        train = kwargs.get("train")
        if isinstance(self.model, DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model
        self.model.train()
        if train:
            model.roi_heads.train_val = False
            model.rpn.train_val = False
            model.train_val = False
            model.transform.train_val = False
        else:
            model.backbone.eval()  # to get feature in eval mode for evaluation
            model.roi_heads.train_val = True
            model.rpn.train_val = True
            model.train_val = True
            model.transform.train_val = True
        target_list = mask_to_dict(last_target, self.c_device)
        if last_input.shape[0] < 2:
            last_input = torch.cat((last_input, last_input))
            target_list.append(target_list[0])

        last_input = [list(last_input), target_list]
        last_target = target_list  # [torch.tensor([1]) for i in last_target]
        return {"last_input": last_input, "last_target": last_target}


class AveragePrecision(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)

    def on_epoch_begin(self, **kwargs):
        self.aps = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        last_output = last_output[0]
        for i in range(len(last_output)):
            last_output[i]["masks"] = last_output[i]["masks"].squeeze()
            if last_output[i]["masks"].shape[0] == 0:
                continue
            if len(last_output[i]["masks"].shape) == 2:
                last_output[i]["masks"] = last_output[i]["masks"][None]
            ap = compute_ap(
                last_target[i]["labels"],
                last_target[i]["masks"],
                last_output[i]["labels"],
                last_output[i]["scores"],
                last_output[i]["masks"],
            )
            self.aps.append(ap)

    def on_epoch_end(self, last_metrics, **kwargs):
        if isinstance(self.model, DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model
        model.roi_heads.train_val = False
        model.rpn.train_val = False
        model.train_val = False
        model.transform.train_val = False
        if self.aps == []:
            self.aps.append(0.0)
        self.aps = torch.mean(torch.tensor(self.aps))
        return add_metrics(last_metrics, self.aps)


def masks_iou(masks1, masks2):
    # Mask R-CNN

    # The MIT License (MIT)

    # Copyright (c) 2017 Matterport, Inc.

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE.

    # Method is based on https://github.com/matterport/Mask_RCNN

    if masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return torch.zeros((masks1.shape[0], masks2.shape[0]))
    masks1 = masks1.permute(1, 2, 0)
    masks2 = masks2.permute(1, 2, 0)
    masks1 = torch.reshape(masks1 > 0.5, (-1, masks1.shape[-1])).type(torch.float64)
    masks2 = torch.reshape(masks2 > 0.5, (-1, masks2.shape[-1])).type(torch.float64)
    area1 = torch.sum(masks1, dim=0)
    area2 = torch.sum(masks2, dim=0)

    intersections = torch.mm(masks1.transpose(1, 0), masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    return overlaps


def compute_matches(
    gt_class_ids,
    gt_masks,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_threshold=0.5,
    detect_threshold=0.5,
):
    # Method is based on https://github.com/matterport/Mask_RCNN
    indices = torch.argsort(pred_scores, descending=True)
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[indices]

    ious_mask = masks_iou(pred_masks, gt_masks)

    pred_match = -1 * np.ones([pred_masks.shape[0]])
    if 0 not in ious_mask.shape:
        max_iou, matches = ious_mask.max(1)
        detected = []
        for i in range(len(pred_class_ids)):
            if (
                max_iou[i] >= iou_threshold
                and pred_scores[i] >= detect_threshold
                and matches[i] not in detected
                and gt_class_ids[matches[i]] == pred_class_ids[i]
            ):
                detected.append(matches[i])
                pred_match[i] = pred_class_ids[i]

    return pred_match


def compute_ap(
    gt_class_ids,
    gt_masks,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_threshold=0.5,
    detect_threshold=0.5,
):
    # Method is based on https://github.com/matterport/Mask_RCNN
    pred_match = compute_matches(
        gt_class_ids,
        gt_masks,
        pred_class_ids,
        pred_scores,
        pred_masks,
        iou_threshold,
        detect_threshold,
    )

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_class_ids)

    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return mAP


def batch_dihedral(x, k):
    flips = []
    if k & 1:
        flips.append(2)
    if k & 2:
        flips.append(3)
    if flips:
        x = torch.flip(x, flips)
    if k & 4:
        x = x.transpose(2, 3)
    return x.contiguous()


def recover_boxes(bboxes, size, k):
    if bboxes.size(0):
        device = bboxes.device
        bboxes = ImageBBox.create(*size, bboxes.detach().cpu())
        bboxes = dihedral_affine(bboxes, k)
        if k == 5 or k == 6:
            bboxes = dihedral_affine(bboxes, 3)
        bboxes = (size[0] * (bboxes.data + 1) / 2).to(device)

    return bboxes


def intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def box_area(b):
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def boxious(box_a, box_b):
    inter = intersect(box_a, box_b)
    union = box_area(box_a).unsqueeze(1) + box_area(box_b).unsqueeze(0) - inter
    return inter / union


def pred_mean_merge(pred, iou_thresold=0.5, same_pred=1):
    bboxes = pred["boxes"]
    masks = pred["masks"].squeeze()
    scores, labels = pred["scores"], pred["labels"]
    ious = boxious(bboxes, bboxes)
    numOfboxes = ious.shape[0]
    # merge predictions with iou>thresold
    ref_boxs, merge_boxids = np.where(ious > iou_thresold)
    _, idx = np.unique(ref_boxs, return_index=True)
    counter = np.array(range(numOfboxes))
    i = 0
    merge_masks, merge_boxes, merge_scores, merge_labels = [], [], [], []
    while i < numOfboxes:
        if counter[i] == -1:
            i += 1
            continue
        if i == numOfboxes - 1:
            matched_idx = merge_boxids[idx[i] :]
        else:
            matched_idx = merge_boxids[idx[i] : idx[i + 1]]
        if len(matched_idx) > same_pred:
            merge_masks.append(masks[matched_idx, :, :].mean(dim=0))
            merge_boxes.append(bboxes[matched_idx, :].mean(dim=0))
            merge_scores.append(scores[matched_idx].mean())
            merge_labels.append(labels[matched_idx[0]])

        counter[np.isin(counter, matched_idx)] = -1
        i += 1
    return (
        torch.stack(merge_masks) if merge_masks else torch.tensor([]),
        torch.stack(merge_boxes) if merge_boxes else torch.tensor([]),
        torch.tensor(merge_scores),
        torch.tensor(merge_labels),
    )


def merge_tta_prediction(predictions, nms_thres=0.3, merge_policy="mean"):
    device = predictions[0]["boxes"].device
    result = {k: [] for k in predictions[0].keys()}
    for pred in predictions:
        for k, v in pred.items():
            result[k].append(v)

    for k, v in result.items():
        result[k] = torch.cat(v).detach().cpu()

    if merge_policy == "mean":
        (
            result["masks"],
            result["boxes"],
            result["scores"],
            result["labels"],
        ) = pred_mean_merge(result, nms_thres)
    else:
        keep = box_ops.batched_nms(
            result["boxes"],
            result["scores"],
            result["labels"],
            nms_thres,
        )

        for k, v in result.items():
            result[k] = result[k][keep]

    for k, v in result.items():
        result[k] = result[k].to(device)

    return result


def predict_tta(model, batch, detect_thresh=0.5, merge_policy="mean"):
    temp = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = detect_thresh
    ttaPreds = [[] for _ in range(batch.shape[0])]
    for k in model.arcgis_tta:
        transforms = batch_dihedral(batch, k)
        pred = model(list(transforms))
        for i, p in enumerate(pred):
            p["masks"] = batch_dihedral(p["masks"], k)
            if k == 5 or k == 6:
                p["masks"] = batch_dihedral(p["masks"], 3)
            p["boxes"] = recover_boxes(p["boxes"], batch.shape[-2:], k)
            ttaPreds[i].append(p)
    for i, pred in enumerate(ttaPreds):
        ttaPreds[i] = merge_tta_prediction(
            pred, model.roi_heads.nms_thresh, merge_policy
        )
    model.roi_heads.score_thresh = temp

    return ttaPreds


def compute_class_AP(
    model,
    dl,
    n_classes,
    show_progress,
    detect_thresh=0.5,
    iou_thresh=0.5,
    mean=False,
    tta_prediction=False,
):
    model.learn.model.eval()
    if mean:
        aps = []
    else:
        aps = [[] for _ in range(n_classes)]
    with torch.no_grad():
        for input, target in progress_bar(dl, display=show_progress):
            if tta_prediction:
                predictions = predict_tta(model.learn.model, input, detect_thresh)
            else:
                predictions = model.learn.model(list(input))
            ground_truth = mask_to_dict(target, model._device)
            for i in range(len(predictions)):
                predictions[i]["masks"] = predictions[i]["masks"].squeeze()
                if predictions[i]["masks"].shape[0] == 0:
                    continue
                if len(predictions[i]["masks"].shape) == 2:
                    predictions[i]["masks"] = predictions[i]["masks"][None]
                if mean:
                    ap = compute_ap(
                        ground_truth[i]["labels"],
                        ground_truth[i]["masks"],
                        predictions[i]["labels"],
                        predictions[i]["scores"],
                        predictions[i]["masks"],
                        iou_thresh,
                        detect_thresh,
                    )
                    aps.append(ap)
                else:
                    for k in range(1, n_classes + 1):
                        gt_labels_index = (
                            (ground_truth[i]["labels"] == k).nonzero().reshape(-1)
                        )
                        gt_labels = ground_truth[i]["labels"][gt_labels_index]
                        gt_masks = ground_truth[i]["masks"][gt_labels_index]
                        pred_labels_index = (
                            (predictions[i]["labels"] == k).nonzero().reshape(-1)
                        )
                        pred_labels = predictions[i]["labels"][pred_labels_index]
                        pred_masks = predictions[i]["masks"][pred_labels_index]
                        pred_scores = predictions[i]["scores"][pred_labels_index]
                        if len(gt_labels):
                            ap = compute_ap(
                                gt_labels,
                                gt_masks,
                                pred_labels,
                                pred_scores,
                                pred_masks,
                                iou_thresh,
                                detect_thresh,
                            )
                            aps[k - 1].append(ap)
    if mean:
        if aps != []:
            aps = np.mean(aps, axis=0)
        else:
            return 0.0
    else:
        for i in range(n_classes):
            if aps[i] != []:
                aps[i] = np.mean(aps[i])
            else:
                aps[i] = 0.0
    if model._device == torch.device("cuda"):
        torch.cuda.empty_cache()
    return aps
