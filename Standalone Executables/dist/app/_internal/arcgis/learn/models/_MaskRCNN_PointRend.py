import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._PointRend import (
    generate_regular_grid_point_coords,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
    roi_mask_point_loss,
    c2_msra_fill,
    c2_xavier_fill,
    StandardPointHead,
)

from torchvision.models.detection.roi_heads import (
    RoIHeads,
    fastrcnn_loss,
    maskrcnn_loss,
    maskrcnn_inference,
)

import torch.nn.functional as F
from torch import nn
from torch.jit.annotations import List, Dict


def create_pointrend(model, num_class):
    # get model parameter to create modified RoI head
    y = {}
    y["box_roi_pool"] = model.roi_heads.box_roi_pool
    y["box_head"] = model.roi_heads.box_head
    y["box_predictor"] = model.roi_heads.box_predictor
    y["fg_iou_thresh"] = model.roi_heads.proposal_matcher.high_threshold
    y["bg_iou_thresh"] = model.roi_heads.proposal_matcher.low_threshold
    y["batch_size_per_image"] = model.roi_heads.fg_bg_sampler.batch_size_per_image
    y["positive_fraction"] = model.roi_heads.fg_bg_sampler.positive_fraction
    y["bbox_reg_weights"] = model.roi_heads.box_coder.weights
    y["score_thresh"] = model.roi_heads.score_thresh
    y["nms_thresh"] = model.roi_heads.nms_thresh
    y["detections_per_img"] = model.roi_heads.detections_per_img

    # change mask head to pointrend mask head

    y["mask_roi_pool"] = MaskRoIPoolHead(num_class)
    y["mask_head"] = CoarseMaskHead(num_class, model.backbone.out_channels)
    y["mask_predictor"] = PointRendHeads(
        num_class, in_channel=model.backbone.out_channels
    )

    model.roi_heads = PointRendROIHeads(**y)

    return model


class PointRendROIHeads(RoIHeads):
    def forward(self, features, proposals, image_shapes, targets=None):
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
                assert (
                    t["labels"].dtype == torch.int64
                ), "target labels must of int64 type"

        if self.training:
            if train_val:
                original_prpsl = [p.clone() for p in proposals]
            (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
            ) = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        if not self.training or train_val:
            if train_val:
                box_features = self.box_roi_pool(features, original_prpsl, image_shapes)
                box_features = self.box_head(box_features)
                class_logits, box_regression = self.box_predictor(box_features)
                boxes, scores, labels = self.postprocess_detections(
                    class_logits, box_regression, original_prpsl, image_shapes
                )
            else:
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

        if self.has_mask():
            mask_pred_gt = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
            if self.training:
                assert matched_idxs is not None
                assert targets is not None
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
                    mask_pred_gt.append(
                        {
                            "proposal_boxes": proposals[img_id][pos],
                            "gt_classes": gt_labels[img_id][pos_matched_idxs[img_id]],
                            "gt_masks": gt_masks[img_id][pos_matched_idxs[img_id]],
                        }
                    )
            else:
                pos_matched_idxs = None
                for res in result:
                    mask_pred_gt.append(
                        {
                            "pred_boxes": res["boxes"],
                            "pred_classes": res["labels"],
                        }
                    )

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_pred_gt)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(features, mask_features, mask_pred_gt)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")
            loss_mask = {}
            loss_mask_point = {}
            if self.training:
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                rcnn_loss_mask = maskrcnn_loss(
                    mask_features, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                loss_mask = {"loss_mask": rcnn_loss_mask}
                loss_mask_point = {"loss_mask_point": mask_logits}
            if not self.training or train_val:
                if train_val:
                    mask_pred_gt = []
                    for res in result:
                        mask_pred_gt.append(
                            {
                                "pred_boxes": res["boxes"],
                                "pred_classes": res["labels"],
                            }
                        )
                    mask_features = self.mask_roi_pool.eval()(features, mask_pred_gt)
                    mask_features = self.mask_head.eval()(mask_features)
                    mask_logits = self.mask_predictor.eval()(
                        features, mask_features, mask_pred_gt
                    )
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)
            losses.update(loss_mask_point)

        return result, losses


def calculate_uncertainty(logits, classes):
    # This code is based on https://github.com/facebookresearch/detectron2/blob/master/projects/PointRend
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.

    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.

    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -(torch.abs(gt_class_logits))


class MaskRoIPoolHead(nn.Module):
    # This code is based on https://github.com/facebookresearch/detectron2/blob/master/projects/PointRend
    def __init__(self, num_class):
        super().__init__()
        self.mask_coarse_in_features = ["0"]
        self.mask_coarse_side_size = 14
        self._feature_scales = {
            "0": 1.0 / 4,
            "1": 1.0 / 8,
            "2": 1.0 / 16,
            "3": 1.0 / 32,
        }  # FPN block 1/stride

    def forward(self, features, proposals):
        if self.training:
            boxes = [x["proposal_boxes"] for x in proposals]
        else:
            boxes = [x["pred_boxes"] for x in proposals]

        point_coords = generate_regular_grid_point_coords(
            sum(len(x) for x in boxes), self.mask_coarse_side_size, boxes[0].device
        )
        mask_coarse_features_list = [features[k] for k in self.mask_coarse_in_features]
        features_scales = [
            self._feature_scales[k] for k in self.mask_coarse_in_features
        ]
        # For regular grids of points, this function is equivalent to `len(features_list)' calls
        # of `ROIAlign` (with `SAMPLING_RATIO=2`), and concat the results.
        mask_features, _ = point_sample_fine_grained_features(
            mask_coarse_features_list, features_scales, boxes, point_coords
        )

        return mask_features


class CoarseMaskHead(nn.Module):
    # This code is based on https://github.com/facebookresearch/detectron2/blob/master/projects/PointRend
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(self, num_class, input_feture_channel):
        """
        The following attributes are parsed from config:
            conv_dim: the output dimension of the conv layers
            fc_dim: the feature dimenstion of the FC layers
            num_fc: the number of FC layers
            output_side_resolution: side resolution of the output square mask prediction
        """
        super(CoarseMaskHead, self).__init__()

        # fmt: off
        self.num_classes            = num_class
        conv_dim                    = 256
        self.fc_dim                 = 1024
        num_fc                      = 2
        self.output_side_resolution = 7
        self.input_channels         = input_feture_channel
        self.input_h                = 14
        self.input_w                = 14
        # fmt: on

        self.conv_layers = []
        if self.input_channels > conv_dim:
            self.reduce_channel_dim_conv = nn.Conv2d(
                self.input_channels,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = nn.Conv2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.conv_layers.append(self.reduce_spatial_dim_conv)

        input_dim = conv_dim * self.input_h * self.input_w
        input_dim //= 4

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(input_dim, self.fc_dim)
            self.add_module("coarse_mask_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = self.fc_dim

        output_dim = (
            self.num_classes * self.output_side_resolution * self.output_side_resolution
        )

        self.prediction = nn.Linear(self.fc_dim, output_dim)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            c2_msra_fill(layer)
        for layer in self.fcs:
            c2_xavier_fill(layer)

    def forward(self, x):
        # unlike BaseMaskRCNNHead, this head only outputs intermediate
        # features, because the features will be used later by PointHead.
        N = x.shape[0]
        x = x.view(N, self.input_channels, self.input_h, self.input_w)
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        return self.prediction(x).view(
            N,
            self.num_classes,
            self.output_side_resolution,
            self.output_side_resolution,
        )


class PointRendHeads(torch.nn.Module):
    # This code is based on https://github.com/facebookresearch/detectron2/blob/master/projects/PointRend
    def __init__(self, num_class, **kwargs):
        super().__init__()
        self._feature_scales = {
            "0": 1.0 / 4,
            "1": 1.0 / 8,
            "2": 1.0 / 16,
            "3": 1.0 / 32,
        }  # FPN block 1/stride
        self.mask_point_in_features = ["0"]
        self.mask_point_train_num_points = 14 * 14
        self.mask_point_oversample_ratio = 3
        self.mask_point_importance_sample_ratio = 0.75
        # next two parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_steps = 5
        self.mask_point_subdivision_num_points = 28 * 28
        in_channel = int(kwargs.get("in_channel", 256))

        in_channels = np.sum([in_channel for f in self.mask_point_in_features])
        self.mask_point_head = StandardPointHead(
            num_class, in_channels, coarse_pred_each_layer=True, fc_dim=in_channel
        )

    def forward(self, features, mask_coarse_logits, instances):
        """
        Forward logic of the mask point head..

        Args:
            features (dict[str, Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """

        mask_features_list = [features[k] for k in self.mask_point_in_features]
        features_scales = [self._feature_scales[k] for k in self.mask_point_in_features]

        if self.training:
            proposal_boxes = [x["proposal_boxes"] for x in instances]
            gt_classes = torch.cat([x["gt_classes"] for x in instances])
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    mask_coarse_logits,
                    lambda logits: calculate_uncertainty(logits, gt_classes),
                    self.mask_point_train_num_points,
                    self.mask_point_oversample_ratio,
                    self.mask_point_importance_sample_ratio,
                )

            (
                fine_grained_features,
                point_coords_wrt_image,
            ) = point_sample_fine_grained_features(
                mask_features_list, features_scales, proposal_boxes, point_coords
            )
            coarse_features = point_sample(
                mask_coarse_logits, point_coords, align_corners=False
            )
            point_logits = self.mask_point_head(fine_grained_features, coarse_features)
            loss_mask_point = roi_mask_point_loss(
                point_logits, instances, point_coords_wrt_image
            )
            return loss_mask_point
        else:
            pred_boxes = [x["pred_boxes"] for x in instances]
            pred_classes = torch.cat([x["pred_classes"] for x in instances])
            # The subdivision code will fail with the empty list of boxes
            if len(pred_classes) == 0:
                return mask_coarse_logits

            mask_logits = mask_coarse_logits.clone()
            for subdivions_step in range(self.mask_point_subdivision_steps):
                mask_logits = F.interpolate(
                    mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                )

                # If `mask_point_subdivision_num_points` is larger or equal to the
                # resolution of the next step, then we can skip this step
                H, W = mask_logits.shape[-2:]
                if (
                    self.mask_point_subdivision_num_points >= 4 * H * W
                    and subdivions_step < self.mask_point_subdivision_steps - 1
                ):
                    continue

                uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.mask_point_subdivision_num_points
                )
                fine_grained_features, _ = point_sample_fine_grained_features(
                    mask_features_list, features_scales, pred_boxes, point_coords
                )
                coarse_features = point_sample(
                    mask_coarse_logits, point_coords, align_corners=False
                )
                point_logits = self.mask_point_head(
                    fine_grained_features, coarse_features
                )

                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(R, C, H, W)
                )

            return mask_logits
