# Copyright 2018-2019 Open-MMLab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# fsaf_x101_64x4d_fpn_1x_coco.py, box AP=42.4 (41.0)
_base_ = "./_base_/models/retinanet_r50_fpn.py"
# model settings
model = dict(
    type="FSAF",
    pretrained="open-mmlab://resnext101_64x4d",
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
    ),
    bbox_head=dict(
        type="FSAFHead",
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        # Only anchor-free branch is implemented. The anchor generator only
        #  generates 1 anchor at each feature point, as a substitute of the
        #  grid of features.
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(_delete_=True, type="TBLRBBoxCoder", normalizer=4.0),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction="none",
        ),
        loss_bbox=dict(
            _delete_=True, type="IoULoss", eps=1e-6, loss_weight=1.0, reduction="none"
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type="CenterRegionAssigner",
            pos_scale=0.2,
            neg_scale=0.2,
            min_pos_iof=0.01,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth"
