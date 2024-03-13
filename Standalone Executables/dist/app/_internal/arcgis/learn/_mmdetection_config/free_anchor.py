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

# retinanet_free_anchor_x101_32x4d_fpn_1x_coco.py, box AP=41.9

_base_ = "./_base_/models/retinanet_r50_fpn.py"
model = dict(
    pretrained="open-mmlab://resnext101_32x4d",
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style="pytorch",
    ),
    bbox_head=dict(
        _delete_=True,
        type="FreeAnchorRetinaHead",
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_bbox=dict(type="SmoothL1Loss", beta=0.11, loss_weight=0.75),
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth"
