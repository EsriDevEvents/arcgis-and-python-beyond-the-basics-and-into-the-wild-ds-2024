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

# libra_faster_rcnn_x101_64x4d_fpn_1x_coco.py, box AP=42.7
_base_ = "./_base_/models/faster_rcnn_r50_fpn.py"
# model settings
model = dict(
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
    neck=[
        dict(
            type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5
        ),
        dict(
            type="BFP",
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type="non_local",
        ),
    ],
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(
                _delete_=True,
                type="BalancedL1Loss",
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0,
            )
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(sampler=dict(neg_pos_ub=5), allowed_border=-1),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type="CombinedSampler",
                num=512,
                pos_fraction=0.25,
                add_gt_as_proposals=True,
                pos_sampler=dict(type="InstanceBalancedPosSampler"),
                neg_sampler=dict(
                    type="IoUBalancedNegSampler",
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3,
                ),
            )
        ),
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth"
