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

# pisa_faster_rcnn_x101_32x4d_fpn_1x_coco.py, box AP=41.9
_base_ = "./_base_/models/faster_rcnn_r50_fpn.py"

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
        norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
    ),
    roi_head=dict(
        type="PISARoIHead",
        bbox_head=dict(loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0)),
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            sampler=dict(
                type="ScoreHLRSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                k=0.5,
                bias=0.0,
            ),
            isr=dict(k=2, bias=0),
            carl=dict(k=1, bias=0.2),
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        )
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco-e4accec4.pth"
