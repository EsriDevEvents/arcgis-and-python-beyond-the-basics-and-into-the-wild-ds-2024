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

# reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py, box AP = 44.2

norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)
model = dict(
    type="RepPointsDetector",
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
        dcn=dict(type="DCN", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        norm_cfg=norm_cfg,
        num_outs=5,
    ),
    bbox_head=dict(
        type="RepPointsHead",
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox_init=dict(type="SmoothL1Loss", beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type="SmoothL1Loss", beta=0.11, loss_weight=1.0),
        transform_method="moment",
    ),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type="PointAssigner", scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        refine=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=100,
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-f87da1ea.pth"
