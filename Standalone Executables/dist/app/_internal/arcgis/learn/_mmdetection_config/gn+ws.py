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

# faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco.py , box AP=42.1

_base_ = "./_base_/models/faster_rcnn_r50_fpn.py"
conv_cfg = dict(type="ConvWS")
norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)
model = dict(
    pretrained="open-mmlab://jhu/resnext101_32x4d_gn_ws",
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style="pytorch",
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
    ),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type="Shared4Conv1FCBBoxHead",
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco_20200212-27da1bc2.pth"
