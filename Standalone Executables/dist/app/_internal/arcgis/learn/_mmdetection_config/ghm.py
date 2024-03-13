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

# retinanet_ghm_x101_64x4d_fpn_1x_coco.py, box AP=41.4
_base_ = "./_base_/models/retinanet_r50_fpn.py"
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
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type="GHMC",
            bins=30,
            momentum=0.75,
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            _delete_=True, type="GHMR", mu=0.02, bins=10, momentum=0.7, loss_weight=10.0
        ),
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131-dd381cef.pth"
