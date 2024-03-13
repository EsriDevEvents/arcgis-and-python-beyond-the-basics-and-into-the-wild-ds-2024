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

# faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py, box AP=42.2

_base_ = "./_base_/models/faster_rcnn_r50_fpn.py"
model = dict(
    pretrained="open-mmlab://regnetx_3.2gf",
    backbone=dict(
        _delete_=True,
        type="RegNet",
        arch="regnetx_3.2gf",
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
    ),
    neck=dict(
        type="FPN", in_channels=[96, 192, 432, 1008], out_channels=256, num_outs=5
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200520_224253-bf85ae3e.pth"
