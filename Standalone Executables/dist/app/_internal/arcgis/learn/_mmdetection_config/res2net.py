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

# cascade_rcnn_r2_101_fpn_20e_coco.py, box AP=45.7

_base_ = "./_base_/models/cascade_rcnn_r50_fpn.py"

model = dict(
    pretrained="open-mmlab://res2net101_v1d_26w_4s",
    backbone=dict(type="Res2Net", depth=101, scales=4, base_width=26),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_rcnn_r2_101_fpn_20e_coco/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth"
