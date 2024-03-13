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

# faster_rcnn_r50_pafpn_1x_coco.py, box AP=37.5
_base_ = "./_base_/models/faster_rcnn_r50_fpn.py"
model = dict(
    neck=dict(
        type="PAFPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5
    )
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth"
