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

# box AP=42.1
_base_ = "./_base_/models/faster_rcnn_r50_fpn.py"
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type="GeneralizedAttention",
                    spatial_range=-1,
                    num_heads=8,
                    attention_type="1111",
                    kv_stride=2,
                ),
                stages=(False, False, True, True),
                position="after_conv2",
            )
        ],
        dcn=dict(type="DCN", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    )
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth"
