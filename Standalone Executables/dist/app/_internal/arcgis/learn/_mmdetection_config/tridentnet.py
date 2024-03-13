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

# tridentnet_r50_caffe_mstrain_3x_coco.py, box AP=40.3

_base_ = "./_base_/models/faster_rcnn_r50_caffe_c4.py"

model = dict(
    type="TridentFasterRCNN",
    pretrained="open-mmlab://detectron2/resnet50_caffe",
    backbone=dict(
        type="TridentResNet",
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1,
    ),
    roi_head=dict(type="TridentRoIHead", num_branch=3, test_branch_idx=1),
    train_cfg=dict(
        rpn_proposal=dict(nms_post=500, max_num=500),
        rcnn=dict(sampler=dict(num=128, pos_fraction=0.5, add_gt_as_proposals=False)),
    ),
)

checkpoint = "https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539-46d227ba.pth"
