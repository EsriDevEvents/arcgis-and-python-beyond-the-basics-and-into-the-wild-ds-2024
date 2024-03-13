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

# error in training loop
# retinanet_r50_nasfpn_crop640_50e_coco.py, box AP= 40.5

_base_ = "./_base_/models/retinanet_r50_fpn.py"
cudnn_benchmark = True
# model settings
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="RetinaNet",
    pretrained="torchvision://resnet50",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
    ),
    neck=dict(type="NASFPN", stack_times=7, norm_cfg=norm_cfg),
    bbox_head=dict(type="RetinaSepBNHead", num_ins=5, norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth"
