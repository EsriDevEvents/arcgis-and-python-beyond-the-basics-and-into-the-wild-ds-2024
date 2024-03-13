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

# Currently only batch_size 1 for inference mode is supported. Found batch_size 2. error is comming so we don't support for now
# detr_r50_8x2_150e_coco.py, box AP=40.1
model = dict(
    type="DETR",
    pretrained="torchvision://resnet50",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
    ),
    bbox_head=dict(
        type="TransformerHead",
        num_classes=80,
        in_channels=2048,
        num_fcs=2,
        transformer=dict(
            type="Transformer",
            embed_dims=256,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            feedforward_channels=2048,
            dropout=0.1,
            act_cfg=dict(type="ReLU", inplace=True),
            norm_cfg=dict(type="LN"),
            num_fcs=2,
            pre_norm=False,
            return_intermediate_dec=True,
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            cls_cost=dict(type="ClassificationCost", weight=1.0),
            reg_cost=dict(type="BBoxL1Cost", weight=5.0),
            iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
        )
    ),
    test_cfg=dict(max_per_img=100),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth"
