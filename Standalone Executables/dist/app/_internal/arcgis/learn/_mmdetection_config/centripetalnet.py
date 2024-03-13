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

# training work but not infrencing due to transforms

model = dict(
    type="CornerNet",
    backbone=dict(
        type="HourglassNet",
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type="BN", requires_grad=True),
    ),
    neck=None,
    bbox_head=dict(
        type="CentripetalHead",
        num_classes=80,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=0,
        loss_heatmap=dict(
            type="GaussianFocalLoss", alpha=2.0, gamma=4.0, loss_weight=1
        ),
        loss_offset=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1),
        loss_guiding_shift=dict(type="SmoothL1Loss", beta=1.0, loss_weight=0.05),
        loss_centripetal_shift=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1),
    ),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        corner_topk=100,
        local_maximum_kernel=3,
        distance_threshold=0.5,
        score_thr=0.05,
        max_per_img=100,
        nms_cfg=dict(type="soft_nms", iou_threshold=0.5, method="gaussian"),
    ),
)

checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth"
