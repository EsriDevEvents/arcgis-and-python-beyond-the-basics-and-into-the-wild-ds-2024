# Copyright 2020 The MMSegmentation Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# model settings
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="CGNet",
        norm_cfg=norm_cfg,
        in_channels=3,
        num_channels=(32, 64, 128),
        num_blocks=(3, 21),
        dilations=(2, 4),
        reductions=(8, 16),
    ),
    decode_head=dict(
        type="FCNHead",
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode="whole"),
)

checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/cgnet/cgnet_512x1024_60k_cityscapes/cgnet_512x1024_60k_cityscapes_20201101_110254-124ea03b.pth"
