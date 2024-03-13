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


_base_ = "./deeplabv3plus.py"
model = dict(
    pretrained="mmcls://mobilenet_v2",
    backbone=dict(
        _delete_=True,
        type="MobileNetV2",
        widen_factor=1.0,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
    ),
    decode_head=dict(in_channels=320, c1_in_channels=24),
    auxiliary_head=dict(in_channels=96),
)

checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-d256dd4b.pth"
