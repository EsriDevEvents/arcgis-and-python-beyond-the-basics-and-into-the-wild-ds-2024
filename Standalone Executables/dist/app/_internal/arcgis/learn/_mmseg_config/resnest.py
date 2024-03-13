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


_base_ = "./deeplabv3.py"
model = dict(
    pretrained="open-mmlab://resnest101",
    backbone=dict(
        type="ResNeSt",
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
    ),
)

checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes/deeplabv3_s101-d8_512x1024_80k_cityscapes_20200807_144429-b73c4270.pth"
