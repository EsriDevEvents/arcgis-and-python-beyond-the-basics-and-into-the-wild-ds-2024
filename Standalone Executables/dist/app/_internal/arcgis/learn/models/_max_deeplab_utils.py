"""
Model architecture code from: https://github.com/conradry/max-deeplab

The following LICENSE applies to this file ONLY.

MIT License

Copyright (c) 2021 Ryan Conrad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# necessary imports
import math
import traceback

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models.resnet import Bottleneck
    from fastai.torch_core import to_cpu
    import numpy as np
    from einops import rearrange, repeat
    from scipy.optimize import linear_sum_assignment
    from ..models._unet_utils import is_contiguous
    from ._inferencing.util import remap
    from .._utils.common import (
        get_nbatches,
        get_symbology_bands,
        dynamic_range_adjustment,
        kwarg_fill_none,
    )

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class _MaXDeepLabModel:
    "This class defines the model architecture of MaXDeepLab."


#############
## Model   ##
#############


class MaXDeepLabS(nn.Module):
    def __init__(self, im_size=224, n_heads=8, n_classes=80, n_masks=50, in_channels=3):
        super(MaXDeepLabS, self).__init__()
        self.encoder = MaXDeepLabSEncoder(
            im_size=im_size, n_heads=n_heads, in_channels=in_channels
        )
        self.decoder = MaXDeepLabSDecoder(
            im_size=im_size, n_heads=n_heads, n_classes=n_classes, n_masks=n_masks
        )

        self.semantic_head = nn.Sequential(
            conv_bn_relu(2048, 256, 5, padding=2, groups=256),
            conv_bn_relu(256, n_classes, 1, with_bn=False, with_relu=False),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
        )

        self.global_memory = nn.Parameter(
            torch.randn((n_masks, 256)), requires_grad=True
        )

    ## TODO: Check what are the two forwards for?
    def forward(self, x):
        return self.conv1x1(self.conv5x5(x))

    def forward(self, P):
        """
        P: pixel NestedTensor (B, 3, H, W)
        """

        # P is a nested tensor, extract the image data
        # see utils.misc.NestedTensor
        # TODO: Replace with the image tensor, can get away with NestedTensor. Updated.
        # P, sizes = P.decompose()
        M = repeat(self.global_memory, "n k -> n b k", b=P.size(0))

        fmaps, mem = self.encoder(P, M)
        semantic_mask = self.semantic_head(fmaps[-1])
        mask_out, classes = self.decoder(fmaps, mem)
        return mask_out, classes, semantic_mask


class MaXDeepLabSEncoder(nn.Module):
    def __init__(
        self, layers=[3, 4, 6, 3], im_size=224, nin_memory=256, n_heads=8, in_channels=3
    ):
        super(MaXDeepLabSEncoder, self).__init__()

        self.base_width = 64
        self.nin = 64
        self.nin_memory = nin_memory
        backbone = MaXDeepLabSBackbone(layers, im_size, n_heads, in_channels)
        stages = backbone.get_stages()
        self.stem = stages[0]
        self.layer1 = stages[1]
        self.layer2 = stages[2]
        self.layer3 = stages[3]

        # overwrite layer 4 and replace with dual path
        del stages[4]
        kernel_size = im_size // 16
        self.nin *= 16
        self.layer4 = self._make_dualpath_layer(
            512, layers[3], n_heads=n_heads, kernel_size=kernel_size
        )

    def _make_dualpath_layer(
        self, planes, n_blocks, stride=1, base_width=64, n_heads=8, kernel_size=20
    ):
        block = DualPathXF
        downsample = None
        if stride != 1 or self.nin != planes * block.expansion:
            downsample = conv_bn_relu(
                self.nin, planes * block.expansion, 1, stride, with_relu=False
            )

        layers = []
        layers.append(
            block(
                self.nin,
                planes,
                self.nin_memory,
                stride,
                downsample,
                base_width=base_width,
                n_heads=n_heads,
                kernel_size=kernel_size,
            )
        )

        self.nin = planes * block.expansion
        kernel_size = kernel_size // stride
        for _ in range(1, n_blocks):
            layers.append(
                block(
                    self.nin,
                    planes,
                    self.nin_memory,
                    stride,
                    base_width=base_width,
                    n_heads=n_heads,
                    kernel_size=kernel_size,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, P, M):
        P1 = self.stem(P)  # H / 4
        P2 = self.layer1(P1)  # H / 4
        P3 = self.layer2(P2)  # H / 8
        P4 = self.layer3(P3)  # H / 16
        DP = self.layer4({"pixel": P4, "memory": M})  # H / 16
        P5, M = DP["pixel"], DP["memory"]

        return [P1, P2, P3, P4, P5], M


class MaXDeepLabSDecoder(nn.Module):
    def __init__(
        self,
        nin_pixel=2048,
        nplanes=512,
        nin_memory=256,
        im_size=640,
        n_heads=8,
        n_classes=19,
        n_masks=50,
    ):
        super(MaXDeepLabSDecoder, self).__init__()
        self.dual_path = DualPathXF(
            nin_pixel,
            nplanes,
            nin_memory,
            base_width=64,
            n_heads=8,
            kernel_size=im_size // 16,
        )

        self.bottleneck1 = DecoderBottleneck(
            nin_pixel, nplanes, upsample_factor=2, compression=4
        )

        nin_pixel = nin_pixel // 4
        nplanes = nplanes // 4
        self.bottleneck2 = DecoderBottleneck(
            nin_pixel, nplanes, upsample_factor=2, compression=2
        )

        nin_pixel = nin_pixel // 2
        self.mask_head = MaskHead(nin_pixel, n_masks)

        self.mem_mask = nn.Sequential(
            linear_bn_relu(nin_memory, nin_memory),
            linear_bn_relu(nin_memory, n_masks, with_relu=False),
        )

        self.mem_class = nn.Sequential(
            linear_bn_relu(nin_memory, nin_memory), nn.Linear(nin_memory, n_classes)
        )

        self.fg_bn = nn.BatchNorm2d(n_masks)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

    def forward(self, P_features, M):
        P1, P2, P3, P4, P5 = P_features
        dp_out = self.dual_path({"pixel": P5, "memory": M})
        P_up, M = dp_out["pixel"], dp_out["memory"]

        P_up = self.bottleneck1(P_up, P3)
        P_up = self.bottleneck2(P_up, P2)
        mask_up = self.mask_head(P_up)  # (B, D, H/4, W/4)

        # handle memory multiplication
        mem_mask = self.mem_mask(M)  # (N, B, D)
        mask_out = torch.einsum("nbd,bdhw->bnhw", mem_mask, mask_up)
        mask_out = self.fg_bn(mask_out)
        mask_out = self.upsample(mask_out)

        classes = self.mem_class(M)  # (N, B, n_classes)
        classes = rearrange(classes, "n b c -> b n c")

        return mask_out, classes


#############
## Backbone
#############


class MaXDeepLabSBackbone(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], im_size=640, n_heads=8, in_channels=3):
        super(MaXDeepLabSBackbone, self).__init__()

        self.base_width = 64
        self.nin = 64
        self.stem = InceptionStem(in_channels, 128)

        self.layer1 = self._make_bottleneck_layer(
            64, layers[0], stride=1, first_layer=True
        )
        self.layer2 = self._make_bottleneck_layer(128, layers[1], stride=2)

        kernel_size = im_size // 8
        self.layer3 = self._make_axial_layer(
            256, layers[2], stride=2, n_heads=n_heads, kernel_size=kernel_size
        )

        kernel_size = im_size // 16
        self.layer4 = self._make_axial_layer(
            512, layers[3], stride=1, n_heads=n_heads, kernel_size=kernel_size
        )

    def get_stages(self):
        return [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def _make_bottleneck_layer(self, planes, n_blocks, stride=1, first_layer=False):
        block = Bottleneck
        downsample = None
        first_block_nin = self.nin * 2 if first_layer else self.nin

        if stride != 1 or self.nin != planes * block.expansion:
            downsample = conv_bn_relu(
                first_block_nin, planes * block.expansion, 1, stride, with_relu=False
            )

        layers = []
        layers.append(
            block(
                first_block_nin, planes, stride, downsample, base_width=self.base_width
            )
        )
        self.nin = planes * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.nin, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def _make_axial_layer(self, planes, n_blocks, stride=1, n_heads=8, kernel_size=56):
        block = AxialBottleneck
        downsample = None
        if stride != 1 or self.nin != planes * block.expansion:
            downsample = conv_bn_relu(
                self.nin, planes * block.expansion, 1, stride, with_relu=False
            )

        layers = []
        layers.append(
            block(
                self.nin,
                planes,
                stride,
                downsample,
                self.base_width,
                n_heads=n_heads,
                kernel_size=kernel_size,
            )
        )

        self.nin = planes * block.expansion
        kernel_size = kernel_size // stride
        for _ in range(1, n_blocks):
            layers.append(
                block(
                    self.nin,
                    planes,
                    base_width=self.base_width,
                    n_heads=n_heads,
                    kernel_size=kernel_size,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, P):
        P = self.stem(P)  # H / 4
        P = self.layer1(P)  # H / 4
        P = self.layer2(P)  # H / 8
        P = self.layer3(P)  # H / 16
        P = self.layer4(P)  # H / 16

        return P


#############
## Blocks
#############


class linear_bn_relu(nn.Module):
    """
    Default layer for linear operations.
    """

    def __init__(self, nin, nout, with_bn=True, with_relu=True):
        super(linear_bn_relu, self).__init__()
        self.l1 = nn.Linear(nin, nout, bias=not with_bn)
        self.bn1 = None
        self.relu = None

        if with_bn:
            self.bn1 = nn.BatchNorm1d(nout)
        if with_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # assumed input is (N, B, C)
        out = self.l1(x)
        if self.bn1 is not None:
            # permute to (B, C, N)
            out = out.permute(1, 2, 0)
            out = self.bn1(out)
            out = out.permute(2, 0, 1)
        if self.relu:
            out = self.relu(out)

        return out


class conv_bn_relu(nn.Module):
    """
    Default layer for convolution operations.
    """

    def __init__(
        self,
        nin,
        nout,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        with_bn=True,
        with_relu=True,
    ):
        super(conv_bn_relu, self).__init__()
        layers = [
            nn.Conv2d(
                nin,
                nout,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=not with_bn,
            )
        ]

        if with_bn:
            layers.append(nn.BatchNorm2d(nout))
        if with_relu:
            layers.append(nn.ReLU(inplace=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InceptionStem(nn.Module):
    """
    Input stem.
    """

    def __init__(self, nin=3, nout=128):
        super(InceptionStem, self).__init__()
        self.net = nn.Sequential(
            conv_bn_relu(nin, nout // 2, 3, stride=2, padding=1),
            conv_bn_relu(nout // 2, nout // 2, 3, stride=1, padding=1),
            conv_bn_relu(nout // 2, nout, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        # (B, NIN, H, W) --> (B, NOUT, H/4, W/4)
        return self.net(x)


class AxialMultiHeadAttention(nn.Module):
    """
    Modified from https://github.com/csrhddlam/axial-deeplab/blob/master/lib/models/axialnet.py.
    """

    def __init__(self, nin, nout, n_heads=8, kernel_size=40, stride=1, axis="height"):
        super(AxialMultiHeadAttention, self).__init__()
        self.nin = nin
        self.nout = nout
        self.n_heads = n_heads
        self.head_nin = nout // n_heads
        self.kernel_size = kernel_size
        self.axis = axis

        self.qkv = nn.Sequential(
            nn.Conv1d(nin, nout * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(nout * 2),
        )
        self.bn_attn = nn.BatchNorm2d(n_heads * 3)
        self.bn_output = nn.BatchNorm1d(nout * 2)

        # (HIN * 2, KS * 2 - 1)
        self.pos_emb = nn.Parameter(
            torch.randn(self.head_nin * 2, kernel_size * 2 - 1), requires_grad=True
        )
        query_index = torch.arange(kernel_size)[None, :]  # (1, KS)
        key_index = torch.arange(kernel_size)[:, None]  # (KS, 1)

        # (KS, 1) - (1, KS) --> (KS, KS)
        relative_index = (key_index - query_index) + (kernel_size - 1)
        self.register_buffer("flat_index", relative_index.view(-1))  # (KS * KS)

        self.avg_pool = nn.AvgPool2d(stride, stride=stride) if stride != 1 else None

        self.reset_parameters()

    def reset_parameters(self):
        # initialize qkv conv1d layer
        self.qkv[0].weight.data.normal_(0, math.sqrt(1.0 / self.nin))
        # and position embedding
        nn.init.normal_(self.pos_emb, 0.0, math.sqrt(1.0 / self.head_nin))

    def forward(self, x):
        if self.axis == "height":
            x = rearrange(x, "n c h w -> n w c h")
        else:
            x = rearrange(x, "n c h w -> n h c w")

        N, W, C_in, H = x.shape
        x = rearrange(x, "n i c j -> (n i) c j")

        # define other useful dimensions
        C_out = self.nout
        kernel_size = self.kernel_size
        n_heads = self.n_heads
        head_nin = self.head_nin
        qkdim = head_nin // 2
        vdim = head_nin
        # NOTE: head_nin * 2 = qkdim + qkdim + vdim

        qkv = self.qkv(x)  # (N * W, C_out * 2, H)
        qkv = rearrange(qkv, "nw (a b) x -> nw a b x", a=n_heads, b=head_nin * 2)
        q, k, v = torch.split(qkv, [qkdim, qkdim, vdim], dim=2)

        embeddings = self.pos_emb[:, self.flat_index]
        embeddings = embeddings.view(head_nin * 2, kernel_size, kernel_size)
        qemb, kemb, vemb = torch.split(embeddings, [qkdim, qkdim, vdim], dim=0)

        # (N * W, n_heads, head_nin / 2, H) x (head_nin / 2, H, H)
        # --> (N * W, n_heads, H, H)
        qr = torch.einsum("bnci,cij->bnij", q, qemb)
        kr = torch.einsum("bnci,cij->bnji", k, kemb)  # note the transpose
        qk = torch.einsum("bnci, bncj->bnij", q, k)

        # (N * W, 3 * n_heads, H, H)
        stacked_attn = self.bn_attn(torch.cat([qk, qr, kr], dim=1))
        stacked_attn = rearrange(
            stacked_attn, "b (a n) i j -> b a n i j", a=3, n=n_heads
        )
        stacked_attn = stacked_attn.sum(1)  # (N * W, n_heads, H, H)
        attn = F.softmax(stacked_attn, dim=3)

        # attend to values
        sv = torch.einsum("bnij,bncj->bnci", attn, v)
        svemb = torch.einsum("bnij,cij->bnci", attn, vemb)

        # (N * W, n_heads, head_nin, 2 * H) --> (N * W, C_out * 2, H)
        stacked_y = torch.cat([sv, svemb], dim=-1)
        stacked_y = rearrange(
            stacked_y, "b n c (k i) -> b (n c k) i", n=n_heads, k=2, i=H
        )
        y = self.bn_output(stacked_y)

        y = y.view(N, W, C_out, 2, H).sum(dim=-2)  # (N, W, C_out, H)

        if self.axis == "height":
            y = rearrange(y, "n w c h -> n c h w")
        else:
            y = rearrange(y, "n h c w -> n c h w")

        if self.avg_pool is not None:
            y = self.avg_pool(y)

        return y


class AxialBottleneck(nn.Module):
    """
    Modified from https://github.com/csrhddlam/axial-deeplab/blob/master/lib/models/axialnet.py.

    ResNet-style bottleneck block with conv3x3 replaced by AxialMultiHeadAttention layers.
    """

    expansion = 4

    def __init__(
        self,
        nin,
        nplanes,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        n_heads=8,
        kernel_size=56,
    ):
        super(AxialBottleneck, self).__init__()

        width = int(nplanes * (base_width / 64.0))
        self.axial_net = nn.Sequential(
            conv_bn_relu(nin, width, kernel_size=1),
            AxialMultiHeadAttention(width, width, n_heads, kernel_size, axis="height"),
            AxialMultiHeadAttention(
                width, width, n_heads, kernel_size, stride=stride, axis="width"
            ),
            nn.ReLU(inplace=False),
            conv_bn_relu(
                width, nplanes * self.expansion, kernel_size=1, with_relu=False
            ),
        )

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.axial_net(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DualPathXF(nn.Module):
    """
    The dual path transformer module.
    """

    expansion = 4

    def __init__(
        self,
        nin_pixel,
        nplanes,
        nin_memory,
        stride=1,
        downsample=None,
        base_width=64,
        n_heads=8,
        kernel_size=20,
    ):
        super(DualPathXF, self).__init__()

        # nplanes = 1024
        self.p2p = AxialBottleneck(
            nin_pixel,
            nplanes,
            stride,
            downsample,
            base_width=base_width,
            n_heads=n_heads,
            kernel_size=kernel_size,
        )

        nin_pixel = nplanes * self.expansion

        self.p2m_conv1 = conv_bn_relu(nin_pixel, nplanes, 1)
        self.p2m_qkv = nn.Sequential(
            nn.Conv2d(nplanes, nplanes * 2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(nplanes * 2),
        )
        self.p2m_conv2 = nn.Sequential(
            nn.Conv2d(nplanes, nplanes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(nplanes * self.expansion),
        )

        # memory qkv
        # nplanes = 512
        self.mem_fc1 = linear_bn_relu(nin_memory, nplanes)
        self.mem_qkv = linear_bn_relu(nplanes, nplanes * 2, with_relu=False)
        self.mem_fc2 = linear_bn_relu(nplanes, nin_memory, with_relu=False)
        self.relu = nn.ReLU(inplace=False)

        self.mem_ffn = nn.Sequential(
            linear_bn_relu(nin_memory, nplanes * self.expansion),
            linear_bn_relu(nplanes * self.expansion, nin_memory, with_relu=False),
        )

        # useful dimensions
        self.n_heads = n_heads
        self.head_nin = nplanes // n_heads
        self.dq = self.head_nin // 2
        self.dv = self.head_nin

    def forward(self, x_dict):
        P = x_dict["pixel"]
        M = x_dict["memory"]

        # useful dimensions
        B, C, H, W = P.size()
        N, B, K = M.size()
        n_heads = self.n_heads  # labeled 'i' in einsums
        head_nin = self.head_nin
        dq = self.dq
        dv = self.dv

        # P is pixel (image), M is memory
        P = self.p2p(P)
        P_identity = P
        M_identity = M

        # apply image path qkv
        # (B, C_out * 2, H, W)
        P_qkv = self.p2m_conv1(P)
        P_qkv = self.p2m_qkv(P_qkv)
        P_qkv = rearrange(P_qkv, "b (i j) h w -> b i j h w", i=n_heads, j=head_nin * 2)
        qp, kp, vp = torch.split(P_qkv, [dq, dq, dv], dim=2)

        # (N, B, K)
        M_qkv = self.mem_fc1(M)
        M_qkv = self.mem_qkv(M_qkv)
        M_qkv = rearrange(M_qkv, "n b (i j) -> n b i j", i=n_heads, j=head_nin * 2)
        qm, km, vm = torch.split(M_qkv, [dq, dq, dv], dim=3)

        # P2M output it ypa in paper
        # qp: (B, n_heads, dq, h, w), km: (N, B, n_heads, dq)
        p2m = torch.einsum("bijhw,nbij->bnihw", qp, km)  # (B, N, n_heads, H, W)
        p2m_attn = F.softmax(p2m, dim=1)
        ypa = torch.einsum(
            "bnihw,nbij->bijhw", p2m_attn, vm
        )  # (B, n_heads, head_nin, H, W)

        # handle m2p and m2m together
        kp = rearrange(kp, "b i j h w -> b i j (h w)")
        km = rearrange(km, "n b i j -> b i j n")
        kpm = torch.cat([kp, km], dim=3)  # (B, n_heads, dq * 2, H * W + N)

        vp = rearrange(vp, "b i j h w -> b i j (h w)")
        vm = rearrange(vm, "n b i j -> b i j n")
        vpm = torch.cat([vp, vm], dim=3)  # (B, n_heads, dv * 2, H * W + N)

        m2m_m2p = torch.einsum("nbij,bijl->nbil", qm, kpm)  # (N, B, n_heads, H * W + N)
        m2m_m2p_attn = F.softmax(m2m_m2p, dim=-1)
        ymb = torch.einsum(
            "nbil,bijl->nbij", m2m_m2p_attn, vpm
        )  # (N, B, n_heads, head_nin)

        P_out = self.p2m_conv2(rearrange(ypa, "b i j h w -> b (i j) h w"))
        P_out += P_identity
        P_out = self.relu(P_out)

        M_out = self.mem_fc2(rearrange(ymb, "n b i j -> n b (i j)"))
        M_out += M_identity
        M_out = self.relu(M_out)

        M_ffn = self.mem_ffn(M_out)
        M_out += M_ffn
        M_out = self.relu(M_out)

        return {"pixel": P_out, "memory": M_out}


class DecoderBottleneck(nn.Module):
    """
    ResNet-style bottleneck block in the decoder with skip connection
    for encoder layer outputs.
    """

    expansion = 2

    def __init__(self, nin, nplanes, upsample_factor=2, compression=4):
        super(DecoderBottleneck, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=upsample_factor, mode="bilinear", align_corners=True
        )
        self.relu = nn.ReLU(inplace=False)

        # see Fig 9. of https://arxiv.org/abs/2003.07853
        self.identity_path = nn.Sequential(
            conv_bn_relu(nin, nin // compression, kernel_size=1, with_relu=False),
            self.upsample,
        )

        self.bottleneck_path = nn.Sequential(
            conv_bn_relu(nin, nplanes, kernel_size=1),
            self.upsample,
            conv_bn_relu(nplanes, nplanes, kernel_size=3, padding=1),
            conv_bn_relu(nplanes, nin // compression, kernel_size=1, with_relu=False),
        )

        self.encoder_feature_path = nn.Sequential(
            conv_bn_relu(
                nin // compression, nin // compression, kernel_size=1, with_relu=False
            )
        )

        self.proj_down = conv_bn_relu(
            nin // compression, nin // compression, kernel_size=1
        )

    def forward(self, x, skip):
        identity = self.identity_path(x)
        bottleneck_out = self.bottleneck_path(x)
        skip_out = self.encoder_feature_path(skip)
        out = self.relu(identity + bottleneck_out + skip_out)

        return self.proj_down(out)


class MaskHead(nn.Module):
    """
    Generates the masks prior to multiplication by global memory.
    """

    def __init__(self, nplanes, nout, kernel_size=5, padding=2, separable=True):
        super(MaskHead, self).__init__()
        groups = nplanes if separable else 1
        self.conv5x5 = conv_bn_relu(
            nplanes, nplanes, kernel_size, padding=padding, groups=groups
        )
        self.conv1x1 = conv_bn_relu(nplanes, nout, 1, with_relu=False)

    def forward(self, x):
        return self.conv1x1(self.conv5x5(x))


#############
## Loss
#############


# @torch.jit.script
def cdice_similarity(input_mask, target_mask, eps=1e-5):
    """
    input mask: (B, N, HW) #probabilities [0, 1]
    target_mask: (B, K, HW) #binary
    """

    input_mask = input_mask.unsqueeze(2)  # (B, N, 1, HW)
    target_mask = target_mask.unsqueeze(1)  # (B, 1, K, HW)
    # (B, N, 1, HW) * (B, 1, K, HW) --> (B, N, K, HW)

    intersections = torch.sum(input_mask * target_mask, dim=-1)
    cardinalities = torch.sum(input_mask + target_mask, dim=-1)
    dice = (2.0 * intersections + eps) / (cardinalities + eps)
    return dice


# @torch.jit.script
def dice_score(input_mask, target_mask, eps=1e-5):
    """
    input mask: (B * K, HW) #probabilities [0, 1]
    target_mask: (B * K, HW) #binary
    """

    dims = tuple(range(1, input_mask.ndimension()))
    intersections = torch.sum(input_mask * target_mask, dims)  # (B, N)
    cardinalities = torch.sum(input_mask + target_mask, dims)
    dice = (2.0 * intersections + eps) / (cardinalities + eps)
    return dice


class HungarianMatcher(nn.Module):
    """
    Heavily inspired by https://github.com/facebookresearch/detr/blob/master/models/matcher.py.
    """

    def __init__(self):
        super(HungarianMatcher, self).__init__()

    @torch.no_grad()
    def forward(
        self, input_class_prob, input_mask, target_class, target_mask, target_sizes
    ):
        """
        input_class: (B, N, N_CLASSES) #probabilities
        input mask: (B, N, H, W) #probabilities [0, 1]
        target_class: (B, K) #long indices
        target_mask: (B, K, H, W) #bool
        target_sizes: (B,) #number of masks that are not padding (i.e. no class)
        """
        device = input_class_prob.device
        B, N = input_class_prob.size()[:2]
        K = target_class.size(-1)

        # we want similarity matrices to size (B, N, K)
        # where N is number of predicted objects and K is number of gt objects
        # (B, N, C)[(B, N, K)] --> (B, N, K)
        sim_class = input_class_prob.gather(
            -1, repeat(target_class, "b k -> b n k", n=N)
        )  # TODO: Understand this part
        sim_dice = cdice_similarity(input_mask, target_mask)
        # final cost matrix (RQ x SQ from the paper, eqn 9)
        sim = (sim_class * sim_dice).cpu()  # (B, N, K)

        # each example in batch, ignore null objects in target (i.e. padding)
        indices = [
            linear_sum_assignment(s[:, :e], maximize=True)
            for s, e in zip(sim, target_sizes)
        ]

        # at this junctions everything is matched, now it's just putting
        # the indices into easily usable formats

        input_pos_indices = []
        target_pos_indices = []
        input_neg_indices = []
        input_indices = np.arange(0, N)
        for i, (inp_idx, tgt_idx) in enumerate(indices):
            input_pos_indices.append(
                torch.as_tensor(inp_idx, dtype=torch.long, device=device)
            )
            target_pos_indices.append(
                torch.as_tensor(tgt_idx, dtype=torch.long, device=device)
            )
            input_neg_indices.append(
                torch.as_tensor(
                    np.setdiff1d(input_indices, inp_idx),
                    dtype=torch.long,
                    device=device,
                )
            )

        # here the lists of indices have variable lengths
        # and sizes; make 1 tensor of size (B * N_pos) for all
        # positives first: shared by input_pos_indices and target_pos_indices
        batch_pos_idx = torch.cat(
            [torch.full_like(pos, i) for i, pos in enumerate(input_pos_indices)]
        )
        batch_neg_idx = torch.cat(
            [torch.full_like(neg, i) for i, neg in enumerate(input_neg_indices)]
        )
        input_pos_indices = torch.cat(input_pos_indices)
        target_pos_indices = torch.cat(target_pos_indices)
        input_neg_indices = torch.cat(input_neg_indices)

        inp_pos_indices = (batch_pos_idx, input_pos_indices)
        tgt_pos_indices = (batch_pos_idx, target_pos_indices)
        inp_neg_indices = (batch_neg_idx, input_neg_indices)
        return inp_pos_indices, tgt_pos_indices, inp_neg_indices


class PQLoss(nn.Module):
    def __init__(self, alpha=0.75, eps=1e-5):
        super(PQLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.xentropy = nn.CrossEntropyLoss(reduction="none")
        self.matcher = HungarianMatcher()
        self.negative_xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, input_class, target_mask, target_class, target_sizes):
        """
        input_class: (B, N, N_CLASSES) #logits
        input mask: (B, N, H, W) #probabilities [0, 1]
        target_class: (B, K) #long indices
        target_mask: (B, K, H, W) #binary
        """
        # apply softmax to get probabilities from logits
        B, N, num_classes = input_class.size()
        input_mask = F.softmax(input_mask, dim=1)
        input_class_prob = F.softmax(input_class, dim=-1)
        input_mask = rearrange(input_mask, "b n h w -> b n (h w)")
        target_mask = rearrange(target_mask, "b k h w -> b k (h w)")

        # match input and target
        inp_pos_indices, tgt_pos_indices, neg_indices = self.matcher(
            input_class_prob, input_mask, target_class, target_mask, target_sizes
        )

        # select masks and labels by indices
        # (B < len(inp_pos_indices) <= B * K)
        # (0 <= len(neg_indices) <= B * (N - K))
        matched_input_class = input_class[inp_pos_indices]
        matched_input_class_prob = input_class_prob[inp_pos_indices]
        matched_target_class = target_class[tgt_pos_indices]
        negative_class = input_class[neg_indices]

        matched_input_mask = input_mask[inp_pos_indices]
        matched_target_mask = target_mask[tgt_pos_indices]
        negative_mask = input_mask[neg_indices]

        # NP is len(inp_pos_indices)
        # NN is len(neg_indices)
        with torch.no_grad():
            class_weight = matched_input_class_prob.gather(
                -1, matched_target_class[:, None]
            )  # (NP,)
            dice_weight = dice_score(
                matched_input_mask, matched_target_mask, self.eps
            )  # (NP,)

        cross_entropy = self.xentropy(
            matched_input_class, matched_target_class
        )  # (NP,)
        dice = dice_score(matched_input_mask, matched_target_mask, self.eps)  # (NP,)

        # eqn 10
        # (1 - dice) is so that the minimum loss value is 0 and not -1
        l_pos = (class_weight * (1 - dice) + dice_weight * cross_entropy).mean()

        # eqn 11
        negative_target_class = torch.zeros(
            size=(len(negative_class),),
            dtype=target_class.dtype,
            device=target_class.device,
        )
        l_neg = self.negative_xentropy(negative_class, negative_target_class).mean()

        # eqn 12
        return self.alpha * l_pos + (1 - self.alpha) * l_neg


# -----------------------
### Auxiliary Losses ###
# -----------------------


class InstanceDiscLoss(nn.Module):
    def __init__(self, temp=0.3, eps=1e-5):
        super(InstanceDiscLoss, self).__init__()
        self.temp = temp
        self.eps = eps

    def forward(self, mask_features, target_mask, target_sizes):
        """
        mask_features: (B, D, H, W) #g
        target_mask: (B, K, H, W) #m
        """

        # downsample input and target by 4 to get (B, H/4, W/4)
        mask_features = mask_features[..., ::4, ::4]
        target_mask = target_mask[..., ::4, ::4]

        device = mask_features.device

        # eqn 16
        t = torch.einsum("bdhw,bkhw->bkd", mask_features, target_mask)
        t = F.normalize(t, dim=-1)  # (B, K, D)

        # get batch and mask indices from target_sizes
        batch_indices = []
        mask_indices = []
        for bi, size in enumerate(target_sizes):
            mindices = torch.arange(0, size, dtype=torch.long, device=device)
            mask_indices.append(mindices)
            batch_indices.append(torch.full_like(mindices, bi))

        batch_indices = torch.cat(
            batch_indices, dim=0
        )  # shape: (torch.prod(target_sizes), )
        mask_indices = torch.cat(mask_indices, dim=0)

        # create logits and apply temperature
        logits = torch.einsum("bdhw,bkd->bkhw", mask_features, t)
        logits = logits[batch_indices, mask_indices]  # (torch.prod(target_sizes), H, W)
        logits /= self.temp

        # select target_masks
        m = target_mask[batch_indices, mask_indices]  # (torch.prod(target_sizes), H, W)

        # flip so that there are HW examples for torch.prod(target_sizes) classes
        logits = rearrange(logits, "k h w -> (h w) k")
        m = rearrange(m, "k h w -> (h w) k")

        # eqn 17
        numerator = torch.logsumexp(m * logits, dim=-1)  # (HW,)
        denominator = torch.logsumexp(logits, dim=-1)  # (HW,)

        # log of quotient is difference of logs
        return (-numerator + denominator).mean()


class MaskIDLoss(nn.Module):
    def __init__(self):
        super(MaskIDLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, target_mask):
        """
        input_mask: (B, N, H, W) #logits
        target_mask: (B, H, W) #long indices of maskID in N
        """
        return self.xentropy(input_mask, target_mask)


class SemanticSegmentationLoss(nn.Module):
    def __init__(self, method="cross_entropy"):
        super(SemanticSegmentationLoss, self).__init__()
        if method != "cross_entropy":
            raise NotImplementedError
        else:
            # they don't specify the loss function
            # could be regular cross entropy or
            # dice loss or focal loss etc.
            # keep it simple for now
            self.xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, target_mask):
        """
        input_mask: (B, NUM_CLASSES, H, W) #logits
        target_mask: (B, H, W) #long indices
        """
        return self.xentropy(input_mask, target_mask)


class MaXDeepLabLoss(nn.Module):
    def __init__(
        self,
        pq_loss_weight=3,
        instance_loss_weight=1,
        maskid_loss_weight=0.3,
        semantic_loss_weight=1,
        alpha=0.75,
        temp=0.3,
        eps=1e-5,
    ):
        super(MaXDeepLabLoss, self).__init__()
        self.pqw = pq_loss_weight
        self.idw = instance_loss_weight
        self.miw = maskid_loss_weight
        self.ssw = semantic_loss_weight
        self.pq_loss = PQLoss(alpha, eps)
        self.instance_loss = InstanceDiscLoss(temp, eps)
        self.maskid_loss = MaskIDLoss()
        self.semantic_loss = SemanticSegmentationLoss()

    def forward(self, input_tuple, target_tuple):
        """
        input_tuple: (input_masks, input_classes, input_semantic_segmentation) Tensors
        target_tuple: (gt_masks, gt_classes, gt_semantic_segmentation) NestedTensors
        """
        input_masks, input_classes, input_ss = input_tuple
        gt_masks, gt_classes, gt_ss = target_tuple
        gt_masks = gt_masks.float()
        gt_ss = gt_ss.squeeze(dim=1)

        # Find the number of targets in each image
        target_sizes = torch.sum(
            gt_classes.clamp(0, 1), dim=(1)
        )  # TODO: Replace with torch.count_nonzero() in Pytorch v1.8

        pq = self.pq_loss(
            input_masks, input_classes, gt_masks, gt_classes, target_sizes
        )
        instdisc = self.instance_loss(input_masks, gt_masks.float(), target_sizes)

        # create the mask for maskid loss using argmax on ground truth
        maskid = self.maskid_loss(input_masks, gt_masks.argmax(1))
        semantic = self.semantic_loss(input_ss, gt_ss)

        loss_items = {
            "pq": pq.item(),
            "semantic": semantic.item(),
            "maskid": maskid.item(),
            "instdisc": instdisc.item(),
        }

        total_loss = (
            self.pqw * pq
            + self.ssw * semantic
            + self.miw * maskid
            + self.idw * instdisc
        )

        return total_loss, loss_items


# -----------------------
### Visualization ###
# -----------------------

import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt


def roll_image(image):
    """helper for displaying images"""
    image = np.rollaxis(image, 0, 3)
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype(np.uint8)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return image


def display_instances(
    image,
    masks,
    class_ids,
    class_names,
    figsize=(16, 16),
    ax=None,
    show_mask=True,
    color_mapping=None,
    instance_classes=[None],
    alpha=0.5,
):
    """
    masks: [height, width, num_instances]
    class_names: list of class names for each mask
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    color_mapping: (optional) A dict of class value and colors to use with each object
    """
    # Number of instances
    N = len(masks)
    assert len(masks) == len(class_names)

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Add black color for NoData
    if color_mapping:
        color_mapping[0] = [0, 0, 0]

    # Generate random colors for instance classes in ground truth and instance predictions
    # and use color mapping for semantic classes
    colors = []
    if color_mapping:
        for i in range(N):
            if class_ids[i] not in instance_classes:
                colors.append(
                    list(map(lambda i: i / 255.0, color_mapping[class_ids[i]]))
                )
            else:
                colors.append(list(map(lambda i: random.random(), range(3))))
    else:
        colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.axis("off")

    masked_image = image.astype(np.uint32).copy()
    for i, mask in enumerate(masks):
        if not np.any(mask):
            continue

        color = colors[i]

        # Label (Uncomment to display labels)
        # y, x = np.where(mask > 0)
        # y1 = np.median(y)
        # x1 = np.median(x)
        # ax.text(x1, y1 + 8, class_names[i], color="w", size=14, backgroundcolor="none")

        # Mask
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color, alpha)

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


def show_results_panoptic(model, rows=5, thresh=0.5, **kwargs):
    alpha = kwargs.get("alpha", 0.5)
    do_random_color = kwargs.get("random_colors", False)
    statistics_type = kwarg_fill_none(
        kwargs, "statistics_type", "dataset"
    )  # Accepted Values `dataset`, `DRA`

    ds_type = DatasetType.Valid
    n_items = rows
    # Limit the number of items/rows to display to batch size
    if model.learn.dl(ds_type).batch_size < n_items:
        n_items = model.learn.dl(ds_type).batch_size

    # Fetch a batch, apply transforms and get predictions through the model
    data = model.learn.data
    xb, yb = data.one_batch(ds_type, detach=False, denorm=False)
    transform_kwargs, kwargs = split_kwargs_by_func(
        kwargs, model._model_conf.transform_input
    )

    # Put images and labels/masks on cpu and denorm
    x, y = to_cpu(xb), to_cpu(yb)
    norm = getattr(data, "norm", False)
    if norm:
        x = data.denorm(x)
        if norm.keywords.get("do_y", False):
            y = data.denorm(y, do_x=True)

    # For multispectral data
    symbology_bands = [0, 1, 2]
    if data._is_multispectral:
        # Get RGB Bands for plotting
        rgb_bands = kwarg_fill_none(kwargs, "rgb_bands", data._symbology_rgb_bands)

        # Get Symbology bands
        symbology_bands = get_symbology_bands(
            rgb_bands, data._extract_bands, data._bands
        )

        # Extract RGB Bands for plotting
        x = x[:, symbology_bands]

    # Apply Image Strecthing
    if statistics_type == "DRA":
        x = dynamic_range_adjustment(x)

    model.learn.model.eval()
    with torch.no_grad():
        preds = model.learn.model(
            model._model_conf.transform_input(xb, **transform_kwargs)
        )

    # Apply postprocessing to preds
    # analyze_kwargs, kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
    # preds = ds.y.analyze_pred(preds, **analyze_kwargs)

    N = model.learn.data.K
    instance_probs = F.softmax(preds[0], dim=1)
    instances = instance_probs.argmax(dim=1)
    instances = F.one_hot(instances, num_classes=N).permute(0, 3, 1, 2)
    class_confidence, classes = F.softmax(preds[1], dim=-1).max(-1)
    semantic = F.softmax(preds[2], dim=1).argmax(dim=1)

    category_dict = model._data.class_mapping.copy()
    category_dict[0] = "NoData"
    instance_classes = model._data.instance_classes

    # Remap to original class mapping if non-contiguous classes
    is_contig = is_contiguous(sorted([0] + list(category_dict.keys())))
    if not is_contig:
        pixel_mapping = sorted(list(category_dict.keys()))
        idx2pixel = {i: d for i, d in enumerate(pixel_mapping)}
        semantic = remap(semantic, idx2pixel)
        classes = remap(classes, idx2pixel)
        y[1] = remap(y[1], idx2pixel)

    # Filter predictions for instances
    inst_cls = classes
    for i in instance_classes:
        inst_cls = torch.where(
            inst_cls == i, torch.tensor(-1).to(classes.device), inst_cls
        )
    keep_pred_instances = torch.where(
        torch.logical_and(inst_cls == -1, class_confidence > thresh)
    )

    pred_instances = []
    pred_classes = []
    pred_class_names = []
    pred_scores = []

    for index in range(n_items):
        keep_pred = keep_pred_instances[1][keep_pred_instances[0] == index]
        pred_instances.append(instances.detach()[index, keep_pred].cpu().numpy())
        pred_classes.append(classes.detach()[index, keep_pred].cpu().numpy())
        pred_scores.append(class_confidence.detach()[index, keep_pred].cpu().numpy())
        pred_class_names.append([category_dict[l] for l in pred_classes[-1]])

    # remove padding instances in gt
    keep_gt_instances = torch.where(y[1] > 0)  # yb[1] are labels

    gt_instances = []
    gt_classes = []
    gt_class_names = []

    pred_semantics = []
    pred_semantic_classes = []
    pred_semantic_class_names = []

    for index in range(n_items):
        keep_gt = keep_gt_instances[1][keep_gt_instances[0] == index]
        gt_instances.append(y[0].detach()[index, keep_gt].cpu().numpy())
        gt_classes.append(y[1].detach()[index, keep_gt].cpu().numpy())
        gt_class_names.append([category_dict[l] for l in gt_classes[-1]])

        # Converting semantics to masks for each class to display in show_results
        semantic_np = np.asarray(semantic[index].detach().cpu())
        semantic_labels = np.unique(semantic[index].detach().cpu())
        semantic_masks = semantic_np == semantic_labels[:, None, None]
        semantic_masks = [mask.astype(np.uint8) for mask in semantic_masks]
        semantic_masks = np.stack(semantic_masks, axis=0)
        pred_semantics.append(semantic_masks)
        pred_semantic_classes.append(semantic_labels)
        pred_semantic_class_names.append(
            [category_dict[l] for l in pred_semantic_classes[-1]]
        )

    if do_random_color:
        color_mapping = None
    else:
        color_mapping = model._data.color_mapping

    f, ax = plt.subplots(n_items, 4, figsize=(18, 6 * n_items), squeeze=False)

    for index in range(n_items):
        # Display image
        display_image = roll_image(x.detach().cpu().numpy()[index])
        ax[index, 0].imshow(display_image)
        ax[index, 0].axis("off")
        # Display ground truth
        display_instances(
            display_image,
            gt_instances[index],
            gt_classes[index],
            gt_class_names[index],
            ax=ax[index, 1],
            color_mapping=color_mapping,
            instance_classes=instance_classes,
            alpha=alpha,
        )
        # Display instance mask predictions
        display_instances(
            display_image,
            pred_instances[index],
            pred_classes[index],
            pred_class_names[index],
            ax=ax[index, 2],
            color_mapping=color_mapping,
            instance_classes=instance_classes,
            alpha=alpha,
        )
        # Display semantic mask predictions
        display_instances(
            display_image,
            pred_semantics[index],
            pred_semantic_classes[index],
            pred_semantic_class_names[index],
            ax=ax[index, 3],
            color_mapping=color_mapping,
            alpha=alpha,
        )

    ax[0, 0].set_title("Image\n", fontsize=20)
    ax[0, 1].set_title("Labels\n", fontsize=20)
    ax[0, 2].set_title("Detected Objects\n", fontsize=20)
    ax[0, 3].set_title("Classified Pixels\n", fontsize=20)
    plt.tight_layout()


def show_batch_panoptic(data, rows=5, alpha=0.5, **kwargs):
    do_random_color = kwargs.get("random_colors", False)
    statistics_type = kwarg_fill_none(
        kwargs, "statistics_type", "dataset"
    )  # Accepted Values `dataset`, `DRA`

    type_data_loader = kwargs.get(
        "data_loader", "training"
    )  # options : traininig, validation, testing
    if type_data_loader == "training":
        data_loader = data.train_dl
    elif type_data_loader == "validation":
        data_loader = data.valid_dl
    elif type_data_loader == "testing":
        data_loader = data.test_dl
    else:
        e = Exception(
            f"could not find {type_data_loader} in data. Please ensure that the data loader type is traininig, validation or testing "
        )
        raise (e)

    category_dict = data.class_mapping
    bsz = data.batch_size
    n_batches = math.ceil(rows / bsz)
    n_items = rows

    xb, yb = get_nbatches(data_loader, n_batches)
    x = torch.cat(xb)
    y_m = torch.cat([yb[i][0] for i in range(len(yb))])
    y_l = torch.cat([yb[i][1] for i in range(len(yb))])
    y_s = torch.cat([yb[i][2] for i in range(len(yb))])
    y = [y_m, y_l, y_s]

    # Remap to original class mapping if non-contiguous classes
    is_contig = is_contiguous(sorted([0] + list(category_dict.keys())))
    if not is_contig:
        pixel_mapping = [0] + list(category_dict.keys())
        idx2pixel = {i: d for i, d in enumerate(pixel_mapping)}
        y[1] = remap(y[1], idx2pixel)

    # Put images and labels/masks on cpu and denorm
    norm = getattr(data, "norm", False)
    if norm:
        x = data.denorm(x)

    # For multispectral data
    symbology_bands = [0, 1, 2]
    if data._is_multispectral:
        # Get RGB Bands for plotting
        rgb_bands = kwarg_fill_none(kwargs, "rgb_bands", data._symbology_rgb_bands)

        # Get Symbology bands
        symbology_bands = get_symbology_bands(
            rgb_bands, data._extract_bands, data._bands
        )

        # Extract RGB Bands for plotting
        x = x[:, symbology_bands]

    # Apply Image Strecthing
    if statistics_type == "DRA":
        x = dynamic_range_adjustment(x)

    # remove padding instances in gt
    keep_gt_instances = torch.where(y[1] > 0)  # y[1] are labels

    gt_instances = []
    gt_classes = []
    gt_class_names = []

    for index in range(n_items):
        keep_gt = keep_gt_instances[1][keep_gt_instances[0] == index]
        gt_instances.append(y[0].detach()[index, keep_gt].cpu().numpy())
        gt_classes.append(y[1].detach()[index, keep_gt].cpu().numpy())
        gt_class_names.append([category_dict[l] for l in gt_classes[-1]])

    if do_random_color:
        color_mapping = None
    else:
        color_mapping = data.color_mapping

    instance_classes = data.instance_classes

    f, ax = plt.subplots(n_items, 2, figsize=(10, 4 * n_items), squeeze=False)

    for index in range(n_items):
        # Display image
        display_image = roll_image(x.detach().cpu().numpy()[index])
        ax[index, 0].imshow(display_image)
        ax[index, 0].axis("off")

        # Display ground truth
        display_instances(
            display_image,
            gt_instances[index],
            gt_classes[index],
            gt_class_names[index],
            ax=ax[index, 1],
            color_mapping=color_mapping,
            instance_classes=instance_classes,
            alpha=alpha,
        )

    ax[0, 0].set_title("Image\n", fontsize=20)
    ax[0, 1].set_title("Labels\n", fontsize=20)
    plt.tight_layout()


# -------------
### Metrics ###
# -------------

from fastprogress.fastprogress import progress_bar
from fastai.basic_data import DatasetType
from fastai.core import split_kwargs_by_func


def compute_panoptic_quality(model, show_progress=True, **kwargs):
    ds_type = DatasetType.Valid
    dl = model._data.valid_dl
    transform_kwargs, kwargs = split_kwargs_by_func(
        kwargs, model._model_conf.transform_input
    )
    matcher = HungarianMatcher()
    eps = 1e-5
    scores = []

    with torch.no_grad():
        for input, target in progress_bar(dl, display=show_progress):
            pred = model.learn.model.eval()(
                model._model_conf.transform_input(input, **transform_kwargs)
            )
            pred_masks, pred_classes, pred_ss = pred
            target_masks, target_classes, target_ss = target
            target_masks = target_masks.float()

            # Find the number of targets in each image
            target_sizes = torch.sum(
                target_classes.clamp(0, 1), dim=(1)
            )  # TODO: Replace with torch.count_nonzero() in Pytorch v1.8

            # apply softmax to get probabilities from logits
            B, N, num_classes = pred_classes.size()
            pred_masks = F.softmax(pred_masks, dim=1)
            pred_class_prob = F.softmax(pred_classes, dim=-1)
            pred_masks = rearrange(pred_masks, "b n h w -> b n (h w)")
            target_masks = rearrange(target_masks, "b k h w -> b k (h w)")

            # match input and target
            inp_pos_indices, tgt_pos_indices, neg_indices = matcher(
                pred_class_prob,
                pred_masks,
                target_classes,
                target_masks,
                target_sizes,
            )

            # select masks and labels by indices
            # (B < len(inp_pos_indices) <= B * K)
            # (0 <= len(neg_indices) <= B * (N - K))
            matched_input_class = pred_classes[inp_pos_indices]
            matched_input_class_prob = pred_class_prob[inp_pos_indices]
            matched_target_class = target_classes[tgt_pos_indices]
            negative_class = pred_classes[neg_indices]

            matched_input_mask = pred_masks[inp_pos_indices]
            matched_target_mask = target_masks[tgt_pos_indices]
            negative_mask = pred_masks[neg_indices]

            class_weight = matched_input_class_prob.gather(
                -1, matched_target_class[:, None]
            )  # (NP,)
            dice_weight = dice_score(
                matched_input_mask, matched_target_mask, eps
            )  # (NP,)

            batch_score = class_weight * dice_weight.unsqueeze(1)
            batch_score = batch_score.mean().cpu()
            scores.append(batch_score)

        scores = np.mean(scores)

    return scores
