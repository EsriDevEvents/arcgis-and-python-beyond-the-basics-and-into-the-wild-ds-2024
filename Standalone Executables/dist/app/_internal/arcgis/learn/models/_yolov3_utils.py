"""
The following license is applicable to this file only.

This software uses some portions from the following software under its license:

Copyright (c) 2018 DeNA Co., Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, and/or sublicense 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software; and

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


chainercv

The MIT License

Copyright (c) 2017 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


# import necessary modules
import numpy as np
import torch
from torch import nn, LongTensor, Tensor
from collections import defaultdict
import warnings

from fastprogress.fastprogress import progress_bar
from fastai.basic_train import LearnerCallback
from ._retinanet_utils import compute_ap, _get_y, IoU_values, tlbr2cthw, cthw2tlbr
from fastai.basic_train import Callback
from fastai.torch_core import add_metrics


def add_conv(in_ch, out_ch, ksize=3, stride=1):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    stage.add_module(
        "conv",
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=ksize,
            stride=stride,
            padding=ksize // 2,
            bias=False,
        ),
    )
    stage.add_module("batch_norm", nn.BatchNorm2d(out_ch))
    stage.add_module("leaky", nn.LeakyReLU(negative_slope=0.1, inplace=True))
    return stage


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch // 2, 1, 1))
            resblock_one.append(add_conv(ch // 2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


def create_yolov3_modules(config_model, ignore_thre):
    """
    Build yolov3 layer modules.
    Args:
        config_model (dict): model configuration.
            See YOLOLayer class for details.
        ignore_thre (float): used in YOLOLayer.
    Returns:
        mlist (ModuleList): YOLOv3 module list.
    """
    # Fetch the number of bands (or channels) in the image
    in_ch = config_model["N_BANDS"]

    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=in_ch, out_ch=32, ksize=3, stride=1))  # 0
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))  # 1
    mlist.append(resblock(ch=64))  # 2
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))  # 3
    mlist.append(resblock(ch=128, nblocks=2))  # 4
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))  # 5
    mlist.append(
        resblock(ch=256, nblocks=8)
    )  # shortcut 1 from here             #6 - shortcut 1
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))  # 7
    mlist.append(
        resblock(ch=512, nblocks=8)
    )  # shortcut 2 from here             #8 - shortcut 2
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))  # 9
    mlist.append(resblock(ch=1024, nblocks=4))  # 10

    # YOLOv3
    mlist.append(resblock(ch=1024, nblocks=2, shortcut=False))  # 11
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))  # 12
    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))  # 13
    mlist.append(
        YOLOLayer(config_model, layer_no=0, in_ch=1024, ignore_thre=ignore_thre)
    )  # 14 - yolo

    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 15
    mlist.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 16 - shortcut 2 concats
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))  # 17
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 18
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))  # 19
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 20
    # 2nd yolo branch
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 21
    mlist.append(
        YOLOLayer(config_model, layer_no=1, in_ch=512, ignore_thre=ignore_thre)
    )  # 22 - yolo

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))  # 23
    mlist.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 24 - shortcut 1 concats
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))  # 25
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))  # 26
    mlist.append(resblock(ch=256, nblocks=2, shortcut=False))  # 27
    mlist.append(
        YOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre=ignore_thre)
    )  # 28 - yolo

    return mlist


class YOLOv3_Model(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """

    def __init__(self, config_model, ignore_thre=0.7):
        """
        Initialization of YOLOv3 class.
        Args:
            config_model (dict): used in YOLOLayer.
            ignore_thre (float): used in YOLOLayer.
        """
        super().__init__()
        self.module_list = create_yolov3_modules(config_model, ignore_thre)

    def forward(self, x, targets=torch.empty((0,), dtype=torch.float32)):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        if not torch.jit.is_scripting():
            dummy = torch.empty((0,), dtype=torch.float32).to(targets.device)
            if torch.equal(targets, dummy):
                targets = None
            train = targets is not None
        else:
            train = False
        output = []
        output_train = []
        if not torch.jit.is_scripting():
            self.loss_dict = defaultdict(float)
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            if i in [14, 22, 28]:
                if not torch.jit.is_scripting():
                    if train:
                        x, y, *loss_dict = module(x, targets)
                        for name, loss in zip(
                            ["xy", "wh", "conf", "cls", "l2"], loss_dict
                        ):
                            self.loss_dict[name] += loss
                        output_train.append(y)
                    else:
                        x = module(x)
                else:
                    x = module(x)
                output.append(x)
            else:
                x = module(x)

            # route layers
            if i in [6, 8, 12, 20]:
                route_layers.append(x)
            if i == 14:
                x = route_layers[2]
            if i == 22:  # yolo 2nd
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)

        if train and not torch.jit.is_scripting():
            return torch.cat(output_train, 1), sum(output)
        else:
            return torch.cat(output, 1)


class YOLOLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """

    def __init__(self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8]  # fixed
        self.anchors = config_model["ANCHORS"]
        self.anch_mask = config_model["ANCH_MASK"][layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model["N_CLASSES"]
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(reduction="sum")
        self.stride = strides[layer_no]
        self.all_anchors_grid = [
            [w / self.stride, h / self.stride] for w, h in self.anchors
        ]
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
        self.ref_anchors = torch.zeros((len(self.all_anchors_grid), 4))  # TODO
        self.ref_anchors[:, 2:] = torch.FloatTensor(self.all_anchors_grid)  # TODO
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=self.n_anchors * (self.n_classes + 5),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def _apply_sigmoid(self, output, n_ch: int):
        indices = [0, 1]
        for i in range(4, n_ch):
            indices.append(i)  # TODO:torch

        # logistic activation for xy, obj, cls #TODO
        output[:, :, :, :, indices] = torch.sigmoid(output[:, :, :, :, indices])

        return output

    def forward(self, xin, labels=torch.empty((0,), dtype=torch.float32)):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N: batchsize and
                K: number of labels (max labels in an image (padded)).
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        if not torch.jit.is_scripting():
            dummy = torch.empty((0,), dtype=torch.float32).to(labels.device)
            if torch.equal(labels, dummy):
                labels = None
        output = self.conv(xin)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        if not torch.jit.is_scripting():
            dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)

        # logistic activation for xy, obj, cls
        if not torch.jit.is_scripting():
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
                output[..., np.r_[:2, 4:n_ch]]
            )
        else:
            output = self._apply_sigmoid(output, n_ch)

        if torch.jit.is_scripting():
            x_shift = (
                torch.broadcast_to(
                    torch.arange(fsize, dtype=torch.float32), output.shape[:4]
                )
                .detach()
                .clone()
                .float()
                .to(xin.device)
            )

            y_shift = (
                torch.broadcast_to(
                    torch.arange(fsize, dtype=torch.float32).reshape(fsize, 1),
                    output.shape[:4],
                )
                .detach()
                .clone()
                .float()
                .to(xin.device)
            )

            masked_anchors = torch.tensor(self.masked_anchors).clone().detach()  # TODO
            w_anchors = (
                torch.broadcast_to(
                    torch.reshape(masked_anchors[:, 0], (1, self.n_anchors, 1, 1)),
                    output.shape[:4],
                )
                .detach()
                .clone()
                .float()
                .to(xin.device)
            )
            h_anchors = (
                torch.broadcast_to(
                    torch.reshape(masked_anchors[:, 1], (1, self.n_anchors, 1, 1)),
                    output.shape[:4],
                )
                .detach()
                .clone()
                .float()
                .to(xin.device)
            )

            pred = output.clone().contiguous()
            pred[..., 0] += x_shift
            pred[..., 1] += y_shift
            pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
            pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors
            pred[..., :4] *= self.stride  # Scale bbox coordinates to image size
            return pred.view(batchsize, -1, n_ch).data  # TODO: here
        else:
            # calculate pred - xywh obj cls
            x_shift = dtype(
                np.broadcast_to(np.arange(fsize, dtype=np.float32), output.shape[:4])
            )
            y_shift = dtype(
                np.broadcast_to(
                    np.arange(fsize, dtype=np.float32).reshape(fsize, 1),
                    output.shape[:4],
                )
            )

            masked_anchors = np.array(self.masked_anchors)

            w_anchors = dtype(
                np.broadcast_to(
                    np.reshape(masked_anchors[:, 0], (1, self.n_anchors, 1, 1)),
                    output.shape[:4],
                )
            )
            h_anchors = dtype(
                np.broadcast_to(
                    np.reshape(masked_anchors[:, 1], (1, self.n_anchors, 1, 1)),
                    output.shape[:4],
                )
            )

            pred = output.clone().contiguous()
            pred[..., 0] += x_shift
            pred[..., 1] += y_shift
            pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
            pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

            # return the predictions when not training
            if labels is None:
                pred[..., :4] *= self.stride  # Scale bbox coordinates to image size
                return pred.view(batchsize, -1, n_ch).data

            pred_train = pred.clone()
            pred_train[..., :4] *= self.stride

            pred = pred[..., :4].data

            # target assignment
            tgt_mask = torch.zeros(
                batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes
            ).type(dtype)
            obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).type(dtype)
            tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).type(
                dtype
            )

            target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).type(
                dtype
            )

            labels = labels.cpu().data
            # Rearrange the labels (dim=1) so that ground truths are ordered before paddings ([0,0,0,0,0])
            labels = labels.flip(1)
            # If there are no bboxes in the batch, create a zeros tensor with consistent shape
            if labels.nelement() == 0:
                labels = torch.zeros(batchsize, 1, 5)

            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

            # Convert to values normalized wrt grid cell size (or stride)
            truth_x_all = labels[:, :, 1] * fsize
            truth_y_all = labels[:, :, 2] * fsize
            truth_w_all = labels[:, :, 3] * fsize
            truth_h_all = labels[:, :, 4] * fsize

            # Find the grid loc of each object
            truth_i_all = truth_x_all.to(torch.int16).numpy()
            truth_j_all = truth_y_all.to(torch.int16).numpy()

            for b in range(batchsize):
                n = int(nlabel[b])  # number of objects in the image
                if n == 0:
                    continue

                truth_box = dtype(np.zeros((n, 4)))
                truth_box[:n, 2] = truth_w_all[b, :n]  # w
                truth_box[:n, 3] = truth_h_all[b, :n]  # h
                truth_i = truth_i_all[b, :n]  # i loc in grid
                truth_j = truth_j_all[b, :n]  # j loc in grid

                # calculate iou between truth and reference anchors
                anchor_ious_all = bboxes_iou(
                    truth_box.cpu(), self.ref_anchors
                )  # shape: (n, 9) - n: #bboxes in the image, 9: #anchors
                best_n_all = np.argmax(
                    anchor_ious_all, axis=1
                )  # tensor shape: (n) - indices of best matched anchor box (0-8) for each bbox
                best_n = (
                    best_n_all % 3
                )  # tensor shape: (n) - indices from the anchor mask, values in (0, 1, 2)
                # Only select the anchors if they are in the anch_mask for this layer, values in [(0,1,2), (3,4,5), (6,7,8)]
                best_n_mask = (
                    (best_n_all == self.anch_mask[0])
                    | (best_n_all == self.anch_mask[1])
                    | (best_n_all == self.anch_mask[2])
                )  # shape: (n) values in (0,1) - either matches one of the designated anchors or not

                truth_box[:n, 0] = truth_x_all[b, :n]
                truth_box[:n, 1] = truth_y_all[b, :n]

                # calculate iou between predictions and ground truth boxes
                pred_ious = bboxes_iou(
                    pred[b].view(-1, 4), truth_box, xyxy=False
                )  # shape: (ch*fsize*fsize, n)
                pred_best_iou, _ = pred_ious.max(
                    dim=1
                )  # Find the max ious values for each pred, shape: (ch*fsize*fsize)
                pred_best_iou = (
                    pred_best_iou > self.ignore_thre
                )  # Create a mask using the thresh
                pred_best_iou = pred_best_iou.view(
                    pred[b].shape[:3]
                )  # shape: (ch, fsize, fsize)
                # set mask to zero (ignore) if pred matches truth
                obj_mask[b] = 1 - pred_best_iou.int()  #

                # If none of the ground truth box matches the anchors for this layer, continue with next image
                if sum(best_n_mask) == 0:
                    continue

                # For every ground truth box
                for ti in range(best_n.shape[0]):
                    if (
                        best_n_mask[ti] == 1
                    ):  # if truth box matches one of the designated anchor
                        i, j = truth_i[ti], truth_j[ti]
                        a = best_n[ti]  # Index of matched anchor box, value in (0,1,2)
                        obj_mask[b, a, j, i] = 1
                        tgt_mask[b, a, j, i, :] = 1
                        target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[
                            b, ti
                        ].to(torch.int16).to(torch.float)
                        target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[
                            b, ti
                        ].to(torch.int16).to(torch.float)
                        target[b, a, j, i, 2] = torch.log(
                            truth_w_all[b, ti]
                            / torch.Tensor(self.masked_anchors)[best_n[ti], 0]
                            + 1e-16
                        )
                        target[b, a, j, i, 3] = torch.log(
                            truth_h_all[b, ti]
                            / torch.Tensor(self.masked_anchors)[best_n[ti], 1]
                            + 1e-16
                        )
                        target[b, a, j, i, 4] = 1
                        target[
                            b, a, j, i, 4 + labels[b, ti, 0].to(torch.int16).numpy()
                        ] = 1
                        tgt_scale[b, a, j, i, :] = torch.sqrt(
                            2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize
                        )

            # loss calculation

            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            bceloss = nn.BCELoss(
                weight=tgt_scale * tgt_scale, reduction="sum"
            )  # weighted BCEloss
            loss_xy = bceloss(output[..., :2], target[..., :2])
            loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
            loss_obj = self.bce_loss(output[..., 4], target[..., 4])
            loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
            loss_l2 = self.l2_loss(output, target)

            loss = (loss_xy + loss_wh + loss_obj + loss_cls).to(torch.float)

            return (
                loss,
                pred_train.view(batchsize, -1, n_ch).data,
                loss_xy,
                loss_wh,
                loss_obj,
                loss_cls,
                loss_l2,
            )


class YOLOv3_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, bbox_tgts, clas_tgts):
        # YOLOv3 model itself outputs loss when training
        if isinstance(output, tuple):
            return output[1]
        return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        # bottom right
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def nms_jit(bbox, thresh: float, score: Tensor):
    if torch.numel(bbox) == 0:
        return torch.zeros((0,), dtype=torch.int32)

    order = torch.argsort(score, descending=True)
    bbox = bbox[order]
    bbox_area = torch.prod(bbox[:, 2:] - bbox[:, :2], dim=1)

    selec = torch.zeros(bbox.shape[0], dtype=torch.bool)
    for i, b in enumerate(bbox):
        tl = torch.maximum(b[:2], bbox[selec, :2])
        br = torch.minimum(b[2:], bbox[selec, 2:])
        # print(tl, br, "\n")
        area = (
            (torch.prod(br - tl, dim=1) * torch.all((tl < br), dim=1))
            .clone()
            .detach()
            .to(dtype=torch.float64)
        )
        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if torch.any(iou >= thresh):
            continue

        selec[i] = True

    selec = torch.where(selec)[0]
    selec = order[selec]
    return selec.long()


def nms(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    from: https://github.com/chainer/chainercv
    """

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            area = np.array(
                np.prod(br - tl, axis=1) * (tl < br).all(axis=1), dtype=np.float64
            )
            iou = area / (bbox_area[i] + bbox_area[selec] - area)
            if (iou >= thresh).any():
                continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def postprocess(
    prediction, chip_size: int, conf_thre: float = 0.7, nms_thre: float = 0.45
):
    """
    Postprocess the output of YOLO model,
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4+1+k)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h, obj_score, cls_scores`
            where `xc` and `yc` represent a center of a bounding box.
        chip_size (int):
            size of the images.
        conf_thre (float):
            confidence threshold ranging from 0 to 1, default 0.7
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1. default 0.45

    Returns:
        output (list of torch tensor):

    """
    # Adding a dimension to handle a single image prediction
    if len(prediction.shape) == 2:
        prediction.unsqueeze_(0)

    if not torch.jit.is_scripting():
        box_corner = prediction.new(prediction.shape)
    else:
        box_corner = prediction.clone().detach()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    if torch.jit.is_scripting():
        dummy = torch.empty((len(prediction), 0, 0, 0)).float()
        output = [dummy for _ in range(len(prediction))]
    else:
        output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        class_pred = torch.max(image_pred[:, 5:], 1)
        class_pred = class_pred[0]  # Taking the class prediction scores
        conf_mask = (image_pred[:, 4] * class_pred >= conf_thre).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get detections with higher confidence scores than the threshold
        ind = (image_pred[:, 5:] * image_pred[:, 4][:, None] >= conf_thre).nonzero()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (
                image_pred[ind[:, 0], :5],
                image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1),
                ind[:, 1].float().unsqueeze(1),
            ),
            1,
        )

        # Iterate through all predicted classes
        if not torch.jit.is_scripting():
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
        else:
            unique_labels = torch.unique(detections[:, detections.shape[1] - 1])
            unique_labels = unique_labels.to(prediction.device)
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[
                detections[:, -1] == c
            ]  # TODO: remove negative index
            if not torch.jit.is_scripting():
                nms_in = detections_class.cpu().numpy()
                nms_out_index = nms(
                    nms_in[:, :4], thresh=nms_thre, score=nms_in[:, 4] * nms_in[:, 5]
                )
            else:
                nms_in = detections_class.detach().clone()
                nms_out_index = nms_jit(
                    nms_in[:, :4], thresh=nms_thre, score=nms_in[:, 4] * nms_in[:, 5]
                )

            detections_class = detections_class[nms_out_index]
            if torch.jit.is_scripting():
                if output[i] is dummy:
                    output[i] = detections_class
                else:
                    output[i] = torch.cat((output[i], detections_class))
            else:
                if output[i] is None:
                    output[i] = detections_class
                else:
                    output[i] = torch.cat((output[i], detections_class))

    if torch.jit.is_scripting():
        if output[0] is dummy:
            return dummy, dummy, dummy  # when there is no detection
    else:
        if output[0] is None:
            return None  # when there is no detection

    bbox_scaled = torch.clamp(
        output[0][:, :4] / (chip_size - 1) * 2 - 1, min=-1, max=1
    )  # Scale: -1 to 1
    bbox_pred = torch.index_select(
        bbox_scaled, dim=1, index=torch.tensor([1, 0, 3, 2]).to(bbox_scaled.device)
    )  # bbox ordered as y1,x1,y2,x2
    preds = output[0][:, -1].to(torch.long) + 1  # Add 1 to account for background class
    scores = output[0][:, 4] * output[0][:, 5]

    return bbox_pred, preds, scores


def transform_targets(last_target):
    """
    Args:
        last_target: List[Tensor(B,N,4), Tensor(B,N)]
    Returns:
        targets (torch.tensor): shape (B, N, 5)
    """
    B, N, _ = last_target[0].shape

    lt = torch.cat(
        (
            last_target[1].unsqueeze(2).type(torch.float),
            last_target[0].type(torch.float),
        ),
        dim=2,
    )
    lt = torch.reshape(lt, (-1, 5))
    tmask = lt[:, 0] != 0.0  # Mask to use to transform only non-zero labels
    target = torch.zeros_like(lt)

    lt[tmask, 1:] = (lt[tmask, 1:] + 1) / 2  # Scale between 0 and 1 from -1 to 1
    target[:, 0] = lt[:, 0]
    target[:, 1] = (lt[:, 2] + lt[:, 4]) / 2
    target[:, 2] = (lt[:, 1] + lt[:, 3]) / 2
    target[:, 3] = lt[:, 4] - lt[:, 2]
    target[:, 4] = lt[:, 3] - lt[:, 1]

    target = torch.reshape(target, (B, N, 5))
    return target


class AppendLabelsCallback(LearnerCallback):
    """
    This callback is used to append labels with input
    to be passed to the YOLOv3 while training.
    """

    def __init__(self, learn):
        super().__init__(learn)

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Returns xb (images,labels), yb (labels) when training"
        if self.learn.predicting:
            self.learn.predicting = False
            return {"last_input": last_input, "last_target": last_target}
        else:
            # On training set xb as (inputs, targets) because YOLOv3 needs both
            self.learn.model.train()
            targets = transform_targets(last_target)
            return {"last_input": (last_input, targets), "last_target": last_target}


from ._ssd_utils import kmeans, avg_iou


def generate_anchors(num_anchor, hw, limit=1000):
    """
    Function to generate anchors using k-means
    Args:
        num_anchors (int) - number of anchors to generate
        hw - List of height width of all bounding boxes in the dataset.
        limit - max num of bounding boxes to consider for k-means
    Returns:
        a list of anchors, shape(num_anchors, 2)
    """

    if limit > len(hw):
        limit = len(hw)
    import random

    idx = random.sample(range(len(hw)), limit)
    hw = np.array(hw)[idx]
    new_centroid = kmeans(hw, num_anchor)
    anchors = (np.ceil(new_centroid)).astype(int)
    anchors = anchors.tolist()
    anchors.sort(key=lambda x: x[0] * x[1])
    return update_size_zero_anchors(anchors)


def update_size_zero_anchors(anchors):
    """
    Function to replace zero size anchors [0,0]
    by incrementally (+1) bigger anchors than the largest one.
    Args:
        anchors, shape(N, 2), where N is number of anchors
    Returns:
        an updated list of anchors.
    """

    zero_anchors = sum([anchor == [0, 0] for anchor in anchors])
    largest_anchor = anchors[-1]

    for i in range(zero_anchors):
        largest_anchor = [j + 1 for j in largest_anchor]
        anchors.append(largest_anchor)

    return anchors[zero_anchors:]


class AveragePrecision(Callback):
    def __init__(self, model, n_classes):
        self.model = model
        self.n_classes = n_classes

    def on_epoch_begin(self, **kwargs):
        self.tps, self.clas, self.p_scores = [], [], []
        self.classes, self.n_gts = (
            LongTensor(range(self.n_classes)),
            torch.zeros(self.n_classes).long(),
        )

    def on_batch_end(self, last_output, last_target, **kwargs):
        tps, p_scores, clas, self.n_gts = compute_cm(
            self.model, last_output[0], last_target, self.n_gts, self.classes
        )
        self.tps.extend(tps)
        self.p_scores.extend(p_scores)
        self.clas.extend(clas)

    def on_epoch_end(self, last_metrics, **kwargs):
        aps = compute_ap_score(
            self.tps, self.p_scores, self.clas, self.n_gts, self.n_classes
        )
        aps = torch.mean(torch.tensor(aps))
        return add_metrics(last_metrics, aps)


def compute_class_AP(
    model, dl, n_classes, show_progress, iou_thresh=0.1, detect_thresh=0.1, num_keep=100
):
    tps, clas, p_scores = [], [], []
    classes, n_gts = LongTensor(range(n_classes)), torch.zeros(n_classes).long()
    model.learn.model.eval()

    with torch.no_grad():
        for input, target in progress_bar(dl, display=show_progress):
            # input, shape: B, C, H, W
            model.learn.predicting = True
            output = model.learn.pred_batch(batch=(input, target))

            tps1, p_scores1, clas1, n_gts = compute_cm(
                model, output, target, n_gts, classes, iou_thresh, detect_thresh
            )
            tps.extend(tps1)
            p_scores.extend(p_scores1)
            clas.extend(clas1)

    aps = compute_ap_score(tps, p_scores, clas, n_gts, n_classes)
    return aps


def compute_cm(
    model, output, target, n_gts, classes, iou_thresh=0.1, detect_thresh=0.1
):
    tps, clas, p_scores = [], [], []
    for i in range(target[0].size(0)):  # range batch-size
        # op - bbox preds, class preds, scores
        op = model._data.y.analyze_pred(
            output[i],
            model=model,
            thresh=detect_thresh,
            nms_overlap=iou_thresh,
            ret_scores=True,
            device=model._device,
        )

        tgt_bbox, tgt_clas = _get_y(target[0][i], target[1][i])

        try:
            bbox_pred, preds, scores = op
            if len(bbox_pred) != 0 and len(tgt_bbox) != 0:
                bbox_pred = bbox_pred.to(model._device)
                preds = preds.to(model._device)
                tgt_bbox = tgt_bbox.to(model._device)

                # Convert the bbox coordinates to center-height-width(cthw) before calculating Intersection Over Union
                ious = IoU_values(tlbr2cthw(bbox_pred), tlbr2cthw(tgt_bbox))
                max_iou, matches = ious.max(1)
                detected = []

                for i in range(len(preds)):
                    if (
                        max_iou[i] >= iou_thresh
                        and matches[i] not in detected
                        and tgt_clas[matches[i]] == preds[i]
                    ):
                        detected.append(matches[i])
                        tps.append(1)
                    else:
                        tps.append(0)
                clas.append(preds.cpu())
                p_scores.append(scores.cpu())
        except:
            pass
        n_gts += ((tgt_clas.cpu()[:, None] - 1) == classes[None, :]).sum(0)

    return tps, p_scores, clas, n_gts


def compute_ap_score(tps, p_scores, clas, n_gts, n_classes):
    # If no true positives are found return an average precision score of 0.
    if len(tps) == 0:
        return [0.0 for cls in range(1, n_classes + 1)]

    tps, p_scores, clas = torch.tensor(tps), torch.cat(p_scores, 0), torch.cat(clas, 0)
    fps = 1 - tps
    idx = p_scores.argsort(descending=True)
    tps, fps, clas = tps[idx], fps[idx], clas[idx]
    aps = []

    for cls in range(1, n_classes + 1):
        tps_cls, fps_cls = (
            tps[clas == cls].float().cumsum(0),
            fps[clas == cls].float().cumsum(0),
        )
        if tps_cls.numel() != 0 and tps_cls[-1] != 0:
            precision = tps_cls / (tps_cls + fps_cls + 1e-8)
            recall = tps_cls / (n_gts[cls - 1] + 1e-8)
            aps.append(compute_ap(precision, recall))
        else:
            aps.append(0.0)
    return aps


def parse_conv_block(m, weights, offset, initflag):
    """
    Initialization of conv layers with batchnorm
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    conv_model = m[0]
    bn_model = m[1]
    param_length = m[1].bias.numel()

    # batchnorm
    for pname in ["bias", "weight", "running_mean", "running_var"]:
        layerparam = getattr(bn_model, pname)

        if initflag:  # yolo initialization - scale to one, bias to zero
            if pname == "weight":
                weights = np.append(weights, np.ones(param_length))
            else:
                weights = np.append(weights, np.zeros(param_length))

        param = torch.from_numpy(weights[offset : offset + param_length]).view_as(
            layerparam
        )
        layerparam.data.copy_(param)
        offset += param_length

    param_length = conv_model.weight.numel()

    # conv
    if initflag:  # yolo initialization
        n, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))

    param = torch.from_numpy(weights[offset : offset + param_length]).view_as(
        conv_model.weight
    )
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights


def parse_yolo_block(m, weights, offset, initflag):
    """
    YOLO Layer (one conv with bias) Initialization
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    conv_model = m._modules["conv"]
    param_length = conv_model.bias.numel()

    if initflag:  # yolo initialization - bias to zero
        weights = np.append(weights, np.zeros(param_length))

    param = torch.from_numpy(weights[offset : offset + param_length]).view_as(
        conv_model.bias
    )
    conv_model.bias.data.copy_(param)
    offset += param_length

    param_length = conv_model.weight.numel()

    if initflag:  # yolo initialization
        n, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))

    param = torch.from_numpy(weights[offset : offset + param_length]).view_as(
        conv_model.weight
    )
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights


def parse_yolo_weights(model, weights_path):
    """
    Parse YOLO (darknet) pre-trained weights data onto the pytorch model
    Args:
        model : pytorch model object
        weights_path (str): path to the YOLO (darknet) pre-trained weights file
    """
    fp = open(weights_path, "rb")

    # skip the header
    header = np.fromfile(fp, dtype=np.int32, count=5)  # not used
    # read weights
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    offset = 0
    initflag = False  # whole yolo weights : False, darknet weights : True

    for m in model.module_list:
        if m._get_name() == "Sequential":
            # normal conv block
            offset, weights = parse_conv_block(m, weights, offset, initflag)

        elif m._get_name() == "resblock":
            # residual block
            for modu in m._modules["module_list"]:
                for blk in modu:
                    offset, weights = parse_conv_block(blk, weights, offset, initflag)

        elif m._get_name() == "YOLOLayer":
            # YOLO Layer (one conv with bias) Initialization
            offset, weights = parse_yolo_block(m, weights, offset, initflag)

        initflag = offset >= len(
            weights
        )  # the end of the weights file. turn the flag on


def download_yolo_weights(weights_path):
    """Download COCO pretrained weights for YOLOv3."""
    from arcgis.gis import GIS

    gis = GIS(set_active=False)
    item = gis.content.get("8b4600eb9a29407bbfe51491ad5bf62c")
    print(f"[INFO] Downloading COCO pretrained weights for YOLOv3 in {weights_path}...")
    filepath = item.download(weights_path)
    return


def coco_config():
    """Function to return YOLOv3 model configurations for COCO dataset."""
    config_model = {}
    config_model["ANCHORS"] = [
        [10, 13],
        [16, 30],
        [33, 23],
        [30, 61],
        [62, 45],
        [59, 119],
        [116, 90],
        [156, 198],
        [373, 326],
    ]
    config_model["ANCH_MASK"] = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    config_model["N_CLASSES"] = 80
    config_model["N_BANDS"] = 3
    return config_model


def coco_class_mapping():
    """Create class mapping for COCO dataset."""

    # 80 COCO class indices on which YOLOv3 is pretrained
    coco_class_ids = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

    # 90 classes of COCO dataset
    coco_label_names = (
        "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "street sign",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "hat",
        "backpack",
        "umbrella",
        "shoe",
        "eye glasses",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "plate",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "mirror",
        "dining table",
        "window",
        "desk",
        "toilet",
        "door",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "blender",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )

    coco_class_mapping = {k: v for k, v in enumerate(coco_label_names)}
    class_mapping = {k: v for k, v in coco_class_mapping.items() if k in coco_class_ids}

    return class_mapping
