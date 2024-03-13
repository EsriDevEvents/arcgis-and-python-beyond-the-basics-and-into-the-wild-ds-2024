# based on https://github.com/foolwood/SiamMask - MIT license
try:
    from fastai.basic_train import LearnerCallback
    from fastai.basic_train import Learner
    import torch
    import torch.nn as nn
    import traceback
    from torch.autograd import Variable
    import math
    import torch.utils.model_zoo as model_zoo
    from collections import namedtuple
    import torch.nn.functional as F
    import numpy as np
    import cv2
    import copy

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class SiamMask(nn.Module):
    def __init__(self, anchors=None, o_sz=127, g_sz=127):
        super().__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])

        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(
        self,
        label_cls,
        label_loc,
        lable_loc_weight,
        label_mask,
        label_mask_weight,
        rpn_pred_cls,
        rpn_pred_loc,
        rpn_pred_mask,
    ):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)

        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(
            rpn_pred_mask, label_mask, label_mask_weight
        )

        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        # print(template_feature)
        feature, search_feature = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        corr_feature = self.mask_model.mask.forward_corr(
            template_feature, search_feature
        )
        rpn_pred_mask = self.refine_model(
            feature[0],
            feature[1],
            feature[2],
            corr_feature,
            pos=None,
            test=torch.empty([1]),
        )

        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return (
            rpn_pred_cls,
            rpn_pred_loc,
            rpn_pred_mask,
            template_feature,
            search_feature,
        )

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input["template"]
        search = input["search"]
        if self.training:
            label_cls = input["label_cls"]
            label_loc = input["label_loc"]
            lable_loc_weight = input["label_loc_weight"]
            label_mask = input["label_mask"]
            label_mask_weight = input["label_mask_weight"]

        (
            rpn_pred_cls,
            rpn_pred_loc,
            rpn_pred_mask,
            template_feature,
            search_feature,
        ) = self.run(template, search, softmax=self.training)

        # return 0
        outputs = dict()

        outputs["predict"] = [
            rpn_pred_loc,
            rpn_pred_cls,
            rpn_pred_mask,
            template_feature,
            search_feature,
        ]
        if self.training:
            (
                rpn_loss_cls,
                rpn_loss_loc,
                rpn_loss_mask,
                iou_acc_mean,
                iou_acc_5,
                iou_acc_7,
            ) = self._add_rpn_loss(
                label_cls,
                label_loc,
                lable_loc_weight,
                label_mask,
                label_mask_weight,
                rpn_pred_cls,
                rpn_pred_loc,
                rpn_pred_mask,
            )
            outputs["losses"] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
            outputs["accuracy"] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc


def get_cls_loss(pred, label, select):
    if select.nelement() == 0:
        return pred.sum() * 0.0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)

    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight:  [b, k, h, w]
    :return: loc loss value
    """
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0:
        return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

    if len(p_m.shape) == 4:
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
        p_m = p_m.view(-1, g_sz * g_sz)
    else:
        p_m = torch.index_select(p_m, 0, pos)

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=0, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)

    mask_uf = torch.index_select(mask_uf, 0, pos)
    loss = F.soft_margin_loss(p_m, mask_uf)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label):
    pred = pred.ge(0).int()
    mask_sum = (pred.eq(1).int()).add(label.eq(1).int())
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn / union

    return (
        torch.mean(iou),
        (torch.sum(iou > 0.5).float() / iou.shape[0]),
        (torch.sum(iou > 0.7).float() / iou.shape[0]),
    )


__all__ = ["ResNet", "resnet50"]

model_urls = {
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        self.feature_size = -1

    def forward(self, x):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{"params": params, "lr": start_lr * feature_mult}]
        return params

    def load_model(self, f="pretrain.model"):
        with open(f) as f:
            pretrained_dict = torch.load(f)
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class MultiStageFeature(Features):
    def __init__(self):
        super(MultiStageFeature, self).__init__()

        self.layers = []
        self.train_num = -1
        self.change_point = []
        self.train_nums = []

    def unfix(self, ratio=0.0):
        if self.train_num == -1:
            self.train_num = 0
            self.unlock()
            self.eval()
        for p, t in reversed(list(zip(self.change_point, self.train_nums))):
            if ratio >= p:
                if self.train_num != t:
                    self.train_num = t
                    self.unlock()
                    return True
                break
        return False

    def train_layers(self):
        return self.layers[: self.train_num]

    def unlock(self):
        for p in self.parameters():
            p.requires_grad = False

        for m in self.train_layers():
            for p in m.parameters():
                p.requires_grad = True

    def train(self, mode):
        self.training = mode
        if mode == False:
            super(MultiStageFeature, self).train(False)
        else:
            for m in self.train_layers():
                m.train(True)

        return self


class Bottleneck(Features):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # padding = (2 - stride) + (dilation // 2 - 1)
        padding = 2 - stride
        assert (
            stride == 1 or dilation == 1
        ), "stride and dilation must have one equals to zero at least"
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual

        out = self.relu(out)

        return out


class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        s = residual.size(3)
        residual = residual[:, :, 1 : s - 1, 1 : s - 1]

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=0, bias=False  # 3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 31x31, 15x15

        self.feature_size = 128 * block.expansion

        if layer3:
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=1, dilation=2
            )  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1, dilation=4
            )  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=3,
                        stride=stride,
                        bias=False,
                        padding=padding,
                        dilation=dd,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dd))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)

        return p0, p1, p2, p3


class ResAdjust(nn.Module):
    def __init__(
        self, block=Bottleneck, out_channels=256, adjust_number=1, fuse_layers=[2, 3, 4]
    ):
        super(ResAdjust, self).__init__()
        self.fuse_layers = set(fuse_layers)

        if 2 in self.fuse_layers:
            self.layer2 = self._make_layer(block, 128, 1, out_channels, adjust_number)
        if 3 in self.fuse_layers:
            self.layer3 = self._make_layer(block, 256, 2, out_channels, adjust_number)
        if 4 in self.fuse_layers:
            self.layer4 = self._make_layer(block, 512, 4, out_channels, adjust_number)

        self.feature_size = out_channels * len(self.fuse_layers)

    def _make_layer(self, block, plances, dilation, out, number=1):
        layers = []

        for _ in range(number):
            layer = block(plances * block.expansion, plances, dilation=dilation)
            layers.append(layer)

        downsample = nn.Sequential(
            nn.Conv2d(
                plances * block.expansion, out, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out),
        )
        layers.append(downsample)

        return nn.Sequential(*layers)

    def forward(self, p2, p3, p4):
        outputs = []

        if 2 in self.fuse_layers:
            outputs.append(self.layer2(p2))
        if 3 in self.fuse_layers:
            outputs.append(self.layer3(p3))
        if 4 in self.fuse_layers:
            outputs.append(self.layer4(p4))
        # return torch.cat(outputs, 1)
        return outputs


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
        except:
            pass
    return model


Corner = namedtuple("Corner", "x1 y1 x2 y2")
BBox = Corner
Center = namedtuple("Center", "x y w h")


def corner2center(corner):
    """
    :param corner: Corner or np.array 4*N
    :return: Center or 4 np.array N
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """
    :param center: Center or np.array 4*N
    :return: Corner or np.array 4*N
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])  # 0-index


def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(
            region[2:4] - region[4:6]
        )
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2

    return cx, cy, w, h


def aug_apply(bbox, param, shape, inv=False, rd=False):
    """
    apply augmentation
    :param bbox: original bbox in image
    :param param: augmentation param, shift/scale
    :param shape: image shape, h, w, (c)
    :param inv: inverse
    :param rd: round bbox
    :return: bbox(, param)
        bbox: augmented bbox
        param: real augmentation param
    """
    if not inv:
        center = corner2center(bbox)
        original_center = center

        real_param = {}
        if "scale" in param:
            scale_x, scale_y = param["scale"]
            imh, imw = shape[:2]
            h, w = center.h, center.w

            scale_x = min(scale_x, float(imw) / w)
            scale_y = min(scale_y, float(imh) / h)

            # center.w *= scale_x
            # center.h *= scale_y
            center = Center(center.x, center.y, center.w * scale_x, center.h * scale_y)

        bbox = center2corner(center)

        if "shift" in param:
            tx, ty = param["shift"]
            x1, y1, x2, y2 = bbox
            imh, imw = shape[:2]

            tx = max(-x1, min(imw - 1 - x2, tx))
            ty = max(-y1, min(imh - 1 - y2, ty))

            bbox = Corner(x1 + tx, y1 + ty, x2 + tx, y2 + ty)

        if rd:
            bbox = Corner(*map(round, bbox))

        current_center = corner2center(bbox)

        real_param["scale"] = (
            current_center.w / original_center.w,
            current_center.h / original_center.h,
        )
        real_param["shift"] = (
            current_center.x - original_center.x,
            current_center.y - original_center.y,
        )

        return bbox, real_param
    else:
        if "scale" in param:
            scale_x, scale_y = param["scale"]
        else:
            scale_x, scale_y = 1.0, 1.0

        if "shift" in param:
            tx, ty = param["shift"]
        else:
            tx, ty = 0, 0

        center = corner2center(bbox)

        center = Center(
            center.x - tx, center.y - ty, center.w / scale_x, center.h / scale_y
        )

        return center2corner(center)


def IoU(rect1, rect2):
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2 - x1) * (y2 - y1)

    target_a = (tx2 - tx1) * (ty2 - ty1)

    inter = ww * hh
    overlap = inter / (area + target_a - inter)

    return overlap


class Anchors:
    def __init__(self, cfg):
        self.stride = 8
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.scales = [8]
        self.round_dight = 0
        self.image_center = 0
        self.size = 0
        self.anchor_density = 1

        self.__dict__.update(cfg)

        self.anchor_num = (
            len(self.scales) * len(self.ratios) * (self.anchor_density**2)
        )
        self.anchors = None  # in single position (anchor_num*4)
        self.all_anchors = None  # in all position 2*(4*anchor_num*h*w)
        self.generate_anchors()

    def generate_anchors(self):
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)

        size = self.stride * self.stride
        count = 0
        anchors_offset = self.stride / self.anchor_density
        anchors_offset = np.arange(self.anchor_density) * anchors_offset
        anchors_offset = anchors_offset - np.mean(anchors_offset)
        x_offsets, y_offsets = np.meshgrid(anchors_offset, anchors_offset)

        for x_offset, y_offset in zip(x_offsets.flatten(), y_offsets.flatten()):
            for r in self.ratios:
                if self.round_dight > 0:
                    ws = round(math.sqrt(size * 1.0 / r), self.round_dight)
                    hs = round(ws * r, self.round_dight)
                else:
                    ws = int(math.sqrt(size * 1.0 / r))
                    hs = int(ws * r)

                for s in self.scales:
                    w = ws * s
                    h = hs * s
                    self.anchors[count][:] = [
                        -w * 0.5 + x_offset,
                        -h * 0.5 + y_offset,
                        w * 0.5 + x_offset,
                        h * 0.5 + y_offset,
                    ][:]
                    count += 1

    def generate_all_anchors(self, im_c, size):
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(
            lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2]
        )
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])
        return True


class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{"params": params, "lr": start_lr * feature_mult}]
        return params


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x: x.requires_grad, self.parameters())
        else:
            params = [
                v for k, v in self.named_parameters() if (key in k) and v.requires_grad
            ]
        params = [{"params": params, "lr": start_lr * feature_mult}]
        return params


class DepthwiseConv2Group(nn.Module):
    def __init__(self):
        super(DepthwiseConv2Group, self).__init__()

    def forward(self, input, kernel):
        batch, channel = kernel.shape[:2]
        input = input.view(
            1, batch * channel, input.size(2), input.size(3)
        )  # 1 * (b*c) * k * k
        kernel = kernel.view(
            batch * channel, 1, kernel.size(2), kernel.size(3)
        )  # (b*c) * 1 * H * W
        feature = F.conv2d(input, kernel, groups=batch * channel)
        feature = feature.view(batch, channel, feature.size(2), feature.size(3))
        return feature


class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

        self.dw_conv2d_group = DepthwiseConv2Group()

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        return self.dw_conv2d_group(input, kernel)

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out

    def conv2d_dw_group(x, kernel):
        batch, channel = kernel.shape[:2]
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
            nn.BatchNorm2d(outplane),
        )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False, backbone="resnet50", pretrained_path=""):
        super(ResDown, self).__init__()
        if backbone == "resnet50":
            self.features = resnet50(layer3=True, layer4=False)

        if pretrain:
            load_pretrain(self.features, pretrained_path)

        self.downsample = ResDownS(1024, 256)

        self.layers = [self.downsample, self.features.layer2, self.features.layer3]
        self.train_nums = [1, 3]
        self.change_point = [0, 0.5]

        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x: x.requires_grad, module.parameters()))
            if len(params):
                return [{"params": params, "lr": lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return p3

    def forward_all(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return output, p3


class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(Mask):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
        return self.mask(z, x)


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.v1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.v2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.h2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.h1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.h0 = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

        for modules in [
            self.v0,
            self.v1,
            self.v2,
            self.h2,
            self.h1,
            self.h0,
            self.deconv,
            self.post0,
            self.post1,
            self.post2,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(
        self, f0, f1, f2, corr_feature, pos=torch.empty(0), test=torch.empty(0)
    ):
        if torch.numel(test) == 0:
            p0 = torch.nn.functional.pad(f0, [16, 16, 16, 16])[
                :, :, 4 * pos[0] : 4 * pos[0] + 61, 4 * pos[1] : 4 * pos[1] + 61
            ]
            p1 = torch.nn.functional.pad(f1, [8, 8, 8, 8])[
                :, :, 2 * pos[0] : 2 * pos[0] + 31, 2 * pos[1] : 2 * pos[1] + 31
            ]
            p2 = torch.nn.functional.pad(f2, [4, 4, 4, 4])[
                :, :, pos[0] : pos[0] + 15, pos[1] : pos[1] + 15
            ]
        else:
            p0 = (
                F.unfold(f0, (61, 61), padding=0, stride=4)
                .permute(0, 2, 1)
                .contiguous()
                .view(-1, 64, 61, 61)
            )
            if not (pos is None):
                p0 = torch.index_select(p0, 0, pos[0])
            p1 = (
                F.unfold(f1, (31, 31), padding=0, stride=2)
                .permute(0, 2, 1)
                .contiguous()
                .view(-1, 256, 31, 31)
            )
            if not (pos is None):
                p1 = torch.index_select(p1, 0, pos[0])
            p2 = (
                F.unfold(f2, (15, 15), padding=0, stride=1)
                .permute(0, 2, 1)
                .contiguous()
                .view(-1, 512, 15, 15)
            )
            if not (pos is None):
                p2 = torch.index_select(p2, 0, pos[0])

        if not (pos is None):
            p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)
        else:
            p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127 * 127)
        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{"params": params, "lr": start_lr * feature_mult}]
        return params


class Custom(SiamMask):
    def __init__(
        self, pretrain=False, backbone="resnet50", pretrained_path="", **kwargs
    ):
        super().__init__(**kwargs)
        self.features = ResDown(
            pretrain=pretrain, backbone=backbone, pretrained_path=pretrained_path
        )
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()

    def refine(self, f, pos=None):
        print("Refine function")
        return self.refine_model(f[0], f[1], f[2], pos)

    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        self.feature, self.search = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, self.search)
        self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        pred_mask = self.mask_model.mask.head(self.corr_feature)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

    def track_refine(self, pos):
        pred_mask = self.refine_model(
            self.feature[0],
            self.feature[1],
            self.feature[2],
            self.corr_feature,
            pos=pos,
            test=torch.empty(0),
        )
        return pred_mask


def build_opt_lr(trainable_params):
    optimizer = torch.optim.SGD(
        trainable_params, 0.001, momentum=0.9, weight_decay=0.0001
    )

    return optimizer


def BNtoFixed(m):
    class_name = m.__class__.__name__
    if class_name.find("BatchNorm") != -1:
        m.eval()


def download_backbone(url, file_name):
    import os
    from pathlib import Path

    weights_path = os.path.join(Path.home(), ".cache", "siammask_checkpoint")
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    weights_file = os.path.join(weights_path, file_name)
    if not os.path.exists(weights_file):
        try:
            import urllib.request

            print(
                f"[INFO] Downloading pretrained weights for SiamMask in {weights_file}"
            )
            urllib.request.urlretrieve(url, weights_file)
        except Exception as e:
            print(e)
            print("[INFO] Can't download pretrained weights for SiamMask.")
    return weights_file


def get_learner(data=None, anchors=None):
    if data is not None:
        file_path = download_backbone(
            url="http://www.robots.ox.ac.uk/~qwang/resnet.model",
            file_name="resnet.model",
        )
        model = Custom(
            anchors=anchors,
            pretrain=True,
            backbone="resnet50",
            pretrained_path=file_path,
        )
        model = model.cuda()

        learn = Learner(
            data=data,
            model=model,
            loss_func=siammask_loss,
            opt_func=build_opt_lr,
            metrics=mIOU,
        )

    return learn


def mIOU(outputs, *args):
    return outputs["accuracy"][0]


def siammask_loss(outputs, *args):
    rpn_cls_loss, rpn_loc_loss, rpn_mask_loss = (
        torch.mean(outputs["losses"][0]),
        torch.mean(outputs["losses"][1]),
        torch.mean(outputs["losses"][2]),
    )

    cls_weight, reg_weight, mask_weight = [1.0, 1.2, 36]

    loss = (
        rpn_cls_loss * cls_weight
        + rpn_loc_loss * reg_weight
        + rpn_mask_loss * mask_weight
    )

    return loss


class train_callback(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Handle new batch `xb`,`yb` in `train` or validation."
        x = {
            "template": torch.autograd.Variable(last_input[0]).cuda(),
            "search": torch.autograd.Variable(last_input[1]).cuda(),
            "label_cls": torch.autograd.Variable(last_input[2]).cuda(),
            "label_loc": torch.autograd.Variable(last_input[3]).cuda(),
            "label_loc_weight": torch.autograd.Variable(last_input[4]).cuda(),
            "label_mask": torch.autograd.Variable(last_input[6]).cuda(),
            "label_mask_weight": torch.autograd.Variable(last_input[7]).cuda(),
        }
        self.learn.model.train()
        last_target = torch.rand((4, 4, 4)).cuda()
        return {"last_input": x, "last_target": last_target}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters share common prefix 'module.'"""
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    if not torch.cuda.is_available():
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")

    try:
        check_keys(model, pretrained_dict)
    except:
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = "features." + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))
    img = to_torch(img).float()
    return img


class TrackerConfig(object):
    penalty_k = 0.09
    window_influence = 0.39
    lr = 0.38
    seg_thr = 0.35
    windowing = "cosine"
    exemplar_size = 127
    instance_size = 255
    total_stride = 8
    out_size = 127
    base_size = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1 + base_size
    context_amount = 0.5
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [
        8,
    ]
    anchor_num = len(ratios) * len(scales)
    round_dight = 0
    anchor = []

    def update(self, newparam=None, anchors=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
        if anchors is not None:
            if isinstance(anchors, dict):
                anchors = Anchors(anchors)
            if isinstance(anchors, Anchors):
                self.total_stride = anchors.stride
                self.ratios = anchors.ratios
                self.scales = anchors.scales
                self.round_dight = anchors.round_dight
        self.renew()

    def renew(self):
        self.score_size = (
            (self.instance_size - self.exemplar_size) // self.total_stride
            + 1
            + self.base_size
        )
        self.anchor_num = len(self.ratios) * len(self.scales)


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode="torch"):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0.0, -context_xmin))
    top_pad = int(max(0.0, -context_ymin))
    right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros(
            (r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8
        )
        te_im[top_pad : top_pad + r, left_pad : left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad : left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad :, left_pad : left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad :, :] = avg_chans
        im_patch_original = te_im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]
    else:
        im_patch_original = im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    return im_to_torch(im_patch) if out_mode in "torch" else im_patch


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = -(score_size // 2) * total_stride
    xx, yy = np.meshgrid(
        [ori + total_stride * dx for dx in range(score_size)],
        [ori + total_stride * dy for dy in range(score_size)],
    )
    xx, yy = (
        np.tile(xx.flatten(), (anchor_num, 1)).flatten(),
        np.tile(yy.flatten(), (anchor_num, 1)).flatten(),
    )
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(im, target_pos, target_sz, model, hp=None, device="cpu"):
    state = dict()
    state["im_h"] = im.shape[0]
    state["im_w"] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)

    p.renew()

    net = copy.deepcopy(model)
    p.scales = model.anchors["scales"]
    p.ratios = model.anchors["ratios"]
    p.anchor_num = model.anchor_num
    p.anchor = generate_anchor(model.anchors, p.score_size)
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.template(z.to(device))

    if p.windowing == "cosine":
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == "uniform":
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state["p"] = p
    state["net"] = net
    state["avg_chans"] = avg_chans
    state["window"] = window
    state["target_pos"] = target_pos
    state["target_sz"] = target_sz
    return state


def siamese_track(
    state, im, mask_enable=False, refine_enable=False, device="cpu", debug=False
):
    p = state["p"]
    net = state["net"]
    avg_chans = state["avg_chans"]
    window = state["window"]
    target_pos = state["target_pos"]
    target_sz = state["target_sz"]

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [
        target_pos[0] - round(s_x) / 2,
        target_pos[1] - round(s_x) / 2,
        round(s_x),
        round(s_x),
    ]

    x_crop = Variable(
        get_subwindow_tracking(
            im, target_pos, p.instance_size, round(s_x), avg_chans
        ).unsqueeze(0)
    )

    if mask_enable:
        score, delta, mask = net.track_mask(x_crop.to(device))
    else:
        score, delta = net.track(x_crop.to(device))

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = (
        F.softmax(
            score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1
        )
        .data[:, 1]
        .cpu()
        .numpy()
    )

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1.0 / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    target_sz_in_crop = target_sz * scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))
    r_c = change(
        (target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :])
    )

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    if mask_enable:
        best_pscore_id_mask = np.unravel_index(
            best_pscore_id, (5, p.score_size, p.score_size)
        )
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        if refine_enable:
            mask = (
                net.track_refine((delta_y, delta_x))
                .to(device)
                .sigmoid()
                .squeeze()
                .view(p.out_size, p.out_size)
                .cpu()
                .data.numpy()
            )
        else:
            mask = (
                mask[0, :, delta_y, delta_x]
                .sigmoid()
                .squeeze()
                .view(p.out_size, p.out_size)
                .cpu()
                .data.numpy()
            )

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c], [0, b, d]]).astype(float)
            crop = cv2.warpAffine(
                image,
                mapping,
                (out_sz[0], out_sz[1]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=padding,
            )
            return crop

        s = crop_box[2] / p.instance_size
        sub_box = [
            crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
            crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
            s * p.exemplar_size,
            s * p.exemplar_size,
        ]
        s = p.out_size / sub_box[2]
        back_box = [
            -sub_box[0] * s,
            -sub_box[1] * s,
            state["im_w"] * s,
            state["im_h"] * s,
        ]
        mask_in_img = crop_back(mask, back_box, (state["im_w"], state["im_h"]))

        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        if cv2.__version__[-5] == "4":
            contours, _ = cv2.findContours(
                target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        else:
            _, contours, _ = cv2.findContours(
                target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array(
                [
                    [location[0], location[1]],
                    [location[0] + location[2], location[1]],
                    [location[0] + location[2], location[1] + location[3]],
                    [location[0], location[1] + location[3]],
                ]
            )

    target_pos[0] = max(0, min(state["im_w"], target_pos[0]))
    target_pos[1] = max(0, min(state["im_h"], target_pos[1]))
    target_sz[0] = max(10, min(state["im_w"], target_sz[0]))
    target_sz[1] = max(10, min(state["im_h"], target_sz[1]))

    state["target_pos"] = target_pos
    state["target_sz"] = target_sz
    state["score"] = score[best_pscore_id]
    state["mask"] = mask_in_img if mask_enable else []
    state["ploygon"] = rbox_in_img if mask_enable else []
    return state


def calculate_iou(rect1, rect2):
    rect1_min_x = rect1[0]
    rect1_min_y = rect1[1]
    rect1_max_x = rect1[0] + rect1[2]
    rect1_max_y = rect1[1] + rect1[3]

    rect2_min_x = rect2[0]
    rect2_min_y = rect2[1]
    rect2_max_x = rect2[0] + rect2[2]
    rect2_max_y = rect2[1] + rect2[3]

    inter_width = min(rect1_max_x, rect2_max_x) - max(rect1_min_x, rect2_min_x)
    inter_height = min(rect1_max_y, rect2_max_y) - max(rect1_min_y, rect2_min_y)

    inter_area = 0

    if inter_width > 0 and inter_height > 0:
        inter_area = inter_width * inter_height

    rect1_area = (rect1_max_x - rect1_min_x) * (rect1_max_y - rect1_min_y)
    rect2_area = (rect2_max_x - rect2_min_x) * (rect2_max_y - rect2_min_y)

    union_area = rect1_area + rect2_area - inter_area

    epsilon = 1e-6
    overlap_ratio = 0

    if inter_area > epsilon and union_area > epsilon:
        overlap_ratio = inter_area / union_area

    return overlap_ratio
