from __future__ import division

from arcgis.learn._utils.common import ArcGISMSImage

try:
    import random
    import logging
    import math
    import sys
    from collections import namedtuple
    from IPython.display import clear_output

    pyv = sys.version[0]

    from os.path import join, isdir
    from os import mkdir, makedirs
    import os
    import json
    import traceback
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from .pointcloud_data import get_device
    from fastai.data_block import DataBunch
    import types
    from PIL import Image
    from .env import (
        raise_fastai_import_error,
        HAS_GDAL,
        gdal_import_exception,
        GDAL_INSTALL_MESSAGE,
    )
except Exception:
    import_exception = traceback.format_exc()
    pass

sample_random = random.Random()
sample_random.seed(123456)

Corner = namedtuple("Corner", "x1 y1 x2 y2")
BBox = Corner
Center = namedtuple("Center", "x y w h")
image_name_len = 0

sep = os.sep


def get_image_for_tracking(path, grayscale=False):
    path = str(os.path.abspath(path))
    if not os.path.exists:
        raise Exception(
            f"The image path {path} could not be found on disk, please verify your training data."
        )

    img = ArcGISMSImage.read_image(path, keep_raw=True)
    if img is None:
        return None

    if len(img.shape) >= 3:
        dim = img.shape[0]
        if dim == 2:
            img = np.dstack((img[1], img[0]))[0]
        elif dim > 2:
            img = np.dstack((img[2], img[1], img[0]))

    if grayscale is True and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


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
        self.anchors = None
        self.all_anchors = None
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

        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])
        return True


class SubDataSet(object):
    def __init__(self, cfg):
        for string in ["root", "anno"]:
            if string not in cfg:
                raise Exception('SubDataSet need "{}"'.format(string))

        self.labels = self.filter_zero(cfg["anno"], cfg)

        def isint(x):
            try:
                int(x)
                return True
            except:
                return False

        to_del = []
        for video in self.labels:
            for track in self.labels[video]:
                frames = self.labels[video][track]
                frames = list(map(int, filter(lambda x: isint(x), frames.keys())))
                frames.sort()
                self.labels[video][track]["frames"] = frames
                if len(frames) <= 0:
                    to_del.append((video, track))

        for video, track in to_del:
            del self.labels[video][track]

        to_del = []
        for video in self.labels:
            if len(self.labels[video]) <= 0:
                to_del.append(video)

        for video in to_del:
            del self.labels[video]

        self.videos = list(self.labels.keys())
        self.root = "/"
        self.start = 0
        self.num = len(self.labels)
        self.num_use = self.num
        self.frame_range = 100
        self.path_format = "{}.{}.{}.jpg"
        self.mask_format = "{}.{}.m.png"

        self.pick = []

        self.__dict__.update(cfg)

        self.has_mask = True

        self.num_use = int(self.num_use)

        self.shuffle()

    def filter_zero(self, anno, cfg):
        name = cfg.get("mark", "")

        out = {}
        tot = 0
        new = 0
        zero = 0

        for video, tracks in anno.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    tot += 1
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                    else:
                        w, h = bbox
                    if w == 0 or h == 0:
                        zero += 1
                        continue
                    new += 1
                    new_frames[frm] = bbox

                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames

            if len(new_tracks) > 0:
                out[video] = new_tracks

        return out

    def shuffle(self):
        lists = list(range(self.start, self.start + self.num))

        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[: self.num_use]
        return self.pick

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = join(self.root, video, self.path_format.format(frame, track, "x"))
        image_anno = self.labels[video][track][frame]

        mask_path = join(self.root, video, self.mask_format.format(frame, track))

        return image_path, image_anno, mask_path

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info["frames"]

        if "hard" not in track_info:
            template_frame = random.randint(0, len(frames) - 1)

            left = max(template_frame - self.frame_range, 0)
            right = min(template_frame + self.frame_range, len(frames) - 1) + 1
            search_range = frames[left:right]
            template_frame = frames[template_frame]
            search_frame = random.choice(search_range)
        else:
            search_frame = random.choice(track_info["hard"])
            left = max(search_frame - self.frame_range, 0)
            right = min(search_frame + self.frame_range, len(frames) - 1) + 1
            template_range = frames[left:right]
            template_frame = random.choice(template_range)
            search_frame = frames[search_frame]

        return (
            self.get_image_anno(video_name, track, template_frame),
            self.get_image_anno(video_name, track, search_frame),
        )

    def get_random_target(self, index=-1):
        if index == -1:
            index = random.randint(0, self.num - 1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info["frames"]
        frame = random.choice(frames)

        return self.get_image_anno(video_name, track, frame)


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(float)
    crop = cv2.warpAffine(
        image,
        mapping,
        (out_sz, out_sz),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding,
    )
    return crop


class Augmentation:
    def __init__(self, cfg):
        self.shift = 0
        self.scale = 0
        self.blur = 0  # False
        self.resize = False
        self.rgbVar = np.array(
            [
                [-0.55919361, 0.98062831, -0.41940627],
                [1.72091413, 0.19879334, -1.82968581],
                [4.64467907, 4.73710203, 4.88324118],
            ],
            dtype=np.float32,
        )
        self.flip = 0

        self.eig_vec = np.array(
            [
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ],
            dtype=np.float32,
        )

        self.eig_val = np.array([[0.2175, 0.0188, 0.0045]], np.float32)

        self.__dict__.update(cfg)

    @staticmethod
    def random():
        return random.random() * 2 - 1.0

    def blur_image(self, image):
        def rand_kernel():
            size = np.random.randn(1)
            size = int(np.round(size)) * 2 + 1
            if size < 0:
                return None
            if random.random() < 0.5:
                return None
            size = min(size, 45)
            kernel = np.zeros((size, size))
            c = int(size / 2)
            wx = random.random()
            kernel[:, c] += 1.0 / size * wx
            kernel[c, :] += 1.0 / size * (1 - wx)
            return kernel

        kernel = rand_kernel()

        if kernel is not None:
            image = cv2.filter2D(image, -1, kernel)
        return image

    def __call__(self, image, bbox, size, gray=False, mask=None):
        if gray:
            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.zeros((grayed.shape[0], grayed.shape[1], 3), np.uint8)
            image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = grayed

        shape = image.shape

        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size - 1, size - 1))

        param = {}
        if self.shift:
            param["shift"] = (
                Augmentation.random() * self.shift,
                Augmentation.random() * self.shift,
            )

        if self.scale:
            param["scale"] = (
                (1.0 + Augmentation.random() * self.scale),
                (1.0 + Augmentation.random() * self.scale),
            )

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1 = crop_bbox.x1
        y1 = crop_bbox.y1

        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            scale_x, scale_y = param["scale"]
            bbox = Corner(
                bbox.x1 / scale_x,
                bbox.y1 / scale_y,
                bbox.x2 / scale_x,
                bbox.y2 / scale_y,
            )

        image = crop_hwc(image, crop_bbox, size)
        if not mask is None:
            mask = crop_hwc(mask, crop_bbox, size)

        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]
        offset = offset.reshape(3)
        image = image - offset

        if self.blur > random.random():
            image = self.blur_image(image)

        if self.resize:
            imageSize = image.shape[:2]
            ratio = max(math.pow(random.random(), 0.5), 0.2)  # 25 ~ 255
            rand_size = (
                int(round(ratio * imageSize[0])),
                int(round(ratio * imageSize[1])),
            )
            image = cv2.resize(image, rand_size)
            image = cv2.resize(image, tuple(imageSize))

        if self.flip and self.flip > Augmentation.random():
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            width = image.shape[1]
            bbox = Corner(width - 1 - bbox.x2, bbox.y1, width - 1 - bbox.x1, bbox.y2)

        return image, bbox, mask


class AnchorTargetLayer:
    def __init__(self, cfg):
        self.thr_high = 0.6
        self.thr_low = 0.3
        self.negative = 16
        self.rpn_batch = 64
        self.positive = 16

        self.__dict__.update(cfg)

    def __call__(self, anchor, target, size, neg=False, need_iou=False):
        anchor_num = anchor.anchors.shape[0]

        cls = np.zeros((anchor_num, size, size), dtype=np.int64)
        cls[...] = -1
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        if neg:
            l = size // 2 - 3
            r = size // 2 + 3 + 1

            cls[:, l:r, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), self.negative)
            cls[:] = -1
            cls[neg] = 0

            if not need_iou:
                return cls, delta, delta_weight
            else:
                overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
                return cls, delta, delta_weight, overlap

        tcx, tcy, tw, th = corner2center(target)

        anchor_box = anchor.all_anchors[0]
        anchor_center = anchor.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = (
            anchor_center[0],
            anchor_center[1],
            anchor_center[2],
            anchor_center[3],
        )

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > self.thr_high)
        neg = np.where(overlap < self.thr_low)

        pos, pos_num = select(pos, self.positive)
        neg, neg_num = select(neg, self.rpn_batch - pos_num)

        cls[pos] = 1
        delta_weight[pos] = 1.0 / (pos_num + 1e-6)

        cls[neg] = 0

        if not need_iou:
            return cls, delta, delta_weight
        else:
            return cls, delta, delta_weight, overlap


class DataSets(Dataset):
    def __init__(
        self,
        data_path,
        annotation,
        num,
        num_epoch=1,
        image_name_len=6,
        annotations=None,
    ):
        super(DataSets, self).__init__()

        anchor = {
            "stride": 8,
            "ratios": [0.33, 0.5, 1, 2, 3],
            "scales": [8],
            "round_dight": 0,
        }
        self.anchors = Anchors(anchor)

        self.template_size = 127
        self.origin_size = 127
        self.search_size = 143
        self.size = 3
        self.base_size = 0
        self.image_name_len = image_name_len
        self.annotations = annotations

        self.crop_size = 0
        self.chip_size = 127
        self._is_multispectral = False

        if (
            self.search_size - self.template_size
        ) / self.anchors.stride + 1 + self.base_size != self.size:
            raise Exception("size not match!")

        self.template_small = False

        self.anchors.generate_all_anchors(im_c=self.search_size // 2, size=self.size)

        self.anchor_target = AnchorTargetLayer({})

        datasets = {
            "siammask_data": {
                "root": os.path.join(data_path, "crop"),
                "anno": annotation,
                "num_use": 200000,
                "frame_range": 20,
            }
        }

        self.all_data = []
        start = 0
        self.num = 0
        self.show_batch_data = annotation
        for name in datasets:
            dataset = datasets[name]
            dataset["mark"] = name
            dataset["start"] = start

            dataset = SubDataSet(dataset)
            self.all_data.append(dataset)

            start += dataset.num
            self.num += dataset.num_use

        aug_cfg = {
            "template": {"shift": 4, "scale": 0.05},
            "search": {"shift": 64, "scale": 0.18, "blur": 0.18},
            "neg": 0.2,
            "gray": 0.25,
        }
        self.template_aug = Augmentation(aug_cfg["template"])
        self.search_aug = Augmentation(aug_cfg["search"])
        self.gray = aug_cfg["gray"]
        self.neg = aug_cfg["neg"]
        self.inner_neg = 0 if "inner_neg" not in aug_cfg else aug_cfg["inner_neg"]

        self.pick = None
        self.num = int(num)
        self.num *= num_epoch
        self.shuffle()

        self.infos = {
            "template": self.template_size,
            "search": self.search_size,
            "template_small": self.template_small,
            "gray": self.gray,
            "neg": self.neg,
            "inner_neg": self.inner_neg,
            "crop_size": self.crop_size,
            "anchor_target": self.anchor_target.__dict__,
            "num": self.num // num_epoch,
        }
        self._img_exts = [".jpg", ".png", ".tif"]

    def imread(self, path):
        img = get_image_for_tracking(path)

        if self.origin_size == self.template_size:
            return img, 1.0

        def map_size(exe, size):
            return int(round(((exe + 1) / (self.origin_size + 1) * (size + 1) - 1)))

        nsize = map_size(self.template_size, img.shape[1])

        img = cv2.resize(img, (nsize, nsize))

        return img, nsize / img.shape[1]

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.all_data:
                sub_p = subset.shuffle()
                p += sub_p

            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick

    def __len__(self):
        return self.num

    def find_dataset(self, index):
        for dataset in self.all_data:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def __getitem__(self, index, debug=False, show_batch=False, show_results=False):
        index = self.pick[index]
        dataset, index = self.find_dataset(index)

        gray = self.gray and self.gray > random.random()
        neg = self.neg and self.neg > random.random()

        if neg:
            template = dataset.get_random_target(index)
            if self.inner_neg and self.inner_neg > random.random():
                search = dataset.get_random_target()
            else:
                search = random.choice(self.all_data).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        display_image = search

        def center_crop(img, size):
            shape = img.shape[1]
            if shape == size:
                return img
            c = shape // 2
            l = c - size // 2
            r = c + size // 2 + 1
            return img[l:r, l:r]

        template_image, scale_z = self.imread(template[0])

        if self.template_small:
            template_image = center_crop(template_image, self.template_size)

        search_image, scale_x = self.imread(search[0])
        if dataset.has_mask and not neg:
            search_mask = (
                get_image_for_tracking(search[2], grayscale=True) > 0
            ).astype(np.float32)
        else:
            search_mask = np.zeros(search_image.shape[:2], dtype=np.float32)

        if self.crop_size > 0:
            search_image = center_crop(search_image, self.crop_size)
            search_mask = center_crop(search_mask, self.crop_size)

        def toBBox(image, shape):
            imh, imw = image.shape[:2]
            if len(shape) == 4:
                w, h = shape[2] - shape[0], shape[3] - shape[1]
            else:
                w, h = shape
            context_amount = 0.5
            exemplar_size = self.template_size
            wc_z = w + context_amount * (w + h)
            hc_z = h + context_amount * (w + h)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            w = w * scale_z
            h = h * scale_z
            cx, cy = imw // 2, imh // 2
            bbox = center2corner(Center(cx, cy, w, h))
            return bbox

        template_box = toBBox(template_image, template[1])
        search_box = toBBox(search_image, search[1])

        template, _, _ = self.template_aug(
            template_image, template_box, self.template_size, gray=gray
        )
        search, bbox, mask = self.search_aug(
            search_image, search_box, self.search_size, gray=gray, mask=search_mask
        )

        if show_batch:
            return display_image, self.image_name_len, self.annotations

        if show_results:
            return display_image

        cls, delta, delta_weight = self.anchor_target(
            self.anchors, bbox, self.size, neg
        )
        if dataset.has_mask and not neg:
            mask_weight = cls.max(axis=0, keepdims=True)
        else:
            mask_weight = np.zeros([1, cls.shape[1], cls.shape[2]], dtype=np.float32)

        template, search = map(
            lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search]
        )

        mask = (np.expand_dims(mask, axis=0) > 0.5) * 2 - 1

        return (
            [
                template,
                search,
                cls,
                delta,
                delta_weight,
                np.array(bbox, np.float32),
                np.array(mask, np.float32),
                np.array(mask_weight, np.float32),
            ],
            [[]],
        )

    def show(self, idx, axes=None):
        global image_name_len
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(15, 7))

        for i, id in enumerate(idx):
            data, img_len, ann = self.__getitem__(id, debug=False, show_batch=True)
            sep = os.sep

            base_folder = data[0].split("crop")[0]
            folder_name = data[0].split(sep)[-2]
            frame_id = data[0].split(sep)[-1].split(".")[1]
            frame_number = data[0].split(sep)[-1].split(".")[0]
            mask_name = folder_name + sep + frame_number.zfill(img_len)
            # key = list(ann[folder_name].keys())[-1]
            for key in list(ann[folder_name].keys()):
                for cnt in ann[folder_name][key]:
                    if cnt["mask_name"] == mask_name:
                        display_data = (cnt["display_mask"], cnt["bbox"])
                        break

            img_path = os.path.join(
                base_folder,
                "JPEGImages",
                folder_name,
                frame_number.zfill(img_len) + ".jpg",
            )

            image = get_image_for_tracking(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bbox = display_data[1]
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]) - int(bbox[0]), int(bbox[3] - int(bbox[1]))),
                (255, 0, 0),
                2,
            )

            overlay = image.copy()
            cv2.drawContours(overlay, display_data[0], -1, (0, 0, 255), -1)

            image_new = cv2.addWeighted(overlay, 0.5, image, 1, 0)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].imshow(image_new)

    def _show_pro(self, idx, axes=None):
        global image_name_len
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(15, 7))

        for i, id in enumerate(idx):
            data, img_len, ann = self.__getitem__(id, debug=False, show_batch=True)
            sep = os.sep

            base_folder = data[0].split("crop")[0]
            folder_name = data[0].split(sep)[-2]
            frame_id = data[0].split(sep)[-1].split(".")[1]
            frame_number = data[0].split(sep)[-1].split(".")[0]
            mask_name = folder_name + sep + frame_number.zfill(img_len)
            for key in list(ann[folder_name].keys()):
                for cnt in ann[folder_name][key]:
                    if cnt["mask_name"] == mask_name:
                        display_data = (cnt["display_mask"], cnt["bbox"])
                        break

            for ext in self._img_exts:
                try:
                    img_path = os.path.join(
                        base_folder,
                        "images",
                        frame_number.zfill(img_len) + ext,
                    )
                    if os.path.isfile(img_path):
                        break
                except:
                    continue

            image = get_image_for_tracking(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox = display_data[1]
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]) - int(bbox[0]), int(bbox[3] - int(bbox[1]))),
                (255, 0, 0),
                2,
            )

            overlay = image.copy()
            cv2.drawContours(overlay, display_data[0], -1, (0, 0, 255), -1)

            image_new = cv2.addWeighted(overlay, 0.5, image, 1, 0)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].imshow(image_new)


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError("Argument xyxy must be a list, tuple, or numpy array.")


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]
    return boxes_from_polys


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(float)
    crop = cv2.warpAffine(
        image,
        mapping,
        (out_sz, out_sz),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding,
    )
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(
    image,
    bbox,
    context_amount=0.5,
    exemplar_size=127,
    instanc_size=255,
    padding=(0, 0, 0),
):
    target_pos = [(bbox[2] + bbox[0]) / 2.0, (bbox[3] + bbox[1]) / 2.0]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_like_SiamFCx(
    image,
    bbox,
    context_amount=0.5,
    exemplar_size=127,
    instanc_size=255,
    padding=(0, 0, 0),
):
    target_pos = [(bbox[2] + bbox[0]) / 2.0, (bbox[3] + bbox[1]) / 2.0]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return x


def crop_video_pro(video, v, crop_path, data_path, instanc_size):
    video_crop_base_path = join(crop_path, video)
    if not isdir(video_crop_base_path):
        makedirs(video_crop_base_path)

    anno_base_path = join(data_path, "labels")
    img_base_path = join(data_path, "images")

    for trackid, o in enumerate(list(v)):
        obj = v[o]
        for frame in obj:
            file_name = frame["file_name"]
            ann_ext = frame["ext"]
            ann_path = join(join(anno_base_path, video), file_name + ann_ext)
            img_path = join(img_base_path, file_name + ann_ext)

            if not os.path.isfile(img_path) and ann_ext == ".png":
                img_path = join(img_base_path, file_name + ".jpg")

            label = get_image_for_tracking(ann_path, grayscale=True)

            im = get_image_for_tracking(img_path)
            if im is None:
                continue

            avg_chans = np.mean(im, axis=(0, 1))
            bbox = frame["bbox"]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            x = crop_like_SiamFCx(
                im, bbox, instanc_size=instanc_size, padding=avg_chans
            )

            cv2.imwrite(
                join(
                    video_crop_base_path,
                    "{:06d}.{:02d}.x.jpg".format(
                        int(file_name.split(sep)[-1]), trackid
                    ),
                ),
                x,
            )

            mask = crop_like_SiamFCx(
                (label == int(o)).astype(np.float32),
                bbox,
                instanc_size=instanc_size,
                padding=0,
            )
            mask = ((mask > 0.2) * 255).astype(np.uint8)
            x[:, :, 0] = mask + (mask == 0) * x[:, :, 0]
            cv2.imwrite(
                join(
                    video_crop_base_path,
                    "{:06d}.{:02d}.m.png".format(
                        int(file_name.split(sep)[-1]), trackid
                    ),
                ),
                mask,
            )


def crop_video(video, v, crop_path, data_path, instanc_size):
    video_crop_base_path = join(crop_path, video)
    if not isdir(video_crop_base_path):
        makedirs(video_crop_base_path)

    anno_base_path = join(data_path, "Annotations")
    img_base_path = join(data_path, "JPEGImages")

    for trackid, o in enumerate(list(v)):
        obj = v[o]
        for frame in obj:
            file_name = frame["file_name"]
            ann_path = join(anno_base_path, file_name + ".png")
            img_path = join(img_base_path, file_name + ".jpg")
            im = get_image_for_tracking(img_path)
            label = get_image_for_tracking(ann_path, grayscale=True)
            avg_chans = np.mean(im, axis=(0, 1))
            bbox = frame["bbox"]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            x = crop_like_SiamFCx(
                im, bbox, instanc_size=instanc_size, padding=avg_chans
            )
            cv2.imwrite(
                join(
                    video_crop_base_path,
                    "{:06d}.{:02d}.x.jpg".format(
                        int(file_name.split(sep)[-1]), trackid
                    ),
                ),
                x,
            )
            mask = crop_like_SiamFCx(
                (label == int(o)).astype(np.float32),
                bbox,
                instanc_size=instanc_size,
                padding=0,
            )
            mask = ((mask > 0.2) * 255).astype(np.uint8)
            x[:, :, 0] = mask + (mask == 0) * x[:, :, 0]
            cv2.imwrite(
                join(
                    video_crop_base_path,
                    "{:06d}.{:02d}.m.png".format(
                        int(file_name.split(sep)[-1]), trackid
                    ),
                ),
                mask,
            )


class Instance(object):
    instID = 0
    pixelCount = 0

    def __init__(self, imgNp, instID):
        if instID == 0:
            return
        self.instID = int(instID)
        self.pixelCount = int(self.getInstancePixels(imgNp, instID))

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

    def toDict(self):
        buildDict = {}
        buildDict["instID"] = self.instID
        buildDict["pixelCount"] = self.pixelCount
        return buildDict

    def __str__(self):
        return "(" + str(self.instID) + ")"


def train_val_split(dataset):
    snippets = dict()
    for k, v in dataset.items():
        video = dict()
        for i, o in enumerate(list(v)):
            obj = v[o]
            snippet = dict()
            trackid = "{:02d}".format(i)
            for frame in obj:
                file_name = frame["file_name"]
                frame_name = "{:06d}".format(int(file_name.split(sep)[-1]))
                bbox = frame["bbox"]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                snippet[frame_name] = bbox
            video[trackid] = snippet
        snippets[k] = video
    return snippets


def printProgressBar(i, max):
    n_bar = 50
    j = i / max
    sys.stdout.write("\r")
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  completed")
    sys.stdout.flush()


def check_data_sanity(path):
    if not os.path.isdir(path):
        raise Exception(f"Invalid directory. Please check the path {path}")

    ann_dir_path = os.path.join(path, "Annotations")
    if not os.path.isdir(ann_dir_path):
        raise Exception(
            f"Invalid directory. Please check the "
            f"annotation folder path {ann_dir_path}. "
            "Please make sure the folder name is Annotations"
        )

    ann_dirs = os.listdir(ann_dir_path)
    img_dir_path = os.path.join(path, "JPEGImages")
    if not os.path.isdir(img_dir_path):
        raise Exception(
            f"Invalid directory. Please check the "
            f"images folder path {img_dir_path}. "
            "Please make sure the folder name is JPEGImages"
        )

    img_dirs = os.listdir(img_dir_path)
    json_file_path = os.path.join(path, "meta.json")
    if not os.path.isfile(json_file_path):
        raise Exception(
            f"Invalid file. Please check the "
            f"meta.json file path {json_file_path}. "
            "Please make sure the name of json is meta.json"
        )
    json_ann = json.load(open(json_file_path))
    total = 0
    for _, video in enumerate(json_ann["videos"]):
        if video in ann_dirs and video in img_dirs:
            total += 1
    if total <= 1:
        raise Exception(
            f"Please input at least two sequences in the {path.name}"
            " directory namely 'Annotations' and 'JPEGImages'."
            "Also, please provide at least two sequences "
            "in meta.json"
        )


def check_pro_data_sanity(path):
    if not os.path.isdir(path):
        raise Exception(f"Invalid directory. Please check the path {path}")

    ann_dir_path = os.path.join(path, "labels")
    if not os.path.isdir(ann_dir_path):
        raise Exception(
            f"Invalid directory. Please check the "
            f"annotation folder path {ann_dir_path}. "
            "Please make sure the folder name is labels"
        )

    img_dir_path = os.path.join(path, "images")
    if not os.path.isdir(img_dir_path):
        raise Exception(
            f"Invalid directory. Please check the "
            f"images folder path {img_dir_path}. "
            "Please make sure the folder name is images"
        )

    total = 0
    seq_paths = [
        f.path
        for f in os.scandir(ann_dir_path)
        if f.is_dir() and len(os.listdir(f.path)) != 0
    ]
    for idx, seq_path in enumerate(seq_paths):
        annotations = [
            file
            for file in os.listdir(seq_path)
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".tif")
        ]
        if len(annotations) > 0:
            total += 1

    if total <= 1:
        raise Exception(
            f"Please input at least two sequences in the {path.name}"
            " directory namely 'labels'."
        )


def prepare_pro_data(path, batch_size, val_split_pct, **kwargs):
    check_pro_data_sanity(path)
    global image_name_len
    data_dir = path
    ann_dir = os.path.join(path, "labels")
    img_dir = os.path.join(path, "images")
    seq_paths = [
        f.path
        for f in os.scandir(ann_dir)
        if f.is_dir() and len(os.listdir(f.path)) != 0
    ]
    num_obj = 0
    num_ann = 0
    all_objects = 0
    print("Extracting info")
    ann_dict = {}
    total = len(seq_paths)
    for idx, seq_path in enumerate(seq_paths):
        seq_name = os.path.basename(seq_path)
        annotation_files = [
            file
            for file in os.listdir(seq_path)
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".tif")
        ]
        annotations = []
        instanceIds = []
        for annotation in annotation_files:
            ann_filename = os.path.join(seq_path, annotation)
            file_name_suffix = os.path.basename(ann_filename)
            file_name_suffix_split = os.path.splitext(file_name_suffix)[0]
            file_name_ext = os.path.splitext(file_name_suffix)[1]
            image_name_len = len(file_name_suffix_split)

            img = get_image_for_tracking(ann_filename, grayscale=True)

            if img is None:
                continue

            h, w = img.shape[:2]
            objects = dict()
            for instanceId in np.unique(img):
                if instanceId == 0:
                    continue
                instance_obj = Instance(img, instanceId)
                instance_obj_dict = instance_obj.toDict()
                mask = (img == instanceId).astype(np.uint8)
                contour, _ = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                polygons = [c.reshape(-1).tolist() for c in contour]
                instance_obj_dict["display_mask"] = contour
                instance_obj_dict["contours"] = [p for p in polygons if len(p) > 4]
                if (
                    len(instance_obj_dict["contours"])
                    and instance_obj_dict["pixelCount"] > 1000
                ):
                    objects[instanceId] = instance_obj_dict

            for objId in objects:
                if len(objects[objId]) == 0:
                    continue
                obj = objects[objId]
                len_p = [len(p) for p in obj["contours"]]
                if min(len_p) <= 4:
                    print("Warning: invalid contours.")
                    continue
                ann = dict()
                ann["h"] = h
                ann["w"] = w
                ann["file_name"] = file_name_suffix_split
                ann["id"] = int(objId)
                ann["segmentation"] = obj["contours"]
                ann["iscrowd"] = 0
                ann["display_mask"] = obj["display_mask"]
                ann["area"] = obj["pixelCount"]
                ann["bbox"] = xyxy_to_xywh(polys_to_boxes([obj["contours"]])).tolist()[
                    0
                ]
                ann["mask_name"] = seq_name + os.sep + file_name_suffix_split
                ann["ext"] = file_name_ext
                annotations.append(ann)
                all_objects += 1
                instanceIds.append(objId)
                num_ann += 1

        instanceIds = sorted(set(instanceIds))
        num_obj += len(instanceIds)
        video_ann = {str(iId): [] for iId in instanceIds}
        for ann in annotations:
            video_ann[str(ann["id"])].append(ann)

        ann_dict[seq_name] = video_ann
        printProgressBar(idx, total)

    items = list(ann_dict.items())
    train_dict = dict(items)

    clear_output()
    crop_path = os.path.join(path, "crop")
    if not os.path.isdir(crop_path):
        os.mkdir(crop_path)
    set_crop_base_path = join(crop_path)
    set_img_base_path = data_dir
    n_video = len(train_dict)

    print("Applying transformations..")

    total = len(train_dict.keys())
    ind_pb = 0
    for k, v in train_dict.items():
        try:
            crop_video_pro(k, v, set_crop_base_path, set_img_base_path, 511)
        except Exception as e:
            print(e)
            break
        ind_pb += 1
        printProgressBar(ind_pb, total)

    val_set = int(((n_video * (val_split_pct * 10)) / 100) * 10)
    val_all_obj = int(((all_objects * (val_split_pct * 10)) / 100) * 10)
    if val_set == 0:
        val_set = 1

    def num_anns(elem):
        ann_dict = elem[1]
        val = 0
        for k, v in ann_dict.items():
            val = val + len(v)
        return val

    items = sorted(items, key=num_anns, reverse=True)
    train_dict = dict(items[:-val_set])
    snippets = train_val_split(train_dict)
    train = {k: v for (k, v) in snippets.items()}
    val_dict = dict(items[-val_set:])

    snippets = train_val_split(val_dict)
    val = {k: v for (k, v) in snippets.items()}

    if all_objects == 0:
        raise Exception(
            "Not enough valid data. Please ensure the labels are correct and enough data is present."
        )

    train_set = DataSets(path, train, all_objects, 5, image_name_len, ann_dict)
    val_set = DataSets(path, val, val_all_obj, 5, image_name_len)

    train_set.shuffle()
    val_set.shuffle()
    init_kwargs = {}
    train_dl = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        **init_kwargs,
    )

    valid_dl = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        **init_kwargs,
    )

    device = get_device()
    data = DataBunch(train_dl, valid_dl, device=device)
    data.path = path
    data.infos = data.train_ds.infos
    data.show_batch = types.MethodType(_show_batch_pro, data)
    data._dataset_type = "ObjectTracking"
    data.train_folders = train
    data.val_folders = val
    clear_output()
    return data


def prepare_object_tracking_data(path, batch_size, val_split_pct, **kwargs):
    check_data_sanity(path)
    global image_name_len
    data_dir = path
    ann_dir = os.path.join(path, "Annotations")
    num_obj = 0
    num_ann = 0
    all_objects = 0
    print("Extracting info")
    ann_dict = {}
    json_ann = json.load(open(os.path.join(data_dir, "meta.json")))
    total = len(json_ann["videos"])
    for vid, video in enumerate(json_ann["videos"]):
        if video not in os.listdir(ann_dir):
            continue
        v = json_ann["videos"][video]
        frames = []
        for obj in v["objects"]:
            o = v["objects"][obj]
            frames.extend(o["frames"])
        frames = sorted(set(frames))
        annotations = []
        instanceIds = []
        for frame in frames:
            file_name = os.path.join(video, frame)
            mask_name = video + os.sep + frame
            mask_filename = os.path.join(ann_dir, file_name + ".png")
            image_name_len = len(frame)
            img = get_image_for_tracking(mask_filename, grayscale=True)
            if img is None:
                continue
            h, w = img.shape[:2]
            objects = dict()
            for instanceId in np.unique(img):
                if instanceId == 0:
                    continue
                instance_obj = Instance(img, instanceId)
                instance_obj_dict = instance_obj.toDict()
                mask = (img == instanceId).astype(np.uint8)
                contour, _ = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                polygons = [c.reshape(-1).tolist() for c in contour]
                instance_obj_dict["display_mask"] = contour
                instance_obj_dict["contours"] = [p for p in polygons if len(p) > 4]
                if (
                    len(instance_obj_dict["contours"])
                    and instance_obj_dict["pixelCount"] > 1000
                ):
                    objects[instanceId] = instance_obj_dict

            for objId in objects:
                if len(objects[objId]) == 0:
                    continue
                obj = objects[objId]
                len_p = [len(p) for p in obj["contours"]]
                if min(len_p) <= 4:
                    print("Warning: invalid contours.")
                    continue

                ann = dict()
                ann["h"] = h
                ann["w"] = w
                ann["file_name"] = file_name
                ann["id"] = int(objId)
                ann["segmentation"] = obj["contours"]
                ann["iscrowd"] = 0
                ann["display_mask"] = obj["display_mask"]
                ann["area"] = obj["pixelCount"]
                ann["bbox"] = xyxy_to_xywh(polys_to_boxes([obj["contours"]])).tolist()[
                    0
                ]
                ann["mask_name"] = mask_name
                annotations.append(ann)
                all_objects += 1
                instanceIds.append(objId)
                num_ann += 1

        instanceIds = sorted(set(instanceIds))
        num_obj += len(instanceIds)
        video_ann = {str(iId): [] for iId in instanceIds}
        for ann in annotations:
            video_ann[str(ann["id"])].append(ann)

        ann_dict[video] = video_ann
        printProgressBar(vid, total)

    items = list(ann_dict.items())
    train_dict = dict(items)

    clear_output()
    crop_path = os.path.join(path, "crop")
    if not isdir(crop_path):
        mkdir(crop_path)
    set_crop_base_path = join(crop_path)
    set_img_base_path = data_dir
    n_video = len(train_dict)
    print("Applying transformations..")
    total = len(train_dict.keys())
    ind_pb = 0
    for k, v in train_dict.items():
        try:
            crop_video(k, v, set_crop_base_path, set_img_base_path, 511)
        except Exception as e:
            print(e)
            break
        ind_pb += 1
        printProgressBar(ind_pb, total)

    val_set = int(((n_video * (val_split_pct * 10)) / 100) * 10)
    val_all_obj = int(((all_objects * (val_split_pct * 10)) / 100) * 10)
    if val_set == 0:
        val_set = 1

    def num_anns(elem):
        ann_dict = elem[1]
        val = 0
        for k, v in ann_dict.items():
            val = val + len(v)
        return val

    items = sorted(items, key=num_anns, reverse=True)
    train_dict = dict(items[:-val_set])

    snippets = train_val_split(train_dict)
    train = {k: v for (k, v) in snippets.items()}
    val_dict = dict(items[-val_set:])

    snippets = train_val_split(val_dict)
    val = {k: v for (k, v) in snippets.items()}

    if all_objects == 0:
        raise Exception(
            "Not enough valid data. Please ensure the labels are correct and enough data is present."
        )

    train_set = DataSets(path, train, all_objects, 5, image_name_len, ann_dict)
    val_set = DataSets(path, val, val_all_obj, 5, image_name_len)

    train_set.shuffle()
    val_set.shuffle()
    init_kwargs = {}
    train_dl = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        **init_kwargs,
    )

    valid_dl = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        **init_kwargs,
    )

    device = get_device()
    data = DataBunch(train_dl, valid_dl, device=device)
    data.path = path
    data.infos = data.train_ds.infos
    data.show_batch = types.MethodType(show_batch, data)
    data._dataset_type = "ObjectTracking"
    data.train_folders = train
    data.val_folders = val
    clear_output()
    return data


def show_batch(self, rows=4, **kwargs):
    img_idxs = [random.randint(0, len(self.train_ds) - 1) for k in range(rows * 2)]
    fig, axes = plt.subplots(nrows=rows, ncols=2, squeeze=False, figsize=(20, rows * 5))
    ind = 0
    for idx in range(0, len(img_idxs), 2):
        self.train_ds.show([img_idxs[idx], img_idxs[idx + 1]], axes[ind])
        ind += 1
    pass


def _show_batch_pro(self, rows=4, **kwargs):
    img_idxs = [random.randint(0, len(self.train_ds) - 1) for k in range(rows * 2)]
    fig, axes = plt.subplots(nrows=rows, ncols=2, squeeze=False, figsize=(20, rows * 5))
    ind = 0
    for idx in range(0, len(img_idxs), 2):
        self.train_ds._show_pro([img_idxs[idx], img_idxs[idx + 1]], axes[ind])
        ind += 1
    pass
