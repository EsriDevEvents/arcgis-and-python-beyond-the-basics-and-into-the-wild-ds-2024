"""
MIT License

Copyright (c) 2020 Ziqiang

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
# based on https://github.com/ZQPei/deep_sort_pytorch
import os
import warnings
import traceback
from pathlib import Path

try:
    from numpy import isin, true_divide
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from fastai.basic_train import LearnerCallback
    from fastai.basic_train import Learner
    from fastai.vision import imagenet_stats
    from fastai.vision.data import ImageDataBunch
    from fastai.vision.transform import ResizeMethod
    from fastai.metrics import accuracy
    from fastai.callback import Callback
    from fastai.torch_core import add_metrics
    from ._arcgis_model import _EmptyData

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

REID_V1 = "reid_v1"
REID_V2 = "reid_v2"

DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 64
DEFAULT_CHANNELS = 3
DEFAULT_NUM_CLASSES = 4
DEFAULT_CLASS_MAPPING = {"obj_a": 1, "obj_b": 2, "obj_c": 3, "obj_d": 4}


class BasicBlock_v1(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock_v1, self).__init__()

        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False), nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False), nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers_v1(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [
                BasicBlock_v1(c_in, c_out, is_downsample=is_downsample),
            ]
        else:
            blocks += [
                BasicBlock_v1(c_out, c_out),
            ]
    return nn.Sequential(*blocks)


# TODO: Handle varied input sizes


class Net_v1(nn.Module):
    def __init__(self, num_classes=625, reid=False):
        super(Net_v1, self).__init__()

        self.img_shape = [3, 128, 64]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers_v1(32, 32, 2, False)
        # 32 64 32
        self.layer2 = make_layers_v1(32, 64, 2, True)
        # 64 32 16
        self.layer3 = make_layers_v1(64, 128, 2, True)
        # 128 16 8
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128 * 16 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
        )
        # 256 1 1
        self.reid = reid
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        if self.reid:
            x = self.dense[0](x)
            x = self.dense[1](x)
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.dense(x)
        # B x 128
        # classifier
        x = self.classifier(x)
        return x


class BasicBlock_v2(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock_v2, self).__init__()

        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False), nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False), nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers_v2(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [
                BasicBlock_v2(c_in, c_out, is_downsample=is_downsample),
            ]
        else:
            blocks += [
                BasicBlock_v2(c_out, c_out),
            ]
    return nn.Sequential(*blocks)


class Net_v2(nn.Module):
    def __init__(self, num_classes=184, reid=False):
        super(Net_v2, self).__init__()

        self.img_shape = [3, 128, 64]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers_v2(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers_v2(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers_v2(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers_v2(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x


def build_opt_lr(trainable_params, **kwargs):
    optimizer = torch.optim.SGD(trainable_params, 0.1, momentum=0.9, weight_decay=5e-4)

    return optimizer


get_reid_loss = torch.nn.CrossEntropyLoss()


def _get_num_classes(data):
    if data is None or isinstance(data, _EmptyData):
        return DEFAULT_NUM_CLASSES
    return max(len(data.train_ds.y.classes), len(data.valid_ds.y.classes))


def get_default_backbone():
    return REID_V2


def get_default_imgsize():
    return (DEFAULT_HEIGHT, DEFAULT_WIDTH)


def get_model(num_classes, backbone, reid=False):
    if backbone is None:
        backbone = get_default_backbone()
    if backbone == REID_V1:
        return Net_v1(num_classes=num_classes, reid=reid)
    elif backbone == REID_V2:
        return Net_v2(num_classes, reid=reid)


def check_data_sanity(data, dataset_types=[]):
    try:
        if data is None:
            return False
        if isinstance(data, ImageDataBunch) and data._dataset_type in dataset_types:
            return True
        else:
            return False
    except Exception as e:
        return False


def _get_model_state(model_path, model=None):
    model_state = None
    if model_path is not None and os.path.isfile(model_path):
        if not torch.cuda.is_available():
            state = torch.load(model_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            state = torch.load(
                model_path, map_location=lambda storage, loc: storage.cuda(device)
            )

            model_state = state
            if "model" in set(state.keys()):
                model_state = state["model"]
            elif isinstance(state, dict) and "net_dict" in state.keys():
                model_state = state["net_dict"]

    if model_state is None and model is not None and hasattr(model, "state_dict"):
        model_state = model.state_dict()

    return model_state


def load_for_prediction(model_path, num_classes, backbone, model):
    model_state = _get_model_state(model_path, model)
    model = get_model(num_classes, backbone, reid=True)
    if model_state is not None:
        model.load_state_dict(model_state)
    else:
        model = None

    return model


def _check_shape(data_in, img_shape):
    try:
        for i, data in enumerate(data_in):
            x = data[0][0]
            c_in = x.size()[0]
            h_in = x.size()[1]
            w_in = x.size()[2]

            if [c_in, h_in, w_in] != img_shape:
                return False
    except BaseException:
        return True
    return True


def _check_data_shape(data, img_shape):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if data is not None and isinstance(data, ImageDataBunch):
            is_valid = True
            if data.train_dl is not None:
                is_valid = is_valid and _check_shape(data.train_dl, img_shape)

            if data.valid_dl is not None:
                is_valid = is_valid and _check_shape(data.valid_dl, img_shape)

            if data.test_dl is not None:
                is_valid = is_valid and _check_shape(data.test_dl, img_shape)

            if not is_valid:
                raise Exception(
                    """\nInvalid input data shape.
                        DeepSort only supports input of the form
                        (channels=3, height=128, width=64)\n"""
                )


def get_fake_data():
    train_tfms = []
    val_tfms = []
    transforms = (train_tfms, val_tfms)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        kwargs_transforms = {}
        kwargs_transforms["size"] = get_default_imgsize()
        kwargs_transforms["resize_method"] = ResizeMethod.SQUISH
        data = ImageDataBunch.single_from_classes(
            Path(os.getcwd()),
            sorted(list(DEFAULT_CLASS_MAPPING.values())),
            ds_tfms=transforms,
        ).normalize(imagenet_stats)

    data.class_mapping = DEFAULT_CLASS_MAPPING
    data.classes = list(data.class_mapping.values())
    data._is_empty = True
    data._dataset_type = "Imagenet"
    data.resize_to = get_default_imgsize()

    return data


def reid_accuracy(input, targs):
    correct = input.max(dim=1)[1].eq(targs).sum().item()
    count = targs.size(0)

    return correct, count


class Accuracy(Callback):
    "Wrap a `func` in a callback for metrics computation."

    def __init__(self, func):
        # If func has a __name__ use this one else it should be a partial
        name = func.__name__ if hasattr(func, "__name__") else func.func.__name__
        self.func, self.name = func, name

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.correct, self.count = 0.0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not isinstance(last_target, (tuple, list)):
            last_target = [last_target]
        correct, count = self.func(last_output, *last_target)
        self.correct += correct
        self.count += count

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, (1.0 * self.correct) / self.count)


def _get_metrics():
    return Accuracy(reid_accuracy)


def get_learner(data=None, num_classes=None, backbone=None, device=torch.device("cpu")):
    learn = None
    if num_classes is None:
        num_classes = _get_num_classes(data)

    model = get_model(num_classes, backbone)
    if model is None:
        return learn

    _check_data_shape(data, model.img_shape)

    model = model.to(device)
    metrics = _get_metrics()
    if data is not None:
        learn = Learner(
            data=data,
            model=model,
            loss_func=get_reid_loss,
            opt_func=build_opt_lr,
            metrics=metrics,
        )
    return learn
