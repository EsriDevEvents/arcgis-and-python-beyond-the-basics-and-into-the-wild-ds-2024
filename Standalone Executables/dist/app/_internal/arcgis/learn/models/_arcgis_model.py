import arcgis
from pathlib import Path
import os
import glob
import shutil
import time
import tempfile
import json
import logging
from .._data import _raise_fastai_import_error
from .._utils.env import (
    HAS_TENSORFLOW,
    raise_tensorflow_import_error,
    _LAMBDA_TEXT_CLASSIFICATION,
    is_arcgispronotebook,
    reload_IPython,
)
from warnings import warn
import contextlib
import io
import sys
import socket
from functools import wraps
import traceback
import inspect
import types
import functools
import warnings

HAS_FASTAI = True
HAS_TENSORBOARDX = True

try:
    if not _LAMBDA_TEXT_CLASSIFICATION:
        from fastai.vision.learner import model_meta, _default_meta
        from .._utils.common import get_post_processed_model
        from torchvision import models

    from fastai.callbacks import TrackerCallback, EarlyStoppingCallback
    from fastai.basic_train import LearnerCallback
    from torch import nn
    import torch
    import numpy as np
    import math
    import warnings
    from fastai.distributed import *
    from fastai.torch_core import distrib_barrier
    import argparse
    from torch.nn.parallel import DistributedDataParallel
    from .._utils.segmentation_loss_functions import dice
    from fastai.basics import partial
    import pandas as pd
    from ... import __version__ as ArcGISLearnVersion
    from ._pointcnn_utils import AverageMetric
    from fastai.core import camel2snake
    import timm
    from .._utils.evaluate_batchsize import estimate_batch_size
    from .._utils.evaluate_batchsize import unsupported_models
    from .._data import prepare_data

    # EarlyStoppingCallback should run as one
    # of the first callback so that stop training flag is set
    # and other callbacks can behave accordingly.
    # e.g: Do not checkpoint final model after early stopping.
    EarlyStoppingCallback._order = -10

except ImportError as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

    class TrackerCallback:
        pass

    class LearnerCallback:
        pass


logger = logging.getLogger()

# For lr computation, skip beginning and trailing values.
losses_skipped = 5
trailing_losses_skipped = 5
model_characteristics_folder = "ModelCharacteristics"

if HAS_FASTAI and not _LAMBDA_TEXT_CLASSIFICATION:
    # Declare the family of backbones to be unpacked and used by different models as supported types
    _vgg_family = [
        models.vgg11.__name__,
        models.vgg11_bn.__name__,
        models.vgg13.__name__,
        models.vgg13_bn.__name__,
        models.vgg16.__name__,
        models.vgg16_bn.__name__,
        models.vgg19.__name__,
        models.vgg19_bn.__name__,
    ]
    _resnet_family = [
        models.resnet18.__name__,
        models.resnet34.__name__,
        models.resnet50.__name__,
        models.resnet101.__name__,
        models.resnet152.__name__,
    ]
    _densenet_family = [
        models.densenet121.__name__,
        models.densenet169.__name__,
        models.densenet161.__name__,
        models.densenet201.__name__,
    ]


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield
    sys.stdout = save_stdout


def _get_device():
    if getattr(arcgis.env, "_processorType", "") == "GPU" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(arcgis.env, "_processorType", "") == "CPU":
        device = torch.device("cpu")
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    return device


class _EmptyDS(object):
    def __init__(self, size):
        self.size = (size, size)


class _EmptyData:
    def __init__(self, path, c, loss_func, chip_size, train_ds=True):
        self.path = path.parent
        if (
            getattr(arcgis.env, "_processorType", "") == "GPU"
            and torch.cuda.is_available()
        ):
            self.device = torch.device("cuda")
        elif getattr(arcgis.env, "_processorType", "") == "CPU":
            self.device = torch.device("cpu")
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.c = c
        self.loss_func = loss_func
        self.chip_size = chip_size

        if train_ds:
            self.train_ds = [[_EmptyDS(chip_size)]]


class _MultiGPUCallback(LearnerCallback):
    """
    Parallelize over multiple GPUs only if multiple GPUs are present.
    """

    def __init__(self, learn):
        super(_MultiGPUCallback, self).__init__(learn)

        self.multi_gpu = torch.cuda.device_count() > 1

    def on_train_begin(self, **kwargs):
        if self.multi_gpu:
            logger.info("Training on multiple GPUs")
            self.learn.model = nn.DataParallel(self.learn.model)

    def on_train_end(self, **kwargs):
        if self.multi_gpu:
            self.learn.model = self.learn.model.module


def _set_multigpu_callback(model):
    if (
        (not hasattr(arcgis.env, "_gpuid"))
        or (arcgis.env._gpuid >= torch.cuda.device_count())
    ) and (not getattr(arcgis.env, "_processorType", False) == "CPU"):
        model.learn.callback_fns.append(_MultiGPUCallback)


def _set_ddp_multigpu(model):
    try:
        parser = argparse.ArgumentParser()
    except IndexError:
        model._multigpu_training = False
        return
    parser.add_argument("--local_rank", type=int)
    args, unknown = parser.parse_known_args()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        model._multigpu_training = False
        return
    model._multigpu_training = True
    args.gpu = args.gpu % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    backend = "nccl"
    if os.name == "nt":
        backend = "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    model._rank_distributed = args.gpu


def _isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


def _create_zip(zipname, path):
    import shutil

    if os.path.exists(os.path.join(path, zipname) + ".dlpk"):
        os.remove(os.path.join(path, zipname) + ".dlpk")

    temp_dir = tempfile.TemporaryDirectory().name
    zip_file = shutil.make_archive(os.path.join(temp_dir, zipname), "zip", path)
    dlpk_base = os.path.splitext(zip_file)[0]
    os.rename(zip_file, dlpk_base + ".dlpk")
    dlpk_file = dlpk_base + ".dlpk"
    shutil.move(dlpk_file, path)


class SaveModelCallback(TrackerCallback):
    def __init__(
        self,
        model,
        every="improvement",
        name="bestmodel",
        load_best_at_end=True,
        **kwargs,
    ):
        super().__init__(learn=model.learn, **kwargs)
        self.model = model
        self.every = every
        self.name = name
        self.load_best_at_end = load_best_at_end

        # set some default value of best epoch attribute
        self.best_epoch = 0
        self.learn._best_epoch = 0

        if self.every not in ["improvement", "epoch"]:
            warn(
                'SaveModel every {} is invalid, falling back to "improvement".'.format(
                    self.every
                )
            )
            self.every = "improvement"

    def on_epoch_end(self, epoch, **kwargs):
        "Compare the value monitored to its best score and maybe save the model."

        if int(os.environ.get("RANK", 0)):
            return

        # do not save model after early stopping kicks in.
        if not kwargs.get("stop_training", False):
            current = self.get_monitor_value()

            if isinstance(current, torch.Tensor):
                if current.is_cuda:
                    current = current.cpu()

            # if a better checkpoint is found.
            better_checkpoint = current is not None and self.operator(
                current, self.best
            )
            if better_checkpoint:
                self.best_epoch = epoch
                self.learn._best_epoch = epoch
                self.best = current

            self.current = current

            if self.every == "epoch":
                self.model._save(
                    f"{self.name}_epoch_{epoch}",
                    zip_files=False,
                    save_html=False,
                    compute_metrics=False,
                )
            # every improvement
            elif better_checkpoint:
                self.remove_previous()
                self.model._save(
                    f"{self.name}_epoch_{epoch}",
                    zip_files=False,
                    save_html=False,
                    compute_metrics=False,
                )

    def remove_previous(self):
        # to avoid creating multiple best checkpoints.
        saved_path = os.path.join(self.model.learn.path, self.model.learn.model_dir)
        for p in glob.glob(os.path.join(saved_path, self.name + "*")):
            shutil.rmtree(p)

    def on_train_end(self, **kwargs):
        "Load the best model."
        if self.load_best_at_end:
            try:
                self.model.load(f"{self.name}_epoch_{self.best_epoch}")
            except FileNotFoundError:
                # don't show message in child process in case of multigpu
                if not int(os.environ.get("RANK", 0)):
                    # logging this to notify about possible errors.
                    print("Could not load the best model.")

            try:
                self.model.save(
                    f"{self.name}_epoch_{self.best_epoch}", compute_metrics=False
                )
            except:
                # logging this to notify about possible errors.
                print("Encountered error in saving checkpoint.")


# Multispectral Models Specific resources start #

valid_init_schemes = ["red_band", "random", "all_random"]
rgb_map = {"r": 0, "red": 0, "g": 1, "green": 1, "b": 2, "blue": 2}


def get_band_mapping(band_name):
    # Extra Logic goes Here
    # For example: NIR --> RED(0); Coastal --> BLUE(2)
    # We can store Custom weights for multispectral models and load them in this logic
    #
    return rgb_map.get(band_name.lower(), None)


def _get_tail(model):
    if hasattr(model, "named_children"):
        child_name, child = next(model.named_children())
        if isinstance(child, nn.Conv2d):
            return child_name, child

    if hasattr(model, "children"):
        for children in model.children():
            try:
                child_name, child = _get_tail(children)
                return child_name, child
            except:
                pass


def _get_ms_tail(tail, data, type_init="random"):
    in_chanls = len(data._extract_bands)
    new_tail = nn.Conv2d(
        in_channels=in_chanls,
        out_channels=tail.out_channels,
        kernel_size=tail.kernel_size,
        stride=tail.stride,
        padding=tail.padding,
        dilation=tail.dilation,
        groups=tail.groups,
        bias=tail.bias is not None,
        padding_mode=tail.padding_mode,
    )
    # referred from https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/helpers.py#L143
    if in_chanls == 1:
        new_tail.weight.data = tail.weight.data.float().sum(dim=1, keepdim=True)
    else:
        repeat = int(math.ceil(in_chanls / 3))
        new_tail.weight.data = (
            (tail.weight.data.float().repeat(1, repeat, 1, 1)[:, :in_chanls, :, :])
            * 3
            / float(in_chanls)
        )
    for i, j in enumerate(data._extract_bands):
        band = str(data._bands[j]).lower()
        b = get_band_mapping(band)  # rgb_map.get(band, None)
        if b is not None and not type_init == "all_random":
            new_tail.weight.data[:, i] = tail.weight.data[:, b]
        else:
            if type_init == "red_band":
                new_tail.weight.data[:, i] = tail.weight.data[
                    :, 0
                ]  # Red Band Weights for all other band weights
            elif type_init == "random" or type_init == "all_random":
                # Random Weights for all other band weights
                pass
    return new_tail


def _set_tail(model, new_tail):
    updated = False
    if hasattr(model, "named_children"):
        child_name, child = next(model.named_children())
        if isinstance(child, nn.Conv2d):
            setattr(model, child_name, new_tail)
            updated = True
    if hasattr(model, "children") and not updated:
        for children in model.children():
            try:
                _set_tail(children, new_tail)
                return
            except:
                pass


def _change_tail(model, data, tail_weights_type=None):
    tail_name, tail = _get_tail(model)
    if tail_weights_type is None:
        tail_weights_type = getattr(arcgis.env, "type_init_tail_parameters", "random")
    if tail_weights_type not in valid_init_schemes:
        raise Exception(
            f"""
        \n'{type_init}' is not a valid scheme for initializing model tail weights.
        \nplease set a valid scheme from 'red_band', 'random' or 'all_random'.
        \n`arcgis.env.type_init_tail_parameters={{valid_scheme}}`
        """
        )
    new_tail = _get_ms_tail(tail, data, type_init=tail_weights_type)
    _set_tail(model, new_tail)
    return model


def _get_backbone_meta(arch_name):
    _model_meta = {i.__name__: j for i, j in model_meta.items()}
    return _model_meta.get(arch_name, _default_meta)


# Multispectral Models Specific resources end #


def _device_check():
    if hasattr(arcgis, "env") and getattr(arcgis.env, "_processorType", "") == "CPU":
        return True

    if not torch.cuda.is_available():
        warnings.warn("Cuda is not available")
        return True

    move_to_cpu = False

    incorrect_binary_warn = """
    Found GPU%d %s which requires CUDA_VERSION >= %d for
    optimal performance and fast startup time, but your PyTorch was compiled
    with CUDA_VERSION %d. Please install the correct PyTorch binary
    using instructions from https://pytorch.org
    It may continue to work on CPU.
    """

    old_gpu_warn = """
    Found GPU%d %s which is of cuda capability %d.%d.
    PyTorch no longer supports this GPU because it is too old.
    The minimum cuda capability that we support is 3.7.
    Go to https://pytorch.org for more info on how to install
    or build a PyTorch version that has been compiled for your
    GPU architecture (Cuda compute capability).
    It may continue to work on CPU.
    """

    CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    for d in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(d)
        major = capability[0]
        name = torch.cuda.get_device_name(d)
        if CUDA_VERSION < 8000 and major >= 6:
            warnings.warn(incorrect_binary_warn % (d, name, 8000, CUDA_VERSION))
            move_to_cpu = True
        elif CUDA_VERSION < 9000 and major >= 7:
            warnings.warn(incorrect_binary_warn % (d, name, 9000, CUDA_VERSION))
            move_to_cpu = True
        elif capability == (3, 0) or major < 3:
            warnings.warn(old_gpu_warn % (d, name, major, capability[1]))
            move_to_cpu = True

    try:
        if not torch._C._cuda_isDriverSufficient():
            move_to_cpu = True
            if torch._C._cuda_getDriverVersion() == 0:
                # found no NVIDIA driver on the system
                warnings.warn(
                    """
                Found no GPU driver. CPU will be used for processing.
                """
                )
            else:
                warnings.warn(
                    """
                The NVIDIA driver on your system is too old (found version {})
                or your GPU architecture is very old and it's not supported.
                Please update your GPU driver by downloading and installing
                a new version from the URL: http://www.nvidia.com/Download/index.aspx
                Alternatively, go to https://pytorch.org for more info
                on how to install or build a PyTorch version that has been
                compiled for your GPU architecture (Cuda compute capability)
                or for your version of Cuda driver. It may continue to work on CPU.
                """.format(
                        str(torch._C._cuda_getDriverVersion())
                    )
                )
    except:
        pass

    return move_to_cpu


class ArcGISModel(object):
    def __init__(self, data, backbone=None, **kwargs):
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)
        if not _LAMBDA_TEXT_CLASSIFICATION:
            move_to_cpu = _device_check()
        else:
            move_to_cpu = True
        # Force move to CPU
        if move_to_cpu:
            arcgis.env._processorType = "CPU"

        self._device = _get_device()

        if backbone is None:
            self._backbone = models.resnet34
        elif type(backbone) is str:
            if hasattr(models, backbone):
                self._backbone = getattr(models, backbone)
            elif hasattr(models.detection, backbone):
                self._backbone = getattr(models.detection, backbone)
            elif "timm:" in backbone:
                bckbn = backbone.split(":")[1]
                if hasattr(timm.models, bckbn):
                    self._backbone = getattr(timm.models, bckbn)
        else:
            self._backbone = backbone

        if not hasattr(self, "_backbone"):
            self._backbone = models.resnet34

        if hasattr(data, "_is_multispectral"):  # multispectral support
            self._is_multispectral = getattr(data, "_is_multispectral")
        else:
            self._is_multispectral = False
        if self._is_multispectral:
            self._imagery_type = data._imagery_type
            self._bands = data._bands
            self._orig_backbone = self._backbone

            @wraps(self._orig_backbone)
            def backbone_wrapper(*args, **inkwargs):
                if "pretrained_backbone" in kwargs:
                    pretrained_backbone = kwargs["pretrained_backbone"]
                    assert type(pretrained_backbone) == bool
                    if len(args) > 0:
                        args = tuple([pretrained_backbone, *args[1:]])
                    else:
                        inkwargs["pretrained"] = pretrained_backbone
                return _change_tail(
                    self._orig_backbone(*args, **inkwargs),
                    data,
                    kwargs.get("tail_weights_type"),
                )

            backbone_wrapper._is_multispectral = True
            self._backbone = backbone_wrapper

        if not hasattr(data, "class_mapping") and hasattr(data, "classes"):
            data.class_mapping = {v: v for v in data.classes}

        if data is not None and getattr(data, "path", None) is None:
            data.path = Path(os.path.abspath("."))

        if getattr(self, "_is_edge_detection", False):
            if len(data.classes) > 2:
                raise Exception(
                    "Found multi-labels in the data, This is a binary segmentation model and hence please export the data with binary labels."
                    # noqa
                )

            data.class_mapping = {1: data.classes[1]}

        self.learn = None
        self._data = data
        self._learning_rate = None
        self._backend = getattr(self, "_backend", "pytorch")
        self._model_metrics_cache = None
        self._slice_lr = True
        self._pretrained_path = kwargs.get("pretrained_path", None)
        if hasattr(self._data, "arcgis_init_kwargs"):
            self._check_data_support_with_pretrained_path()
        self._model_kwargs = kwargs
        if self.__class__.__name__ not in unsupported_models:
            if not getattr(data, "_is_empty", False) and hasattr(
                data, "_estimate_batch"
            ):
                if data._estimate_batch:
                    try:
                        data._estimate_batch = False
                        batch_size = estimate_batch_size(self, mode="none")
                        self._data.train_dl.batch_size = (
                            batch_size.recommended_batchsize
                        )
                        self._data.valid_dl.batch_size = (
                            batch_size.recommended_batchsize
                        )
                    except Exception as e:
                        data._estimate_batch = True

    def _check_data_support_with_pretrained_path(self):
        if self._data is not None and self._pretrained_path is not None:
            with open(Path(self._pretrained_path).with_suffix(".emd")) as f:
                emd = json.load(f)
            if self._data.chip_size != emd["ImageHeight"]:
                import copy
                from .._data import prepare_data
                import logging

                logger = logging.getLogger()
                logger.warning(
                    f"""Setting the `chip_size` of input data ({self._data.chip_size}) to same as input model's ({emd["ImageHeight"]})."""
                )
                arcgis_init_kwargs = copy.deepcopy(self._data.arcgis_init_kwargs)
                arcgis_init_kwargs["chip_size"] = emd["ImageHeight"]
                arcgis_init_kwargs["resize_to"] = emd["resize_to"]
                self._data = prepare_data(**arcgis_init_kwargs)

    def _check_backbone_support(self, backbone):
        "Fetches the backbone name and returns True if it is in the list of supported backbones"
        backbone_name = backbone if type(backbone) is str else backbone.__name__
        if type(backbone) is not str and "timm" in backbone.__module__:
            backbone_name = "timm:" + backbone.__name__
        return False if backbone_name not in self.supported_backbones else True

    def _check_dataset_support(self, data):
        "Fetches the dataset name and returns True if it is in the list of supported dataset type"
        if hasattr(data, "dataset_type"):
            if getattr(data, "dataset_type") not in self.supported_datasets:
                raise Exception(
                    f"Enter only compatible datasets from {', '.join(self.supported_datasets)}"
                )

    def _arcgis_init_callback(self):
        if self._is_multispectral:
            if self._data._train_tail:
                params_iterator = self.learn.model.parameters()
                next(
                    params_iterator
                ).requires_grad = True  # make first conv weights learnable

                tail_name, first_layer = _get_tail(self.learn.model)

                if (
                    first_layer.bias is not None
                    or self.__class__.__name__ == "MaskRCNN"
                    or self.__class__.__name__ == "ModelExtension"
                ):
                    # make first conv bias weights learnable
                    # In case of maskrcnn make the batch norm trainable
                    next(params_iterator).requires_grad = True
                self.learn.create_opt(slice(3e-3))
        if hasattr(self, "_show_results_multispectral"):
            self.show_results = self._show_results_multispectral

    # function for checking if data exists for using class functions.
    def _check_requisites(self):
        if isinstance(self._data, _EmptyData) or getattr(
            self._data, "_is_empty", False
        ):
            raise Exception("Can't call this function without data.")

    # function for checking if tensorflow is installed otherwise raise error.
    def _check_tf(self):
        if not HAS_TENSORFLOW:
            raise_tensorflow_import_error()

    def _init_tensorflow(self, data, backbone):
        self._check_tf()

        from .._utils.common import get_color_array
        from .._utils.common_tf import (
            handle_backbone_parameter,
            get_input_shape,
            check_backbone_is_mobile_optimized,
        )

        # Get color Array
        color_array = get_color_array(data.color_mapping)
        if len(data.color_mapping) == (data.c - 1):
            # Add Background color
            color_array = np.concatenate(
                [np.array([[0.0, 0.0, 0.0, 0.0]]), color_array]
            )
        data._multispectral_color_array = color_array

        # Handle Backbone
        self._backbone = handle_backbone_parameter(backbone)

        self._backbone_mobile_optimized = check_backbone_is_mobile_optimized(
            self._backbone
        )

        # Initialize Backbone
        in_shape = get_input_shape(data.chip_size)
        self._backbone_initalized = self._backbone(
            input_shape=in_shape, include_top=False, weights="imagenet"
        )

        self._backbone_initalized.trainable = False
        self._device = torch.device("cpu")
        self._data = data

    def lr_find(self, allow_plot=True):
        """
        Runs the Learning Rate Finder. Helps in choosing the
        optimum learning rate for training the model.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        allow_plot              Optional boolean. Display the plot of losses
                                against the learning rates and mark the optimal
                                value of the learning rate on the plot.
                                The default value is 'True'.
        =====================   ===========================================
        """
        self._check_requisites()
        temp1 = self.learn.path
        metrics = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                metrics = self.learn.metrics
                self.learn.metrics = []
                # ddp training
                if getattr(self, "_multigpu_training", False):
                    self.learn.lr_find()
                    distrib_barrier()
                    # remove tmp.pth created during lr_find in parent process
                    if not int(os.environ.get("RANK", 0)):
                        os.remove(
                            Path(self.learn.path) / self.learn.model_dir / "tmp.pth"
                        )
                else:
                    with tempfile.TemporaryDirectory(
                        prefix="arcgisTemp_"
                    ) as _tempfolder:
                        self.learn.path = Path(_tempfolder)
                        self.learn.lr_find()
            except Exception as e:
                # if some error comes in lr_find
                raise e
            finally:
                self.learn.metrics = metrics
                # Revert
                self.learn.path = temp1

            #
            self.learn.path = temp1

            from IPython.display import clear_output

            clear_output()
            lr, index = self._find_lr()
            if allow_plot:
                self._show_lr_plot(index)

        return lr

    def _show_lr_plot(
        self,
        index,
        losses_skipped=losses_skipped,
        trailing_losses_skipped=trailing_losses_skipped,
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        losses = self.learn.recorder.losses
        lrs = self.learn.recorder.lrs
        final_losses_skipped = 0
        if (
            len(self.learn.recorder.losses[losses_skipped:-trailing_losses_skipped])
            >= 5
        ):
            losses = self.learn.recorder.losses[losses_skipped:-trailing_losses_skipped]
            lrs = self.learn.recorder.lrs[losses_skipped:-trailing_losses_skipped]
            final_losses_skipped = losses_skipped
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.0e"))
        ax.plot(
            self.learn.recorder.lrs[index],
            self.learn.recorder.losses[index],
            markersize=10,
            marker="o",
            color="red",
        )

        plt.show()

    def _find_lr(
        self,
        losses_skipped=losses_skipped,
        trailing_losses_skipped=trailing_losses_skipped,
        section_factor=3,
    ):
        losses = self.learn.recorder.losses
        lrs = self.learn.recorder.lrs
        final_losses_skipped = 0
        if (
            len(self.learn.recorder.losses[losses_skipped:-trailing_losses_skipped])
            >= 5
        ):
            losses = self.learn.recorder.losses[losses_skipped:-trailing_losses_skipped]
            lrs = self.learn.recorder.lrs[losses_skipped:-trailing_losses_skipped]
            final_losses_skipped = losses_skipped

        n = len(losses)

        max_start = 0
        max_end = 0

        lds = [1] * n

        for i in range(1, n):
            for j in range(0, i):
                if losses[i] < losses[j] and lds[i] < lds[j] + 1:
                    lds[i] = lds[j] + 1
                if lds[max_end] < lds[i]:
                    max_end = i
                    max_start = max_end - lds[max_end]

        sections = (max_end - max_start) / section_factor
        final_index = max_start + int(sections) + int(sections / 2)
        return lrs[final_index], final_losses_skipped + final_index

    @property
    def _model_metrics(self):
        raise NotImplementedError

    @property
    def available_metrics(self):
        """
        List of available metrics that are displayed in the training
        table. Set `monitor` value to be one of these while calling
        the `fit` method.
        """
        metrics = ["valid_loss"]
        for m in self.learn.metrics:
            if isinstance(m, AverageMetric) or isinstance(m, functools.partial):
                metrics.append(m.func.__name__)
            elif isinstance(m, types.FunctionType):
                metrics.append(m.__name__)
            else:
                metrics.append(camel2snake(m.__class__.__name__))

        return metrics

    def fit(
        self,
        epochs=10,
        lr=None,
        one_cycle=True,
        early_stopping=False,
        checkpoint=True,  # "all", "best", True, False ("best" and True are same.)
        tensorboard=False,
        monitor="valid_loss",  # whatever is passed here, earlystopping and checkpointing will use that.
        **kwargs,
    ):
        """
        Train the model for the specified number of epochs and using the
        specified learning rates

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        epochs                  Required integer. Number of cycles of training
                                on the data. Increase it if underfitting.
        ---------------------   -------------------------------------------
        lr                      Optional float or slice of floats. Learning rate
                                to be used for training the model. If ``lr=None``,
                                an optimal learning rate is automatically deduced
                                for training the model.
        ---------------------   -------------------------------------------
        one_cycle               Optional boolean. Parameter to select 1cycle
                                learning rate schedule. If set to `False` no
                                learning rate schedule is used.
        ---------------------   -------------------------------------------
        early_stopping          Optional boolean. Parameter to add early stopping.
                                If set to 'True' training will stop if parameter
                                `monitor` value stops improving for 5 epochs.
                                A minimum difference of 0.001 is required for
                                it to be considered an improvement.
        ---------------------   -------------------------------------------
        checkpoint              Optional boolean or string.
                                Parameter to save checkpoint during training.
                                If set to `True` the best model
                                based on `monitor` will be saved during
                                training. If set to 'all', all checkpoints
                                are saved. If set to False, checkpointing will
                                be off. Setting this parameter loads the best
                                model at the end of training.
        ---------------------   -------------------------------------------
        tensorboard             Optional boolean. Parameter to write the training log.
                                If set to 'True' the log will be saved at
                                `<dataset-path>/training_log` which can be visualized in
                                tensorboard. Required tensorboardx version=2.1

                                The default value is 'False'.

                                .. note::
                                    Not applicable for Text Models
        ---------------------   -------------------------------------------
        monitor                 Optional string. Parameter specifies
                                which metric to monitor while checkpointing
                                and early stopping. Defaults to 'valid_loss'. Value
                                should be one of the metric that is displayed in
                                the training table. Use `{model_name}.available_metrics`
                                to list the available metrics to set here.
        =====================   ===========================================
        """
        if os.environ.get("BLOCK_MODEL_TRAINING", 0) == "1":
            raise Exception(f"This model cannot be trained in ArcGIS Online Notebooks")

        if getattr(self, "_is_mm3d", False):
            self.learn.model.prediction = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._check_requisites()

            if lr is None:
                print("Finding optimum learning rate.")

                lr = self.lr_find(allow_plot=False)
                if self._slice_lr is True and len(self.learn.layer_groups) > 1:
                    lr = slice(lr / 10, lr)

            self._learning_rate = lr
            self._model_metrics_cache = None
            if (
                getattr(self._data, "_dataset_type", None) == "Classified_Tiles"
                and (
                    dice.__qualname__
                    not in [
                        metric.func.__qualname__
                        if hasattr(metric, "func")
                        else metric.__qualname__
                        for metric in self.learn.metrics
                    ]
                )
                and not getattr(self, "_is_edge_detection", False)
            ):
                self.learn.metrics.extend([dice])
            if arcgis.env.verbose:
                logger.info("Fitting the model.")

            if (
                not (type(self).__name__) == "EfficientDet"
                and getattr(self, "_backend", "pytorch") == "tensorflow"
            ):
                checkpoint = False

            callbacks = kwargs["callbacks"] if "callbacks" in kwargs.keys() else []
            kwargs.pop("callbacks", None)
            monitored_names = self.available_metrics
            if monitor not in monitored_names:
                raise Exception(f"`monitor` must be set to one from {monitored_names}")
            self.monitor = monitor
            if early_stopping:
                callbacks.append(
                    EarlyStoppingCallback(
                        learn=self.learn, monitor=monitor, min_delta=0.001, patience=5
                    )
                )
            self._is_checkpointed = checkpoint
            if checkpoint:
                from datetime import datetime

                now = datetime.now()
                if checkpoint != True and checkpoint != "all":
                    raise Exception(
                        "Checkpoint can only be set to a boolean, or 'all'."
                    )
                every = "improvement" if checkpoint is True else "epoch"
                save_callback_params = kwargs.get(
                    "save_callback_params", {"monitor": monitor, "every": every}
                )
                callbacks.append(
                    SaveModelCallback(
                        self,
                        name=now.strftime("checkpoint_%Y-%m-%d_%H-%M-%S"),
                        **save_callback_params,
                    )
                )
            kwargs.pop("save_callback_params", None)
            # If tensorboardx is installed write a log with name as timestamp
            if tensorboard:
                try:
                    import tensorboardX

                    # LearnerTensorboardWriter uses SummaryWriter from tensorboardX
                    from fastai.callbacks.tensorboard import LearnerTensorboardWriter
                    from .._utils.tensorboard_utils import ArcGISTBCallback
                except:
                    raise
                training_id = time.strftime("log_%Y-%m-%d_%H-%M-%S")
                log_path = Path(self._data.path) / "training_log"
                abs_path = os.path.abspath(log_path)
                training_id = type(self).__name__ + "_" + training_id
                callbacks.append(
                    partial(
                        ArcGISTBCallback,
                        base_dir=log_path,
                        name=training_id,
                        arcgis_model=self,
                    )(learn=self.learn)
                )
                hostname = socket.gethostname()
                print(
                    "Monitor training on Tensorboard using the following command: 'tensorboard --host={} --logdir=\"{}\"'".format(
                        hostname, abs_path
                    )
                )
            # Send out a warning if tensorboardX is not installed
            elif tensorboard:
                warn(
                    "Install tensorboardX 2.1 'pip install tensorboardx==2.1' to write training log"
                )

            self._fit_callbacks = callbacks
            if one_cycle:
                self.learn.fit_one_cycle(epochs, lr, callbacks=callbacks, **kwargs)
            else:
                self.learn.fit(epochs, lr, callbacks=callbacks, **kwargs)

    def unfreeze(self):
        """
        Unfreezes the earlier layers of the model for fine-tuning.
        """
        self.learn.unfreeze()

    def plot_losses(self):
        """
        Plot validation and training losses after fitting the model.
        """
        self._check_requisites()
        try:
            self.learn.recorder.plot_losses()
        except:
            raise Exception("You need to train your model to compute losses")

    def _create_emd_template(
        self, path, compute_metrics=True, save_inference_file=True
    ):
        _emd_template = {}

        # For old models - add lr, ModelName
        if isinstance(self._data, _EmptyData) or getattr(
            self._data, "_is_empty", False
        ):
            _emd_template = self._data.emd
            _emd_template["ModelFormat"] = "NCHW"
            if self._backend == "tensorflow" and self._framework == "tflite":
                _emd_template["ModelFormat"] = "NHWC"
            _emd_template["ModelFile"] = path.name
            if not _emd_template.get("ModelName"):
                _emd_template["ModelName"] = type(self).__name__

            if not _emd_template.get("LearningRate"):
                _emd_template["LearningRate"] = "0.0"
            if _emd_template["ModelName"] in ["MaskRCNN", "UnetClassifier", "CycleGAN"]:
                _emd_template["SupportsVariableTileSize"] = True
            else:
                _emd_template["SupportsVariableTileSize"] = False
            _emd_template["ArcGISLearnVersion"] = ArcGISLearnVersion

            return _emd_template

        if self._backbone is None:
            backbone = self._backbone
        else:
            if self._backend == "tensorflow":
                backbone = self._backbone._keras_api_names[-1].split(".")[-1]
            else:
                if "timm" in self._backbone.__module__:
                    backbone = "timm:" + self._backbone.__name__
                else:
                    backbone = self._backbone.__name__
            if backbone == "backbone_wrapper":
                backbone = self._orig_backbone.__name__

        _emd_template = self._get_emd_params(save_inference_file)

        _emd_template["ModelFormat"] = "NCHW"
        if self._backend == "tensorflow" and self._framework == "tflite":
            _emd_template["ModelFormat"] = "NHWC"
        if getattr(self, "_data", None) is not None:
            _emd_template["MinCellSize"] = getattr(self._data, "_emd", {}).get(
                "MinCellSize", None
            )
            _emd_template["MaxCellSize"] = getattr(self._data, "_emd", {}).get(
                "MaxCellSize", None
            )

        _emd_template["SupportsVariableTileSize"] = _emd_template.get(
            "SupportsVariableTileSize", False
        )
        _emd_template["ArcGISLearnVersion"] = ArcGISLearnVersion

        if getattr(self, "_fit_callbacks", None) is not None:
            checkpoint_callback = [
                c for c in self._fit_callbacks if isinstance(c, SaveModelCallback)
            ]
            if checkpoint_callback != []:
                checkpoint_callback = checkpoint_callback[0]
                key = getattr(self, "monitor", "valid_loss")
                val = checkpoint_callback.best
                if isinstance(val, torch.Tensor):
                    val = val.cpu().item()
                else:
                    val = float(val)
                _emd_template[f"monitored_{key}"] = val

        if isinstance(self._learning_rate, slice):
            _emd_lr = slice(
                "{0:1.4e}".format(self._learning_rate.start),
                "{0:1.4e}".format(self._learning_rate.stop),
            )
        elif self._learning_rate is not None:
            _emd_lr = "{0:1.4e}".format(self._learning_rate)
        else:
            _emd_lr = None

        _emd_template["ModelFile"] = path.name

        if hasattr(self._data, "chip_size"):
            chip_size = self._data.chip_size
            if not isinstance(chip_size, tuple):
                chip_size = (chip_size, chip_size)

            _emd_template["ImageHeight"] = chip_size[0]
            _emd_template["ImageWidth"] = chip_size[1]

        if hasattr(self._data, "_image_space_used"):
            _emd_template["ImageSpaceUsed"] = self._data._image_space_used

        _emd_template["LearningRate"] = str(_emd_lr)
        _emd_template["ModelName"] = type(self).__name__.replace("_", "")
        _emd_template["backend"] = self._backend

        if getattr(self, "_is_mmsegdet", False):
            model_params = {
                "model_name": self._kwargs["model"],
                "backend": self._backend,
            }
        elif getattr(self, "model_type", False) == "SR3":
            model_params = {"backbone": "SR3", "backend": self._backend}
        else:
            model_params = {"backbone": backbone, "backend": self._backend}
        if _emd_template.get("ModelParameters", None) is None:
            _emd_template["ModelParameters"] = model_params
        else:
            for _key in model_params:
                _emd_template["ModelParameters"][_key] = model_params[_key]

        if compute_metrics:
            if self._model_metrics_cache == None:
                print("Computing model metrics...")
                self._model_metrics_cache = self._model_metrics
            _emd_template.update(self._model_metrics_cache)

        resize_to = None
        if hasattr(self._data, "resize_to") and self._data.resize_to:
            resize_to = self._data.resize_to

        _emd_template["resize_to"] = resize_to

        # Check if model is Multispectral and dump parameters for that
        _emd_template["IsMultispectral"] = getattr(self, "_is_multispectral", False)
        if _emd_template.get("IsMultispectral", False):
            _emd_template["Bands"] = self._data._bands
            _emd_template["ImageryType"] = self._data._imagery_type
            if getattr(self._data, "_dataset_type", None) != "ChangeDetection":
                _emd_template["ExtractBands"] = self._data._extract_bands
            if not getattr(self._data, "_dataset_type", None) == "SuperResolution":
                _emd_template["NormalizationStats"] = {
                    "band_min_values": self._data._band_min_values,
                    "band_max_values": self._data._band_max_values,
                    "band_mean_values": self._data._band_mean_values,
                    "band_std_values": self._data._band_std_values,
                    "scaled_min_values": self._data._scaled_min_values,
                    "scaled_max_values": self._data._scaled_max_values,
                    "scaled_mean_values": self._data._scaled_mean_values,
                    "scaled_std_values": self._data._scaled_std_values,
                }
                for _stat in _emd_template["NormalizationStats"]:
                    if _emd_template["NormalizationStats"][_stat] is not None:
                        _emd_template["NormalizationStats"][_stat] = _emd_template[
                            "NormalizationStats"
                        ][_stat].tolist()
                _emd_template["DoNormalize"] = self._data._do_normalize
        if (
            getattr(self._data, "_dataset_type", None) == "Pix2Pix"
            or getattr(self._data, "_dataset_type", None) == "CycleGAN"
            or getattr(self._data, "_dataset_type", None) == "WNet_cGAN"
        ):
            _emd_template["ExtractBands"] = self._data._extract_bands
            _emd_template["NormalizationStats"] = {
                "band_min_values": self._data._band_min_values,
                "band_max_values": self._data._band_max_values,
                "band_mean_values": self._data._band_mean_values,
                "band_std_values": self._data._band_std_values,
                "scaled_min_values": self._data._scaled_min_values,
                "scaled_max_values": self._data._scaled_max_values,
                "scaled_mean_values": self._data._scaled_mean_values,
                "scaled_std_values": self._data._scaled_std_values,
            }
            for _stat in _emd_template["NormalizationStats"]:
                if _emd_template["NormalizationStats"][_stat] is not None:
                    _emd_template["NormalizationStats"][_stat] = _emd_template[
                        "NormalizationStats"
                    ][_stat].tolist()
            if getattr(self._data, "_dataset_type", None) == "WNet_cGAN":
                _emd_template["NormalizationStats_b"] = {
                    "band_min_values": self._data._band_min_values_b,
                    "band_max_values": self._data._band_max_values_b,
                    "band_mean_values": self._data._band_mean_values_b,
                    "band_std_values": self._data._band_std_values_b,
                    "scaled_min_values": self._data._scaled_min_values_b,
                    "scaled_max_values": self._data._scaled_max_values_b,
                    "scaled_mean_values": self._data._scaled_mean_values_b,
                    "scaled_std_values": self._data._scaled_std_values_b,
                }
                for _stat in _emd_template["NormalizationStats_b"]:
                    if _emd_template["NormalizationStats_b"][_stat] is not None:
                        _emd_template["NormalizationStats_b"][_stat] = _emd_template[
                            "NormalizationStats_b"
                        ][_stat].tolist()
            if getattr(self._data, "_dataset_type", None) == "CycleGAN":
                _emd_template["n_channel_rev"] = len(
                    _emd_template["NormalizationStats"]["band_min_values"]
                )
        if getattr(self._data, "_dataset_type", None) == "Classified_Tiles":
            if not getattr(self, "_is_edge_detection", False):
                if not getattr(self, "_orient_data", False):
                    if compute_metrics:
                        _emd_template[
                            "per_class_metrics"
                        ] = self.per_class_metrics().to_json()
        return _emd_template

    @staticmethod
    def _write_emd(_emd_template, path):
        json.dump(_emd_template, open(path, "w"), indent=4)

        return path.stem

    def _get_emd_params(self, save_inference_file):
        return {}

    @staticmethod
    def _create_html(path_model):
        import base64

        model_characteristics_dir = os.path.join(
            path_model.parent.absolute(), model_characteristics_folder
        )
        loss_graph = os.path.join(model_characteristics_dir, "loss_graph.png")
        show_results = os.path.join(model_characteristics_dir, "show_results.png")
        confusion_matrix = os.path.join(
            model_characteristics_dir, "confusion_matrix.png"
        )
        metrics_file = os.path.join(model_characteristics_dir, "metrics.html")
        results_file = os.path.join(model_characteristics_dir, "results.html")
        iframe_showresults = os.path.exists(
            os.path.join(model_characteristics_dir, "show_results.html")
        )

        encoded_losses_img = None
        if os.path.exists(loss_graph):
            encoded_losses_img = "data:image/png;base64,{0}".format(
                base64.b64encode(open(loss_graph, "rb").read()).decode("utf-8")
            )

        encoded_showresults = None
        if os.path.exists(show_results):
            encoded_showresults = "data:image/png;base64,{0}".format(
                base64.b64encode(open(show_results, "rb").read()).decode("utf-8")
            )

        confusion_matrix_img = None
        if os.path.exists(confusion_matrix):
            confusion_matrix_img = "data:image/png;base64,{0}".format(
                base64.b64encode(open(confusion_matrix, "rb").read()).decode("utf-8")
            )

        metrics_html = None
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics_html = f.read()

        results_html = None
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results_html = f.read()

        html_file_path = os.path.join(path_model.parent, "model_metrics.html")

        emd_path = os.path.join(path_model.parent, path_model.stem + ".emd")
        if not os.path.exists(emd_path):
            return

        emd_template = json.load(open(emd_path, "r"))

        encoded_losses = ""
        if encoded_losses_img:
            encoded_losses = f"""
                <p><b>Training and Validation loss</b></p>
                <img src="{encoded_losses_img}" alt="training and validation losses">
            """

        if emd_template.get("ModelParameters", {}).get("model_name", False):
            HTML_TEMPLATE = f"""        
                <p><b> {emd_template.get("ModelName").replace('>', '').replace('<', '')} </b></p>
                <p><b>Model Name:</b> {emd_template.get('ModelParameters', {}).get('model_name')}</p>
                <p><b>Learning Rate:</b> {emd_template.get('LearningRate')}</p>
                {encoded_losses}
            """

        else:
            HTML_TEMPLATE = f"""        
                    <p><b> {emd_template.get("ModelName").replace('>', '').replace('<', '')} </b></p>
                    <p><b>Backbone:</b> {emd_template.get('ModelParameters', {}).get('backbone')}</p>
                    <p><b>Learning Rate:</b> {emd_template.get('LearningRate')}</p>
                    {encoded_losses}
            """

        model_analysis = None
        if confusion_matrix_img:
            model_analysis = f"""
                    <img src="{confusion_matrix_img}" alt="Confusion Matrix">
            """

        if emd_template.get("accuracy"):
            model_analysis = f"""
            <p><b>Accuracy:</b> {emd_template.get('accuracy')}</p>
        """
        if emd_template.get("mIoU"):
            model_analysis = f"""
            <p><b>mIoU:</b> {emd_template.get('mIoU')}</p>
        """

        if emd_template.get("average_precision_score"):
            model_analysis = f"""
            <p><b>Average Precision Score:</b> {emd_template.get('average_precision_score')}</p>
        """

        if emd_template.get("score"):
            model_analysis = f"""
            <p><b>Score:</b> {emd_template.get('score')}</p>
        """

        if emd_template.get("PSNR"):
            model_analysis = f"""
            <p><b>PSNR Metric:</b> {emd_template.get('PSNR')}</p>
            <p><b>SSIM Metric:</b> {emd_template.get('SSIM')}</p>
        """
            # FID is supported for RGB only
            if emd_template.get("FID"):
                model_analysis = (
                    model_analysis
                    + f"""
                <p><b>FID Metric:</b> {emd_template.get('FID')}</p>
                """
                )

        if emd_template.get("per_class_metrics"):
            html_table = pd.read_json(emd_template.get("per_class_metrics")).to_html()
            model_analysis = f"""
            <p><b>Per class metrics:</b> {html_table}</p>
        """
        if emd_template.get("FID_A"):
            model_analysis = f"""
            <p><b>FID A:</b> {emd_template.get('FID_A')}</p>
            <p><b>FID B:</b> {emd_template.get('FID_B')}</p>
        """
        if emd_template.get("panoptic_quality"):
            model_analysis = f"""
            <p><b>Panoptic Quality:</b> {emd_template.get('panoptic_quality')}</p>
        """

        if model_analysis:
            HTML_TEMPLATE += f"""
            <p><b>Analysis of the model</b></p>
            {model_analysis}
        """

        if encoded_showresults:
            HTML_TEMPLATE += f"""
                <p><b>Sample Results</b></p>
                <img src="{encoded_showresults}" alt="Sample Results">
            """

        # For PointCNN and 3d models.
        if iframe_showresults:
            HTML_TEMPLATE += f"""
                <p><b>Sample Results</b><p>
                <iframe src="ModelCharacteristics/show_results.html" style="width:100%;height:70%;" scrolling="no" frameborder="0">
                    </iframe>
            """

        if metrics_html:
            HTML_TEMPLATE += f"""
                <p><b>Metrics per label</b></p>
                {metrics_html}
            """

        if results_html:
            HTML_TEMPLATE += f"""
                <p><b>Sample Results</b></p>
                {results_html}
            """

        file = open(html_file_path, "w")
        file.write(HTML_TEMPLATE)
        file.close()

    def _save(
        self,
        name_or_path,
        framework="PyTorch",
        zip_files=True,
        save_html=True,
        publish=False,
        gis=None,
        compute_metrics=True,
        save_optimizer=False,
        save_inference_file=True,
        **kwargs,
    ):
        save_format = kwargs.get("save_format", "default")  # 'default', 'tflite'
        post_processed = kwargs.get("post_processed", True)  # True, False
        quantized = kwargs.get("quantized", False)  # True, False
        temp = self.learn.path
        temp1 = self.learn.model_dir
        self._framework = framework
        if "\\" in name_or_path or "/" in name_or_path:
            path = Path(name_or_path)
            name = path.parts[-1]
            # to make fastai save to both path and with name
            self.learn.path = path
            self.learn.model_dir = ""
            if not os.path.exists(self.learn.path):
                os.makedirs(self.learn.path)
        else:
            # fixing fastai bug
            # self.learn.path = self.learn.path.parent
            self.learn.model_dir = Path(self.learn.model_dir) / name_or_path
            if not os.path.exists(self.learn.path / self.learn.model_dir):
                os.makedirs(self.learn.path / self.learn.model_dir)
            name = name_or_path

        script_paths = []
        onnx_paths = []
        tflite_paths = []
        try:
            _framework = framework.lower()
            if self._backend == "tensorflow" and _framework == "tflite":
                saved_path = self._save_tflite(
                    name, post_processed=post_processed, quantized=quantized
                )
            else:
                if self._backend != "tensorflow" and _framework == "tflite":
                    supported_models = [
                        "FeatureClassifier",
                    ]
                    if (type(self).__name__) in supported_models:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            tflite_paths = self._save_pytorch_tflite(name)
                    else:
                        raise Exception(
                            "This pytorch model cannot be saved in tflite format"
                        )
                if self._backend == "pytorch" and _framework == "torchscript":
                    supported_models = [
                        "MaskRCNN",
                        "SingleShotDetector",
                        "YOLOv3",
                        "RetinaNet",
                        "SiamMask",
                    ]
                    if (type(self).__name__) in supported_models:
                        if type(self).__name__ != "SiamMask":
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                script_paths = self._save_pytorch_torchscript(name)
                    else:
                        raise Exception(
                            "This pytorch model cannot be saved in torchscript format"
                        )

                if isinstance(self.learn.model, (DistributedDataParallel)):
                    if not int(os.environ.get("RANK", 0)):
                        saved_path = self.learn.save(
                            name, return_path=True, with_opt=save_optimizer
                        )
                    return

                saved_path = self.learn.save(
                    name, return_path=True, with_opt=save_optimizer
                )

            # undoing changes to self.learn.path
        except Exception as e:
            raise e
        finally:
            self.learn.path = temp
            self.framework = framework
            self.learn.model_dir = temp1

        if (type(self).__name__) == "EfficientDet":
            _emd_template = self._create_emd_template(
                saved_path, compute_metrics, save_inference_file
            )
        else:
            _emd_template = self._create_emd_template(
                saved_path.with_suffix(".pth"), compute_metrics, save_inference_file
            )

        if framework.lower() == "tf-onnx":
            batch_size = kwargs.get("batch_size", 16)

            with nostdout():
                self._save_as_tfonnx(saved_path, batch_size)

            self._create_tfonnx_emd_template(
                _emd_template, saved_path.with_suffix(".onnx"), batch_size
            )
            os.remove(saved_path.with_suffix(".pth"))

        if self._backend != "tensorflow" and framework.lower() == "tflite":
            if len(tflite_paths) != 0:
                _script_save_params = {"tf": tflite_paths[0], "sm": tflite_paths[1]}
                _emd_template["TFLite"] = _script_save_params

        # TODO: merge all
        if framework.lower() == "torchscript":
            if len(script_paths) != 0:  # TODO: change_siammask
                _script_save_params = {"GPU": script_paths[1], "CPU": script_paths[0]}
                _emd_template["TorchScript"] = _script_save_params
            else:
                from ._siammask_utils import Custom
                from ._siammask_utils import load_pretrain

                siammask = Custom(anchors=self.anchors)
                if "\\" in name_or_path or "/" in name_or_path:
                    models_path = os.path.join(name_or_path)
                else:
                    models_path = os.path.join(
                        self.learn.path, self.learn.model_dir, name
                    )
                if not os.path.exists(models_path):
                    os.makedirs(models_path)

                siammask = load_pretrain(
                    siammask, os.path.join(models_path, name + ".pth")
                )
                outdir = os.path.join(models_path, "torch_scripts")
                if not os.path.isdir(outdir):
                    os.mkdir(outdir)

                scripted_feature_extractor = torch.jit.script(
                    siammask.features.features
                )
                scripted_feature_extractor.save(
                    os.path.join(outdir, "feature_extractor.pt")
                )

                scripted_feature_downsampler = torch.jit.script(
                    siammask.features.downsample
                )
                scripted_feature_downsampler.save(
                    os.path.join(outdir, "feature_downsampler.pt")
                )

                scripted_rpn_model = torch.jit.script(siammask.rpn_model)
                scripted_rpn_model.save(os.path.join(outdir, "rpn_model.pt"))

                scripted_mask_conv_kernel = torch.jit.script(
                    siammask.mask_model.mask.conv_kernel
                )
                scripted_mask_conv_kernel.save(
                    os.path.join(outdir, "mask_conv_kernel.pt")
                )

                scripted_mask_conv_search = torch.jit.script(
                    siammask.mask_model.mask.conv_search
                )
                scripted_mask_conv_search.save(
                    os.path.join(outdir, "mask_conv_search.pt")
                )

                scripted_mask_depthwise_conv = torch.jit.script(
                    siammask.mask_model.mask.conv2d_dw_group
                )
                scripted_mask_depthwise_conv.save(
                    os.path.join(outdir, "mask_depthwise_conv.pt")
                )

                scripted_refine_model = torch.jit.script(siammask.refine_model)
                scripted_refine_model.save(os.path.join(outdir, "refine_model.pt"))
                temp_emd_template = _emd_template.copy()
                temp_emd_template["ModelFile"] = "."
                temp_emd_template["ModelFiles"] = [
                    "feature_extractor.pt",
                    "feature_downsampler.pt",
                    "rpn_model.pt",
                    "mask_conv_kernel.pt",
                    "mask_conv_search.pt",
                    "mask_depthwise_conv.pt",
                    "refine_model.pt",
                ]

                if os.path.exists(os.path.join(outdir, name + ".emd")):
                    os.remove(os.path.join(outdir, name + ".emd"))

                import zipfile

                dlpk_Name = os.path.join(outdir, name + ".dlpk")
                if os.path.exists(dlpk_Name):
                    os.remove(dlpk_Name)

                out_file = open(os.path.join(outdir, name + ".emd"), "w")
                json.dump(temp_emd_template, out_file, indent=4)
                out_file.close()
                dlpk_Name = os.path.join(outdir, name + ".dlpk")
                f = zipfile.ZipFile(dlpk_Name, "w")
                cwd = os.getcwd()
                os.chdir(outdir)
                for files in temp_emd_template["ModelFiles"]:
                    f.write(files)

                f.write(name + ".emd")
                f.close()
                os.chdir(cwd)

        if _emd_template.get("InferenceFunction", False):
            if (
                _emd_template["ModelType"]
                not in [
                    "ObjectDetection",
                    "ImageClassification",
                    "InstanceDetection",
                    "ObjectClassification",
                    "CycleGAN",
                    "Pix2Pix",
                    "SuperResolution",
                    "ImageCaptioner",
                    "PanopticSegmenter",
                ]
                or save_inference_file
            ):
                inference_file = _emd_template["InferenceFunction"]
                if "[Functions]" in inference_file:
                    inference_file = inference_file[
                        len("[Functions]System\\DeepLearning\\ArcGISLearn\\") :
                    ]
                    _emd_template["InferenceFunction"] = inference_file

                with open(
                    saved_path.parent / _emd_template["InferenceFunction"], "w"
                ) as f:
                    f.write(self._code)
            if not save_inference_file:
                inference_file = _emd_template["InferenceFunction"]
                if "[Functions]" not in inference_file:
                    _emd_template["InferenceFunction"] = (
                        "[Functions]System\\DeepLearning\\ArcGISLearn\\"
                        + _emd_template["InferenceFunction"]
                    )

        ArcGISModel._write_emd(_emd_template, saved_path.with_suffix(".emd"))
        zip_name = saved_path.stem

        if save_html:
            # Backup env var
            bak_IS_ARCGISPRONOTEBOOK = arcgis.learn._utils.env._IS_ARCGISPRONOTEBOOK
            arcgis.learn._utils.env.switch = False
            try:
                # Do not call plt.show()
                arcgis.learn._utils.env._IS_ARCGISPRONOTEBOOK = False
                #
                self._save_model_characteristics(
                    saved_path.parent.absolute() / model_characteristics_folder
                )
                ArcGISModel._create_html(saved_path)
            except:
                pass
            finally:
                # Restore env var
                arcgis.learn._utils.env._IS_ARCGISPRONOTEBOOK = bak_IS_ARCGISPRONOTEBOOK
                is_arcgispronotebook()

        if _emd_template.get("ModelConfigurationFile", False):
            with open(
                saved_path.parent / _emd_template["ModelConfigurationFile"], "w"
            ) as f:
                f.write(inspect.getsource(self._model_conf_class))

        if zip_files:
            _create_zip(str(zip_name), str(saved_path.parent))

        if arcgis.env.verbose:
            print("Created model files at {spp}".format(spp=saved_path.parent))

        if publish:
            self._publish_dlpk(
                (saved_path.parent / os.path.basename(saved_path)).with_suffix(".dlpk"),
                gis=gis,
                overwrite=kwargs.get("overwrite", False),
            )

        return saved_path.parent

    def _save_tflite(self, name, post_processed=True, quantized=False):
        if post_processed or quantized:
            input_normalization = quantized is False
            return self.learn._save_tflite(
                name,
                return_path=True,
                model_to_save=self._get_post_processed_model(
                    input_normalization=input_normalization
                ),
                quantized=quantized,
                data=self._data,
            )
        return self.learn._save_tflite(name)

    def _save_pytorch_tflite(self, name):
        pass

    def _script(self, model, inp):
        scripted_model = torch.jit.script(model, inp)
        scripted_model.eval()
        return scripted_model

    def _trace(self, model, inp, check_trace=False):
        traced_model = torch.jit.trace(model, inp, check_trace=check_trace)
        traced_model.eval()
        return traced_model

    def _save_pytorch_torchscript(self, name):
        pass

    def _get_post_processed_model(self, input_normalization=True):
        return get_post_processed_model(self, input_normalization=input_normalization)

    def _save_model_characteristics(self, model_characteristics_dir):
        import shutil
        import matplotlib.pyplot as plt

        if isinstance(self._data, _EmptyData) or getattr(
            self._data, "_is_empty", False
        ):
            if not os.path.exists(
                os.path.join(self._data.emd_path.parent, model_characteristics_folder)
            ):
                return
            temp_path = tempfile.NamedTemporaryFile().name
            shutil.copytree(
                os.path.join(self._data.emd_path.parent, model_characteristics_folder),
                temp_path,
            )
            if os.path.exists(
                os.path.join(model_characteristics_dir, model_characteristics_dir)
            ):
                shutil.rmtree(
                    os.path.join(model_characteristics_dir, model_characteristics_dir),
                    ignore_errors=True,
                )

            shutil.copytree(
                temp_path,
                os.path.join(model_characteristics_dir, model_characteristics_dir),
            )

            return

        if not os.path.exists(
            os.path.join(model_characteristics_dir, model_characteristics_dir)
        ):
            os.mkdir(os.path.join(model_characteristics_dir, model_characteristics_dir))

        if hasattr(self.learn, "recorder"):
            try:
                self.learn.recorder.plot_losses()
                plt.savefig(os.path.join(model_characteristics_dir, "loss_graph.png"))
                plt.close()
            except:
                plt.close()

        if self.__str__() in [
            "<PointCNN>",
            "<RandLANet>",
            "<SQNSeg>",
            "<MMDetection3D>",
        ]:
            self.show_results(save_html=True, save_path=model_characteristics_dir)
        elif self.__str__() in [
            "<TextClassifier>",
            "<TransformerEntityRecognizer>",
            "<SequenceToSequence>",
            "<TimeSeriesModel>",
        ]:
            pass
        elif hasattr(self, "show_results"):
            self.show_results()
            plt.savefig(os.path.join(model_characteristics_dir, "show_results.png"))
            plt.close()

        if hasattr(self, "_save_confusion_matrix"):
            self._save_confusion_matrix(model_characteristics_dir)

    def _publish_dlpk(self, dlpk_path, gis=None, overwrite=False):
        gis_user = arcgis.env.active_gis if gis is None else gis
        if not gis_user:
            warn("No active gis user found!")
            return

        if not os.path.exists(dlpk_path):
            warn("DLPK file not found!")
            return

        emd_path = os.path.join(dlpk_path.parent, dlpk_path.stem + ".emd")

        if not os.path.exists(emd_path):
            warn("EMD File not found!")
            return

        emd_data = json.load(open(emd_path, "r"))
        formatted_description = f"""
                <p><b> {emd_data.get('ModelName').replace('>', '').replace('<', '')} </b></p>
                <p><b>Backbone:</b> {emd_data.get('ModelParameters', {}).get('backbone')}</p>
                <p><b>Learning Rate:</b> {emd_data.get('LearningRate')}</p>
        """

        if emd_data.get("accuracy"):
            formatted_description = (
                formatted_description
                + f"""
                <p><b>Analysis of the model</b></p>
                <p><b>Accuracy:</b> {emd_data.get('accuracy')}</p>
            """
            )

        if emd_data.get("average_precision_score"):
            formatted_description = (
                formatted_description
                + f"""
                <p><b>Analysis of the model</b></p>
                <p><b>Average Precision Score:</b> {emd_data.get('average_precision_score')}</p>
            """
            )

        item = gis_user.content.add(
            {
                "type": "Deep Learning Package",
                "description": formatted_description,
                "title": dlpk_path.stem,
                "overwrite": "true" if overwrite else "false",
            },
            data=str(dlpk_path.absolute()),
        )

        print(f"Published DLPK Item Id: {item.itemid}")

        model_characteristics_dir = os.path.join(
            dlpk_path.parent.absolute(), model_characteristics_folder
        )
        screenshots = [
            os.path.join(model_characteristics_dir, screenshot)
            for screenshot in os.listdir(model_characteristics_dir)
        ]

        item.update(item_properties={"screenshots": screenshots})

    def _create_tfonnx_emd_template(self, _emd_template, saved_path, batch_size):
        _emd_template.update(self._get_tfonnx_emd_params())
        _emd_template["BatchSize"] = batch_size
        _emd_template["ModelFile"] = saved_path.name

        return _emd_template

    def _get_tfonnx_emd_params(self):
        # Raises error if framework specified is TF-ONNX but is not supported by the model
        raise NotImplementedError(
            "TF-ONNX framework is currently not supported by this model."
        )

    def _save_as_tfonnx(self, saved_path, batch_size):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import onnx
                from onnx_tf.backend import prepare
        except:
            raise Exception(
                "Could not find the required deep learning dependencies. Ensure you have installed the required dependent libraries. See https://developers.arcgis.com/python/guide/deep-learning/."
            )

        batch_size = int(math.sqrt(int(batch_size))) ** 2
        dummy_input = torch.randn(
            batch_size,
            3,
            self._data.chip_size,
            self._data.chip_size,
            device=self._device,
            requires_grad=True,
        )
        torch.onnx.export(
            self.learn.model, dummy_input, saved_path.with_suffix(".onnx")
        )

    def save(
        self,
        name_or_path,
        framework="PyTorch",
        publish=False,
        gis=None,
        compute_metrics=True,
        save_optimizer=False,
        save_inference_file=True,
        **kwargs,
    ):
        """
        Saves the model weights, creates an Esri Model Definition and Deep
        Learning Package zip for deployment to Image Server or ArcGIS Pro.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Name of the model to save. It
                                stores it at the pre-defined location. If path
                                is passed then it stores at the specified path
                                with model name as directory name and creates
                                all the intermediate directories.
        ---------------------   -------------------------------------------
        framework               Optional string. Exports the model in the
                                specified framework format ('PyTorch', 'tflite'
                                'torchscript', and 'TF-ONXX' (deprecated)).
                                Only models saved with the default framework
                                (PyTorch) can be loaded using `from_model`.
                                ``tflite`` framework (experimental support) is
                                supported by :class:`~arcgis.learn.SingleShotDetector`
                                - tensorflow backend only,
                                :class:`~arcgis.learn.FeatureClassifier`(not supported
                                for transformer backbones) and
                                :class:`~arcgis.learn.RetinaNet` - tensorflow
                                backend only.``torchscript`` format is supported by
                                :class:`~arcgis.learn.SiamMask`,
                                :class:`~arcgis.learn.MaskRCNN`,
                                :class:`~arcgis.learn.SingleShotDetector`,
                                :class:`~arcgis.learn.YOLOv3` and
                                :class:`~arcgis.learn.RetinaNet`.
                                For usage of SiamMask model in ArcGIS Pro >= 2.8,
                                load the ``PyTorch`` framework saved model
                                and export it with ``torchscript`` framework
                                using ArcGIS API for Python >= v1.8.5.
                                For usage of SiamMask model in ArcGIS Pro 2.9,
                                set framework to ``torchscript`` and use the
                                model files additionally generated inside
                                'torch_scripts' folder.
                                If framework is ``TF-ONNX`` (Only supported for
                                :class:`~arcgis.learn.SingleShotDetector`),
                                ``batch_size`` can be passed as an optional
                                keyword argument.
        ---------------------   -------------------------------------------
        publish                 Optional boolean. Publishes the DLPK as an item.
        ---------------------   -------------------------------------------
        gis                     Optional :class:`~arcgis.gis.GIS`  Object.
                                Used for publishing the item. If not specified
                                then active gis user is taken.
        ---------------------   -------------------------------------------
        compute_metrics         Optional boolean. Used for computing model
                                metrics.
        ---------------------   -------------------------------------------
        save_optimizer          Optional boolean. Used for saving the model-optimizer
                                state along with the model. Default is set to False
        ---------------------   -------------------------------------------
        save_inference_file     Optional boolean. Used for saving the inference file
                                along with the model.
                                If False, the model will not work with ArcGIS Pro 2.6
                                or earlier. Default is set to True.
        ---------------------   -------------------------------------------
        kwargs                  Optional Parameters:
                                Boolean `overwrite` if True, it will overwrite
                                the item on ArcGIS Online/Enterprise, default False.
        =====================   ===========================================
        """
        if int(os.environ.get("RANK", 0)):
            return
        return self._save(
            name_or_path,
            framework=framework,
            publish=publish,
            gis=gis,
            compute_metrics=compute_metrics,
            save_optimizer=save_optimizer,
            save_inference_file=save_inference_file,
            **kwargs,
        )

    def load(self, name_or_path, **kwargs):
        """
        Loads a compatible saved model for inferencing or fine tuning from the disk.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Name or Path to
                                Deep Learning Package (DLPK) or
                                Esri Model Definition(EMD) file.
        =====================   ===========================================
        """
        temp = self.learn.path
        if "\\" in name_or_path or "/" in name_or_path:
            path = Path(name_or_path)
            # to make fastai from both path and with name
            if path.is_file():
                name = path.stem
                self.learn.path = path.parent
            else:
                name = path.parts[-1]
                self.learn.path = path
            self.learn.model_dir = ""
        else:
            # fixing fastai bug
            # self.learn.path = self.learn.path.parent
            self.learn.model_dir = Path(self.learn.model_dir) / name_or_path
            name = name_or_path

        try:
            device = getattr(self, "_map_location", None)
            self.learn.load(name, purge=False, device=device)
        except Exception as e:
            raise e
        finally:
            # undoing changes to self.learn.path
            self.learn.path = temp
            self.learn.model_dir = "models"
