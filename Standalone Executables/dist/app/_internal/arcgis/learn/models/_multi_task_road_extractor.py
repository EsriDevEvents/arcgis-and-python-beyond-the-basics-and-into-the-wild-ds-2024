import json
import logging
import random
import traceback
from functools import partial
from pathlib import Path
import glob
import warnings
import os

from arcgis.learn._data import _raise_fastai_import_error
from ._arcgis_model import ArcGISModel
from ._codetemplate import image_classifier_prf

HAS_OPENCV = True
HAS_FASTAI = True
HAS_ARCPY = True

# Try to import the necessary modules
# Exception will turn the HAS_FASTAI flag to false
# so that relevant exception can be raised
try:
    import numpy as np
    import torch
    from torch import optim
    from torchvision import models
    from arcgis.learn.models._linknet_utils import (
        LinkNetMultiTaskModel,
        accuracy,
        miou,
        road_orient_loss,
    )
    from arcgis.learn.models._linknet_utils import compute_miou
    from arcgis.learn.models._hourglass_utils import StackHourglassMultiTaskModel
    from ._road_mtl_learner import MultiTaskRoadLearner
    from ._unet_utils import show_results_multispectral
    from arcgis.learn.models._arcgis_model import _resnet_family, _EmptyData
    from arcgis.learn._utils.common import (
        get_multispectral_data_params_from_emd,
        _get_emd_path,
    )
    from arcgis.learn._utils.segmentation_loss_functions import dice
    from arcgis.learn.models._arcgis_model import _device_check
    from ._arcgis_model import _set_multigpu_callback
    from ._timm_utils import filter_timm_models, get_backbone
    from .._data_utils._pixel_classifier_data import ClassifiedTilesData
    from .._data_utils._road_orient_data import RoadOrientation
    from .._utils.env import is_arcgispronotebook
    from matplotlib import pyplot as plt
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

logger = logging.getLogger()


def safe_json(data):
    """
    Check if data is JSON Serializable.
    It can be used while saving emd_parameters, as it dumps
    the data using JSON which requires JSON serializanble data only.
    """
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


class MultiTaskRoadExtractor(ArcGISModel):
    """
    Creates a Multi-Task Learning model for binary segmentation of roads. Supports RGB
    and Multispectral Imagery.
    Implementation based on https://doi.org/10.1109/CVPR.2019.01063 .

    =====================   =====================================================
    **Parameter**            **Description**
    ---------------------   -----------------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -----------------------------------------------------
    backbone                Optional String. Backbone convolutional neural network
                            model used for feature extraction. If hourglass is chosen as
                            the mtl_model (Architecture), then this parameter is
                            ignored as hourglass uses a special customised
                            architecture.
                            This parameter is used with `linknet` model.
                            Default: 'resnet34'
                            Supported backbones: ResNet family and specified Timm
                            models(experimental support) from :func:`~arcgis.learn.MultiTaskRoadExtractor.backbones`.
    ---------------------   -----------------------------------------------------
    pretrained_path         Optional String. Path where a compatible pre-trained
                            model is saved. Accepts a Deep Learning Package
                            (DLPK) or Esri Model Definition(EMD) file.
    =====================   =====================================================

    **kwargs**

    =============================   =============================================
    **Parameter**                    **Description**
    -----------------------------   ---------------------------------------------
    mtl_model                       Optional String. It is used to create model
                                    from linknet or
                                    hourglass based neural architectures.
                                    Supported: 'linknet', 'hourglass'.
                                    Default: 'hourglass'
    -----------------------------   ---------------------------------------------
    gaussian_thresh                 Optional float. Sets the gaussian threshold
                                    which allows to set the required road width.
                                    Range: 0.0 to 1.0
                                    Default: 0.76
    -----------------------------   ---------------------------------------------
    orient_bin_size                 Optional Int. Sets the bin size for
                                    orientation angles.
                                    Default: 20
    -----------------------------   ---------------------------------------------
    orient_theta                    Optional Int. Sets the width of orientation
                                    mask.
                                    Default: 8
    =============================   =============================================

    :return: :class:`~arcgis.learn.MultiTaskRoadExtractor` Object
    """

    def __init__(
        self,
        data,
        backbone=None,
        pretrained_path=None,
        *args,
        **kwargs,
    ):
        # if data.sub_dataset_type != "RoadOrientation":
        #    raise Exception(
        #        "This model only works for road extraction. And In order to use this model it's corresponding model-specific parameters should also be passed at prepare_data."
        #    )
        # Set default backbone to be 'resnet34'

        if pretrained_path is not None:
            pretrained_backbone = False
        else:
            pretrained_backbone = True

        self._validate_kwargs(**kwargs)
        # if backbone is None:
        #    backbone = models.resnet34
        super().__init__(data, backbone, pretrained_path=pretrained_path, **kwargs)
        self._slice_lr = False  # Road models just have a single layer group due to which we cant slice the lr.
        predefined_mtl_model = None
        # Causes Divide by zero error in  fastai library
        if hasattr(
            self._data, "orig_path"
        ):  # If true the data is not empty class obtained from from_model
            if pretrained_path is not None:
                if os.path.isdir(pretrained_path):
                    for file in os.listdir(pretrained_path):
                        if file.endswith(".emd"):
                            try:
                                emd = json.load(
                                    open(os.path.join(pretrained_path, file))
                                )
                                bin_size = emd["RoadOrientation"]["orient_bin_size"]
                                predefined_mtl_model = emd["ModelParameters"][
                                    "mtl_model"
                                ]
                            except Exception as e:
                                raise e
                        elif file.endswith(".dlpk"):
                            try:
                                emd_path = _get_emd_path(Path(file))
                                with open(emd_path) as f:
                                    emd = json.load(f)
                                bin_size = emd["RoadOrientation"]["orient_bin_size"]
                                predefined_mtl_model = emd["ModelParameters"][
                                    "mtl_model"
                                ]
                            except Exception as e:
                                raise e
                else:
                    try:
                        emd_path = _get_emd_path(pretrained_path)
                        with open(emd_path) as f:
                            emd = json.load(f)
                        bin_size = emd["RoadOrientation"]["orient_bin_size"]
                        predefined_mtl_model = emd["ModelParameters"]["mtl_model"]
                    except Exception as e:
                        raise e
                try:
                    bin_size
                except:
                    print(
                        "Could not find the emd file in the specified path. Please provide correct path"
                    )

                kwargs[
                    "orient_bin_size"
                ] = bin_size  # To ensure the data is consistent across multiple runs

            self._orient_data = self._get_road_orient_data(data, **kwargs)
            if len(data.classes) > 2:
                raise Exception(
                    "Found multi-labels in the data, This is a binary segmentation model and hence please export the data with binary labels."
                    # noqa
                )
            self._data.classes = self._orient_data.classes
            class_key = (
                self._orient_data.classes[0]
                if isinstance(self._orient_data.classes[0], int)
                else self._orient_data.classes[1]
            )
            self._data.class_mapping = {class_key: str(class_key)}
            self._orient_data.class_mapping = self._data.class_mapping
            self._orient_data._imagery_type = (
                self._data._imagery_type if hasattr(data, "_imagery_type") else None
            )
            if self._data._imagery_type == "MS":
                self._orient_data._image_space_used = (
                    self._data._image_space_used
                    if hasattr(data, "_image_space_used")
                    else None
                )
                self._orient_data._is_multispectral = (
                    self._data._is_multispectral
                    if hasattr(data, "_is_multispectral")
                    else None
                )
                self._orient_data._band_max_values = (
                    self._data._band_max_values
                    if hasattr(data, "_band_max_values")
                    else None
                )
                self._orient_data._band_mean_values = (
                    self._data._band_mean_values
                    if hasattr(data, "_band_mean_values")
                    else None
                )
                self._orient_data._band_min_values = (
                    self._data._band_min_values
                    if hasattr(data, "_band_min_values")
                    else None
                )
                self._orient_data._band_std_values = (
                    self._data._band_std_values
                    if hasattr(data, "_band_std_values")
                    else None
                )
                self._orient_data._bands = (
                    self._data._bands if hasattr(data, "_bands") else None
                )
                self._orient_data._extract_bands = (
                    self._data._extract_bands
                    if hasattr(data, "_extract_bands")
                    else None
                )
                self._orient_data._image_space_used = (
                    self._data._image_space_used
                    if hasattr(data, "_image_space_used")
                    else None
                )
                self._orient_data._min_max_scaler = (
                    self._data._min_max_scaler
                    if hasattr(data, "_min_max_scaler")
                    else None
                )
                self._orient_data._min_max_scaler_tfm = (
                    self._data._min_max_scaler_tfm
                    if hasattr(data, "_min_max_scaler_tfm")
                    else None
                )
                self._orient_data._multispectral_color_array = (
                    self._data._multispectral_color_array * 255
                    if hasattr(data, "_multispectral_color_array")
                    else None
                )
                self._orient_data._multispectral_color_mapping = (
                    self._data._multispectral_color_mapping
                    if hasattr(data, "_multispectral_color_mapping")
                    else None
                )
                self._orient_data._norm_pct = (
                    self._data._norm_pct if hasattr(data, "_norm_pct") else None
                )
                self._orient_data._rgb_bands = (
                    self._data._rgb_bands if hasattr(data, "_rgb_bands") else None
                )
                self._orient_data._scaled_max_values = (
                    self._data._scaled_max_values
                    if hasattr(data, "_scaled_max_values")
                    else None
                )
                self._orient_data._scaled_mean_values = (
                    self._data._scaled_mean_values
                    if hasattr(data, "_scaled_mean_values")
                    else None
                )
                self._orient_data._scaled_min_values = (
                    self._data._scaled_min_values
                    if hasattr(data, "_scaled_min_values")
                    else None
                )
                self._orient_data._scaled_std_values = (
                    self._data._scaled_std_values
                    if hasattr(data, "_scaled_std_values")
                    else None
                )
                self._orient_data._symbology_rgb_bands = (
                    self._data._symbology_rgb_bands
                    if hasattr(data, "_symbology_rgb_bands")
                    else None
                )
                self._orient_data.stats = (
                    self._data.stats if hasattr(data, "stats") else None
                )
                self._orient_data._do_normalize = (
                    self._data._do_normalize if hasattr(data, "_do_normalize") else None
                )
                self._orient_data.train_ds.tfms = (
                    self._data.train_ds.tfms if hasattr(data.train_ds, "tfms") else None
                )
                self._orient_data.valid_ds.tfms = (
                    self._data.valid_ds.tfms if hasattr(data.valid_ds, "tfms") else None
                )
                if self._orient_data._do_normalize:
                    self._orient_data = self._orient_data.normalize(
                        stats=(
                            self._orient_data._scaled_mean_values,
                            self._orient_data._scaled_std_values,
                        ),
                        do_x=True,
                        do_y=False,
                    )
                # self._data = self._orient_data
                self.show_results = self._show_results_multispectral
        else:
            self._orient_data = (
                self._data
            )  # Else use the data attributes obtained from emd
            # if self._orient_data._do_normalize:
            #    self._orient_data = self._orient_data.normalize(stats=(self._orient_data._scaled_mean_values, self._orient_data._scaled_std_values), do_x=True, do_y=False)
        # self._orient_data.train_ds.x._imagery_type = self._orient_data._imagery_type
        # self._orient_data.valid_ds.x._imagery_type = self._orient_data._imagery_type
        self._ignore_classes = kwargs.get("ignore_classes", [])
        if self._ignore_classes != [] and len(self._orient_data.classes) <= 3:
            raise Exception(
                "`ignore_classes` parameter can only be used when the dataset has more than 2 classes."  # noqa
            )

        data_classes = list(self._orient_data.class_mapping.keys())
        if 0 not in list(self._orient_data.class_mapping.values()):
            self._ignore_mapped_class = [
                data_classes.index(k) + 1 for k in self._ignore_classes if k != 0
            ]
        else:
            self._ignore_mapped_class = [
                data_classes.index(k) + 1 for k in self._ignore_classes
            ]
        if self._ignore_classes != []:
            if 0 not in self._ignore_mapped_class:
                self._ignore_mapped_class.insert(0, 0)

        pixel_accuracy = partial(
            accuracy, ignore_mapped_class=self._ignore_mapped_class
        )
        road_iou = partial(miou, ignore_mapped_class=self._ignore_mapped_class)
        dice_coeff = partial(dice, ignore_classes=[[0]])

        n_bands = len(getattr(self._data, "_extract_bands", [0, 1, 2]))
        _backbone = self._backbone
        if hasattr(self, "_orig_backbone"):
            _backbone = self._orig_backbone

        # Check if a backbone provided is compatible, use resnet50 as default
        if not self._check_backbone_support(_backbone):
            raise Exception(
                f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"  # noqa
            )

        self.name = "MultiTaskRoadExtractor"
        self._code = image_classifier_prf

        self._chip_size = (self._orient_data.chip_size, self._orient_data.chip_size)

        # Cut-off the backbone before the penultimate layer
        self._encoder = get_backbone(self._backbone, pretrained_backbone)

        # Initialize the model, loss function and the Learner object
        mtl_models = {
            "linknet": LinkNetMultiTaskModel,
            "hourglass": StackHourglassMultiTaskModel,
        }
        if predefined_mtl_model:
            self._mtl_model = predefined_mtl_model
        else:
            self._mtl_model = kwargs.get("mtl_model", "hourglass")
            self._mtl_model = (
                self._mtl_model if self._mtl_model in mtl_models.keys() else "hourglass"
            )

        if self._mtl_model == "hourglass" and backbone is not None:
            logger.warning(
                "The Hourglass model does not support user specified backbones. The backbone parameter will be ignored."
            )

        self._model_init_kwargs = kwargs.get("model_init_kwargs", {})

        if "timm" in self._backbone.__module__ and self._mtl_model == "linknet":
            self._model_init_kwargs["is_timm"] = True

        self._model = mtl_models[self._mtl_model](
            self._encoder,
            task1_classes=self._orient_data.c,
            task2_classes=self._orient_data.orient_c,
            chip_size=self._chip_size,
            n_bands=n_bands,
            **self._model_init_kwargs,
        )
        self._loss_f = road_orient_loss(
            n_classes=self._orient_data.c,
            loss_weights=kwargs.get("loss_weights", (1.0, 1.0)),
        )
        self.learner_params = kwargs.get("learner_params", {})
        learner_kwargs = self.learner_params.copy()
        self._opt_func_name = kwargs.get("opt_func", None)
        if self._opt_func_name:
            _opt_func = self._get_optimizer(self._opt_func_name)
            if _opt_func:
                self._opt_func_args = kwargs.get("opt_func_args", {})
                learner_kwargs["opt_func"] = partial(_opt_func, **self._opt_func_args)

        self.learn = MultiTaskRoadLearner(
            self._orient_data,
            self._model,
            loss_func=self._loss_f,
            metrics=[pixel_accuracy, road_iou, dice_coeff],
            **learner_kwargs,
        )
        if hasattr(self._data, "path"):
            self.learn.path = self._data.path
        self.learn.model = self.learn.model.to(self._device)
        _set_multigpu_callback(self)
        if pretrained_path is not None:
            super().load(str(pretrained_path))
        self._arcgis_init_callback()  # make first conv weights learnable

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy", "miou", "dice"]

    def mIOU(self, mean=False, show_progress=True):
        """
        Computes mean IOU on the validation set for each class.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        mean                    Optional bool. If False returns class-wise
                                mean IOU, otherwise returns mean iou of all
                                classes combined.
        ---------------------   -------------------------------------------
        show_progress           Optional bool. Displays the prgress bar if
                                True.
        =====================   ===========================================

        :return: `dict` if mean is False otherwise `float`
        """
        # self._check_requisites()
        if hasattr(self.learn.data, "emd") and (self._learning_rate is None):
            return self.learn.data.emd[
                "mIoU"
            ]  # if model is loaded without data then reads the miou value from emd
        num_classes = torch.arange(self._orient_data.c)
        miou = compute_miou(
            self,
            self._orient_data.valid_dl,
            mean,
            num_classes,
            show_progress,
            self._ignore_mapped_class,
        )
        if mean:
            miou = [
                miou[i] for i in range(len(miou)) if i not in self._ignore_mapped_class
            ]
            return np.mean(miou)
        if self._ignore_mapped_class == []:
            return dict(zip(["0"] + self._orient_data.classes[1:], miou))
        else:
            class_values = [0] + list(self._orient_data.class_mapping.keys())
            return {
                class_values[i]: miou[i]
                for i in range(len(miou))
                if i not in self._ignore_mapped_class
            }

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    def _get_road_orient_data(self, data, **kwargs):
        road_data_obj = ClassifiedTilesData(
            path=self._data.orig_path,
            class_mapping={},
            chip_size=self._data.chip_size,
            val_split_pct=self._data._val_split_pct,
            batch_size=self._data.batch_size,
            transforms=None,
            seed=42,
            dataset_type=self._data.dataset_type,
            resize_to=self._data.resize_to,
            **kwargs,
        )
        road_orient_obj = RoadOrientation(self._data, road_data_obj, **kwargs)
        orient_data = road_orient_obj.get_databunch(data, **kwargs)
        return orient_data

    def _get_optimizer(self, optim_name):
        optim_funcs = {"sgd": optim.SGD}
        return optim_funcs.get(optim_name, None)

    def _validate_kwargs(self, **kwargs):
        gauss_thresh = kwargs.get("gaussian_thresh", 0.6)
        orient_bin_size = kwargs.get("orient_bin_size", 20)
        orient_theta = kwargs.get("orient_theta", 8)
        assert (
            gauss_thresh > 0.0 and gauss_thresh < 1
        ), "gauss_thresh should be between 0 and 1"
        assert (
            orient_bin_size > 0 and orient_bin_size < 360
        ), "orient_bin_size should be between 1 and 360"
        assert (
            orient_theta > 0 and orient_theta < 10
        ), "orient_theta should be between 1 and 10"

    def unfreeze(self):
        """
        Unfreezes the earlier layers of the model for fine-tuning.
        """
        for _, param in self.learn.model.named_parameters():
            param.requires_grad = True

    # Return a list of supported backbones names
    @property
    def supported_backbones(self):
        """Supported list of backbones for this model."""
        return MultiTaskRoadExtractor._supported_backbones()

    @staticmethod
    def backbones():
        """Supported list of backbones for this model."""
        return MultiTaskRoadExtractor._supported_backbones()

    @staticmethod
    def _supported_backbones():
        timm_models = filter_timm_models(
            [
                "*dpn*",
                "*inception*",
                "*nasnet*",
                "*tf_efficientnet_cc*",
                "*repvgg*",
                "*resnetblur*",
                "*selecsls*",
                "*tresnet*",
                "*hrnet*",
                "*rexnet*",
                "*mixnet*",
            ]
        )
        timm_backbones = list(map(lambda m: "timm:" + m, timm_models))
        return [*_resnet_family] + timm_backbones

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return MultiTaskRoadExtractor._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Classified_Tiles"]

    def fit(self, epochs=10, lr=None, **kwargs):
        if isinstance(lr, slice):
            lr = lr.stop
        # setting monitor because earlystopping also uses this value.
        super().fit(epochs, lr=lr, monitor=kwargs.pop("monitor", "miou"), **kwargs)

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISImageClassifier.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISImageClassifier.py"
        _emd_template["ModelConfiguration"] = "_road_infrencing"
        _emd_template["ModelType"] = "ImageClassification"
        _emd_template["ExtractBands"] = [0, 1, 2]

        _emd_template["SupportsVariableTileSize"] = True

        model_params = {
            "backbone": self._backbone,
            "backend": self._backend,
            "opt_func": self._opt_func_name if hasattr(self, "_opt_func_name") else "",
            "opt_func_args": self._opt_func_args
            if hasattr(self, "_opt_func_args")
            else "",
            "loss": self._loss_f.__class__.__name__,
            "loss_weights": self._loss_f.__dict__["loss_weights"],
            "mtl_model": self._mtl_model,
            "model_init_kwargs": self._model_init_kwargs,
            "learner_params": self.learner_params,
        }
        _emd_template["ModelParameters"] = model_params
        _emd_template["Classes"] = []

        class_data = {}
        # 0th index is background
        for _, class_name in enumerate(self._data.classes[1:]):
            inverse_class_mapping = {
                int(v): k for k, v in self._data.class_mapping.items()
            }
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)]
            class_data["Color"] = color
            _emd_template["Classes"].append(class_data.copy())

        if hasattr(self._orient_data, "parent_obj"):
            data_params = {}
            for param_key, param_value in self._orient_data.parent_obj.__dict__.items():
                if "files" in param_key:
                    continue
                if hasattr(param_value, "__dict__"):
                    continue
                if not safe_json(param_value):
                    param_value = str(param_value)
                data_params[f"{param_key}"] = param_value
            _emd_template[self._orient_data.parent_obj.__class__.__name__] = data_params

        return _emd_template

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a Multi-Task Learning model for binary segmentation from a
        Deep Learning Package(DLPK) or Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data` function or None for
                                inferencing.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.MultiTaskRoadExtractor` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)
        model_file = Path(emd["ModelFile"])
        chip_size = emd["ImageWidth"]
        model_params = emd["ModelParameters"]
        orient_c = int(360.0 / emd["RoadOrientation"]["orient_bin_size"]) + 1

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}

        if data is None:
            force_cpu = _device_check()
            data = _EmptyData(
                path=emd_path.parent.parent,
                loss_func=None,
                c=len(class_mapping) + 1,
                chip_size=chip_size,
            )
            data.orient_c = orient_c
            data.class_mapping = class_mapping
            data._is_empty = True
            data = get_multispectral_data_params_from_emd(data, emd)
            data.emd_path = emd_path
            data.emd = emd

        return cls(data, **model_params, pretrained_path=emd_path)

    def show_results(self, rows=2, **kwargs):
        """
        Shows the ground truth and predictions of model side by side.

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Number of rows of data to be displayed, if
                                batch size is smaller, then the rows will
                                display the value provided for batch size.
        ---------------------   -------------------------------------------
        alpha                   Optional Float. Opacity parameter for label
                                overlay on image. Float [0.0 - 1.0]
                                Default: 0.6
        =====================   ===========================================

        """
        self._check_requisites()
        self.return_fig = kwargs.get("return_fig", False)
        fig = self.learn.show_results(rows=rows, **kwargs)
        if is_arcgispronotebook():
            plt.show()
        if self.return_fig:
            return fig

    def _show_results_multispectral(
        self, rows=5, alpha=0.7, **kwargs
    ):  # parameters adjusted in kwargs
        """
        Shows the ground truth and predictions of model side by side.

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Number of rows of data to be displayed, if
                                batch size is smaller, then the rows will
                                display the value provided for batch size.
        ---------------------   -------------------------------------------
        alpha                   Optional Float. Opacity parameter for label
                                overlay on image. Float [0.0 - 1.0]
                                Default: 0.7
        =====================   ===========================================

        """
        return_fig = kwargs.get("return_fig", False)
        ret_val = show_results_multispectral(self, nrows=rows, alpha=alpha, **kwargs)
        if return_fig:
            fig, ax = ret_val
            return fig

    @property
    def _model_metrics(self):
        acc, mean_iou = self._get_model_metrics()
        return {
            "accuracy": "{}".format(acc),
            "mIoU": "{}".format(mean_iou),
        }

    def _get_model_metrics(self, **kwargs):
        checkpoint = getattr(self, "_is_checkpointed", False)
        if not hasattr(self.learn, "recorder"):
            return 0.0, 0.0

        if len(self.learn.recorder.metrics) == 0:
            return 0.0, 0.0

        try:
            model_accuracy = self.learn.recorder.metrics[-1][0]
            model_iou = self.learn.recorder.metrics[-1][1]
            if checkpoint:
                metrics_array = np.array(self.learn.recorder.metrics)
                model_iou = np.max(metrics_array[:, 1])
                model_accuracy = metrics_array[np.argmax(metrics_array[:, 1]), 0]
        except BaseException:
            logger.debug("Cannot retrieve model accuracy.")
            model_accuracy = 0.0

        return float(model_accuracy), float(model_iou)
