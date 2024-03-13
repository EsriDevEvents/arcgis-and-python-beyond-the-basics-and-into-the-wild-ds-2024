import json
import logging
import random
import traceback
from functools import partial
from pathlib import Path

from .._data import _raise_fastai_import_error
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
    from fastai.vision.learner import create_body
    from torch import optim
    from torchvision import models
    from fastai.basic_train import Learner
    from arcgis.learn.models._linknet_utils import (
        LinkNetModel,
        accuracy,
        miou,
    )
    from fastai.basic_data import DatasetType
    from arcgis.learn._utils.segmentation_loss_functions import mIoULoss
    from arcgis.learn.models._arcgis_model import _resnet_family, _EmptyData
    from arcgis.learn._utils.common import (
        get_multispectral_data_params_from_emd,
        safe_json,
    )
    from fastai.callback import annealing_no
    from fastai.callbacks.general_sched import TrainingPhase, GeneralScheduler
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class LinkNet(ArcGISModel):
    """
    Model architecture from https://arxiv.org/pdf/1707.03718.pdf.
    Creates a LinkNet Image Segmentation / Pixel Classification model.
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used for
                            creating the base of the `LinkNet`, which
                            is `resnet34` by default.
                            Compatible backbones: 'resnet18', 'resnet34',
                            'resnet50', 'resnet101', 'resnet152'
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================
    :return: `LinkNet` Object
    """

    def __init__(
        self,
        data,
        backbone=None,
        pretrained_path=None,
        *args,
        **kwargs,
    ):
        # Set default backbone to be 'resnet34'
        if backbone is None:
            backbone = models.resnet34
        super().__init__(data, backbone, **kwargs)

        self._ignore_classes = kwargs.get("ignore_classes", [])
        if self._ignore_classes != [] and len(self._data.classes) <= 3:
            raise Exception(
                "`ignore_classes` parameter can only be used when the dataset has more than 2 classes."  # noqa
            )

        data_classes = list(self._data.class_mapping.keys())
        if 0 not in list(data.class_mapping.values()):
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
        mean_iou = partial(miou, ignore_mapped_class=self._ignore_mapped_class)

        n_bands = len(getattr(self._data, "_extract_bands", [0, 1, 2]))
        _backbone = self._backbone
        if hasattr(self, "_orig_backbone"):
            _backbone = self._orig_backbone

        # Check if a backbone provided is compatible, use resnet50 as default
        if not self._check_backbone_support(_backbone):
            raise Exception(
                f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"  # noqa
            )

        self.name = "LinkNet"
        self._code = image_classifier_prf

        self._chip_size = (self._data.chip_size, self._data.chip_size)

        # Cut-off the backbone before the penultimate layer
        self._encoder = create_body(self._backbone, -2)

        # Initialize the model, loss function and the Learner object
        self._model = LinkNetModel(
            self._encoder,
            n_classes=self._data.c,
            chip_size=self._chip_size,
            n_bands=n_bands,
        )
        self._loss_f = mIoULoss(n_classes=self._data.c)
        learner_kwargs = {}
        self._opt_func_name = kwargs.get("opt_func", None)
        if self._opt_func_name:
            _opt_func = self._get_optimizer(self._opt_func_name)
            if _opt_func:
                self._opt_func_args = kwargs.get("opt_func_args", {})
                learner_kwargs["opt_func"] = partial(_opt_func, **self._opt_func_args)

        self.learn = Learner(
            self._data,
            self._model,
            loss_func=self._loss_f,
            metrics=[pixel_accuracy, mean_iou],
            **learner_kwargs,
        )
        if pretrained_path is not None:
            self.load(str(pretrained_path))
        self._arcgis_init_callback()  # make first conv weights learnable

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    def _get_optimizer(self, optim_name):
        optim_funcs = {"sgd": optim.SGD}
        return optim_funcs.get(optim_name, None)

    # Return a list of supported backbones names
    @property
    def supported_backbones(self):
        """
        Supported torchvision backbones for this model.
        """
        return LinkNet._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return [*_resnet_family]

    def _get_emd_params(self):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["InferenceFunction"] = "ArcGISImageClassifier.py"
        _emd_template["ModelConfiguration"] = "_LinkNet_Inference"
        _emd_template["ExtractBands"] = [0, 1, 2]
        model_params = {
            "backbone": self._backbone,
            "backend": self._backend,
            "opt_func": self._opt_func_name if hasattr(self, "_opt_func_name") else "",
            "opt_func_args": self._opt_func_args
            if hasattr(self, "_opt_func_args")
            else "",
        }
        _emd_template["ModelParameters"] = model_params
        _emd_template["Classes"] = []

        class_data = {}
        # 0th index is background
        for _, class_name in enumerate(self._data.classes[1:]):
            inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)]
            class_data["Color"] = color
            _emd_template["Classes"].append(class_data.copy())

        if hasattr(self._data, "_base"):
            data_params = {}
            for param_key, param_value in self._data._base.__dict__.items():
                if "files" in param_key:
                    continue
                if hasattr(param_value, "__dict__"):
                    continue
                if not safe_json(param_value):
                    param_value = str(param_value)
                data_params[f"{param_key}"] = param_value
            _emd_template[self._data._base.__class__.__name__] = data_params

        return _emd_template

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a LinkNet Pixel Classifier from an Esri Model Definition (EMD) file.
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Esri Model Definition
                                file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data` function or None for
                                inferencing.
        =====================   ===========================================
        :return: `LinkNet` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = Path(emd_path)
        emd = json.load(open(emd_path))
        model_file = Path(emd["ModelFile"])
        chip_size = emd["ImageWidth"]
        model_params = emd["ModelParameters"]

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}

        if data is None:
            data = _EmptyData(
                path=emd_path.parent.parent,
                loss_func=None,
                c=len(class_mapping) + 1,
                chip_size=chip_size,
            )
            data.class_mapping = class_mapping
            data = get_multispectral_data_params_from_emd(data, emd)
            data.emd_path = emd_path
            data.emd = emd

        return cls(data, **model_params, pretrained_path=str(model_file))

    def show_results(self, rows=5, **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================

        """
        self._check_requisites()
        ds_type = kwargs.get("ds_type", DatasetType.Valid)
        min_data_rows = min(
            len(self._data.dl(ds_type)), self._data.dl(ds_type).batch_size
        )
        if rows > min_data_rows:
            rows = min_data_rows
        self.learn.show_results(rows=rows, **kwargs)

    @property
    def _model_metrics(self):
        acc, mean_iou = self._get_model_metrics()
        return {
            "accuracy": "{0:1.4e}".format(acc),
            "mIoU": "{0:1.4e}".format(mean_iou),
        }

    def _get_model_metrics(self, **kwargs):
        checkpoint = getattr(self, "_is_checkpointed", False)
        if not hasattr(self.learn, "recorder"):
            return 0.0

        if len(self.learn.recorder.metrics) == 0:
            return 0.0

        try:
            model_accuracy = self.learn.recorder.metrics[-1][0]
            model_iou = self.learn.recorder.metrics[-1][1]
            if checkpoint:
                metrics_array = np.array(self.learn.recorder.metrics)
                model_iou = np.max(metrics_array[:, 1])
                model_accuracy = metrics_array[np.argmax(metrics_array[:, 1]), 0]
        except BaseException:
            logger = logging.getLogger()
            logger.debug("Cannot retrieve model accuracy.")
            model_accuracy = 0.0

        return float(model_accuracy), float(model_iou)
