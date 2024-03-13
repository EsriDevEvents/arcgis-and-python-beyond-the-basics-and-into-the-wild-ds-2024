import json
from pathlib import Path
from ._codetemplate import image_classifier_prf
from ._arcgis_model import _EmptyData
from functools import partial
from .._data import _raise_fastai_import_error
import traceback
import logging

logger = logging.getLogger()

try:
    from ._arcgis_model import (
        ArcGISModel,
        _resnet_family,
        _set_ddp_multigpu,
        _isnotebook,
    )
    from ._timm_utils import filter_timm_models, timm_config
    import torch
    from fastai.vision.learner import unet_learner, cnn_config
    import numpy as np
    from fastai.layers import CrossEntropyFlat
    from .._utils.segmentation_loss_functions import FocalLoss, MixUpCallback, DiceLoss
    from ._unet_utils import (
        is_no_color,
        LabelCallback,
        predict_batch,
        show_results_multispectral,
    )
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from .._utils.classified_tiles import per_class_metrics
    from ._psp_utils import accuracy
    from ._deeplab_utils import compute_miou
    from matplotlib import pyplot as plt
    from .._utils.env import is_arcgispronotebook

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )

    class NnModule:
        pass

    HAS_FASTAI = False


class UnetClassifier(ArcGISModel):
    """
    Creates a Unet like classifier based on given pretrained encoder.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    backbone                Optional string. Backbone convolutional neural network
                            model used for feature extraction, which
                            is `resnet34` by default.
                            Supported backbones: ResNet family and specified Timm
                            models(experimental support) from :func:`~arcgis.learn.UnetClassifier.backbones`.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    ---------------------   -------------------------------------------
    backend                 Optional string. Controls the backend framework to be used
                            for this model, which is 'pytorch' by default.

                            valid options are 'pytorch', 'tensorflow'
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    class_balancing         Optional boolean. If True, it will balance the
                            cross-entropy loss inverse to the frequency
                            of pixels per class. Default: False.
    ---------------------   -------------------------------------------
    mixup                   Optional boolean. If True, it will use mixup
                            augmentation and mixup loss. Default: False
    ---------------------   -------------------------------------------
    focal_loss              Optional boolean. If True, it will use focal loss
                            Default: False
    ---------------------   -------------------------------------------
    dice_loss_fraction      Optional float.
                            Min_val=0, Max_val=1
                            If > 0 , model will use a combination of default or
                            focal(if focal=True) loss with the specified fraction
                            of dice loss.
                            E.g.
                            for dice = 0.3, loss = (1-0.3)*default loss + 0.3*dice
                            Default: 0
    ---------------------   -------------------------------------------
    dice_loss_average       Optional str.
                            micro: Micro dice coefficient will be used for loss
                            calculation.
                            macro: Macro dice coefficient will be used for loss
                            calculation.
                            A macro-average will compute the metric independently
                            for each class and then take the average (hence treating
                            all classes equally), whereas a micro-average will
                            aggregate the contributions of all classes to compute the
                            average metric. In a multi-class classification setup,
                            micro-average is preferable if you suspect there might be
                            class imbalance (i.e you may have many more examples of
                            one class than of other classes)
                            Default: 'micro'
    ---------------------   -------------------------------------------
    ignore_classes          Optional list. It will contain the list of class
                            values on which model will not incur loss.
                            Default: []
    =====================   ===========================================

    :return: :class:`~arcgis.learn.UnetClassifier` Object
    """

    def __init__(
        self,
        data,
        backbone=None,
        pretrained_path=None,
        backend="pytorch",
        *args,
        **kwargs,
    ):
        if pretrained_path is not None:
            backbone_pretrained = False
        else:
            backbone_pretrained = True

        self._backend = backend
        if self._backend == "tensorflow":
            super().__init__(data, None)
            self._intialize_tensorflow(data, backbone, pretrained_path, kwargs)
        else:
            super().__init__(data, backbone, pretrained_path=pretrained_path, **kwargs)
            data = self._data

            self._check_dataset_support(self._data)
            if not (self._check_backbone_support(getattr(self, "_backbone", backbone))):
                raise Exception(
                    f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"
                )

            self._ignore_classes = kwargs.get("ignore_classes", [])
            if self._ignore_classes != [] and len(data.classes) <= 3:
                raise Exception(
                    f"`ignore_classes` parameter can only be used when the dataset has more than 2 classes."
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
                global accuracy
                accuracy = partial(
                    accuracy, ignore_mapped_class=self._ignore_mapped_class
                )

            self.mixup = kwargs.get("mixup", False)
            self.class_balancing = kwargs.get("class_balancing", False)
            self.focal_loss = kwargs.get("focal_loss", False)
            self.dice_loss_fraction = kwargs.get("dice_loss_fraction", False)
            self.weighted_dice = kwargs.get("weighted_dice", False)
            self.dice_loss_average = kwargs.get("dice_loss_average", "micro")

            self._code = image_classifier_prf

            backbone_cut = None
            backbone_split = None

            if hasattr(self, "_orig_backbone"):
                _backbone_meta = cnn_config(self._orig_backbone)
                backbone_cut = _backbone_meta["cut"]
                backbone_split = _backbone_meta["split"]

            if "timm" in self._backbone.__module__:
                for bckbn in ["densenet", "inception_v4", "vgg"]:
                    if bckbn in self._backbone.__name__:
                        from torch import nn
                        from fastai.vision.learner import has_pool_type

                        def bckbn_cut(m):
                            ll = list(enumerate(m.children()))
                            cut = next(i for i, o in reversed(ll) if has_pool_type(o))
                            m = nn.Sequential(*list(m.children())[:cut])
                            return m[0]

                        backbone_cut = bckbn_cut
                        backbone_split = None
                        break
                else:
                    timm_meta = timm_config(self._backbone)
                    backbone_cut = timm_meta["cut"]
                    backbone_split = timm_meta["split"]

                if (
                    "nasnet" in self._backbone.__name__
                    or "repvgg" in self._backbone.__name__
                ):
                    backbone_cut = None

            if not _isnotebook():
                _set_ddp_multigpu(self)
                if self._multigpu_training:
                    self.learn = unet_learner(
                        data,
                        arch=self._backbone,
                        pretrained=backbone_pretrained,
                        metrics=accuracy,
                        wd=1e-2,
                        bottle=True,
                        last_cross=True,
                        cut=backbone_cut,
                        split_on=backbone_split,
                    ).to_distributed(self._rank_distributed)
                    self._map_location = {
                        "cuda:%d" % 0: "cuda:%d" % self._rank_distributed
                    }
                else:
                    self.learn = unet_learner(
                        data,
                        arch=self._backbone,
                        pretrained=backbone_pretrained,
                        metrics=accuracy,
                        wd=1e-2,
                        bottle=True,
                        last_cross=True,
                        cut=backbone_cut,
                        split_on=backbone_split,
                    )
            else:
                self.learn = unet_learner(
                    data,
                    arch=self._backbone,
                    pretrained=backbone_pretrained,
                    metrics=accuracy,
                    wd=1e-2,
                    bottle=True,
                    last_cross=True,
                    cut=backbone_cut,
                    split_on=backbone_split,
                )

            class_weight = None
            if self.class_balancing:
                if data.class_weight is not None:
                    # Handle condition when nodata is already at pixel value 0 in data
                    if (data.c - 1) == data.class_weight.shape[0]:
                        class_weight = (
                            torch.tensor(
                                [data.class_weight.mean()] + data.class_weight.tolist()
                            )
                            .float()
                            .to(self._device)
                        )
                    else:
                        class_weight = (
                            torch.tensor(data.class_weight).float().to(self._device)
                        )
                else:
                    if getattr(data, "overflow_encountered", False):
                        logger.warning(
                            "Overflow Encountered. Ignoring `class_balancing` parameter."
                        )
                        class_weight = [1] * len(data.classes)
                    else:
                        logger.warning(
                            "Could not find 'NumPixelsPerClass' in 'esri_accumulated_stats.json'. Ignoring `class_balancing` parameter."
                        )

            if self._ignore_classes != []:
                if not self.class_balancing:
                    class_weight = torch.tensor([1] * data.c).float().to(self._device)
                class_weight[self._ignore_mapped_class] = 0.0

            self._final_class_weight = class_weight
            self.learn.loss_func = CrossEntropyFlat(class_weight, axis=1)

            if self.focal_loss:
                self.learn.loss_func = FocalLoss(self.learn.loss_func)
            if self.mixup:
                self.learn.callbacks.append(MixUpCallback(self.learn))

            if self.dice_loss_fraction:
                self.learn.loss_func = DiceLoss(
                    self.learn.loss_func,
                    self.dice_loss_fraction,
                    weighted_dice=self.weighted_dice,
                    dice_average=self.dice_loss_average,
                )

            self._arcgis_init_callback()  # make first conv weights learnable
            self.learn.callbacks.append(
                LabelCallback(self.learn)
            )  # appending label callback

            self.learn.model = self.learn.model.to(self._device)
            # _set_multigpu_callback(self) # MultiGPU doesn't work for U-Net. (Fastai-Forums)
            if pretrained_path is not None:
                self.load(pretrained_path)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy"]

    @property
    def supported_backbones(self):
        """Supported list of backbones for this model."""
        return UnetClassifier._supported_backbones()

    @staticmethod
    def backbones():
        """Supported list of backbones for this model."""
        return UnetClassifier._supported_backbones()

    @staticmethod
    def _supported_backbones():
        timm_models = filter_timm_models(
            [
                "*dpn*",
                "*hrnet*",
                "nasnetalarge",
                "pnasnet5large",
                "*repvgg*",
                "*selecsls*",
                "*tresnet*",
            ]
        )
        timm_backbones = list(map(lambda m: "timm:" + m, timm_models))
        return [*_resnet_family] + timm_backbones

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return UnetClassifier._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Classified_Tiles"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a Unet like classifier from an Esri Model Definition (EMD) file.

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

        :return: :class:`~arcgis.learn.UnetClassifier` Object
        """
        return cls.from_emd(data, emd_path)

    @classmethod
    def from_emd(cls, data, emd_path):
        """
        Creates a Unet like classifier from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data` function or None for
                                inferencing.
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Esri Model Definition
                                file.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.UnetClassifier` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd["ModelParameters"]

        try:
            class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}
            color_mapping = {i["Value"]: i["Color"] for i in emd["Classes"]}
        except KeyError:
            class_mapping = {i["ClassValue"]: i["ClassName"] for i in emd["Classes"]}
            color_mapping = {i["ClassValue"]: i["Color"] for i in emd["Classes"]}

        resize_to = emd.get("resize_to")

        if data is None:
            data = _EmptyData(
                path=emd_path.parent.parent,
                loss_func=None,
                c=len(class_mapping) + 1,
                chip_size=emd["ImageHeight"],
            )
            data.class_mapping = class_mapping
            data.color_mapping = color_mapping
            data = get_multispectral_data_params_from_emd(data, emd)

            data.emd_path = emd_path
            data.emd = emd
            data._is_empty = True

        data.resize_to = resize_to

        return cls(data, **model_params, pretrained_path=str(model_file))

    @property
    def _model_metrics(self):
        return {"accuracy": "{0:1.4e}".format(self._get_model_metrics())}

    def _get_emd_params(self, save_inference_file):
        import random

        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_unet"
        _emd_template["ModelType"] = "ImageClassification"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISImageClassifier.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISImageClassifier.py"
        _emd_template["ExtractBands"] = [0, 1, 2]
        _emd_template["ignore_mapped_class"] = self._ignore_mapped_class
        _emd_template["SupportsVariableTileSize"] = True

        _emd_template["Classes"] = []
        class_data = {}
        for i, class_name in enumerate(
            self._data.classes[1:]
        ):  # 0th index is background
            inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = (
                [random.choice(range(256)) for i in range(3)]
                if is_no_color(self._data.color_mapping)
                else self._data.color_mapping[inverse_class_mapping[class_name]]
            )
            class_data["Color"] = color
            _emd_template["Classes"].append(class_data.copy())

        return _emd_template

    def _predict_batch(self, imagetensor_batch):
        return predict_batch(self, imagetensor_batch)

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
        self.learn.callbacks = [
            x for x in self.learn.callbacks if not isinstance(x, LabelCallback)
        ]
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)
        self.learn.show_results(
            rows=rows, ignore_mapped_class=self._ignore_mapped_class, **kwargs
        )
        if is_arcgispronotebook():
            plt.show()

    def accuracy(self):
        """Computes per pixel accuracy on validation set."""
        try:
            return self.learn.validate()[1].tolist()
        except Exception as e:
            accuracy = self._data.emd.get("accuracy")
            if accuracy:
                return accuracy
            else:
                logger.error("Metric not found in the loaded model")

    def _get_model_metrics(self, **kwargs):
        checkpoint = getattr(self, "_is_checkpointed", False)
        if not hasattr(self.learn, "recorder"):
            return 0.0

        try:
            model_accuracy = self.learn.recorder.metrics[-1][0]
            if checkpoint:
                val_losses = self.learn.recorder.val_losses
                model_accuracy = self.learn.recorder.metrics[
                    val_losses.index(min(val_losses))
                ][0]
        except:
            model_accuracy = 0.0

        return float(model_accuracy)

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
        show_progress           Optional bool. Displays the progress bar if
                                True.
        =====================   ===========================================

        :return: `dict` if mean is False otherwise `float`
        """
        self._check_requisites()
        num_classes = torch.arange(self._data.c)
        miou = compute_miou(
            self,
            self._data.valid_dl,
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
            return dict(zip(["0"] + self._data.classes[1:], miou))
        else:
            class_values = [0] + list(self._data.class_mapping.keys())
            return {
                class_values[i]: miou[i]
                for i in range(len(miou))
                if i not in self._ignore_mapped_class
            }

    ## Tensorflow specific functions start ##
    def _intialize_tensorflow(self, data, backbone, pretrained_path, kwargs):
        self._check_tf()
        self._ignore_mapped_class = []

        import tensorflow as tf
        from .._utils.common import get_color_array
        from .._utils.common_tf import handle_backbone_parameter, get_input_shape
        from .._model_archs.unet_tf import get_unet_tf_model
        from tensorflow.keras.losses import (
            SparseCategoricalCrossentropy,
            BinaryCrossentropy,
        )
        from .._utils.fastai_tf_fit import TfLearner, defaults
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from .._utils.common import kwarg_fill_none

        if data._is_multispectral:
            raise Exception(
                'Multispectral data is not supported with backend="tensorflow"'
            )

        # Intialize Tensorflow
        self._init_tensorflow(data, backbone)

        # Loss Function
        # self._loss_function_tf_ = BinaryCrossentropy(from_logits=True)
        self._loss_function_tf_ = SparseCategoricalCrossentropy(
            from_logits=True, reduction="auto"
        )

        self._mobile_optimized = kwarg_fill_none(
            kwargs, "mobile_optimized", self._backbone_mobile_optimized
        )

        # Create Unet Model
        model = get_unet_tf_model(
            self._backbone_initalized, data, mobile_optimized=self._mobile_optimized
        )

        self.learn = TfLearner(
            data,
            model,
            opt_func=Adam,
            loss_func=self._loss_function_tf,
            true_wd=True,
            bn_wd=True,
            wd=defaults.wd,
            train_bn=True,
        )

        self.learn.unfreeze()
        self.learn.freeze_to(len(self._backbone_initalized.layers))

        self.show_results = self._show_results_multispectral

        self._code = image_classifier_prf

    def _loss_function_tf(self, target, predictions):
        import tensorflow as tf

        # print(target.shape, predictions.shape)
        # print(target.dtype, predictions.dtype)
        # print(tf.unique(tf.reshape(target, [-1]))[0])
        # print('\n', tf.unique(tf.reshape(predictions, [-1]))[0])
        # print(tf.unique(tf.reshape(target, [-1])).numpy(), tf.unique(tf.reshape(predictions, [-1])))
        target = tf.squeeze(target, axis=1)

        # from .._utils.pixel_classification import segmentation_mask_to_one_hot
        # from .._utils.fastai_tf_fit import _pytorch_to_tf
        # target = _pytorch_to_tf(segmentation_mask_to_one_hot(target.cpu().numpy(), self._data.c).permute(0, 2, 3, 1))

        return self._loss_function_tf_(target, predictions)

    ## Tensorflow specific functions end ##

    def per_class_metrics(self, ignore_classes=[]):
        """
        Computer per class precision, recall and f1-score on validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        self                    segmentation model object -> [PSPNetClassifier | UnetClassifier | DeepLab]
        ---------------------   -------------------------------------------
        ignore_classes          Optional list. It will contain the list of class
                                values on which model will not incur loss.
                                Default: []
        =====================   ===========================================

        Returns per class precision, recall and f1 scores
        """
        ignore_classes = np.unique(self._ignore_classes + ignore_classes).tolist()
        try:
            self._check_requisites()
            ## Calling imported function `per_class_metrics`
            return per_class_metrics(self, ignore_classes)
        except:
            import pandas as pd

            if "per_class_metrics" in self._data.emd.keys():
                return pd.read_json(self._data.emd["per_class_metrics"])
            else:
                logger.error("Metric not found in the loaded model")
