import json
from pathlib import Path
from ._codetemplate import image_classifier_prf
from ._arcgis_model import ArcGISModel
from functools import partial
import logging

logger = logging.getLogger()

try:
    from ._arcgis_model import (
        _resnet_family,
        _densenet_family,
        _vgg_family,
    )
    from ._timm_utils import filter_timm_models
    from ._unet_utils import is_no_color, show_results_multispectral
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision import models
    from ._unet_utils import LabelCallback
    from ._arcgis_model import _EmptyData
    from fastai.layers import CrossEntropyFlat
    from .._utils.segmentation_loss_functions import FocalLoss, MixUpCallback, DiceLoss
    from ._psp_utils import PSPNet, _pspnet_learner, _pspnet_learner_with_unet, accuracy
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from .._utils.classified_tiles import per_class_metrics
    import numpy as np
    from fastai.torch_core import split_model_idx
    from fastai.vision import flatten_model
    from ._deeplab_utils import compute_miou
    from ._PointRend import PointRend_target_transform
    from .._utils.env import is_arcgispronotebook
    from matplotlib import pyplot as plt

    HAS_FASTAI = True
except Exception as e:
    HAS_FASTAI = False


class PSPNetClassifier(ArcGISModel):

    """
    Model architecture from https://arxiv.org/abs/1612.01105.
    Creates a PSPNet Image Segmentation/ Pixel Classification model.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    backbone                Optional string. Backbone convolutional neural network
                            model used for feature extraction, which
                            is `resnet50` by default.
                            Supported backbones: ResNet, DenseNet, VGG families
                            and specified Timm models(experimental support) from
                            :func:`~arcgis.learn.PSPNetClassifier.backbones`.
    ---------------------   -------------------------------------------
    use_unet                Optional Bool. Specify whether to use Unet-Decoder or not,
                            Default True.
    ---------------------   -------------------------------------------
    pyramid_sizes           Optional List. The sizes at which the feature map is pooled at.
                            Currently set to the best set reported in the paper,
                            i.e, (1, 2, 3, 6)
    ---------------------   -------------------------------------------
    pretrained              Optional Bool. If True, use the pretrained backbone
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained PSPNet model is
                            saved.
    ---------------------   -------------------------------------------
    unet_aux_loss           Optional. Bool If True will use auxiliary loss for PSUnet.
                            Default set to False. This flag is applicable only when
                            use_unet is True.
    ---------------------   -------------------------------------------
    pointrend               Optional boolean. If True, it will use PointRend
                            architecture on top of the segmentation head.
                            Default: False. PointRend architecture from
                            https://arxiv.org/pdf/1912.08193.pdf.
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
    focal_loss              Optional boolean. If True, it will use focal loss.
                            Default: False
    ---------------------   -------------------------------------------
    dice_loss_fraction      Optional float.
                            Min_val=0, Max_val=1
                            If > 0 , model will use a combination of default or
                            focal(if focal=True) loss with the specified fraction
                            of dice loss.

                            Example:

                                for dice = 0.3, loss = (1-0.3)*default loss + 0.3*dice

                            Default: 0
    ---------------------   -------------------------------------------
    dice_loss_average       Optional str.

                            * "``micro``": Micro dice coefficient will be used for loss calculation.

                            * "``macro``": Macro dice coefficient will be used for loss calculation.

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
    ---------------------   -------------------------------------------
    keep_dilation           Optional boolean. When PointRend architecture is used,
                            keep_dilation=True can potentially improve accuracy
                            at the cost of memory consumption. Default: False
    =====================   ===========================================

    :return: :class:`~arcgis.learn.PSPNetClassifier` Object
    """

    def __init__(
        self,
        data,
        backbone=None,
        use_unet=True,
        pyramid_sizes=[1, 2, 3, 6],
        pretrained_path=None,
        unet_aux_loss=False,
        pointrend=False,
        *args,
        **kwargs,
    ):
        # Set default backbone to be 'resnet50'
        if backbone is None:
            backbone = models.resnet50

        if pretrained_path is not None:
            backbone_pretrained = False
        else:
            backbone_pretrained = True

        self._check_dataset_support(data)
        if not (self._check_backbone_support(backbone)):
            raise Exception(
                f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"
            )

        super().__init__(data, backbone, pretrained_path=pretrained_path, **kwargs)

        if pointrend:
            use_unet = False

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
            accuracy = partial(accuracy, ignore_mapped_class=self._ignore_mapped_class)
        self.mixup = kwargs.get("mixup", False)
        self.class_balancing = kwargs.get("class_balancing", False)
        self.focal_loss = kwargs.get("focal_loss", False)
        self.dice_loss_fraction = kwargs.get("dice_loss_fraction", False)
        self.weighted_dice = kwargs.get("weighted_dice", False)
        self.dice_loss_average = kwargs.get("dice_loss_average", "micro")
        self.keep_dilation = kwargs.get("keep_dilation", False)
        self._vggv2 = kwargs.get("vggv2", True)
        self._code = image_classifier_prf
        self.pyramid_sizes = pyramid_sizes
        self._use_unet = use_unet
        self._unet_aux_loss = unet_aux_loss
        self._pointrend = pointrend

        if use_unet:
            self.learn = _pspnet_learner_with_unet(
                data,
                backbone=self._backbone,
                chip_size=self._data.chip_size,
                pyramid_sizes=pyramid_sizes,
                pretrained=backbone_pretrained,
                metrics=accuracy,
                unet_aux_loss=unet_aux_loss,
                vggv2=self._vggv2,
            )

            if self.class_balancing and data.class_weight is not None:
                class_weight = (
                    torch.tensor(
                        [data.class_weight.mean()] + data.class_weight.tolist()
                    )
                    .float()
                    .to(self._device)
                )
                self.learn.loss_func = CrossEntropyFlat(class_weight, axis=1)

        else:
            self.learn = _pspnet_learner(
                data,
                backbone=self._backbone,
                chip_size=self._data.chip_size,
                pyramid_sizes=pyramid_sizes,
                pretrained=backbone_pretrained,
                pointrend=self._pointrend,
                keep_dilation=self.keep_dilation,
                metrics=accuracy,
            )

        self._map_location = getattr(self.learn, "_map_location_multi_gpu", None)
        if self._map_location is not None:
            self._multigpu_training = True

        if self.mixup:
            self.learn.callbacks.append(MixUpCallback(self.learn))

        self.learn.callbacks.append(
            LabelCallback(self.learn)
        )  # appending label callback

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

        self.learn.loss_func = CrossEntropyFlat(class_weight, axis=1)
        self._final_class_weight = class_weight

        if unet_aux_loss or not use_unet:
            self.learn.loss_func = self._psp_loss
        if self.focal_loss:
            self.learn.loss_func = FocalLoss(self.learn.loss_func)
        if self.dice_loss_fraction:
            self.learn.loss_func = DiceLoss(
                self.learn.loss_func,
                self.dice_loss_fraction,
                weighted_dice=self.weighted_dice,
                dice_average=self.dice_loss_average,
            )
        self.learn.model = self.learn.model.to(self._device)
        self.freeze()
        self._arcgis_init_callback()  # make first conv weights learnable

        if pretrained_path is not None:
            self.load(pretrained_path)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy"]

    # Return a list of supported backbones names
    @property
    def supported_backbones(self):
        """Supported list of backbones for this model."""
        return PSPNetClassifier._supported_backbones()

    @staticmethod
    def backbones():
        """Supported list of backbones for this model."""
        return PSPNetClassifier._supported_backbones()

    @staticmethod
    def _supported_backbones():
        timm_models = filter_timm_models(
            [
                "*dpn*",
                "*inception*",
                "*hrnet*",
                "*nasnet*",
                "*repvgg*",
                "*resnetblur*",
                "*selecsls*",
                "*tresnet*",
                "*mobilenet*",
                "*hardcorenas*",
            ]
        )
        timm_backbones = list(map(lambda m: "timm:" + m, timm_models))
        return [*_resnet_family, *_densenet_family, *_vgg_family] + timm_backbones

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return PSPNetClassifier._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Classified_Tiles"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a PSPNet classifier from an Esri Model Definition (EMD) file.

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

        :return: :class:`~arcgis.learn.PSPNetClassifier` Object
        """
        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd["ModelParameters"]
        model_params["vggv2"] = model_params.get("vggv2", False)

        try:
            class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}
            color_mapping = {i["Value"]: i["Color"] for i in emd["Classes"]}
        except KeyError:
            class_mapping = {i["ClassValue"]: i["ClassName"] for i in emd["Classes"]}
            color_mapping = {i["ClassValue"]: i["Color"] for i in emd["Classes"]}

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

        return cls(data, **model_params, pretrained_path=str(model_file))

    def _psp_loss(self, outputs, targets, **kwargs):
        targets = targets.squeeze(1).detach()

        criterion = nn.CrossEntropyLoss(weight=self._final_class_weight).to(
            self._device
        )
        if self.learn.model.training:
            out = outputs[0]
            aux = outputs[1]
            if self._pointrend:
                pointrend_out = outputs[2][0]
                pointrend_coord = outputs[2][1]
                pointrend_target = PointRend_target_transform(targets, pointrend_coord)
        else:  # validation
            out = outputs
        main_loss = criterion(out, targets)

        if self.learn.model.training:
            aux_loss = criterion(aux, targets)
            if self._pointrend:
                pointrend_loss = criterion(pointrend_out, pointrend_target)
                total_loss = main_loss + 0.4 * aux_loss + pointrend_loss
            else:
                total_loss = main_loss + 0.4 * aux_loss
            return total_loss
        else:
            return main_loss

    def freeze(self):
        "Freezes the pretrained backbone."
        for idx, i in enumerate(flatten_model(self.learn.model)):
            if hasattr(i, "dilation"):
                if isinstance(i, (nn.BatchNorm2d)):
                    continue
                dilation = i.dilation
                dilation = dilation[0] if isinstance(dilation, tuple) else dilation
                if dilation > 1:
                    break
            for p in i.parameters():
                p.requires_grad = False

        self.learn.layer_groups = split_model_idx(
            self.learn.model, [idx]
        )  ## Could also call self.learn.freeze after this line because layer groups are now present.
        self.learn.create_opt(lr=3e-3)

    def unfreeze(self):
        """
        Unfreezes the earlier layers of the model for fine-tuning.
        """
        for _, param in self.learn.model.named_parameters():
            param.requires_grad = True

    def accuracy(self, input=None, target=None, void_code=0, class_mapping=None):
        """Computes per pixel accuracy."""
        if input is not None or target is not None:
            accuracy(input, target)
        else:
            try:
                return self.learn.validate()[1].tolist()
            except Exception as e:
                accuracy = self._data.emd.get("accuracy")
                if accuracy:
                    return accuracy
                else:
                    logger.error("Metric not found in the loaded model")

    def _get_emd_params(self, save_inference_file):
        import random

        _emd_template = {"ModelParameters": {}}
        _emd_template["ModelType"] = "ImageClassification"
        _emd_template["ModelParameters"]["pyramid_sizes"] = self.pyramid_sizes
        _emd_template["ModelParameters"]["use_unet"] = self._use_unet
        _emd_template["ModelParameters"]["pointrend"] = self._pointrend
        _emd_template["ModelParameters"]["keep_dilation"] = self.keep_dilation
        _emd_template["ModelParameters"]["vggv2"] = self._vggv2
        _emd_template["ModelParameters"]["unet_aux_loss"] = self._unet_aux_loss
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_psp"
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
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)
        self.learn.show_results(
            rows=rows, ignore_mapped_class=self._ignore_mapped_class, **kwargs
        )
        if is_arcgispronotebook():
            plt.show()

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
        return {"accuracy": "{0:1.4e}".format(self._get_model_metrics())}

    def _get_model_metrics(self, **kwargs):
        checkpoint = getattr(self, "_is_checkpointed", False)
        if not hasattr(self.learn, "recorder"):
            return 0.0

        if len(self.learn.recorder.metrics) == 0:
            return 0.0

        try:
            model_accuracy = self.learn.recorder.metrics[-1][0]
            if checkpoint:
                model_accuracy = self.learn.recorder.metrics[self.learn._best_epoch][
                    0
                ]  # index using best epoch.
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

            return pd.read_json(self._data.emd["per_class_metrics"])
