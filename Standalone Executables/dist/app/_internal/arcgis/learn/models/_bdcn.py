from pathlib import Path
import json
from ._model_extension import ModelExtension
from ._arcgis_model import _EmptyData

try:
    from fastai.vision import flatten_model
    import torch
    from fastai.torch_core import split_model_idx
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from ._arcgis_model import _resnet_family, _vgg_family
    from ._timm_utils import filter_timm_models
    from ._hed_utils import DDPCallback

    HAS_FASTAI = True

except Exception as e:
    HAS_FASTAI = False


class CustomBDCN:
    """
    Create class with following fixed function names and the number of arguents to train your model from external source
    """

    try:
        import torch
        from torchvision import models
        from arcgis.learn.models import _bdcn_utils as bdcn
    except:
        pass

    def get_model(self, data, backbone=None, **kwargs):
        """
        In this fuction you have to define your model with following two arguments!

        """
        pretrained_backbone = kwargs.get("pretrained_backbone", True)

        if backbone is None:
            self._backbone = self.models.vgg19
        elif type(backbone) is str:
            if hasattr(self.models, backbone):
                self._backbone = getattr(self.models, backbone)
            elif hasattr(self.models.detection, backbone):
                self._backbone = getattr(self.models.detection, backbone)
            elif "timm:" in backbone:
                import timm

                bckbn = backbone.split(":")[1]
                if hasattr(timm.models, bckbn):
                    self._backbone = getattr(timm.models, bckbn)
        else:
            self._backbone = backbone

        model = self.bdcn._BDCNModel(
            self._backbone, data.chip_size, pretrained=pretrained_backbone
        )

        return model

    def on_batch_begin(self, learn, model_input_batch, model_target_batch, **kwargs):
        return model_input_batch, model_target_batch

    def transform_input(self, xb):
        return xb

    def transform_input_multispectral(self, xb):
        return xb

    def loss(self, model_output, *model_target):
        final_loss = self.bdcn.bdcn_loss(model_output, *model_target)

        return final_loss

    def post_process(self, pred, thres=0.5, thinning=True, prob_raster=False):
        """
        In this function you have to return list with appended output for each image in the batch with shape [C=1,H,W]!

        """

        from skimage.morphology import skeletonize, binary_dilation
        import numpy as np

        post_processed_pred = []
        if isinstance(pred, list):
            pred = pred[-1]
        if prob_raster:
            return pred
        elif thinning:
            for p in pred:
                p = self.torch.unsqueeze(
                    self.torch.tensor(
                        skeletonize(
                            binary_dilation(
                                np.squeeze((p >= thres).byte().cpu().numpy())
                            )
                        )
                    ),
                    dim=0,
                )
                post_processed_pred.append(p)
            return post_processed_pred
        else:
            return (pred >= thres).byte()


class BDCNEdgeDetector(ModelExtension):
    """
    Model architecture from https://arxiv.org/pdf/1902.10903.pdf.
    Creates a :class:`~arcgis.learn.BDCNEdgeDetector` model

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    backbone                Optional string. Backbone convolutional neural network
                            model used for feature extraction, which is `vgg19` by
                            default.
                            Supported backbones: ResNet, Vgg family and specified Timm
                            models(experimental support) from :func:`~arcgis.learn.BDCNEdgeDetector.backbones`.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.BDCNEdgeDetector` Object
    """

    def __init__(self, data, backbone="vgg19", pretrained_path=None):
        self._check_dataset_support(data)
        backbone_name = backbone if type(backbone) is str else backbone.__name__
        if backbone_name not in self.supported_backbones:
            raise Exception(
                f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"
            )

        super().__init__(data, CustomBDCN, backbone, pretrained_path)

        if getattr(self, "_multigpu_training", False):
            self.learn.callbacks.append(DDPCallback(self.learn, self._rank_distributed))
        self._freeze()

    def unfreeze(self):
        for _, param in self.learn.model.named_parameters():
            param.requires_grad = True

    def _freeze(self):
        "Freezes the pretrained backbone."
        count = 0
        count_strided_conv = 0
        for idx, i in enumerate(flatten_model(self.learn.model.backbone)):
            if isinstance(i, (torch.nn.BatchNorm2d)):
                continue

            for p in i.parameters():
                p.requires_grad = False

            if isinstance(i, torch.nn.MaxPool2d):
                count += 1
                if count == 3:
                    break
            if isinstance(i, torch.nn.Conv2d):
                if i.stride[0] == 2:
                    count_strided_conv += 1
                    if count_strided_conv == 4:
                        break

        self.learn.layer_groups = split_model_idx(self.learn.model, [idx])
        self.learn.create_opt(lr=3e-3)

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy", "f1_score"]

    @property
    def _is_edge_detection(self):
        return True

    @staticmethod
    def backbones():
        """Supported list of backbones for this model."""
        return BDCNEdgeDetector._supported_backbones()

    @property
    def supported_backbones(self):
        """Supported list of backbones for this model."""
        return BDCNEdgeDetector._supported_backbones()

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
                "*ghostnet*",
            ]
        )
        timm_backbones = list(map(lambda m: "timm:" + m, timm_models))
        return [*_resnet_family, *_vgg_family] + timm_backbones

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return BDCNEdgeDetector._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Classified_Tiles"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a :class:`~arcgis.learn.BDCNEdgeDetector` object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data`  function or None for
                                inferencing.

        =====================   ===========================================

        :return: :class:`~arcgis.learn.BDCNEdgeDetector` Object
        """
        emd_path = _get_emd_path(emd_path)

        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        backbone = emd["ModelParameters"]["backbone"]

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
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            data.classes = ["background"]
            for k, v in class_mapping.items():
                data.classes.append(v)
            data = get_multispectral_data_params_from_emd(data, emd)
            data.dataset_type = emd["DatasetType"]

        return cls(data, backbone, pretrained_path=str(model_file))

    def compute_precision_recall(self, thresh=0.5, buffer=3, show_progress=True):
        """
        Computes precision, recall and f1 score on validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        thresh                  Optional float. The probability on which
                                the detection will be considered edge pixel.
        ---------------------   -------------------------------------------
        buffer                  Optional int. pixels in neighborhood to
                                consider true detection.
        =====================   ===========================================

        :return: `dict`
        """

    def show_results(self, rows=5, thresh=0.5, thinning=True, **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.
        """
