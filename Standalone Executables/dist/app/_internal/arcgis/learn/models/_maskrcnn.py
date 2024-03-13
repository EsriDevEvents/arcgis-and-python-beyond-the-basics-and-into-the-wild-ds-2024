from ._arcgis_model import ArcGISModel
from ._arcgis_model import _EmptyData, _change_tail

HAS_OPENCV = True
try:
    from pathlib import Path
    import warnings
    import json
    import math
    import PIL
    import torch
    import torch.nn as nn
    from collections import OrderedDict
    from fastai.vision.learner import create_body
    from fastai.callbacks.hooks import num_features_model
    from fastai.vision import flatten_model
    from fastai.vision.image import pil2tensor
    from fastai.core import split_kwargs_by_func
    from torchvision import models
    from skimage.measure import find_contours
    from ._codetemplate import instance_detector_prf
    from arcgis.learn.models._deepsort_predict_utils import non_max_suppression
    import numpy as np
    import types
    from ._arcgis_model import (
        _get_backbone_meta,
        _resnet_family,
        _set_ddp_multigpu,
        _isnotebook,
    )
    from ._timm_utils import timm_config, filter_timm_models, _get_feature_size
    from torchvision import models
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from fastai.basic_train import Learner
    from ._maskrcnn_utils import (
        is_no_color,
        mask_rcnn_loss,
        train_callback,
        compute_class_AP,
        predict_tta,
        AveragePrecision,
        forward_roi,
        postprocess_transform,
        post_nms_top_n,
        pre_nms_top_n,
        eager_outputs_modified,
    )
    from .._image_utils import _get_image_chips, _draw_predictions

    from ._MaskRCNN_PointRend import create_pointrend
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from fastai.torch_core import split_model_idx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from .._utils.common import get_nbatches, image_batch_stretcher, read_image
    from .._utils.env import is_arcgispronotebook

    HAS_FASTAI = True
except Exception as e:
    # raise Exception(e)
    HAS_FASTAI = False

try:
    import cv2
except Exception:
    HAS_OPENCV = False


class TimmFPNBackbone(nn.Module):
    def __init__(self, backbone, chip_size=224):
        from fastai.callbacks.hooks import hook_outputs
        from fastai.callbacks.hooks import model_sizes
        from torchvision.ops.feature_pyramid_network import (
            FeaturePyramidNetwork,
            LastLevelMaxPool,
        )
        from ._hed_utils import get_hooks

        super().__init__()
        self.backbone = backbone
        hooks = get_hooks(self.backbone, chip_size)
        self.hook = hook_outputs(hooks[1:])
        model_sizes(self.backbone, size=(chip_size, chip_size))
        layer_num_channels = [k.stored.shape[1] for k in self.hook]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=layer_num_channels,
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 256

    def forward(self, x):
        x = self.backbone(x)
        out = OrderedDict()
        features = self.hook.stored
        for k, v in enumerate(features):
            out[str(k)] = v
        x = self.fpn(out)
        return x


def chips_to_batch(chips, model_height, model_width, batch_size=1):
    dtype = np.float32
    band_count = 3
    if len(chips) != 0:
        dtype = chips[0].dtype

    batch = np.zeros(
        shape=(batch_size, band_count, model_height, model_width),
        dtype=dtype,
    )
    for b in range(batch_size):
        if b < len(chips):
            batch[b, :, :model_height, :model_height] = chips[b]

    return batch


def grid_anchors(self, grid_sizes, strides):
    anchors = []
    cell_anchors = self.cell_anchors
    assert cell_anchors is not None

    for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
        grid_height, grid_width = size
        stride_height, stride_width = stride
        device = base_anchors.device

        # For output anchor, compute [x_center, y_center, x_center, y_center]
        shifts_x = (
            torch.arange(0, grid_width, dtype=torch.float32, device=device)
            * stride_width
        )
        shifts_y = (
            torch.arange(0, grid_height, dtype=torch.float32, device=device)
            * stride_height
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        # For every (base anchor, output anchor) pair,
        # offset each zero-centered base anchor by the center of the output anchor.
        anchors.append(
            (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
        )

    return anchors


class MaskRCNNTracer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def _dict_to_tuple(self, out_dict):
        return (
            out_dict["boxes"],
            out_dict["scores"],
            out_dict["labels"],
            out_dict["masks"],
        )

    def forward(self, inp):
        out = self.model(inp)
        return self._dict_to_tuple(out[0])


class MaskRCNN(ArcGISModel):
    """
    Model architecture from https://arxiv.org/abs/1703.06870.
    Creates a :class:`~arcgis.learn.MaskRCNN` Instance segmentation model,
    based on https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    backbone                Optional string. Backbone convolutional neural network
                            model used for feature extraction, which
                            is `resnet50` by default.
                            Supported backbones: ResNet family and specified Timm
                            models(experimental support) from :func:`~arcgis.learn.MaskRCNN.backbones`.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    ---------------------   -------------------------------------------
    pointrend               Optional boolean. If True, it will use PointRend
                            architecture on top of the segmentation head.
                            Default: False. PointRend architecture from
                            https://arxiv.org/pdf/1912.08193.pdf.
    =====================   ===========================================

    **kwargs**

    =============================   =============================================
    **Parameter**                    **Description**
    -----------------------------   ---------------------------------------------
    rpn_pre_nms_top_n_train         Optional int. Number of proposals to keep before
                                    applying NMS during training.
                                    Default: 2000
    -----------------------------   ---------------------------------------------
    rpn_pre_nms_top_n_test          Optional int. Number of proposals to keep before
                                    applying NMS during testing.
                                    Default: 1000
    -----------------------------   ---------------------------------------------
    rpn_post_nms_top_n_train        Optional int. Number of proposals to keep after
                                    applying NMS during training.
                                    Default: 2000
    -----------------------------   ---------------------------------------------
    rpn_post_nms_top_n_test         Optional int. Number of proposals to keep after
                                    applying NMS during testing.
                                    Default: 1000
    -----------------------------   ---------------------------------------------
    rpn_nms_thresh                  Optional float. NMS threshold used for postprocessing
                                    the RPN proposals.
                                    Default: 0.7
    -----------------------------   ---------------------------------------------
    rpn_fg_iou_thresh               Optional float. Minimum IoU between the anchor
                                    and the GT box so that they can be considered
                                    as positive during training of the RPN.
                                    Default: 0.7
    -----------------------------   ---------------------------------------------
    rpn_bg_iou_thresh               Optional float. Maximum IoU between the anchor and
                                    the GT box so that they can be considered as negative
                                    during training of the RPN.
                                    Default: 0.3
    -----------------------------   ---------------------------------------------
    rpn_batch_size_per_image        Optional int. Number of anchors that are sampled
                                    during training of the RPN for computing the loss.
                                    Default: 256
    -----------------------------   ---------------------------------------------
    rpn_positive_fraction           Optional float. Proportion of positive anchors in a
                                    mini-batch during training of the RPN.
                                    Default: 0.5
    -----------------------------   ---------------------------------------------
    box_score_thresh                Optional float. During inference, only return proposals
                                    with a classification score greater than box_score_thresh
                                    Default: 0.05
    -----------------------------   ---------------------------------------------
    box_nms_thresh                  Optional float. NMS threshold for the prediction head.
                                    Used during inference.
                                    Default: 0.5
    -----------------------------   ---------------------------------------------
    box_detections_per_img          Optional int. Maximum number of detections per
                                    image, for all classes.
                                    Default: 100
    -----------------------------   ---------------------------------------------
    box_fg_iou_thresh               Optional float. Minimum IoU between the proposals and
                                    the GT box so that they can be considered as positive
                                    during training of the classification head.
                                    Default: 0.5
    -----------------------------   ---------------------------------------------
    box_bg_iou_thresh               Optional float. Maximum IoU between the proposals and
                                    the GT box so that they can be considered as negative
                                    during training of the classification head.
                                    Default: 0.5
    -----------------------------   ---------------------------------------------
    box_batch_size_per_image        Optional int. Number of proposals that are sampled during
                                    training of the classification head.
                                    Default: 512
    -----------------------------   ---------------------------------------------
    box_positive_fraction           Optional float. Proportion of positive proposals in a
                                    mini-batch during training of the classification head.
                                    Default: 0.25
    =============================   =============================================

    :return:
        :class:`~arcgis.learn.MaskRCNN` Object
    """

    def __init__(
        self,
        data,
        backbone=None,
        pretrained_path=None,
        pointrend=False,
        *args,
        **kwargs,
    ):
        # Set default backbone to be 'resnet50'
        if backbone is None:
            backbone = models.resnet50

        if pretrained_path is not None:
            pretrained_backbone = False
        else:
            pretrained_backbone = True

        self._check_dataset_support(data)
        if not (self._check_backbone_support(backbone)):
            raise Exception(
                f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"
            )

        super().__init__(data, backbone, pretrained_path=pretrained_path, **kwargs)
        if self._is_multispectral:
            self._backbone_ms = self._backbone
            self._backbone = self._orig_backbone
            scaled_mean_values = data._scaled_mean_values[data._extract_bands].tolist()
            scaled_std_values = data._scaled_std_values[data._extract_bands].tolist()

        self._code = instance_detector_prf

        self._pointrend = pointrend

        self.maskrcnn_kwargs, kwargs = split_kwargs_by_func(
            kwargs, models.detection.MaskRCNN.__init__
        )

        if (
            self._backbone.__name__ == "resnet50"
            and "timm" not in self._backbone.__module__
        ):
            model = models.detection.maskrcnn_resnet50_fpn(
                pretrained=pretrained_backbone,
                pretrained_backbone=False,
                min_size=1.5 * data.chip_size,
                max_size=2 * data.chip_size,
                **self.maskrcnn_kwargs,
            )

            if self._is_multispectral:
                model.backbone = _change_tail(model.backbone, data)
                model.transform.image_mean = scaled_mean_values
                model.transform.image_std = scaled_std_values
        elif (
            self._backbone.__name__ in ["resnet18", "resnet34"]
            and not pointrend
            and "timm" not in self._backbone.__module__
        ):
            if self._is_multispectral:
                backbone_small = create_body(
                    self._backbone_ms,
                    pretrained=pretrained_backbone,
                    cut=_get_backbone_meta(self._backbone.__name__)["cut"],
                )
                backbone_small.out_channels = 512
                model = models.detection.MaskRCNN(
                    backbone_small,
                    91,
                    min_size=1.5 * data.chip_size,
                    max_size=2 * data.chip_size,
                    image_mean=scaled_mean_values,
                    image_std=scaled_std_values,
                    **self.maskrcnn_kwargs,
                )
            else:
                backbone_small = create_body(
                    self._backbone, pretrained=pretrained_backbone
                )
                backbone_small.out_channels = 512
                model = models.detection.MaskRCNN(
                    backbone_small,
                    91,
                    min_size=1.5 * data.chip_size,
                    max_size=2 * data.chip_size,
                    **self.maskrcnn_kwargs,
                )
            model.rpn.anchor_generator.grid_anchors = types.MethodType(
                grid_anchors, model.rpn.anchor_generator
            )
        else:
            if "timm" in self._backbone.__module__:
                backbone_cut = timm_config(self._backbone)["cut"]
                backbone_fpn = create_body(
                    self._backbone, pretrained_backbone, backbone_cut
                )
                try:
                    backbone_fpn = TimmFPNBackbone(backbone_fpn, data.chip_size)
                except:
                    if "tresnet" in self._backbone.__module__:
                        backbone_fpn.out_channels = _get_feature_size(
                            self._backbone, backbone_cut
                        )[-1][1]
                    else:
                        backbone_fpn.out_channels = num_features_model(
                            torch.nn.Sequential(*backbone_fpn.children())
                        )
            else:
                backbone_fpn = resnet_fpn_backbone(
                    self._backbone.__name__, pretrained=pretrained_backbone
                )
            if self._is_multispectral:
                backbone_fpn = _change_tail(backbone_fpn, data)
                model = models.detection.MaskRCNN(
                    backbone_fpn,
                    91,
                    min_size=1.5 * data.chip_size,
                    max_size=2 * data.chip_size,
                    image_mean=scaled_mean_values,
                    image_std=scaled_std_values,
                    **self.maskrcnn_kwargs,
                )
            else:
                model = models.detection.MaskRCNN(
                    backbone_fpn,
                    91,
                    min_size=1.5 * data.chip_size,
                    max_size=2 * data.chip_size,
                    **self.maskrcnn_kwargs,
                )
            if "timm" in self._backbone.__module__:
                model.rpn.anchor_generator.grid_anchors = types.MethodType(
                    grid_anchors, model.rpn.anchor_generator
                )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, data.c)

        if pointrend:
            model = create_pointrend(model, data.c)
        else:
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, hidden_layer, data.c
            )

        if not _isnotebook():
            _set_ddp_multigpu(self)
            if self._multigpu_training:
                self.learn = Learner(
                    data, model, loss_func=mask_rcnn_loss
                ).to_distributed(self._rank_distributed)
                self._map_location = {"cuda:%d" % 0: "cuda:%d" % self._rank_distributed}
            else:
                self.learn = Learner(data, model, loss_func=mask_rcnn_loss)
        else:
            self.learn = Learner(data, model, loss_func=mask_rcnn_loss)
        self.learn.callbacks.append(train_callback(self.learn))
        if not pointrend:
            self.learn.model.roi_heads.forward = types.MethodType(
                forward_roi, self.learn.model.roi_heads
            )
        self.learn.model.eager_outputs = types.MethodType(
            eager_outputs_modified, self.learn.model
        )
        self.learn.model.transform.postprocess = types.MethodType(
            postprocess_transform, self.learn.model.transform
        )
        self.learn.model.rpn.post_nms_top_n = types.MethodType(
            post_nms_top_n, self.learn.model.rpn
        )
        self.learn.model.rpn.pre_nms_top_n = types.MethodType(
            pre_nms_top_n, self.learn.model.rpn
        )
        self.learn.metrics = [AveragePrecision(self.learn)]
        self.learn.model = self.learn.model.to(self._device)
        self.learn.c_device = self._device

        # fixes for zero division error when slice is passed
        idx = self._freeze()
        self.learn.layer_groups = split_model_idx(self.learn.model, [idx])
        self.learn.create_opt(lr=3e-3)

        # make first conv weights learnable
        self._arcgis_init_callback()

        if pretrained_path is not None:
            self.load(pretrained_path)

        if hasattr(data, "_image_space_used"):
            if data._image_space_used == "MAP_SPACE":
                self.learn.model.arcgis_tta = list(range(8))
            else:
                self.learn.model.arcgis_tta = [0, 2]
        if self._is_multispectral:
            self._orig_backbone = self._backbone
            self._backbone = self._backbone_ms

    def unfreeze(self):
        for _, param in self.learn.model.named_parameters():
            param.requires_grad = True

    def _freeze(self):
        "Freezes the pretrained backbone."
        for idx, i in enumerate(flatten_model(self.learn.model.backbone)):
            if isinstance(i, (torch.nn.BatchNorm2d)):
                continue
            for p in i.parameters():
                p.requires_grad = False
        return idx

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @staticmethod
    def _available_metrics():
        return ["valid_loss"]

    @property
    def supported_backbones(self):
        """Supported list of backbones for this model."""
        return MaskRCNN._supported_backbones()

    @staticmethod
    def backbones():
        """Supported list of backbones for this model."""
        return MaskRCNN._supported_backbones()

    @staticmethod
    def _supported_backbones():
        timm_models = filter_timm_models(["*repvgg*", "*tresnet*"])
        timm_backbones = list(map(lambda m: "timm:" + m, timm_models))
        return [*_resnet_family] + timm_backbones

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return MaskRCNN._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["RCNN_Masks"]

    @classmethod
    def from_model(cls, emd_path, data=None, **kwargs):
        """
        Creates a :class:`~arcgis.learn.MaskRCNN` Instance segmentation object from an Esri Model Definition (EMD) file.

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

        :return: :class:`~arcgis.learn.MaskRCNN` Object
        """

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd["ModelParameters"]
        maskrcnn_kwargs = emd.get("MaskRCNNkwargs", {})

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
                chip_size=kwargs.get("chip_size", emd["ImageHeight"]),
            )
            data.resize_to = emd.get("resize_to", None)
            data.class_mapping = class_mapping
            data.color_mapping = color_mapping
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            data = get_multispectral_data_params_from_emd(data, emd)

        return cls(
            data, **model_params, pretrained_path=str(model_file), **maskrcnn_kwargs
        )

    def _save_pytorch_torchscript(self, name, save=True):
        model = self.learn.model
        model.eval()

        cpu = torch.device("cpu")
        device = cpu
        model = model.to(cpu)

        if hasattr(self._data, "chip_size"):
            chip_size = self._data.chip_size
            if not isinstance(chip_size, tuple):
                chip_size = (chip_size, chip_size)
        inp = torch.rand(1, 3, chip_size[0], chip_size[1]).to(cpu)

        model = MaskRCNNTracer(model)

        traced_model = None
        try:
            with torch.no_grad():
                traced_model = self._trace(model, inp, True)
        except:
            with torch.no_grad():
                traced_model = self._trace(model, inp, True)

        model.to(device)
        saved_path_cpu = (
            self.learn.path / self.learn.model_dir / f"{name}-cpu.pt"
        ).__str__()
        saved_path_gpu = ""
        torch.jit.save(traced_model, saved_path_cpu)

        if not save:
            [traced_model, traced_model]
        return [f"{name}-cpu.pt", saved_path_gpu]

    def _save_pytorch_tflite(self, name):
        import tensorflow as tf
        import logging

        tf.get_logger().setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import onnx
            import onnx_tf
            from onnx_tf.backend import prepare

        traced_models = self._save_pytorch_torchscript(name, False)
        if traced_models[0] is None:
            return ["", ""]

        cpu = torch.device("cpu")
        device = cpu
        traced_model = traced_models[0].to(device)

        if hasattr(self._data, "chip_size"):
            chip_size = self._data.chip_size
            if not isinstance(chip_size, tuple):
                chip_size = (chip_size, chip_size)
        num_input_channels = list(self.learn.model.parameters())[0].shape[1]
        inp = torch.randn([1, num_input_channels, chip_size[0], chip_size[1]]).to(
            device
        )

        save_path_tflite = self.learn.path / self.learn.model_dir / f"{name}.tflite"
        save_path_onnx = self.learn.path / self.learn.model_dir / f"{name}.onnx"
        save_path_pb = self.learn.path / self.learn.model_dir / f"{name}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                traced_model,
                inp,
                save_path_onnx,
                export_params=True,
                do_constant_folding=False,
                verbose=True,
                input_names=["input"],
                output_names=["output"],
                opset_version=11,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arcgis_onnx = onnx.load(save_path_onnx)
            tf_onnx = prepare(arcgis_onnx, logging_level="ERROR")
            tf_onnx.export_graph(str(save_path_pb))

        model = tf.saved_model.load(save_path_pb)
        concrete_func = model.signatures[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # for full accuracy
        tf_lite_model = converter.convert()
        open(save_path_tflite, "wb").write(tf_lite_model)

        return [save_path_tflite, save_path_onnx]

    def _get_emd_params(self, save_inference_file):
        import random

        _emd_template = {"ModelParameters": {}}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_maskrcnn_inferencing"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISInstanceDetector.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISInstanceDetector.py"
        _emd_template["ModelType"] = "InstanceDetection"
        _emd_template["MaskRCNNkwargs"] = self.maskrcnn_kwargs
        _emd_template["ModelParameters"]["pointrend"] = self._pointrend
        _emd_template["ExtractBands"] = [0, 1, 2]
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

    @property
    def _model_metrics(self):
        return {
            "average_precision_score": self.average_precision_score(show_progress=True)
        }

    def _predict_batch(self, images, detect_thresh=0.5, tta_prediction=False):
        model = self.learn.model
        model.eval()
        model = model.to(self._device)
        normed_batch_tensor = images.to(self._device)
        if tta_prediction:
            predictions = predict_tta(model, normed_batch_tensor, detect_thresh)
        else:
            temp = model.roi_heads.score_thresh
            model.roi_heads.score_thresh = detect_thresh
            predictions = model(list(normed_batch_tensor))
            model.roi_heads.score_thresh = temp
        normed_batch_tensor.detach().cpu()
        del normed_batch_tensor
        return predictions

    def _predict_results(self, xb, detect_thresh=0.5, tta_prediction=False):
        predictions = self._predict_batch(xb, detect_thresh, tta_prediction)
        predictionsf = []
        for i in range(len(predictions)):
            predictionsf.append({})
            predictionsf[i]["masks"] = predictions[i]["masks"].detach().cpu().numpy()
            predictionsf[i]["boxes"] = predictions[i]["boxes"].detach().cpu().numpy()
            predictionsf[i]["labels"] = predictions[i]["labels"].detach().cpu().numpy()
            predictionsf[i]["scores"] = predictions[i]["scores"].detach().cpu().numpy()
            del predictions[i]["masks"]
            del predictions[i]["boxes"]
            del predictions[i]["labels"]
            del predictions[i]["scores"]
        if self._device == torch.device("cuda"):
            torch.cuda.empty_cache()
        return predictionsf

    def _predict_postprocess(self, predictions, threshold=0.5, box_threshold=0.5):
        pred_mask = []
        pred_box = []

        for i in range(len(predictions)):
            out = predictions[i]["masks"].squeeze()
            pred_box.append([])

            if out.shape[0] != 0:  # handle for prediction with n masks
                if (
                    len(out.shape) == 2
                ):  # for out dimension hxw (in case of only one predicted mask)
                    out = out[None]
                ymask = np.where(out[0] > threshold, 1, 0)
                if predictions[i]["scores"][0] > box_threshold:
                    pred_box[i].append(predictions[i]["boxes"][0])
                for j in range(1, out.shape[0]):
                    ym1 = np.where(out[j] > threshold, j + 1, 0)
                    ymask += ym1
                    if predictions[i]["scores"][j] > box_threshold:
                        pred_box[i].append(predictions[i]["boxes"][j])
            else:
                ymask = np.zeros(
                    (self._data.chip_size, self._data.chip_size)
                )  # handle for not predicted masks
            pred_mask.append(ymask)
        return pred_mask, pred_box

    def show_results(
        self,
        rows=4,
        mode="mask",
        mask_threshold=0.5,
        box_threshold=0.7,
        tta_prediction=False,
        imsize=5,
        index=0,
        alpha=0.5,
        cmap="tab20",
        **kwargs,
    ):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        ---------------------   -------------------------------------------
        mode                    Required arguments within ['bbox', 'mask', 'bbox_mask'].
                                    * ``bbox`` - For visualizing only bounding boxes.
                                    * ``mask`` - For visualizing only mask
                                    * ``bbox_mask`` - For visualizing both mask and bounding boxes.
        ---------------------   -------------------------------------------
        mask_threshold          Optional float. The probability above which
                                a pixel will be considered mask.
        ---------------------   -------------------------------------------
        box_threshold           Optional float. The probability above which
                                a detection will be considered valid.
        ---------------------   -------------------------------------------
        tta_prediction          Optional bool. Perform test time augmentation
                                while predicting
        =====================   ===========================================
        """
        self._check_requisites()
        if mode not in ["bbox", "mask", "bbox_mask"]:
            raise Exception("mode can be only ['bbox', 'mask', 'bbox_mask']")

        # Get Number of items
        nrows = rows
        ncols = 2

        type_data_loader = kwargs.get(
            "data_loader", "validation"
        )  # options : traininig, validation, testing
        if type_data_loader == "training":
            data_loader = self._data.train_dl
        elif type_data_loader == "validation":
            data_loader = self._data.valid_dl
        elif type_data_loader == "testing":
            data_loader = self._data.test_dl
        else:
            e = Exception(
                f"could not find {type_data_loader} in data. Please ensure that the data loader type is traininig, validation or testing "
            )
            raise (e)

        statistics_type = kwargs.get(
            "statistics_type", "dataset"
        )  # Accepted Values `dataset`, `DRA`
        stretch_type = kwargs.get(
            "stretch_type", "minmax"
        )  # Accepted Values `minmax`, `percentclip`

        cmap_fn = getattr(matplotlib.cm, cmap)
        return_fig = kwargs.get("return_fig", False)

        x_batch, y_batch = get_nbatches(data_loader, nrows)
        x_batch = torch.cat(x_batch)
        y_batch = torch.cat(y_batch)

        nrows = min(nrows, len(x_batch))

        title_font_size = 16
        if kwargs.get("top", None) is not None:
            top = kwargs.get("top")
        else:
            top = 1 - (math.sqrt(title_font_size) / math.sqrt(100 * nrows * imsize))

        # Get Predictions
        prediction_store = []
        for i in range(0, x_batch.shape[0], self._data.batch_size):
            prediction_store.extend(
                self._predict_results(
                    x_batch[i : i + self._data.batch_size],
                    box_threshold,
                    tta_prediction,
                )
            )
        pred_mask, pred_box = self._predict_postprocess(
            prediction_store, mask_threshold, box_threshold
        )

        if self._is_multispectral:
            rgb_bands = kwargs.get("rgb_bands", self._data._symbology_rgb_bands)

            e = Exception(
                "`rgb_bands` should be a valid band_order, list or tuple of length 3 or 1."
            )
            symbology_bands = []
            if not (len(rgb_bands) == 3 or len(rgb_bands) == 1):
                raise (e)
            for b in rgb_bands:
                if type(b) == str:
                    b_index = self._bands.index(b)
                elif type(b) == int:
                    self._bands[
                        b
                    ]  # To check if the band index specified by the user really exists.
                    b_index = b
                else:
                    raise (e)
                b_index = self._data._extract_bands.index(b_index)
                symbology_bands.append(b_index)

            # Denormalize X
            if self._data._do_normalize:
                x_batch = (
                    self._data._scaled_std_values[self._data._extract_bands]
                    .view(1, -1, 1, 1)
                    .to(x_batch)
                    * x_batch
                ) + self._data._scaled_mean_values[self._data._extract_bands].view(
                    1, -1, 1, 1
                ).to(
                    x_batch
                )

            # Extract RGB Bands
            symbology_x_batch = x_batch[:, symbology_bands]
            if stretch_type is not None:
                symbology_x_batch = image_batch_stretcher(
                    symbology_x_batch, stretch_type, statistics_type
                )

            # Channel first to channel last for plotting
            symbology_x_batch = symbology_x_batch.permute(0, 2, 3, 1)
            # Clamp float values to range 0 - 1
            if symbology_x_batch.mean() < 1:
                symbology_x_batch = symbology_x_batch.clamp(0, 1)
        else:
            symbology_x_batch = x_batch.permute(0, 2, 3, 1)

        # Squeeze channels if single channel (1, 224, 224) -> (224, 224)
        if symbology_x_batch.shape[-1] == 1:
            symbology_x_batch = symbology_x_batch.squeeze()

        fig, ax = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * imsize, nrows * imsize)
        )
        fig.suptitle("Ground Truth / Predictions", fontsize=title_font_size)
        for i in range(nrows):
            if nrows == 1:
                ax_i = ax
            else:
                ax_i = ax[i]

            # Ground Truth
            ax_i[0].imshow(symbology_x_batch[i].cpu())
            ax_i[0].axis("off")
            if mode in ["mask", "bbox_mask"]:
                n_instance = y_batch[i].unique().shape[0]
                y_merged = y_batch[i].max(dim=0)[0].cpu().numpy()
                try:
                    y_rgba = cmap_fn.resampled(n_instance)(y_merged)
                except:
                    y_rgba = cmap_fn._resample(n_instance)(y_merged)
                y_rgba[y_merged == 0] = 0
                y_rgba[:, :, -1] = alpha
                ax_i[0].imshow(y_rgba)
            ax_i[0].axis("off")

            # Predictions
            ax_i[1].imshow(symbology_x_batch[i].cpu())
            ax_i[1].axis("off")
            if mode in ["mask", "bbox_mask"]:
                n_instance = np.unique(pred_mask[i]).shape[0]
                try:
                    p_rgba = cmap_fn.resampled(n_instance)(pred_mask[i])
                except:
                    p_rgba = cmap_fn._resample(n_instance)(pred_mask[i])
                p_rgba[pred_mask[i] == 0] = 0
                p_rgba[:, :, -1] = alpha
                ax_i[1].imshow(p_rgba)
            if mode in ["bbox_mask", "bbox"]:
                if pred_box[i] != []:
                    for num_boxes in pred_box[i]:
                        rect = patches.Rectangle(
                            (num_boxes[0], num_boxes[1]),
                            num_boxes[2] - num_boxes[0],
                            num_boxes[3] - num_boxes[1],
                            linewidth=1,
                            edgecolor="r",
                            facecolor="none",
                        )
                        ax_i[1].add_patch(rect)
            ax_i[1].axis("off")
        plt.subplots_adjust(top=top)
        if self._device == torch.device("cuda"):
            torch.cuda.empty_cache()

        if is_arcgispronotebook():
            plt.show()
        if return_fig:
            return fig

    def average_precision_score(
        self,
        detect_thresh=0.5,
        iou_thresh=0.5,
        mean=False,
        show_progress=True,
        tta_prediction=False,
    ):
        """
        Computes average precision on the validation set for each class.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        detect_thresh           Optional float. The probability above which
                                a detection will be considered for computing
                                average precision.
        ---------------------   -------------------------------------------
        iou_thresh              Optional float. The intersection over union
                                threshold with the ground truth mask, above
                                which a predicted mask will be
                                considered a true positive.
        ---------------------   -------------------------------------------
        mean                    Optional bool. If False returns class-wise
                                average precision otherwise returns mean
                                average precision.
        ---------------------   -------------------------------------------
        tta_prediction          Optional bool. Perform test time augmentation
                                while predicting
        =====================   ===========================================
        :return: `dict` if mean is False otherwise `float`
        """
        self._check_requisites()
        if mean:
            aps = compute_class_AP(
                self,
                self._data.valid_dl,
                1,
                show_progress,
                detect_thresh,
                iou_thresh,
                mean,
                tta_prediction,
            )
            return aps
        else:
            aps = compute_class_AP(
                self,
                self._data.valid_dl,
                self._data.c - 1,
                show_progress,
                detect_thresh,
                iou_thresh,
                tta_prediction=tta_prediction,
            )
            return dict(zip(self._data.classes[1:], aps))

    def _get_transformed_image(self, image, resize=False):
        if resize:
            resize_to = None
            if self._data.chip_size is not None:
                resize_to = self._data.resize_to
            elif self._data.resize_to is not None:
                resize_to = self._data.chip_size

            if resize_to is not None:
                if isinstance(resize_to, tuple):
                    image = cv2.resize(image, resize_to)
                else:
                    image = cv2.resize(image, (resize_to, resize_to))
        return image

    def _pixel_mask_image(
        self,
        batch,
        orig_img_dim,
        min_obj_size,
        threshold,
        offsets,
        scale_ratio,
        pred_mask,
        pred_box,
        pred_class,
        pred_score,
        extra_chips=0,
        tta_prediction=False,
    ):
        predictions = self._predict_batch(
            torch.tensor(batch).float(), threshold, tta_prediction
        )

        for batch_idx in range(len(predictions) - extra_chips):
            offset = offsets[batch_idx]
            masks = predictions[batch_idx]["masks"].squeeze().detach().cpu().numpy()
            if masks.shape[0] != 0:
                if len(masks.shape) == 2:
                    masks = masks[None]

                for n, mask in enumerate(masks):
                    if predictions[batch_idx]["scores"][n].tolist() >= threshold:
                        contours = find_contours(mask, 0.5, fully_connected="high")
                        coord_list = []
                        for c_idx, contour in enumerate(contours):
                            contour[:, 0] = (contour[:, 0] + (offset[0])) * scale_ratio[
                                0
                            ]
                            contour[:, 1] = (contour[:, 1] + (offset[1])) * scale_ratio[
                                1
                            ]
                            if c_idx == 0:
                                coord_list.append(contour[:, [1, 0]].tolist())
                            else:
                                coord_list.append(
                                    list(reversed(contour[:, [1, 0]].tolist()))
                                )
                        box = (
                            predictions[batch_idx]["boxes"][n]
                            .cpu()
                            .detach()
                            .numpy()
                            .tolist()
                        )

                        box[0] = (box[0] + offset[1]) * scale_ratio[1]
                        box[2] = (box[2] + offset[1]) * scale_ratio[1]
                        box[1] = (box[1] + offset[0]) * scale_ratio[0]
                        box[3] = (box[3] + offset[0]) * scale_ratio[0]

                        if box[0] < 0:
                            box[0] = 0

                        if box[1] < 0:
                            box[1] = 0

                        if box[2] >= orig_img_dim[1]:
                            box[2] = orig_img_dim[1] - 1

                        if box[3] >= orig_img_dim[0]:
                            box[3] = orig_img_dim[0] - 1

                        box[2] -= box[0]
                        box[3] -= box[1]
                        area = box[2] * box[3]
                        if area > 0 and math.sqrt(area) >= min_obj_size:
                            pred_box.append(box)
                            pred_class.append(
                                predictions[batch_idx]["labels"][n].tolist()
                            )
                            pred_score.append(
                                predictions[batch_idx]["scores"][n].tolist()
                            )
                            pred_mask.append(coord_list)

    def _get_batched_predictions(
        self,
        chips,
        tytx,
        orig_img_dim,
        scale_ratio,
        threshold=0.5,
        nms_overlap=0.3,
        min_obj_size=1,
        batch_size=1,
        tta_prediction=False,
    ):
        data = []
        offsets = []

        pred_mask = []
        pred_box = []
        pred_class = []
        pred_score = []

        data_counter = 0
        for idx in range(len(chips)):
            chip = chips[idx]
            if self._data._is_multispectral:
                t = torch.tensor(
                    np.rollaxis(chip["chip"], -1, 0).astype(np.float32),
                    dtype=torch.float32,
                )[None]
                scaled_t = self._data._min_max_scaler(t)[0]
                frame = scaled_t[self._data._extract_bands].detach().cpu().numpy()
            else:
                frame = (
                    pil2tensor(
                        PIL.Image.fromarray(
                            cv2.cvtColor(chip["chip"], cv2.COLOR_BGR2RGB)
                        ),
                        dtype=np.float32,
                    )
                    .div_(255)
                    .detach()
                    .cpu()
                    .numpy()
                )

            data.append(frame)
            offsets.append((chip["ymin"], chip["xmin"]))

            data_counter += 1

            if data_counter % batch_size == 0 or idx == len(chips) - 1:
                batch = chips_to_batch(data, tytx, tytx, batch_size)

                self._pixel_mask_image(
                    batch,
                    orig_img_dim,
                    min_obj_size,
                    threshold,
                    offsets,
                    scale_ratio,
                    pred_mask=pred_mask,
                    pred_box=pred_box,
                    pred_class=pred_class,
                    pred_score=pred_score,
                    extra_chips=batch_size - len(data),
                    tta_prediction=tta_prediction,
                )
                data = []
                offsets = []
                data_counter = 0

        import copy

        pred_box_indices = non_max_suppression(
            copy.deepcopy(np.array(pred_box)), nms_overlap, pred_score
        )
        filtered_mask = [pred_mask[i] for i in pred_box_indices]
        filtered_box = [pred_box[i] for i in pred_box_indices]
        filtered_class = [
            self._data.class_mapping[pred_class[i]] for i in pred_box_indices
        ]
        filtered_score = [pred_score[i] for i in pred_box_indices]

        return filtered_mask, filtered_box, filtered_class, filtered_score

    def predict(
        self,
        image_path,
        threshold=0.5,
        nms_overlap=0.1,
        return_scores=True,
        visualize=False,
        resize=False,
        tta_prediction=False,
        **kwargs,
    ):
        """
        Predicts and displays the results of a trained model on a single image.
        This method is only supported for RGB images.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        image_path              Required. Path to the image file to make the
                                predictions on.
        ---------------------   -------------------------------------------
        thresh                  Optional float. The probability above which
                                a detection will be considered valid.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        ---------------------   -------------------------------------------
        return_scores           Optional boolean.
                                Will return the probability scores of the
                                bounding box predictions if True.
        ---------------------   -------------------------------------------
        visualize               Optional boolean. Displays the image with
                                predicted bounding boxes if True.
        ---------------------   -------------------------------------------
        resize                  Optional boolean. Resizes the image to the
                                same size (chip_size parameter in prepare_data)
                                that the model was trained on, before detecting
                                objects. Note that if resize_to parameter was
                                used in prepare_data, the image is resized to
                                that size instead.

                                By default, this parameter is false and the
                                detections are run in a sliding window fashion
                                by applying the model on cropped sections of
                                the image (of the same size as the model was
                                trained on).
        ---------------------   -------------------------------------------
        tta_prediction          Optional bool. Perform test time augmentation
                                while predicting
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        batch_size              Optional int. Batch size to be used
                                during tiled inferencing
        ---------------------   -------------------------------------------
        min_obj_size            Optional int. Minimum object size
                                to be detected.
        =====================   ===========================================

        :return: 'List' of xmin, ymin, width, height, labels, scores, of predicted bounding boxes on the given image
        """

        if not HAS_OPENCV:
            raise Exception(
                "This function requires opencv 4.0.1.24. Install it using pip install opencv-python==4.0.1.24"
            )

        if isinstance(image_path, str):
            import os

            if not os.path.isfile(image_path):
                return None, None, None
            if self._data._is_multispectral:
                resize_to = None
                if resize:
                    if self._data.resize_to is not None:
                        resize_to = self._data.resize_to
                    elif self._data.chip_size is not None:
                        resize_to = self._data.chip_size
                image = read_image(image_path, resize_to)
                resize = False
            else:
                image = cv2.imread(image_path)
        else:
            image = image_path

        orig_height, orig_width, _ = image.shape
        orig_frame = image.copy()

        image = self._get_transformed_image(image, resize)
        height, width, _ = image.shape

        tytx = int(kwargs.get("tile_size", self._data.chip_size))
        batch_size = int(kwargs.get("batch_size", 1))
        min_obj_size = int(kwargs.get("min_obj_size", 1))

        if not resize:
            chips = _get_image_chips(image, tytx)
        else:
            chips = [
                {
                    "width": width,
                    "height": height,
                    "xmin": 0,
                    "ymin": 0,
                    "chip": image,
                    "predictions": [],
                }
            ]

        masks, predictions, labels, scores = self._get_batched_predictions(
            chips,
            tytx,
            (orig_height, orig_width),
            (orig_height / (1.0 * height), orig_width / (1.0 * width)),
            threshold,
            nms_overlap,
            min_obj_size,
            batch_size,
            tta_prediction,
        )

        if visualize:
            if self._data._is_multispectral:
                t = torch.tensor(
                    np.rollaxis(orig_frame, -1, 0).astype(np.float32),
                    dtype=torch.float32,
                )[None]
                scaled_t = self._data._min_max_scaler(t)[0]
                orig_frame = (
                    (scaled_t * 255)
                    .round()
                    .numpy()
                    .astype(np.uint8)[self._data._symbology_rgb_bands]
                )
                orig_frame = np.rollaxis(orig_frame, 0, 3)
                a = np.zeros(orig_frame.shape, dtype=np.uint8)
                a[:] = orig_frame[:]
                if len(labels) > 0:
                    image = _draw_predictions(a, predictions, labels)
                else:
                    image = orig_frame
            else:
                image = _draw_predictions(orig_frame, predictions, labels)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.xticks([])
            plt.yticks([])
            plt.imshow(PIL.Image.fromarray(image))

        if return_scores:
            return masks, predictions, labels, scores
        else:
            return masks, predictions, labels
