from ._arcgis_model import ArcGISModel, _get_device
from ._timm_utils import timm_config, filter_timm_models
from pathlib import Path
import json
from ._codetemplate import code
import random
import statistics
import warnings
from .._data import _raise_fastai_import_error
import traceback

HAS_OPENCV = True
HAS_FASTAI = True

# Try to import the necessary modules
# Exception will turn the HAS_FASTAI flag to false so that relevant exception can be raised
try:
    import torch
    from torch import Tensor
    import numpy as np
    import pandas as pd
    import PIL
    from fastai.vision.learner import create_body
    from fastai.vision import ImageList
    from fastai.vision import imagenet_stats, normalize
    from fastai.vision.image import open_image, bb2hw, image2np, Image, pil2tensor
    from fastai.core import ifnone
    from torchvision import models
    from .._utils.pascal_voc_rectangles import (
        ObjectDetectionCategoryList,
        show_results_multispectral,
    )
    from ._retinanet_utils import (
        RetinaNetModel,
        RetinaNetFocalLoss,
        compute_class_AP,
        get_predictions,
        AveragePrecision,
        _process_bboxes_jit,
    )
    from fastai.callbacks import EarlyStoppingCallback
    from fastai.basic_train import Learner
    from ._arcgis_model import _resnet_family
    from .._image_utils import (
        _get_image_chips,
        _get_transformed_predictions,
        _draw_predictions,
        _exclude_detection,
    )
    from .._video_utils import VideoUtils
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from fastprogress.fastprogress import progress_bar
    from .._utils.env import is_arcgispronotebook
    import matplotlib.pyplot as plt
    from .._utils.utils import chips_to_batch
    from .._utils.pascal_voc_rectangles import _reconstruct
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

try:
    import cv2
except:
    HAS_OPENCV = False

from typing import Tuple, List


class RetinaNetTracer(torch.nn.Module):
    def __init__(self, model, device, sizes, ratios, scales):
        super().__init__()
        self.model = model.to(device)
        self.crit_vals = [
            torch.tensor(sizes),
            torch.tensor(ratios),
            torch.tensor(scales),
        ]

    def _process_bboxes(self, output: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        return _process_bboxes_jit(output, self.crit_vals)

    def forward(self, inp: List[Tensor]):
        out = self.model(inp)
        out_final = self._process_bboxes(out)
        return out_final


class RetinaNet(ArcGISModel):
    """
    Creates a RetinaNet Object Detector with the specified zoom scales
    and aspect ratios.
    Based on the `Fast.ai notebook <https://github.com/fastai/fastai_dev/blob/master/dev_nb/102a_coco.ipynb>`_

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    scales                  Optional list of float values. Zoom scales of anchor boxes.
    ---------------------   -------------------------------------------
    ratios                  Optional list of float values. Aspect ratios of anchor
                            boxes.
    ---------------------   -------------------------------------------
    backbone                Optional string. Backbone convolutional neural network
                            model used for feature extraction, which
                            is `resnet50` by default.
                            Supported backbones: ResNet family and specified Timm
                            models(experimental support) from :func:`~arcgis.learn.RetinaNet.backbones`.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    :return:
        :class:`~arcgis.learn.RetinaNet` Object
    """

    def __init__(
        self,
        data,
        scales=None,
        ratios=None,
        backbone=None,
        pretrained_path=None,
        *args,
        **kwargs,
    ):
        if pretrained_path is not None:
            backbone_pretrained = False
        else:
            backbone_pretrained = True

        # Set default backbone to be 'resnet50'
        if backbone is None:
            backbone = models.resnet50

        self._check_dataset_support(data)
        if not (self._check_backbone_support(backbone)):
            raise Exception(
                f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"
            )

        super().__init__(data, backbone, pretrained_path=pretrained_path, **kwargs)
        data = self._data

        n_bands = len(getattr(self._data, "_extract_bands", [0, 1, 2]))
        _backbone = self._backbone
        if hasattr(self, "_orig_backbone"):
            _backbone = self._orig_backbone

        self.name = "RetinaNet"
        self._code = code

        self.scales = ifnone(scales, [1, 2 ** (-1 / 3), 2 ** (-2 / 3)])
        self.ratios = ifnone(ratios, [1 / 2, 1, 2])
        self._n_anchors = len(self.scales) * len(self.ratios)

        self._data = data
        self._chip_size = (data.chip_size, data.chip_size)

        if "timm" in self._backbone.__module__:
            backbone_cut = timm_config(self._backbone)["cut"]
        else:
            backbone_cut = None

        # Cut-off the backbone before the penultimate layer
        self._encoder = create_body(self._backbone, backbone_pretrained, backbone_cut)

        # Initialize the model, loss function and the Learner object
        self._model = RetinaNetModel(
            self._backbone,
            backbone_pretrained,
            backbone_cut,
            n_classes=data.c - 1,
            final_bias=-4,
            chip_size=self._chip_size,
            n_anchors=self._n_anchors,
            n_bands=n_bands,
        )
        self._loss_f = RetinaNetFocalLoss(
            sizes=self._model.sizes,
            scales=self.scales,
            ratios=self.ratios,
            device=self._device,
        )
        self.learn = Learner(data, self._model, loss_func=self._loss_f)
        self.learn.metrics = [AveragePrecision(self, data.c - 1)]
        if (
            "resnet" in self._backbone.__name__
            and "timm" not in self._backbone.__module__
        ):
            self.learn.split([self._model.encoder[6], self._model.c5top5])
        else:
            self.learn.split([self._model.c5top5])
        self.learn.freeze()
        if pretrained_path is not None:
            self.load(str(pretrained_path))
        self._arcgis_init_callback()  # make first conv weights learnable

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    def _save_device_model(self, model, device, save_path):
        model.eval()
        if hasattr(self._data, "chip_size"):
            chip_size = self._data.chip_size
            if not isinstance(chip_size, tuple):
                chip_size = (chip_size, chip_size)
        num_input_channels = list(self.learn.model.parameters())[0].shape[1]
        inp = torch.rand(1, num_input_channels, chip_size[0], chip_size[1]).to(device)

        ratios = [float(ratio) for ratio in self.ratios]
        scales = [float(ratio) for ratio in self.scales]
        sizes = self._model.sizes
        model = RetinaNetTracer(model, device, sizes, ratios, scales)
        model = model.to(device)
        traced_model = None
        with torch.no_grad():
            traced_model = self._trace(model, inp, True)  # TODO: more ease of use
        torch.jit.save(traced_model, save_path)

        return traced_model

    def _save_pytorch_torchscript(self, name, save=True):
        traced_model_cpu = None
        traced_model_gpu = None

        model = self.learn.model
        model.eval()
        device = self._device

        cpu = torch.device("cpu")
        save_path_cpu = (
            self.learn.path / self.learn.model_dir / f"{name}-cpu.pt"
        ).__str__()
        traced_model_cpu = self._save_device_model(model, cpu, save_path_cpu)
        save_path_cpu = f"{name}-cpu.pt"

        save_path_gpu = ""
        if torch.cuda.is_available():
            gpu = torch.device("cuda")
            save_path_gpu = (
                self.learn.path / self.learn.model_dir / f"{name}-gpu.pt"
            ).__str__()
            traced_model_gpu = self._save_device_model(model, gpu, save_path_gpu)
            save_path_gpu = f"{name}-gpu.pt"

        model.to(device)

        if not save:
            return [traced_model_cpu, traced_model_gpu]
        return [save_path_cpu, save_path_gpu]

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

        converter = tf.lite.TFLiteConverter.from_saved_model(str(save_path_pb))
        converter.experimental_new_converter = True
        converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]
        converter.target_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tflite_model = converter.convert()
        with tf.io.gfile.GFile(save_path_tflite, "wb") as f:
            f.write(tflite_model)

        return [save_path_tflite, save_path_onnx]

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "average_precision"]

    # Return a list of supported backbones names
    @property
    def supported_backbones(self):
        """Supported list of backbones for this model."""
        return RetinaNet._supported_backbones()

    @staticmethod
    def backbones():
        """Supported list of backbones for this model."""
        return RetinaNet._supported_backbones()

    @staticmethod
    def _supported_backbones():
        timm_models = filter_timm_models(["*repvgg*", "*tresnet*"])
        timm_backbones = list(map(lambda m: "timm:" + m, timm_models))
        return [*_resnet_family] + timm_backbones

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return RetinaNet._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["PASCAL_VOC_rectangles", "KITTI_rectangles"]

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISObjectDetector.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISObjectDetector.py"
        _emd_template["ModelConfiguration"] = "_RetinaNet_Inference"
        _emd_template["ModelType"] = "ObjectDetection"
        _emd_template["ExtractBands"] = [0, 1, 2]
        _emd_template["ModelParameters"] = {}
        _emd_template["ModelParameters"][
            "scales"
        ] = (
            self._loss_f.scales
        )  # Scales and Ratios are attributes of RetinaNetFocalLoss object _loss_f
        _emd_template["ModelParameters"]["ratios"] = self._loss_f.ratios
        _emd_template["Classes"] = []

        class_data = {}
        for i, class_name in enumerate(
            self._data.classes[1:]
        ):  # 0th index is background
            inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)]
            class_data["Color"] = color
            _emd_template["Classes"].append(class_data.copy())

        return _emd_template

    @property
    def _model_metrics(self):
        return {"accuracy": self.average_precision_score(show_progress=True)}

    def _analyze_pred(
        self, pred, thresh=0.5, nms_overlap=0.1, ret_scores=True, device=None
    ):
        return get_predictions(
            pred, crit=self._loss_f, detect_thresh=thresh, nms_overlap=nms_overlap
        )

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a RetinaNet Object Detector from an Esri Model Definition (EMD) file.

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

        :return:
            :class:`~arcgis.learn.RetinaNet` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)
        emd_path = _get_emd_path(emd_path)
        emd = json.load(open(emd_path))
        model_file = Path(emd["ModelFile"])
        chip_size = emd["ImageWidth"]

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}

        resize_to = emd.get("resize_to")
        if isinstance(resize_to, list):
            resize_to = (resize_to[0], resize_to[1])

        data_passed = True
        # Create an image databunch for when loading the model using emd (without training data)
        if data is None:
            data_passed = False
            train_tfms = []
            val_tfms = []
            ds_tfms = (train_tfms, val_tfms)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)

                sd = ImageList([], path=emd_path.parent.parent.parent).split_by_idx([])
                data = (
                    sd.label_const(
                        0,
                        label_cls=ObjectDetectionCategoryList,
                        classes=list(class_mapping.values()),
                    )
                    .transform(ds_tfms)
                    .databunch(device=_get_device())
                    .normalize(imagenet_stats)
                )

            data.chip_size = chip_size
            data.class_mapping = class_mapping
            data.classes = ["background"] + list(class_mapping.values())
            data = get_multispectral_data_params_from_emd(data, emd)
            # Add 1 for background class
            data.c += 1
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd

        data.resize_to = resize_to
        ret = cls(data, **emd["ModelParameters"], pretrained_path=model_file)

        if not data_passed:
            ret.learn.data.single_ds.classes = ret._data.classes
            ret.learn.data.single_ds.y.classes = ret._data.classes

        return ret

    def show_results(self, rows=5, thresh=0.5, nms_overlap=0.1):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        ---------------------   -------------------------------------------
        thresh                  Optional float. The probability above which
                                a detection will be considered valid.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        =====================   ===========================================

        """
        self._check_requisites()

        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)

        self.learn.show_results(
            rows=rows, thresh=thresh, nms_overlap=nms_overlap, model=self
        )
        if is_arcgispronotebook():
            plt.show()

    def _show_results_multispectral(
        self, rows=5, thresh=0.3, nms_overlap=0.1, alpha=1, **kwargs
    ):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        ---------------------   -------------------------------------------
        thresh                  Optional float. The probability above which
                                a detection will be considered valid.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        ---------------------   -------------------------------------------
        alpha                   Optional Float.
                                Opacity of the lables for the corresponding
                                images. Values range between 0 and 1, where
                                1 means opaque.
        =====================   ===========================================

        """
        return_fig = kwargs.get("return_fig", False)
        ret_val = show_results_multispectral(
            self,
            nrows=rows,
            thresh=thresh,
            nms_overlap=nms_overlap,
            alpha=alpha,
            **kwargs,
        )
        if return_fig:
            fig, ax = ret_val
            return fig

    def predict_video(
        self,
        input_video_path,
        metadata_file,
        threshold=0.5,
        nms_overlap=0.1,
        track=False,
        visualize=False,
        output_file_path=None,
        multiplex=False,
        multiplex_file_path=None,
        tracker_options={
            "assignment_iou_thrd": 0.3,
            "vanish_frames": 40,
            "detect_frames": 10,
        },
        visual_options={
            "show_scores": True,
            "show_labels": True,
            "thickness": 2,
            "fontface": 0,
            "color": (255, 255, 255),
        },
        resize=False,
    ):
        """
        Runs prediction on a video and appends the output VMTI predictions in the metadata file.
        This method is only supported for RGB images.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        input_video_path        Required. Path to the video file to make the
                                predictions on.
        ---------------------   -------------------------------------------
        metadata_file           Required. Path to the metadata csv file where
                                the predictions will be saved in VMTI format.
        ---------------------   -------------------------------------------
        threshold               Optional float. The probability above which
                                a detection will be considered.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        ---------------------   -------------------------------------------
        track                   Optional bool. Set this parameter as True to
                                enable object tracking.
        ---------------------   -------------------------------------------
        visualize               Optional boolean. If True a video is saved
                                with prediction results.
        ---------------------   -------------------------------------------
        output_file_path        Optional path. Path of the final video to be saved.
                                If not supplied, video will be saved at path input_video_path
                                appended with _prediction.
        ---------------------   -------------------------------------------
        multiplex               Optional boolean. Runs Multiplex using the VMTI detections.
        ---------------------   -------------------------------------------
        multiplex_file_path     Optional path. Path of the multiplexed video to be saved.
                                By default a new file with _multiplex.MOV extension is saved
                                in the same folder.
        ---------------------   -------------------------------------------
        tracking_options        Optional dictionary. Set different parameters for
                                object tracking. assignment_iou_thrd parameter is used
                                to assign threshold for assignment of trackers,
                                vanish_frames is the number of frames the object should
                                be absent to consider it as vanished, detect_frames
                                is the number of frames an object should be detected
                                to track it.
        ---------------------   -------------------------------------------
        visual_options          Optional dictionary. Set different parameters for
                                visualization.
                                show_scores boolean, to view scores on predictions,
                                show_labels boolean, to view labels on predictions,
                                thickness integer, to set the thickness level of box,
                                fontface integer, fontface value from opencv values,
                                color tuple (B, G, R), tuple containing values between
                                0-255.
        ---------------------   -------------------------------------------
        resize                  Optional boolean. Resizes the video frames to the same size
                                (chip_size parameter in prepare_data) that the model was
                                trained on, before detecting objects. Note that if
                                resize_to parameter was used in prepare_data,
                                the video frames are resized to that size instead.

                                By default, this parameter is false and the detections
                                are run in a sliding window fashion by applying the
                                model on cropped sections of the frame (of the same
                                size as the model was trained on).
        =====================   ===========================================

        """

        VideoUtils.predict_video(
            self,
            input_video_path,
            metadata_file,
            threshold,
            nms_overlap,
            track,
            visualize,
            output_file_path,
            multiplex,
            multiplex_file_path,
            tracker_options,
            visual_options,
            resize,
        )

    def _predict_batch(self, images):
        model = self.learn.model
        model.eval()
        model = model.to(self._device)
        normed_batch_tensor = images.to(self._device)
        predictions = model(normed_batch_tensor)
        normed_batch_tensor.detach().cpu()
        del normed_batch_tensor
        return predictions

    def _get_batched_predictions(self, chips, tytx, norm, batch_size=1):
        data = []
        data_counter = 0
        final_class = []
        final_bbox = []
        for idx in range(len(chips)):
            chip = chips[idx]
            frame = np.moveaxis(
                norm(cv2.cvtColor(chip["chip"], cv2.COLOR_BGR2RGB)), -1, 0
            )
            data.append(frame)
            data_counter += 1
            if data_counter % batch_size == 0 or idx == len(chips) - 1:
                batch = chips_to_batch(data, tytx, tytx, batch_size)
                batch_classes, batch_bboxes = self._predict_batch(
                    torch.tensor(batch).float()
                )
                extra_chips = batch_size - len(data)
                batch_output_class = (
                    batch_classes[: (len(batch_classes) - extra_chips)].detach().cpu()
                )
                batch_output_bbox = (
                    batch_bboxes[: (len(batch_bboxes) - extra_chips)].detach().cpu()
                )
                final_class.append(batch_output_class)
                final_bbox.append(batch_output_bbox)
                data = []
                data_counter = 0
        return torch.cat(final_class), torch.cat(final_bbox)

    def predict(
        self,
        image_path,
        threshold=0.5,
        nms_overlap=0.1,
        return_scores=True,
        visualize=False,
        resize=False,
        batch_size=1,
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
        batch_size              Optional int. Batch size to be used
                                during tiled inferencing. Deafult value 1.
        =====================   ===========================================

        :return: 'List' of xmin, ymin, width, height of predicted bounding boxes on the given image
        """

        if not HAS_OPENCV:
            raise Exception(
                "This function requires opencv 4.0.1.24. Install it using pip install opencv-python==4.0.1.24"
            )

        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path

        if image is None:
            raise Exception(str("No such file or directory: %s" % (image_path)))

        orig_height, orig_width, _ = image.shape
        orig_frame = image.copy()

        if resize and self._data.resize_to is None and self._data.chip_size is not None:
            image = cv2.resize(image, (self._data.chip_size, self._data.chip_size))

        if self._data.resize_to is not None:
            if isinstance(self._data.resize_to, tuple):
                image = cv2.resize(image, self._data.resize_to)
            else:
                image = cv2.resize(image, (self._data.resize_to, self._data.resize_to))

        height, width, _ = image.shape
        tytx = self._data.chip_size

        if self._data.chip_size is not None:
            chips = _get_image_chips(image, self._data.chip_size)
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

        valid_tfms = self._data.valid_ds.tfms
        self._data.valid_ds.tfms = []

        include_pad_detections = False
        if len(chips) == 1:
            include_pad_detections = True

        imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mean = 255 * np.array(imagenet_stats[0], dtype=np.float32)
        std = 255 * np.array(imagenet_stats[1], dtype=np.float32)
        norm = lambda x: (x - mean) / std

        from .._utils.pascal_voc_rectangles import modified_getitem
        from fastai.data_block import LabelList

        orig_getitem = LabelList.__getitem__
        LabelList.__getitem__ = modified_getitem

        try:
            pred_class, pred_bbox = self._get_batched_predictions(
                chips, tytx, norm, batch_size
            )

            class dummy:
                pass

            dummy_x = dummy()
            dummy_x.size = [tytx, tytx]
            for chip_idx, (pc, pb) in enumerate(zip(pred_class, pred_bbox)):
                pc = pc.detach().clone()
                pb = pb.detach().clone()
                pp_output = self._analyze_pred(
                    pred=(pc, pb), thresh=threshold, nms_overlap=nms_overlap
                )
                bbox = _reconstruct(
                    pp_output, dummy_x, pad_idx=0, classes=self._data.classes
                )
                if bbox is not None:
                    scores = bbox.scores
                    bboxes, lbls = bbox._compute_boxes()
                    bboxes.add_(1).mul_(
                        torch.tensor(
                            [
                                chips[chip_idx]["height"] / 2,
                                chips[chip_idx]["width"] / 2,
                                chips[chip_idx]["height"] / 2,
                                chips[chip_idx]["width"] / 2,
                            ]
                        )
                    ).long()
                    for index, bbox in enumerate(bboxes):
                        if lbls is not None:
                            label = lbls[index]
                        else:
                            label = "Default"

                        data = bb2hw(bbox)
                        if include_pad_detections or not _exclude_detection(
                            (data[0], data[1], data[2], data[3]),
                            chips[chip_idx]["width"],
                            chips[chip_idx]["height"],
                        ):
                            chips[chip_idx]["predictions"].append(
                                {
                                    "xmin": data[0],
                                    "ymin": data[1],
                                    "width": data[2],
                                    "height": data[3],
                                    "score": float(scores[index]),
                                    "label": label,
                                }
                            )
        finally:
            LabelList.__getitem__ = orig_getitem

        self._data.valid_ds.tfms = valid_tfms

        predictions, labels, scores = _get_transformed_predictions(chips)

        # Scale the predictions to original image and clip the predictions to image dims
        y_ratio = orig_height / height
        x_ratio = orig_width / width
        for index, prediction in enumerate(predictions):
            prediction[0] = prediction[0] * x_ratio
            prediction[1] = prediction[1] * y_ratio
            prediction[2] = prediction[2] * x_ratio
            prediction[3] = prediction[3] * y_ratio

            # Clip xmin
            if prediction[0] < 0:
                prediction[2] = prediction[2] + prediction[0]
                prediction[0] = 1

            # Clip width when xmax greater than original width
            if prediction[0] + prediction[2] > orig_width:
                prediction[2] = (prediction[0] + prediction[2]) - orig_width

            # Clip ymin
            if prediction[1] < 0:
                prediction[3] = prediction[3] + prediction[1]
                prediction[1] = 1

            # Clip height when ymax greater than original height
            if prediction[1] + prediction[3] > orig_height:
                prediction[3] = (prediction[1] + prediction[3]) - orig_height

            predictions[index] = [
                prediction[0],
                prediction[1],
                prediction[2],
                prediction[3],
            ]

        if visualize:
            image = _draw_predictions(orig_frame, predictions, labels)
            import matplotlib.pyplot as plt

            plt.xticks([])
            plt.yticks([])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(PIL.Image.fromarray(image))

        if return_scores:
            return predictions, labels, scores
        else:
            return predictions, labels

    def average_precision_score(
        self, detect_thresh=0.5, iou_thresh=0.1, mean=False, show_progress=True
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
                                threshold with the ground truth labels, above
                                which a predicted bounding box will be
                                considered a true positive.
        ---------------------   -------------------------------------------
        mean                    Optional bool. If False returns class-wise
                                average precision otherwise returns mean
                                average precision.
        =====================   ===========================================

        :return: `dict` if mean is False otherwise `float`
        """
        self._check_requisites()
        aps = compute_class_AP(
            self,
            self._data.valid_dl,
            self._data.c - 1,
            show_progress,
            detect_thresh=detect_thresh,
            iou_thresh=iou_thresh,
        )
        if mean:
            return statistics.mean(aps)
        else:
            return dict(zip(self._data.classes[1:], aps))
