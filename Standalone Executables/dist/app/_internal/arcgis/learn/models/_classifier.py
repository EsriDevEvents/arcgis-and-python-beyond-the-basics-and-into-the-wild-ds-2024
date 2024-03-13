import arcgis as _arcgis
from ._arcgis_model import ArcGISModel, _change_tail
from ..._impl.common._deprecate import deprecated
from .._data import _check_esri_files, _raise_fastai_import_error
import random
import math
import traceback

try:
    import pandas
    import gc
    import tempfile
    import numpy as np
    import json
    import os
    import io
    import shutil
    import warnings
    from pathlib import Path
    from ._codetemplate import feature_classifier_prf
    import torch
    import torch.nn.functional as F
    from torchvision import models
    import fastai
    from fastai.metrics import accuracy, MultiLabelFbeta
    from fastai.callbacks import *
    from fastai.vision import Image
    import fastai
    from .._utils.metrics import accuracy_multi
    from fastai.vision.image import open_image
    from fastai.data_block import MultiCategoryList
    from fastai.vision.data import ImageDataBunch, ImageList
    from fastai.vision import imagenet_stats, normalize, flatten_model
    from fastai.torch_core import split_model_idx
    from fastai.basic_train import Learner, LearnerCallback
    from torch.utils.data.sampler import WeightedRandomSampler
    from fastai.vision.learner import (
        cnn_learner,
        ClassificationInterpretation,
        cnn_config,
    )
    from ._arcgis_model import _set_multigpu_callback, _resnet_family, _get_device
    from fastai.vision.transform import (
        crop,
        rotate,
        dihedral_affine,
        brightness,
        contrast,
    )
    import glob
    import time
    import xml.etree.ElementTree as ElementTree
    import PIL.Image
    import PIL.ExifTags
    from torch.nn import Module as NnModule
    from .._utils.common import (
        get_multispectral_data_params_from_emd,
        _get_emd_path,
    )
    from .._utils.env import is_arcgispronotebook
    from matplotlib import pyplot as plt
    from .._utils.image_classification import adapt_fastai_databunch
    import copy
    import timm
    from ._timm_utils import (
        timm_config,
        filter_timm_models,
        _get_feature_size,
        test_cnn_trnsfrmr,
        gradcam_trnsfrmr,
        reshape_tensor,
        complete_transformer_backbone_name,
    )
    from fastai.vision import learner

    learner._test_cnn = test_cnn_trnsfrmr
    ClassificationInterpretation.GradCAM = gradcam_trnsfrmr

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )

    class NnModule:
        pass

    HAS_FASTAI = False

HAS_ARCPY = True
try:
    from arcgis.auth.tools import LazyLoader

    arcpy = LazyLoader("arcpy")
except Exception:
    HAS_ARCPY = False


def _mobilenet_split(m: NnModule):
    return m[0][0][0], m[1]


def _prediction_function(predictions):
    classes = {}
    max_prediction_value = 0
    max_prediction_class = None
    for prediction in predictions:
        if not classes.get(prediction[0]):
            classes[prediction[0]] = prediction[1]
        else:
            classes[prediction[0]] = classes[prediction[0]] + prediction[1]
        if max_prediction_value < classes[prediction[0]]:
            max_prediction_value = classes[prediction[0]]
            max_prediction_class = prediction[0]

    return max_prediction_class, max_prediction_value


class FeatureClassifierTF(torch.nn.Module):
    def __init__(self, head):
        super(FeatureClassifierTF, self).__init__()
        self._head = head

    def forward(self, x):
        x = self._head(x)
        x = torch.nn.functional.softmax(x[0], dim=0)
        return x


class FeatureClassifier(ArcGISModel):
    """
    Creates an image classifier to classify the area occupied by a
    geographical feature based on the imagery it overlaps with.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    backbone                Optional string. Backbone convolutional neural network
                            model used for feature extraction, which is ``resnet34``
                            by default.
                            Supported backbones: ResNet family and specified Timm
                            models(experimental support) from :func:`~arcgis.learn.FeatureClassifier.backbones`.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    ---------------------   -------------------------------------------
    mixup                   Optional boolean. If set to True, it creates
                            new training images by randomly mixing training set images.

                            The default is set to False.
    ---------------------   -------------------------------------------
    oversample              Optional boolean. If set to True, it oversamples unbalanced
                            classes of the dataset during training. Not supported with
                            MultiLabel dataset.
    ---------------------   -------------------------------------------
    backend                 Optional string. Controls the backend framework to be used
                            for this model, which is 'pytorch' by default.

                            valid options are "``pytorch``", "``tensorflow``"
    =====================   ===========================================

    :return: :class:`~arcgis.learn.FeatureClassifier` Object
    """

    def __init__(
        self,
        data,
        backbone="resnet34",
        pretrained_path=None,
        mixup=False,
        oversample=False,
        backend="pytorch",
        *args,
        **kwargs,
    ):
        # condition when databunch is from fastai
        # it will not contain class_mapping
        if not hasattr(data, "class_mapping"):
            data = adapt_fastai_databunch(data)

        self._free_memory()
        backbone = complete_transformer_backbone_name(backbone, data.chip_size)
        if not (
            self._check_backbone_support(backbone)
            or backbone in self._transformer_backbone_original_names()
        ):
            raise Exception(
                f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"
            )

        self._check_dataset_support(data)

        self._backend = backend
        if self._backend == "tensorflow":
            super().__init__(data, None)
            self._intialize_tensorflow(data, backbone, pretrained_path, mixup, kwargs)
        else:
            super().__init__(data, backbone, pretrained_path=pretrained_path, **kwargs)
            data = self._data

            backbone_cut = None
            backbone_split = None

            _backbone = self._backbone
            if hasattr(self, "_orig_backbone"):
                _backbone = self._orig_backbone
                _backbone_meta = cnn_config(self._orig_backbone)
                backbone_cut = _backbone_meta["cut"]
                backbone_split = _backbone_meta["split"]

            if _backbone == models.mobilenet_v2:
                backbone_cut = -1
                backbone_split = _mobilenet_split

            self._code = feature_classifier_prf

            if getattr(data, "_dataset_type", "Labeled_Tiles") == "MultiLabeled_Tiles":
                # ToDo: allow option to change `thresh` parameter by user
                accuracy_multi.__name__ = "accuracy"

                class MultLabelFbetaModified(MultiLabelFbeta):
                    def fbeta_score(self, precision, recall):
                        beta2 = self.beta**2
                        fbeta = (
                            (1 + beta2)
                            * (precision * recall)
                            / ((beta2 * precision + recall) + self.eps)
                        )
                        if isinstance(fbeta, torch.Tensor):
                            if fbeta.is_cuda:
                                fbeta = fbeta.cpu()
                        return fbeta

                MultLabelFbetaModified.__name__ = "MultiLabelFbeta"
                metrics = [accuracy_multi, MultLabelFbetaModified()]
            else:
                metrics = accuracy

            if "timm" in self._backbone.__module__:
                timm_meta = timm_config(self._backbone)
                backbone_cut = timm_meta["cut"]
                backbone_split = timm_meta["split"]

            if "tresnet" in self._backbone.__module__:
                from fastai.vision import create_head

                nf = 2 * _get_feature_size(self._backbone, backbone_cut)[-1][1]
                head = create_head(nf, data.c)
            else:
                head = None

            self._transformer = (
                type(backbone) is str
                and backbone in FeatureClassifier._transformer_backbone_original_names()
            )

            if self._transformer:
                from ._timm_utils import create_transformer_FeatureClassifier

                trnsfrmr_model = create_transformer_FeatureClassifier(
                    self._backbone.__name__,
                    num_classes=data.c,
                    img_size=self._data.chip_size,
                    pretrained=True,
                )
                if self._is_multispectral:
                    trnsfrmr_model = _change_tail(trnsfrmr_model, data)

                self.learn = Learner(
                    data,
                    model=trnsfrmr_model,
                    metrics=metrics,
                )
                idx = self._freeze()
                if trnsfrmr_model[0].__class__.__name__ == "CoaT":
                    idx = 8
                self.learn.layer_groups = split_model_idx(self.learn.model, [idx])
                self.learn.create_opt(lr=3e-3)
            else:
                self.learn = cnn_learner(
                    data,
                    self._backbone,
                    metrics=metrics,
                    cut=backbone_cut,
                    split_on=backbone_split,
                    custom_head=head,
                )
            if oversample:
                self.learn.callbacks.append(OverSamplingCallback(self.learn))
            self._arcgis_init_callback()  # make first conv weights learnable

            # Add Mixup data augmentation
            if mixup:
                # For mixup to work with multilabel call it with parameter stack_y=False
                stack_y = (
                    getattr(data, "_dataset_type", "Labeled_Tiles") == "Labeled_Tiles"
                )
                if (
                    getattr(data, "_dataset_type", "Labeled_Tiles") == "Labeled_Tiles"
                ) or (
                    getattr(data, "_dataset_type", "Labeled_Tiles")
                    == "MultiLabeled_Tiles"
                ):
                    self.learn = self.learn.mixup(stack_y=stack_y)
                else:
                    self.learn = self.learn.mixup()

            self.learn.model = self.learn.model.to(self._device)

            _set_multigpu_callback(self)
            if pretrained_path is not None:
                self.load(pretrained_path)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    def _freeze(self):
        layers = flatten_model(self.learn.model[0])
        idx = len(layers) // 2
        start_idx = 0
        if self._is_multispectral:
            start_idx = 1
        for layer in layers[start_idx:idx]:
            if (
                isinstance(layer, (torch.nn.BatchNorm2d))
                or isinstance(layer, (fastai.torch_core.ParameterModule))
                or isinstance(layer, (torch.nn.BatchNorm1d))
                or isinstance(layer, (torch.nn.LayerNorm))
            ):
                continue
            for p in layer.parameters():
                p.requires_grad = False

        return idx

    def _free_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy"]

    @property
    def supported_backbones(self):
        """Supported list of backbones for this model."""
        return FeatureClassifier._supported_backbones()

    @staticmethod
    def transformer_backbones():
        from ._timm_utils import shortened_transformer_backbone

        transformer_model = shortened_transformer_backbone()
        transformer_model_name = [
            "*coat*",
            "*convit*",
            "*levit*",
            "*twins*",
            "*visformer*",
        ]
        for tr in transformer_model_name:
            transformer_model.extend(timm.list_models(tr, pretrained=True))
        transformer_model = sorted(transformer_model, key=lambda x: x.split("_")[0])
        transformer_model = list(map(lambda m: "timm:" + m, transformer_model))
        return transformer_model

    @staticmethod
    def _transformer_backbone_original_names():
        """Supported list of transformer backbones for this model."""
        transformer_model_name = [
            "*cait*",
            "*coat*",
            "*convit*",
            "*deit*",
            "*levit*",
            "*pit*",
            "*swin*",
            "*tnt*",
            "*twins*",
            "*visformer*",
            "vit_*",
        ]
        trnsfrmr_model = []
        for tr in transformer_model_name:
            trnsfrmr_model.extend(timm.list_models(tr, pretrained=True))
        return list(map(lambda m: "timm:" + m, trnsfrmr_model))

    @staticmethod
    def backbones():
        """Supported list of backbones for this model."""
        return FeatureClassifier._supported_backbones()

    @staticmethod
    def _supported_backbones():
        timm_models = filter_timm_models(["*repvgg*", "*tresnet*"])
        timm_backbones = list(map(lambda m: "timm:" + m, timm_models))
        transformer_backbones = FeatureClassifier.transformer_backbones()
        return [*_resnet_family, models.mobilenet_v2.__name__] + sorted(
            timm_backbones + transformer_backbones
        )

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return FeatureClassifier._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Labeled_Tiles", "MultiLabeled_Tiles", "Imagenet"]

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
        self.learn.show_results(rows=rows, **kwargs)
        if is_arcgispronotebook():
            plt.show()

    def _show_results_multispectral(self, rows=5, **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================

        """
        from .._utils.image_classification import IC_show_results

        return_fig = kwargs.get("return_fig", False)
        fig = IC_show_results(self, nrows=rows, **kwargs)
        if return_fig:
            fig1, axs = fig
            return fig1

    def predict(self, img_path, visualize=False, gradcam=False):
        """
        Runs prediction on an Image. Works with RGB images only.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        img_path                Required. Path to the image file to make the
                                predictions on.
        ---------------------   -------------------------------------------
        visualize               Optional: Set this parameter to True to
                                visualize the image being predicted.
        ---------------------   -------------------------------------------
        gradcam                 Optional: Set this parameter to True to
                                get gradcam visualization to help with
                                explanability of the prediction. If set
                                to True, visualize parameter must also
                                be set to True.
        =====================   ===========================================

        :return: prediction label and confidence
        """
        img = open_image(img_path)
        pred = self.learn.predict(img)
        if visualize == True:
            gradCam = self._gradCAM(img, pred[0], grad_vis=gradcam)
        return pred

    def _predict_batch(self, imagetensor_batch):
        predictions = (
            self.learn.model.eval()(imagetensor_batch.to(self._device)).detach().cpu()
        )
        predictions_conf, predicted_classes = torch.max(predictions, dim=-1)
        predicted_classes = predicted_classes.tolist()
        predictions_conf = (predictions_conf * 100).tolist()
        return predicted_classes, predictions_conf

    def _save_confusion_matrix(self, path):
        from IPython.utils import io

        with io.capture_output() as captured:
            self.plot_confusion_matrix()
            plt.savefig(os.path.join(path, "confusion_matrix.png"), bbox_inches="tight")
            plt.close()

    @property
    def _model_metrics(self):
        return {}

    def _save_pytorch_tflite(self, name):
        import tensorflow as tf
        import logging

        tf.get_logger().setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import onnx
            import onnx_tf
            from onnx_tf.backend import prepare

        model = self.learn.model
        model.eval()
        device = self._device
        cpu = torch.device("cpu")
        model.to(cpu)

        if hasattr(self._data, "chip_size"):
            chip_size = self._data.chip_size
            if not isinstance(chip_size, tuple):
                chip_size = (chip_size, chip_size)
        num_input_channels = len(getattr(self._data, "_extract_bands", [0, 1, 2]))
        inp = torch.randn([1, num_input_channels, chip_size[0], chip_size[1]]).to(cpu)
        inp_np = inp.detach().cpu().numpy()
        base = f"{name}-base"
        path_base_onnx = self.learn.path / self.learn.model_dir / f"{base}.onnx"
        path_save_pb = self.learn.path / self.learn.model_dir / f"{name}-pb"

        activated_model = FeatureClassifierTF(model)
        activated_model.eval()
        activated_model.to(cpu)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model=activated_model,
                args=inp,
                f=path_base_onnx,
                verbose=False,
                export_params=True,
                do_constant_folding=True,  # fold constant values for optimization
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )

            onnx_base_model = onnx.load(str(path_base_onnx))
            tf_rep_base = prepare(onnx_base_model)
            tf_rep_base.export_graph(str(path_save_pb))

        model.to(device)

        path_save_tflite = self.learn.path / self.learn.model_dir / f"{name}.tflite"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tf_model = tf.saved_model.load(str(path_save_pb))
            infer = tf_model.signatures["serving_default"]
            concrete_func = tf_model.signatures[
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            ]
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            tflite_model = converter.convert()

            # Save the model
            with open(path_save_tflite, "wb") as f:
                f.write(tflite_model)

        return [f"{name}.tflite", f"{name}-pb"]

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_FeatureClassifier"
        _emd_template["ModelType"] = "ObjectClassification"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISObjectClassifier.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISObjectClassifier.py"
        _emd_template["MetaDataMode"] = self._data._dataset_type
        _emd_template["ExtractBands"] = [0, 1, 2]
        _emd_template["CropSizeFixed"] = int(
            getattr(self._data, "_emd", {}).get("CropTileMode", "Fixed_Size")
            == "Fixed_Size"
        )
        _emd_template["BlackenAroundFeature"] = int(
            getattr(self._data, "_emd", {}).get("BlackenAroundFeature", False)
        )
        _emd_template["ImageSpaceUsed"] = "MAP_SPACE"
        _emd_template["Classes"] = []
        class_data = {}

        if self._data._dataset_type == "MultiLabeled_Tiles":
            self._data.class_mapping = {k: v for k, v in enumerate(self._data.classes)}
        inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}

        for i, class_name in enumerate(self._data.classes):
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)]
            class_data["Color"] = color
            _emd_template["Classes"].append(class_data.copy())

        # if getattr(self, '_is_multispectral', False):
        #     _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        #     _emd_template["ModelConfiguration"] = "_FeatureClassifier"
        #     if save_inference_file:
        #         _emd_template["InferenceFunction"] = "ArcGISObjectClassifier.py"
        #     else:
        #         _emd_template["InferenceFunction"] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISObjectClassifier.py"

        return _emd_template

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a Feature classifier from an Esri Model Definition (EMD) file.

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

        :return:
            :class:`~arcgis.learn.FeatureClassifier` Object

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
        chip_size = emd["ImageWidth"]

        try:
            class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}
            color_mapping = {i["Value"]: i["Color"] for i in emd["Classes"]}
        except KeyError:
            class_mapping = {i["ClassValue"]: i["ClassName"] for i in emd["Classes"]}
            color_mapping = {i["ClassValue"]: i["Color"] for i in emd["Classes"]}

        if data is None:
            ranges = (0, 1)
            train_tfms = [
                rotate(degrees=30, p=0.5),
                crop(size=chip_size, p=1.0, row_pct=ranges, col_pct=ranges),
                dihedral_affine(),
                brightness(change=(0.4, 0.6)),
                contrast(scale=(0.75, 1.5)),
                # rand_zoom(scale=(0.75, 1.5))
            ]
            val_tfms = [crop(size=chip_size, p=1.0, row_pct=0.5, col_pct=0.5)]
            transforms = (train_tfms, val_tfms)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)

                if ("MetaDataMode" in emd) and (
                    emd["MetaDataMode"] == "MultiLabeled_Tiles"
                ):
                    img_list = ImageList([], path=emd_path.parent.parent).split_by_idx(
                        []
                    )
                    data = (
                        img_list.label_const(
                            0,
                            label_cls=MultiCategoryList,
                            classes=list(class_mapping.values()),
                        )
                        .transform(transforms)
                        .databunch()
                        .normalize(imagenet_stats)
                    )
                    data._dataset_type = "MultiLabeled_Tiles"
                else:
                    data = ImageDataBunch.single_from_classes(
                        emd_path.parent.parent,
                        sorted(list(class_mapping.values())),
                        ds_tfms=transforms,
                        size=chip_size,
                    ).normalize(imagenet_stats)

            data.chip_size = chip_size
            data.class_mapping = class_mapping
            data.classes = list(class_mapping.values())
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            data = get_multispectral_data_params_from_emd(data, emd)
            data.device = _get_device()

        resize_to = emd.get("resize_to")
        data.resize_to = resize_to

        return cls(data, **model_params, pretrained_path=str(model_file))

    def plot_confusion_matrix(self, **kwargs):
        """
        Plots a confusion matrix of the model predictions to evaluate accuracy
        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        thresh                  confidence score threshold for multilabel predictions,
                                defaults to 0.5
        =====================   ===========================================
        """
        self._check_requisites()
        if self._data._dataset_type == "MultiLabeled_Tiles":
            # Get x, y from validation dataset
            data_loader = self._data.valid_dl
            nbatches = math.ceil(len(self._data.valid_ds) / self._data.batch_size)
            from .._utils.common import get_nbatches

            x_batch, y_batch = get_nbatches(data_loader, nbatches)
            x_batch = torch.cat(x_batch)
            y_batch = torch.cat(y_batch)
            score_thresh = kwargs.get("thresh", 0.5)

            # Get predictions
            predictions = []
            learn_temp = copy.copy(self.learn)
            for i in range(0, x_batch.shape[0], self._data.batch_size):
                batch_preds = learn_temp.pred_batch(
                    batch=(
                        x_batch[i : i + self._data.batch_size],
                        y_batch[i : i + self._data.batch_size],
                    )
                )
                predictions.append(batch_preds)
            predictions = torch.cat(predictions)
            one_hot_preds = predictions >= score_thresh

            # Use Scikit-learn multilabel confusion matrix
            from sklearn.metrics import multilabel_confusion_matrix

            y_true = y_batch.to("cpu").numpy()
            y_pred = one_hot_preds.to("cpu").numpy()
            confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

            # Plot the classwise confusion matrix
            nrows = self._data.c
            plt_size = 4
            fig, axs = plt.subplots(nrows=nrows, figsize=(plt_size, (nrows) * plt_size))
            fig.suptitle("Confusion Matrix", fontsize=16)
            top = 1 - (math.sqrt(16) / math.sqrt(100 * nrows * plt_size))
            fig.subplots_adjust(top=top, hspace=0.5)

            for i, (classname, matrix) in enumerate(
                zip(self._data.classes, confusion_matrix)
            ):
                cm = np.fliplr(np.flipud(matrix))
                axi = axs[i]
                cmap = "Blues"
                axi.imshow(cm, interpolation="nearest", cmap=cmap)
                title = classname
                axi.set_title(title)
                tick_marks = np.arange(2)
                axi.set_xticks(ticks=tick_marks)
                axi.set_xticklabels([classname, "Rest"])
                axi.set_yticks(ticks=tick_marks)
                axi.set_yticklabels([classname, "Rest"])
                axi.set_ylabel("Actual")
                axi.set_xlabel("Predicted")
                axi.grid(False)

                import itertools

                thresh = cm.max() / 2.0
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    coeff = f"{cm[i, j]}"
                    axi.text(
                        j,
                        i,
                        coeff,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )

        # For single label classification
        else:
            self._check_requisites()
            learn_temp = copy.copy(self.learn)

            # Reassigning the function from vision.learner because fastai sets it from tabular.learner
            from fastai.vision.learner import _cl_int_from_learner

            ClassificationInterpretation.from_learner = _cl_int_from_learner
            interp = ClassificationInterpretation.from_learner(learn_temp)

            nrows = self._data.c
            # figsize range: 4 <= (no. of classes + 15)/4 <=20
            fs = min(max(4, (nrows + 15) / 4), 20)
            interp.plot_confusion_matrix(figsize=(fs, fs))

    def plot_hard_examples(self, num_examples):
        """
        Plots the hard examples with their heatmaps.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        num_examples            Number of hard examples to plot
                                :meth:`~arcgis.learn.prepare_data`  function.
        =====================   ===========================================
        """
        self._check_requisites()
        # handling bug in fastai.
        if num_examples == 1:
            num_examples = 2
        learn_temp = copy.copy(self.learn)
        from arcgis.learn._utils.labeled_tiles import plot_multi_top_losses_modified

        ClassificationInterpretation.plot_multi_top_losses = (
            plot_multi_top_losses_modified
        )
        interp = ClassificationInterpretation.from_learner(learn_temp)
        heatmap = True
        if self._backend == "tensorflow":
            heatmap = False
        if self._data._dataset_type == "MultiLabeled_Tiles":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Add plt.show to avoid issues from previous plots from other method
                    try:
                        plt.show()
                    except:
                        pass
                    interp.plot_multi_top_losses(num_examples, figsize=(5, 5))
            except IndexError:
                from IPython.display import clear_output

                clear_output(wait=True)
                print("No mismatches found.")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = interp.plot_top_losses(
                num_examples, figsize=(15, 15), heatmap=heatmap, return_fig=True
            )
        # fastai way of calculating num nrows and ncols
        cols = math.ceil(math.sqrt(num_examples))
        rows = math.ceil(num_examples / cols)
        axes = fig.axes
        # get number of empty axes from behind.
        num_empty_ax = rows * cols - num_examples
        # delete those from back.
        for k in range(num_empty_ax):
            fig.delaxes(axes[-(k + 1)])

    @staticmethod
    def _convert_to_degrees(value, reference):
        d0 = value[0][0]
        d1 = value[0][1]
        d = float(d0) / float(d1)

        m0 = value[1][0]
        m1 = value[1][1]
        m = float(m0) / float(m1)

        s0 = value[2][0]
        s1 = value[2][1]
        s = float(s0) / float(s1)

        degrees = d + (m / 60.0) + (s / 3600.0)

        if reference == "S" or reference == "W":
            degrees = 0 - degrees

        return degrees

    def predict_folder_and_create_layer(
        self,
        folder,
        feature_layer_name,
        gis=None,
        prediction_field="predict",
        confidence_field="confidence",
    ):
        """
        Predicts on images present in the given folder and creates a feature layer.
        The images stored in the folder contain GPS information as part of EXIF metadata.
        Works with RGB images only.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        folder                  Required String. Folder containing images to inference on.
        ---------------------   -------------------------------------------
        feature_layer_name      Required String. The name of the feature layer used to publish.
        ---------------------   -------------------------------------------
        gis                     Optional :class:`~arcgis.gis.GIS`  Object, the GIS on which this tool runs. If not specified,
                                the active GIS is used.
        ---------------------   -------------------------------------------
        prediction_field        Optional String. The field name to use to add predictions.
        ---------------------   -------------------------------------------
        confidence_field        Optional String. The field name to use to add confidence.
        =====================   ===========================================

        :return: :class:`~arcgis.features.FeatureCollection` Object
        """
        return self._create_feature_layer(
            self._extract_images_geo_data(folder),
            _arcgis.env.active_gis if gis is None else gis,
            feature_layer_name,
            prediction_field,
            confidence_field,
        )

    def _extract_images_geo_data(self, folder):
        ALLOWED_FILE_FORMATS = ["tif", "jpg", "png"]

        files = []

        for ext in ALLOWED_FILE_FORMATS:
            files.extend(glob.glob(os.path.join(folder, "*." + ext)))

        images_data = []

        for file in files:
            img = PIL.Image.open(file)
            exif = {
                PIL.ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in PIL.ExifTags.TAGS
            }

            images_data.append(
                {
                    "image_path": file,
                    "y": FeatureClassifier._convert_to_degrees(
                        exif["GPSInfo"][2], exif["GPSInfo"][1]
                    ),
                    "x": FeatureClassifier._convert_to_degrees(
                        exif["GPSInfo"][4], exif["GPSInfo"][3]
                    ),
                }
            )

        return images_data

    def _create_feature_layer(
        self,
        images_data,
        gis_user,
        feature_layer_name,
        prediction_field,
        confidence_field,
    ):
        data = []
        images = {}
        for image_data in images_data:
            prediction = self.predict(image_data["image_path"])
            images[os.path.basename(image_data["image_path"])] = image_data[
                "image_path"
            ]
            data.append(
                [
                    os.path.basename(image_data["image_path"]),
                    prediction[0].obj,
                    prediction[2].data.max().tolist(),
                    image_data["x"],
                    image_data["y"],
                ]
            )

        dataframe = pandas.DataFrame(
            data, columns=["image_name", prediction_field, confidence_field, "X", "Y"]
        )
        spatial_dataframe = dataframe.spatial.from_xy(
            df=dataframe, sr=4326, x_column="X", y_column="Y"
        )

        feature_collection = gis_user.content.import_data(
            spatial_dataframe, title=feature_layer_name
        )

        feature_layer = feature_collection.layers[0]
        feature_layer.manager.add_to_definition({"hasAttachments": True})

        df = feature_layer.query(as_df=True)
        object_field = feature_layer.properties["objectIdField"]

        for image_name, image_path in images.items():
            object_id = df.loc[df["image_name"] == image_name, object_field].values[
                0
            ]  # assuming image_name is unique
            if np.isnan(object_id):
                continue  # skipping those values which are not present.
            feature_layer.attachments.add(object_id, image_path)

        return feature_collection

    @staticmethod
    def _update_predictions_layer(
        feature_layer, features_to_update, output_label_field, confidence_field=None
    ):
        field_template = {
            "name": output_label_field,
            "type": "esriFieldTypeString",
            "alias": output_label_field,
            "sqlType": "sqlTypeOther",
            "length": 256,
            "nullable": True,
            "editable": True,
            "visible": True,
            "domain": None,
            "defaultValue": "",
        }

        confidence_field_template = {
            "name": confidence_field,
            "type": "esriFieldTypeString",
            "alias": confidence_field,
            "sqlType": "sqlTypeOther",
            "length": 256,
            "nullable": True,
            "editable": True,
            "visible": True,
            "domain": None,
            "defaultValue": "",
        }

        feature_layer.manager.add_to_definition({"fields": [field_template]})

        if confidence_field:
            feature_layer.manager.add_to_definition(
                {"fields": [confidence_field_template]}
            )

        try:
            start = 0
            stop = 100
            count = 100

            features_updated = features_to_update[start:stop]
            response = feature_layer.edit_features(updates=features_updated)

            for resp in response.get("updateResults", []):
                if resp.get("success", False):
                    continue
                warnings.warn(f"Something went wrong for data {resp}")

            time.sleep(2)
            while count == len(features_updated):
                start = stop
                stop = stop + 100
                features_updated = features_to_update[start:stop]
                response = feature_layer.edit_features(updates=features_updated)
                for resp in response.get("updateResults", []):
                    if resp.get("success", False):
                        continue
                    warnings.warn(f"Something went wrong for data {resp}")
                time.sleep(2)
        except Exception:
            feature_layer.manager.delete_from_definition({"fields": [field_template]})
            if confidence_field:
                feature_layer.manager.delete_from_definition(
                    {"fields": [confidence_field_template]}
                )

            return False

        return True

    def _classify_attachments(
        self,
        feature_layer,
        data_folder,
        feature_attachments_mapping,
        input_label_field,
        output_label_field,
        confidence_field=None,
        predict_function=_prediction_function,
    ):
        features = feature_layer.query().features
        features_to_update = []

        for feature in features:
            feature_attachments = feature_attachments_mapping.get(
                str(feature.attributes[input_label_field])
            ) or feature_attachments_mapping.get(
                int(feature.attributes[input_label_field])
            )
            if not feature_attachments:
                continue

            predictions = []
            for attachment in feature_attachments:
                prediction = self.predict(os.path.join(data_folder, attachment))
                predictions.append(
                    (prediction[0].obj, prediction[2].data.max().tolist())
                )

            final_prediction = predict_function(predictions)

            feature.attributes[output_label_field] = final_prediction[0]
            if confidence_field:
                feature.attributes[confidence_field] = final_prediction[1]

            features_to_update.append(feature)

        return features_to_update

    def _classify_labeled_tiles(
        self,
        feature_layer,
        labeled_tiles_directory,
        input_label_field,
        output_label_field,
        confidence_field=None,
    ):
        ALLOWED_FILE_FORMATS = ["tif", "jpg", "png"]
        IMAGES_FOLDER = "images/"
        LABELS_FOLDER = "labels/"

        files = []

        for ext in ALLOWED_FILE_FORMATS:
            files.extend(
                glob.glob(
                    os.path.join(labeled_tiles_directory, IMAGES_FOLDER + "*." + ext)
                )
            )

        predictions = {}
        for file in files:
            xml_path = os.path.join(
                os.path.dirname(os.path.dirname(file)),
                os.path.join(
                    LABELS_FOLDER, os.path.basename(file).split(".")[0] + ".xml"
                ),
            )

            if not os.path.exists(xml_path):
                continue

            tree = ElementTree.parse(xml_path)
            root = tree.getroot()

            name_field = root.findall("object/name")
            if len(name_field) != 1:
                continue

            file_prediction = self.predict(file)

            predictions[name_field[0].text] = {
                "prediction": file_prediction[0].obj,
                "score": str(file_prediction[2].data.max().tolist()),
            }

        features = feature_layer.query(output_fields=[input_label_field]).features
        features_to_update = []
        for feature in features:
            if predictions.get(str(feature.attributes[input_label_field])):
                feature.attributes[output_label_field] = predictions.get(
                    str(feature.attributes[input_label_field])
                )["prediction"]
                if confidence_field:
                    feature.attributes[confidence_field] = predictions.get(
                        str(feature.attributes[input_label_field])
                    )["score"]

                features_to_update.append(feature)

        return features_to_update

    def classify_features(
        self,
        feature_layer,
        labeled_tiles_directory,
        input_label_field,
        output_label_field,
        confidence_field=None,
        predict_function=None,
    ):
        """
        Deprecated in ArcGIS version 1.9.1 and later: Use the Classify Objects Using Deep Learning tool or :meth:`~arcgis.learn.classify_objects`

        Classifies the exported images and updates the feature layer with the prediction results in the ``output_label_field``.
        Works with RGB images only.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        feature_layer                            Required. :class:`~arcgis.features.FeatureLayer` for classification.
        ------------------------------------     --------------------------------------------------------------------
        labeled_tiles_directory                  Required. Folder structure containing images and labels folder. The
                                                 chips should have been generated using the export training data tool in
                                                 the Labeled Tiles format, and the labels should contain the OBJECTIDs
                                                 of the features to be classified.
        ------------------------------------     --------------------------------------------------------------------
        input_label_field                        Required. Value field name which created the labeled tiles. This field
                                                 should contain the OBJECTIDs of the features to be classified. In case of
                                                 attachments this field is not used.
        ------------------------------------     --------------------------------------------------------------------
        output_label_field                       Required. Output column name to be added in the layer which contains predictions.
        ------------------------------------     --------------------------------------------------------------------
        confidence_field                         Optional. Output column name to be added in the layer which contains the confidence score.
        ------------------------------------     --------------------------------------------------------------------
        predict_function                         Optional. Used for calculation of final prediction result when each feature
                                                 has more than one attachment. The ``predict_function`` takes as input a list of tuples.
                                                 Each tuple has first element as the class predicted and second element is the confidence score.
                                                 The function should return the final tuple classifying the feature and its confidence
        ====================================     ====================================================================

        :return:
            Boolean : True/False if operation is successful

        """

        if predict_function is None:
            predict_function = _prediction_function

        if input_label_field and _check_esri_files(Path(labeled_tiles_directory)):
            features_to_update = self._classify_labeled_tiles(
                feature_layer,
                labeled_tiles_directory,
                input_label_field,
                output_label_field,
                confidence_field,
            )
        elif os.path.exists(os.path.join(labeled_tiles_directory, "mapping.txt")):
            json_file = os.path.join(labeled_tiles_directory, "mapping.txt")
            with open(json_file) as file:
                feature_attachments_mapping = json.load(file)

            features_to_update = self._classify_attachments(
                feature_layer,
                labeled_tiles_directory,
                feature_attachments_mapping,
                feature_layer.properties["objectIdField"],
                output_label_field,
                confidence_field,
                predict_function,
            )
        else:
            return False

        return FeatureClassifier._update_predictions_layer(
            feature_layer, features_to_update, output_label_field, confidence_field
        )

    def _categorize_feature_layer(
        self,
        feature_layer,
        raster,
        class_value_field,
        class_name_field,
        confidence_field,
        cell_size,
        coordinate_system,
        predict_function,
        batch_size,
        overwrite,
    ):
        # class values
        class_values = list(self._data.class_mapping.keys())

        # normalization stats
        norm_mean = torch.tensor(imagenet_stats[0])
        norm_std = torch.tensor(imagenet_stats[1])

        # Check and create Fields
        class_name_field_template = {
            "name": class_name_field.lower(),
            "type": "esriFieldTypeString",
            "alias": class_name_field,
            "sqlType": "sqlTypeOther",
            "length": 256,
            "nullable": True,
            "editable": True,
            "visible": True,
            "domain": None,
            "defaultValue": "",
        }

        class_value_field_template = {
            "name": class_value_field.lower(),
            "type": "esriFieldTypeInteger",
            "alias": class_value_field,
            "sqlType": "sqlTypeOther",
            "nullable": True,
            "editable": True,
            "visible": True,
            "domain": None,
            "defaultValue": -999,
        }

        to_delete = []
        to_create = []

        feature_layer_fields = {
            f["name"].lower(): f for f in feature_layer.properties["fields"]
        }
        oid_field = feature_layer.properties["objectIdField"]

        if class_value_field_template["name"] in feature_layer_fields:
            if overwrite:
                to_delete.append(
                    feature_layer_fields[class_value_field_template["name"]]
                )
            else:
                e = Exception(
                    f"The specified class_value_field '{class_value_field}' already exists, please specify a different name or set `overwrite=True`"
                )
                raise (e)
        to_create.append(class_value_field_template)

        if class_name_field_template["name"] in feature_layer_fields:
            if overwrite:
                to_delete.append(
                    feature_layer_fields[class_name_field_template["name"]]
                )
            else:
                e = Exception(
                    f"The specified class_name_field '{class_name_field}' already exists, please specify a different name or set `overwrite=True`"
                )
                raise (e)
        to_create.append(class_name_field_template)

        if confidence_field is not None:
            confidence_field_template = {
                "name": confidence_field.lower(),
                "type": "esriFieldTypeDouble",
                "alias": confidence_field,
                "sqlType": "sqlTypeDouble",
                "nullable": True,
                "editable": True,
                "visible": True,
                "domain": None,
                "defaultValue": -999,
            }
            if confidence_field_template["name"] in feature_layer_fields:
                if overwrite:
                    to_delete.append(
                        feature_layer_fields[confidence_field_template["name"]]
                    )
                else:
                    e = Exception(
                        f"The specified confidence_field '{confidence_field}' already exists, please specify a different name or set `overwrite=True`"
                    )
                    raise (e)
            to_create.append(confidence_field_template)

        feature_layer.manager.delete_from_definition({"fields": to_delete})
        feature_layer.manager.add_to_definition({"fields": to_create})

        # Get features for updation
        fields_to_update = [oid_field, class_value_field, class_name_field]
        if confidence_field is not None:
            fields_to_update.append(confidence_field)
        feature_layer_features = feature_layer.query(
            out_fields=",".join(fields_to_update), return_geometry=False
        ).features
        update_store = {}

        if raster is not None:
            if not HAS_ARCPY:
                raise Exception("This function requires arcpy.")

            # Arcpy Environment to export data
            arcpy.env.cellSize = cell_size
            if coordinate_system is not None:
                arcpy.env.outputCoordinateSystem = coordinate_system
                arcpy.env.cartographicCoordinateSystem = coordinate_system

            feature_layer_url = feature_layer.url

            if feature_layer._token is not None:
                feature_layer_url = feature_layer_url + f"?token={feature_layer._token}"

            # Create Temporary ID field
            tempid_field = _tempid_field = "f_fcuid"
            i = 1
            while tempid_field in feature_layer_fields:
                tempid_field = _tempid_field + str(i)
                i += 1
            # arcpy.AddField_management(feature_layer_url, tempid_field, "LONG")
            tempid_field_template = {
                "name": tempid_field,
                "type": "esriFieldTypeInteger",
                "alias": tempid_field,
                "sqlType": "sqlTypeOther",
                "nullable": True,
                "editable": True,
                "visible": True,
                "domain": None,
                "defaultValue": -999,
            }
            feature_layer.manager.add_to_definition({"fields": [tempid_field_template]})
            # arcpy.CalculateField_management(feature_layer_url, tempid_field, f"{oid_field}", "SQL")
            feature_layer.calculate(
                where="1=1",
                calc_expression={
                    "field": tempid_field,
                    "sqlExpression": f"{oid_field}",
                },
            )

            temp_folder = arcpy.env.scratchFolder
            temp_datafldr = os.path.join(
                temp_folder, "categorize_features_" + str(int(time.time()))
            )
            result = arcpy.ia.ExportTrainingDataForDeepLearning(
                in_raster=raster,
                out_folder=temp_datafldr,
                in_class_data=feature_layer_url,
                image_chip_format="TIFF",
                tile_size_x=self._data.chip_size,
                tile_size_y=self._data.chip_size,
                stride_x=0,
                stride_y=0,
                output_nofeature_tiles="ALL_TILES",
                metadata_format="Labeled_Tiles",
                start_index=0,
                class_value_field=tempid_field,
                buffer_radius=0,
                in_mask_polygons=None,
                rotation_angle=0,
            )
            # cleanup
            # arcpy.DeleteField_management(feature_layer_url, [ tempid_field ])
            feature_layer.manager.delete_from_definition(
                {"fields": [tempid_field_template]}
            )

            image_list = ImageList.from_folder(os.path.join(temp_datafldr, "images"))

            def get_id(imagepath):
                with open(
                    os.path.join(
                        temp_datafldr,
                        "labels",
                        os.path.basename(imagepath)[:-3] + "xml",
                    )
                ) as f:
                    return int(f.read().split("<name>")[1].split("<")[0])

            for i in range(0, len(image_list), batch_size):
                # Get Temporary Ids
                tempids = [get_id(f) for f in image_list.items[i : i + batch_size]]

                # Get Image batch
                image_batch = torch.stack(
                    [im.data for im in image_list[i : i + batch_size]]
                )
                image_batch = normalize(image_batch, mean=norm_mean, std=norm_std)

                # Get Predications
                predicted_classes, predictions_conf = self._predict_batch(image_batch)

                # push prediction to store
                for ui, oid in enumerate(tempids):
                    classvalue = class_values[predicted_classes[ui]]
                    update_store[oid] = {
                        oid_field: oid,
                        class_value_field: classvalue,
                        class_name_field: self._data.class_mapping[classvalue],
                    }
                    if confidence_field is not None:
                        update_store[oid][confidence_field] = predictions_conf[ui]

            # Cleanup
            arcpy.Delete_management(temp_datafldr)
            shutil.rmtree(temp_datafldr, ignore_errors=True)

        else:
            out_folder = tempfile.TemporaryDirectory().name
            os.mkdir(out_folder)
            feature_layer.export_attachments(out_folder)
            with open(os.path.join(out_folder, "mapping.txt")) as file:
                feature_attachments_mapping = json.load(file)
                images_store = []
                for oid in feature_attachments_mapping:
                    for im in feature_attachments_mapping[oid]:
                        images_store.append(
                            {"oid": oid, "im": os.path.join(out_folder, im)}
                        )
            update_store_scratch = {}
            for i in range(0, len(images_store), batch_size):
                rel_objectids = []
                image_batch = []
                for r in images_store[i : i + batch_size]:
                    im = open_image(r["im"])  # Read Bytes
                    im = im.resize(self._data.chip_size)  # Resize
                    image_batch.append(im.data)  # Convert to tensor
                    rel_objectids.append(int(r["oid"]))
                image_batch = torch.stack(image_batch)
                image_batch = normalize(image_batch, mean=norm_mean, std=norm_std)
                # Get Predictions and save to scratch
                predicted_classes, predictions_conf = self._predict_batch(image_batch)
                for ai, oid in enumerate(rel_objectids):
                    if update_store_scratch.get(oid) is None:
                        update_store_scratch[oid] = []
                    update_store_scratch[oid].append(
                        [predicted_classes[ai], predictions_conf[ai]]
                    )
            # Prepare final updated features
            for oid in update_store_scratch:
                max_prediction_class, max_prediction_value = predict_function(
                    update_store_scratch[oid]
                )
                if max_prediction_class is not None:
                    classvalue = class_values[max_prediction_class]
                    classname = self._data.class_mapping[classvalue]
                else:
                    classvalue = None
                    classname = None
                update_store[oid] = {
                    oid_field: oid,
                    class_value_field: classvalue,
                    class_name_field: classname,
                }
                if confidence_field is not None:
                    update_store[oid][confidence_field] = max_prediction_value

        # Update Features
        features_to_update = []
        for feat in feature_layer_features:
            if update_store.get(feat.attributes[oid_field]) is not None:
                updated_attributes = update_store[feat.attributes[oid_field]]
                for f in fields_to_update:
                    feat.attributes[f] = updated_attributes[f]
                features_to_update.append(feat)
        step = 100
        for si in range(0, len(features_to_update), step):
            feature_batch = features_to_update[si : si + step]
            response = feature_layer.edit_features(updates=feature_batch)
            for resp in response.get("updateResults", []):
                if resp.get("success", False):
                    continue
                warnings.warn(f"Something went wrong for data {resp}")
            time.sleep(2)
        return True

    def _categorize_feature_class(
        self,
        feature_class,
        raster,
        class_value_field,
        class_name_field,
        confidence_field,
        cell_size,
        coordinate_system,
        predict_function,
        batch_size,
        overwrite,
    ):
        # class values
        class_values = list(self._data.class_mapping.keys())

        if not HAS_ARCPY:
            raise Exception("This function requires arcpy to access feature class.")
        arcpy.env.overwriteOutput = overwrite

        if batch_size is None:
            batch_size = self._data.batch_size

        if predict_function is None:
            predict_function = _prediction_function

        norm_mean = torch.tensor(imagenet_stats[0])
        norm_std = torch.tensor(imagenet_stats[1])

        fcdesc = arcpy.Describe(feature_class)
        oid_field = fcdesc.OIDFieldName
        if not (fcdesc.dataType == "FeatureClass" and fcdesc.shapeType == "Polygon"):
            e = Exception(
                f"The specified FeatureClass at '{feature_class}' is not valid, it should be Polygon FeatureClass"
            )
            raise (e)
        fields = arcpy.ListFields(feature_class)
        field_names = [f.name for f in fields]
        if class_value_field in field_names:
            if not overwrite:
                e = Exception(
                    f"The specified class_value_field '{class_value_field}' already exists in the target FeatureClass, please specify a different name or set `overwrite=True`"
                )
                raise (e)
        arcpy.DeleteField_management(feature_class, [class_value_field])
        arcpy.AddField_management(feature_class, class_value_field, "LONG")

        if class_name_field in field_names:
            if not overwrite:
                e = Exception(
                    f"The specified class_name_field '{class_name_field}' already exists in the target FeatureClass, please specify a different name or set `overwrite=True`"
                )
                raise (e)
        arcpy.DeleteField_management(feature_class, [class_name_field])
        arcpy.AddField_management(feature_class, class_name_field, "TEXT")

        if confidence_field is not None:
            if confidence_field in field_names:
                if not overwrite:
                    e = Exception(
                        f"The specified confidence_field '{confidence_field}' already exists in the target FeatureClass, please specify a different name or set `overwrite=True`"
                    )
                    raise (e)
            arcpy.DeleteField_management(feature_class, [confidence_field])
            arcpy.AddField_management(feature_class, confidence_field, "DOUBLE")

        if raster is not None:
            # Arcpy Environment to export data
            arcpy.env.cellSize = cell_size
            if coordinate_system is not None:
                arcpy.env.outputCoordinateSystem = coordinate_system
                arcpy.env.cartographicCoordinateSystem = coordinate_system

            tempid_field = _tempid_field = "f_fcuid"
            i = 1
            while tempid_field in field_names:
                tempid_field = _tempid_field + str(i)
                i += 1
            arcpy.AddField_management(feature_class, tempid_field, "LONG")
            arcpy.CalculateField_management(
                feature_class, tempid_field, f"!{oid_field}!"
            )

            temp_folder = arcpy.env.scratchFolder
            temp_datafldr = os.path.join(
                temp_folder, "categorize_features_" + str(int(time.time()))
            )
            result = arcpy.ia.ExportTrainingDataForDeepLearning(
                in_raster=raster,
                out_folder=temp_datafldr,
                in_class_data=feature_class,
                image_chip_format="TIFF",
                tile_size_x=self._data.chip_size,
                tile_size_y=self._data.chip_size,
                stride_x=0,
                stride_y=0,
                output_nofeature_tiles="ALL_TILES",
                metadata_format="Labeled_Tiles",
                start_index=0,
                class_value_field=tempid_field,
                buffer_radius=0,
                in_mask_polygons=None,
                rotation_angle=0,
            )
            # cleanup
            arcpy.DeleteField_management(feature_class, [tempid_field])
            image_list = ImageList.from_folder(os.path.join(temp_datafldr, "images"))

            def get_id(imagepath):
                with open(
                    os.path.join(
                        temp_datafldr,
                        "labels",
                        os.path.basename(imagepath)[:-3] + "xml",
                    )
                ) as f:
                    return int(f.read().split("<name>")[1].split("<")[0])

            for i in range(0, len(image_list), batch_size):
                # Get Temporary Ids
                tempids = [get_id(f) for f in image_list.items[i : i + batch_size]]

                # Get Image batch
                image_batch = torch.stack(
                    [im.data for im in image_list[i : i + batch_size]]
                )
                image_batch = normalize(image_batch, mean=norm_mean, std=norm_std)

                # Get Predications
                predicted_classes, predictions_conf = self._predict_batch(image_batch)

                # Update Feature Class
                where_clause = f"{oid_field} IN ({','.join(str(e) for e in tempids)})"
                update_cursor = arcpy.UpdateCursor(
                    feature_class,
                    where_clause=where_clause,
                    sort_fields=f"{oid_field} A",
                )
                for row in update_cursor:
                    row_tempid = row.getValue(oid_field)
                    ui = tempids.index(row_tempid)
                    classvalue = class_values[predicted_classes[ui]]
                    row.setValue(class_value_field, classvalue)
                    row.setValue(class_name_field, self._data.class_mapping[classvalue])
                    if confidence_field is not None:
                        row.setValue(confidence_field, predictions_conf[ui])
                    update_cursor.updateRow(row)

                # Remove Locks
                del row
                del update_cursor

            # Cleanup
            arcpy.Delete_management(temp_datafldr)
            shutil.rmtree(temp_datafldr, ignore_errors=True)

        else:
            feature_class_attach = feature_class + "__ATTACH"
            nrows = arcpy.GetCount_management(feature_class_attach)[0]
            store = {}
            for i in range(0, int(nrows), batch_size):
                attachment_ids = []
                rel_objectids = []
                image_batch = []

                # Get Image Batch
                with arcpy.da.SearchCursor(
                    feature_class_attach, ["ATTACHMENTID", "REL_OBJECTID", "DATA"]
                ) as search_cursor:
                    for c, item in enumerate(search_cursor):
                        if c >= i and c < i + batch_size:
                            attachment_ids.append(item[0])
                            rel_objectids.append(item[1])
                            attachment = item[-1]
                            im = open_image(
                                io.BytesIO(attachment.tobytes())
                            )  # Read Bytes
                            im = im.resize(self._data.chip_size)  # Resize
                            image_batch.append(im.data)  # Convert to tensor
                            del item
                            del attachment
                            # del im
                image_batch = torch.stack(image_batch)
                image_batch = normalize(image_batch, mean=norm_mean, std=norm_std)

                # Get Predictions and save to store
                predicted_classes, predictions_conf = self._predict_batch(image_batch)
                for ai in range(len(attachment_ids)):
                    if store.get(rel_objectids[ai]) is None:
                        store[rel_objectids[ai]] = []
                    store[rel_objectids[ai]].append(
                        [predicted_classes[ai], predictions_conf[ai]]
                    )

            # Update Feature Class
            update_cursor = arcpy.UpdateCursor(feature_class)
            for row in update_cursor:
                row_oid = row.getValue(oid_field)
                max_prediction_class, max_prediction_value = predict_function(
                    store[row_oid]
                )
                if max_prediction_class is not None:
                    classvalue = class_values[max_prediction_class]
                    classname = self._data.class_mapping[classvalue]
                else:
                    classvalue = None
                    classname = None
                row.setValue(class_value_field, classvalue)
                row.setValue(class_name_field, classname)
                if confidence_field is not None:
                    row.setValue(confidence_field, max_prediction_value)
                update_cursor.updateRow(row)

            # Remove Locks
            del row
            del update_cursor
        return True

    def _gradCAM(
        self, im, cl, heatmap_thresh: int = 16, image: bool = True, grad_vis=False
    ):
        if isinstance(cl, fastai.core.MultiCategory):
            if not cl.raw:  # If the predictions are all 0, including for None class
                xb_norm, _ = self._data.one_item(im, detach=False, denorm=True)
                xb, _ = self._data.one_item(im, detach=False, denorm=False)
                xb_im = Image(xb[0])
                xb_im_denorm = Image(xb_norm[0])
                _, ax = plt.subplots(figsize=(6, 6))
                xb_im_denorm.show(ax, title=f"Predicted class: None")
                return
            else:
                cat = cl.raw  # Handles MuliCategory types
                cat1 = cat[0]
        else:
            cat1 = int(cl)
        m = self.learn.model.eval()
        xb_norm, _ = self._data.one_item(im, detach=False, denorm=True)
        xb, _ = self._data.one_item(
            im, detach=False, denorm=False
        )  # put into a minibatch of batch size = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with hook_output(m[0]) as hook_a:
                with hook_output(m[0], grad=True) as hook_g:
                    preds = m(xb)
                    preds[0, cat1].backward()
        acts = hook_a.stored[0].cpu()  # activation maps
        grad = hook_g.stored[0][0].cpu()
        if self._transformer:
            acts = reshape_tensor(acts)
            grad = reshape_tensor(grad)

        if (acts.shape[-1] * acts.shape[-2]) >= heatmap_thresh:
            grad_chan = grad.mean(1).mean(1)
            mult = F.relu(((acts * grad_chan[..., None, None])).sum(0))
            if image:
                xb_im = Image(xb[0])
                xb_im_denorm = Image(xb_norm[0])
                sz = list(xb_im.shape[-2:])
                if grad_vis == True:
                    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
                    xb_im_denorm.show(ax[0], title=f"Predicted class: {cl}")
                    xb_im_denorm.show(ax[1], title=f"Predicted class: {cl}")
                    ax[1].imshow(
                        mult,
                        alpha=0.4,
                        extent=(0, *sz[::-1], 0),
                        interpolation="bilinear",
                        cmap="hot",
                    )
                else:
                    _, ax = plt.subplots(figsize=(6, 6))
                    xb_im_denorm.show(ax, title=f"Predicted class: {cl}")
            return mult

    @deprecated(
        deprecated_in="1.7.1",
        details="Please use :meth:`~arcgis.learn.classify_objects` instead",
    )
    def categorize_features(
        self,
        feature_layer,
        raster=None,
        class_value_field="class_val",
        class_name_field="prediction",
        confidence_field="confidence",
        cell_size=1,
        coordinate_system=None,
        predict_function=None,
        batch_size=64,
        overwrite=False,
    ):
        """
        Categorizes each feature by classifying its attachments or an image of its geographical area (using the provided Imagery Layer)
        and updates the feature layer with the prediction results in the ``output_label_field``.
        Deprecated, Use the Classify Objects Using Deep Learning tool or :meth:`~arcgis.learn.classify_objects`

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        feature_layer                            Required. Public :class:`~arcgis.features.FeatureLayer` or path of local feature class for classification with read, write, edit permissions.
        ------------------------------------     --------------------------------------------------------------------
        raster                                   Optional. :class:`~arcgis.raster.ImageryLayer` or path of local raster to be used for exporting image chips. (Requires arcpy)
        ------------------------------------     --------------------------------------------------------------------
        class_value_field                        Required string. Output field to be added in the layer, containing class value of predictions.
        ------------------------------------     --------------------------------------------------------------------
        class_name_field                         Required string. Output field to be added in the layer, containing class name of predictions.
        ------------------------------------     --------------------------------------------------------------------
        confidence_field                         Optional string. Output column name to be added in the layer which contains the confidence score.
        ------------------------------------     --------------------------------------------------------------------
        cell_size                                Optional float. Cell size to be used for exporting the image chips.
        ------------------------------------     --------------------------------------------------------------------
        coordinate_system                        Optional. Cartographic Coordinate System to be used for exporting the image chips.
        ------------------------------------     --------------------------------------------------------------------
        predict_function                         Optional list of tuples. Used for calculation of final prediction result when each feature
                                                 has more than one attachment. The ``predict_function`` takes as input a list of tuples.
                                                 Each tuple has first element as the class predicted and second element is the confidence score.
                                                 The function should return the final tuple classifying the feature and its confidence.
        ------------------------------------     --------------------------------------------------------------------
        batch_size                               Optional integer. The no of images or tiles to process in a single go.

                                                 The default value is 64.
        ------------------------------------     --------------------------------------------------------------------
        overwrite                                Optional boolean. If set to True the output fields will be overwritten by new values.

                                                 The default value is False.
        ====================================     ====================================================================

        :return:
            Boolean : True if operation is successful, False otherwise

        """

        import arcgis
        from arcgis.features import FeatureLayer
        from arcgis.raster import ImageryLayer
        from arcgis.gis import Item

        class_value_field = class_value_field.lower()
        class_name_field = class_name_field.lower()
        confidence_field = confidence_field.lower()

        if predict_function is None:
            predict_function = _prediction_function

        if isinstance(raster, str):
            if "http" in raster:
                raster = ImageryLayer(raster, gis=arcgis.env.active_gis)
        if isinstance(raster, ImageryLayer):
            raster_url = raster.url
            if raster._token is not None:
                raster_url = raster_url + f"?token={raster._token}"
            raster = raster_url
        elif isinstance(raster, Item):  # handles Mapserver
            raster_url = raster.url
            if raster.layers[0]._token is not None:
                raster_url = raster_url + f"?token={raster.layers[0]._token}"
            raster = raster_url

        if isinstance(feature_layer, str):
            if "http" in feature_layer:
                feature_layer = FeatureLayer(feature_layer, gis=arcgis.env.active_gis)
            else:
                return self._categorize_feature_class(
                    feature_class=feature_layer,
                    raster=raster,
                    class_value_field=class_value_field,
                    class_name_field=class_name_field,
                    confidence_field=confidence_field,
                    cell_size=cell_size,
                    coordinate_system=coordinate_system,
                    predict_function=predict_function,
                    batch_size=batch_size,
                    overwrite=overwrite,
                )

        if isinstance(feature_layer, FeatureLayer):
            return self._categorize_feature_layer(
                feature_layer=feature_layer,
                raster=raster,
                class_value_field=class_value_field,
                class_name_field=class_name_field,
                confidence_field=confidence_field,
                cell_size=cell_size,
                coordinate_system=coordinate_system,
                predict_function=predict_function,
                batch_size=batch_size,
                overwrite=overwrite,
            )
        else:
            e = Exception("Could not understand layer type")
            raise (e)

    ## Tensorflow specific functions start ##
    def _intialize_tensorflow(self, data, backbone, drop, pretrained_path, kwargs):
        self._check_tf()

        from .._utils.fastai_tf_fit import TfLearner
        import tensorflow as tf
        from tensorflow.keras.losses import CategoricalCrossentropy
        from tensorflow.keras.models import Model
        from tensorflow.keras import applications
        from tensorflow.keras.optimizers import Adam
        from fastai.basics import defaults
        from .._utils.image_classification import TF_IC_get_head_output

        if data._is_multispectral:
            raise Exception(
                'Multispectral data is not supported with backend="tensorflow"'
            )

        # Pyramid Scheme in head
        self._fpn = kwargs.get("fpn", True)

        # prepare color array
        alpha = 0.7
        color_mapping = getattr(data, "color_mapping", None)
        if color_mapping is None:
            color_array = torch.tensor([[1.0, 1.0, 1.0]]).float()
        else:
            color_array = torch.tensor(list(color_mapping.values())).float() / 255
        alpha_tensor = torch.tensor([alpha] * len(color_array)).view(-1, 1).float()
        color_array = torch.cat([color_array, alpha_tensor], dim=-1)
        background_color = torch.tensor([[0, 0, 0, 0]]).float()
        data._multispectral_color_array = torch.cat([background_color, color_array])

        self.ssd_version = 1  # ssd_version
        if backbone is None:
            backbone = "ResNet50"

        if type(backbone) == str:
            backbone = getattr(applications, backbone)

        self._backbone = backbone

        x, y = next(iter(data.train_dl))
        if tf.keras.backend.image_data_format() == "channels_last":
            in_shape = [x.shape[-1], x.shape[-1], 3]
        else:
            in_shape = [3, x.shape[-1], x.shape[-1]]

        self._backbone_initalized = self._backbone(
            input_shape=in_shape, include_top=False, weights="imagenet"
        )
        self._backbone_initalized.trainable = False

        self._device = torch.device("cpu")
        self._data = data

        self._loss_function_tf_ = CategoricalCrossentropy(
            from_logits=True, reduction="sum"
        )
        self._loss_function_tf_noreduction = CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        output_layer = TF_IC_get_head_output(self)

        model = Model(inputs=self._backbone_initalized.input, outputs=output_layer)

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

        self._code = feature_classifier_prf

    def _loss_function_tf(self, target, predictions, reduction=True):
        import tensorflow as tf

        if target.ndim == 2:
            target_masks = target
        else:
            target_masks = tf.gather(tf.eye(self._data.c), target)
        if reduction:
            return self._loss_function_tf_(target_masks, predictions)
        else:
            return self._loss_function_tf_noreduction(target_masks, predictions)


if HAS_FASTAI:

    class OverSamplingCallback(LearnerCallback):
        """
        The OverSamplingCallback support handles unbalanced dataset (dataset with rare classes). It is used to oversample data during training.
        """

        def __init__(self, learn: Learner, weights: torch.Tensor = None):
            super().__init__(learn)
            self.weights = weights

        def on_train_begin(self, **kwargs):
            ds, dl = self.data.train_ds, self.data.train_dl
            self.labels = ds.y.items.astype(int)
            assert np.issubdtype(
                self.labels.dtype, np.integer
            ), "Can only oversample integer values"
            _, self.label_counts = np.unique(self.labels, return_counts=True)
            if self.weights is None:
                self.weights = torch.DoubleTensor((1 / self.label_counts)[self.labels])
            self.total_len_oversample = int(self.data.c * np.max(self.label_counts))
            sampler = WeightedRandomSampler(self.weights, self.total_len_oversample)
            self.data.train_dl = dl.new(shuffle=False, sampler=sampler)
