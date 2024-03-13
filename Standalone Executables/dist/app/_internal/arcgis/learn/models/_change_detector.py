from pathlib import Path
import json
import random
from ._arcgis_model import ArcGISModel, _EmptyData
import traceback
from .._utils.env import raise_fastai_import_error
from ._codetemplate import image_classifier_prf

try:
    from ._change_detector_utils import get_learner
    from .._utils.change_detection_data import show_results
    from ._arcgis_model import _resnet_family
    from .._utils.common import _get_emd_path

    HAS_FASTAI = True
except ImportError:
    import_exception = traceback.format_exc()
    HAS_FASTAI = False


class ChangeDetector(ArcGISModel):

    """
    Creates a Change Detection model.

    A Spatial-Temporal Attention-Based Method and a New Dataset
    for Remote Sensing Image Change Detection -
    https://www.mdpi.com/2072-4292/12/10/1662

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object
                            from :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used
                            for creating the encoder of the :class:`~arcgis.learn.ConnectNet`,
                            which is `resnet18` by default. It supports
                            the ResNet family of backbones.
    ---------------------   -------------------------------------------
    attention_type          Optional string. It's value can be either be "PAM"
                            (Pyramid Attention Module) or "BAM"
                            (Basic Attention Module).
                            Defaults to "PAM".
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.ConnectNet` object
    """

    def __init__(
        self, data, backbone=None, attention_type="PAM", pretrained_path=None, **kwargs
    ):
        if not HAS_FASTAI:
            raise_fastai_import_error(
                import_exception=import_exception, message="", installation_steps=" "
            )

        if backbone is None:
            backbone = "resnet18"

        if not self._check_backbone_support(backbone):
            raise Exception(
                f"Enter only compatible backbones from {', '.join(self.supported_backbones)}"
            )

        super().__init__(data, backbone, pretrained_path=pretrained_path)
        backbone = self._backbone.__name__.lower()
        self.SA_type = attention_type
        self.learn = get_learner(self._data, backbone, self.SA_type)
        self._code = image_classifier_prf
        self._arcgis_init_callback()  # make first conv weights learnable
        if pretrained_path is not None:
            self.load(pretrained_path)

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "precision", "recall", "f1"]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @property
    def supported_backbones(self):
        """Supported torchvision backbones for this model."""
        return ChangeDetector._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return [*_resnet_family]

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return ChangeDetector._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["ChangeDetection", "Classified_Tiles"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a ChangeDetector model from an Esri Model Definition (EMD)
        file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Optional fastai Databunch. Returned
                                data object from :meth:`~arcgis.learn.prepare_data`  function or
                                None for inferencing.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.ConnectNet` Object
        """
        emd_path = _get_emd_path(emd_path)
        emd_path = Path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd["ModelParameters"]
        if data is None:
            data = _EmptyData(
                path=emd_path.parent,
                loss_func=None,
                c=2,  # change, no_change
                chip_size=emd["ImageHeight"],
            )
            data._is_empty = True
            data._imagery_type = None
            data.emd_path = emd_path
            data.emd = emd
            for key, value in emd["DataAttributes"].items():
                setattr(data, key, value)

        return cls(data, **model_params, pretrained_path=str(model_file))

    @property
    def _model_metrics(self):
        metrics = self._get_model_metrics()
        return {
            "precision": float(metrics[0]),
            "recall": float(metrics[1]),
            "f1": float(metrics[2]),
        }

    def _get_model_metrics(self, **kwargs):
        checkpoint = getattr(self, "_is_checkpointed", False)
        if not hasattr(self.learn, "recorder"):
            return [0.0, 0.0, 0.0]

        if len(self.learn.recorder.metrics) == 0:
            return [0.0, 0.0, 0.0]

        model_accuracy = self.learn.recorder.metrics[-1]
        if checkpoint:
            val_losses = self.learn.recorder.val_losses
            model_accuracy = self.learn.recorder.metrics[
                self.learn._best_epoch  # index using best epoch.
            ]

        return model_accuracy

    def _get_emd_params(self, save_inference_file):
        _emd_template = {"DataAttributes": {}, "ModelParameters": {}}
        # add encoder parameters
        _emd_template["ModelParameters"]["attention_type"] = self.SA_type
        # chip size
        _emd_template["DataAttributes"]["chip_size"] = self._data.chip_size
        _emd_template["DataAttributes"][
            "_is_multispectral"
        ] = self._data._is_multispectral
        if self._data._is_multispectral:
            _emd_template["DataAttributes"]["_imagery_type"] = self._data._imagery_type
            _emd_template["DataAttributes"]["_bands"] = self._data._bands
            _emd_template["DataAttributes"]["_rgb_bands"] = self._data._rgb_bands
            _emd_template["DataAttributes"][
                "_extract_bands"
            ] = self._data._extract_bands
            _emd_template["DataAttributes"]["_train_tail"] = self._data._train_tail
            _emd_template["DataAttributes"][
                "_band_min_values"
            ] = self._data._band_min_values.tolist()
            _emd_template["DataAttributes"][
                "_band_max_values"
            ] = self._data._band_max_values.tolist()

        # normalization stats
        norm_stats = []
        for k in self._data.norm_stats:
            norm_stats.append(k)
        _emd_template["DataAttributes"]["norm_stats"] = list(norm_stats)
        _emd_template["DataAttributes"]["class_mapping"] = self._data.class_mapping
        _emd_template["DataAttributes"]["color_mapping"] = self._data.color_mapping
        _emd_template["DataAttributes"]["classes"] = self._data.classes
        _emd_template["DataAttributes"]["batch_size"] = self._data.batch_size
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        # object classifier config can be used.
        _emd_template["ModelConfiguration"] = "change_detection"
        # Model Type
        _emd_template["ModelType"] = "ImageClassification"
        # Inference function of object classifier.
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISImageClassifier.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISImageClassifier.py"
        if self._is_multispectral:
            # change this when we start to honour extract bands parameter.
            _emd_template["ExtractBands"] = list(
                range(len(self._data._extract_bands) * 2)
            )
        else:
            # To extract bands from concatenated images (RGB)
            _emd_template["ExtractBands"] = list(range(6))
        _emd_template["Classes"] = []
        class_data = {}
        for (
            value,
            class_name,
        ) in self._data.class_mapping.items():  # 0th index is background
            class_data["Value"] = value
            class_data["Name"] = class_name
            class_data["Color"] = self._data.color_mapping[value]
            _emd_template["Classes"].append(class_data.copy())

        return _emd_template

    def show_results(self, rows=4, **kwargs):
        """
        Displays the results of a trained model on the validation set.
        """
        show_results(self, rows, **kwargs)

    def precision_recall_score(self):
        """
        Computes precision, recall and f1 score.
        """
        from .._utils.classified_tiles import per_class_metrics

        return per_class_metrics(self, postprocess_type="CD")

    def _score_author(self):
        """
        Internal implementation required for comparing
        the author's way of computing scores.
        """
        from ._change_detector_utils import calculate_author_metrics

        return calculate_author_metrics(self)

    def predict(self, before_image, after_image, **kwargs):
        """
        Predict on a pair of images.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        before_image            Required string. Path to image from before.
        ---------------------   -------------------------------------------
        after_image             Required string. Path to image from later.
        =====================   ===========================================

        **Kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        crop_predict            Optional Boolean. If True, It will predict
                                using a sliding window strategy. Typically, used
                                when image size is larger than the `chip_size`
                                the model is trained on. Default False.
        ---------------------   -------------------------------------------
        visualize               Optional Boolean. If True, It will plot
                                the predictions on the notebook. Default False.
        ---------------------   -------------------------------------------
        save                    Optional Boolean. If true will write the
                                prediction file on the disk. Default False.
        =====================   ===========================================

        :return: PyTorch Tensor of the change mask.
        """
        from .._utils.change_detection_data import predict

        return predict(self, before_image, after_image, **kwargs)
