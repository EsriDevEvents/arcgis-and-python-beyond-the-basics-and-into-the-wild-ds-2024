from pathlib import Path
import json
from ._arcgis_model import ArcGISModel, _EmptyData
import traceback
from .._utils.env import raise_fastai_import_error
import logging

logger = logging.getLogger()

try:
    from ._image_captioning_utils import (
        image_captioner_learner,
        predict_image,
        get_bleu,
    )
    from .._utils.image_captioning_data import show_results
    from ._arcgis_model import _resnet_family
    from .._utils.common import _get_emd_path
    from ._codetemplate import image_captioning_prf

    HAS_FASTAI = True
except ImportError:
    import_exception = traceback.format_exc()
    HAS_FASTAI = False


class ImageCaptioner(ArcGISModel):

    """
    Creates an Image Captioning model.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object
                            from :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used
                            for creating the encoder of the :class:`~arcgis.learn.ImageCaptioner` ,
                            which is `resnet34` by default. It supports
                            the ResNet family of backbones.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    decoder_params          Optional dictionary. The keys of the dictionary are
                            `embed_size`, `hidden_size`, `attention_size`,
                            `teacher_forcing`, `dropout` and
                            `pretrained_embeddings`.

                            Default values:

                                | decoder_params={
                                |                     'embed_size':100,
                                |                     'hidden_size':100,
                                |                     'attention_size':100,
                                |                     'teacher_forcing':1,
                                |                     'dropout':0.1,
                                |                     'pretrained_emb':False
                                |                 }

                            Parameter Explanation:

                            - 'embed_size': Size of embedding to be used during training.
                            - 'hidden_size': Size of hidden layer.
                            - 'attention_size': Size of intermediate attention layer.
                            - 'teacher_forcing': Probability of teacher forcing.
                            - 'dropout': Dropout probability.
                            - 'pretrained_emb': If true, it will use fasttext embeddings.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.ImageCaptioner`  Object
    """

    def __init__(self, data, backbone=None, pretrained_path=None, **kwargs):
        if not HAS_FASTAI:
            raise_fastai_import_error(
                import_exception=import_exception, message="", installation_steps=" "
            )

        super().__init__(data, backbone, **kwargs)

        if pretrained_path is not None:
            pretrained_backbone = False
        else:
            pretrained_backbone = True

        self._code = image_captioning_prf
        self.decoder_params = kwargs.get("decoder_params", {})
        self.learn = image_captioner_learner(
            self._data,
            self._backbone,
            decoder_params=self.decoder_params,
            metrics=kwargs.get("metrics", None),
            pretrained=pretrained_backbone,
        )
        if pretrained_path is not None:
            self.load(pretrained_path)  # Load model and vocab

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy", "corpus_bleu"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a ImageCaptioner model from an Esri Model Definition (EMD)
        file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Optional fastai Databunch. Returned
                                data object from :meth:`~arcgis.learn.prepare_data` function or
                                None for inferencing.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.ImageCaptioner`  Object
        """

        from fastai.text.transform import Vocab

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd["ModelParameters"]
        if data is None:
            data = _EmptyData(
                path=emd_path.parent, loss_func=None, c=0, chip_size=emd["ImageHeight"]
            )

            data.emd_path = emd_path
            data.emd = emd
            data._is_empty = True
            for key, value in emd["DataAttributes"].items():
                setattr(data, key, value)

            vocab_path = emd_path.parent / "vocab"
            data.vocab = Vocab.load(vocab_path)  # load vocab.

        return cls(data, **model_params, pretrained_path=str(model_file))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @property
    def supported_backbones(self):
        """Supported torchvision backbones for this model."""
        return ImageCaptioner._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return [*_resnet_family]

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return ImageCaptioner._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["ImageCaptioning"]

    @property
    def _model_metrics(self):
        return {"accuracy": self._get_model_metrics()}

    def _get_model_metrics(self, **kwargs):
        checkpoint = getattr(self, "_is_checkpointed", False)
        if not hasattr(self.learn, "recorder"):
            return 0.0

        if len(self.learn.recorder.metrics) == 0:
            return 0.0

        model_accuracy = self.learn.recorder.metrics[-1][0]
        if checkpoint:
            val_losses = self.learn.recorder.val_losses
            model_accuracy = self.learn.recorder.metrics[
                self.learn._best_epoch  # index using best epoch.
            ][0]

        return float(model_accuracy)

    def bleu_score(self, **kwargs):
        """
        Computes bleu score over validation set.

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        beam_width              Optional int. The size of beam to be used
                                during beam search decoding. Default is 5.
        ---------------------   -------------------------------------------
        max_len                 Optional int. The maximum length of the
                                sentence to be decoded. Default is 20.
        =====================   ===========================================

        """
        if isinstance(self._data, _EmptyData):
            scores = self._data.emd.get("Metrics")
            if scores is None:
                logger.error("Metric not found in the loaded model")
                return
            else:
                return json.loads(scores)
        return get_bleu(self, self._data, *kwargs)

    def _get_emd_params(self, save_inference_file):
        _emd_template = {"DataAttributes": {}, "ModelParameters": {}}
        # arcgis.learn.models._inferencing
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        # object classifier config can be used.
        _emd_template["ModelConfiguration"] = "_image_captioner_inference"
        # handle for different types of spectrums
        _emd_template["ExtractBands"] = [0, 1, 2]
        _emd_template["ModelType"] = "ImageCaptioner"
        # Inference function of object classifier.
        _emd_template["InferenceFunction"] = "ArcGISObjectClassifier.py"

        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISImageCaptioner.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISImageCaptioner.py"

        # add encoder parameters
        _emd_template["ModelParameters"]["decoder_params"] = self.decoder_params
        # chip size
        _emd_template["DataAttributes"]["chip_size"] = self._data.chip_size
        # normalization stats
        norm_stats = []
        for k in self._data.norm_stats:
            norm_stats.append(k.tolist())
        _emd_template["DataAttributes"]["norm_stats"] = list(norm_stats)

        _emd_template["CropSizeFixed"] = 1
        _emd_template["BlackenAroundFeature"] = 0
        _emd_template["SingleLabelFieldFound"] = "Caption"

        return _emd_template

    def predict(self, path, visualize=True, **kwargs):
        return predict_image(self, path, visualize=visualize, **kwargs)

    def show_results(self, rows=4, **kwargs):
        """
        Shows the ground truth and predictions of model side by side.

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        beam_width              Optional int. The size of beam to be used
                                during beam search decoding. Default is 5.
        ---------------------   -------------------------------------------
        max_len                 Optional int. The maximum length of the
                                sentence to be decoded. Default is 20.
        =====================   ===========================================

        """
        self._check_requisites()
        return_fig = kwargs.get("return_fig", False)
        fig = show_results(self, rows=rows, **kwargs)
        if return_fig:
            return fig

    def _save(
        self, name_or_path, framework="PyTorch", publish=False, gis=None, **kwargs
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
        framework               Optional string. Defines the framework of the
                                model.
                                (Only supported by :class:`~arcgis.learn.SingleShotDetector`.)
                                If framework used is ``TF-ONNX``,
                                ``batch_size`` can be passed as an optional
                                keyword argument.

                                Framework choice: 'PyTorch' and 'TF-ONNX'
        ---------------------   -------------------------------------------
        publish                 Optional boolean. Publishes the DLPK as an
                                item.
        ---------------------   -------------------------------------------
        gis                     Optional :class:`~arcgis.gis.GIS`  Object. Used for publishing the
                                item.
                                If not specified then active gis user is taken.
        ---------------------   -------------------------------------------
        kwargs                  Optional Parameters:
                                Boolean `overwrite` if True, it will overwrite
                                the item on ArcGIS Online/Enterprise.
                                Default is False.
        =====================   ===========================================
        """
        from ._arcgis_model import _create_zip

        zip_files = kwargs.pop("zip_files", True)
        path = super()._save(
            name_or_path,
            framework=framework,
            publish=publish,
            gis=gis,
            zip_files=False,
            **kwargs
        )
        self._data.vocab.save(path / "vocab")
        if zip_files:
            _create_zip(path.name, str(path))
        return path

    def load(self, name_or_path):
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
        from fastai.text.transform import Vocab

        super().load(name_or_path)
        model_path = self.learn.path / "models"
        path_like = "/" in name_or_path or "\\" in name_or_path
        name = Path(name_or_path).name if path_like else name_or_path
        if path_like:
            model_path = Path(name_or_path)
            if model_path.is_file():
                vocab_path = model_path.parent / "vocab"
            else:
                vocab_path = model_path / "vocab"
        else:
            vocab_path = model_path / name / "vocab"
        self._data.vocab = Vocab.load(vocab_path)  # load vocab.
