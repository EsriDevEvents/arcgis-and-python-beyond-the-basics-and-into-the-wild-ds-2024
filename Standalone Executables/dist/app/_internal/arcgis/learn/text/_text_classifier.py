import copy
from functools import partial
from pathlib import Path
import sys
import json
import warnings
import traceback
from ..models._arcgis_model import ArcGISModel, model_characteristics_folder
from .._utils._shap_masker import custom_tokenizer

HAS_NUMPY = True
HAS_FASTAI = True

try:
    import torch
    import torch.nn as nn
    import pandas as pd
    from fastai.text.transform import Tokenizer
    from fastprogress.fastprogress import progress_bar
    from fastai.basic_train import Learner, DatasetType
    from fastai.train import to_fp16
    from fastai.metrics import accuracy, error_rate, accuracy_thresh

    # from transformers import AdamW
    from transformers import AutoTokenizer, AutoConfig
    from .._utils.env import _LAMBDA_TEXT_CLASSIFICATION

    if not _LAMBDA_TEXT_CLASSIFICATION:
        from sklearn.metrics import classification_report
    from ._arcgis_transformer import ModelBackbone, infer_model_type
    from .._utils.common import _get_emd_path
    from .._utils.text_data import (
        TextDataObject,
        save_data_in_model_metrics_html,
        copy_metrics,
    )
    from .._utils.text_transforms import TransformersBaseTokenizer, TransformersVocab
    from ._transformer_text_classifier import (
        TransformerForTextClassification,
        backbone_models_reverse_map,
        transformer_architectures,
        transformer_seq_length,
    )
    from transformers import logging

    logging.get_logger("filelock").setLevel(logging.ERROR)
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False
    from ._transformer_text_classifier import (
        transformer_architectures,
        transformer_seq_length,
    )

    class TransformerForTextClassification:
        _supported_backbones = transformer_architectures

else:
    warnings.filterwarnings("ignore", category=UserWarning, module="fastai")

try:
    import numpy as np

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except:
    HAS_NUMPY = False


warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


class TextClassifier(ArcGISModel):
    """
    Creates a :class:`~arcgis.learn.text.TextClassifier` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Optional data object returned from :class:`~arcgis.learn.prepare_textdata` function.
                            data object can be `None`, in case where someone wants to use a
                            Hugging Face Transformer model fine-tuned on classification task.
                            In this case the model should be used directly for inference.
    ---------------------   -------------------------------------------
    backbone                Optional string. Specifying the HuggingFace
                            transformer model name to be used to train the
                            classifier. Default set to `bert-base-cased`.

                            To learn more about the available models or
                            choose models that are suitable for your dataset,
                            kindly visit:- https://huggingface.co/transformers/pretrained_models.html

                            To learn more about the available transformer models fine-tuned
                            on Text Classification Task, kindly visit:-
                            https://huggingface.co/models?pipeline_tag=text-classification
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    verbose                 Optional string. Default set to `error`. The
                            log level you want to set. It means the amount
                            of information you want to display while training
                            or calling the various methods of this class.
                            Allowed values are - `debug`, `info`, `warning`,
                            `error` and `critical`.
    ---------------------   -------------------------------------------
    seq_len                 Optional Integer. Default set to 512. Maximum
                            sequence length (at sub-word level after tokenization)
                            of the training data to be considered for training
                            the model.
    ---------------------   -------------------------------------------
    thresh                  Optional Float. This parameter is used to set
                            the threshold value to pick labels in case of
                            multi-label text classification problem. Default
                            value is set to 0.25
    ---------------------   -------------------------------------------
    mixed_precision         Optional Bool. Default set to False. If set
                            True, then mixed precision training is used
                            to train the model
    ---------------------   -------------------------------------------
    pretrained_path         Optional String. Path where pre-trained model
                            is saved. Accepts a Deep Learning Package
                            (DLPK) or Esri Model Definition(EMD) file.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.text.TextClassifier` Object
    """

    # supported transformer backbones
    supported_backbones = transformer_architectures

    def __init__(self, data, backbone="bert-base-cased", **kwargs):
        if not HAS_FASTAI:
            from .._data import _raise_fastai_import_error

            _raise_fastai_import_error(import_exception=import_exception)

        self.logger = logging.get_logger()
        if kwargs.get("verbose", None):
            self.logger.setLevel(kwargs.get("verbose").upper())
        else:
            self.logger.setLevel(logging.ERROR)

        model_backbone = ModelBackbone(backbone)
        super().__init__(data, model_backbone)
        self._emodel = None
        self._emask = None
        self.shap_values = None
        self.is_multilabel_problem = False
        self.thresh = kwargs.get("thresh", 0.25)
        self._mixed_precision = kwargs.get("mixed_precision", False)
        self._seq_len = kwargs.get("seq_len", transformer_seq_length)
        model_config = kwargs.get("model_config", None)
        if data is None:
            model = TextClassifier.from_pretrained(backbone, **kwargs)
            self.learn = model.learn
            self._data = model._data
        else:
            self._create_text_learner_object(
                data,
                backbone,
                kwargs.get("pretrained_path", None),
                config=model_config,
                mixed_precision=self._mixed_precision,
                seq_len=self._seq_len,
            )

            self.learn.model = self.learn.model.to(self._device)
            layer_groups = self.learn.model.get_layer_groups()
            self.learn.split(layer_groups)
            # self._freeze()

    def _create_text_learner_object(
        self,
        data,
        backbone,
        pretrained_path=None,
        mixed_precision=False,
        seq_len=transformer_seq_length,
        config=None,
    ):
        model_type = infer_model_type(backbone, transformer_architectures)
        self.logger.info(f"Inferred Backbone: {model_type}")
        pretrained_model_name = backbone

        if not config:
            config = AutoConfig.from_pretrained(pretrained_model_name)
        transformer_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, config=config
        )

        # pad_first = bool(model_type in ['xlnet'])
        pad_first = True if transformer_tokenizer.padding_side == "left" else False
        pad_idx = transformer_tokenizer.pad_token_id

        base_tokenizer = TransformersBaseTokenizer(
            pretrained_tokenizer=transformer_tokenizer, seq_len=seq_len
        )
        tokenizer = Tokenizer(tok_func=base_tokenizer, pre_rules=[], post_rules=[])
        if sys.platform == "win32":
            tokenizer.n_cpus = 1
        vocab = TransformersVocab(tokenizer=transformer_tokenizer)

        if data._is_empty or data._backbone != backbone:
            self.logger.info("Creating DataBunch")
            classes = None
            data._prepare_databunch(
                tokenizer=tokenizer,
                vocab=vocab,
                pad_first=pad_first,
                pad_idx=pad_idx,
                backbone=backbone,
                classes=classes,
                logger=self.logger,
            )

        databunch = data.get_databunch()
        if config.label2id != databunch.train_ds.c2i:
            config.label2id = databunch.train_ds.c2i
            config.id2label = {y: x for x, y in config.label2id.items()}

        if pretrained_path is not None:
            pretrained_path = str(_get_emd_path(pretrained_path))

        model = TransformerForTextClassification(
            architecture=model_type,
            pretrained_model_name=pretrained_model_name,
            config=config,
            pretrained_model_path=pretrained_path,
            seq_len=seq_len,
        )
        model.init_model()
        # opt_func = partial(AdamW, correct_bias=False)

        self.is_multilabel_problem = True if len(data._label_cols) > 1 else False
        if self.is_multilabel_problem:
            accuracy_multi = partial(accuracy_thresh, thresh=self.thresh)
            accuracy_multi.__name__ = "accuracy"
            metrics = [accuracy_multi]
            loss_func = nn.BCEWithLogitsLoss()
            # self.learn = Learner(databunch, model, opt_func=opt_func, loss_func=loss_func, metrics=metrics)
            self.learn = Learner(
                databunch, model, loss_func=loss_func, metrics=metrics, path=data.path
            )
        else:
            metrics = [accuracy, error_rate]

            # self.learn = Learner(databunch, model, opt_func=opt_func, metrics=metrics)
            self.learn = Learner(databunch, model, metrics=metrics, path=data.path)

        if pretrained_path is not None:
            self.load(pretrained_path)

        if mixed_precision:
            if model_type in ["xlnet", "mobilebert"]:
                error_message = (
                    f"Mixed precision training is not supported for transformer model - {model_type.upper()}."
                    "\nKindly turn off the `mixed_precision` flag to use this model in its default mode,"
                    f" or choose a different transformer architectures from - {transformer_architectures}"
                )
                raise Exception(error_message)
            self.logger.info("Converting model to 16 Bit Floating Point precision")
            self.learn = to_fp16(self.learn)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy", "error_rate"]

    @classmethod
    def available_backbone_models(cls, architecture):
        """
        Get available models for the given transformer backbone

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        architecture            Required string. name of the transformer
                                backbone one wish to use. To learn more about
                                the available models or choose models that are
                                suitable for your dataset, kindly visit:-
                                https://huggingface.co/transformers/pretrained_models.html
        =====================   ===========================================

        :return: a tuple containing the available models for the given transformer backbone
        """
        if not HAS_FASTAI:
            from .._data import _raise_fastai_import_error

            _raise_fastai_import_error(import_exception=import_exception)
        return TransformerForTextClassification._available_backbone_models(architecture)

    def freeze(self):
        """
        Freeze up to last layer group to train only the last layer group of the model.
        """
        self.learn.freeze()

    @classmethod
    def from_pretrained(cls, backbone, **kwargs):
        """
        Creates an TextClassifier model object from an already fine-tuned
        Hugging Face Transformer backbone.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        backbone                Required string. Specify the Hugging Face Transformer
                                backbone name fine-tuned on Text Classification task.

                                To get more details on available transformer models
                                fine-tuned on Text Classification Task, kindly visit:-
                                https://huggingface.co/models?pipeline_tag=text-classification

        =====================   ===========================================

        :return: :class:`~arcgis.learn.text.TextClassifier` Object
        """
        if not HAS_FASTAI:
            from .._data import _raise_fastai_import_error

            _raise_fastai_import_error(import_exception=import_exception)

        model_config = AutoConfig.from_pretrained(backbone)
        class_labels = list(model_config.id2label.values())
        data = TextDataObject(task="classification")
        data._backbone = backbone
        data.create_empty_object_for_classification("", [], class_labels)

        cls_object = cls(data, backbone, model_config=model_config, **kwargs)
        data.emd, data.emd_path = cls_object._get_emd_params(), None
        cls_object._data._is_empty = True
        return cls_object

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates an TextClassifier model object from a Deep Learning
        Package(DLPK) or Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :class:`~arcgis.learn.prepare_textdata` function or None for
                                inferencing.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.text.TextClassifier` model Object
        """
        if not HAS_FASTAI:
            from .._data import _raise_fastai_import_error

            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        pretrained_model = emd["PretrainedModel"]
        mixed_precision = emd["MixedPrecisionTraining"]
        text_cols = emd["TextColumns"]
        label_cols = emd["LabelColumns"]
        class_labels = list(emd["Label2Id"].keys())
        is_multilabel_problem = emd["IsMultilabelClassificationProblem"]
        thresh = emd.get("Threshold")
        seq_len = emd.get("SequenceLength", transformer_seq_length)

        data_is_none = False
        if data is None:
            data_is_none = True
            data = TextDataObject(task="classification")
            data._backbone = pretrained_model
            data.create_empty_object_for_classification(
                text_cols, label_cols, class_labels, is_multilabel_problem
            )
            data.emd, data.emd_path = emd, emd_path.parent
        cls_object = cls(
            data,
            pretrained_model,
            pretrained_path=str(emd_path),
            mixed_precision=mixed_precision,
            thresh=thresh,
            seq_len=seq_len,
        )
        if data_is_none:
            cls_object._data._is_empty = True
        return cls_object

    def load(self, name_or_path):
        """
        Loads a saved TextClassifier model from disk.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        =====================   ===========================================
        """
        if "\\" in str(name_or_path) or "/" in str(name_or_path):
            name_or_path = str(_get_emd_path(name_or_path))
        else:
            name_or_path = Path("models") / name_or_path
            name_or_path = str(_get_emd_path(name_or_path))
        return super().load(name_or_path)

    def save(
        self,
        name_or_path,
        framework="PyTorch",
        publish=False,
        gis=None,
        compute_metrics=True,
        save_optimizer=False,
        **kwargs,
    ):
        """
        Saves the model weights, creates an Esri Model Definition and Deep
        Learning Package zip for deployment.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Folder path to save the model.
        ---------------------   -------------------------------------------
        framework               Optional string. Defines the framework of the
                                model. (Only supported by :class:`~arcgis.learn.SingleShotDetector`, currently.)
                                If framework used is ``TF-ONNX``, ``batch_size`` can be
                                passed as an optional keyword argument.

                                Framework choice: 'PyTorch' and 'TF-ONNX'
        ---------------------   -------------------------------------------
        publish                 Optional boolean. Publishes the DLPK as an item.
        ---------------------   -------------------------------------------
        gis                     Optional :class:`~arcgis.gis.GIS`  Object. Used for publishing the item.
                                If not specified then active gis user is taken.
        ---------------------   -------------------------------------------
        compute_metrics         Optional boolean. Used for computing model
                                metrics.
        ---------------------   -------------------------------------------
        save_optimizer          Optional boolean. Used for saving the model-optimizer
                                state along with the model. Default is set to False.
        ---------------------   -------------------------------------------
        kwargs                  Optional Parameters:
                                Boolean `overwrite` if True, it will overwrite
                                the item on ArcGIS Online/Enterprise, default False.
                                Boolean `zip_files` if True, it will create the Deep
                                Learning Package (DLPK) file while saving the model.
        =====================   ===========================================

        :return: the qualified path at which the model is saved
        """
        from ..models._arcgis_model import _create_zip

        zip_files = kwargs.pop("zip_files", True)
        overwrite = kwargs.pop("overwrite", False)
        path = super().save(
            name_or_path,
            framework,
            publish=False,
            gis=None,
            compute_metrics=compute_metrics,
            save_optimizer=save_optimizer,
            zip_files=False,
            **kwargs,
        )

        self._save_df_to_html(path)

        if zip_files:
            _create_zip(path.name, str(path))

        if publish:
            self._publish_dlpk(
                (path / path.stem).with_suffix(".dlpk"), gis=gis, overwrite=overwrite
            )

        return Path(path)

    @property
    def _model_metrics(self):
        from IPython.utils import io

        with io.capture_output() as captured:
            metrics = {"Accuracy": self.accuracy()}
            per_class_metric_df = self.metrics_per_label()
            metrics["MetricsPerLabel"] = json.dumps(
                per_class_metric_df.transpose().to_dict()
            )
        return metrics

    def _get_emd_params(self, save_inference_file=True):
        _emd_template = {}
        is_multilabel_problem = True if len(self._data._label_cols) > 1 else False
        _emd_template["Architecture"] = self.learn.model._transformer_architecture
        _emd_template[
            "PretrainedModel"
        ] = self.learn.model._transformer_pretrained_model_name
        _emd_template["ModelType"] = "Transformer"
        _emd_template["MixedPrecisionTraining"] = self._mixed_precision
        _emd_template["TextColumns"] = self._data._text_cols
        _emd_template["LabelColumns"] = self._data._label_cols
        _emd_template["Label2Id"] = self.learn.model._config.label2id
        _emd_template["SequenceLength"] = self._seq_len
        if is_multilabel_problem:
            _emd_template["Threshold"] = self.thresh
        _emd_template["IsMultilabelClassificationProblem"] = is_multilabel_problem
        return _emd_template

    def show_results(self, rows=5, **kwargs):
        """
        Prints the rows of the dataframe with target and prediction columns.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional Integer.
                                Number of rows to print.
        =====================   ===========================================

        :return: dataframe
        """
        self._check_requisites()
        if kwargs.get("thresh") is None and self.is_multilabel_problem:
            kwargs.update({"thresh": self.thresh})
        return self.learn.show_results(rows=rows, **kwargs)

    def accuracy(self):
        """
        Calculates the following  metric:

        * accuracy:   the number of correctly predicted labels in the validation set divided by the total number of items in the validation set

        :return: a floating point number depicting the accuracy of the classification model.
        """
        try:
            self._check_requisites()
        except Exception as e:
            acc = self._data.emd.get("Accuracy")
            if acc:
                return acc
            else:
                self.logger.error("Metric not found in the loaded model")
        else:
            if not HAS_NUMPY:
                self.logger.error("This function requires numpy.")
                return
            if hasattr(self.learn, "recorder"):
                metrics_names = self.learn.recorder.metrics_names
                metrics_values = self.learn.recorder.metrics
                if len(metrics_names) > 0 and len(metrics_values) > 0:
                    metrics = {
                        x: round(metrics_values[-1][i].item(), 4)
                        for i, x in enumerate(metrics_names)
                    }
                    metric = metrics["accuracy"]
                else:
                    metric = self._calculate_model_metric()
            else:
                metric = self._calculate_model_metric()
            return metric

    def _calculate_model_metric(self):
        self.logger.info("Calculating Model Metrics")
        validation_dataframe = self._data._valid_df

        if self.is_multilabel_problem:
            predictions = [
                x[2]
                for x in self.predict(
                    validation_dataframe[self._data._text_cols].tolist()
                )
            ]
            labels = [
                [int(getattr(item, column)) for column in self._data._label_cols]
                for idx, item in validation_dataframe.iterrows()
            ]

            metric = accuracy_thresh(
                torch.tensor(predictions),
                torch.tensor(labels),
                thresh=self.thresh,
                sigmoid=False,
            ).item()
        else:
            predictions = [
                x[1]
                for x in self.predict(
                    validation_dataframe[self._data._text_cols].tolist()
                )
            ]
            labels = [x[0] for x in validation_dataframe[self._data._label_cols].values]
            metric = round((np.sum(np.array(predictions) == labels) / len(labels)), 4)

        return metric

    def _predict(self, text, thresh=None):
        if thresh is None:
            thresh = self.thresh
        result = self.learn.model.predict_class(
            text, self._device, self.is_multilabel_problem, thresh
        )
        return result

    def _predict_batch(self, text, thresh=None):
        if thresh is None:
            thresh = self.thresh
        result = self.learn.model.predict_class_batch(
            text, self._device, self.is_multilabel_problem, thresh
        )
        return result

    def predict(
        self,
        text_or_list,
        show_progress=True,
        thresh=None,
        explain=False,
        explain_index=None,
        batch_size=64,
    ):
        """
        Predicts the class label(s) for the input text

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required String or List. text or a list of
                                texts for which we wish to find the class label(s).
        ---------------------   -------------------------------------------
        show_progress           optional Bool. If set to True, will display a
                                progress bar depicting the items processed so far.
                                Applicable only when a list of text is passed
        ---------------------   -------------------------------------------
        thresh                  Optional Float. The threshold value set to get
                                the class label(s). Applicable only for multi-label
                                classification task. Default is the value set
                                during the model creation time, otherwise the value
                                of 0.25 is set.
        ---------------------   -------------------------------------------
        explain                 Optional Bool. If set to True it shall generate SHAP
                                based explanation. Kindly visit:-
                                https://shap.readthedocs.io/en/latest/
        ---------------------   -------------------------------------------
        explain_index           Optional List. Index of the rows for which explanation
                                is required.  If the value is None, it will generate
                                an explanation for every row.
        ---------------------   -------------------------------------------
        batch_size              Optional integer.
                                Number of inputs to be processed at once.
                                Try reducing the batch size in case of out of
                                memory errors.
                                Default value : 64
        =====================   ===========================================

        :return: * In case of single label classification problem, a tuple containing the text, its predicted class label and the confidence score.

                 * In case of multi label classification problem, a tuple containing the text, its predicted class labels, a list containing 1's for the predicted labels, 0's otherwise and list containing a score for each label
        """
        if explain:
            try:
                import shap
            except:
                warnings.warn(
                    "SHAP is not installed. Model explainablity will not be available"
                )
                explain = False
                explain_index = None

        if self.is_multilabel_problem is False and thresh is not None:
            self.logger.error(
                "Passing a threshold value for non multi-label classification task "
                "will not have any affect on the predicting the class label"
            )

        sliced_text_list = []

        if isinstance(text_or_list, (list, tuple, np.ndarray)):
            preds = []
            if len(text_or_list) < batch_size:
                batch_size = len(text_or_list)

            remaining_len = len(text_or_list)
            if len(text_or_list) % batch_size == 0:
                iter_val = len(text_or_list) // batch_size
            else:
                iter_val = (len(text_or_list) // batch_size) + 1
            lower_range = 0
            upper_range = batch_size
            if show_progress:
                for ind in progress_bar(range(iter_val)):
                    prediction = self._predict_batch(
                        text_or_list[lower_range:upper_range], thresh
                    )
                    preds.extend(prediction)
                    remaining_len -= batch_size
                    lower_range = upper_range
                    if remaining_len >= batch_size:
                        upper_range = lower_range + batch_size
                    else:
                        upper_range = upper_range + remaining_len
            else:
                for ind in range(iter_val):
                    prediction = self._predict_batch(
                        text_or_list[lower_range:upper_range], thresh
                    )
                    preds.extend(prediction)
                    remaining_len -= batch_size
                    lower_range = upper_range
                    if remaining_len >= batch_size:
                        upper_range = lower_range + batch_size
                    else:
                        upper_range = upper_range + remaining_len
            result = [(text, *pred) for text, pred in zip(text_or_list, preds)]
            if explain:
                if isinstance(explain_index, int):
                    sliced_text_list = text_or_list[explain_index : explain_index + 1]
                elif isinstance(explain_index, (list, tuple, np.ndarray)):
                    for i in explain_index:
                        if i < len(text_or_list):
                            sliced_text_list.append(text_or_list[i])
                else:
                    sliced_text_list = copy.deepcopy(text_or_list)
                    warnings.warn(
                        "No Index is supplied. Going ahead with all the inputs"
                    )
        else:
            preds = self._predict(text_or_list, thresh)
            result = (text_or_list, *preds)
            if explain:
                sliced_text_list = [text_or_list]

        if explain:
            self._emodel, self._emask = self._wrapped_model_for_explnation()
            self._explain(sliced_text_list)

        return result

    def _save_df_to_html(self, path):
        if getattr(self._data, "_is_empty", False):
            if self._data.emd_path:
                copy_metrics(self._data.emd_path, path, model_characteristics_folder)
            return
        validation_dataframe = self._data._valid_df.sample(n=5)
        if self.is_multilabel_problem:
            predictions = [
                x[1]
                for x in self.predict(
                    validation_dataframe[self._data._text_cols].tolist(),
                    show_progress=False,
                )
            ]
            labels = [
                ";".join(
                    [
                        column
                        for column in self._data._label_cols
                        if int(getattr(item, column))
                    ]
                )
                for idx, item in validation_dataframe.iterrows()
            ]

        else:
            predictions = [
                x[1]
                for x in self.predict(
                    validation_dataframe[self._data._text_cols].tolist(),
                    show_progress=False,
                )
            ]
            labels = [x[0] for x in validation_dataframe[self._data._label_cols].values]

        new_df = pd.DataFrame(
            validation_dataframe[self._data._text_cols].values, columns=["source"]
        )
        new_df["target"] = labels
        new_df["predictions"] = predictions

        df_str = new_df.to_html(index=False, justify="left").replace(">\n", ">")

        msg = "<p><b>Sample Results</b></p>"

        text = f"\n\n{msg}\n\n{df_str}"

        save_data_in_model_metrics_html(text, path, model_characteristics_folder)

    def metrics_per_label(self):
        """
        :return: precision, recall and f1 score for each label in the classification model.
        """
        try:
            self._check_requisites()
        except Exception as e:
            metrics_per_label = self._data.emd.get("MetricsPerLabel")
            if metrics_per_label:
                metrics_per_label = json.loads(metrics_per_label)
                return self._create_dataframe_from_dict(metrics_per_label)
            else:
                self.logger.error("Metric not found in the loaded model")
        else:
            validation_dataframe = self._data._valid_df
            if self.is_multilabel_problem:
                predictions = [
                    x[2]
                    for x in self.predict(
                        validation_dataframe[self._data._text_cols].tolist()
                    )
                ]
                labels = [
                    [int(getattr(item, column)) for column in self._data._label_cols]
                    for idx, item in validation_dataframe.iterrows()
                ]
                target_names = self._data._label_cols
                output_dict = classification_report(
                    labels,
                    predictions,
                    target_names=target_names,
                    zero_division=1,
                    output_dict=True,
                )
            else:
                predictions = [
                    x[1]
                    for x in self.predict(
                        validation_dataframe[self._data._text_cols].tolist()
                    )
                ]
                labels = [
                    x[0] for x in validation_dataframe[self._data._label_cols].values
                ]
                target_names = self.learn.model._config.label2id.keys()
                if len(target_names) != len(set(labels)):
                    warnings.warn(
                        f'Validation dataset classes {list(set(labels))} does not match the training dataset \
classes {list(target_names)}, you could use "stratify=True" with prepare_textdata or try increasing the minority class \
samples. Metrics are only being calculated for classes present in the validation dataset.'
                    )
                    target_names = set(labels)
                output_dict = classification_report(
                    labels,
                    predictions,
                    target_names=target_names,
                    zero_division=1,
                    output_dict=True,
                )

            return self._create_dataframe_from_dict(output_dict)

    @staticmethod
    def _create_dataframe_from_dict(out_dict):
        out_dict.pop("accuracy", None)
        out_dict.pop("micro avg", None)
        out_dict.pop("macro avg", None)
        out_dict.pop("samples avg", None)
        out_dict.pop("weighted avg", None)
        df = pd.DataFrame(out_dict)
        # df.drop("support", inplace=True)
        dataframe = df.T.round(4)
        column_mappings = {
            "precision": "Precision_score",
            "recall": "Recall_score",
            "f1-score": "F1_score",
            "support": "Support",
        }
        dataframe.rename(columns=column_mappings, inplace=True)
        return dataframe

    def get_misclassified_records(self):
        """
        :return: get misclassified records for this classification model.
        """
        self._check_requisites()
        validation_dataframe = self._data._valid_df
        misclassified_records, text_col, label_col = (
            [],
            self._data._text_cols,
            self._data._label_cols,
        )
        if self.is_multilabel_problem:
            predictions = [
                x[1]
                for x in self.predict(
                    validation_dataframe[self._data._text_cols].tolist()
                )
            ]
            labels = [
                ";".join(
                    [
                        column
                        for column in self._data._label_cols
                        if int(getattr(item, column))
                    ]
                )
                for idx, item in validation_dataframe.iterrows()
            ]
            for index, row in validation_dataframe.iterrows():
                label, prediction = labels[index], predictions[index]
                if label != prediction:
                    misclassified_records.append((row[text_col], label, prediction))
        else:
            predictions = [
                x[1]
                for x in self.predict(
                    validation_dataframe[self._data._text_cols].tolist()
                )
            ]
            label_col = (
                label_col if isinstance(label_col, (str, bytes)) else label_col[0]
            )
            for index, row in validation_dataframe.iterrows():
                label, prediction = row[label_col], predictions[index]
                if label != prediction:
                    misclassified_records.append((row[text_col], label, prediction))

        return pd.DataFrame(
            misclassified_records, columns=[text_col, "Target", "Prediction"]
        )

    def _explain(self, text_or_list, custom_tok=True):
        # """
        # To Generate explanation for a single or a batch of input strings.
        #
        # This function is a wrapper around SHAP explainer function for the Language model.
        # It relies on two underlying methods.
        # 1. It will wrap the logits of the model
        # 2. It will produce single class as an output.
        # EntityRecognizer
        #
        # =====================   ===========================================
        # **Parameter**            **Description**
        # ---------------------   -------------------------------------------
        # text_or_list            Required String or List. text or a list of
        #                         texts for which we wish to find the class label(s).
        #
        # custom_tok              Setting this argument to True will return the explanation
        #                         based on the word boundary token.
        # =====================   ===========================================
        # :return: None
        #
        # """
        has_shap = True
        try:
            import shap
        except:
            has_shap = False
            warnings.warn(
                "SHAP is not installed. Model explainablity will not be available"
            )
        if has_shap:
            if isinstance(text_or_list, str):
                text_or_list = [text_or_list]
            elif not isinstance(text_or_list, list):
                raise Exception(f" This module takes string or list as an input")
            # Build custom masker
            masker = None
            if custom_tok:
                masker = shap.maskers.Text(custom_tokenizer)
            # create labels
            labels = sorted(
                self.learn.model._config.label2id,
                key=self.learn.model._config.label2id.get,
            )
            explainer = shap.Explainer(self._logit_wrapper, masker, output_names=labels)
            text_or_list = [
                text if len(text.split(" ")) > 1 else text + "  "
                for text in text_or_list
            ]
            self.shap_values = explainer(text_or_list)
            shap.plots.text(self.shap_values)

    def _wrapped_model_for_explnation(self):
        # """
        # It will return the wrapped transformer for the classification task. It will return the
        # transformer from learner as well as the tokenizer.
        # """
        return self.learn.model._transformer, self.learn.model._tokenizer

    def _logit_wrapper(self, input_sent):
        input_sent = list(input_sent)
        # This code is modified to accomodate the masking structure and making the implementation verbose.
        encoded_dict = self._emask(
            input_sent,
            max_length=self._seq_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        logits = self._emodel(
            encoded_dict["input_ids"].cuda(), encoded_dict["attention_mask"].cuda()
        )[0]
        results = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return results
