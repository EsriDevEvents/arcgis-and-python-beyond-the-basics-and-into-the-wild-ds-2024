import os, sys
import tempfile
from functools import partial
from pathlib import Path
import json
import warnings
import traceback
from ..models._arcgis_model import ArcGISModel, model_characteristics_folder

HAS_FASTAI = True

try:
    import transformers
    import torch
    import torch.nn as nn
    import pandas as pd
    from fastai.text.data import NumericalizeProcessor, TokenizeProcessor
    from fastai.text.transform import Tokenizer
    from fastprogress.fastprogress import progress_bar
    from fastai.basic_train import Learner, DatasetType
    from fastai.train import to_fp16
    from fastai.metrics import accuracy
    from transformers import AutoTokenizer, AutoConfig
    from .._utils.text_data import (
        TextDataObject,
        save_data_in_model_metrics_html,
        copy_metrics,
    )
    from .._utils._seq2seq_utils import (
        SequenceToSequenceLearner,
        CorpusBLEU,
        seq2seq_loss,
        seq2seq_acc,
    )
    from .._utils.text_transforms import TransformersBaseTokenizer, TransformersVocab
    from ._arcgis_transformer import ModelBackbone, infer_model_type
    from .._utils.common import _get_emd_path
    from ._transformer_seq2seq import (
        TransformerForSequenceToSequence,
        backbone_models_reverse_map,
        transformer_architectures,
        transformer_seq_length,
    )
    from transformers import logging
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False
    from ._transformer_seq2seq import transformer_architectures, transformer_seq_length
else:
    warnings.filterwarnings("ignore", category=UserWarning, module="fastai")


class SequenceToSequence(ArcGISModel):
    """
    Creates a :class:`~arcgis.learn.text.SequenceToSequence` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required text data object, returned from
                            :class:`~arcgis.learn.prepare_textdata` function.
    ---------------------   -------------------------------------------
    backbone                Optional string. Specifying the HuggingFace
                            transformer model name to be used to train the
                            model. Default set to 't5-base'.

                            To learn more about the available models or
                            choose models that are suitable for your dataset,
                            kindly visit:- https://huggingface.co/transformers/pretrained_models.html
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
    mixed_precision         Optional Bool. Default set to False. If set
                            True, then mixed precision training is used
                            to train the model
    ---------------------   -------------------------------------------
    pretrained_path         Optional String. Path where pre-trained model
                            is saved. Accepts a Deep Learning Package
                            (DLPK) or Esri Model Definition(EMD) file.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.text.SequenceToSequence` model object for sequence_translation task.
    """

    # supported transformer backbones
    supported_backbones = transformer_architectures

    def __init__(self, data, backbone="t5-base", **kwargs):
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
        self._mixed_precision = kwargs.get("mixed_precision", False)
        self._seq_len = kwargs.get("seq_len", transformer_seq_length)
        self.shap_values = None
        self._create_text_learner_object(
            data,
            backbone,
            kwargs.get("pretrained_path", None),
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
    ):
        self._model_type = infer_model_type(backbone, transformer_architectures)
        self.logger.info(f"Inferred Backbone: {self._model_type}")
        pretrained_model_name = backbone
        transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        pad_first = True if transformer_tokenizer.padding_side == "left" else False
        pad_idx = transformer_tokenizer.pad_token_id
        base_tokenizer = TransformersBaseTokenizer(
            pretrained_tokenizer=transformer_tokenizer, seq_len=seq_len
        )
        tokenizer = Tokenizer(tok_func=base_tokenizer, pre_rules=[], post_rules=[])
        if sys.platform == "win32":
            tokenizer.n_cpus = 1
        vocab = TransformersVocab(tokenizer=transformer_tokenizer)
        numericalize_processor = NumericalizeProcessor(vocab=vocab)
        tokenize_processor = TokenizeProcessor(
            tokenizer=tokenizer, include_bos=False, include_eos=False
        )
        transformer_processor = [tokenize_processor, numericalize_processor]
        if data._is_empty or data._backbone != backbone:
            self.logger.info("Creating DataBunch")
            data._prepare_seq2seq_databunch(
                transformer_processor=transformer_processor,
                pad_first=pad_first,
                pad_idx=pad_idx,
                model_type=self._model_type,
                backbone=backbone,
            )
        databunch = data.get_databunch()

        config = AutoConfig.from_pretrained(pretrained_model_name)

        if pretrained_path is not None:
            pretrained_path = str(_get_emd_path(pretrained_path))

        model = TransformerForSequenceToSequence(
            architecture=self._model_type,
            pretrained_model_name=pretrained_model_name,
            config=config,
            pretrained_model_path=pretrained_path,
            seq_len=seq_len,
        )
        from IPython.utils import io

        with io.capture_output() as captured:
            model.init_model()  # output will be cleared
        n_y_vocab = model._config.vocab_size
        metrics = [seq2seq_acc, CorpusBLEU(n_y_vocab)]
        # metrics=[accuracy, CorpusBLEU(n_y_vocab)]
        loss_func = seq2seq_loss
        from fastai.layers import LabelSmoothingCrossEntropy, FlattenedLoss

        # loss_func = FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1)

        self.learn = SequenceToSequenceLearner(
            databunch, model, metrics=metrics, loss_func=loss_func, path=data.path
        )

        if pretrained_path is not None:
            self.load(pretrained_path)

        if mixed_precision:
            if self._model_type in ["t5"]:
                error_message = (
                    f"Mixed precision training is not supported for transformer model - {self._model_type.upper()}."
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
        return ["valid_loss", "seq2seq_acc", "corpus_bleu"]

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
        return TransformerForSequenceToSequence._available_backbone_models(architecture)

    def freeze(self):
        """
        Freeze up to last layer group to train only the last layer group of the model.
        """
        self.learn.freeze()

    @classmethod
    def from_model(cls, emd_path, data=None, **kwargs):
        """
        Creates an SequenceToSequence model object from a Deep Learning
        Package(DLPK) or Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Optional fastai Databunch. Returned data
                                object from :class:`~arcgis.learn.prepare_textdata` function or None for
                                inferencing.
                                Default value: None
        =====================   ===========================================

        :return: :class:`~arcgis.learn.text.SequenceToSequence` Object
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
        seq_len = emd.get("SequenceLength", transformer_seq_length)

        data_is_none = False
        if data is None:
            data_is_none = True
            data = TextDataObject(task="sequence_translation")
            data._backbone = pretrained_model
            data.create_empty_seq2seq_data(text_cols, label_cols)
            data.emd, data.emd_path = emd, emd_path.parent
        cls_object = cls(
            data,
            pretrained_model,
            pretrained_path=str(emd_path),
            mixed_precision=mixed_precision,
            seq_len=seq_len,
            **kwargs,
        )
        if data_is_none:
            cls_object._data._is_empty = True
        return cls_object

    def load(self, name_or_path):
        """
        Loads a saved SequenceToSequence model from disk.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        =====================   ===========================================
        """
        if "\\" in str(name_or_path) or "/" in str(name_or_path):
            name_or_path = str(_get_emd_path(name_or_path))
        return super().load(name_or_path, strict=False)

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
        if "\\" in name_or_path or "/" in name_or_path:
            path = name_or_path
        else:
            path = os.path.join(self._data.path, "models", name_or_path)
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))

        path = super().save(
            path,
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
            metrics = self.get_model_metrics()

        return metrics

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        # metrics = self.get_model_metrics()
        # _emd_template.update(metrics)
        is_multilabel_problem = True if len(self._data._label_cols) > 1 else False
        _emd_template["Architecture"] = self.learn.model._transformer_architecture
        _emd_template[
            "PretrainedModel"
        ] = self.learn.model._transformer_pretrained_model_name
        _emd_template["ModelType"] = "Transformer"
        _emd_template["MixedPrecisionTraining"] = self._mixed_precision
        _emd_template["TextColumns"] = self._data._text_cols
        _emd_template["LabelColumns"] = self._data._label_cols
        _emd_template["SequenceLength"] = self._seq_len
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
        rows = (
            rows if (rows <= self.learn.data.batch_size) else self.learn.data.batch_size
        )
        return self.learn.show_results(rows=rows, **kwargs)

    def get_model_metrics(self):
        """
        Calculates the following  metrics:

        * accuracy:   the number of correctly predicted labels in the validation set divided by the total number of items in the validation set

        * bleu-score  This value indicates the similarity between model predictions and the ground truth text. Maximum value is 1

        :return: a dictionary containing the metrics for classification model.
        """
        try:
            self._check_requisites()
        except Exception as e:
            acc, bleu = self._data.emd.get("seq2seq_acc"), self._data.emd.get("bleu")
            if acc or bleu:
                return {"seq2seq_acc": acc, "bleu": bleu}
            else:
                self.logger.error("Metric not found in the loaded model")
        else:
            if hasattr(self.learn, "recorder"):
                metrics_names = self.learn.recorder.metrics_names
                metrics_values = self.learn.recorder.metrics
                if len(metrics_names) > 0 and len(metrics_values) > 0:
                    metrics = {
                        x: round(float(metrics_values[-1][i]), 4)
                        for i, x in enumerate(metrics_names)
                    }
                else:
                    metrics = self._calculate_model_metrics()
            else:
                metrics = self._calculate_model_metrics()
            return metrics

    def _calculate_model_metrics(self):
        self._check_requisites()
        self.logger.info("Calculating Model Metrics")
        metrics_names = ["accuracy", "bleu"]
        metrics_values = self.learn.validate()[1:]  # 0th value is validation loss
        metrics = {}
        if len(metrics_names) > 0 and len(metrics_values) > 0:
            metrics = {
                x: round(float(metrics_values[i]), 4)
                for i, x in enumerate(metrics_names)
            }
        return metrics

    def predict(
        self,
        text_or_list,
        batch_size=64,
        show_progress=True,
        explain=False,
        explain_index=None,
        **kwargs,
    ):
        """
        Predicts the translated outcome.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required input string or list of input strings.
        ---------------------   -------------------------------------------
        batch_size              Optional integer.
                                Number of inputs to be processed at once.
                                Try reducing the batch size in case of out of
                                memory errors.
                                Default value : 64
        ---------------------   -------------------------------------------
        show_progress           Optional bool.
                                To show or not to show the progress of prediction task.
                                Default value : True
        ---------------------   -------------------------------------------
        explain                 Optional bool.
                                To enable shap based importance
                                Default value : False
        ---------------------   -------------------------------------------
        explain_index           Optional list.
                                Index of the input rows for which the importance score will
                                be generated
                                Default value : None
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        num_beams               Optional integer.
                                Number of beams for beam search. 1 means no beam search.
                                Default value is set to 1
        ---------------------   -------------------------------------------
        max_length              Optional integer.
                                The maximum length of the sequence to be generated.
                                Default value is set to 20
        ---------------------   -------------------------------------------
        min_length              Optional integer.
                                The minimum length of the sequence to be generated.
                                Default value is set to 10
        =====================   ===========================================

        :return: list of tuples(input , predicted output strings).
        """

        if isinstance(text_or_list, str):
            text_or_list = [text_or_list]
        preds = self.learn.predict(text_or_list, batch_size, show_progress, **kwargs)

        # select the specific rows for the explanation
        text_list_for_exp = []
        try:
            if explain:
                if isinstance(explain_index, list):
                    # validate the index
                    temp_index = []
                    invalid_index = []
                    for index in explain_index:
                        if isinstance(index, int):
                            temp_index.append(index)
                        else:
                            invalid_index.append(index)

                    if invalid_index:
                        with warnings.catch_warnings():
                            warnings.simplefilter("always", UserWarning)
                            warnings.warn(
                                f"Index {invalid_index} are not valid. Indices/index must be integer. Ignoring "
                                f"{invalid_index} for processing."
                            )

                    if temp_index:
                        for i in temp_index:
                            if i < len(text_or_list):
                                text_list_for_exp.append(text_or_list[i])
                            else:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("always", UserWarning)
                                    warnings.warn(
                                        f"Value of index {i} should be less than/equal to {len(text_or_list) -1}."
                                    )
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("always", UserWarning)
                            warnings.warn(
                                f"No valid indices were supplied. Please change your input to list of integers"
                            )

                elif isinstance(explain_index, int):
                    if explain_index < len(text_or_list):
                        text_list_for_exp = text_or_list[explain_index]
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("always", UserWarning)
                            warnings.warn(
                                f"Value of index {explain_index} should be less than/equal to {len(text_or_list) - 1}."
                            )
                else:
                    exp_rows = 5 if len(text_or_list) > 5 else len(text_or_list)
                    for i in range(exp_rows):
                        text_list_for_exp.append(text_or_list[i])
                    with warnings.catch_warnings():
                        warnings.simplefilter("always", UserWarning)
                        warnings.warn(
                            f"Generating explanation for first {exp_rows} rows "
                        )

            if explain and len(text_list_for_exp):
                self._explain(text_list_for_exp, **kwargs)

        except:
            with warnings.catch_warnings():
                warnings.simplefilter("always", UserWarning)
                warnings.warn(
                    f"SHAP workflow has encountered an error. Failed to generate an explanation."
                )

        return list(zip(text_or_list, preds))

    def _save_df_to_html(self, path):
        if getattr(self._data, "_is_empty", False):
            copy_metrics(self._data.emd_path, path, model_characteristics_folder)
            return
        validation_dataframe = self._data._valid_df.sample(n=5)

        model_output = self.predict(
            validation_dataframe[self._data._text_cols].tolist(), show_progress=False
        )
        predictions = [pred for _, pred in model_output]
        labels = [
            x[0] for x in validation_dataframe[self._data._label_cols].values.tolist()
        ]
        new_df = pd.DataFrame(
            validation_dataframe[self._data._text_cols].values, columns=["input"]
        )
        new_df["target"] = labels
        new_df["predictions"] = predictions

        df_str = new_df.to_html(index=False, justify="left").replace(">\n", ">")

        msg = "<p><b>Sample Results</b></p>"

        text = f"\n\n{msg}\n\n{df_str}"

        save_data_in_model_metrics_html(text, path, model_characteristics_folder)

    def plot_losses(self, show=True):
        """
        Plot training and validation losses.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        show                    Optional bool. Defaults to True
                                If set to False, figure will not be plotted
                                but will be returned, when set to True function
                                will plot the figure and return nothing.
        =====================   ===========================================

        :return: `matplotlib.figure.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_
        """
        self._check_requisites()
        import matplotlib.pyplot as plt

        if not hasattr(self.learn, "recorder"):  # return none if the recorder is empty
            self.logger.error(
                "Model needs to be trained first. Please call `model.fit()` to train the model."
                " Then call this method to plot/return the loss curve."
            )
            return
        return_fig = not show
        fig = self.learn.recorder.plot_losses(return_fig=return_fig)
        if return_fig:
            plt.close()
            return fig

    def _explain(self, text_list_for_exp, **kwargs):
        """
        Generates SHAP based explanation.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        text_list_for_exp       Required input string or list of input strings.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        num_beams               Optional integer.
                                Number of beams for beam search. 1 means no beam search.
                                Default value is set to 1
        ---------------------   -------------------------------------------
        max_length              Optional integer.
                                The maximum length of the sequence to be generated.
                                Default value is set to 20
        ---------------------   -------------------------------------------
        min_length              Optional integer.
                                The minimum length of the sequence to be generated.
                                Default value is set to 10
        =====================   ===========================================

        :return: list of tuples(input , predicted output strings).
        """
        has_shap = True
        try:
            import shap
        except:
            has_shap = False
            warnings.warn(
                "SHAP is not installed. Model explainablity will not be available"
            )
        if has_shap:
            if isinstance(text_list_for_exp, str):
                text_list_for_exp = [text_list_for_exp]

            # collect the model and tokenizer
            emodel = self.learn.model._transformer
            emask = self.learn.model._tokenizer

            ###
            # There is an issue with the fast tokenizer. So selecting the slow tokenizer
            ###
            if isinstance(emask, transformers.PreTrainedTokenizerFast):
                if emask.name_or_path.find("T5"):
                    emask = transformers.T5TokenizerFast.from_pretrained(
                        "google/t5-v1_1-base", from_slow=True
                    )

            # initialize the explainer.
            explainer = shap.Explainer(emodel, emask)

            # validated the kwargs. So that we can match maximum length as passed to inference model
            try:
                for key, val in kwargs.items():
                    if hasattr(explainer.masker.model.inner_model.config, key):
                        explainer.masker.model.inner_model.config.__dict__[key] = val
            except:
                pass

            # generate shap values and generate plot.
            for record in text_list_for_exp:
                self.shap_values = explainer([record], fixed_context=0)
                shap.plots.text(self.shap_values)

    # def summarize(self, text_or_list, num_beams=4, max_len=50):
    #     if self._model_type not in['t5']:
    #         return logger.warning(f'Function not implemented for {self._model_type} models')
    #     if isinstance(text_or_list, str):
    #        text_or_list = ['summarize : ' + text_or_list]
    #     elif isinstance(text_or_list, list):
    #         text_or_list = ['summarize : '+ x for x in text_or_list]
    #     return self.learn.predict(text_or_list)

    # def qna(self, question_text_or_list, context_text_or_list, num_beams=4, max_len=50):
    #     if self._model_type not in['t5']:
    #         return logger.warning(f'Function not implemented for {self._model_type} models')
    #     if type(question_text_or_list) != type(context_text_or_list):
    #         return('Questions and context must either be both of string type or equal length lists of strings.')
    #     if isinstance(question_text_or_list, str) and isinstance(context_text_or_list, str):
    #         question_text_or_list = 'question: ' + question_text_or_list
    #         context_text_or_list = ' context: ' + context_text_or_list
    #         text_or_list = [question_text_or_list + context_text_or_list]
    #     elif isinstance(question_text_or_list, list) and isinstance(context_text_or_list, list):
    #         question_text_or_list = ['question: ' + x for x in  question_text_or_list]
    #         context_text_or_list = [' context: ' + x for x in  context_text_or_list]
    #         text_or_list = [x+y for x,y in zip(question_text_or_list,context_text_or_list)]
    #     return self.learn.predict(text_or_list)
