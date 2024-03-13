import os
import json
import random
import traceback
from pathlib import Path
from functools import partial
from ..models._arcgis_model import ArcGISModel, model_characteristics_folder

HAS_FASTAI = True

try:
    import torch
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    from transformers import logging

    logging.get_logger("filelock").setLevel(logging.ERROR)
    from fastai.train import to_fp16
    from fastprogress.fastprogress import progress_bar
    from fastai.basic_train import Learner, DatasetType
    from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
    )
    from .._utils.common import _get_emd_path
    from .._utils.text_transforms import get_results
    from .._utils.text_data import (
        save_data_in_model_metrics_html,
        TextDataObject,
        copy_metrics,
    )
    from ._arcgis_transformer import (
        ArcGISTransformer,
        ModelBackbone,
        infer_model_type,
        transformer_seq_length,
    )
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False
    transformer_seq_length = 512

    class ArcGISTransformer:
        pass


backbone_models_map = {
    "bert": (
        "bert-base-cased",
        "bert-base-uncased",
        "bert-large-cased",
        "bert-large-uncased",
    ),
    "albert": (
        "albert-base-v1",
        "albert-base-v2",
        "albert-large-v1",
        "albert-large-v2",
        "albert-xlarge-v1",
        "albert-xlarge-v2",
        "albert-xxlarge-v1",
        "albert-xxlarge-v2",
    ),
    "roberta": ("roberta-base", "roberta-large", "distilroberta-base"),
    "distilbert": ("distilbert-base-cased", "distilbert-base-uncased"),
    "xlnet": ("xlnet-base-cased", "xlnet-large-cased"),
    "xlm": (
        "xlm-mlm-ende-1024",
        "xlm-mlm-enfr-1024",
        "xlm-mlm-xnli15-1024",
        "xlm-mlm-en-2048",
    ),
    "xlm-roberta": ("xlm-roberta-base", "xlm-roberta-large"),
    "longformer": ("allenai/longformer-base-4096",),
    "electra": ("google/electra-base-discriminator", "google/electra-base-generator"),
    "mobilebert": ("google/mobilebert-uncased",),
    "camembert": ("camembert-base",),
    "flaubert": (
        "flaubert/flaubert_small_cased",
        "flaubert/flaubert_base_cased",
        "flaubert/flaubert_base_uncased",
        "flaubert/flaubert_large_cased",
    ),
    "funnel": (
        "funnel-transformer/small",
        "funnel-transformer/small-base",
        "funnel-transformer/medium",
        "funnel-transformer/medium-base",
    ),
}

transformer_architectures = [
    "BERT",
    "RoBERTa",
    "DistilBERT",
    "ALBERT",
    "CamemBERT",
    "MobileBERT",
    "XLNet",
    "XLM",
    "XLM-RoBERTa",
    "FlauBERT",
    "ELECTRA",
    "Longformer",
    "Funnel",
]

backbone_models_reverse_map = {
    x: key for key, val in backbone_models_map.items() for x in val
}


class TransformerNERLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, labels, masks):
        # TransformerNER model itself outputs loss when forward method is called
        return output[0]


class TransformerForEntityRecognition(ArcGISTransformer):
    _supported_backbones = transformer_architectures

    def __init__(
        self,
        architecture,
        pretrained_model_name,
        config=None,
        pretrained_model_path=None,
        seq_len=transformer_seq_length,
    ):
        super().__init__(
            architecture,
            pretrained_model_name,
            config,
            pretrained_model_path,
            task="ner",
        )
        self._seq_len = seq_len
        self._max_seq_len = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % type(self).__name__

    def _process_config(self):
        """
        Process the model config to convert the class values to integers in the
        id2label attribute (which is a dictionary) of the model config
        """
        self._config.id2label = {int(x): y for x, y in self._config.id2label.items()}

    def _set_max_seq_length(self):
        """
        Set the max-seq-len parameter to avoid out of memory issues for transformers
        like Bart, Longformer, XLNet
        """
        self._max_seq_len = min(self._tokenizer.model_max_length, self._seq_len)

    @classmethod
    def _available_backbone_models(cls, architecture):
        """
        Provides a list of available models for a given transformer architecture

        =====================   =================================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------------
        architecture            Required string. The transformer architecture for
                                which we wish to get the available models
        ---------------------   -------------------------------------------------
        return: tuple containing available models for the given architecture
        """
        if architecture.lower() in backbone_models_map:
            return backbone_models_map[architecture.lower()]
        else:
            return (
                f"Error, wrong architecture name - {architecture} supplied. "
                f"Please choose from {cls._supported_backbones}"
            )

    def save(self, model_path):
        # Not required as this functionality will be implemented by the ArcGISModel child class
        raise NotImplementedError

    @classmethod
    def load_pretrained_model(cls, path):
        # Not required as this functionality will be implemented by the ArcGISModel child class
        raise NotImplementedError

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        if labels is not None:
            output = self._transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
        else:
            # Models like DistilBERT, RoBERTa, XLM-RoBERTa, Longformer, Bart, ELECTRA
            # and CamemBERT does not return token_type_ids in encode_plus method
            labels = token_type_ids
            output = self._transformer(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        return output

    def _load_transformer(self):
        """
        Method to load the transformer model. For this class we want to load the
        transformer model for performing Token Classification tasks
        """
        if self._pretrained_model_path and os.path.exists(self._pretrained_model_path):
            base_path, extension = os.path.splitext(self._pretrained_model_path)
            path = base_path + ".pth"
            if extension == ".emd" and os.path.exists(path):
                self._transformer = AutoModelForTokenClassification.from_pretrained(
                    path, config=self._config
                )
            else:
                self._transformer = AutoModelForTokenClassification.from_pretrained(
                    self._pretrained_model_path
                )
        else:
            self._transformer = AutoModelForTokenClassification.from_pretrained(
                self._transformer_pretrained_model_name, config=self._config
            )

    @staticmethod
    def get_active_predictions_labels_per_batch(output, label, attention_mask):
        y_pred, y_true = [], []
        predictions = output[1].argmax(2)
        active_tensors = attention_mask.view(-1) == 1

        predictions_flat = predictions.view(-1).tolist()
        labels_flat = label.view(-1).tolist()
        for index, item in enumerate(active_tensors.tolist()):
            if item:
                y_pred.append(predictions_flat[index])
                y_true.append(labels_flat[index])
        return y_pred, y_true

    @staticmethod
    def precision_score(output, label, attention_mask, average="micro"):
        (
            y_pred,
            y_true,
        ) = TransformerForEntityRecognition.get_active_predictions_labels_per_batch(
            output, label, attention_mask
        )
        return torch.tensor(
            round(precision_score(y_true, y_pred, average=average, zero_division=1), 4)
        )

    @staticmethod
    def recall_score(output, label, attention_mask, average="micro"):
        (
            y_pred,
            y_true,
        ) = TransformerForEntityRecognition.get_active_predictions_labels_per_batch(
            output, label, attention_mask
        )
        return torch.tensor(
            round(recall_score(y_true, y_pred, average=average, zero_division=1), 4)
        )

    @staticmethod
    def f1_score(output, label, attention_mask, average="micro"):
        (
            y_pred,
            y_true,
        ) = TransformerForEntityRecognition.get_active_predictions_labels_per_batch(
            output, label, attention_mask
        )
        return torch.tensor(
            round(f1_score(y_true, y_pred, average=average, zero_division=1), 4)
        )

    @staticmethod
    def get_active_predictions_labels(data_loader, learner, show_progress=True):
        predictions, labels = [], []
        dataset = list(iter(data_loader))
        if show_progress:
            for i in progress_bar(range(len(data_loader))):
                batch = dataset[i]
                output = learner.model.forward(*batch[0])
                (
                    batch_preds,
                    batch_labels,
                ) = TransformerForEntityRecognition.get_active_predictions_labels_per_batch(
                    output, *batch[1]
                )
                predictions.extend(batch_preds)
                labels.extend(batch_labels)
        else:
            results = [
                TransformerForEntityRecognition.get_active_predictions_labels_per_batch(
                    learner.model.forward(*batch[0]), *batch[1]
                )
                for batch in dataset
            ]

            for item in results:
                predictions.extend(item[0])
                labels.extend(item[1])

        return predictions, labels

    def generate_inference(self, text_list, device):
        tokenizer = self._tokenizer
        max_seq_length = self._max_seq_len
        model_type = self._transformer_architecture
        token_list = [x.split(" ") for x in text_list]
        if model_type in ["roberta", "bart", "longformer"]:
            token_list = [[f" {y}" for y in x] for x in token_list]

        encodings = tokenizer.batch_encode_plus(
            token_list,
            max_length=max_seq_length,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        ).to(device)

        input_ids, attention_mask = (
            encodings.get("input_ids"),
            encodings.get("attention_mask"),
        )
        token_type_ids = encodings.get("token_type_ids")
        # Models like DistilBERT, RoBERTa, XLM-RoBERTa, Longformer, Bart, ELECTRA
        # and CamemBERT does not return token_type_ids in encode_plus method
        if token_type_ids is not None:
            output = self._transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            output = self._transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )

        batch_tokens = input_ids.tolist()
        batch_labels = output[0].argmax(2).tolist()

        return batch_tokens, batch_labels


class _TransformerEntityRecognizer(ArcGISModel):
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
        self.stats = "macro"
        self._is_empty = True
        self.train_ds, self.val_ds = None, None
        self._address_tag = self._data._address_tag
        self._mixed_precision = kwargs.get("mixed_precision", False)
        self._seq_len = kwargs.get("seq_len", transformer_seq_length)
        model_config = kwargs.get("model_config", None)
        self.path = getattr(data, "working_dir", data.path)
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
        # Freeze the model by default
        # self._freeze()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % type(self).__name__.replace("_", "")

    @classmethod
    def available_backbone_models(cls, architecture):
        if not HAS_FASTAI:
            from .._data import _raise_fastai_import_error

            _raise_fastai_import_error(import_exception=import_exception)
        return TransformerForEntityRecognition._available_backbone_models(architecture)

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
            pretrained_model_name, config=config, use_fast=False
        )
        if data._is_empty or data._backbone != backbone:
            self.logger.info("Creating DataBunch")
            data._prepare_databunch(
                tokenizer=transformer_tokenizer,
                model_type=model_type,
                seq_len=seq_len,
                backbone=backbone,
                logger=self.logger,
            )

        databunch = data.get_databunch()

        if config.label2id != data._label2id:
            config.label2id = data._label2id
            config.id2label = {y: x for x, y in config.label2id.items()}
            config._num_labels = len(config.label2id)

        if pretrained_path is not None:
            pretrained_path = str(_get_emd_path(pretrained_path))

        model = TransformerForEntityRecognition(
            architecture=model_type,
            pretrained_model_name=pretrained_model_name,
            config=config,
            pretrained_model_path=pretrained_path,
            seq_len=seq_len,
        )

        model.init_model()

        loss_func = TransformerNERLoss()

        f1 = partial(TransformerForEntityRecognition.f1_score, average=self.stats)
        recall = partial(
            TransformerForEntityRecognition.recall_score, average=self.stats
        )
        precision = partial(
            TransformerForEntityRecognition.precision_score, average=self.stats
        )
        metrics = [precision, recall, f1]

        self.learn = Learner(
            databunch, model, loss_func=loss_func, metrics=metrics, path=self.path
        )

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

        if databunch.is_empty:
            self.train_ds = None
            self.val_ds = None
            self._is_empty = True
        else:
            self.train_ds = databunch.train_ds
            self.valid_ds = databunch.valid_ds
            self._is_empty = False

        self.entities = data._unique_tags
        self._address_tag = data._address_tag

    def freeze(self):
        """
        Freeze up to last layer group to train only the last layer group of the model.
        """
        self.learn.freeze()

    def _save_df_to_html(self, path):
        if getattr(self._data, "_is_empty", False):
            if self._data.emd_path:
                copy_metrics(self._data.emd_path, path, model_characteristics_folder)
            return

        metrics_per_label = self.metrics_per_label(show_progress=False)
        metrics_per_label_str = metrics_per_label.to_html().replace(">\n", ">")

        dataframe = self.show_results()
        show_result_df_str = dataframe.to_html(index=False, justify="left").replace(
            ">\n", ">"
        )

        show_result_msg = "<p><b>Sample Results</b></p>"
        metrics_per_label_msg = "<p><b>Metrics per label</b></p>"

        text = f"\n\n{metrics_per_label_msg}\n\n{metrics_per_label_str}\n\n{show_result_msg}\n\n{show_result_df_str}"

        save_data_in_model_metrics_html(text, path, model_characteristics_folder)

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
            metrics = self._calculate_model_metrics()
            per_class_metric_df = self.metrics_per_label()
            metrics["metrics_per_label"] = per_class_metric_df.transpose().to_dict()
        return {"Metrics": json.dumps(metrics)}

    def _get_emd_params(self, save_inference_file=True):
        _emd_template = {}
        _emd_template["Architecture"] = self.learn.model._transformer_architecture
        _emd_template[
            "PretrainedModel"
        ] = self.learn.model._transformer_pretrained_model_name
        _emd_template["ModelType"] = "Transformer"
        _emd_template["MixedPrecisionTraining"] = self._mixed_precision
        _emd_template["AddressTag"] = self._address_tag
        _emd_template["Labels"] = list(self.learn.model._config.label2id.keys())
        _emd_template["Label2Id"] = self.learn.model._config.label2id
        _emd_template["SequenceLength"] = self._seq_len
        return _emd_template

    def load(self, name_or_path):
        """
        Loads a saved EntityRecognizer model from disk.

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
            name_or_path = os.path.join(self.path, "models", name_or_path)
            name_or_path = str(_get_emd_path(name_or_path))
        return super().load(name_or_path)

    @classmethod
    def _from_pretrained(cls, backbone, label2id, **kwargs):
        if not HAS_FASTAI:
            from .._data import _raise_fastai_import_error

            _raise_fastai_import_error(import_exception=import_exception)

        entities = list(label2id.keys())
        data = TextDataObject(task="ner")
        data.create_empty_object_for_ner(entities, "Address", label2id)
        data._is_empty, data._label2id = True, label2id
        cls_object = cls(data, backbone, **kwargs)
        cls_object._data._is_empty = True
        data.emd, data.emd_path = cls_object._get_emd_params(), None
        return cls_object

    @classmethod
    def from_model(cls, emd_path, data=None):
        if not HAS_FASTAI:
            from .._data import _raise_fastai_import_error

            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = Path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        mixed_precision = emd["MixedPrecisionTraining"]
        entities = set(emd["Labels"])
        label2id = emd["Label2Id"]
        address_tag = emd["AddressTag"]
        pretrained_model = emd["PretrainedModel"]
        seq_len = emd.get("SequenceLength", transformer_seq_length)
        data_is_none = False
        if data is None:
            data_is_none = True
            data = TextDataObject(task="ner")
            data.create_empty_object_for_ner(entities, address_tag, label2id)
            data.emd, data.emd_path = emd, emd_path.parent

        data._is_empty = True
        data._label2id = label2id
        cls_object = cls(
            data,
            pretrained_model,
            pretrained_path=str(emd_path),
            mixed_precision=mixed_precision,
            seq_len=seq_len,
        )
        if data_is_none:
            cls_object._data._is_empty = True
        cls_object.path = getattr(data, "working_dir", data.path)
        return cls_object

    def extract_entities(
        self, text_list, batch_size=4, drop=True, debug=False, show_progress=True
    ):
        results, columns, file_names = [], [], []
        if isinstance(text_list, (str, bytes)):
            path = text_list
            text_list, skipped_docs = [], []
            item_names = os.listdir(path)
            for item_name in item_names:
                try:
                    with open(
                        f"{path}/{item_name}", "r", encoding="utf-16", errors="ignore"
                    ) as f:
                        text_list.append(f.read())
                    file_names.append(item_name)
                except:
                    try:
                        with open(
                            f"{path}/{item_name}",
                            "r",
                            encoding="utf-8",
                            errors="ignore",
                        ) as f:
                            text_list.append(f.read())
                        file_names.append(item_name)
                    except Exception as e:
                        self.logger.exception(e)
                        skipped_docs.append(item_name)
            if len(skipped_docs):
                print(
                    "Unable to read the following documents ", ", ".join(skipped_docs)
                )

        tokenizer, id2label = (
            self.learn.model._tokenizer,
            self.learn.model._config.id2label,
        )
        model_type = self.learn.model._transformer_architecture
        self.logger.info(
            f"Generating Inference using - {model_type} transformer model."
        )
        for i in progress_bar(
            range(0, len(text_list), batch_size), display=show_progress
        ):
            tokens, labels = self.learn.model.generate_inference(
                text_list[i : i + batch_size], self._device
            )
            if debug:
                batch_results = get_results(
                    tokens, labels, tokenizer, id2label, model_type, len(tokens)
                )
                results.extend(batch_results)
            else:
                batch_results, columns = self._process_results(
                    tokens,
                    labels,
                    return_dataframe=False,
                    drop=drop,
                    start_index=i,
                    file_names=file_names[i : i + batch_size],
                )
                results.extend(batch_results)

        if debug:
            return results
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.fillna("", inplace=True)
        return dataframe

    def _process_results(
        self,
        tokens,
        predictions,
        return_dataframe=True,
        drop=False,
        start_index=0,
        file_names=[],
    ):
        data_list = []
        tokenizer = self.learn.model._tokenizer
        id2label = self.learn.model._config.id2label
        model_type = self.learn.model._transformer_architecture
        columns = {x.split("-")[-1] for x in self._data._unique_tags}

        results = get_results(
            tokens, predictions, tokenizer, id2label, model_type, num_items=len(tokens)
        )

        columns.discard("O")
        address_tag, text_tag = self._address_tag, "Text"
        has_address = True if address_tag in self.entities else False

        if has_address:
            cols = [x for x in columns if x not in [text_tag, address_tag]]
        else:
            cols = [x for x in columns if x not in [text_tag]]

        for index, row in enumerate(results):
            text = row[text_tag]
            if has_address:
                values = [", ".join(row.get(column, "")) for column in cols]
                address_list = row.get(address_tag)
                # Don't append the row if drop flag is True and address_list is None
                if drop is True and address_list is None:
                    continue
                if address_list is None:
                    address_list = [""]

                file_name_column = (
                    file_names[index]
                    if len(file_names)
                    else f"Example_{index + start_index}"
                )
                for address in address_list:
                    data_list.append([text, file_name_column, address, *values])
            else:
                file_name_column = (
                    file_names[index]
                    if len(file_names)
                    else f"Example_{index + start_index}"
                )
                values = [", ".join(row.get(column, "")) for column in cols]
                data_list.append([text, file_name_column, *values])

        # data_list = data_list[:self._data._bs]

        df_columns = (
            [text_tag, "Filename", address_tag, *cols]
            if has_address
            else [text_tag, "Filename", *cols]
        )
        if return_dataframe:
            dataframe = pd.DataFrame(data_list, columns=df_columns)
            dataframe.fillna("", inplace=True)
            return dataframe
        else:
            return data_list, df_columns

    def show_results(self, ds_type="valid"):
        """
        Runs entity extraction on a random batch from the mentioned ds_type.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        ds_type                 Optional string, defaults to valid.
        =====================   ===========================================

        :return: Pandas DataFrame
        """
        self._check_requisites()
        databunch = self._data.get_databunch()
        if ds_type.lower() == "valid":
            x, y = random.sample(list(databunch.valid_dl), 1)[0]
        elif ds_type.lower() == "train":
            x, y = random.sample(list(databunch.train_dl), 1)[0]
        else:
            return "Please provide a valid ds_type:['valid'|'train']"
        output = self.learn.model.forward(*x)
        predictions = output[1].argmax(2).tolist()
        tokens = x[0].tolist()
        df = self._process_results(tokens, predictions)
        return df

    def _calculate_model_metrics(self, metric_type="all"):
        metrics = {}
        databunch = self._data.get_databunch()
        (
            predictions,
            labels,
        ) = TransformerForEntityRecognition.get_active_predictions_labels(
            databunch.valid_dl, self.learn
        )

        if metric_type in ["precision_score", "all"]:
            metrics["precision_score"] = round(
                precision_score(
                    labels, predictions, average=self.stats, zero_division=1
                ),
                2,
            )

        if metric_type in ["recall_score", "all"]:
            metrics["recall_score"] = round(
                recall_score(labels, predictions, average=self.stats, zero_division=1),
                2,
            )

        if metric_type in ["f1_score", "all"]:
            metrics["f1_score"] = round(
                f1_score(labels, predictions, average=self.stats, zero_division=1), 2
            )

        if metric_type in ["accuracy_score", "all"]:
            metrics["accuracy"] = round(accuracy_score(labels, predictions), 2)

        return metrics

    def _get_metric(self, metric_type):
        try:
            self._check_requisites()
        except Exception as e:
            metrics = self._data.emd.get("Metrics")
            if metrics:
                return json.loads(metrics).get(metric_type)
            else:
                self.logger.error("Metric not found in the loaded model")
        else:
            if hasattr(self.learn, "recorder"):
                metrics_names = self.learn.recorder.metrics_names
                metrics_values = self.learn.recorder.metrics
                if metric_type in self.learn.recorder.metrics_names:
                    index = metrics_names.index(metric_type)
                    return round(metrics_values[-1][index].item(), 2)
                else:
                    metrics = self._calculate_model_metrics(metric_type)
                    return metrics[metric_type]
            else:
                metrics = self._calculate_model_metrics(metric_type)
                return metrics[metric_type]

    def precision_score(self):
        return self._get_metric("precision_score")

    def recall_score(self):
        return self._get_metric("recall_score")

    def f1_score(self):
        return self._get_metric("f1_score")

    def metrics_per_label(self, show_progress=True):
        try:
            self._check_requisites()
        except Exception as e:
            metrics = self._data.emd.get("Metrics")
            if metrics:
                per_label_metrics = json.loads(metrics).get("metrics_per_label", {})
                return self._create_dataframe_from_dict(per_label_metrics)
            else:
                self.logger.error("Metric not found in the loaded model")
        else:
            databunch = self._data.get_databunch()
            (
                predictions,
                labels,
            ) = TransformerForEntityRecognition.get_active_predictions_labels(
                databunch.valid_dl, self.learn, show_progress=show_progress
            )
            target_names = databunch.train_ds.label2id
            output = classification_report(
                labels,
                predictions,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            )
            return self._create_dataframe_from_dict(output)

    @staticmethod
    def _create_dataframe_from_dict(out_dict):
        out_dict.pop("accuracy", None)
        out_dict.pop("macro avg", None)
        out_dict.pop("weighted avg", None)
        df = pd.DataFrame(out_dict)
        df.drop("support", inplace=True, errors="ignore")
        dataframe = df.T.round(2)
        column_mappings = {
            "precision": "Precision_score",
            "recall": "Recall_score",
            "f1-score": "F1_score",
        }
        dataframe.rename(columns=column_mappings, inplace=True)
        return dataframe

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
