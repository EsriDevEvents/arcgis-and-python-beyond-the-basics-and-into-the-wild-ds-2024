import os
import json
import traceback
from .._data import _raise_fastai_import_error

HAS_TRANSFORMERS = True

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from ._arcgis_transformer import ArcGISTransformer
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMERS = False

    class ArcGISTransformer:
        pass


backbone_models_map = {
    "t5": ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
    + ["See all T5 models at https://huggingface.co/models?filter=t5 "],
    "bart": [
        "facebook/bart-base",
        "facebook/bart-large",
        "facebook/bart-large-mnli",
        "facebook/bart-large-cnn",
        "facebook/bart-large-xsum",
        "facebook/mbart-large-en-ro",
    ]
    + ["See all BART models at https://huggingface.co/models?filter=bart "],
    "marian": [
        "See all Marian models at https://huggingface.co/models?search=Helsinki-NLP "
    ],
}

transformer_architectures = ["T5", "Bart", "Marian"]

backbone_models_reverse_map = {
    x: key for key, val in backbone_models_map.items() for x in val
}

transformer_seq_length = 512


class TransformerForSequenceToSequence(ArcGISTransformer):
    _supported_backbones = transformer_architectures

    def __init__(
        self,
        architecture,
        pretrained_model_name,
        config=None,
        pretrained_model_path=None,
        seq_len=transformer_seq_length,
    ):
        if not HAS_TRANSFORMERS:
            _raise_fastai_import_error(import_exception=import_exception)
        self.architecture = architecture
        self._seq_len = seq_len
        super().__init__(
            architecture,
            pretrained_model_name,
            config,
            pretrained_model_path,
            task="sequence_translation",
        )
        self._seq_len = seq_len
        self._max_seq_len = None
        self.pretrained_model_name = pretrained_model_name

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % type(self).__name__

    def _process_config(self):
        """
        Process the model config to convert the class values to integers in the
        id2label attribute (which is a dictionary) of the model config
        """
        pass
        # self._config.id2label = {int(x): y for x, y in self._config.id2label.items()}

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
                f"PLease choose from {cls._supported_backbones}"
            )

    def save(self, model_path):
        """
        Method to save the fine-tuned model to the disk

        =====================   =================================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------------
        model_path              Required string. The disk location where the
                                fine-tuned model has to be saved
        ---------------------   -------------------------------------------------
        """
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_params = {
            "architecture": self._transformer_architecture,
            "pretrained_model": self._transformer_pretrained_model_name,
        }
        file_path = os.path.join(model_path, self._outfile)
        out_file = open(file_path, "w")
        json.dump(model_params, out_file)
        self._transformer.save_pretrained(model_path)

    @classmethod
    def load_pretrained_model(cls, path):
        """
        Method to load the fine-tuned model which was saved on the disk

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        path                    Required string. The disk location from
                                where the fine-tuned model has to be loaded
        ---------------------   -------------------------------------------
        """
        architecture_file_path = os.path.join(path, cls._outfile)
        if os.path.exists(architecture_file_path):
            with open(architecture_file_path) as f:
                data = json.load(f)
            architecture = data.get("architecture")
            pretrained_model_name = data.get("pretrained_model")
            obj = TransformerForSequenceToSequence(
                architecture, pretrained_model_name, pretrained_model_path=path
            )
            obj.init_model()
            return obj
        else:
            raise Exception(f"{cls._outfile} not present at {path}")

    def forward(self, input_ids, labels, **kwargs):
        """
        Return only the logits from the transfomer model

        =====================   ==============================================
        **Parameter**            **Description**
        ---------------------   ----------------------------------------------
        input_ids               tensor object. tensor containing the token-ids
                                got by calling tokenizer.encode method to the
                                input text. works same for mini-batches
        ---------------------   ----------------------------------------------

        :return: the logits from the transformer model
        """
        if self.architecture in ["t5", "bart", "marian"]:
            pad_id = self._tokenizer.pad_token_id
            attention_mask = (input_ids != pad_id).int()
            output_dict = self._transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            logits = output_dict.get("logits")
            loss = output_dict.get("loss")
            return loss, logits

    def _load_transformer(self):
        """
        Method to load the transformer model. For this class we want to load the
        transformer model for performing Sequence Classification tasks
        """
        if self._pretrained_model_path and os.path.exists(self._pretrained_model_path):
            base_path, extension = os.path.splitext(self._pretrained_model_path)
            path = base_path + ".pth"
            if extension == ".emd" and os.path.exists(path):
                self._transformer = AutoModelForSeq2SeqLM.from_pretrained(
                    path, config=self._config
                )
            else:
                self._transformer = AutoModelForSeq2SeqLM.from_pretrained(
                    self._pretrained_model_path
                )
        else:
            self._transformer = AutoModelForSeq2SeqLM.from_pretrained(
                self._transformer_pretrained_model_name, config=self._config
            )
