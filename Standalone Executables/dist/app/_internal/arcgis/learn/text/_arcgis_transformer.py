import os
import abc
import traceback
from ._transformers_layer_group import split_into_layer_groups


HAS_TRANSFORMER = True

try:
    import torch.nn as nn
    from transformers import AutoConfig, AutoTokenizer
except Exception as e:
    transformer_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMER = False

transformer_seq_length = 512


def infer_model_type(model_name, transformer_architectures):
    model_type = "Others"
    model_name = model_name.split("/")[-1]
    for architecture in sorted(transformer_architectures, key=len, reverse=True):
        if model_name.startswith(architecture.lower()) or model_name.endswith(
            architecture.lower()
        ):
            model_type = architecture.lower()
            break
        elif model_name.startswith("opus-mt"):
            model_type = "marian"
            break
    return model_type


class ModelBackbone:
    def __init__(self, name):
        self._name = name

    @property
    def __name__(self):
        return f"{ModelBackbone.__name__}: {self._name}"


class ArcGISTransformer(nn.Module, metaclass=abc.ABCMeta):
    _outfile = "model_architecture.json"

    def __init__(
        self,
        architecture,
        pretrained_model_name,
        config=None,
        pretrained_model_path=None,
        task=None,
    ):
        super(ArcGISTransformer, self).__init__()
        self._transformer_architecture = architecture
        self._transformer_pretrained_model_name = pretrained_model_name
        self._pretrained_model_path = pretrained_model_path
        self._config = config
        self._task = task
        self._tokenizer = None
        self._transformer = None
        self._layer_groups = []

    def _load_config(self):
        """
        Load the model config from the specified path if not passed
        in the constructor and call the post processing function on it.
        """
        if self._config is None:
            if self._pretrained_model_path is None:
                raise Exception(
                    "Error, either provide model config while initializing the object or "
                    "provide pretrained-model-path containing config.json"
                )
            else:
                config_file_path = os.path.join(
                    self._pretrained_model_path, "config.json"
                )
                if os.path.exists(config_file_path):
                    self._config = AutoConfig.from_pretrained(config_file_path)
                else:
                    raise Exception(
                        f"Error, config.json not present at {config_file_path}"
                    )
        self._process_config()

    def _load_layer_groups(self):
        """
        Split the Transformer model into layer groups. Appropriate method
        will be called to split the model into different layers depending
        on the model architecture and the task selected
        """
        self._layer_groups = split_into_layer_groups(
            self._transformer, self._transformer_architecture, self._task
        )

    def get_layer_groups(self):
        """
        Returns the layer groups of the transformer architecture.
        :return: A list of list containing the layer group information of the model
        """
        return self._layer_groups

    def _load_tokenizer(self):
        """
        Loads the appropriate tokenizer for tokenizing the text
        depending on the transformer model-name parameter
        """
        # FastTokenizer not working for EntityRecognizer in transformers library version 4.5.1
        use_fast = False if self._task == "ner" else True
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._transformer_pretrained_model_name,
            config=self._config,
            use_fast=use_fast,
        )
        self._set_max_seq_length()

    def init_model(self):
        """
        A wrapper function to load necessary attributes like model-config,
        model-tokeniser, transformer-model and layer-groups
        """
        self._load_config()
        self._load_tokenizer()
        self._load_transformer()
        self._load_layer_groups()

    @abc.abstractmethod
    def _process_config(self):
        """
        Post processing method applied to the model config. Class inheriting this
        class will apply their specific post processing on model config
        """
        pass

    @abc.abstractmethod
    def _set_max_seq_length(self):
        """
        Set the max sequence length parameter. This is useful in cases where a transformer
        transformer backbone doesn't have a max sequence length parameter for example XLNet,
        Longformer, Bart etc. This can leads to out of memory issue while training on GPUs
        """
        pass

    @abc.abstractmethod
    def _load_transformer(self):
        """
        Method to load the appropriate transformer model. Class inheriting this
        class will load specific model architectures to solve specific nlp tasks
        """
        pass

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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def forward(self, input_ids, **kwargs):
        pass
