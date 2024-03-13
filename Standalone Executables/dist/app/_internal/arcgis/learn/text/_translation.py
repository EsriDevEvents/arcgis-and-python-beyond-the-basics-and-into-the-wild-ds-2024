import traceback
from .._data import _raise_fastai_import_error
from ._inference_only_models import InferenceOnlyModel

HAS_TRANSFORMER = True

try:
    from pathlib import Path
    import json
    import os
    import torch
    from transformers import pipeline, logging
    from fastprogress.fastprogress import progress_bar
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception as e:
    transformer_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMER = False


class TextTranslator(InferenceOnlyModel):
    """
    Creates a :class:`~arcgis.learn.text.TextTranslator` Object.
    Based on the Hugging Face transformers library
    To learn more about the available models for translation task,
    kindly visit:- https://huggingface.co/models?pipeline_tag=translation&search=Helsinki

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    source_language         Optional string. Specify the language of the
                            text you would like to get the translation of.
                            Default value is 'es' (Spanish)
    ---------------------   -------------------------------------------
    target_language         Optional string. The language into which one
                            wishes to translate the input text.
                            Default value is 'en' (English)
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    pretrained_path         Option str. Path to a directory, where pretrained
                            model files are saved.
                            If pretrained_path is provided, the model is
                            loaded from that path on the local disk.
    ---------------------   -------------------------------------------
    working_dir             Option str. Path to a directory on local filesystem.
                            If directory is not present, it will be created.
                            This directory is used as the location to save the
                            model.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.text.TextTranslator` Object
    """

    #: supported transformer architectures
    supported_backbones = ["MarianMT"]

    def __init__(self, source_language="es", target_language="en", **kwargs):
        self._source_lang = source_language
        self._target_lang = target_language
        self._task = f"translation_{self._source_lang}_to_{self._target_lang}"
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)
        super().__init__(task=self._task, **kwargs)

    def _load_model(self):
        if self._pretrained_path:
            pretrained_path = r"{}".format(self._pretrained_path)
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                f"Helsinki-NLP/opus-mt-{self._source_lang}-{self._target_lang}"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                f"Helsinki-NLP/opus-mt-{self._source_lang}-{self._target_lang}"
            )

        device = torch.device(
            "cpu" if self._device < 0 else "cuda:{}".format(self._device)
        )
        self.model.to(device)

    def translate(self, text_or_list, show_progress=True, **kwargs):
        """
        Translate the given text or list of text into the target language

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. A text/passage
                                or a list of texts/passages to translate.
        ---------------------   -------------------------------------------
        show_progress           optional Bool. If set to True, will display a
                                progress bar depicting the items processed so far.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        min_length              Optional integer. The minimum length of the
                                sequence to be generated.
                                Default value is set to to `min_length` parameter
                                of the model config.
        ---------------------   -------------------------------------------
        max_length              Optional integer. The maximum length of the
                                sequence to be generated.
                                Default value is set to to `max_length` parameter
                                of the model config.
        ---------------------   -------------------------------------------
        num_return_sequences    Optional integer. The number of independently
                                computed returned sequences for each element
                                in the batch.
                                Default value is set to 1.
        ---------------------   -------------------------------------------
        num_beams               Optional integer. Number of beams for beam
                                search. 1 means no beam search.
                                Default value is set to 1.
        ---------------------   -------------------------------------------
        length_penalty          Optional float. Exponential penalty to the
                                length. 1.0 means no penalty. Set to values < 1.0
                                in order to encourage the model to generate
                                shorter sequences, to a value > 1.0 in order to
                                encourage the model to produce longer sequences.
                                Default value is set to 1.0.
        ---------------------   -------------------------------------------
        early_stopping          Optional bool. Whether to stop the beam search
                                when at least ``num_beams`` sentences are
                                finished per batch or not.
                                Default value is set to False.
        =====================   ===========================================

        :return: a list or a list of list containing the translation of the input prompt(s) / sentence(s) to the target language
        """
        results = []
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        min_length, max_length = kwargs.get("min_length"), kwargs.get("max_length")
        if min_length and max_length and min_length > max_length:
            error_message = (
                f"Value of `min_length` parameter({min_length}) cannot be "
                f"greater than the value of `max_length` parameter({max_length})."
            )
            raise Exception(error_message)

        if not isinstance(text_or_list, (list, tuple)):
            text_or_list = [text_or_list]

        for i in progress_bar(range(len(text_or_list)), display=show_progress):
            inputs = self._tokenizer.encode(
                f"{text_or_list[i]} {self._tokenizer.eos_token}", return_tensors="pt"
            ).to(self._device)
            outputs = self.model.generate(inputs, **kwargs)
            result = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            result = [{"translated_text": x} for x in result]
            if num_return_sequences == 1:
                result = result[0]
            results.append(result)
        return results

    def _create_emd(self, name_or_path):
        emd_template = {}
        emd_template.update({"ModelName": self.__class__.__name__})
        emd_template.update({"architectures": self.model.config.architectures})
        emd_template.update({"source_lang": self._source_lang})
        emd_template.update({"target_lang": self._target_lang})
        path = Path(name_or_path)
        name = path.parts[-1]
        with open(os.path.join(path, f"{name}.emd"), "w") as f:
            f.write(json.dumps(emd_template))

    def save(self, name_or_path):
        """
        Saves the translator model files on a specified path on the local disk.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Path to save
                                model files on the local disk.
        =====================   ===========================================

        :return: Absolute path for the saved model
        """
        if "\\" in name_or_path or "/" in name_or_path:
            path = name_or_path
        else:
            path = os.path.join(self.working_dir, "models", name_or_path)
        self.model.save_pretrained(path)
        self.model.config.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        self._create_emd(path)
        return Path(path).absolute()
