import traceback
from .._data import _raise_fastai_import_error
from ._inference_only_models import InferenceOnlyModel

HAS_TRANSFORMER = True

try:
    import torch
    from transformers import pipeline, logging
    from fastprogress.fastprogress import progress_bar

    try:
        # For version 3.3.0
        from transformers.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    except ModuleNotFoundError as e:
        # For version 4.5.1
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        )
    EXPECTED_MODEL_TYPES = [
        x.__name__.replace("Config", "")
        for x in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys()
    ]
except Exception as e:
    transformer_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMER = False
    EXPECTED_MODEL_TYPES = []


class TextSummarizer(InferenceOnlyModel):
    """
    Creates a :class:`~arcgis.learn.text.TextSummarizer` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    backbone                Optional string. Specify the HuggingFace
                            transformer model name which will be used to
                            summarize the text.

                            To learn more about the available models for
                            summarization task, kindly visit:-
                            https://huggingface.co/models?pipeline_tag=summarization
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

    :return: :class:`~arcgis.learn.text.TextSummarizer` Object
    """

    #: supported transformer architectures
    supported_backbones = EXPECTED_MODEL_TYPES

    def __init__(self, backbone=None, **kwargs):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)
        super().__init__(backbone=backbone, task="summarization", **kwargs)

    def summarize(self, text_or_list, show_progress=True, **kwargs):
        """
        Summarize the given text or list of text

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. A text/passage
                                or a list of texts/passages to generate the
                                summary for.
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

        :return: a list or a list of list containing the summary/summaries for the input prompt(s) / sentence(s)
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
            result = self.model(text_or_list[i], **kwargs)
            if num_return_sequences == 1:
                result = result[0]
            results.append(result)
        return results
