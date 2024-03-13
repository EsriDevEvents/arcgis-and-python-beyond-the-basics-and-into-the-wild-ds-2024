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
        from transformers.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING
    except ModuleNotFoundError as e:
        # For version 4.5.1
        from transformers.models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING
    EXPECTED_MODEL_TYPES = [
        x.__name__.replace("Config", "") for x in MODEL_FOR_MASKED_LM_MAPPING.keys()
    ]
except Exception as e:
    transformer_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMER = False
    EXPECTED_MODEL_TYPES = []


class FillMask(InferenceOnlyModel):
    """
    Creates a :class:`~arcgis.learn.text.FillMask` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    backbone                Optional string. Specify the HuggingFace
                            transformer model name which will be used to
                            generate the suggestion token.

                            To learn more about the available models for
                            fill-mask task, kindly visit:-
                            https://huggingface.co/models?pipeline_tag=fill-mask
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

    :return: :class:`~arcgis.learn.text.FillMask` Object
    """

    #: supported transformer architectures
    supported_backbones = EXPECTED_MODEL_TYPES

    def __init__(self, backbone=None, **kwargs):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)
        super().__init__(backbone=backbone, task="fill-mask", **kwargs)

    def _load_model(self):
        try:
            # For version 3.3.0
            if self._pretrained_path:
                self.model = pipeline(
                    self._task,
                    model=self._pretrained_path,
                    device=self._device,
                    topk=10,
                )
            else:
                self.model = pipeline(
                    self._task, model=self._backbone, device=self._device, topk=10
                )
        except TypeError as e:
            # For version 4.5.1
            if self._pretrained_path:
                self.model = pipeline(
                    self._task,
                    model=self._pretrained_path,
                    device=self._device,
                    top_k=10,
                )
            else:
                self.model = pipeline(
                    self._task, model=self._backbone, device=self._device, top_k=10
                )
        except Exception as e:
            error_message = (
                f"`{self._backbone}` is not valid backbone name for {self._task} task.\n"
                f"For selecting backbone name for {self._task} task, kindly visit:- "
                f"https://huggingface.co/models?pipeline_tag={self._task} "
            )
            raise Exception(error_message)

    def predict_token(self, text_or_list, num_suggestions=5, show_progress=True):
        """
        Summarize the given text or list of text

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. A text/sentence
                                or a list of texts/sentences for which on wishes
                                to generate the recommendations for masked-token.
        ---------------------   -------------------------------------------
        num_suggestions         Optional Integer. The number of suggestions to
                                return. The maximum number of suggestion that
                                can be generated for a `missing-token` is 10.
        ---------------------   -------------------------------------------
        show_progress           optional Bool. If set to True, will display a
                                progress bar depicting the items processed so far.
        =====================   ===========================================

        :return: A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (:obj:`str`)  -- The corresponding input with the mask token prediction.
            - **score** (:obj:`float`)   -- The corresponding probability.
            - **token_str** (:obj:`str`) -- The predicted token (to replace the masked one).
        """
        results = []
        if not isinstance(text_or_list, (list, tuple)):
            text_or_list = [text_or_list]
        self._do_sanity(text_or_list)

        for i in progress_bar(range(len(text_or_list)), display=show_progress):
            text = text_or_list[i].replace("__", self.model.tokenizer.mask_token)
            result = self._process_result(self.model(text)[:num_suggestions])
            if num_suggestions == 1:
                result = result[0]
            results.append(result)
        return results

    @staticmethod
    def _do_sanity(text_list):
        for text in text_list:
            if "__" not in text:
                error_message = (
                    f"Text - `{text}` is mising `__` token. Use the `__` token to "
                    "specify where you want to generate the suggestions in the text."
                )
                raise Exception(error_message)

    def _process_result(self, result_list):
        for item in result_list:
            _ = item.pop("token", -1)
            token = item["token_str"]
            if isinstance(token, (str, bytes)):
                token = [token]
            sequence = item["sequence"]
            item["token_str"] = self.model.tokenizer.convert_tokens_to_string(
                token
            ).strip()
            item["sequence"] = self.model.tokenizer.decode(
                self.model.tokenizer.encode(sequence, add_special_tokens=False),
                skip_special_tokens=True,
            )

        return result_list
