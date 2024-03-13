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
        from transformers.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    except ModuleNotFoundError as e:
        # For version 4.5.1
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        )
    EXPECTED_MODEL_TYPES = [
        x.__name__.replace("Config", "")
        for x in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()
    ]
except Exception as e:
    transformer_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMER = False
    EXPECTED_MODEL_TYPES = []


class ZeroShotClassifier(InferenceOnlyModel):
    """
    Creates a :class:`~arcgis.learn.text.ZeroShotClassifier` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    backbone                Optional string. Specifying the HuggingFace
                            transformer model name which will be used to
                            predict the answers from a given passage/context.

                            To learn more about the available models for
                            zero-shot-classification task, kindly visit:-
                            https://huggingface.co/models?pipeline_tag=zero-shot-classification
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

    :return: :class:`~arcgis.learn.text.ZeroShotClassifier` Object
    """

    #: supported transformer architectures
    supported_backbones = EXPECTED_MODEL_TYPES

    def __init__(self, backbone=None, **kwargs):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)
        super().__init__(backbone=backbone, task="zero-shot-classification", **kwargs)

    def predict(self, text_or_list, candidate_labels, show_progress=True, **kwargs):
        """
        Predicts the class label(s) for the input text

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. The sequence or a
                                list of sequences to classify.
        ---------------------   -------------------------------------------
        candidate_labels        Required string or list. The set of possible
                                class labels to classify each sequence into.
                                Can be a single label, a string of
                                comma-separated labels, or a list of labels.
        ---------------------   -------------------------------------------
        show_progress           optional Bool. If set to True, will display a
                                progress bar depicting the items processed so far.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        multi_class             Optional boolean. Whether or not multiple
                                candidate labels can be true.
                                Default value is set to False.
        ---------------------   -------------------------------------------
        hypothesis              Optional string. The template used to turn each
                                label into an NLI-style hypothesis. This template
                                must include a {} or similar syntax for the
                                candidate label to be inserted into the template.
                                Default value is set to `"This example is {}."`.
        =====================   ===========================================

        :return: a list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **sequence** (:obj:`str`) -- The sequence for which this is the output.
            - **labels** (:obj:`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (:obj:`List[float]`) -- The probabilities for each of the labels.
        """
        results = []
        multi_class = kwargs.get("multi_class", False)
        hypothesis = kwargs.get("hypothesis", "This example is {}.")
        if not isinstance(text_or_list, (list, tuple)):
            text_or_list = [text_or_list]

        for i in progress_bar(range(len(text_or_list)), display=show_progress):
            result = self.model(
                text_or_list[i],
                candidate_labels,
                multi_class=multi_class,
                hypothesis_template=hypothesis,
            )
            results.append(result)
        return results
