import traceback
import warnings
from .._data import _raise_fastai_import_error
from ._inference_only_models import InferenceOnlyModel
from functools import partial


HAS_TRANSFORMER = True

try:
    import torch
    from transformers import pipeline, logging
    from fastprogress.fastprogress import progress_bar

    try:
        # For version 3.3.0
        from transformers.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
    except ModuleNotFoundError as e:
        # For version 4.5.1
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        )
    EXPECTED_MODEL_TYPES = [
        x.__name__.replace("Config", "")
        for x in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()
    ]
except Exception as e:
    transformer_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMER = False
    EXPECTED_MODEL_TYPES = []


class QuestionAnswering(InferenceOnlyModel):
    """
    Creates a :class:`~arcgis.learn.text.QuestionAnswering` Object.
    Based on the Hugging Face transformers library

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    backbone                Optional string. Specify the HuggingFace
                            transformer model name which will be used to
                            extract the answers from a given passage/context.

                            To learn more about the available models for
                            question-answering task, kindly visit:-
                            https://huggingface.co/models?pipeline_tag=question-answering
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

    :return: :class:`~arcgis.learn.text.QuestionAnswering` Object
    """

    #: supported transformer architectures
    supported_backbones = EXPECTED_MODEL_TYPES

    def __init__(self, backbone=None, **kwargs):
        if not HAS_TRANSFORMER:
            _raise_fastai_import_error(import_exception=transformer_exception)
        super().__init__(backbone=backbone, task="question-answering", **kwargs)

    def get_answer(
        self,
        text_or_list,
        context,
        show_progress=True,
        explain=False,
        explain_start_word=True,
        explain_index=None,
        **kwargs,
    ):
        """
        Find answers for the asked questions from the given passage/context

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or list. Questions or a list
                                of questions one wishes to seek an answer for.
        ---------------------   -------------------------------------------
        context                 Required string. The context associated with
                                the question(s) which contains the answers.
        ---------------------   -------------------------------------------
        show_progress           optional Bool. If set to True, will display a
                                progress bar depicting the items processed so far.
        ---------------------   -------------------------------------------
        explain                 optional Bool. If set to True, will generate
                                a shap based explanation
        ---------------------   -------------------------------------------
        explain_start_word      optional Bool.
                                E.g. Context: Point cloud datasets are typically
                                collected using Lidar sensors (
                                light detection and ranging )
                                Question: "How is Point cloud dataset collected?"
                                Answer: Lidar Sensors

                                If set to True, will generate
                                a shap based explanation for start word. if set
                                to False, will generate explanation for last word
                                of the answer.

                                In the above example, if the value of `explain_start_word`
                                is `True`, it will generate the importance of different context
                                words that leads to selection of "Lidar" as a starting word
                                of the span. If `explain_start_word` is set to `False`
                                then it will generate explanation for the word `sensors`
        ---------------------   -------------------------------------------
        explain_index           optional List. Index of the question for which answer
                                needs to be generated
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        num_answers             Optional integer. The number of answers to
                                return. The answers will be chosen by order
                                of likelihood.
                                Default value is set to 1.
        ---------------------   -------------------------------------------
        max_answer_length       Optional integer. The maximum length of the
                                predicted answers.
                                Default value is set to 15.
        ---------------------   -------------------------------------------
        max_question_length     Optional integer. The maximum length of the
                                question after tokenization. Questions will be
                                truncated if needed.
                                Default value is set to 64.
        ---------------------   -------------------------------------------
        impossible_answer       Optional bool. Whether or not we accept impossible
                                as an answer.
                                Default value is set to False
        =====================   ===========================================

        :return: a list or a list of list containing the answer(s) for the input question(s)
        """
        results, kwargs_dict = [], {}

        kwargs_dict["topk"] = kwargs.get("num_answers", 1)
        kwargs_dict["max_answer_len"] = kwargs.get("max_answer_length", 15)
        kwargs_dict["max_question_len"] = kwargs.get("max_question_length", 64)
        kwargs_dict["handle_impossible_answer"] = kwargs.get("impossible_answer", False)

        if not isinstance(text_or_list, (list, tuple)):
            text_or_list = [text_or_list]

        for i in progress_bar(range(len(text_or_list)), display=show_progress):
            results.append(
                self.model(question=text_or_list[i], context=context, **kwargs_dict)
            )
        try:
            if explain:
                temp_text_or_list = []
                if explain_index is not None:
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
                                    temp_text_or_list.append(text_or_list[i])
                                else:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("always", UserWarning)
                                        warnings.warn(
                                            f"Value of index {i} should be less than/equal to {len(text_or_list) - 1}."
                                        )
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter("always", UserWarning)
                                warnings.warn(
                                    f"No valid indices were supplied. Please change your input to list of integers"
                                )

                    elif isinstance(explain_index, int):
                        if explain_index < len(text_or_list):
                            temp_text_or_list = [text_or_list[explain_index]]
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter("always", UserWarning)
                                warnings.warn(
                                    f"Value of index {explain_index} should be less than/equal to {len(text_or_list) - 1}."
                                )
                else:
                    temp_text_or_list = text_or_list

                if len(temp_text_or_list) > 0:
                    if explain_start_word:
                        self._explain(temp_text_or_list, context, True)
                    else:
                        self._explain(temp_text_or_list, context, False)
        except:
            with warnings.catch_warnings():
                warnings.simplefilter("always", UserWarning)
                warnings.warn(
                    f"SHAP workflow has encountered an error. Failed to generate an explanation."
                )

        return self._process_result(results, text_or_list)

    @staticmethod
    def _process_result(result_list, question_list):
        processed_results = []
        for result, question in zip(result_list, question_list):
            if isinstance(result, dict):
                tmp_dict = {
                    "question": question,
                    "answer": result["answer"],
                    "score": result["score"],
                }
                processed_results.append(tmp_dict)
            elif isinstance(result, list):
                item_list = [
                    {
                        "question": question,
                        "answer": item["answer"],
                        "score": item["score"],
                    }
                    for item in result
                ]
                processed_results.append(item_list)

        return processed_results

    def _logit_wrapper(self, part_start, questions):
        outs = []
        for q in questions:
            question, context = q.split("[SEP]")
            d = self.model.tokenizer(question, context, truncation="only_second")
            out = self.model.model.forward(
                **{k: torch.tensor(d[k]).reshape(1, -1).to(self._device) for k in d}
            )
            logits = out.start_logits if part_start else out.end_logits
            outs.append(logits.reshape(-1).detach().cpu().numpy())
        return outs

    def _output_token_decode(self, inputs):
        question, context = inputs.split("[SEP]")
        d = self.model.tokenizer(question, context, truncation="only_second")
        return [self.model.tokenizer.decode([id]) for id in d["input_ids"]]

    def _explain(self, questions, context, explain_start_token=True):
        IS_SHAP = True
        try:
            import shap
        except:
            IS_SHAP = False

        if IS_SHAP:
            logit_start = partial(self._logit_wrapper, True)
            logit_end = partial(self._logit_wrapper, False)
            notify = []
            for i in questions:
                try:
                    val = i + "[SEP]" + context
                    if explain_start_token:
                        explainer_start = shap.Explainer(
                            logit_start,
                            self.model.tokenizer,
                            output_names=self._output_token_decode(val),
                        )
                    else:
                        explainer_start = shap.Explainer(
                            logit_end,
                            self.model.tokenizer,
                            output_names=self._output_token_decode(
                                i + "[SEP]" + context
                            ),
                        )
                    shap_values_start = explainer_start([val])
                    shap.plots.text(shap_values_start)
                except ValueError as err:
                    notify.append(
                        f"SHAP based explanation failed for question {i} due to internal error."
                    )

            if len(notify):
                for i in notify:
                    print(i)
        else:
            warnings.warn("SHAP is not installed.")
