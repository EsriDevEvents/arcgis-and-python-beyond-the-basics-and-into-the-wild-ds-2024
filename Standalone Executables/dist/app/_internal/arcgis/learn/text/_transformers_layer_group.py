import logging
import traceback
from functools import partial


HAS_FASTAI = True

try:
    import torch.nn as nn
    from fastai.torch_core import flatten_model
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

logger = logging.getLogger()


def split_into_layer_groups(model, architecture, task="classification"):
    """
    Method responsible for getting the correct layer
    group splitter function and calling it to split the
    transformer model into different layer groups

    =====================   =================================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------------
    model                   Required ModelObject. The transformer model for
                            which we want to get the layer groups
    ---------------------   -------------------------------------------------
    ---------------------   -------------------------------------------------
    architecture            Required string. The transformer architecture for
                            which we wish to get the layer groups. This param
                            will be used to call the correct function to split
                            model layers
    ---------------------   -------------------------------------------------
    return: A list containing model layer groups
    """
    if task == "classification":
        splitter = get_layer_group_splitter_for_classification(architecture)
    elif task == "ner":
        splitter = get_layer_group_splitter_for_ner(architecture)
    elif task == "sequence_translation":
        splitter = get_layer_group_splitter_for_sequence_translation(architecture)
    else:
        raise Exception(
            f"Wrong task - {task} selected. Allowed values are 'ner', 'classification','sequence_translation'"
        )

    # logger.info(f"Invoking - {splitter.__name__} function for splitting {architecture} model into layer groups")
    return splitter(model, architecture)


def get_layer_group_splitter_for_ner(architecture):
    """
    This function will return the appropriate function which will
    then be used to split the transformer model into layer groups

    =====================   =================================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------------
    architecture            Required string. The transformer architecture for
                            which we wish to get the layer groups. This param
                            will be used to return the correct function to
                            split model layers
    ---------------------   -------------------------------------------------
    """
    if architecture in ["bert", "roberta", "distilbert", "xlm-roberta", "electra"]:
        return _bert_layer_splitter_for_ner
    elif architecture == "albert":
        return _albert_layer_splitter
    elif architecture == "xlnet":
        return _xlnet_layer_splitter_for_ner
    elif architecture in ["xlm"]:
        _xlm_layer_splitter_for_ner = partial(_xlm_layer_splitter, task="ner")
        _xlm_layer_splitter_for_ner.__name__ = "xlm_layer_splitter_for_ner"
        return _xlm_layer_splitter_for_ner
    else:
        return naive_model_splitter


def get_layer_group_splitter_for_classification(architecture):
    """
    This function will return the appropriate function which will
    then be used to split the transformer model into layer groups

    =====================   =================================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------------
    architecture            Required string. The transformer architecture for
                            which we wish to get the layer groups. This param
                            will be used to return the correct function to
                            split model layers
    ---------------------   -------------------------------------------------
    """
    if architecture in ["bert", "roberta", "distilbert", "xlm-roberta", "electra"]:
        return _bert_layer_splitter_for_classification
    elif architecture == "albert":
        return _albert_layer_splitter
    elif architecture == "xlnet":
        return _xlnet_layer_splitter_for_classification
    elif architecture in ["xlm", "flaubert"]:
        return _xlm_layer_splitter
    else:
        return naive_model_splitter


def get_layer_group_splitter_for_sequence_translation(architecture):
    """
    This function will return the appropriate function which will
    then be used to split the transformer model into layer groups

    =====================   =================================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------------
    architecture            Required string. The transformer architecture for
                            which we wish to get the layer groups. This param
                            will be used to return the correct function to
                            split model layers
    ---------------------   -------------------------------------------------
    """
    if architecture in ["t5"]:
        return _t5_conditional_generation_splitter
    elif architecture in ["bart", "mbart", "marian"]:
        return _bart_conditional_generation_splitter
    else:
        return naive_model_splitter


def naive_model_splitter(model, model_name):
    if not HAS_FASTAI:
        from .._data import _raise_fastai_import_error

        _raise_fastai_import_error(import_exception=import_exception)
    linear_layer_found = False
    index, last_index = -1, -1
    layers = flatten_model(model)
    for idx, layer in reversed(list(enumerate(layers))):
        if isinstance(layer, nn.modules.linear.Linear):
            if linear_layer_found is False:
                last_index = idx
                linear_layer_found = True
                continue
            else:
                current_index = idx
                index = last_index if last_index - current_index > 4 else current_index
                break

    return [nn.Sequential(*layers[:index]), nn.Sequential(*layers[index:])]


def split_into_chunks(arr, chunk_size=3):
    # split array into 3 groups
    """
    Function to split the list into equal sized chunks
    """
    # n = round(len(arr)/ 3)
    # return [arr[:n], arr[n: 2*n], arr[2*n:]]
    return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]


def _bert_layer_splitter_for_ner(model, model_name):
    """
    Split BERT, RoBERTa, DistilBERT and XLM-RoBERTa Models into layer groups
    """
    # hack to handle `xlm-roberta` string name in model_name
    model_name = model_name.split("-")[-1]
    if hasattr(model, model_name):
        model_obj = getattr(model, model_name)
        embedder = model_obj.embeddings
        pooler = None if model_name in ["distilbert", "electra"] else model_obj.pooler
        layers = (
            model_obj.transformer.layer
            if model_name == "distilbert"
            else model_obj.encoder.layer
        )

        chunk_size = 3 if model_name == "distilbert" else 4
        chunks = split_into_chunks(layers, chunk_size=chunk_size)

        classifier = [model.dropout, model.classifier]

        if model_name in ["distilbert", "electra"]:
            groups = [[embedder], *chunks, classifier]
        else:
            groups = [[embedder], *chunks, [pooler] + classifier]
        return groups
    else:
        raise Exception("Error in splitting the model into layer groups")


def _bert_layer_splitter_for_classification(model, model_name):
    """
    Split BERT, RoBERTa, DistilBERT and XLM-RoBERTa Models into layer groups
    """
    # hack to handle `xlm-roberta` string name in model_name
    model_name = model_name.split("-")[-1]
    if hasattr(model, model_name):
        model_obj = getattr(model, model_name)
        embedder = model_obj.embeddings
        pooler = None if model_name in ["distilbert", "electra"] else model_obj.pooler
        layers = (
            model_obj.transformer.layer
            if model_name == "distilbert"
            else model_obj.encoder.layer
        )

        chunk_size = 4
        if model_name in ["bert"]:
            classifier = [model.dropout, model.classifier]
        elif model_name in ["roberta", "xlm-roberta", "electra"]:
            classifier = [model.classifier]
        else:
            chunk_size = 3
            classifier = [model.pre_classifier, model.classifier, model.dropout]

        chunks = split_into_chunks(layers, chunk_size=chunk_size)

        if model_name in ["distilbert", "electra"]:
            groups = [[embedder], *chunks, classifier]
        else:
            groups = [[embedder], *chunks, [pooler] + classifier]
        return groups
    else:
        raise Exception("Error in splitting the model into layer groups")


def _albert_layer_splitter(model, model_name):
    """
    Split Hugging Face ALBERT Model into layer groups
    """
    if hasattr(model, model_name):
        model_obj = getattr(model, model_name)
        embedder = model_obj.embeddings
        encoder = model_obj.encoder
        pooler = [model_obj.pooler, model_obj.pooler_activation]
        classifier = [model.dropout, model.classifier]
        groups = [[embedder], [encoder], pooler + classifier]
        return groups
    else:
        raise Exception("Error in splitting the model into layer groups")


def _xlnet_layer_splitter_for_ner(model, model_name):
    """
    Split Hugging Face XLNet Model into layer groups
    """
    if hasattr(model, "transformer"):
        model_obj = getattr(model, "transformer")
        embedder = model_obj.word_embedding
        layers = model_obj.layer
        # chunks = [layers[i: i+grouping_param] for i in range(0, len(layers), grouping_param)]
        chunks = split_into_chunks(layers, chunk_size=4)
        classifier = [model_obj.dropout, model.classifier]
        groups = [[embedder], *chunks, classifier]
        return groups
    else:
        raise Exception("Error in splitting the model into layer groups")


def _xlnet_layer_splitter_for_classification(model, model_name):
    """
    Split Hugging Face XLNet Model into layer groups
    """
    if hasattr(model, "transformer"):
        model_obj = getattr(model, "transformer")
        embedder = model_obj.word_embedding
        layers = model_obj.layer
        # chunks = [layers[i: i+grouping_param] for i in range(0, len(layers), grouping_param)]
        chunks = split_into_chunks(layers, chunk_size=4)
        seq_summary = [model.sequence_summary]
        classifier = [model.logits_proj]
        groups = [[embedder], *chunks, seq_summary + classifier]
        return groups
    else:
        raise Exception("Error in splitting the model into layer groups")


def _xlm_layer_splitter(model, model_name, task="classification"):
    """
    Split Hugging Face XLM Model into layer groups
    """
    if hasattr(model, "transformer"):
        model_obj = getattr(model, "transformer")
        embedder = [
            model_obj.position_embeddings,
            model_obj.embeddings,
            model_obj.layer_norm_emb,
        ]

        attentions = model_obj.attentions
        attention_chunks = split_into_chunks(attentions, chunk_size=6)
        # attention_chunks = [attentions[i: i+grouping_param] for i in range(0, len(attentions), grouping_param)]

        layer_norm1 = model_obj.layer_norm1
        layer_norm1_chunks = split_into_chunks(layer_norm1, chunk_size=6)
        # layer_norm1_chunks = [layer_norm1[i: i+grouping_param] for i in range(0, len(layer_norm1), grouping_param)]

        ffns = model_obj.ffns
        ffns_chunks = split_into_chunks(ffns, chunk_size=6)
        # ffns_chunks = [ffns[i: i+grouping_param] for i in range(0, len(ffns), grouping_param)]

        layer_norm2 = model_obj.layer_norm2
        layer_norm2_chunks = split_into_chunks(layer_norm2, chunk_size=6)
        # layer_norm2_chunks = [layer_norm2[i: i+grouping_param] for i in range(0, len(layer_norm2), grouping_param)]

        if task == "classification":
            classifier = [model.sequence_summary]
        elif task == "ner":
            classifier = [model.dropout, model.classifier]
        else:
            raise Exception(
                f"Wrong task - {task} selected. Allowed values are 'ner', 'classification'"
            )
        groups = [
            embedder,
            *attention_chunks,
            *layer_norm1_chunks,
            *ffns_chunks,
            *layer_norm2_chunks,
            classifier,
        ]
        return groups
    else:
        raise Exception("Error in splitting the model into layer groups")


def _t5_conditional_generation_splitter(model, model_name):
    """
    Split Hugging Face t5 Model into layer groups
    """
    try:
        t = model
        groups = [[t.encoder.block, t.decoder.block], [t.lm_head]]
        return groups
    except:
        raise Exception("Error in splitting the model into layer groups")


def _bart_conditional_generation_splitter(model, model_name):
    """
    Split Hugging Face t5 Model into layer groups
    """
    try:
        t = model.model
        groups = [
            [
                t.encoder.embed_positions,
                t.encoder.layers,
                t.decoder.embed_positions,
                t.decoder.layers,
            ],
            [t.shared],
        ]
        return groups
    except:
        raise Exception("Error in splitting the model into layer groups")
