import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import re
import traceback
import collections
from functools import partial

HAS_TRANSFORMER = True

try:
    import torch
    import pandas as pd
    from fastai.data_block import DataBunch
    from fastai.basic_data import DatasetType
    from torch.utils.data import Dataset, DataLoader
    from transformers import PreTrainedTokenizer, AutoTokenizer
    from fastai.text import (
        List,
        BaseTokenizer,
        Vocab,
        Collection,
        SortishSampler,
        SortSampler,
    )
    from fastai.torch_core import to_data, BatchSamples, Tuple, LongTensor
except Exception as e:
    transformer_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMER = False

    class BaseTokenizer:
        pass

    class Vocab:
        pass


def process_text(text):
    text = re.sub(" +", " ", text)
    text = re.sub(" ([@.#$/:-]) ?", r"\1", text)
    # Reomve leading and trailing special characters
    while text and text[0] in [".", ",", "-", ":", "@", "$"]:
        text = text[1:]
    while text and text[-1] in [".", ",", "-", ":", "@", "$"]:
        text = text[:-1]
    return text.strip()


models_subwords_map = {
    "distilbert": ("##", False),
    "bert": ("##", False),
    "electra": ("##", False),
    "xlnet": ("▁", True),
    "camembert": ("▁", True),
    "albert": ("▁", True),
    "xlm-roberta": ("▁", True),
    "roberta": ("Ġ", True),
    "longformer": ("Ġ", True),
    "bart": ("Ġ", True),
    "mobilebert": ("##", False),
    "funnel": ("##", False),
}


class TransformersBaseTokenizer(BaseTokenizer):
    """
    Wrapper around PreTrainedTokenizer to be compatible with fast.ai
    """

    def __init__(
        self,
        pretrained_tokenizer: PreTrainedTokenizer,
        lang="en",
        seq_len=512,
        **kwargs,
    ):
        super().__init__(lang)
        self._pretrained_tokenizer = pretrained_tokenizer
        self._seq_len = seq_len
        self.max_seq_len = min(pretrained_tokenizer.model_max_length, self._seq_len)

    def __call__(self, *args, **kwargs):
        return self

    # new improved tokenizer which can support any transformer model
    def tokenizer(self, t: str) -> List[str]:
        ids = self._pretrained_tokenizer.encode(
            t, max_length=self.max_seq_len, truncation=True
        )
        tokens = self._pretrained_tokenizer.convert_ids_to_tokens(ids)
        return tokens


class TransformersVocab(Vocab):
    """
    Contain the correspondence between numbers and tokens and numericalize.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(itos=[])
        self.tokenizer = tokenizer
        self.special_token_map = self.tokenizer.special_tokens_map
        self.has_cls_token = True if self.special_token_map.get("cls_token") else False
        self.has_sep_token = True if self.special_token_map.get("sep_token") else False
        self.has_unk_token = True if self.special_token_map.get("unk_token") else False
        self.has_pad_token = True if self.special_token_map.get("pad_token") else False

    def numericalize(self, t: Collection[str]) -> List[int]:
        """
        Convert a list of tokens `t` to their ids.
        """
        return self.tokenizer.convert_tokens_to_ids(t)

    def textify(self, nums: Collection[int], sep=" ") -> str:
        """
        Convert a list of `nums` to their tokens.
        """
        # text = self.tokenizer.decode(nums, skip_special_tokens=True)
        text = self.tokenizer.decode(nums)
        if self.has_cls_token:
            text = text.replace(self.tokenizer.cls_token, "")
        if self.has_sep_token:
            text = text.replace(self.tokenizer.sep_token, "")
        if self.has_unk_token:
            text = text.replace(self.tokenizer.unk_token, "")
        if self.has_pad_token:
            text = text.replace(self.tokenizer.pad_token, "")
        text = re.sub(" +", " ", text)
        return text.strip()

    def __getstate__(self):
        return {"itos": self.itos, "tokenizer": self.tokenizer}

    def __setstate__(self, state: dict):
        self.itos = state["itos"]
        self.tokenizer = state["tokenizer"]
        self.stoi = collections.defaultdict(
            int, {v: k for k, v in enumerate(self.itos)}
        )


class TransformerNERDataset(Dataset):
    def __init__(
        self,
        tokens_collection,
        tags_collection,
        tokenizer,
        label2id,
        model_type="bert",
        seq_length=512,
    ):
        self.tokens_collection = tokens_collection
        self.tags_collection = tags_collection
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {y: x for x, y in self.label2id.items()}
        self.model_type = model_type
        self.ignore_token_list = ["\n"]
        self.otag_id = (
            self.label2id["O"] if "O" in self.label2id else self.label2id["o"]
        )
        self.max_seq_len = min(tokenizer.model_max_length, seq_length)

    def __len__(self):
        return len(self.tokens_collection)

    def __getitem__(self, index):
        tokens = self.tokens_collection[index]
        tags = self.tags_collection[index]
        ids, target_tags = [], []

        tokens = [x for x in tokens if x not in self.ignore_token_list]
        if self.model_type in ["roberta", "bart", "longformer"]:
            # tokens = [f" {x}" if index != 0 else x for index, x in enumerate(tokens)]
            tokens = [f" {x}" for index, x in enumerate(tokens)]

        for index, token in enumerate(tokens):
            # Transformer tokenizer can split the input word at sub-word level.
            # For example - `uninvited` can be broken into: 'un', '##in', '##vi', '##ted'
            token_breakup_list = self.tokenizer.encode(token, add_special_tokens=False)
            num_items = len(token_breakup_list)
            ids.extend(token_breakup_list)
            tag_id = self.label2id[tags[index]]
            target_tags.extend([tag_id] * num_items)

        # self.max_seq_len - 2, because we will be adding a [CLS] & [SEP] token when creating the DataBunch
        ids = ids[: self.max_seq_len - 2]
        target_tags = target_tags[: self.max_seq_len - 2]

        if self.model_type in ["xlnet"]:
            ids = ids + [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]
            target_tags = target_tags + [self.otag_id, self.otag_id]
        elif self.model_type in ["xlm", "flaubert"]:
            ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.sep_token_id]
            target_tags = [self.otag_id] + target_tags + [self.otag_id]
        else:
            ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
            target_tags = [self.otag_id] + target_tags + [self.otag_id]

        return {"ids": ids, "target": target_tags, "otag_id": self.otag_id}


def collate_func(
    samples: BatchSamples, tokenizer: PreTrainedTokenizer, seq_len: int = 512
):
    # Function that collect samples and adds padding
    samples = to_data(samples)
    batch_sentences = [x["ids"] for x in samples]
    batch_target = [x["target"] for x in samples]
    otag_id = samples[0]["otag_id"]

    # [CLS] & [SEP] token are already added from the items returned from TransformerNERDataset
    max_len_batch = max([len(x) for x in batch_target])
    max_seq_len = min(tokenizer.model_max_length, seq_len)
    max_len = min(max_len_batch, max_seq_len)

    res_x = tokenizer.batch_encode_plus(
        batch_sentences,
        max_length=max_len,
        padding=True,
        truncation=True,
        is_split_into_words=True,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids, attention_masks = res_x.get("input_ids"), res_x.get("attention_mask")
    token_type_ids = res_x.get("token_type_ids")

    res_y = torch.zeros(len(samples), max_len).long() + otag_id

    pad_first = True if tokenizer.padding_side == "left" else False
    for i, s in enumerate(batch_target):
        if pad_first:
            res_y[i, -len(s) :] = LongTensor(s)
        else:
            res_y[i, : len(s) :] = LongTensor(s)

    # res_y is added in the list as the model has the ability to return the loss if the labels are provided
    if token_type_ids is not None:
        return (
            [input_ids, attention_masks, token_type_ids, res_y],
            [
                res_y,
                attention_masks,
            ],
        )
    else:
        return [input_ids, attention_masks, res_y], [res_y, attention_masks]


class TransformerNERDataBunch(DataBunch):
    "Create a `TextDataBunch` suitable for training an RNN classifier."

    @classmethod
    def create(
        cls,
        train_ds,
        valid_ds,
        tokenizer: PreTrainedTokenizer,
        bs: int = 4,
        val_bs: int = None,
        device: torch.device = None,
        no_check: bool = False,
        seq_len: int = 512,
        **dl_kwargs,
    ) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch`. Passes `**dl_kwargs` on to `DataLoader()`"
        if val_bs is None:
            val_bs = bs
        # write own collate function
        collate_fn = partial(collate_func, tokenizer=tokenizer, seq_len=seq_len)
        train_sampler = SortishSampler(
            train_ds, key=lambda t: len(train_ds[t]["target"]), bs=bs
        )
        train_dl = DataLoader(
            train_ds, batch_size=bs, sampler=train_sampler, drop_last=False, **dl_kwargs
        )
        dataloaders = [train_dl]

        sampler = SortSampler(valid_ds, key=lambda t: len(valid_ds[t]["target"]))
        dataloaders.append(
            DataLoader(
                valid_ds,
                batch_size=val_bs,
                sampler=sampler,
                drop_last=False,
                **dl_kwargs,
            )
        )
        return cls(
            *dataloaders, device=device, collate_fn=collate_fn, no_check=no_check
        )

    @property
    def is_empty(self) -> bool:
        return not (
            (self.train_dl and len(self.train_ds) != 0)
            or (self.valid_dl and len(self.valid_ds) != 0)
            or (self.test_dl and len(self.test_ds) != 0)
        )

    def show_batch(
        self,
        rows: int = 5,
        ds_type: DatasetType = DatasetType.Train,
        reverse: bool = False,
        **kwargs,
    ):
        x, y = self.one_batch(ds_type, True, True)
        batch_tokens = x[0].tolist()
        batch_labels = y[0].tolist()
        num_items = min(rows, self.dl(DatasetType.Train).batch_size)
        results = get_results(
            batch_tokens,
            batch_labels,
            self.train_ds.tokenizer,
            self.train_ds.id2label,
            self.train_ds.model_type,
            num_items,
        )
        df = pd.DataFrame(
            results,
        )
        df.fillna("", inplace=True)
        return df


def get_previous_tokens(token_ids, index, model_type, tok):
    res = models_subwords_map.get(model_type)
    if res is None:
        return [token_ids[index]]

    i, whole_word_ids = index, []
    spl_char, start_token_contain_spl_char = res
    while i > 0:
        curr_token = tok.convert_ids_to_tokens(token_ids[i])
        whole_word_ids.insert(0, token_ids[i])
        if curr_token.startswith(spl_char):
            if start_token_contain_spl_char:
                break
        else:
            if not start_token_contain_spl_char:
                break
        i -= 1
    return whole_word_ids


def get_next_tokens(token_ids, index, model_type, tok):
    res = models_subwords_map.get(model_type)
    if res is None:
        return []

    # i, whole_word_ids = index+1, [token_ids[index]]
    i, whole_word_ids = index + 1, []
    spl_char, start_token_contain_spl_char = res

    while i < len(token_ids):
        curr_token = tok.convert_ids_to_tokens(token_ids[i])
        if curr_token.startswith(spl_char):
            if start_token_contain_spl_char:
                break
        else:
            if not start_token_contain_spl_char:
                break
        whole_word_ids.append(token_ids[i])
        i += 1
    return whole_word_ids


def get_results(batch_tokens, batch_labels, tokenizer, id2label, model_type, num_items):
    results = []
    for index, (tokens, labels) in enumerate(zip(batch_tokens, batch_labels)):
        labels = [id2label[x] for x in labels]
        if index >= num_items:
            break
        prev_label, token_list, entities = labels[0], [(0, tokens[0])], []
        for token_index, (token, label) in enumerate(list(zip(tokens[1:], labels[1:]))):
            label = label.split("-")[-1]
            if label == "O":
                if prev_label != "O":
                    entities.append((token_list, prev_label))
                token_list, prev_label = list(), label

            if prev_label == label:
                token_list.append((token_index + 1, token))
            else:
                if prev_label != "O":
                    entities.append((token_list, prev_label))
                token_list = list()
                prev_label = label
                token_list.append((token_index + 1, token))
        if token_list:
            entities.append((token_list, prev_label))

        # entities is a list [([(4, 1196), (5, 345)], "Address"), ([(34, 444), (35, 3834)], "Person")]
        entity_list = []
        for item in entities:
            token_list, entity = item
            # print("Before - ", tokenizer.decode([x[1] for x in token_list]))
            prev_tokens = get_previous_tokens(
                tokens, token_list[0][0], model_type, tokenizer
            )
            end_tokens = get_next_tokens(
                tokens, token_list[-1][0], model_type, tokenizer
            )

            token_list = prev_tokens + [x[1] for x in token_list[1:]] + end_tokens
            # print("After - ", tokenizer.decode(token_list))
            entity_text = process_text(
                tokenizer.decode(token_list, skip_special_tokens=True)
            )
            if len(entity_text) > 2:
                entity_list.append((entity_text, entity))

        entity_dict = {}
        text = process_text(
            tokenizer.decode(batch_tokens[index], skip_special_tokens=True)
        )
        entity_dict["Text"] = text
        _ = [entity_dict.setdefault(x[1], []).append(x[0]) for x in entity_list if x[0]]
        results.append(entity_dict)
    return results
