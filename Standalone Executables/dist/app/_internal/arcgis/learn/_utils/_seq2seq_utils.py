import re
import os
import sys
import copy
from copy import deepcopy
import types
import random
import warnings
import traceback
import pandas as pd
from functools import partial

HAS_FASTAI = True
try:
    from enum import Enum
    import torch
    import arcgis
    from fastprogress.fastprogress import progress_bar
    from fastai.text import (
        TextList,
        TextClasDataBunch,
        pad_collate,
        TextDataBunch,
        SortishSampler,
        SortSampler,
        ItemList,
        ItemBase,
        Text,
    )
    from fastai.text import *
    from fastai.layers import CrossEntropyFlat
    from fastai.basic_train import Learner
    from fastai.torch_core import to_data
    from fastai.callback import Callback
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

HAS_NUMPY = True
try:
    import numpy as np

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except:
    HAS_NUMPY = False


max_len = 100
DatasetType = Enum("DatasetType", "Train Valid Test Single Fix")


class SequenceToSequenceDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training."

    @classmethod
    def create(
        cls,
        train_ds,
        valid_ds,
        test_ds=None,
        path=".",
        bs=32,
        pad_idx=1,
        dl_tfms=None,
        pad_first=False,
        no_check=False,
        backwards=False,
        val_bs=None,
        collate_fn=None,
        **dl_kwargs
    ):
        "Function that transform the `datasets` in a `DataBunch`. Passes `**dl_kwargs` on to `DataLoader()`"
        device = dl_kwargs.pop("device", None)
        from torch.utils.data import DataLoader

        datasets = cls._init_ds(train_ds, valid_ds)
        val_bs = bs
        collate_fn = partial(
            seq2seq_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards
        )
        train_sampler = SortishSampler(
            datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs // 2
        )
        train_dl = DataLoader(
            datasets[0],
            batch_size=bs,
            sampler=train_sampler,
            drop_last=True,
            **dl_kwargs
        )
        dataloaders = [train_dl]
        for i, ds in enumerate(datasets[1:]):
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(
                DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs)
            )
        return cls(
            *dataloaders,
            dl_tfms=dl_tfms,
            path=path,
            collate_fn=collate_fn,
            no_check=no_check,
            device=device
        )


class SequenceToSequenceTextList(TextList):
    _bunch = SequenceToSequenceDataBunch
    _label_cls = TextList

    def reconstruct(self, t: torch.Tensor):
        if isinstance(t, (list)):  # handling the 'shift_tfm' dataloader transform
            t = t[0]
        elif t.ndim == 2:
            t = t.argmax(-1)
        idx_min = (t != self.pad_idx).nonzero().min()
        idx_max = (t != self.pad_idx).nonzero().max()
        return Text(
            t[idx_min : idx_max + 1], self.vocab.textify(t[idx_min : idx_max + 1])
        )


class SequenceToSequenceLearner(Learner):
    def __init__(self, data, model, metrics=None, loss_func=None, **learn_kwargs):
        # self.model_name = model_name
        pretrained_model_name = model.pretrained_model_name
        super().__init__(
            data, model, metrics=metrics, loss_func=loss_func, **learn_kwargs
        )
        self.tokenizer = (
            self.model._tokenizer
        )  # AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.__class__.__name__ == "MarianTokenizer":
            self.tokenizer.clean_up_tokenization = clean_up_tokenization

    def show_results(
        self, ds_type=DatasetType.Valid, rows: int = 5, num_beams=4, max_len=50
    ):
        if not HAS_NUMPY:
            raise Exception("This function requires numpy.")
        from IPython.display import display, HTML

        "Show `rows` result of predictions on `ds_type` dataset."
        ds = self.dl(ds_type).dataset
        x, y = self.data.one_batch(ds_type, detach=False, denorm=False)
        preds = self.pred_batch(batch=(x, y))
        xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(rows)]
        ys = [ds.x.reconstruct(grab_idx(y, i)) for i in range(rows)]
        zs = self.model._transformer.generate(
            x[0], num_beams=num_beams, max_length=max_len
        )
        items, names = [], ["text", "target", "pred"]
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            txt_x = self.tokenizer.decode(x.data, skip_special_tokens=True)
            txt_y = self.tokenizer.decode(y.data, skip_special_tokens=True)
            txt_z = self.tokenizer.decode(z, skip_special_tokens=True)
            items.append([txt_x, txt_y, txt_z])
        items = np.array(items)
        df = pd.DataFrame({n: items[:, i] for i, n in enumerate(names)}, columns=names)
        with pd.option_context("display.max_colwidth", 0):
            display(HTML(df.to_html(index=False)))

    def predict_batch(self, batch_text, num_beams, max_length, min_length):
        tok = self.model._tokenizer
        encoded_input_batch = tok.batch_encode_plus(
            batch_text, padding=True, return_tensors="pt"
        )["input_ids"]
        encoded_output_batch = self.model._transformer.generate(
            encoded_input_batch.to(self.model._transformer.device.type),
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
        )
        decoded_output_batch = tok.batch_decode(
            encoded_output_batch, skip_special_tokens=True
        )
        return decoded_output_batch, encoded_output_batch

    def predict(
        self, text_list, batch_size, show_progress, **kwargs
    ):  # num_beams=4, max_len=50):
        tok = self.model._tokenizer
        num_beams = kwargs.get("num_beams", 1)
        max_length = kwargs.get("max_length", 20)
        min_length = kwargs.get("min_length", 10)

        if isinstance(text_list, (list)):
            decoded_output = []
            for i in progress_bar(
                range(0, len(text_list), batch_size), display=show_progress
            ):
                batch_text = text_list[i : i + batch_size]
                decoded_output_batch, encoded_output_batch = self.predict_batch(
                    batch_text,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                )
                decoded_output.extend(decoded_output_batch)
        else:
            encoded_input = text_list
            # encoded_input.unsqueeze_(0)
            encoded_output = self.model._transformer.generate(
                encoded_input.to(self.model._transformer.device.type),
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
            )
            decoded_output = tok.batch_decode(encoded_output, skip_special_tokens=True)
        return decoded_output


class NGram:
    if not HAS_NUMPY:
        raise Exception("This function requires numpy.")

    def __init__(self, ngram, max_n=5000):
        self.ngram, self.max_n = ngram, max_n

    def __eq__(self, other):
        if len(self.ngram) != len(other.ngram):
            return False
        return np.all(np.array(self.ngram) == np.array(other.ngram))

    def __hash__(self):
        return int(sum([o * self.max_n**i for i, o in enumerate(self.ngram)]))


def get_grams(x, n, max_n=5000):
    return (
        x
        if n == 1
        else [NGram(x[i : i + n], max_n=max_n) for i in range(len(x) - n + 1)]
    )


def get_correct_ngrams(pred, targ, n, max_n=5000):
    pred_grams, targ_grams = (
        get_grams(pred, n, max_n=max_n),
        get_grams(targ, n, max_n=max_n),
    )
    pred_cnt, targ_cnt = Counter(pred_grams), Counter(targ_grams)
    return sum([min(c, targ_cnt[g]) for g, c in pred_cnt.items()]), len(pred_grams)


class CorpusBLEU(Callback):
    def __init__(self, vocab_sz):
        self.vocab_sz = vocab_sz
        self.name = "bleu"

    def on_epoch_begin(self, **kwargs):
        self.pred_len, self.targ_len, self.corrects, self.counts = (
            0,
            0,
            [0] * 4,
            [0] * 4,
        )

    def on_batch_end(self, last_output, last_target, **kwargs):
        loss, last_output = last_output
        last_output = last_output.argmax(dim=-1)
        for pred, targ in zip(last_output.cpu().numpy(), last_target.cpu().numpy()):
            self.pred_len += len(pred)
            self.targ_len += len(targ)
            for i in range(4):
                c, t = get_correct_ngrams(pred, targ, i + 1, max_n=self.vocab_sz)
                self.corrects[i] += c
                self.counts[i] += t

    def on_epoch_end(self, last_metrics, **kwargs):
        precs = [c / t for c, t in zip(self.corrects, self.counts)]
        len_penalty = (
            exp(1 - self.targ_len / self.pred_len)
            if self.pred_len < self.targ_len
            else 1
        )
        bleu = len_penalty * ((precs[0] * precs[1] * precs[2] * precs[3]) ** 0.25)
        return add_metrics(last_metrics, bleu)


def seq2seq_loss(out, targ, pad_idx=1):
    loss, _ = out
    return loss


def seq2seq_acc(out, targ, pad_idx=1):
    loss, out = out
    bs, targ_len = targ.size()
    _, out_len, vs = out.size()
    if targ_len > out_len:
        out = F.pad(out, (0, 0, 0, targ_len - out_len, 0, 0), value=pad_idx)
    if out_len > targ_len:
        targ = F.pad(targ, (0, out_len - targ_len, 0, 0), value=pad_idx)
    out = out.argmax(2)
    return (out == targ).float().mean()


def seq2seq_collate(samples, pad_idx=1, pad_first=True, backwards=False):
    "Function that collect samples and adds padding. Flips token order if needed"
    samples = to_data(samples)
    max_len_x, max_len_y = (
        max([len(s[0]) for s in samples]),
        max([len(s[1]) for s in samples]),
    )
    res_x = torch.zeros(len(samples), max_len_x).long() + pad_idx
    res_y = torch.zeros(len(samples), max_len_y).long() + pad_idx
    if backwards:
        pad_first = not pad_first
    for i, s in enumerate(samples):
        if pad_first:
            res_x[i, -len(s[0]) :], res_y[i, -len(s[1]) :] = (
                torch.LongTensor(s[0]),
                torch.LongTensor(s[1]),
            )
        else:
            res_x[i, : len(s[0])], res_y[i, : len(s[1])] = (
                torch.LongTensor(s[0]),
                torch.LongTensor(s[1]),
            )
    if backwards:
        res_x, res_y = res_x.flip(1), res_y.flip(1)
    return res_x, res_y


# def shift_tfm(b):
#     #transform to modify input for teacher forcing
#     x,y = b
#     y = torch.nn.functional.pad(y, (1, 0), value=0)
#     return [x,y[:,:-1]], y[:,1:]


def teacher_forcing_tfm(b):
    x, y = b
    return [x, y], y


def clean_up_tokenization(out_string: str) -> str:
    """
    Additional handling of '▁' in clean_up_tokenization for marian tokenizer.
    """
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    out_string = re.sub(r"▁(?=\S)", " ", out_string)
    return out_string
