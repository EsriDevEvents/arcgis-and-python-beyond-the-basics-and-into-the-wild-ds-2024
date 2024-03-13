import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from fastai.vision import create_body
from fastai.basic_train import Learner
import torch.nn.functional as F
from fastai.basic_train import LearnerCallback
import random
from functools import partial
from .._utils.image_captioning_data import show_image_and_text
from fastai.vision import open_image
import matplotlib.pyplot as plt
from fastai.callbacks.hooks import model_sizes
import numpy as np
from fastai.callback import Callback
from fastai.torch_core import add_metrics
from fastai.text import Counter, exp
from .._utils.image_captioning_data import BeamSearchAttention as BeamSearch
from random import choice
from .._utils.image_captioning_data import normalize
from fastai.torch_core import split_model_idx, flatten_model
from pathlib import Path
import shutil
import os
import sys
import tempfile
import fasttext
import fasttext.util


EPS = 1e-5


class EncoderAttention(nn.Module):
    def __init__(self, backbone, cut=None, pretrained=True):
        """Load the pretrained backbone and replace top fc layer."""
        super().__init__()
        self.backbone = create_body(backbone, cut=cut, pretrained=pretrained)
        # Get number of channels of backbone.
        self.feature_size = model_sizes(self.backbone, size=(200, 200))[-1][1]

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.backbone(images)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).view(B, H * W, C)
        return features.contiguous()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_size, hidden_size, attention_size):
        super().__init__()
        self.W1 = nn.Linear(encoder_size, attention_size)
        self.W2 = nn.Linear(hidden_size, attention_size)
        self.V = nn.Linear(attention_size, 1)

    def forward(self, features, hidden):
        # features shape: B, spatial_size x spatial size, num_channels
        # hidden shape: B, hidden_size
        hidden = hidden.unsqueeze(1)
        feature_attention = self.W1(features)
        hidden_attention = self.W2(hidden)
        B, HW, C = features.shape
        C_att = feature_attention.shape[-1]
        B_hidden = hidden_attention.shape[0]

        # This will be required while doing beam search
        if not self.training and B_hidden != B:
            beam_size = int(B_hidden / B)
            feature_attention = feature_attention.unsqueeze(1).repeat(
                1, beam_size, 1, 1
            )
            feature_attention = feature_attention.view(B * beam_size, HW, C_att)
            features = features.unsqueeze(1).repeat(1, beam_size, 1, 1)
            features = features.view(B * beam_size, HW, C)

        score = torch.tanh(feature_attention + hidden_attention)
        attention_weights = F.softmax(self.V(score), dim=1)
        context_vector = attention_weights * features
        context_vector = context_vector.sum(dim=1)
        return context_vector, attention_weights


def select_nucleus(outp, p=0.5):
    # print("select_nuc", outp.shape)
    # import pdb; pdb.set_trace();
    outp = outp.squeeze(0)
    probs = F.softmax(outp, dim=-1)
    idxs = torch.argsort(probs, descending=True)
    res, cumsum = [], 0.0
    for idx in idxs:
        res.append(idx)
        cumsum += probs[idx]
        if cumsum > p:
            return idxs.new_tensor([choice(res)])


class DecoderAttention(nn.Module):
    def __init__(
        self,
        embed_size,
        feature_size,
        hidden_size,
        vocab_size,
        attention_size,
        max_seq_length=20,
        pretrained_embeddings=False,
        vocab=None,
        teacher_forcing=True,
        dropout=0.5,
    ):
        """
        Decoder will take in features and will generate
        captions from those. It defines the basic LSTM,
        attention and few linear layers.
        """
        super(DecoderAttention, self).__init__()
        self.vocab = vocab
        if pretrained_embeddings is not False:
            self.embed = self.load_embedding(pretrained_embeddings, vocab)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTMCell(2 * embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed_features = nn.Linear(feature_size, embed_size)
        self.hidden_init = nn.Linear(embed_size, hidden_size)
        self.c_init = nn.Linear(embed_size, hidden_size)
        self.attention = BahdanauAttention(embed_size, hidden_size, attention_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.teacher_forcing = teacher_forcing
        self.dropout = nn.Dropout(dropout)
        self.bos_tag = self.vocab.stoi["xxbos"]

    def forward(self, features, captions_lengths, return_packed=False):
        """
        :param feature: features from encoder B, HXW, C
        :param captions_length: tuple containing captions and length
        :param return packed: tells whether to return packed sequence or not
        """
        # import pdb; pdb.set_trace();
        features = self.embed_features(features)
        captions, lengths = captions_lengths
        # import pdb; pdb.set_trace();
        embeddings = self.embed(captions)
        # we will not process xxeos token.
        packed = pack_padded_sequence(embeddings, lengths.cpu() - 1, batch_first=True)
        self.init_hidden(features)
        hx, cx = self.hx, self.cx
        outputs = []
        total_processed = 0
        next_words = (
            torch.tensor([self.bos_tag] * features.shape[0]).to(features.device).long()
        )
        for batch_size in packed[1]:
            features = features[:batch_size]  # batch_size, HXW, feature_space
            hx = hx[:batch_size]  # batch_size, hidden_dim
            cx = cx[:batch_size]  # batch_size, hidden_dim
            # context_vector: batch_size, attention_size
            # attention_map: batch_size, HXW, 1
            context_vector, attention_map = self.attention(features, hx)
            input_batch = packed[0][total_processed : total_processed + batch_size]
            # select input batch on based of teacher forcing prob.
            if self.teacher_forcing < random.random():
                input_batch = self.embed(next_words)
                input_batch = input_batch[:batch_size]

            # Adding attention image to input batch
            input_batch = torch.cat([input_batch, context_vector], dim=1)

            hx, cx = self.lstm(
                input_batch, (hx, cx)
            )  # batch_size, embedding_dim + attention_size
            current_op = self.out(self.dropout(hx))  # batch_size, vocab_size
            next_words = current_op.argmax(dim=-1)
            outputs.append(current_op)
            total_processed += batch_size

        return torch.cat(outputs), packed[1]

    def init_hidden(self, features):
        # mean does work of adaptive average pool
        self.hx = self.hidden_init(features.mean(1))
        self.cx = self.c_init(features.mean(1))

    def load_embedding(self, pretrained_embedding, vocab):
        # Initialize zeros vectors
        vectors = torch.zeros(len(vocab.itos), pretrained_embedding.get_dimension())
        # Load vectors from pretrained embeddings.
        for index, token in enumerate(vocab.itos):
            vectors[index] = torch.tensor(pretrained_embedding.get_word_vector(token))
        # Using pretrained vectors from fastext to initialize embeddings.
        layer = nn.Embedding.from_pretrained(vectors)
        return layer

    def decode_step(self, current_words, im, hidden):
        # This function is used in nucleus decoding for which
        # we have not given support
        with torch.no_grad():
            hx, cx = hidden
            embedding = self.embed(current_words)
            context_vector, attention_map = self.attention(im, hx[[0]])
            if embedding.shape[0] != context_vector.shape[0]:
                context_vector = context_vector[[0]].repeat(embedding.shape[0], 1)

            if embedding.shape[0] != hx.shape[0]:
                hx = hx[[0]].repeat(embedding.shape[0], 1)
                cx = cx[[0]].repeat(embedding.shape[0], 1)

            input_text = torch.cat([embedding, context_vector], dim=1)

            hx, cx = self.lstm(input_text, (hx, cx))
            out = self.out(hx)
        return out, (hx, cx), attention_map

    def sample_nucleus(self, im, p, max_len):
        # Nucleus decoding is not getting used.
        self.init_hidden(im)
        hidden = self.hx, self.cx
        predicted_sentence = []
        bos_tag = [self.vocab.stoi["xxbos"]]
        current_words = torch.tensor(bos_tag).to(im.device).long()
        for i in range(max_len):
            out, hidden, attention_map = self.decode_step(current_words, im, hidden)
            current_words = select_nucleus(out, p=p)
            print(current_words)
            predicted_sentence.append(current_words.item())
        return self.vocab.textify(predicted_sentence)

    def sample(self, features, beam_width, max_len):
        """
        takes in a batch of features
        """
        batch_size = features.shape[0]
        bos = self.vocab.stoi["xxbos"]
        eos = self.vocab.stoi["xxeos"]

        # decode step function it will be used in Beam Search.
        # Takes in images featurs, current input and hidden state
        def decode_step(self, im, current_words, hidden):
            with torch.no_grad():
                hx, cx = hidden["hx"], hidden["cx"]
                embedding = self.embed(current_words)
                context_vector, attention_map = self.attention(im, hx)
                input_text = torch.cat([embedding, context_vector], dim=1)
                hx, cx = self.lstm(input_text, (hx, cx))
                out = self.out(hx).log_softmax(-1)
            return out, {"hx": hx, "cx": cx, "attention_map": attention_map}

        # Initialize beam search with parameters.
        beam_search = BeamSearch(
            end_index=eos,
            max_steps=max_len,
            beam_size=beam_width,
            per_node_beam_size=beam_width // 2,
        )
        self.init_hidden(features)
        states = {"hx": self.hx, "cx": self.cx}
        start_predictions = torch.tensor([bos] * batch_size).to(features.device)
        # Initialize the function with features
        partial_decode_step = partial(decode_step, self, features)
        # Do beam search
        output = beam_search.search(start_predictions, states, partial_decode_step)

        captions = []
        # output[0] contains tokens of captions ranked
        # by probability.
        for caption in output[0]:
            caption = caption[0]
            captions.append(self.vocab.textify(caption[caption != eos]))

        return captions, (output[0], output[2])


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, captions):
        features = self.encoder(inputs)
        return self.decoder(features, captions)

    def sample(self, input_batch, beam_width=8, max_len=25):
        self.eval()
        if type(input_batch) is tuple:
            input_batch, _ = input_batch

        features = self.encoder(input_batch)
        features = self.decoder.embed_features(features)
        captions, attention_maps = self.decoder.sample(
            features, beam_width=beam_width, max_len=max_len
        )
        return input_batch, captions, attention_maps


def loss_function_attention(inputs, captions, lengths):
    # skipping xxbos because that is going as input
    # and we need to predict next indexes onwards
    packed = pack_padded_sequence(captions[:, 1:], lengths.cpu() - 1, batch_first=True)
    return F.cross_entropy(inputs[0], packed[0])


def load_fasttext_embeddings(language="en"):
    embeddings_file = f"cc.{language}.300.bin"
    embeddings_path = os.path.join(Path.home(), ".cache", "embeddings")
    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)
    embeddings_file_path = os.path.join(embeddings_path, embeddings_file)
    # print(embeddings_file_path, os.path.exists(embeddings_file_path))
    if not os.path.exists(embeddings_file_path):
        fasttext.util.download_model(language, if_exists="ignore")
        shutil.move(embeddings_file, embeddings_path)
        os.remove(embeddings_file + ".gz")

    orig_stderr = sys.stderr
    temp_f = tempfile.TemporaryFile(mode="w")
    sys.stderr = temp_f
    ft = fasttext.load_model(embeddings_file_path)
    sys.stderr = orig_stderr
    return ft


def image_captioner_learner(
    data, backbone, attention=True, decoder_params=None, metrics=None, pretrained=True
):
    if attention:
        pretrained_embeddings = decoder_params.get("pretrained_embeddings", False)

        decoder_params = {
            "embed_size": decoder_params.get("embed_size", 100),
            "hidden_size": decoder_params.get("hidden_size", 100),
            "attention_size": decoder_params.get("attention_size", 100),
            "max_seq_length": decoder_params.get("max_seq_length", 20),
            "teacher_forcing": decoder_params.get("teacher_forcing", 1),
            "dropout": decoder_params.get("dropout", 0.1),
        }

        # Download pretrained embeddings if not already present.
        if pretrained_embeddings:
            ft = load_fasttext_embeddings(data.lang)
            if ft.get_dimension() != decoder_params["embed_size"]:
                decoder_params["embed_size"] = ft.get_dimension()
            decoder_params["pretrained_embeddings"] = ft

        decoder_params["vocab"] = data.vocab
        decoder_params["vocab_size"] = len(data.vocab.itos)
        # create ecoder using backbone
        encoder = EncoderAttention(backbone, pretrained=pretrained)
        # get channels from encoder which were computed during init.
        decoder_params["feature_size"] = encoder.feature_size
        # create LSTM decoder
        decoder = DecoderAttention(**decoder_params)
    else:
        raise NotImplementedError
    model = EncoderDecoder(encoder, decoder)

    # create learner using data.
    learn = Learner(
        data,
        model,
        loss_func=loss_function_attention,
        metrics=[accuracy, CorpusBLEU(len(data.vocab.itos))],
    )
    # Append callback
    learn.callbacks.append(ModifyInputCallback(learn))
    # learn.callbacks.append(ReduceTeacherForcing(learn))
    # split model for discriminative learning rate at embedding layer.
    split_layer_groups(learn)
    # return learner
    return learn


def split_layer_groups(learn):
    layers = flatten_model(learn.model)
    for idx, l in enumerate(layers):
        if "embed" in layers:
            break
    learn.layer_groups = split_model_idx(learn.model, [idx])


class ModifyInputCallback(LearnerCallback):
    def on_batch_begin(self, last_input, last_target, **kwargs):
        """
        Also include target in the input while training the model,
        because we do teacher forcing for the first 5 epochs.
        """
        return {"last_input": (last_input, last_target), "last_target": last_target}


class ReduceTeacherForcing(LearnerCallback):
    def on_epoch_end(self, **kwargs):
        """
        Reduce teacher forcing by a factor of 0.7 after each epoch.
        """
        if kwargs["epoch"] > 5:
            # 0.7 factor is experimental, can be improved based
            # on experimentation or defacto standards.
            # It will remain 1 for the first 5 epochs.
            self.learn.model.decoder.teacher_forcing *= 0.99


def accuracy(inputs, captions, lengths):
    # skipping xxbos because that is going as input
    # and we need to predict next indexes onwards
    packed = pack_padded_sequence(captions[:, 1:], lengths.cpu() - 1, batch_first=True)
    return (inputs[0].argmax(dim=-1) == packed[0]).float().mean()


class NGram:
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


def sentence_bleu(pred, targ, max_n=5000):
    corrects = [get_correct_ngrams(pred, targ, n, max_n=max_n) for n in range(1, 5)]
    n_precs = [c / t for c, t in corrects]
    len_penalty = exp(1 - len(targ) / len(pred)) if len(pred) < len(targ) else 1
    return len_penalty * ((n_precs[0] * n_precs[1] * n_precs[2] * n_precs[3]) ** 0.25)


def pad_tensor(tensor, batch_size):
    length = tensor.shape[0]
    n = batch_size - length
    return torch.cat([tensor, torch.zeros(n).to(tensor.device).long()], dim=0)[:, None]


def pack_into_batch(predictions, batches, batch_size):
    batch_of_captions = []
    total_processed = 0
    for l in batches.tolist():
        batch_of_captions.append(
            pad_tensor(predictions[total_processed : total_processed + l], batch_size)
        )
        total_processed += l
    return torch.cat(batch_of_captions, dim=1)


def post_process(predictions, gt_captions, batches):
    """
    predictions: padded sequence.
    gt_captions: padded_sequence ground truth.
    """
    batch_size = gt_captions.shape[0]
    predictions = predictions.argmax(dim=-1)
    padded_batch = pack_into_batch(predictions, batches, batch_size).cpu().numpy()
    gt_captions = gt_captions[:, 1:-1].cpu().numpy()
    pred_captions = []
    gt_captions_list = []
    for n in range(padded_batch.shape[0]):
        pred_captions.append(np.trim_zeros(padded_batch[n], trim="b"))
        gt_captions_list.append(np.trim_zeros(gt_captions[n], trim="b"))
    return pred_captions, gt_captions_list


class CorpusBLEU(Callback):
    def __init__(self, vocab_sz):
        self.vocab_sz = vocab_sz
        self.name = "bleu"
        self.pred_len, self.targ_len, self.corrects, self.counts = (
            0,
            0,
            [0] * 4,
            [0] * 4,
        )

    def on_epoch_begin(self, **kwargs):
        self.pred_len, self.targ_len, self.corrects, self.counts = (
            0,
            0,
            [0] * 4,
            [0] * 4,
        )

    def on_batch_end(self, last_output, last_target, **kwargs):
        preds, targs = post_process(last_output[0], last_target[0], last_output[1])
        for pred, targ in zip(preds, targs):
            self.pred_len += len(pred)
            self.targ_len += len(targ)
            for i in range(4):
                c, t = get_correct_ngrams(pred, targ, i + 1, max_n=self.vocab_sz)
                self.corrects[i] += c
                self.counts[i] += t

    def on_epoch_end(self, last_metrics, **kwargs):
        precs = [c / (t + EPS) for c, t in zip(self.corrects, self.counts)]
        len_penalty = (
            exp(1 - self.targ_len / (self.pred_len + EPS))
            if self.pred_len < self.targ_len
            else 1
        )
        bleu = len_penalty * ((precs[0] * precs[1] * precs[2] * precs[3]) ** 0.25)
        return add_metrics(last_metrics, bleu)


def predict_image(
    self, image, visualize, beam_width=5, max_len=20, visualize_attention=False
):
    import skimage
    import math

    # if PIL image/ ndarray convert to torch.tensor
    image = open_image(image)
    # normalize and resize
    image = image.resize(self._data.chip_size)
    image_tensor = normalize(image.px, *self._data.norm_stats)
    image_tensor = image_tensor[None].to(self._device)  # batch size 1
    # get captions
    _, caption, att_vars = self.learn.model.sample(
        image_tensor, beam_width=beam_width, max_len=max_len
    )
    if visualize:
        fig, ax = plt.subplots()
        show_image_and_text(ax, image.px, caption[0], False)

    if visualize_attention:
        cap = att_vars[0][0, 0]
        length = len(cap[cap != 3])
        b = 0  # batch index is zero
        nrows = math.ceil(length / 5)
        fig, axs = plt.subplots(nrows, 5, figsize=((20.0, 10.0)), squeeze=False)
        axs = axs.flatten()
        for k, atts in enumerate(att_vars[1]):
            if cap[k] != 3:
                ax = axs[k]
                attn_map = atts.view(1, beam_width, 7, 7)[b][0].cpu().numpy()
                image = skimage.transform.resize(attn_map, [224, 224])
                image = np.flip(image, 0)
                image = np.flip(image, 1)
                string = self._data.vocab.itos[cap[k].item()]
                ax.text(
                    0,
                    1,
                    "%s" % (string),
                    color="black",
                    backgroundcolor="white",
                    fontsize=12,
                )
                ax.imshow(image, cmap="gray")
            else:
                break

        for z in range(k, len(axs)):
            fig.delaxes(axs[z])

    return caption[0]


def get_bleu(self, data, beam_width=5, max_len=20):
    """
    Computes BLEU score considering after beam search.
    """
    captions, captions_gts = [], []

    # iterate over the validation set and
    # predictions and ground truth captions
    for x_in, current_caption_gt in data.valid_dl:
        captions_gts.extend(
            [cap.tolist()[: length.item()] for cap, length in zip(*current_caption_gt)]
        )
        _, current_captions, _ = self.learn.model.sample(
            x_in, beam_width=beam_width, max_len=max_len
        )
        captions.extend(current_captions)

    # compute counting of 1,2,3 and 4 grams
    pred_len, targ_len, corrects, counts = 0, 0, [0] * 4, [0] * 4
    for pred, targ in zip(captions, captions_gts):
        targ = targ[1:-1]
        pred = data.vocab.numericalize(data.tokenizer._process_all_1([pred])[0])
        pred_len += len(pred)
        targ_len += len(targ)
        for i in range(4):
            c, t = get_correct_ngrams(pred, targ, i + 1, max_n=len(data.vocab.itos))
            corrects[i] += c
            counts[i] += t

    # compute precision of all type of bleu
    n_precs = [c / (t + EPS) for c, t in zip(corrects, counts)]
    precs = n_precs

    # compute overall bleu as https://www.aclweb.org/anthology/P02-1040.pdf
    len_penalty = exp(1 - targ_len / (pred_len + EPS)) if pred_len < targ_len else 1
    bleu = len_penalty * ((precs[0] * precs[1] * precs[2] * precs[3]) ** 0.25)

    BLEU = {
        "bleu-1": n_precs[0],
        "bleu-2": n_precs[1],
        "bleu-3": n_precs[2],
        "bleu-4": n_precs[3],
        "BLEU": bleu,
    }

    return BLEU
