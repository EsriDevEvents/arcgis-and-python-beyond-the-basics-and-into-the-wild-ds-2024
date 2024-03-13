import os
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from fastai.callback import Callback
from fastai.torch_core import add_metrics
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from fastprogress.fastprogress import progress_bar


def get_decoder(n_neurons):
    layers = []
    for i in range(len(n_neurons) - 1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i + 1]))
        if i < (len(n_neurons) - 2):
            layers.extend([nn.BatchNorm1d(n_neurons[i + 1]), nn.ReLU()])
    m = nn.Sequential(*layers)
    return m


"""
Pixel-Set encoder module
author: Vivien Sainte Fare Garnot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PixelSetEncoder(nn.Module):
    def __init__(self, input_dim, mlp1, pooling, mlp2, with_extra, extra_size):
        """
        Pixel-set encoder.
        Args:
            input_dim (int): Number of channels of the input tensors
            mlp1 (list):  Dimensions of the successive feature spaces of MLP1
            pooling (str): Pixel-embedding pooling strategy, can be chosen in ('mean','std','max,'min')
                or any underscore-separated combination thereof.
            mlp2 (list): Dimensions of the successive feature spaces of MLP2
            with_extra (bool): Whether additional pre-computed features are passed between the two MLPs
            extra_size (int, optional): Number of channels of the additional features, if any.
        """

        super(PixelSetEncoder, self).__init__()

        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)
        self.pooling = pooling

        self.with_extra = with_extra
        self.extra_size = extra_size

        self.name = "PSE-{}-{}-{}".format(
            "|".join(list(map(str, self.mlp1_dim))),
            pooling,
            "|".join(list(map(str, self.mlp2_dim))),
        )

        self.output_dim = (
            input_dim * len(pooling.split("_"))
            if len(self.mlp2_dim) == 0
            else self.mlp2_dim[-1]
        )

        inter_dim = self.mlp1_dim[-1] * len(pooling.split("_"))

        if self.with_extra:
            self.name += "Extra"
            inter_dim += self.extra_size

        assert input_dim == mlp1[0]
        assert inter_dim == mlp2[0]
        # Feature extraction
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(linlayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        # MLP after pooling
        layers = []
        for i in range(len(self.mlp2_dim) - 1):
            layers.append(nn.Linear(self.mlp2_dim[i], self.mlp2_dim[i + 1]))
            layers.append(nn.BatchNorm1d(self.mlp2_dim[i + 1]))
            if i < len(self.mlp2_dim) - 2:
                layers.append(nn.ReLU())
        self.mlp2 = nn.Sequential(*layers)

    def forward(self, input):
        """
        The input of the PSE is a tuple of tensors as yielded by the PixelSetData class:
          (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
        Pixel-Set : Batch_size x (Sequence length) x Channel x Number of pixels
        Pixel-Mask : Batch_size x (Sequence length) x Number of pixels
        Extra-features : Batch_size x (Sequence length) x Number of features
        If the input tensors have a temporal dimension, it will be combined with the batch dimension so that the
        complete sequences are processed at once. Then the temporal dimension is separated back to produce a tensor of
        shape Batch_size x Sequence length x Embedding dimension
        """
        a, b = input

        if len(a) == 2:
            out, mask = a
            extra = b
            if len(extra) == 2:
                extra, bm = extra
        else:
            out, mask = a, b

        if len(out.shape) == 4:
            # Combine batch and temporal dimensions in case of sequential input
            reshape_needed = True
            batch, temp = out.shape[:2]

            out = out.view(batch * temp, *out.shape[2:])
            mask = mask.view(batch * temp, -1)
            if self.with_extra:
                extra = extra.view(batch * temp, -1)
        else:
            reshape_needed = False

        out = self.mlp1(out)
        out = torch.cat(
            [pooling_methods[n](out, mask) for n in self.pooling.split("_")], dim=1
        )

        if self.with_extra:
            out = torch.cat([out, extra], dim=1)
        out = self.mlp2(out)

        if reshape_needed:
            out = out.view(batch, temp, -1)
        return out


class linlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        out = input.permute((0, 2, 1))  # to channel last
        out = self.lin(out)

        out = out.permute((0, 2, 1))  # to channel first
        out = self.bn(out)
        out = F.relu(out)

        return out


def masked_mean(x, mask):
    out = x.permute((1, 0, 2))
    out = out * mask
    out = out.sum(dim=-1) / mask.sum(dim=-1)
    out = out.permute((1, 0))
    return out


def masked_std(x, mask):
    m = masked_mean(x, mask)
    out = x.permute((2, 0, 1))
    out = out - m
    out = out.permute((2, 1, 0))

    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2

    out = (out**2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 10e-32)  # To ensure differentiability
    out = out.permute(1, 0)
    return out


def maximum(x, mask):
    return x.max(dim=-1)[0].squeeze()


def minimum(x, mask):
    return x.min(dim=-1)[0].squeeze()


pooling_methods = {
    "mean": masked_mean,
    "std": masked_std,
    "max": maximum,
    "min": minimum,
}

"""
Temporal Attention Encoder module
Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.
paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""


class TemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        n_head,
        d_k,
        d_model,
        n_neurons,
        dropout,
        T,
        len_max_seq,
        positions,
    ):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
        """

        super(TemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)

        self.name = "TAE_dk{}_{}Heads_{}_T{}_do{}".format(
            d_k, n_head, "|".join(list(map(str, self.n_neurons))), T, dropout
        )

        if positions is None:
            positions = len_max_seq + 1
        else:
            self.name += "_bespokePos"

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(positions, self.in_channels, T=T), freeze=True
        )

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(
                nn.Conv1d(in_channels, d_model, 1), nn.LayerNorm([d_model, len_max_seq])
            )
            self.name += "_dmodel{}".format(d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.outlayernorm = nn.LayerNorm(self.d_model)

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )

        assert self.n_neurons[0] == n_head * self.d_model
        assert self.n_neurons[-1] == self.d_model
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend(
                [
                    nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                    nn.BatchNorm1d(self.n_neurons[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.positions is None:
            src_pos = (
                torch.arange(1, seq_len + 1, dtype=torch.long)
                .expand(sz_b, seq_len)
                .to(x.device)
            )
        else:
            src_pos = (
                torch.arange(0, seq_len, dtype=torch.long)
                .expand(sz_b, seq_len)
                .to(x.device)
            )
        enc_output = x + self.position_enc(src_pos)

        if self.inconv is not None:
            enc_output = self.inconv(enc_output.permute(0, 2, 1)).permute(0, 2, 1)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)

        enc_output = (
            enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)
        )  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        return enc_output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(n_head * d_k), nn.Linear(n_head * d_k, n_head * d_k)
        )

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(q).view(sz_b, seq_len, n_head, d_k)
        q = q.mean(dim=1).squeeze()  # MEAN query
        q = self.fc2(q.view(sz_b, n_head * d_k)).view(sz_b, n_head, d_k)
        q = q.permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)

        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = v.repeat(n_head, 1, 1)  # (n*b) x lv x d_in

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, 1, d_in)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


def get_sinusoid_encoding_table(positions, d_hid, T):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt1 = F.log_softmax(input, dim=1)
        logpt2 = logpt1.gather(1, target.type(torch.int64))
        logpt3 = logpt2.view(-1)
        pt = Variable(logpt3.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.type(torch.int64).data.view(-1))
            logpt = logpt3 * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt3
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class miou(Callback):
    def __init__(self, nclass):
        self.nclass = nclass

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.y_pred = last_output
        self.y_true = last_target

    def on_epoch_end(self, last_metrics, **kwargs):
        y_pred = self.y_pred.argmax(dim=1)
        iou = 0
        n_classes = self.nclass
        n_observed = n_classes
        for i in range(n_classes):
            y_t = (self.y_true == i).int()
            y_p = (y_pred == i).int()

            inter = torch.sum(y_t * y_p)
            union = torch.sum((y_t + y_p > 0).int())

            if union == 0:
                n_observed -= 1
            else:
                iou += inter / union
        miou = iou / n_observed
        return add_metrics(last_metrics, miou.detach().cpu().numpy())


def mIou_new(y_true, y_pred, cls_list):
    """
    Mean Intersect over Union metric.
    Computes the one versus all IoU for each class and returns the average.
    Classes that do not appear in the provided set are not counted in the average.
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes
    Returns:
        mean Iou (float)
    """
    iou = 0
    n_observed = len(cls_list)
    for i in cls_list:
        y_t = (np.array(y_true) == i).astype(int)
        y_p = (np.array(y_pred) == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union
    return iou / n_observed


def weight_init(m):
    """
    Initializes a model's parameters.
    Credits to: https://gist.github.com/jeasinema
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=0, std=1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=0, std=1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=0, std=1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        try:
            init.normal_(m.bias.data)
        except AttributeError:
            pass
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim, len_max_seq, positions, out_class, **kwargs):
        super(PseTae, self).__init__()

        mlp1 = kwargs.get("mlp1", [32, 64])
        mlp1inp = mlp1.copy()
        mlp1inp.insert(0, input_dim)

        pooling = kwargs.get("pooling", "mean_std")
        mlp2 = kwargs.get("mlp2", [128, 128])

        if pooling not in ["mean", "std", "max", "min"]:
            if mlp2[0] != mlp1[-1] * 2:
                raise Exception("MLP2 input should be double of MLP1 output")
        else:
            del mlp2[0]
            mlp2.insert(0, mlp1[1])

        n_head = kwargs.get("n_head", 4)
        d_k = kwargs.get("d_k", 32)

        mlp3 = []
        mlp3.extend([mlp2[1]] * 2)
        mlp3.insert(0, mlp2[1] * n_head)

        dropout = kwargs.get("dropout", 0.2)

        T = kwargs.get("T", 1000)

        mlp4 = kwargs.get("mlp4", [64, 32])
        mlp4inp = mlp4.copy()
        mlp4inp.insert(0, mlp3[-1])
        mlp4inp.append(out_class)

        d_model = int(mlp3[0] / n_head)

        self.spatial_encoder = PixelSetEncoder(
            input_dim,
            mlp1=mlp1inp,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=False,
            extra_size=4,
        )
        self.temporal_encoder = TemporalAttentionEncoder(
            in_channels=mlp2[-1],
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            n_neurons=mlp3,
            dropout=dropout,
            T=T,
            len_max_seq=len_max_seq,
            positions=positions,
        )
        self.decoder = get_decoder(mlp4inp)
        self.name = "_".join([self.spatial_encoder.name, self.temporal_encoder.name])

    def forward(self, input, target):
        """
        Args:
           input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
           Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
           Pixel-Mask : Batch_size x Sequence length x Number of pixels
           Extra-features : Batch_size x Sequence length x Number of features
        """
        out = self.spatial_encoder((input, target))
        out = self.temporal_encoder(out)
        out = self.decoder(out)
        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                s / total * 100, t / total * 100, c / total * 100
            )
        )


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_eval(data, model, class_dict, convertmap):
    validarr = torch.cat([i[0][None, :, :, :] for i, j in data.valid_ds], axis=0)
    batch_size = data.batch_size
    labsarr = torch.stack([j for i, j in data.valid_ds])
    final_img = torch.moveaxis(validarr, 3, 1)[:, :, :, :, None]
    img_arr = torch.reshape(
        final_img,
        (
            final_img.shape[0] * final_img.shape[1],
            final_img.shape[2],
            final_img.shape[3],
            1,
        ),
    ).to(model._device)
    final_labs = [validarr.shape[3] * [labsarr[l]] for l in range(validarr.shape[0])]
    final_labs = torch.stack(sum(final_labs, [])).to(model._device)
    divided = DataLoader(img_arr, batch_size=batch_size, pin_memory=False)
    prediction = []
    for i in progress_bar(divided):
        sim = torch.ones(i.shape[0], i.shape[1], 1).to(model._device)
        model.eval()
        with torch.no_grad():
            pred = model(i, sim)
        prediction.append(pred.argmax(dim=1))
    if convertmap:
        prediction = np.array(
            [convertmap.get(item, item) for item in torch.cat(prediction).cpu().numpy()]
        )
        final_labs = np.array(
            [convertmap.get(item, item) for item in final_labs.cpu().numpy()]
        )

    preds = np.array([class_dict.get(item, item) for item in prediction])
    trues = np.array([class_dict.get(item, item) for item in final_labs])
    mats = confusion_matrix_analysis(confusion_matrix(preds, trues), class_dict)
    miou = mIou_new(preds, trues, list(class_dict.values()))
    return mats, miou


def confusion_matrix_analysis(mat, cls_dict):
    """
    This method computes all the performance metrics from the confusion matrix.
    In addition to overall accuracy, the precision, recall, f-score and IoU for
    each class is computed.The class-wise metrics are averaged to provide overall
    indicators in two ways (MICRO and MACRO average).
    Args:
        mat (array): confusion matrix
    Returns:
        per_class (dict) : per class metrics
        overall (dict): overall metrics
    """
    TP = 0
    FP = 0
    FN = 0

    per_class = {}
    zero_divide = lambda n, d: 0 if n == 0 or d == 0 else n / d

    for j in range(mat.shape[0]):
        d = {}
        tp = np.sum(mat[j, j])
        fp = np.sum(mat[:, j]) - tp
        fn = np.sum(mat[j, :]) - tp

        d["IoU"] = zero_divide(tp, tp + fp + fn)
        d["Precision"] = zero_divide(tp, tp + fp)
        d["Recall"] = zero_divide(tp, tp + fn)
        d["F1-score"] = zero_divide(2 * tp, 2 * tp + fp + fn)

        per_class[str(list(cls_dict.values())[j])] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    overall["micro_IoU"] = TP / (TP + FP + FN)
    overall["micro_Precision"] = TP / (TP + FP)
    overall["micro_Recall"] = TP / (TP + FN)
    overall["micro_F1-score"] = 2 * TP / (2 * TP + FP + FN)

    macro = pd.DataFrame(per_class).transpose().mean()
    overall["MACRO_IoU"] = macro.loc["IoU"]
    overall["MACRO_Precision"] = macro.loc["Precision"]
    overall["MACRO_Recall"] = macro.loc["Recall"]
    overall["MACRO_F1-score"] = macro.loc["F1-score"]

    overall["Accuracy"] = np.sum(np.diag(mat)) / np.sum(mat)

    return per_class, overall
