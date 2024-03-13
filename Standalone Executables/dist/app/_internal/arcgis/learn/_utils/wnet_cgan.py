from logging import exception
from fastai.vision import (
    ItemBase,
    ItemList,
    Tensor,
    ImageList,
    Tuple,
    Path,
    get_transforms,
    random,
    open_image,
    Image,
    math,
    plt,
    torch,
    Learner,
    partial,
    optim,
    ifnone,
    image2np,
)
from .._utils.common import (
    ArcGISImageList,
    ArcGISMSImage,
    get_top_padding,
    get_nbatches,
)
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from .._utils.cyclegan import get_activations, InceptionV3
import os
import json


class ImageTuple(ItemBase):
    def __init__(self, img1, img2, img3):
        self.img1, self.img2, self.img3 = img1, img2, img3
        self.obj, self.data = (
            (img1, img2, img3),
            [-1 + 2 * img1.data, -1 + 2 * img2.data, -1 + 2 * img3.data],
        )
        self.data2 = [-1 + 2 * img3.data, -1 + 2 * img2.data, -1 + 2 * img1.data]
        self.shape = img1.shape

    def apply_tfms(self, tfms, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms, **kwargs)
        self.img2 = self.img2.apply_tfms(tfms, **kwargs)
        self.img3 = self.img3.apply_tfms(tfms, **kwargs)
        return self

    def to_one(self):
        return Image(
            0.5
            + torch.cat(
                [
                    self.data[0][:3, :, :],
                    self.data[1][:3, :, :],
                    self.data[2][:3, :, :],
                ],
                2,
            )
            / 2
        )

    def to_one_pred(self):
        return Image(0.5 + (self.data2[0]) / 2)

    def __repr__(self):
        return f"{self.__class__.__name__}{(self.img1.shape, self.img2.shape, self.img3.shape)}"


class TargetTupleList(ItemList):
    def reconstruct(self, t: Tensor):
        if len(t.size()) == 0:
            return t
        return ImageTuple(
            Image(t[0] / 2 + 0.5), Image(t[1] / 2 + 0.5), Image(t[2] / 2 + 0.5)
        )


_batch_stats_a = None
_batch_stats_b = None
_batch_stats_c = None


class ImageTupleListMS2(ArcGISImageList):
    _label_cls = TargetTupleList

    def __init__(
        self,
        items,
        itemsB=None,
        itemsC=None,
        itemsB_valid=None,
        itemsC_valid=None,
        **kwargs,
    ):
        self.itemsB = itemsB
        self.itemsC = itemsC
        self.itemsB_valid = itemsB_valid
        self.itemsC_valid = itemsC_valid
        super().__init__(items, **kwargs)

    def new(self, items, **kwargs):
        return super().new(
            items,
            itemsB=self.itemsB,
            itemsC=self.itemsC,
            itemsB_valid=self.itemsB_valid,
            itemsC_valid=self.itemsC_valid,
            **kwargs,
        )

    def get(self, i):
        if len(self.items) == len(self.itemsB):
            img1 = super().get(i)
            fn1 = self.itemsB[i]
            img2 = ArcGISMSImage.open(fn1)
            fn = self.itemsC[i]
            img3 = ArcGISMSImage.open(fn)

            max_of = np.max([img1.shape[0], img2.shape[0], img3.shape[0]])

            if max_of != img1.shape[0]:
                cont = []
                last_tile = np.expand_dims(img1.data[img1.shape[0] - 1, :, :], 0)
                res = abs(max_of - img1.shape[0])
                for i in range(res):
                    img1 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img1.data, last_tile), axis=0))
                    )
            if max_of != img2.shape[0]:
                cont = []
                last_tile = np.expand_dims(img2.data[img2.shape[0] - 1, :, :], 0)
                res = abs(max_of - img2.shape[0])
                for i in range(res):
                    img2 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img2.data, last_tile), axis=0))
                    )
            if max_of != img3.shape[0]:
                cont = []
                last_tile = np.expand_dims(img3.data[img3.shape[0] - 1, :, :], 0)
                res = abs(max_of - img3.shape[0])
                for i in range(res):
                    img3 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img3.data, last_tile), axis=0))
                    )
            if max_of == 1:
                last_tile1, last_tile2, last_tile3 = (
                    np.expand_dims(img1.data[0, :, :], 0),
                    np.expand_dims(img2.data[0, :, :], 0),
                    np.expand_dims(img3.data[0, :, :], 0),
                )
                for i in range(2):
                    img1 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img1.data, last_tile1), axis=0))
                    )
                    img2 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img2.data, last_tile2), axis=0))
                    )
                    img3 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img3.data, last_tile3), axis=0))
                    )
        else:
            img1 = super().get(i)
            fn1 = self.itemsB_valid[i]
            img2 = ArcGISMSImage.open(fn1)
            fn = self.itemsC_valid[i]
            img3 = ArcGISMSImage.open(fn)
            max_of = np.max([img1.shape[0], img2.shape[0], img3.shape[0]])

            if max_of != img1.shape[0]:
                cont = []
                last_tile = np.expand_dims(img1.data[img1.shape[0] - 1, :, :], 0)
                res = abs(max_of - img1.shape[0])
                for i in range(res):
                    img1 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img1.data, last_tile), axis=0))
                    )
            if max_of != img2.shape[0]:
                cont = []
                last_tile = np.expand_dims(img2.data[img2.shape[0] - 1, :, :], 0)
                res = abs(max_of - img2.shape[0])
                for i in range(res):
                    img2 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img2.data, last_tile), axis=0))
                    )
            if max_of != img3.shape[0]:
                cont = []
                last_tile = np.expand_dims(img3.data[img3.shape[0] - 1, :, :], 0)
                res = abs(max_of - img3.shape[0])
                for i in range(res):
                    img3 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img3.data, last_tile), axis=0))
                    )
            if max_of == 1:
                last_tile1, last_tile2, last_tile3 = (
                    np.expand_dims(img1.data[0, :, :], 0),
                    np.expand_dims(img2.data[0, :, :], 0),
                    np.expand_dims(img3.data[0, :, :], 0),
                )
                for i in range(2):
                    img1 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img1.data, last_tile1), axis=0))
                    )
                    img2 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img2.data, last_tile2), axis=0))
                    )
                    img3 = ArcGISMSImage(
                        torch.tensor(np.concatenate((img3.data, last_tile3), axis=0))
                    )

        global _batch_stats_a
        global _batch_stats_b
        global _batch_stats_c

        img1_scaled = _tensor_scaler_tfm(
            img1.data,
            min_values=_batch_stats_a["band_min_values"],
            max_values=_batch_stats_a["band_max_values"],
            mode="minmax",
        )
        img2_scaled = _tensor_scaler_tfm(
            img2.data,
            min_values=_batch_stats_b["band_min_values"],
            max_values=_batch_stats_b["band_max_values"],
            mode="minmax",
        )
        img3_scaled = _tensor_scaler_tfm(
            img3.data,
            min_values=_batch_stats_c["band_min_values"],
            max_values=_batch_stats_c["band_max_values"],
            mode="minmax",
        )

        self.img1_scaled = ArcGISMSImage(img1_scaled)
        self.img2_scaled = ArcGISMSImage(img2_scaled)
        self.img3_scaled = ArcGISMSImage(img3_scaled)

        return ImageTuple(self.img1_scaled, self.img2_scaled, self.img3_scaled)

    def show(self, i, axes=None, rgb_bands=None):
        if rgb_bands is None:
            rgb_bands = [0, 1, 2]
        self[i]
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(15, 7))

        self.img1_scaled.show(axes[0], rgb_bands=rgb_bands)
        self.img2_scaled.show(axes[1], rgb_bands=rgb_bands)
        self.img3_scaled.show(axes[2], rgb_bands=rgb_bands)

    def reconstruct(self, t: Tensor):
        if len(t) == 3:
            return ImageTuple(
                ArcGISMSImage(t[0] / 2 + 0.5),
                ArcGISMSImage(t[1] / 2 + 0.5),
                ArcGISMSImage(t[2] / 2 + 0.5),
            )
        else:
            return ImageTuple(
                ArcGISMSImage(t[0] / 2 + 0.5),
                ArcGISMSImage(t[1] / 2 + 0.5),
                ArcGISMSImage(t[1] / 2 + 0.5),
            )

    @classmethod
    def from_folders(
        cls,
        path,
        folderA,
        folderB,
        folderC,
        batch_stats_a,
        batch_stats_b,
        batch_stats_c,
        **kwargs,
    ):
        itemsB = ImageList.from_folder(folderB).items
        itemsC = ImageList.from_folder(folderC).items
        res = super().from_folder(
            folderA,
            itemsB=itemsB,
            itemsC=itemsC,
            itemsB_valid=itemsB,
            itemsC_valid=itemsC,
            **kwargs,
        )
        res.path = path

        global _batch_stats_a
        global _batch_stats_b
        global _batch_stats_c

        _batch_stats_a = batch_stats_a
        _batch_stats_b = batch_stats_b
        _batch_stats_c = batch_stats_c
        return res

    def split_by_idxs(self, train_idx, valid_idx):
        "Split the data between `train_idx` and `valid_idx`."

        train_idx_alt = []
        for i in range(train_idx.shape[0]):
            if train_idx[i] < len(self.itemsB):
                train_idx_alt.append(train_idx[i])

        valid_idx_alt = []
        for j in range(valid_idx.shape[0]):
            if valid_idx[j] < len(self.itemsB):
                valid_idx_alt.append(valid_idx[j])

        self.itemsB_valid = self.itemsB_valid[valid_idx_alt]
        self.itemsC_valid = self.itemsC_valid[valid_idx_alt]
        self.itemsB = self.itemsB[train_idx_alt]
        self.itemsC = self.itemsC[train_idx_alt]
        return self.split_by_list(self[train_idx_alt], self[valid_idx_alt])

    def show_xys(self, xs, ys, figsize: Tuple[int, int] = (12, 6), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."

        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows, rows, figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize: Tuple[int, int] = None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""

        figsize = ifnone(figsize, (12, 3 * len(xs)))
        fig, axs = plt.subplots(len(xs), 2, figsize=figsize)
        if axs.ndim == 1:  # fix for rows=1
            axs = axs.reshape(1, 2)
        fig.suptitle("Images / Ground truth / Predictions", weight="bold", size=14)
        for i, (x, z) in enumerate(zip(xs, zs)):
            x.to_one().show(ax=axs[i, 0], **kwargs)
            z.to_one_pred().show(ax=axs[i, 1], **kwargs)


def calculate_activation_statistics(batch_size, data_len, batch_list):
    act = get_activations(batch_size, data_len, batch_list)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _tensor_scaler_tfm(tensor_batch, min_values, max_values, mode="minmax"):
    from .._data import _tensor_scaler

    x = tensor_batch
    if x.shape[0] > min_values.shape[0]:
        res = x.shape[0] - min_values.shape[0]
        last_val = torch.tensor([min_values[min_values.shape[0] - 1]])
        for i in range(res):
            min_values = torch.tensor(np.concatenate((min_values, last_val), axis=0))

    if x.shape[0] > max_values.shape[0]:
        res = x.shape[0] - max_values.shape[0]
        last_val = torch.tensor([max_values[max_values.shape[0] - 1]])
        for i in range(res):
            max_values = torch.tensor(np.concatenate((max_values, last_val), axis=0))
    max_values = max_values.view(-1, 1, 1).to(x.device)
    min_values = min_values.view(-1, 1, 1).to(x.device)
    x = _tensor_scaler(x, min_values, max_values, mode, create_view=False)
    return x


def _batch_stats_json(
    path, img_list, norm_pct, stats_file_name="esri_normalization_stats.json"
):
    from .._data import _get_batch_stats

    if len(img_list) < 300:
        norm_pct = 1

    dummy_stats = {
        "batch_stats_for_norm_pct_0": {
            "band_min_values": None,
            "band_max_values": None,
            "band_mean_values": None,
            "band_std_values": None,
            "scaled_min_values": None,
            "scaled_max_values": None,
            "scaled_mean_values": None,
            "scaled_std_values": None,
        }
    }

    path_save = Path(os.path.split(path)[0])
    normstats_json_path = os.path.abspath(path_save / ".." / stats_file_name)

    if not os.path.exists(normstats_json_path):
        normstats = dummy_stats
        with open(normstats_json_path, "w", encoding="utf-8") as f:
            json.dump(normstats, f, ensure_ascii=False, indent=4)
    else:
        with open(normstats_json_path) as f:
            normstats = json.load(f)

    emd_path = Path(os.path.abspath(path_save / "esri_model_definition.emd"))
    with open(emd_path) as f:
        emd_stats = json.load(f)

    if stats_file_name == "esri_normalization_stats_a.json":
        domain_stats = emd_stats.get("AllTilesStats")
    elif stats_file_name == "esri_normalization_stats_b.json":
        domain_stats = emd_stats.get("AllTilesStats")
    else:
        domain_stats = emd_stats.get("AllTilesStats2")

    mini, maxi, mean, std = [], [], [], []
    for i in domain_stats:
        mini.append(i.get("Min"))
        maxi.append(i.get("Max"))
        mean.append(i.get("Mean"))
        std.append(i.get("StdDev"))

    data_stats = [mini, maxi, mean, std]
    data_stats_tensors = [
        torch.tensor(mini),
        torch.tensor(maxi),
        torch.tensor(mean),
        torch.tensor(std),
    ]

    norm_pct_search = f"batch_stats_for_norm_pct_{round(norm_pct*100)}"
    if norm_pct_search in normstats:
        batch_stats = normstats[norm_pct_search]
        for l, s in enumerate(batch_stats):
            if l == len(data_stats):
                break
            batch_stats[s] = data_stats[l]

        for s in batch_stats:
            if batch_stats[s] is not None:
                batch_stats[s] = torch.tensor(batch_stats[s])
    else:
        batch_stats = {
            "band_min_values": None,
            "band_max_values": None,
            "band_mean_values": None,
            "band_std_values": None,
            "scaled_min_values": None,
            "scaled_max_values": None,
            "scaled_mean_values": None,
            "scaled_std_values": None,
        }
        for l, s in enumerate(batch_stats):
            if l == len(data_stats):
                break
            batch_stats[s] = data_stats[l]

        for s in batch_stats:
            if batch_stats[s] is not None:
                batch_stats[s] = torch.Tensor(batch_stats[s])
        normstats[norm_pct_search] = dict(batch_stats)
        for s in normstats[norm_pct_search]:
            if normstats[norm_pct_search][s] is not None:
                normstats[norm_pct_search][s] = normstats[norm_pct_search][s].tolist()
        with open(normstats_json_path, "w", encoding="utf-8") as f:
            json.dump(normstats, f, ensure_ascii=False, indent=4)

    return batch_stats


def prepare_data_wnetcgan(path, norm_pct, val_split_pct, seed, databunch_kwargs):
    path_a = path / "train_A_C" / "images"
    path_c = path / "train_A_C" / "images2"
    path_b = path / "train_B" / "images"

    img_list_a = ArcGISImageList.from_folder(path_a)
    img_list_b = ArcGISImageList.from_folder(path_b)
    img_list_c = ArcGISImageList.from_folder(path_c)

    batch_stats_a = _batch_stats_json(
        path_a, img_list_a, norm_pct, stats_file_name="esri_normalization_stats_a.json"
    )
    batch_stats_b = _batch_stats_json(
        path_b, img_list_b, norm_pct, stats_file_name="esri_normalization_stats_b.json"
    )
    batch_stats_c = _batch_stats_json(
        path_c, img_list_c, norm_pct, stats_file_name="esri_normalization_stats_c.json"
    )

    data = (
        ImageTupleListMS2.from_folders(
            path,
            path_a,
            path_b,
            path_c,
            batch_stats_a,
            batch_stats_b,
            batch_stats_c,
        )
        .split_by_rand_pct(val_split_pct, seed=seed)
        .label_empty()
        .databunch(**databunch_kwargs)
    )

    data._band_min_values = batch_stats_a["band_min_values"]
    data._band_max_values = batch_stats_a["band_max_values"]
    data._band_mean_values = batch_stats_a["band_mean_values"]
    data._band_std_values = batch_stats_a["band_std_values"]
    data._scaled_min_values = batch_stats_a["scaled_min_values"]
    data._scaled_max_values = batch_stats_a["scaled_max_values"]
    data._scaled_mean_values = batch_stats_a["scaled_mean_values"]
    data._scaled_std_values = batch_stats_a["scaled_std_values"]

    data._band_min_values_b = batch_stats_b["band_min_values"]
    data._band_max_values_b = batch_stats_b["band_max_values"]
    data._band_mean_values_b = batch_stats_b["band_mean_values"]
    data._band_std_values_b = batch_stats_b["band_std_values"]
    data._scaled_min_values_b = batch_stats_b["scaled_min_values"]
    data._scaled_max_values_b = batch_stats_b["scaled_max_values"]
    data._scaled_mean_values_b = batch_stats_b["scaled_mean_values"]
    data._scaled_std_values_b = batch_stats_b["scaled_std_values"]

    data._band_min_values_c = batch_stats_c["band_min_values"]
    data._band_max_values_c = batch_stats_c["band_max_values"]
    data._band_mean_values_c = batch_stats_c["band_mean_values"]
    data._band_std_values_c = batch_stats_c["band_std_values"]
    data._scaled_min_values_c = batch_stats_c["scaled_min_values"]
    data._scaled_max_values_c = batch_stats_c["scaled_max_values"]
    data._scaled_mean_values_c = batch_stats_c["scaled_mean_values"]
    data._scaled_std_values_c = batch_stats_c["scaled_std_values"]

    # add dataset_type
    data._dataset_type = "WNet_cGAN"
    return data


def display_row(axes, display, rgb_bands=None):
    if rgb_bands is None:
        rgb_bands = [0, 1, 2]
    for i, ax in enumerate(axes):
        if type(display[i]) is ArcGISMSImage:
            display[i].show(ax, rgb_bands)
        else:
            ax.imshow(display[i])
        ax.axis("off")


def show_batch_wnet(self, rows=4, **kwargs):
    rgb_bands = kwargs.get("rgb_bands", None)
    fig, axes = plt.subplots(nrows=rows, ncols=3, squeeze=False, figsize=(20, rows * 5))
    top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
    plt.subplots_adjust(top=top)
    fig.suptitle("Input 1 / Input 2 / Label", fontsize=16)
    img_idxs = [random.randint(0, len(self.train_ds) - 1) for k in range(rows)]
    for idx, im_idx in enumerate(img_idxs):
        self.train_ds.show(im_idx, axes[idx], rgb_bands)


def show_results(self, rows):
    self.learn.model.eval()
    x_batch, y_batch = get_nbatches(
        self._data.valid_dl, math.ceil(rows / self._data.batch_size)
    )

    x_A, x_B, x_C = (
        [x[0] for x in x_batch],
        [x[1] for x in x_batch],
        [x[2] for x in x_batch],
    )

    x_A = torch.cat(x_A)
    x_B = torch.cat(x_B)
    x_C = torch.cat(x_C)

    top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
    activ = []

    for i in range(0, x_B.shape[0], self._data.batch_size):
        with torch.no_grad():
            preds = self.learn.model(
                x_A[i : i + self._data.batch_size].detach(),
                x_B[i : i + self._data.batch_size].detach(),
                x_C[i : i + self._data.batch_size].detach(),
            )
        if self._data.nband_c == 1:
            all_batchs = []
            for batc in range(preds[0].shape[0]):
                first_pred = preds[0][batc, 1, :, :]
                real_preds = np.array(
                    [first_pred.clone().cpu().numpy()] * self._data.n_channel
                )
                all_batchs.append(real_preds)
            preds_ = torch.tensor(all_batchs).cuda()
            activ.append(preds_)
        else:
            activ.append(preds[0])

    activations = torch.cat(activ)
    if activations.shape[0] == x_A.shape[0]:
        x_A = 0.5 + x_A.cpu() / 2
        x_B = 0.5 + x_B.cpu() / 2
        x_C = 0.5 + x_C.cpu() / 2
        activations = 0.5 + activations.cpu() / 2

        rows = min(rows, x_A.shape[0])
        fig, axs = plt.subplots(
            nrows=rows, ncols=4, figsize=(4 * 5, rows * 5), squeeze=False
        )
        plt.subplots_adjust(top=top)
        fig.suptitle("Input 1 / Input 2 / Label / Predictions")
        for r in range(rows):
            if self._data._is_multispectral:
                display_row(
                    axs[r],
                    (
                        ArcGISMSImage(x_A[r]),
                        ArcGISMSImage(x_B[r]),
                        ArcGISMSImage(x_C[r]),
                        ArcGISMSImage(activations[r]),
                    ),
                )
            else:
                display_row(
                    axs[r],
                    (
                        image2np(x_A[r]),
                        image2np(x_B[r]),
                        image2np(x_C[r]),
                        image2np(activations[r]),
                    ),
                )
