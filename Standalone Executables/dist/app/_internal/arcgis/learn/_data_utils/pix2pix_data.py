import os, random, sys, math
import json
import numpy as np
from pathlib import Path
from fastai.data_block import get_files as gf
import torch
from fastai.vision import (
    open_mask,
    get_transforms,
    plt,
    Image,
    ImageSegment,
    image2np,
)
from torch.utils.data import Dataset, DataLoader
from fastai.data_block import DataBunch
from fastai.vision.image import _resolve_tfms
import arcgis
from .._utils.common import (
    get_nbatches,
    get_top_padding,
    ArcGISMSImage,
    ArcGISImageList,
)
import types
from functools import partial
from .._data import _prepare_working_dir
from .._utils.cyclegan import image_extensions
from .._data import _tensor_scaler
from .._utils.superres import show_batch
import warnings

stats = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]


def get_mask_map(img_paths):
    img_list = []
    for img_path in img_paths:
        img = open_mask(img_path)
        img = img.data.numpy()
        img_list.append(np.unique(img))
    img_list = np.concatenate(img_list)
    img_list = np.sort(img_list)
    img_list = np.unique(img_list)
    return img_list


def rescale_mask(mask, map):
    mask = mask.data
    for i, j in enumerate(map):
        mask[mask == j] = i
    mask = mask.float()
    return ImageSegment(mask)


def get_device():
    if getattr(arcgis.env, "_processorType", "") == "GPU" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(arcgis.env, "_processorType", "") == "CPU":
        device = torch.device("cpu")
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    return device


def is_old_format_pix2pix(path):
    """
    Function that returns 'True' if the manually created dir structure is being used.
    """
    return os.path.exists(path / "Images" / "train_a") and os.path.exists(
        path / "Images" / "train_b"
    )


def is_thematic_data(path):
    """
    Function that returns 'True' if thematic labels are present instead of rgb as source image
    """
    return os.path.exists(path / "labels") and os.path.exists(path / "images")


def pix2pix_paths(path):
    """
    Function to return the appropriate image dir paths in the provided dataset dir.
    """
    if is_old_format_pix2pix(path):
        return (path / "Images" / "train_a", path / "Images" / "train_b")
    elif is_thematic_data(path):  # only works with Pix2PixHD
        return (path / "labels", path / "images")
    else:
        return (path / "images", path / "images2")


def folder_check_pix2pix(path):
    """
    Function to check if the correct dir structure is provided.
    """
    img_folder1 = os.path.exists(path / "Images" / "train_a") or os.path.exists(
        path / "images"
    )
    img_folder2 = (
        os.path.exists(path / "Images" / "train_b")
        or os.path.exists(path / "images2")
        or os.path.exists(path / "labels")
    )
    if not all([img_folder1, img_folder2]):
        raise Exception(
            f"""You might be using an incorrect format to train your model. \nPlease ensure your training data has the following folder structure:
                ├─dataset folder name
                    ├─images
                    ├─images2   """
        )


def _tensor_scaler_tfm(tensor_batch, min_values, max_values, mode="minmax"):
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


def _norm_stats(path):
    emd_path = Path(os.path.abspath(path / "esri_model_definition.emd"))
    with open(emd_path) as f:
        emd_stats = json.load(f)

    domain_stats1 = emd_stats.get("AllTilesStats")
    domain_stats2 = emd_stats.get("AllTilesStats2", domain_stats1)

    data_stats = [domain_stats1, domain_stats2]
    batch_stats = [
        {
            "band_min_values": torch.tensor([i.get("Min") for i in stats]),
            "band_max_values": torch.tensor([i.get("Max") for i in stats]),
            "band_mean_values": torch.tensor([i.get("Mean") for i in stats]),
            "band_std_values": torch.tensor([i.get("StdDev") for i in stats]),
            "scaled_min_values": None,
            "scaled_max_values": None,
            "scaled_mean_values": None,
            "scaled_std_values": None,
        }
        for stats in data_stats
    ]
    return batch_stats[0], batch_stats[1]


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

    normstats_json_path = os.path.abspath(path / stats_file_name)

    if not os.path.exists(normstats_json_path):
        normstats = dummy_stats
        with open(normstats_json_path, "w", encoding="utf-8") as f:
            json.dump(normstats, f, ensure_ascii=False, indent=4)
    else:
        with open(normstats_json_path) as f:
            normstats = json.load(f)

    emd_path = Path(os.path.abspath(path / "esri_model_definition.emd"))
    with open(emd_path) as f:
        emd_stats = json.load(f)
    if "AllTilesStats2" not in emd_stats:
        emd_stats = None
    else:
        if stats_file_name == "esri_normalization_stats_a.json":
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
        if emd_stats is not None:
            for l, s in enumerate(batch_stats):
                if l == len(data_stats):
                    break
                batch_stats[s] = data_stats[l]

            for s in batch_stats:
                if batch_stats[s] is not None:
                    batch_stats[s] = torch.tensor(batch_stats[s])
        else:
            for s in batch_stats:
                if batch_stats[s] is not None:
                    batch_stats[s] = torch.tensor(batch_stats[s])

    else:
        if emd_stats is not None:
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
        else:
            batch_stats = _get_batch_stats(img_list, norm_pct)
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


def multispectral_additions(data):
    # batch_stats_a = _batch_stats_json(data.path, data.x_a, norm_pct, stats_file_name="esri_normalization_stats_a.json")
    # batch_stats_b = _batch_stats_json(data.path, data.x_b, norm_pct, stats_file_name="esri_normalization_stats_b.json")

    data._band_min_values = data.batch_stats_a["band_min_values"]
    data._band_max_values = data.batch_stats_a["band_max_values"]
    data._band_mean_values = data.batch_stats_a["band_mean_values"]
    data._band_std_values = data.batch_stats_a["band_std_values"]
    data._scaled_min_values = data.batch_stats_a["scaled_min_values"]
    data._scaled_max_values = data.batch_stats_a["scaled_max_values"]
    data._scaled_mean_values = data.batch_stats_a["scaled_mean_values"]
    data._scaled_std_values = data.batch_stats_a["scaled_std_values"]

    data._band_min_values_b = data.batch_stats_b["band_min_values"]
    data._band_max_values_b = data.batch_stats_b["band_max_values"]
    data._band_mean_values_b = data.batch_stats_b["band_mean_values"]
    data._band_std_values_b = data.batch_stats_b["band_std_values"]
    data._scaled_min_values_b = data.batch_stats_b["scaled_min_values"]
    data._scaled_max_values_b = data.batch_stats_b["scaled_max_values"]
    data._scaled_mean_values_b = data.batch_stats_b["scaled_mean_values"]
    data._scaled_std_values_b = data.batch_stats_b["scaled_std_values"]


def get_files(*args, **kwargs):
    return sorted(gf(*args, **kwargs))


def apply_tfms(images, other_tfms, resize_to):
    image_A, image_B = images
    # To fix when the disk image size is smaller than
    # crop size, otherwise crop_tfm behaves strange    if min(image_B.shape[1:]) / size < 1:
    #         image_A = image_A.resize(size)
    #         image_B = image_B.resize(size
    # )
    # image_inst = image_inst.resize(size)
    image_A = image_A.resize((3, resize_to, resize_to))
    image_B = image_B.resize((3, resize_to, resize_to))

    image_A = image_A.apply_tfms(other_tfms, do_resolve=False)

    image_B = image_B.apply_tfms(other_tfms, do_resolve=False)
    return image_A, image_B


def concat_bands(image_A, image_B):
    if image_A.shape[0] < image_B.shape[0]:
        cont = []
        last_tile = np.expand_dims(image_A.data[image_A.shape[0] - 1, :, :], 0)
        res = abs(image_B.shape[0] - image_A.shape[0])
        for i in range(res):
            image_A = ArcGISMSImage(
                torch.tensor(np.concatenate((image_A.data, last_tile), axis=0))
            )
    if image_B.shape[0] < image_A.shape[0]:
        cont = []
        last_tile = np.expand_dims(image_B.data[image_B.shape[0] - 1, :, :], 0)
        res = abs(image_A.shape[0] - image_B.shape[0])
        for i in range(res):
            image_B = ArcGISMSImage(
                torch.tensor(np.concatenate((image_B.data, last_tile), axis=0))
            )

    return image_A, image_B


def _get_transforms(transforms, model=None):
    if transforms is None:
        if model:
            transforms = get_transforms(
                do_flip=True,
                flip_vert=True,
                max_rotate=90.0,
                max_zoom=0,
                max_lighting=None,
                max_warp=None,
                p_affine=0.75,
                xtra_tfms=None,
            )
        else:
            transforms = get_transforms(flip_vert=True, max_lighting=0.3, max_warp=0.0)
    elif transforms is False:
        transforms = ([], [])

    return transforms


# images_A = labels
# images_B = rgb_imgs
# images_inst = instance_labels


class Pix2PixHDDataset(Dataset):
    def __init__(
        self,
        path,
        image_list_A,
        image_list_B,
        resize_to,
        transforms=None,
        norm_stats=None,
        flip_vert=True,
        split=None,
        label_nc=False,
        _is_multispectral=False,
        imagery_type="RGB",
        batch_stats_a=None,
        batch_stats_b=None,
        rgb_bands=[0, 1, 2],
    ):
        self.path = Path(path)
        self.image_list_A = image_list_A
        self.image_list_B = image_list_B
        # self.image_list_inst = image_list_inst

        self.train_tfms, self.val_tfms = _get_transforms(transforms)
        self.split = split
        self.norm_stats = norm_stats
        self.label_nc = label_nc
        self.x_a = ArcGISImageList(image_list_A)
        self.x_b = ArcGISImageList(image_list_B)
        if resize_to is None:
            resize_to = self.x_a[0].data.shape[-1]
        if label_nc:  # fix for https://github.com/NVIDIA/pix2pixHD/issues/100
            if resize_to % 32 != 0:
                resize_to = resize_to - resize_to % 32
        self.resize_to = resize_to
        self.y = 0

        # MS
        self._is_multispectral = _is_multispectral
        self.n_channel = (
            self.x_a[0].shape[0]
            if self.x_a[0].shape[0] > self.x_b[0].shape[0]
            else self.x_b[0].shape[0]
        )
        if self._is_multispectral:
            self.norm_stats = [[0.5] * self.n_channel, [0.5] * self.n_channel]
        self.batch_stats_a = batch_stats_a
        self.batch_stats_b = batch_stats_b
        self.imagery_type = imagery_type
        self.rgb_bands = rgb_bands
        if label_nc:
            self.mask_map = get_mask_map(image_list_A)

    def __len__(self):
        return len(self.image_list_B)

    def __getitem__(self, idx):
        if os.path.isdir(self.path / "labels"):
            self.image_list_B = [
                Path(
                    os.path.join(
                        os.path.split(i)[0].replace("images2", "labels"),
                        os.path.split(i)[1],
                    )
                )
                for i in self.image_list_B
            ]
        if self._is_multispectral:
            # to add mask loading
            image_A = ArcGISMSImage.open(
                self.image_list_A[idx], imagery_type=self.imagery_type
            )
            image_B = ArcGISMSImage.open(
                self.image_list_B[idx], imagery_type=self.imagery_type
            )

            image_A, image_B = concat_bands(image_A, image_B)

            image_A = _tensor_scaler_tfm(
                image_A.data,
                min_values=self.batch_stats_a["band_min_values"],
                max_values=self.batch_stats_a["band_max_values"],
                mode="minmax",
            )
            image_B = _tensor_scaler_tfm(
                image_B.data,
                min_values=self.batch_stats_b["band_min_values"],
                max_values=self.batch_stats_b["band_max_values"],
                mode="minmax",
            )

            image_A = ArcGISMSImage(image_A)
            image_B = ArcGISMSImage(image_B)

        else:
            min_vals_a = self.batch_stats_a["band_min_values"].tolist()
            max_vals_a = self.batch_stats_a["band_max_values"].tolist()
            min_vals_b = self.batch_stats_b["band_min_values"].tolist()
            max_vals_b = self.batch_stats_b["band_max_values"].tolist()
            rgb_mins_a, rgb_maxs_a = (
                [float(i) for i in min_vals_a],
                [float(i) for i in max_vals_a],
            )
            rgb_mins_b, rgb_maxs_b = (
                [float(i) for i in min_vals_b],
                [float(i) for i in max_vals_b],
            )
            if self.label_nc == False:
                image_A = ArcGISMSImage.open(
                    self.image_list_A[idx],
                    imagery_type=self.imagery_type,
                    div=(rgb_mins_a, rgb_maxs_a),
                )
            else:
                image_A = ImageSegment(ArcGISMSImage.open(self.image_list_A[idx]).data)
                image_A = rescale_mask(image_A, self.mask_map)

            image_B = ArcGISMSImage.open(
                self.image_list_B[idx],
                imagery_type=self.imagery_type,
                div=(rgb_mins_b, rgb_maxs_b),
            )

        _resolve_tfms(self.train_tfms)
        _resolve_tfms(self.val_tfms)

        images = (image_A, image_B)
        if self.split == "train":
            images = apply_tfms(images, self.train_tfms, self.resize_to)
        else:
            images = apply_tfms(images, self.val_tfms, self.resize_to)
        image_A, image_B = images

        if self.label_nc == False:
            self.image_A = image_A
            image_A = normalize(image_A.px, *self.norm_stats)
            # image_A = -1+2*image_A.data
        else:
            self.image_A = image_A
            image_A = image_A.data

        self.image_B = image_B
        image_B = normalize(image_B.px, *self.norm_stats)
        return (image_A, image_B), self.y

    def __repr__(self):
        item = self.__getitem__(0)
        return f"{self.__class__.__name__}{(item[0][0].shape, item[0][1].shape)}, items_A:{len(self.image_list_A)}, items_B:{len(self.image_list_B)}"

    def show(self, idx, axes=None, rgb_bands=None):
        if rgb_bands is None:
            rgb_bands = self.rgb_bands
        self[idx]
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(15, 7))

        if self.label_nc == True:
            self.image_A.show(axes[0])
        else:
            self.image_A.show(axes[0], rgb_bands=rgb_bands)
        self.image_B.show(axes[1], rgb_bands=rgb_bands)


class SR3Dataset(Pix2PixHDDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_multispec = self._is_multispectral
        self._n_channel = self.x_a[0].shape[0]
        self.resize_to = self.x_b[0].data.shape[-1]
        self._image_stats, self._image_stats2 = None, None
        self.train_tfms, self.val_tfms = _get_transforms(transforms=None, model="SR3")

    def __getitem__(self, idx):
        min_vals_a = self.batch_stats_a["band_min_values"]
        max_vals_a = self.batch_stats_a["band_max_values"]
        min_vals_b = self.batch_stats_b["band_min_values"]
        max_vals_b = self.batch_stats_b["band_max_values"]
        if self._is_multispectral:
            # to add mask loading
            image_A = ArcGISMSImage.open(
                self.image_list_A[idx], imagery_type=self.imagery_type
            )
            image_B = ArcGISMSImage.open(
                self.image_list_B[idx], imagery_type=self.imagery_type
            )

            image_A, image_B = concat_bands(image_A, image_B)

            image_A = _tensor_scaler_tfm(
                image_A.data,
                min_values=min_vals_a,
                max_values=max_vals_a,
                mode="minmax",
            )
            image_B = _tensor_scaler_tfm(
                image_B.data,
                min_values=min_vals_b,
                max_values=max_vals_b,
                mode="minmax",
            )

            image_A = ArcGISMSImage(image_A)
            image_B = ArcGISMSImage(image_B)

        else:
            image_A = ArcGISMSImage.open(
                self.image_list_A[idx],
                imagery_type=self.imagery_type,
                div=255,
            )
            image_B = ArcGISMSImage.open(
                self.image_list_B[idx],
                imagery_type=self.imagery_type,
                div=255,
            )

        _resolve_tfms(self.train_tfms)
        _resolve_tfms(self.val_tfms)

        images = (image_A, image_B)
        if self.split == "train":
            images = apply_tfms(images, self.train_tfms, self.resize_to)
        else:
            images = apply_tfms(images, self.val_tfms, self.resize_to)
        image_A, image_B = images

        self.image_A = image_A
        image_A = -1 + 2 * image_A.px  # normalize(image_A.px, *self.norm_stats)
        image_A = image_A.data

        self.image_B = image_B
        image_B = -1 + 2 * image_B.px  # normalize(image_B.px, *self.norm_stats)
        image_B = image_B.data
        return (image_A, image_B), image_B

    def __repr__(self):
        item = self.__getitem__(0)
        return f"{self.__class__.__name__}{(item[0][0].shape, item[0][1].shape)}, items_A:{len(self.image_list_A)}, items_B:{len(self.image_list_B)}"


def create_train_val_sets(
    path,
    val_split_pct,
    transforms,
    resize_to,
    norm_stats,
    flip_vert,
    label_nc=False,
    _is_multispectral=False,
    rgb_bands=None,
    norm_pct=1,
    model_type="Pix2Pix",
    **kwargs,
):
    path_lr = kwargs.get("path_lr", None)
    path_hr = kwargs.get("path_hr", None)

    imagery_type = kwargs.get("imagery_type", "None")
    path = Path(path)
    if model_type == "Pix2Pix":
        path_A, path_B = pix2pix_paths(path)
    else:
        path_A, path_B = path_lr, path_hr

    images_A = get_files(path_A, extensions=image_extensions, recurse=True)

    images_B = get_files(path_B, extensions=image_extensions, recurse=True)

    batch_stats_a, batch_stats_b = None, None

    if model_type == "Pix2Pix":
        batch_stats_a = _batch_stats_json(
            path,
            ArcGISImageList(images_A),
            norm_pct,
            stats_file_name="esri_normalization_stats_a.json",
        )
        batch_stats_b = _batch_stats_json(
            path,
            ArcGISImageList(images_B),
            norm_pct,
            stats_file_name="esri_normalization_stats_b.json",
        )
    else:
        batch_stats_a, batch_stats_b = _norm_stats(path)

    total_num_images = len(images_B)
    val_num_images = int(total_num_images * val_split_pct)
    zipped = list(zip(images_A, images_B))

    random.shuffle(zipped)

    images_A, images_B = zip(*zipped)

    train_images_A, train_images_B = (
        images_A[val_num_images:],
        images_B[val_num_images:],
    )

    val_images_A, val_images_B = (images_A[:val_num_images], images_B[:val_num_images])

    if model_type == "Pix2Pix":
        datasetclass = Pix2PixHDDataset
    else:
        datasetclass = SR3Dataset

    train_dataset = datasetclass(
        path,
        train_images_A,
        train_images_B,
        transforms=transforms,
        resize_to=resize_to,
        norm_stats=norm_stats,
        flip_vert=flip_vert,
        split="train",
        label_nc=label_nc,
        _is_multispectral=_is_multispectral,
        imagery_type=imagery_type,
        batch_stats_a=batch_stats_a,
        batch_stats_b=batch_stats_b,
        rgb_bands=rgb_bands,
    )
    val_dataset = datasetclass(
        path,
        val_images_A,
        val_images_B,
        transforms=transforms,
        resize_to=resize_to,
        norm_stats=norm_stats,
        flip_vert=flip_vert,
        split="val",
        label_nc=label_nc,
        _is_multispectral=_is_multispectral,
        imagery_type=imagery_type,
        batch_stats_a=batch_stats_a,
        batch_stats_b=batch_stats_b,
        rgb_bands=rgb_bands,
    )

    return train_dataset, val_dataset


def create_dataloaders(datasets, batch_size, dataloader_kwargs):
    return [DataLoader(d, batch_size, **dataloader_kwargs) for d in datasets]


def prepare_pix2pix_data(
    path,
    batch_size,
    val_split_pct,
    transforms,
    resize_to,
    norm_pct,
    _is_multispectral,
    working_dir,
    seed,
    dataset_type,
    **kwargs,
):
    norm_stats = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # kwargs.get('norm_stats', stats)
    flip_vert = kwargs.get("imagery_type", "satellite") != "oriented"
    split = kwargs.get("split", "random")
    if dataset_type == "Pix2Pix":
        label_nc = is_thematic_data(path) or kwargs.get("label_nc", False)
    else:
        label_nc = False
    if label_nc:
        _is_multispectral = False
    imagery_type = kwargs.get("imagery_type")
    rgb_bands = kwargs.get("rgb_bands")
    datasets = create_train_val_sets(
        path,
        val_split_pct,
        transforms=transforms,
        resize_to=resize_to,
        norm_stats=norm_stats,
        flip_vert=flip_vert,
        label_nc=label_nc,
        _is_multispectral=_is_multispectral,
        rgb_bands=rgb_bands,
        norm_pct=norm_pct,
        model_type=dataset_type,
        **kwargs,
    )

    databunch_kwargs = (
        {"num_workers": 0}
        if sys.platform == "win32"
        else {"num_workers": os.cpu_count() - 4}
    )

    train_dl, valid_dl = create_dataloaders(datasets, batch_size, databunch_kwargs)
    device = get_device()
    data = DataBunch(train_dl, valid_dl, device=device)
    data.path = data.train_ds.path
    data._do_normalize = None
    data._imagery_type = imagery_type
    data._norm_pct = norm_pct
    # data.n_channel = data.train_ds[0][0][0].shape[0]
    multispectral_additions(data)
    if label_nc:
        data.label_nc = len(data.mask_map) + 1
    data.chip_size = data.resize_to
    if working_dir is not None:
        data.path = Path(os.path.abspath(working_dir))
    data._temp_folder = _prepare_working_dir(data.path)
    data.show_batch = types.MethodType(show_batch, data)
    data._dataset_type = dataset_type  # "Pix2Pix"
    data._downsampling_factor = kwargs.get("downsample_factor", None)
    data.val_split_pct = val_split_pct
    data.resize_to = resize_to
    data.seed = seed
    data._extract_bands = None

    return data


def show_batch(self, rows=4, **kwargs):
    """
    This function randomly picks a few training chips and visualizes them.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional int. Number of rows of results
                            to be displayed.
    =====================   ===========================================
    """
    rgb_bands = kwargs.get("rgb_bands", None)
    fig, axes = plt.subplots(nrows=rows, ncols=2, squeeze=False, figsize=(20, rows * 5))
    top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
    plt.subplots_adjust(top=top)
    # fig.suptitle("Input / Ground Truth", fontsize=16)
    axes[0, 0].title.set_text("Input")
    axes[0, 1].title.set_text("Target")
    img_idxs = [random.randint(0, len(self.train_ds) - 1) for k in range(rows)]
    for idx, im_idx in enumerate(img_idxs):
        self.train_ds.show(im_idx, axes[idx], rgb_bands)


def to_device(z, device):
    return torch.tensor(z).to(device)


def normalize(x, mean, std):
    "Normalize `x` with `mean` and `std`."
    to_tensor = partial(to_device, device=x.device)

    if type(mean[0]) is not torch.Tensor:
        mean = to_tensor(mean)
        std = to_tensor(std)
    return (x - mean[..., None, None]) / std[..., None, None]


def denormalize(x, mean, std, do_x=True):
    "Denormalize `x` with `mean` and `std`."
    to_tensor = partial(to_device, device=x.device)
    if type(mean[0]) is not torch.Tensor:
        mean = to_tensor(mean)
        std = to_tensor(std)
    return (
        x.cpu().float() * std[..., None, None] + mean[..., None, None]
        if do_x
        else x.cpu()
    )


def display_row(axes, display, rgb_bands=None):
    if rgb_bands is None:
        rgb_bands = [0, 1, 2]
    for i, ax in enumerate(axes):
        if type(display[i]) is ArcGISMSImage:
            display[i].show(ax, rgb_bands)
        else:
            ax.imshow(display[i])
        ax.axis("off")


def post_process(dists):
    if type(dists) is np.ndarray:
        dists = torch.from_numpy(dists).permute(2, 0, 1)
    return (dists > 1.0).long()


def show_results(self, rows, **kwargs):
    self.learn.model.eval()
    x_batch, y_batch = get_nbatches(
        self._data.valid_dl, math.ceil(rows / self._data.batch_size)
    )
    x_A, x_B = [x[0] for x in x_batch], [x[1] for x in x_batch]

    x_A = torch.cat(x_A)
    x_B = torch.cat(x_B)

    top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
    activ = []

    for i in range(0, x_B.shape[0], self._data.batch_size):
        with torch.no_grad():
            preds = self.learn.model(
                x_A[i : i + self._data.batch_size].detach(),
                x_B[i : i + self._data.batch_size].detach(),
            )
        activ.append(preds[0])

    activations = torch.cat(activ)
    if self._data.label_nc == 0:
        x_A = denormalize(x_A.cpu(), *self._data.norm_stats)
    x_B = denormalize(x_B.cpu(), *self._data.norm_stats)
    activations = denormalize(activations.cpu(), *self._data.norm_stats)
    rows = min(rows, x_A.shape[0])

    fig, axs = plt.subplots(
        nrows=rows, ncols=3, figsize=(4 * 5, rows * 5), squeeze=False
    )
    plt.subplots_adjust(top=top)
    axs[0, 0].title.set_text("Input")
    axs[0, 1].title.set_text("Target")
    axs[0, 2].title.set_text("Prediction")
    for r in range(rows):
        if self._data._is_multispectral:
            display_row(
                axs[r],
                (
                    ArcGISMSImage(x_A[r]),
                    ArcGISMSImage(x_B[r]),
                    ArcGISMSImage(activations[r]),
                ),
                kwargs.get("rgb_bands", None),
            )
        else:
            display_row(
                axs[r],
                (image2np(x_A[r]), image2np(x_B[r]), image2np(activations[r])),
            )


def predict(self, img_path):
    img_path = Path(img_path)
    if self._data.label_nc == 0:
        raw_img = ArcGISMSImage.open(
            img_path, imagery_type=self._data.imagery_type, div=255
        )
    else:
        raw_img = ImageSegment(ArcGISMSImage.open(img_path).data)
        raw_img = rescale_mask(raw_img, self._data.mask_map)
    raw_img = raw_img.resize(self._data.chip_size)

    if self._data._is_multispectral:
        n_band = self._data.n_channel
        if n_band > raw_img.shape[0]:
            cont = []
            last_tile = np.expand_dims(raw_img.data[raw_img.shape[0] - 1, :, :], 0)
            res = abs(n_band - raw_img.shape[0])
            for i in range(res):
                raw_img = Image(
                    torch.tensor(np.concatenate((raw_img.data, last_tile), axis=0))
                )

    if self._data.label_nc == 0:
        raw_img_tensor = normalize(raw_img.px, *self._data.norm_stats)
    else:
        raw_img_tensor = raw_img.px
    raw_img_tensor = raw_img_tensor[None].to(self._device)
    self.learn.model.eval()
    with torch.no_grad():
        prediction = (
            self.learn.model(raw_img_tensor, raw_img_tensor)[0].detach()[0].cpu()
        )

    pred_denorm = denormalize(prediction, *self._data.norm_stats)
    pred_denorm = ArcGISMSImage(pred_denorm)
    pred_denorm = pred_denorm.show()
    return pred_denorm


def rgb_or_ms(im_path):
    """
    Function that returns the imagery type (RGB or ms) of an image.
    """
    try:
        from osgeo import gdal

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = gdal.Open(im_path)
        if ds.RasterCount != 3 or ds.GetRasterBand(1).DataType != gdal.GDT_Byte:
            return "ms"
        else:
            return "RGB"
    except:
        return None
