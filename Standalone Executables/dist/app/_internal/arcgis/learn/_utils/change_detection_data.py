from pathlib import Path
from .env import raise_fastai_import_error
import traceback
import types
import sys
import os
import random
import mimetypes
from functools import partial
import math
import numpy as np
import json

try:
    import torch
    from fastai.vision import open_image, open_mask, image2np
    from fastai.vision.image import ImageSegment
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset, DataLoader
    from fastai.data_block import DataBunch
    from .pointcloud_data import get_device
    from fastai.vision.transform import get_transforms, crop
    from fastai.vision import ImageList
    from fastai.vision.image import _resolve_tfms
    from fastai.data_block import get_files as gf
    from .common import get_nbatches, get_top_padding
    from .common import ArcGISMSImage, ArcGISImageList
    from .._data import _extract_bands_tfm, _tensor_scaler, _tensor_scaler_tfm
    from .._data import _get_batch_stats, sniff_rgb_bands
    from .._data import _prepare_working_dir

    HAS_FASTAI = True
except ImportError:
    import_exception = traceback.format_exc()
    HAS_FASTAI = False


def remap_label(data, class_mapping):
    """
    class_mapping will be required in case of multiclasss
    change detection.
    """
    remapped = torch.zeros_like(data)
    no_change_value = 0
    # hardcoded
    remapped[data == no_change_value] = 1  # no change
    remapped[data != no_change_value] = -1  # change
    return remapped


def is_contiguous(class_values):
    flag = True
    for i in range(len(class_values)):
        if class_values[i] + 1 != class_values[i + 1]:
            flag = False
            break
    return flag


# SA_type
# crop_predict


def multispectral_additions(
    data, _is_multispectral, rgb_bands, bands, extract_bands, norm_pct, **kwargs
):
    # Normalize multispectral imagery by calculating stats
    json_file = data.path / "esri_model_definition.emd"
    if json_file.exists():
        with open(json_file) as f:
            emd = json.load(f)
        if "InputRastersProps" in emd:
            # Starting with ArcGIS Pro 2.7 and Python API for ArcGIS 1.9, the following multispectral kwargs have been
            # deprecated. This is done in favour of the newly added support for Imagery statistics and metadata in the
            # IA > Export Training data for Deep Learining GP Tool.
            #
            #   bands, rgb_bands, norm_pct,
            #
            data._emd = emd
            data._sensor_name = emd["InputRastersProps"]["SensorName"]
            bands = data._band_names = emd["InputRastersProps"]["BandNames"]
            # data._band_mapping = {i: k for i, k in enumerate(bands)}
            # data._band_mapping_reverse = {k: i for i, k in data._band_mapping.items()}
            data._nbands = len(data._band_names)
            band_min_values = []
            band_max_values = []
            band_mean_values = []
            band_std_values = []
            for band_stats in emd["AllTilesStats"]:
                band_min_values.append(band_stats["Min"])
                band_max_values.append(band_stats["Max"])
                band_mean_values.append(band_stats["Mean"])
                band_std_values.append(band_stats["StdDev"])

            data._rgb_bands = rgb_bands
            data._symbology_rgb_bands = rgb_bands

            data._band_min_values = torch.tensor(band_min_values, dtype=torch.float32)
            data._band_max_values = torch.tensor(band_max_values, dtype=torch.float32)
            data._band_mean_values = torch.tensor(band_mean_values, dtype=torch.float32)
            data._band_std_values = torch.tensor(band_std_values, dtype=torch.float32)
            data._scaled_min_values = torch.zeros((data._nbands,), dtype=torch.float32)
            data._scaled_max_values = torch.ones((data._nbands,), dtype=torch.float32)
            data._scaled_mean_values = _tensor_scaler(
                data._band_mean_values,
                min_values=data._band_min_values,
                max_values=data._band_max_values,
                mode="minmax",
            )

            data._scaled_std_values = (
                (data._band_std_values**2)
                * (data._scaled_mean_values / data._band_mean_values)
            ) ** 0.5

            # Handover to next section
            norm_pct = 1
            bands = data._band_names
            rgb_bands = symbology_rgb_bands = sniff_rgb_bands(data._band_names)
            if rgb_bands is None:
                rgb_bands = []
                if len(data._band_names) < 3:
                    symbology_rgb_bands = [0]  # Panchromatic
                else:
                    symbology_rgb_bands = [
                        0,
                        1,
                        2,
                    ]  # Case where could not find RGB in multiband imagery

    else:
        if len(data.train_ds) < 300:
            norm_pct = 1

        # Statistics
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
        normstats_json_path = os.path.abspath(
            data.path / "esri_normalization_stats.json"
        )
        if not os.path.exists(normstats_json_path):
            normstats = dummy_stats
            with open(normstats_json_path, "w", encoding="utf-8") as f:
                json.dump(normstats, f, ensure_ascii=False, indent=4)
        else:
            with open(normstats_json_path) as f:
                normstats = json.load(f)

        norm_pct_search = f"batch_stats_for_norm_pct_{round(norm_pct*100)}"
        if norm_pct_search in normstats:
            batch_stats = normstats[norm_pct_search]
            for s in batch_stats:
                if batch_stats[s] is not None:
                    batch_stats[s] = torch.tensor(batch_stats[s])
        else:
            batch_stats = _get_batch_stats(
                data.x, norm_pct, scaled_std=False, reshape=True
            )
            normstats[norm_pct_search] = dict(batch_stats)
            for s in normstats[norm_pct_search]:
                if normstats[norm_pct_search][s] is not None:
                    normstats[norm_pct_search][s] = normstats[norm_pct_search][
                        s
                    ].tolist()
            with open(normstats_json_path, "w", encoding="utf-8") as f:
                json.dump(normstats, f, ensure_ascii=False, indent=4)

        # batch_stats -> [band_min_values, band_max_values, band_mean_values, band_std_values, scaled_min_values, scaled_max_values, scaled_mean_values, scaled_std_values]
        data._band_min_values = batch_stats["band_min_values"]
        data._band_max_values = batch_stats["band_max_values"]
        data._band_mean_values = batch_stats["band_mean_values"]
        data._band_std_values = batch_stats["band_std_values"]
        data._scaled_min_values = batch_stats["scaled_min_values"]
        data._scaled_max_values = batch_stats["scaled_max_values"]
        data._scaled_mean_values = batch_stats["scaled_mean_values"]
        data._scaled_std_values = batch_stats["scaled_std_values"]

        # Prevent Divide by zeros
        data._band_max_values[data._band_min_values == data._band_max_values] += 1
        # data._scaled_std_values[data._scaled_std_values == 0]+=1e-02

    # Scaling
    data._min_max_scaler = partial(
        _tensor_scaler,
        min_values=data._band_min_values,
        max_values=data._band_max_values,
        mode="minmax",
    )
    data._min_max_scaler_tfm = partial(
        _tensor_scaler_tfm,
        min_values=data._band_min_values,
        max_values=data._band_max_values,
        mode="minmax",
    )
    # data.add_tfm(data._min_max_scaler_tfm)

    # Transforms
    def _scaling_tfm(x):
        # Scales Fastai Image Scaling | MS Image Values -> 0 - 1 range
        return x.__class__(data._min_max_scaler_tfm((x.data, None))[0][0])

    ## Fastai need tfm, order and resolve.
    class dummy:
        pass

    _scaling_tfm.tfm = dummy()
    _scaling_tfm.tfm.order = 0
    _scaling_tfm.resolve = dummy

    ## Scaling the images before applying any  other transform
    if getattr(data.train_ds, "train_tfms") is not None:
        data.train_ds.train_tfms = [_scaling_tfm] + data.train_ds.train_tfms
    else:
        data.train_ds.train_tfms = [_scaling_tfm]

    if getattr(data.valid_ds, "val_tfms") is not None:
        data.valid_ds.val_tfms = [_scaling_tfm] + data.valid_ds.val_tfms
    else:
        data.valid_ds.val_tfms = [_scaling_tfm]

    data._is_multispectral = _is_multispectral
    if data._is_multispectral:
        data._bands = bands
        data._norm_pct = norm_pct
        data._rgb_bands = rgb_bands
        data._symbology_rgb_bands = rgb_bands
        # Prepare unknown bands list if bands data is missing
        if data._bands is None:
            n_bands = data.x[0].data.shape[0]
            if n_bands == 1:  # Handle Pancromatic case
                data._bands = ["p"]
                data._symbology_rgb_bands = [0]
            else:
                data._bands = ["u" for i in range(n_bands)]
                if n_bands == 2:  # Handle Data with two channels
                    data._symbology_rgb_bands = [0]
        else:
            n_bands = len(bands)
        #
        if data._rgb_bands is None:
            data._rgb_bands = []

        #
        if data._symbology_rgb_bands is None:
            data._symbology_rgb_bands = [0, 1, 2][: min(n_bands, 3)]
        elif data._symbology_rgb_bands == []:
            data._symbology_rgb_bands = list(range(n_bands))[: min(n_bands, 3)]

        # Complete symbology rgb bands
        if len(data._bands) > 2 and len(data._symbology_rgb_bands) < 3:
            data._symbology_rgb_bands += [
                min(max(data._symbology_rgb_bands) + 1, len(data._bands) - 1)
                for i in range(3 - len(data._symbology_rgb_bands))
            ]

        # Overwrite band values at r g b indexes with 'r' 'g' 'b'
        for i, band_idx in enumerate(data._rgb_bands):
            if band_idx is not None:
                if data._bands[band_idx] == "u":
                    data._bands[band_idx] = ["r", "g", "b"][i]

        # # Attach custom show batch
        # if _show_batch_multispectral is not None:
        #     data.show_batch = types.MethodType( _show_batch_multispectral, data )

        # Apply filter band transformation if user has specified extract_bands otherwise add a generic extract_bands
        """
        extract_bands : List containing band indices of the bands from imagery on which the model would be trained.
                        Useful for benchmarking and applied training, for reference see examples below.

                        4 band naip ['r, 'g', 'b', 'nir'] + extract_bands=[0, 1, 2] -> 3 band naip with bands ['r', 'g', 'b']

        """

        data._extract_bands = extract_bands
        if data._extract_bands is None:
            data._extract_bands = list(range(len(data._bands)))
        else:
            data._extract_bands_tfm = partial(
                _extract_bands_tfm, band_indices=data._extract_bands
            )
            data.add_tfm(data._extract_bands_tfm)

        # Tail Training Override
        _train_tail = True
        if [data._bands[i] for i in data._extract_bands] == ["r", "g", "b"]:
            _train_tail = False
        data._train_tail = kwargs.get("train_tail", _train_tail)
    pass


class ChangeDetectionDataset(Dataset):
    def __init__(
        self,
        path,
        before_list,
        after_list,
        label_list,
        class_mapping,
        color_mapping,
        chip_size=224,
        transforms=None,
        flip_vert=True,
        split=None,
        norm_stats=None,
        _is_multispectral=False,
        rgb_bands=[0, 1, 2],
        imagery_type="RGB",
    ):
        """
        Initialize dataset with lists of files

        Args:
            before_list (list): list of previous images
            after_list (list): list of after images
            label_list (list): list of segmentation maps
        """
        self.path = Path(path)
        self.before_list = before_list
        self.after_list = after_list
        self.label_list = label_list
        # get transforms based on user passed transforms
        self.train_tfms, self.val_tfms = _get_transforms(
            transforms, flip_vert=flip_vert
        )

        self.split = split
        self.norm_stats = norm_stats
        self.class_mapping = class_mapping
        self.color_mapping = color_mapping
        self.x = ArcGISImageList(self.before_list + self.after_list)
        self.n_c = self.x[0].data.shape[0]

        # if transforms is False, chip_size will be equal to actual size.
        # no cropping will take place.
        if not transforms:
            chip_size = self.x[0].data.shape[1]

        crop_tfm = [crop(size=chip_size, row_pct=(0, 1), col_pct=(0, 1))]

        self.train_crop_tfm = crop_tfm
        # val crop tfm is just center crop.
        self.val_crop_tfm = [crop(size=chip_size, row_pct=0.5, col_pct=0.5)]
        self.chip_size = chip_size

        # MS
        self._is_multispectral = _is_multispectral
        if _is_multispectral:
            self.norm_stats = [[0.5] * self.n_c, [0.5] * self.n_c]
        self.rgb_bands = rgb_bands
        self.imagery_type = imagery_type

    def __len__(self):
        return len(self.before_list)

    def __getitem__(self, idx):
        if self._is_multispectral:
            image_before = ArcGISMSImage.open(
                self.before_list[idx], imagery_type=self.imagery_type
            )
            image_after = ArcGISMSImage.open(
                self.after_list[idx], imagery_type=self.imagery_type
            )
        else:
            image_before = ArcGISMSImage.open(
                self.before_list[idx], imagery_type=self.imagery_type, div=255
            )
            image_after = ArcGISMSImage.open(
                self.after_list[idx], imagery_type=self.imagery_type, div=255
            )
        change_label = ImageSegment(ArcGISMSImage.open(self.label_list[idx]).data)

        assert (
            image_before.shape[1:] == image_after.shape[1:] == change_label.shape[1:]
        ), f"The image size of {self.before_list[idx]}, {self.after_list[idx]}, {self.label_list[idx]} is not same"
        # resolving transforms first so that
        # both before and after images are
        # cropped and zoomed the same way.
        _resolve_tfms(self.train_tfms)
        _resolve_tfms(self.val_tfms)
        _resolve_tfms(self.train_crop_tfm)
        _resolve_tfms(self.val_crop_tfm)

        images = (image_before, image_after, change_label)

        if self.split == "train":
            images = apply_tfms(images, self.train_crop_tfm, self.train_tfms)
        else:
            images = apply_tfms(images, self.val_crop_tfm, self.val_tfms)

        image_before, image_after, change_label = images

        self.image_before = image_before
        self.image_after = image_after
        self.change_label = change_label

        change_label = remap_label(change_label.data, self.class_mapping)
        image_before = normalize(image_before.px, *self.norm_stats)
        image_after = normalize(image_after.px, *self.norm_stats)

        return (image_before, image_after), change_label

    def show(self, idx, axes=None, rgb_bands=None):
        if rgb_bands is None:
            rgb_bands = self.rgb_bands
        self[idx]
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(15, 7))
        self.image_before.show(axes[0], rgb_bands=rgb_bands)
        self.image_after.show(axes[1], rgb_bands=rgb_bands)
        # Single channel that's indexed [0]
        change_data = self.change_label.data.numpy()[0]
        axes[2].imshow(get_colored_label(change_data, self.color_mapping))
        axes[2].axis("off")


def get_colored_label(change_data, color_mapping):
    colored_label = np.zeros(change_data.shape + (3,))
    # This logic will need to be changed in multiclass
    # change detection
    colored_label[change_data == 0] = color_mapping[0]
    # Get key except 0, dict of length 2
    all_classes = list(color_mapping.keys())
    all_classes.remove(0)
    change_value = all_classes[0]
    colored_label[change_data != 0] = color_mapping[change_value]
    colored_label = colored_label.astype(int)
    return colored_label


def map_color(element):
    """
    This will be required in case of multiclass CD.
    """
    # change_data = np.vectorize(map_color, signature='()->(3)')(change_data)
    return np.array(self.color_mapping.get(element))


def apply_tfms(images, crop_tfm, other_tfms):
    """
    For some reason crop transform is getting applied
    with others hence, applying it differently.
    """
    image_before, image_after, change_label = images
    size = crop_tfm[0].kwargs["size"]
    # To fix when the disk image size is smaller than
    # crop size, otherwise crop_tfm behaves strange
    if min(image_before.shape[1:]) / size < 1:
        image_before = image_before.resize(size)
        image_after = image_after.resize(size)
        change_label = change_label.resize(size)

    image_before = image_before.apply_tfms(crop_tfm, do_resolve=False)
    image_before = image_before.apply_tfms(other_tfms, do_resolve=False)

    image_after = image_after.apply_tfms(crop_tfm, do_resolve=False)
    image_after = image_after.apply_tfms(other_tfms, do_resolve=False)

    change_label = change_label.apply_tfms(crop_tfm, do_resolve=False)

    # Multispectral tfm not on labels.
    if other_tfms != [] and str(type(other_tfms[0])) == "<class 'function'>":
        change_label = change_label.apply_tfms(other_tfms[1:], do_resolve=False)
    else:
        change_label = change_label.apply_tfms(other_tfms, do_resolve=False)

    return image_before, image_after, change_label


def _get_transforms(transforms, flip_vert):
    if transforms is None:
        transforms = get_transforms(
            flip_vert=flip_vert,
            max_lighting=0.3,
            max_warp=0.0,
        )

    elif transforms is False:
        transforms = ([], [])

    return transforms


image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)

# These stats are used in original implementation.
stats = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
# stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


def folder_check(path):
    dirs = os.listdir(path)
    images_before = "images" in dirs or "images_before" in dirs
    images_after = "images2" in dirs or "images_after" in dirs
    labels = "labels" in dirs
    if not all([images_before, images_after, labels]):
        raise Exception(
            f"Three folders must be present in the {path.name}"
            " directory namely 'images', 'images2' and 'labels'"
            " or 'images_before', 'images_after' and 'labels'"
        )


def is_old_format_change_detection(path):
    folder_check(path)
    return os.path.exists(path / "images_before") and os.path.exists(
        path / "images_after"
    )


def folder_names(path):
    if is_old_format_change_detection(path):
        return ("images_before", "images_after")
    else:
        return ("images", "images2")


def get_files(*args, **kwargs):
    return sorted(gf(*args, **kwargs))


def create_train_val_sets(
    path,
    val_split_pct,
    chip_size,
    transforms,
    norm_stats,
    flip_vert,
    class_mapping,
    color_mapping,
    split_type="random",
    _is_multispectral=False,
    rgb_bands=None,
    imagery_type="RGB",
):
    """Create train and val datasets based on type of split.


    Args:
        path (string): path to dataset. It can either contain train and val
                       folder. It can also contain a single folder with three
                       folders.
                           dataroot:
                                    ├─images_before
                                    ├─images_after
                                    ├─labels

        val_split_pct (float): percentage of data to split in validation
        if the split_type is "random"

        chip_size (integer): final size of integer to train

        transforms (fastai transforms): if None, defaults are used, if False,
                                        no transforms are used if passed they
                                        need to be passed as tuple of list of
                                        fastai transforms
                                        ([train_tfms],[val_tfms])

        norm_stats (list of list): [[mean per channel], [std per channel]]

        split_type (str, optional): If split_type='folder' will use train
                                    val folders. Defaults to 'random'.

        tuple: train and validation dataset
    """

    path = Path(path)
    if split_type == "folder":
        if (path / "train").exists() and (path / "val").exists():
            folder_check(path / "train")
            folder_check(path / "val")
            train_images_before = get_files(
                path / "train" / "images", extensions=image_extensions
            )
            train_images_after = get_files(
                path / "train" / "images2", extensions=image_extensions
            )
            train_labels = get_files(
                path / "train" / "labels", extensions=image_extensions
            )
            val_images_before = get_files(
                path / "val" / "images", extensions=image_extensions
            )
            val_images_after = get_files(
                path / "val" / "images2", extensions=image_extensions
            )
            val_labels = get_files(path / "val" / "labels", extensions=image_extensions)
        else:
            raise Exception(
                f"Split Type: {split_type}." "'train' and 'val' folder must be present"
            )

    elif split_type == "random":
        folder_check(path)
        images_before_dir, images_after_dir = folder_names(path)
        images_before = get_files(
            path / images_before_dir, extensions=image_extensions, recurse=True
        )
        images_after = get_files(
            path / images_after_dir, extensions=image_extensions, recurse=True
        )
        labels = get_files(path / "labels", extensions=image_extensions, recurse=True)

        total_num_images = len(images_before)
        val_num_images = int(total_num_images * val_split_pct)

        zipped = list(zip(images_before, images_after, labels))

        random.shuffle(zipped)
        images_before, images_after, labels = zip(*zipped)

        train_images_before, train_images_after, train_labels = (
            images_before[val_num_images:],
            images_after[val_num_images:],
            labels[val_num_images:],
        )

        val_images_before, val_images_after, val_labels = (
            images_before[:val_num_images],
            images_after[:val_num_images],
            labels[:val_num_images],
        )

    else:
        raise Exception(f"Split Type: {split_type}. is undefined.")

    train_dataset = ChangeDetectionDataset(
        path,
        train_images_before,
        train_images_after,
        train_labels,
        class_mapping,
        color_mapping,
        chip_size=chip_size,
        transforms=transforms,
        flip_vert=flip_vert,
        split="train",
        norm_stats=norm_stats,
        _is_multispectral=_is_multispectral,
        rgb_bands=rgb_bands,
        imagery_type=imagery_type,
    )

    val_dataset = ChangeDetectionDataset(
        path,
        val_images_before,
        val_images_after,
        val_labels,
        class_mapping,
        color_mapping,
        chip_size=chip_size,
        transforms=transforms,
        flip_vert=flip_vert,
        split="val",
        norm_stats=norm_stats,
        _is_multispectral=_is_multispectral,
        rgb_bands=rgb_bands,
        imagery_type=imagery_type,
    )

    return train_dataset, val_dataset


def create_dataloaders(datasets, batch_size, dataloader_kwargs):
    return [DataLoader(d, batch_size=batch_size, **dataloader_kwargs) for d in datasets]


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


def prepare_change_detection_data(
    path, chip_sz, batch_size, val_split_pct, transforms, **kwargs
):
    if not HAS_FASTAI:
        raise_fastai_import_error(
            import_exception=import_exception, message="", installation_steps=" "
        )
    norm_stats = kwargs.get("norm_stats", stats)
    imagery_type = kwargs.get("imagery_type")
    _is_multispectral = kwargs.get("_is_multispectral")
    rgb_bands = kwargs.get("rgb_bands")
    bands = kwargs.get("bands")
    norm_pct = kwargs.get("norm_pct")
    extract_bands = kwargs.get("extract_bands")
    flip_vert = kwargs.get("imagery_type", "satellite") != "oriented"
    class_mapping = kwargs.get("class_mapping", None)
    color_mapping = kwargs.get("color_mapping", None)
    split = kwargs.get("split", "random")
    working_dir = kwargs.get("working_dir", None)

    # Hardcoded for LEVIR-CD dataset.
    if class_mapping is None:
        class_mapping = {0: "NoChange", 255: "Change"}

    if color_mapping is None:
        color_mapping = {0: [0, 0, 0], 255: [255, 255, 255]}

    # create datasets
    datasets = create_train_val_sets(
        path,
        val_split_pct,
        chip_size=chip_sz,
        transforms=transforms,
        norm_stats=norm_stats,
        flip_vert=flip_vert,
        class_mapping=class_mapping,
        color_mapping=color_mapping,
        split_type=split,
        _is_multispectral=_is_multispectral,
        rgb_bands=rgb_bands,
        imagery_type=imagery_type,
    )

    # num_workers=0 for win and maximum for linux
    databunch_kwargs = (
        {"num_workers": 0}
        if sys.platform == "win32"
        else {"num_workers": os.cpu_count() - 4}
    )
    # create dataloaders
    train_dl, valid_dl = create_dataloaders(datasets, batch_size, databunch_kwargs)

    # create Databunch
    device = get_device()
    data = DataBunch(train_dl, valid_dl, device=device)
    data.path = data.train_ds.path
    data._imagery_type = imagery_type
    data._do_normalize = None
    # Multispectral additions
    if _is_multispectral:
        multispectral_additions(data, **kwargs)
    # Attach show batch to databunch
    data.show_batch = types.MethodType(show_batch, data)
    # attach chip size
    data.chip_size = chip_sz
    # change or no-change
    assert 0 in class_mapping.keys()
    data.c = len(class_mapping)
    data.class_mapping = class_mapping
    data.color_mapping = color_mapping
    data.classes = list(data.class_mapping.values())
    # add dataset_type
    data._dataset_type = "ChangeDetection"
    if working_dir is not None:
        data.path = Path(os.path.abspath(working_dir))
    data._temp_folder = _prepare_working_dir(data.path)
    # return databunch.
    return data


def show_batch(self, rows=4, **kwargs):
    rgb_bands = kwargs.get("rgb_bands", None)
    fig, axes = plt.subplots(nrows=rows, ncols=3, squeeze=False, figsize=(20, rows * 5))
    top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
    plt.subplots_adjust(top=top)
    fig.suptitle("Image Before / Image After / Ground Truth", fontsize=16)
    img_idxs = [random.randint(0, len(self.train_ds) - 1) for k in range(rows)]
    for idx, im_idx in enumerate(img_idxs):
        self.train_ds.show(im_idx, axes[idx], rgb_bands)


# After model training.
def post_process(dists):
    if type(dists) is np.ndarray:
        dists = torch.from_numpy(dists).permute(2, 0, 1)
    return (dists > 1.0).long()


def display_row(axes, display, rgb_bands=None):
    if rgb_bands is None:
        rgb_bands = [0, 1, 2]
    for i, ax in enumerate(axes):
        if type(display[i]) is ArcGISMSImage:
            display[i].show(ax, rgb_bands)
        else:
            ax.imshow(display[i])
        ax.axis("off")


def post_process_y(y):
    # This will change in multi-class
    y[y == -1] = 255
    y[y == 1] = 0
    return y


def show_results(self, rows, **kwargs):
    self.learn.model.eval()
    rgb_bands = kwargs.get("rgb_bands", self._data.rgb_bands)
    x_batch, y_batch = get_nbatches(
        self._data.valid_dl, math.ceil(rows / self._data.batch_size)
    )
    x_before, x_after = [x[0] for x in x_batch], [x[1] for x in x_batch]  # , x_batch[1]
    x_before = torch.cat(x_before)
    x_after = torch.cat(x_after)
    y_batch = post_process_y(torch.cat(y_batch)).cpu()

    top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
    # Get Predictions
    activations = []
    for i in range(0, x_after.shape[0], self._data.batch_size):
        with torch.no_grad():
            preds = (
                self.learn.model(
                    x_before[i : i + self._data.batch_size],
                    x_after[i : i + self._data.batch_size],
                )
                .detach()
                .cpu()
            )
        activations.append(preds)

    # Analyze Pred
    activations = torch.cat(activations)
    predictions = post_process(activations)

    # Denormalize
    x_before = denormalize(x_before.cpu(), *self._data.norm_stats)
    x_after = denormalize(x_after.cpu(), *self._data.norm_stats)

    # Size for plotting
    rows = min(rows, x_after.shape[0])
    fig, axs = plt.subplots(
        nrows=rows, ncols=4, figsize=(4 * 5, rows * 5), squeeze=False
    )

    plt.subplots_adjust(top=top)
    fig.suptitle("Image Before / Image After / Ground Truth / Prediction", fontsize=16)
    for r in range(rows):
        if self._data._is_multispectral:
            display_row(
                axs[r],
                (
                    ArcGISMSImage(x_before[r]),
                    ArcGISMSImage(x_after[r]),
                    get_colored_label(image2np(y_batch[r]), self._data.color_mapping),
                    get_colored_label(
                        predictions[r][0].numpy(), self._data.color_mapping
                    ),
                ),
                rgb_bands,
            )
        else:
            display_row(
                axs[r],
                (
                    image2np(x_before[r]),
                    image2np(x_after[r]),
                    get_colored_label(image2np(y_batch[r]), self._data.color_mapping),
                    get_colored_label(
                        predictions[r][0].numpy(), self._data.color_mapping
                    ),
                ),
            )


# Predict on a pair of images.
def predict(
    self,
    image_before,
    image_after,
    crop_predict=False,
    visualize=False,
    padding=None,
    save=False,
):
    if self._data._is_multispectral:
        image_before = ArcGISMSImage.open(
            image_before, imagery_type=self._data._imagery_type
        )
        image_after = ArcGISMSImage.open(
            image_after, imagery_type=self._data._imagery_type
        )

        if not isinstance(self._data._band_min_values, torch.Tensor):
            self._data._band_min_values = (
                torch.tensor(self._data._band_min_values).clone().to(self._device)
            )
            self._data._band_max_values = (
                torch.tensor(self._data._band_max_values).clone().to(self._device)
            )
        _min_max_scaler_tfm = partial(
            _tensor_scaler_tfm,
            min_values=self._data._band_min_values,
            max_values=self._data._band_max_values,
            mode="minmax",
        )

        # scaling Transforms
        def _scaling_tfm(x):
            # Scales Fastai Image Scaling | MS Image Values -> 0 - 1 range
            return x.__class__(_min_max_scaler_tfm((x.data, None))[0][0])

        class dummy:
            pass

        _scaling_tfm.tfm = dummy()
        _scaling_tfm.tfm.order = 0
        _scaling_tfm.resolve = dummy

        image_before = image_before.apply_tfms([_scaling_tfm])
        image_after = image_after.apply_tfms([_scaling_tfm])

    else:
        image_before = ArcGISMSImage.open(
            image_before, imagery_type=self._data._imagery_type, div=255
        )
        image_after = ArcGISMSImage.open(
            image_after, imagery_type=self._data._imagery_type, div=255
        )

    assert image_before.shape == image_after.shape
    if crop_predict:
        if image_before.shape[1] >= self._data.chip_size:
            prediction = chunk_predict_stich(
                self, image_before, image_after, self._data.chip_size, padding=padding
            )
    else:
        image_before = image_before.resize(self._data.chip_size)
        image_after = image_after.resize(self._data.chip_size)
        image_before_tensor = normalize(image_before.px, *self._data.norm_stats)
        image_after_tensor = normalize(image_after.px, *self._data.norm_stats)
        image_before_tensor = image_before_tensor[None].to(self._device)  # batch size 1
        image_after_tensor = image_after_tensor[None].to(self._device)  # batch size 1
        self.learn.model.eval()
        with torch.no_grad():
            prediction = self.learn.model(
                image_before_tensor, image_after_tensor
            ).detach()[0]
    prediction = post_process(prediction)
    from fastai.vision import Image

    prediction = Image(prediction[[0]])
    if visualize:
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle("Image Before / Image After /  Prediction", fontsize=16)
        if self._data._is_multispectral:
            rgb_bands = self._data._rgb_bands
            if rgb_bands == []:
                rgb_bands = [0, 1, 2]
            image_before.show(ax=ax[0], rgb_bands=rgb_bands)
            image_after.show(ax=ax[1], rgb_bands=rgb_bands)
            prediction.show(ax=ax[2], rgb_bands=rgb_bands)
        else:
            image_before.show(ax=ax[0])
            image_after.show(ax=ax[1])
            prediction.show(ax=ax[2])
    if save:
        prediction.save("detected_change.png")
    return prediction.px.cpu()


def chunk_predict_stich(self, image_before, image_after, chip_size, padding=None):
    batch_size = self._data.batch_size
    if padding is None:
        padding = chip_size // 8
    chunker = ImageChunker(chip_size, chip_size, padding)
    device = image_before.device
    image_before = image_before.px.cpu().numpy().transpose(1, 2, 0)
    image_after = image_after.px.cpu().numpy().transpose(1, 2, 0)
    # This variable is required later.
    _image_before = image_before.copy()
    # Chunk it.
    image_before = chunker.dimension_preprocess(image_before)
    image_after = chunker.dimension_preprocess(image_after)
    image_before = torch.from_numpy(image_before.transpose(0, 3, 1, 2)).to(device)
    image_after = torch.from_numpy(image_after.transpose(0, 3, 1, 2)).to(device)
    image_before = normalize(image_before, *self._data.norm_stats).to(self._device)
    image_after = normalize(image_after, *self._data.norm_stats).to(self._device)
    self.learn.model.eval()
    activations = []
    for i in range(0, image_before.shape[0], batch_size):
        with torch.no_grad():
            preds = (
                self.learn.model(
                    image_before[i : i + batch_size], image_after[i : i + batch_size]
                )
                .detach()
                .cpu()
            )
        activations.append(preds)
    activations = torch.cat(activations).numpy().transpose(0, 2, 3, 1)
    activations = chunker.dimension_postprocess(activations, _image_before)
    return activations


# Implementation from https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/util.py
class ImageChunker(object):
    def __init__(self, rows, cols, overlap):
        self.rows = rows
        self.cols = cols
        self.overlap = overlap

    def perform_chunking(self, img_size, chunk_size):
        """
        Given an image dimension img_size, return list of (start, stop)
        tuples to perform chunking of chunk_size
        """
        chunks, i = [], 0
        while True:
            chunks.append(
                (
                    i * (chunk_size - self.overlap / 2),
                    i * (chunk_size - self.overlap / 2) + chunk_size,
                )
            )
            i += 1
            if chunks[-1][1] > img_size:
                break
        n_count = len(chunks)
        chunks[-1] = tuple(
            x - (n_count * chunk_size - img_size - (n_count - 1) * self.overlap / 2)
            for x in chunks[-1]
        )
        chunks = [(int(x), int(y)) for x, y in chunks]
        return chunks

    def get_chunks(self, img, scale=1):
        """
        Get width and height lists of (start, stop) tuples for chunking of img.
        """
        x_chunks, y_chunks = [(0, self.rows)], [(0, self.cols)]
        if img.shape[0] > self.rows:
            x_chunks = self.perform_chunking(img.shape[0], self.rows)
        else:
            x_chunks = [(0, img.shape[0])]
        if img.shape[1] > self.cols:
            y_chunks = self.perform_chunking(img.shape[1], self.cols)
        else:
            y_chunks = [(0, img.shape[1])]
        return x_chunks, y_chunks

    def dimension_preprocess(self, img, padding=True):
        """
        In case of prediction on image of different size than 512x512,
        this function is used to add padding and chunk up the image into pieces
        of 512x512, which can then later be reconstructed into the original image
        using the dimension_postprocess() function.
        """

        # Assert single image input
        assert len(img.shape) == 3, "Image dimension expected to be (H, W, C)"

        # Check if we are adding padding for too small images
        if padding:
            # Check if height is too small
            if img.shape[0] < self.rows:
                padding = np.ones(
                    (self.rows - img.shape[0], img.shape[1], img.shape[2])
                )
                img = np.concatenate((img, padding), axis=0)

            # Check if width is too small
            if img.shape[1] < self.cols:
                padding = np.ones(
                    (img.shape[0], self.cols - img.shape[1], img.shape[2])
                )
                img = np.concatenate((img, padding), axis=1)

        # Get chunking of the image
        x_chunks, y_chunks = self.get_chunks(img)
        # Chunk up the image
        images = []
        for x in x_chunks:
            for y in y_chunks:
                images.append(img[x[0] : x[1], y[0] : y[1], :])
        images = np.array(images)
        return images

    def dimension_postprocess(
        self, chunked_images, original_image, scale=1, padding=True
    ):
        """
        In case of prediction on image of different size than 512x512,
        the dimension_preprocess  function is used to add padding and chunk
        up the image into pieces of 512x512, and this function is used to
        reconstruct these pieces into the original image.
        """

        # Assert input dimensions
        assert (
            len(original_image.shape) == 3
        ), "Image dimension expected to be (H, W, C)"
        assert (
            len(chunked_images.shape) == 4
        ), "Chunked images dimension expected to be (B, H, W, C)"

        # Check if we are adding padding for too small images
        if padding:
            # Check if height is too small
            if original_image.shape[0] < self.rows:
                new_images = []
                for img in chunked_images:
                    new_images.append(img[0 : scale * original_image.shape[0], :, :])
                chunked_images = np.array(new_images)

            # Check if width is too small
            if original_image.shape[1] < self.cols:
                new_images = []
                for img in chunked_images:
                    new_images.append(img[:, 0 : scale * original_image.shape[1], :])
                chunked_images = np.array(new_images)

        # Put reconstruction into this array
        new_shape = (
            original_image.shape[0] * scale,
            original_image.shape[1] * scale,
            original_image.shape[2],
        )
        reconstruction = np.zeros(new_shape)

        # Get the chunks for this image
        x_chunks, y_chunks = self.get_chunks(original_image)

        i = 0
        s = scale
        for x in x_chunks:
            for y in y_chunks:
                prior_fill = reconstruction != 0
                chunk = np.zeros(new_shape)
                chunk[x[0] * s : x[1] * s, y[0] * s : y[1] * s, :] += chunked_images[i]
                chunk_fill = chunk != 0

                reconstruction += chunk
                reconstruction[prior_fill & chunk_fill] = (
                    reconstruction[prior_fill & chunk_fill] / 2
                )

                i += 1

        return reconstruction
