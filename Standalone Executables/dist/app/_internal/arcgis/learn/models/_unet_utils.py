from fastai.vision import ImageSegment, Image
from fastai.vision.image import open_image, show_image, pil2tensor
from fastai.vision.data import SegmentationProcessor, ImageList
from fastai.layers import CrossEntropyFlat
from fastai.basic_train import LearnerCallback
from fastai.core import is_listy
from .._utils.common import (
    ArcGISMSImage,
    get_top_padding,
    kwarg_fill_none,
    find_data_loader,
    get_nbatches,
    dynamic_range_adjustment,
    image_tensor_checks_plotting,
    get_symbology_bands,
    predict_batch,
    denorm_x,
    get_nbatches,
    GDAL_INSTALL_MESSAGE,
    image_batch_stretcher,
)
from .._utils.env import HAS_GDAL
from .._utils.pixel_classification import analyze_pred_pixel_classification
from .._utils.env import is_arcgispronotebook
from .._utils.utils import check_imbalance
import torch
import warnings
import PIL
import numpy as np
import os
import math
from typing import Callable
import warnings


def _class_array_to_rbg(ca: "classified_array", cm: "color_mapping", nodata=0):
    im = np.expand_dims(ca, axis=2).repeat(3, axis=2)
    white_mask = im[im == nodata]
    for x in np.unique(im):
        if not x == nodata:
            for i in range(3):
                im[:, :, i][im[:, :, i] == x] = cm[x][i]
    im[white_mask] = 255
    return im


# def _show_batch_unet_multispectral(self, nrows=3, ncols=3, n_items=None, index=0, rgb_bands=None, nodata=0, alpha=0.7, imsize=5): # Proposed Parameters
def _show_batch_unet_multispectral(
    self, rows=3, alpha=0.7, **kwargs
):  # parameters adjusted in kwargs
    import matplotlib.pyplot as plt
    from .._data import _tensor_scaler

    nrows = rows
    ncols = 3
    if kwargs.get("ncols", None) is not None:
        ncols = kwargs.get("ncols")

    n_items = None
    if kwargs.get("n_items", None) is not None:
        n_items = kwargs.get("n_items")

    type_data_loader = kwargs.get(
        "data_loader", "training"
    )  # options : traininig, validation, testing
    if type_data_loader == "training":
        data_loader = self.train_dl
    elif type_data_loader == "validation":
        data_loader = self.valid_dl
    elif type_data_loader == "testing":
        data_loader = self.test_dl
    else:
        e = Exception(
            f"could not find {type_data_loader} in data. Please ensure that the data loader type is traininig, validation or testing "
        )
        raise (e)

    rgb_bands = self._symbology_rgb_bands
    if kwargs.get("rgb_bands", None) is not None:
        rgb_bands = kwargs.get("rgb_bands")

    nodata = 0
    if kwargs.get("nodata", None) is not None:
        nodata = kwargs.get("nodata")

    index = 0
    if kwargs.get("index", None) is not None:
        index = kwargs.get("index")

    imsize = 5
    if kwargs.get("imsize", None) is not None:
        imsize = kwargs.get("imsize")

    statistics_type = kwargs.get(
        "statistics_type", "dataset"
    )  # Accepted Values `dataset`, `DRA`
    stretch_type = kwargs.get(
        "stretch_type", "minmax"
    )  # Accepted Values `minmax`, `percentclip`

    e = Exception(
        "`rgb_bands` should be a valid band_order, list or tuple of length 3 or 1."
    )
    symbology_bands = []
    if not (len(rgb_bands) == 3 or len(rgb_bands) == 1):
        raise (e)
    for b in rgb_bands:
        if type(b) == str:
            b_index = self._bands.index(b)
        elif type(b) == int:
            self._bands[
                b
            ]  # To check if the band index specified by the user really exists.
            b_index = b
        else:
            raise (e)
        b_index = self._extract_bands.index(b_index)
        symbology_bands.append(b_index)

    # Get Batch
    if n_items is None:
        n_items = nrows * ncols
    else:
        nrows = math.ceil(n_items / ncols)
    n_items = min(n_items, len(self.x))

    x_batch, y_batch = get_nbatches(data_loader, n_items)
    x_batch = torch.cat(x_batch)
    y_batch = torch.cat(y_batch)
    # Denormalize X
    x_batch = (
        self._scaled_std_values[self._extract_bands].view(1, -1, 1, 1).to(x_batch)
        * x_batch
    ) + self._scaled_mean_values[self._extract_bands].view(1, -1, 1, 1).to(x_batch)

    # Extract RGB Bands
    symbology_x_batch = x_batch[:, symbology_bands]
    if stretch_type is not None:
        symbology_x_batch = image_batch_stretcher(
            symbology_x_batch, stretch_type, statistics_type
        )

    # Channel first to channel last and clamp float values to range 0 - 1 for plotting
    symbology_x_batch = symbology_x_batch.permute(0, 2, 3, 1)
    # Clamp float values to range 0 - 1
    if symbology_x_batch.mean() < 1:
        symbology_x_batch = symbology_x_batch.clamp(0, 1)

    # Squeeze channels if single channel (1, 224, 224) -> (224, 224)
    if symbology_x_batch.shape[-1] == 1:
        symbology_x_batch = symbology_x_batch.squeeze()

    # Get color Array
    color_array = self._multispectral_color_array
    color_array[1:, 3] = alpha

    # Size for plotting
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * imsize, nrows * imsize)
    )
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if nrows == 1 and ncols == 1:
                axi = axs
            else:
                axi = axs[r][c]
            if idx < symbology_x_batch.shape[0]:
                axi.imshow(symbology_x_batch[idx])
                y_rgb = color_array[y_batch[idx][0]]  # .cpu().numpy()
                # y_rgb = _class_array_to_rbg(y_batch[idx][0], self._multispectral_color_mapping, nodata)
                axi.imshow(y_rgb, alpha=alpha)
                axi.axis("off")
            else:
                axi.axis("off")
            idx += 1
    #


class ArcGISImageSegment(Image):
    "Support applying transforms to segmentation masks data in `px`."

    def __init__(self, x, color_mapping=None):
        super(ArcGISImageSegment, self).__init__(x)
        self.color_mapping = color_mapping

    def lighting(self, func, *args, **kwargs):
        return self

    def refresh(self):
        self.sample_kwargs["mode"] = "nearest"
        return super().refresh()

    @property
    def data(self):
        "Return this image pixels as a `LongTensor`."
        return self.px.long()

    def show(
        self,
        ax=None,
        figsize: tuple = (3, 3),
        title=None,
        hide_axis: bool = True,
        cmap=None,
        alpha: float = 0.5,
        **kwargs,
    ):
        "Show the `ImageSegment` on `ax`."
        if is_no_color(self.color_mapping):
            ## This condition will not be true.
            ax = show_image(
                self,
                ax=ax,
                hide_axis=hide_axis,
                cmap="tab20",
                figsize=figsize,
                interpolation="nearest",
                alpha=alpha,
                vmin=0,
                **kwargs,
            )
        else:
            color_mapping = torch.tensor(list(self.color_mapping.values()))
            color_mapping = torch.cat(
                (
                    color_mapping.float() / 255,
                    torch.tensor([float(alpha)] * len(color_mapping)).view(-1, 1),
                ),
                dim=1,
            )
            color_mapping = torch.cat(
                (torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, -1), color_mapping), dim=0
            )
            try:
                color_im = color_mapping[self.data[0]].permute(2, 0, 1)
            except IndexError as e:
                if HAS_GDAL:
                    message = f"Encountered invalid values in training label values, please check your training data."
                else:
                    message = (
                        f"Encountered invalid values while reading training labels. Please install gdal for better support.\n\n"
                        + GDAL_INSTALL_MESSAGE
                    )

                raise Exception(f"{e} \n\n{message}")

            ax = show_image(
                color_im,
                ax=ax,
                hide_axis=hide_axis,
                cmap=cmap,
                figsize=figsize,
                interpolation="nearest",
                alpha=alpha,
                vmin=0,
                **kwargs,
            )
        if title:
            ax.set_title(title)


class ArcGISMultispectralImageSegment:
    def __init__(self, tensor):
        self.data = tensor
        self.size = tensor.shape
        self.shape = tensor.shape


def is_no_color(color_mapping):
    if isinstance(color_mapping, dict):
        color_mapping = list(color_mapping.values())
    return (np.array(color_mapping) == [-1.0, -1.0, -1.0]).any()


def is_contiguous(class_values):
    flag = True
    for i in range(len(class_values) - 1):
        if class_values[i] + 1 != class_values[i + 1]:
            flag = False
    return flag


def map_to_contiguous(tensor, mapping):
    modified_tensor = torch.zeros_like(tensor)
    for i, value in enumerate(mapping):
        modified_tensor[tensor == value] = i
    return modified_tensor


class ArcGISSegmentationLabelList(ImageList):
    "`ItemList` for segmentation masks."
    _processor = SegmentationProcessor

    def __init__(
        self, items, classes=None, class_mapping=None, color_mapping=None, **kwargs
    ):
        super().__init__(items, **kwargs)
        self.class_mapping = class_mapping
        self.color_mapping = color_mapping
        self.copy_new.append("classes")
        self.classes, self.loss_func = classes, CrossEntropyFlat(axis=1)
        self.is_contiguous = is_contiguous(
            sorted([0] + list(self.class_mapping.keys()))
        )
        if not self.is_contiguous:
            self.pixel_mapping = [0] + list(self.class_mapping.keys())

    def _open_rgb(self, fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
            x = PIL.Image.open(fn)
            if x.palette is not None:
                x = x.convert("P")
            else:
                x = x.convert("L")
            x = pil2tensor(x, np.float32)

        if not self.is_contiguous:
            x = map_to_contiguous(x, self.pixel_mapping)

        return ArcGISImageSegment(x, color_mapping=self.color_mapping)

    def analyze_pred(
        self, pred, thresh=0.5, ignore_mapped_class=[], model=None, thinning=None
    ):
        if getattr(model, "_is_model_extension", False):
            if thinning is None:
                pred = model._model_conf.post_process(pred, thresh)
            else:
                pred = model._model_conf.post_process(pred, thresh, thinning)
            return pred

        if is_listy(pred):
            pred = pred[0]

        if ignore_mapped_class == []:
            return pred.argmax(dim=0)[None]
        else:
            for k in ignore_mapped_class:
                pred[k] = -1
            return pred.argmax(dim=0)[None]

    def reconstruct(self, t):
        return ArcGISImageSegment(t, color_mapping=self.color_mapping)

    def open(self, fn):
        x = ArcGISMSImage.open(fn).data
        if not self.is_contiguous:
            x = map_to_contiguous(x, self.pixel_mapping)
        return ArcGISImageSegment(x, color_mapping=self.color_mapping)


class ArcGISSegmentationItemList(ImageList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls, _square_show_res = ArcGISSegmentationLabelList, False
    _div = None
    _imagery_type = None

    def open(self, fn):
        return ArcGISMSImage.open(fn, div=self._div, imagery_type=self._imagery_type)

    def check_class_imbalance(
        self, func: Callable, class_mapping, stratify=False, class_imbalance_pct=0.01
    ):
        try:
            labelval = [(func(o)) for o in self.items]
            total_sample = np.concatenate(labelval)
            total_sample = total_sample[total_sample != "0"]
            unique_sample = set(total_sample)
            imabalanced_class_list = []

            for sample in unique_sample:
                if (total_sample == sample).sum() < len(
                    total_sample
                ) * class_imbalance_pct:
                    imabalanced_class_list.append(class_mapping[int(sample)])

            if stratify == True and len(imabalanced_class_list) > 0:
                warnings.warn(
                    f'We see a class imbalance in the dataset. The class(es) {",".join(imabalanced_class_list)} doesnt have enough data points in your dataset.'
                )
            elif stratify == False and len(imabalanced_class_list) > 0:
                warnings.warn(
                    f'We see a class imbalance in the dataset. The class(es) {",".join(imabalanced_class_list)} doesnt have enough data points in your dataset. Although, class imbalance cannot be overcome easily, adding the parameter stratify = True will to a certain extent help get over this problem.'
                )
        except Exception as e:
            warnings.warn(f"Unable to check for class imbalance [reason : {e}]")

        return self

    def label_list_from_func(self, func: Callable):
        "Apply `func` to every input to get its label."
        import pandas as pd

        self._list_of_labels = ["_".join(func(o)) for o in self.items]
        self._idx_label_tuple_list = [
            (i, label) for i, label in enumerate(self._list_of_labels)
        ]
        label_series = pd.Series(self._list_of_labels)
        single_instance_labels = list(
            label_series.value_counts()[label_series.value_counts() == 1].index
        )
        self._label_idx_mapping = {
            label: i for i, label in enumerate(self._list_of_labels)
        }
        for (
            label
        ) in single_instance_labels:  # adding duplicate instance of unique labels
            self._idx_label_tuple_list.append((self._label_idx_mapping[label], label))
        return self

    def stratified_split_by_pct(self, valid_pct: float = 0.2, seed: int = None):
        try:
            "Split the items in a stratified manner by putting `valid_pct` in the validation set, optional `seed` can be passed."
            from sklearn.model_selection import train_test_split
            import random, math

            if valid_pct == 0.0:
                return self.split_none()
            if seed is not None:
                np.random.seed(seed)
            if (
                len(set(self._list_of_labels)) > len(self._list_of_labels) * valid_pct
            ):  # if validation samples length is less than unique labels
                classes = len(set(self._list_of_labels))
                xlen = len(self._list_of_labels)
                sample_shortage = math.ceil((classes - xlen * valid_pct) / valid_pct)
                print(sample_shortage)
                extra_samples = random.choices(
                    self._idx_label_tuple_list, k=sample_shortage
                )
                self._idx_label_tuple_list.extend(extra_samples)
            X, y = [], []
            for index, label in self._idx_label_tuple_list:
                X.append(index)
                y.append(label)
            train_idx, val_idx, _, _ = train_test_split(
                X, y, test_size=valid_pct, random_state=seed, stratify=y
            )
            return self.split_by_idxs(train_idx, val_idx)
        except Exception as e:
            warnings.warn(
                f"Unable to perform stratified splitting [reason : {e}], falling back to random split"
            )
            return self.split_by_rand_pct(valid_pct=valid_pct, seed=seed)


class ArcGISSegmentationMSLabelList(ArcGISSegmentationLabelList):
    def open(self, fn):
        from osgeo import gdal

        path = str(os.path.abspath(fn))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = gdal.Open(path).ReadAsArray()
        x = torch.tensor(x.astype(np.float32))[None]
        if not self.is_contiguous:
            x = map_to_contiguous(x, self.pixel_mapping)
        return ArcGISImageSegment(x, color_mapping=self.color_mapping)


class LabelCallback(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.label_mapping = {
            value: (idx + 1)
            for idx, value in enumerate(learn.data.class_mapping.keys())
        }

    def on_batch_begin(self, last_input, last_target, **kwargs):
        """
        This callback is not used anymore.
        Not using this callback has increased the unet model training speed.
        """
        # modified_target = torch.zeros_like(last_target)
        # for label, idx in self.label_mapping.items():
        #     modified_target[last_target==label] = idx
        return {"last_input": last_input, "last_target": last_target}


# def show_results_multispectral(self, nrows=3, index=0, type_ds='valid', rgb_bands=None, nodata=0, alpha=0.7, imsize=5, top=0.97): # Proposed Parameters
def show_results_multispectral(
    self, nrows=5, alpha=0.7, **kwargs
):  # parameters adjusted in kwargs
    import matplotlib.pyplot as plt

    # Get Number of items
    ncols = 2
    return_fig = kwargs.get("return_fig", False)
    type_data_loader = kwarg_fill_none(
        kwargs, "data_loader", "validation"
    )  # options : traininig, validation, testing
    data_loader = find_data_loader(type_data_loader, self._data)
    if getattr(self, "name", "") in ["MultiTaskRoadExtractor"]:
        data_loader = find_data_loader(type_data_loader, self._orient_data)
        self._data.batch_size = 1

    nodata = kwarg_fill_none(kwargs, "nodata", 0)

    index = kwarg_fill_none(kwargs, "index", 0)

    imsize = kwarg_fill_none(kwargs, "imsize", 5)

    top = kwargs.get("top", None)
    title_font_size = 16
    if top is None:
        top = get_top_padding(
            title_font_size=title_font_size, nrows=nrows, imsize=imsize
        )

    statistics_type = kwarg_fill_none(
        kwargs, "statistics_type", "dataset"
    )  # Accepted Values `dataset`, `DRA`
    stretch_type = kwargs.get(
        "stretch_type", "minmax"
    )  # Accepted Values `minmax`, `percentclip`

    # get batches
    x_batch, y_batch = get_nbatches(
        data_loader, math.ceil(nrows / self._data.batch_size)
    )
    symbology_x_batch = x_batch = torch.cat(x_batch)
    if getattr(self, "name", "") in ["MultiTaskRoadExtractor"]:
        y_batch = (
            torch.stack([item for sublist in y_batch for item in sublist[0]])
            .type(torch.long)
            .unsqueeze(1)
        )
    else:
        y_batch = torch.cat(y_batch)

    symbology_bands = [0, 1, 2]
    if self._is_multispectral:
        # Get RGB Bands for plotting
        rgb_bands = kwarg_fill_none(
            kwargs, "rgb_bands", self._data._symbology_rgb_bands
        )

        # Get Symbology bands
        symbology_bands = get_symbology_bands(
            rgb_bands, self._data._extract_bands, self._data._bands
        )

    # Get Predictions
    activation_store = []
    for i in range(0, x_batch.shape[0], self._data.batch_size):
        activations = predict_batch(self, x_batch[i : i + self._data.batch_size])
        activation_store.append(activations)

    # Analyze Pred
    if getattr(self, "name", "") in ["MultiTaskRoadExtractor"]:
        activation_store = [x[0] for x in activation_store]
    predictions = analyze_pred_pixel_classification(self, activation_store)

    # Denormalize X
    x_batch = denorm_x(x_batch, self)

    # Extract RGB Bands for plotting
    symbology_x_batch = x_batch[:, symbology_bands]
    if stretch_type is not None:
        symbology_x_batch = image_batch_stretcher(
            symbology_x_batch, stretch_type, statistics_type
        )

    # Apply Image Strecthing
    if statistics_type == "DRA":
        symbology_x_batch = dynamic_range_adjustment(symbology_x_batch)

    symbology_x_batch = image_tensor_checks_plotting(symbology_x_batch)

    # Get color Array
    color_array = self._data._multispectral_color_array
    color_array[1:, 3] = alpha

    # Size for plotting
    nrows = min(nrows, symbology_x_batch.shape[0])
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * imsize, nrows * imsize)
    )
    plt.subplots_adjust(top=top)
    fig.suptitle("Ground Truth / Predictions", fontsize=title_font_size)
    for r in range(nrows):
        if nrows == 1:
            axi = axs
        else:
            axi = axs[r]
        if r < symbology_x_batch.shape[0]:
            axi[0].imshow(symbology_x_batch[r].cpu().numpy())
            y_rgb = color_array[y_batch[r][0]]
            axi[0].imshow(y_rgb, alpha=alpha)
            axi[1].imshow(symbology_x_batch[r].cpu().numpy())
            p_rgb = color_array[predictions[r]]
            axi[1].imshow(p_rgb, alpha=alpha)
        axi[0].axis("off")
        axi[1].axis("off")
    #
    if is_arcgispronotebook():
        plt.show()
    if return_fig:
        return fig, axs
