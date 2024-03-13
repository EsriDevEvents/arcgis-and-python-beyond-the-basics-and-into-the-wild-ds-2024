import os
import traceback
import json
import math
import warnings
from zipfile import ZipFile
import tempfile
from pathlib import Path
from typing import Callable
import copy
from .utils import check_imbalance

HAS_FASTAI = False
try:
    from .env import (
        raise_fastai_import_error,
        HAS_GDAL,
        gdal_import_exception,
        GDAL_INSTALL_MESSAGE,
        _LAMBDA_TEXT_CLASSIFICATION,
    )

    if not _LAMBDA_TEXT_CLASSIFICATION:
        from fastai.vision.data import ImageList
        from fastai.vision import Image, imagenet_stats, pil2tensor
        import PIL
    elif _LAMBDA_TEXT_CLASSIFICATION:

        class Image:
            pass

        class ImageList:
            pass

    from fastai.data_block import get_files
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    HAS_FASTAI = True
except Exception:
    import_exception = traceback.format_exc()

    class Image:
        pass

    class ImageList:
        pass


def read_image(path, resize_to: int = None, keep_raw=False):
    """
    path: file path of image on disk.

    returns an array of image pixel values in channel last format.
    """
    path = str(os.path.abspath(path))
    if not os.path.exists:
        raise Exception(
            f"The image path {path} could not be found on disk, please verify your training data."
        )

    gdal_error = None
    try:
        if not HAS_GDAL:
            gdal_error = f"""{gdal_import_exception} \n\n{GDAL_INSTALL_MESSAGE}"""
        else:
            from osgeo import gdal

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = gdal.Open(path)
            if resize_to is None or keep_raw:
                arr = ds.ReadAsArray()
            else:
                gdal_dtype = ds.GetRasterBand(1).DataType
                transform = ds.GetGeoTransform()
                dx = transform[1]
                dy = transform[-1]
                #
                dx_new = (ds.RasterXSize / resize_to) * dx
                dy_new = (ds.RasterYSize / resize_to) * dy
                #
                ds_new = gdal.Warp(
                    "",
                    path,
                    dstSRS=ds.GetProjection(),
                    format="VRT",
                    outputType=gdal_dtype,
                    xRes=dx_new,
                    yRes=dy_new,
                )
                arr = ds_new.ReadAsArray()
            if len(arr.shape) > 2 and not keep_raw:
                arr = np.rollaxis(arr, 0, 3)
            return arr
    except Exception as _gdal_error:
        gdal_error = str(_gdal_error)

    # Attach gdal error
    message = f"""
       Tried opening image using gdal and encountered the following error
       \n\n{gdal_error}
       """

    if resize_to is not None:
        raise Exception(
            f"`resize_to` parameter is only supported using gdal. \n" + message
        )

    try:
        from skimage.io import imread

        with PIL.Image.open(path).convert("RGB") as im:
            arr = np.array(im)
            im.close()
        return arr
    except Exception as _pillow_error:
        pillow_error = str(_pillow_error)

    # Attach pillow error
    message += f"""
       \n===================================================================
       \n\nTried opening image using pillow and encountered the following error
       \n\n{pillow_error}
       """
    raise Exception(message)


class ArcGISMSImage(Image):
    def show(
        self,
        ax=None,
        rgb_bands=None,
        show_axis=False,
        title=None,
        return_ax=False,
    ):
        if rgb_bands is None:
            rgb_bands = getattr(self, "rgb_bands", [0, 1, 2])
        if ax is None:
            ax = plt.subplot(1, 1, 1)
        symbology_data = self.data[rgb_bands]
        im_shape = symbology_data.shape
        min_vals = symbology_data.view(im_shape[0], -1).min(dim=1)[0]
        max_vals = symbology_data.view(im_shape[0], -1).max(dim=1)[0]
        strechted_data = (symbology_data - min_vals.view(im_shape[0], 1, 1)) / (
            max_vals.view(im_shape[0], 1, 1) - min_vals.view(im_shape[0], 1, 1) + 0.001
        )
        data_to_plot = strechted_data.permute(1, 2, 0)
        if not show_axis:
            ax.axis("off")
        ax.imshow(data_to_plot)
        if title is not None:
            ax.set_title(title)
        if return_ax:
            return ax

    def print_method(self):
        return self.show()

    def _repr_png_(self):
        return self.show()

    def _repr_jpeg_(self):
        return self.show()

    @classmethod
    def open_gdal(cls, path):
        try:
            from osgeo import gdal
        except ImportError as e:
            message = f"""
            {e}\n\nPlease install gdal using the following command
            \nconda install gdal=2.3.3
            """
            raise Exception(message)
        path = str(os.path.abspath(path))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = gdal.Open(path).ReadAsArray()
        x = torch.tensor(x.astype(np.float32))
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        return cls(x)

    @staticmethod
    def read_image(path, keep_raw=False):
        return read_image(path, keep_raw=keep_raw)

    @classmethod
    def open(cls, path, cast_to=None, div=None, imagery_type=None):
        if cast_to is None:
            cast_to = np.float32
        path = str(os.path.abspath(path))
        if not os.path.exists:
            raise Exception(
                f"The image path {path} could not be found on disk, please verify your training data."
            )

        read = False
        gdal_error = None
        pillow_error = None
        try:
            from osgeo import gdal

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = gdal.Open(path).ReadAsArray()
            # Ignore Alpha Channel
            if x.shape[0] == 4 and imagery_type == "RGB":
                x = x[:3]
            x = torch.tensor(x.astype(cast_to))
            read = True
        except Exception as _gdal_error:
            gdal_error = str(_gdal_error)
            pass

        if not read:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", UserWarning
                    )  # EXIF warning from TiffPlugin
                    with PIL.Image.open(path).convert("RGB") as im:
                        x = pil2tensor(im, cast_to)
                        im.close()
                read = True
            except Exception as _pillow_error:
                pillow_error = str(_pillow_error)
                pass

        if not read:
            # Attach gdal error
            message = f"""
            Tried opening image using gdal and encountered the following error
            \n\n{gdal_error}
            """
            if (gdal_error) == ModuleNotFoundError:
                message += GDAL_INSTALL_MESSAGE
            # Attach pillow error
            message += f"""
            \n===================================================================
            \n\nTried opening image using pillow and encountered the following error
            \n\n{pillow_error}
            """

            raise Exception(message)
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            if div is not None:
                if isinstance(div, tuple):
                    min_values, max_values = div
                    if not isinstance(min_values, torch.Tensor):
                        min_values, max_values = (
                            torch.tensor(min_values),
                            torch.tensor(max_values),
                        )
                    for i in range(x.shape[0]):
                        arr = x[i, :, :]
                        arr[torch.isnan(arr)] = min_values[i]
                    min_values = min_values.to(x).view(-1, 1, 1)
                    max_values = max_values.to(x).view(-1, 1, 1)
                    x = (x - min_values) / (max_values - min_values + 1e-04)
                else:
                    x = x / div

                # Remove data values which are outside our data range.
                # Data Range is as read from EMD.
                # Handles No data values.
                x.clamp_(0, 1)
        return cls(x)


class ArcGISImageList(ImageList):
    "`ImageList` suitable for classification tasks."
    _square_show_res = False
    _div = None
    _imagery_type = None

    def open(self, fn):
        return ArcGISMSImage.open(fn, div=self._div, imagery_type=self._imagery_type)

    def check_class_imbalance(
        self, func: Callable, stratify=False, class_imbalance_pct=0.01
    ):
        try:
            labelval = [(func(o)) for o in self.items]
            if any(isinstance(el, list) for el in labelval):
                total_sample = np.concatenate(np.array(labelval))
            else:
                total_sample = np.array(labelval)
            unique_sample = set(total_sample)
            check_imbalance(total_sample, unique_sample, class_imbalance_pct, stratify)
        except Exception as e:
            warnings.warn(f"Unable to check for class imbalance [reason : {e}]")
        return self

    def label_list_from_func(self, func: Callable, val_split_pct):
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
        req_instances_per_class = int(0.8 / val_split_pct)
        classes_below_req_intances = list(
            label_series.value_counts()[
                label_series.value_counts() < req_instances_per_class
            ].index
        )
        if len(classes_below_req_intances) > 0:
            always_warn(
                f'For valid statification all classes should have more than {req_instances_per_class} data points, \
class(es) {",".join(classes_below_req_intances)} in your data does not meet the condition.'
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


class ArcGISImageListRGB(ArcGISImageList):
    _div = 255


def get_multispectral_data_params_from_emd(data, emd):
    data._is_multispectral = emd.get("IsMultispectral", False)
    if data._is_multispectral:
        data._bands = emd.get("Bands")
        data._imagery_type = emd.get("ImageryType")
        data._extract_bands = emd.get("ExtractBands")
        data._train_tail = False  # Hardcoded because we are never going to train a model with empty data
        normalization_stats = dict(
            emd.get("NormalizationStats")
        )  # Copy the normalization stats so that self._data.emd has no tensors other wise it will raise error while creating emd
        for _stat in normalization_stats:
            if normalization_stats[_stat] is not None:
                normalization_stats[_stat] = torch.tensor(normalization_stats[_stat])
            setattr(data, ("_" + _stat), normalization_stats[_stat])
        data._do_normalize = emd.get("DoNormalize")
    return data


def get_post_processed_model(arcgis_model, input_normalization=True):
    if arcgis_model._backend == "tensorflow":
        from .postprocessing_tf import get_post_processed_model_tf

        return get_post_processed_model_tf(
            arcgis_model, input_normalization=input_normalization
        )


def get_color_array(color_mapping: dict, alpha=0.7):
    color_array = np.array(list(color_mapping.values()), dtype=float) / 255
    color_array = np.concatenate(
        [
            color_array,
            np.repeat([alpha], color_array.shape[0]).reshape(color_array.shape[0], 1),
        ],
        axis=-1,
    )
    return color_array


## show_batch() show_results() helper functions start ##


def to_torch_tensor(x):
    if type(x) == torch.Tensor:
        return x.clone().detach()
    return torch.tensor(x)


def get_symbology_bands(rgb_bands, extract_bands, bands):
    e = Exception(
        "`rgb_bands` should be a valid band_order, list or tuple of length 3 or 1."
    )
    symbology_bands = []
    if not (len(rgb_bands) == 3 or len(rgb_bands) == 1):
        raise (e)
    for b in rgb_bands:
        if type(b) == str:
            b_index = bands.index(b)
        elif type(b) == int:
            # check if the band index specified by the user really exists.
            bands[b]
            b_index = b
        else:
            raise (e)
        b_index = extract_bands.index(b_index)
        symbology_bands.append(b_index)
    return symbology_bands


def get_top_padding(title_font_size, nrows, imsize):
    top = 1 - (math.sqrt(title_font_size) / math.sqrt(100 * nrows * imsize))
    return top


def kwarg_fill_none(kwargs, kwarg_key, default_value=None):
    value = kwargs.get(kwarg_key, None)
    if value is None:
        return default_value
    return value


def find_data_loader(type_data_loader, data):
    if type_data_loader == "training":
        data_loader = data.train_dl
    elif type_data_loader == "validation":
        data_loader = data.valid_dl
    elif type_data_loader == "testing":
        data_loader = data.test_dl
    else:
        e = Exception(f"could not find {type_data_loader} in data.")
        raise (e)
    return data_loader


def get_nbatches(data_loader, nbatches):
    x_batch, y_batch = [], []
    dl_iterater = iter(data_loader)
    get_next = True
    i = 0
    while i < nbatches and get_next:
        i += 1
        try:
            x, y = next(dl_iterater)
            x_batch.append(x)
            y_batch.append(y)
        except StopIteration:
            get_next = False
    return x_batch, y_batch


def image_tensor_checks_plotting(imagetensor_batch):
    symbology_x_batch = imagetensor_batch
    # Channel first to channel last for plotting
    symbology_x_batch = symbology_x_batch.permute(0, 2, 3, 1)

    # Clamp float values to range 0 - 1
    if symbology_x_batch.mean() < 1:
        symbology_x_batch = symbology_x_batch.clamp(0, 1)

    # Squeeze channels if single channel (224, 224, 1) -> (224, 224)
    if symbology_x_batch.shape[-1] == 1:
        symbology_x_batch = symbology_x_batch.squeeze(-1)
    return symbology_x_batch


def denorm_x(imagetensor_batch, self=None):
    """
    denormalizes a imagetensor_batch for plotting
    -------------------------
    imagetensor_batch: imagebatch with shape (batch, bands, rows, columns)

    self: optional. can be an instance of ArcGISModel or arcgis.learn data object(databunch)
    -------------------------
    returns denormalized imagetensor_batch
    """
    from .. import models

    if isinstance(self, models._arcgis_model.ArcGISModel):
        data = self._data
    else:
        data = self
    if data is not None and data._is_multispectral:
        return denorm_image(
            imagetensor_batch,
            mean=data._scaled_mean_values[data._extract_bands],
            std=data._scaled_std_values[data._extract_bands],
        )
    return denorm_image(imagetensor_batch)


def denorm_image(imagetensor_batch, mean=None, std=None):
    # prepare normalization stats
    if mean is None or std is None:
        mean = imagenet_stats[0]
        std = imagenet_stats[1]
    mean = to_torch_tensor(mean).to(imagetensor_batch).view(1, -1, 1, 1)
    std = to_torch_tensor(std).to(imagetensor_batch).view(1, -1, 1, 1)
    return (imagetensor_batch * std) + mean


def predict_batch(self, imagetensor_batch):
    if self._backend == "pytorch":
        if getattr(self, "_is_model_extension", False):
            if self._is_multispectral:
                imagetensor_batch = self._model_conf.transform_input_multispectral(
                    imagetensor_batch
                )
            else:
                imagetensor_batch = self._model_conf.transform_input(imagetensor_batch)
            predictions = self.learn.model.eval()(imagetensor_batch)
        else:
            predictions = self.learn.model.eval()(
                imagetensor_batch.to(self._device).float()
            )
        if type(predictions) in [tuple, list]:
            predictions = [x.detach() for x in predictions]
        else:
            predictions = predictions.detach()
        # predictions = self.learn.model.eval()(imagetensor_batch.to(self._device).float()).detach()
        return predictions
    elif self._backend == "tensorflow":
        from .common_tf import predict_batch_tf

        return predict_batch_tf(self, imagetensor_batch)


## show_batch() show_results() helper functions end ##

## Image Stretching Functions start ##


def get_band_percent_minmax(values, min_clip, max_clip):
    return (
        values[math.floor(values.shape[0] * min_clip)],
        values[values.shape[0] - math.ceil(values.shape[0] * max_clip)],
    )


def get_percent_minmax(imagetensor_batch, min_clip=0.0025, max_clip=0.005):
    shp = imagetensor_batch.shape
    _imagetensor_batch = imagetensor_batch.transpose(1, 0).reshape(shp[1], -1)
    min_vals = []
    max_vals = []
    for i in range(shp[1]):
        v = get_band_percent_minmax(_imagetensor_batch[i].unique(), min_clip, max_clip)
        min_vals.append(v[0])
        max_vals.append(v[1])
    return (
        torch.tensor(
            min_vals,
            dtype=imagetensor_batch.dtype,
            device=imagetensor_batch.device,
        ),
        torch.tensor(
            max_vals,
            dtype=imagetensor_batch.dtype,
            device=imagetensor_batch.device,
        ),
    )


def image_batch_stretcher(
    imagetensor_batch, stretch_type="minmax", statistics_type=None
):
    shp = imagetensor_batch.shape
    if statistics_type == "DRA":
        if stretch_type == "minmax":
            min_vals = imagetensor_batch.view(shp[0], shp[1], -1).min(dim=2)[0]
            max_vals = imagetensor_batch.view(shp[0], shp[1], -1).max(dim=2)[0]
        elif stretch_type == "percentclip":
            min_vals = []
            max_vals = []
            for idx in range(shp[0]):
                v = get_percent_minmax(imagetensor_batch[idx : idx + 1])
                min_vals.append(v[0])
                max_vals.append(v[1])
            min_vals = torch.stack(min_vals)
            max_vals = torch.stack(max_vals)
        else:
            raise NotImplementedError
        min_vals = min_vals.view(shp[0], shp[1], 1, 1)
        max_vals = max_vals.view(shp[0], shp[1], 1, 1)
    else:
        if stretch_type == "minmax":
            min_vals = (
                imagetensor_batch.transpose(1, 0).reshape(shp[1], -1).min(dim=1)[0]
            )
            max_vals = (
                imagetensor_batch.transpose(1, 0).reshape(shp[1], -1).max(dim=1)[0]
            )
        elif stretch_type == "percentclip":
            min_vals, max_vals = get_percent_minmax(imagetensor_batch)
        else:
            raise NotImplementedError
        min_vals = min_vals.view(1, shp[1], 1, 1)
        max_vals = max_vals.view(1, shp[1], 1, 1)
    #
    imagetensor_batch = (imagetensor_batch - min_vals) / (max_vals - min_vals + 0.001)
    imagetensor_batch = imagetensor_batch.clamp(0, 1)
    return imagetensor_batch


def dynamic_range_adjustment(imagetensor_batch):
    shp = imagetensor_batch.shape
    min_vals = imagetensor_batch.view(shp[0], shp[1], -1).min(dim=2)[0]
    max_vals = imagetensor_batch.view(shp[0], shp[1], -1).max(dim=2)[0]
    imagetensor_batch = (imagetensor_batch - min_vals.view(shp[0], shp[1], 1, 1)) / (
        max_vals.view(shp[0], shp[1], 1, 1)
        - min_vals.view(shp[0], shp[1], 1, 1)
        + 0.001
    )
    imagetensor_batch = imagetensor_batch.clamp(0, 1)
    return imagetensor_batch


## Image Stretching Functions end ##


def load_model(emd_path, data=None):
    from .. import models

    # if not HAS_FASTAI:
    #     raise_fastai_import_error(import_exception=import_exception)

    _emd_path = os.path.abspath(emd_path)
    if not os.path.exists(emd_path):
        raise Exception(
            f"Could not find an EMD file at the specified path does not exist '{emd_path}'"
        )

    with open(_emd_path) as f:
        emd = json.load(f)
    model_name = "".join(char for char in emd["ModelName"] if char.isalnum())
    model_cls = getattr(models, model_name, None)

    if model_cls is None:
        raise Exception(
            f"Failed to load model, Could not find class '{model_name}' in arcgis.learn.models"
        )

    model_obj = model_cls.from_model(_emd_path, data=data)

    return model_obj


def _temp_dlpk(dlpk_path):
    with ZipFile(dlpk_path, "r") as zip_obj:
        temp_dir = tempfile.TemporaryDirectory().name
        zip_obj.extractall(temp_dir)
    return temp_dir


def _get_emd_path(emd_path):
    emd_path = Path(emd_path)
    if emd_path.suffix == ".dlpk":
        temp_path = _temp_dlpk(emd_path)
        emd_path = Path(temp_path)
        # return cls.from_model(temp_path)

    if emd_path.suffix != ".emd":
        list_files = get_files(emd_path, extensions=[".emd"])
        try:
            assert len(list_files) == 1
        except AssertionError as e:
            raise Exception(
                f"There are multiple emd files in the saved path.Save the model in a different path."
            )
        # return cls.from_model(list_files[0])
        emd_path = list_files[0]
    return emd_path


def _get_gpu_device_id(max_memory=0.8):
    """
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    """
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory, available = 0, 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available


def _get_device_id():
    import arcgis
    from ..models._arcgis_model import _device_check

    move_to_cpu = _device_check()
    if move_to_cpu:
        arcgis.env._processorType = "CPU"

    if getattr(arcgis.env, "_processorType", "") == "GPU" and torch.cuda.is_available():
        device = _get_gpu_device_id()
    elif getattr(arcgis.env, "_processorType", "") == "CPU":
        device = -1
    else:
        device = _get_gpu_device_id() if torch.cuda.is_available() else -1

    return device


def always_warn(warning_message):
    warning_filters = copy.deepcopy(warnings.filters)
    warnings.simplefilter("always", UserWarning)
    warnings.warn(warning_message)
    warnings.filters = warning_filters
