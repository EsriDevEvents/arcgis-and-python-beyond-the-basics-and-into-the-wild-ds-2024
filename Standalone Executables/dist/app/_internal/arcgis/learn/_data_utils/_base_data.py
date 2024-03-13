import json
import os
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .._utils.env import raise_fastai_import_error

HAS_FASTAI = True

try:
    import arcgis
    import numpy as np
    import torch
    from arcgis.learn._utils.common import ArcGISMSImage
    from arcgis.learn._utils.env import ARCGIS_ENABLE_TF_BACKEND
    from fastai.torch_core import set_all_seed
    from PIL import Image
except ImportError as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

# Constants
ESRI_MODEL_DEFINITION = "esri_model_definition.emd"
ESRI_MAP = "map.txt"
ESRI_STATS = "esri_accumulated_stats.json"
ESRI_STATS_METADATAMODE = "MetaDataMode"
ESRI_STATS_IMAGESPACEUSED = "ImageSpaceUsed"

MAP_SPACE = "MAP_SPACE"
PIXEL_SPACE = "PIXEL_SPACE"


class ArcgisData(object):
    def __init__(
        self,
        path: Union[str, Path],
        class_mapping: Dict,
        chip_size: int = 256,
        val_split_pct: float = 0.1,
        batch_size: Union[int, Tuple[int]] = 64,
        transforms: List = [],
        seed: int = 42,
        dataset_type=None,
        resize_to: int = None,
        **kwargs,
    ):
        """
        Base class for all data object used by Arcgis learn/training modules.
        This data object consists of training and validation datasets and used as base class for
        different data object formt such as object detection, pixel classification etc.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        path                    Required string. Path to data directory.
        ---------------------   -------------------------------------------
        class_mapping           Optional dictionary. Mapping from id to
                                its string label.
                                For dataset_type=IOB, BILUO or ner_json:
                                    Provide address field as class mapping
                                    in below format:
                                    class_mapping={'address_tag':'address_field'}.
                                    Field defined as 'address_tag' will be treated
                                    as a location. In cases where trained model extracts
                                    multiple locations from a single document, that
                                    document will be replicated for each location.

        ---------------------   -------------------------------------------
        val_split_pct           Optional float. Percentage of training data to keep
                                as validation.
        ---------------------   -------------------------------------------
        batch_size              Optional integer. Batch size for mini batch gradient
                                descent (Reduce it if getting CUDA Out of Memory
                                Errors).
        ---------------------   -------------------------------------------
        transforms              Optional tuple. Fast.ai transforms for data
                                augmentation of training and validation datasets
                                respectively (We have set good defaults which work
                                for satellite imagery well). If transforms is set
                                to `False` no transformation will take place and
                                `chip_size` parameter will also not take effect.
                                If the dataset_type is 'PointCloud', use
                                :class:`~arcgis.learn.Transform3d` .
        ---------------------   -------------------------------------------
        collate_fn              Optional function. Passed to PyTorch to collate data
                                into batches(usually default works).
        ---------------------   -------------------------------------------
        seed                    Optional integer. Random seed for reproducible
                                train-validation split.
        ---------------------   -------------------------------------------
        dataset_type            Optional string.
        ---------------------   -------------------------------------------
        resize_to               Optional integer. Resize the image to given size.
        =====================   ===========================================

        **Keyword Arguments**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        imagery_type            Optional string. Type of imagery used to export
                                the training data, valid values are:
                                    - 'naip'
                                    - 'sentinel2'
                                    - 'landsat8'
                                    - 'ms' - any other type of imagery
        ---------------------   -------------------------------------------
        bands                   Optional list. Bands of the imagery used to export
                                training data.
                                For example ['r', 'g', 'b', 'nir', 'u']
                                where 'nir' is near infrared band and 'u' is a miscellaneous band.
        ---------------------   -------------------------------------------
        rgb_bands               Optional list. Indices of red, green and blue bands
                                in the imagery used to export the training data.
                                for example: [2, 1, 0]
        ---------------------   -------------------------------------------
        extract_bands           Optional list. Indices of bands to be used for
                                training the model, same as in the imagery used to
                                export the training data.
                                for example: [3, 1, 0] where we will not be using
                                the band at index 2 to train our model.
        ---------------------   -------------------------------------------
        norm_pct                Optional float. Percentage of training data to be
                                used for calculating imagery statistics for
                                normalizing the data.
                                Default is 0.3 (30%) of data.
        =====================   ===========================================

        :return: data object
        """
        if not HAS_FASTAI:
            raise_fastai_import_error(import_exception=import_exception)

        self.seed = seed
        set_all_seed(self.seed)
        self.path = Path(path) if isinstance(path, str) else path
        assert (
            self.path.exists()
        ), "Invalid input path. Please ensure that the input path is correct."

        self.kwargs = kwargs
        self.has_esri_files = self._check_esri_files(self.path)
        self.emd, self.esri_stats, self.esri_map_lines = None, None, None
        if self.has_esri_files:
            self.emd = self._get_emd()
            self.esri_stats = self._get_stats_json()
            self.esri_map_lines = self._get_map_text_lines()

        self._image_space_used = (
            self.emd.get("ImageSpaceUsed", MAP_SPACE)
            if self.has_esri_files
            else PIXEL_SPACE
        )

        self.batch_size = batch_size[0] if isinstance(batch_size, tuple) else batch_size
        self.train_batch_size, self.valid_batch_size = (
            (batch_size, batch_size) if isinstance(batch_size, int) else batch_size
        )
        self.transforms = transforms
        self.val_split_pct = val_split_pct
        self.dataset_type = dataset_type
        self.data_count = kwargs.get("data_count", None)

        # Update Chip Size and extension for dataset
        image_size, extenstion = self._get_image_size_and_extension()
        self.dataset_image_size = image_size
        self.extension = extenstion
        self.resize_to = resize_to
        self.chip_size = chip_size if chip_size else image_size
        if self.chip_size >= image_size:
            self.chip_size = image_size

        self.images_folder = "images"
        self.labels_folder = "labels"
        self.img_root = os.path.join(self.path, self.images_folder)
        self.gt_root = os.path.join(self.path, self.labels_folder)
        self.data_num_workers = kwargs.get("data_num_workers", 16)
        self.downsample_factor = kwargs.get("downsample_factor", None)
        self.norm_pct = min(max(0, kwargs.get("norm_pct", 0.3)), 1)
        self.imagery_type = self._get_imagery_type()
        self.imagery_bands = self._get_imagery_bands(self.imagery_type)
        self.rgb_bands = self._get_rgb_bands()
        self.imagery_is_multispectral = bool(
            self.imagery_bands
            or self.rgb_bands
            or self.imagery_type not in ["RGB", "ASSUMED_RGB"]
        )
        # Hack to update back the imagery to multispectral if it is "RGB"
        self.imagery_type = (
            "MULTISPECTRAL"
            if self.imagery_is_multispectral
            and self.imagery_type in ["RGB", "ASSUMED_RGB"]
            else self.imagery_type
        )

        # TODO: Refactor Databunch  Args
        self.databunch_kwargs = (
            {"num_workers": 0}
            if sys.platform == "win32"
            else {"num_workers": min(self.data_num_workers, self.batch_size)}
        )

        force_cpu = arcgis.learn.models._arcgis_model._device_check()

        if hasattr(arcgis, "env") and force_cpu == 1:
            arcgis.env._processorType = "CPU"

        # if getattr(arcgis.env, "_processorType", "") == "CPU":
        #    self.databunch_kwargs["device"] = torch.device('cpu')

        if ARCGIS_ENABLE_TF_BACKEND:
            self.databunch_kwargs["device"] = torch.device("cpu")
            self.databunch_kwargs["pin_memory"] = False

    def get_databunch(self):
        raise NotImplementedError(
            f"Dataset of class {self.__name__} is not implemented."
        )

    def set_databunch_attributes(self, data):
        """
        Add/Update FastAI Databunch properties
        """
        data.chip_size = data.train_ds[0][0].shape[-1]
        data._dataset_type = self._get_dataset_type(self.path)
        data.class_mapping = self.class_mapping
        data.color_mapping = self.color_mapping
        data.orig_path = self.path
        data.resize_to = self.resize_to
        data.downsample_factor = self.downsample_factor
        # data._norm_pct = norm_pct
        data._image_space_used = self._image_space_used
        # if self.imagery_is_multispectral:
        #    data = self.set_multispectral_data_attributes(data)
        # if not self.imagery_is_multispectral:
        #    data._imagery_type = self.imagery_type
        #    data.train_ds.x._imagery_type = data._imagery_type
        #    data.valid_ds.x._imagery_type = data._imagery_type
        return data

    def _get_emd(self):
        """
        Read Esri Model Definition json file and return data.
        """
        json_file = Path.joinpath(self.path, ESRI_MODEL_DEFINITION)
        with open(json_file) as f:
            return json.load(f)

    def _get_stats_json(self):
        """
        Get data statistcs file.
        """
        stats_file = Path.joinpath(self.path, ESRI_STATS)
        with open(stats_file) as f:
            return json.load(f)

    def _get_map_text_lines(self, lines_count=2):
        """
        Get maps.txt file conating image/label pair paths.
        """
        map_file = Path.joinpath(self.path, ESRI_MAP)
        with open(map_file) as f:
            lines = []
            while True:
                line = f.readline()
                if len(line.split()) == 2 and len(lines) < lines_count:
                    lines.append(line)
                else:
                    break
            return lines

    def _get_image_size_and_extension(self) -> Tuple[int, str]:
        """
        Get Image size and extension of images from maps.txt file and return as tuple.
        """
        if self.esri_map_lines is None:
            return (None, None)
        map_first_line = self.esri_map_lines[0]
        try:
            img_size = ArcGISMSImage.open_gdal(
                (self.path / (map_first_line.split()[0]).replace("\\", os.sep))
            ).shape[-1]
        except BaseException:
            img_size = Image.open(
                (self.path / (map_first_line.split()[0]).replace("\\", os.sep))
            ).size[-1]

        extension_label = map_first_line.split()[1].split(".")[-1].lower()
        extension_img = map_first_line.split()[0].split(".")[-1].lower()
        return img_size, [extension_img, extension_label]

    def _get_imagery_type(self):
        """
        Get the imagery Type = (RGB or MultiSpectral) from EMD or input kwargs
        """
        input_imagery_type = self.kwargs.get("imagery_type", "ASSUMED_RGB")
        input_imagery_type = (
            input_imagery_type.upper()
            if input_imagery_type in ["ms", "rgb"]
            else input_imagery_type
        )
        # Multispectral support from EMD Not Implemented Yet
        # And it will give imagery type from kwargs = "imagery_type"
        _imagery_type = self.emd.get("imagery_type", input_imagery_type)

        return _imagery_type

    def _get_imagery_bands(self, imagery_type):
        """
        Get the imagery bands in following order
        1. Based on given imagery type define,d in ArcGIS - landsat8, naip, sentinel2
        2. Then get bands from input kwargs and overwrite
        3. And check if bands are available in EMD file and overwrite input kwargs bands
        """
        bands_based_on_imagery = (
            self._imagery_type_lib().get(imagery_type).get("bands", None)
            if imagery_type and imagery_type.lower() not in ["rgb", "assumed_rgb"]
            else None
        )
        input_bands = self.kwargs.get("bands", bands_based_on_imagery)
        _bands = self.emd.get("bands", input_bands)
        if _bands:
            _bands = [b.lower() for b in _bands if isinstance(b, str)]
        return _bands

    def _get_rgb_bands(self):
        rgb_bands = self.kwargs.get("rgb_bands", None)
        if self.imagery_bands:
            rgb_bands = [
                self.imagery_bands.index(b)
                for b in ["r", "g", "b"]
                if b in self.imagery_bands
            ]

        return rgb_bands

    # Static Methods
    @staticmethod
    def _check_esri_files(path: Path) -> bool:
        """
        Chek if all the esri files are present:
        1. esri_model_definition.emd
        2. map.txt
        3. esri_accumulated_stats.json
        """
        return (
            Path.joinpath(path, ESRI_MODEL_DEFINITION).exists()
            and Path.joinpath(path, ESRI_MAP).exists()
            and Path.joinpath(path, ESRI_STATS).exists()
        )

    @staticmethod
    def _get_dataset_type(path: Path) -> bool:
        """
        Get dataset type from esri_accumulated_stats.json from param = `MetaDataMode`
        """
        stats_file = Path.joinpath(path, ESRI_STATS)
        with open(stats_file) as f:
            stats = json.load(f)
            return stats["MetaDataMode"]

    @staticmethod
    def _imagery_type_lib():
        imagery_type_lib = {
            "landsat8": {
                "bands": [
                    "ca",
                    "b",
                    "g",
                    "r",
                    "nir",
                    "swir",
                    "swir",
                    "c",
                    "qa",
                    "tir",
                    "tir",
                ],
                "bands_info": {},  # incomplete
            },
            "naip": {"bands": ["r", "g", "b", "nir"], "bands_info": {}},  # incomplete
            "sentinel2": {
                "bands": [
                    "ca",
                    "b",
                    "g",
                    "r",
                    "vre",
                    "vre",
                    "vre",
                    "nir",
                    "nnir",
                    "wv",
                    "swirc",
                    "swir",
                    "swir",
                ],
                "bands_info": {  # incomplete
                    "b1": {"Name": "costal", "max": 10000, "min": 10000},
                    "b2": {"Name": "blue", "max": 10000, "min": 10000},
                },
            },
            "MS": {},
        }

        return imagery_type_lib
