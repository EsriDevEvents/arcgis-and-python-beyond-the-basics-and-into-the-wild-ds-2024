import math
import os
import traceback
import types
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

HAS_FASTAI = True

try:
    import torch
    from arcgis.learn._data_utils._base_data import MAP_SPACE, ArcgisData
    from arcgis.learn._data_utils._road_orient_data import RoadOrientation
    from arcgis.learn._utils.classified_tiles import show_batch_classified_tiles
    from arcgis.learn.models._unet_utils import ArcGISSegmentationItemList
    from fastai.vision.data import imagenet_stats
    from fastai.vision.transform import ResizeMethod, get_transforms

    # from fastai.vision.transform import brightness as brightness_tfm
    # from fastai.vision.transform import contrast as contrast_tfm
    # from fastai.vision.transform import crop as crop_tfm
    # from fastai.vision.transform import dihedral_affine as dihedral_affine_tfm
    # from fastai.vision.transform import rotate as rotate_tfm
except ImportError as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class ClassifiedTilesData(ArcgisData):
    def __init__(
        self,
        path: Path,
        class_mapping: Dict,
        val_split_pct: float,
        batch_size: Union[int, Tuple[int]],
        transforms: List = [],
        seed: int = 42,
        **kwargs,
    ):
        """
        This class create data object based on classified tiles with data format of pixel classification.
        """
        super().__init__(
            path=path,
            class_mapping=class_mapping,
            batch_size=batch_size,
            val_split_pct=val_split_pct,
            transforms=transforms,
            seed=seed,
            **kwargs,
        )
        self.class_mapping = self._get_class_mapping(class_mapping)
        self.color_mapping = {
            (i.get("Value", 0) or i.get("ClassValue", 0)): i["Color"]
            for i in self.emd.get("Classes", [])
        }

    def get_databunch(self, sub_dataset_type=None, **kwargs):
        """
        Get FastAI databunch.
        """
        if sub_dataset_type == "RoadOrientation":
            road_orient_obj = RoadOrientation(self, **kwargs)
            data = road_orient_obj.get_databunch()
        else:
            self.databunch_kwargs.update(bs=self.batch_size)
            data = self.get_common_databunch()
            if self.imagery_is_multispectral:
                data.show_batch = types.MethodType(show_batch_classified_tiles, data)

                # The default fastai collate_fn was causing memory leak on tensors
                def classified_tiles_collate_fn(samples):
                    r = (
                        torch.stack([x[0].data for x in samples]),
                        torch.stack([x[1].data for x in samples]),
                    )
                    return r

                self.databunch_kwargs.update(collate_fn=classified_tiles_collate_fn)
            else:
                data.show_batch = types.MethodType(
                    types.FunctionType(
                        data.show_batch.__code__,
                        data.show_batch.__globals__,
                        data.show_batch.__name__,
                        (
                            min(int(math.sqrt(data.batch_size)), 5),
                            *data.show_batch.__defaults__[1:],
                        ),
                        data.show_batch.__closure__,
                    ),
                    data,
                )

        if data is None:
            return None
        # Get base Class object
        data._base = self
        data.sub_dataset_type = sub_dataset_type
        return self.set_databunch_attributes(data)

    def set_databunch_attributes(self, data):
        """
        Add/Update FastAI Databunch properties
        """
        data = super().set_databunch_attributes(data)
        if self.esri_stats:
            pixel_stats = self.esri_stats.get("ClassPixelStats", None)
            data.num_pixels_per_class = (
                pixel_stats.get("NumPixelsPerClass", None) if pixel_stats else None
            )
            data.class_weight = None
            if data.num_pixels_per_class:
                num_pixels_per_class = np.array(
                    data.num_pixels_per_class, dtype=np.int64
                )
                if num_pixels_per_class.sum() < 0:
                    data.overflow_encountered = True
                    data.class_weight = None
                else:
                    data.class_weight = (
                        num_pixels_per_class.sum() / num_pixels_per_class
                    )
        return data

    def get_common_databunch(self):
        """
        Get Common Databunch
        """
        not_label_count = [0]

        def get_y_func(x, ext=self.extension):
            return x.parents[1] / self.labels_folder / (x.stem + ".{}".format(ext))

        def image_without_label(imagefile, not_label_count=[0], ext=self.extension):
            xmlfile = (
                imagefile.parents[1]
                / self.labels_folder
                / (imagefile.stem + ".{}".format(ext))
            )
            if not os.path.exists(xmlfile):
                not_label_count[0] += 1
                return False
            return True

        remove_image_without_label = partial(
            image_without_label, not_label_count=not_label_count
        )
        data = (
            ArcGISSegmentationItemList.from_folder(self.path / self.images_folder)
            .sample_data(self.data_count)
            .filter_by_func(remove_image_without_label)
            .split_by_rand_pct(self.val_split_pct, seed=self.seed)
            .label_from_func(
                get_y_func,
                classes=(["NoData"] + list(self.class_mapping.values())),
                class_mapping=self.class_mapping,
                color_mapping=self.color_mapping,
            )
        )
        if self.transforms is None:
            self.transforms = self._get_common_transforms()
        # if self.chip_size is not None and self.dataset_image_size != self.chip_size:
        #     self.transforms[0].append(
        #         crop_tfm(size=self.chip_size, p=1.0, row_pct=0.5, col_pct=0.5)
        #     )
        #     self.transforms[1].append(
        #         crop_tfm(size=self.chip_size, p=1.0, row_pct=0.5, col_pct=0.5)
        #     )
        kwargs_transforms = {}
        kwargs_transforms["tfm_y"] = True
        kwargs_transforms["size"] = self.chip_size
        data = (
            data.transform(self.transforms, **kwargs_transforms).databunch(
                **self.databunch_kwargs
            )
            # .normalize(imagenet_stats)
        )
        if not self.imagery_is_multispectral:
            data = data.normalize(imagenet_stats)

        data.train_ds.x._div = 255.0
        data.valid_ds.x._div = 255.0
        # # First Apply transforms and then resize to given resize_to
        # data = data.transform(self.transforms, **kwargs_transforms)
        # if self.resize_to:
        #     data = data.transform([], size=self.resize_to, resize_method=ResizeMethod.SQUISH)
        # data = data.databunch(**self.databunch_kwargs)
        # if not self.imagery_is_multispectral:
        #     data = data.normalize(imagenet_stats)
        return data

    def _get_common_transforms(self):
        if self._image_space_used == MAP_SPACE:
            base_transforms = get_transforms(
                flip_vert=True, max_rotate=90.0, max_zoom=3.0, max_lighting=0.5
            )
        else:
            base_transforms = get_transforms(max_zoom=3.0, max_lighting=0.5)

        return base_transforms

    def _get_class_mapping(self, class_mapping):
        # Create Class Mapping from EMD if not specified by user
        # Validate user defined class_mapping keys with emd (issue #3064)
        # Get classmapping from emd file.
        try:
            emd_class_mapping = {i["Value"]: i["Name"] for i in self.emd["Classes"]}
        except KeyError:
            emd_class_mapping = {
                i["ClassValue"]: i["ClassName"] for i in self.emd["Classes"]
            }

        # Change all keys to int.
        updated_class_mapping = (
            {int(key): value for key, value in class_mapping.items()}
            if class_mapping
            else {}
        )
        # Map values from user defined classmapping to emd classmapping.
        for key, _ in emd_class_mapping.items():
            if updated_class_mapping.get(key) is not None:
                emd_class_mapping[key] = class_mapping[key]

        if emd_class_mapping.get(None):
            del emd_class_mapping[None]

        return emd_class_mapping
