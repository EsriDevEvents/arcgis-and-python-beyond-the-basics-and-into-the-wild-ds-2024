# MIT License

# PointCNN
# Copyright (c) 2018 Shandong University
# Copyright (c) 2018 Yangyan Li, Rui Bu, Mingchao Sun, Baoquan Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
import importlib
import sys
import warnings
import os
import math
from pathlib import Path
import json
import types
import random
import logging
import shutil
import copy
from functools import partial
import re
import warnings

logger = logging.getLogger()

try:
    from torch.utils.data import (
        DataLoader,
        Dataset,
        SubsetRandomSampler,
        SequentialSampler,
    )
    import torch
    import numpy as np
    from fastai.data_block import DataBunch
    import arcgis
    from fastai.data_block import ItemList
    from fastprogress.fastprogress import master_bar, progress_bar
    from scipy.spatial.transform import Rotation as R
    from ..models._rand_lanet_utils import batch_preprocess_dict
except ImportError:
    # To avoid breaking builds.
    class Dataset:
        pass

    class ItemList:
        pass


def try_imports(list_of_modules):
    ## Not a generic function.
    try:
        for module in list_of_modules:
            importlib.import_module(module)
    except Exception as e:
        raise Exception(
            f"""This function requires {' '.join(list_of_modules)}. Visit https://developers.arcgis.com/python/guide/install-and-set-up and https://developers.arcgis.com/python/guide/point-cloud-segmentation-using-pointcnn for installing the dependencies."""
        )


def try_import(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        if module == "plotly":
            raise Exception(
                "This function requires plotly. Visit https://developers.arcgis.com/python/guide/install-and-set-up and https://developers.arcgis.com/python/guide/point-cloud-segmentation-using-pointcnn for installing the dependencies."
            )
        elif module == "laspy":
            raise Exception(
                "This function requires laspy. Visit https://developers.arcgis.com/python/guide/install-and-set-up and https://developers.arcgis.com/python/guide/point-cloud-segmentation-using-pointcnn for installing the dependencies."
            )
        elif module == "h5py":
            raise Exception(
                f"This function requires h5py. Visit https://developers.arcgis.com/python/guide/install-and-set-up and https://developers.arcgis.com/python/guide/point-cloud-segmentation-using-pointcnn for installing the dependencies."
            )
        else:
            raise Exception(
                f"This function requires {module}. Please install it in your environment."
            )


def pad_tensor(cur_tensor, max_points, to_float=True):
    cur_points = cur_tensor.shape[0]
    if cur_points < max_points:
        remaining_points = max_points - cur_points
        if len(cur_tensor.shape) < 2:
            remaining_tensor = torch.zeros(remaining_points)
        else:
            remaining_tensor = torch.zeros(remaining_points, cur_tensor.shape[1])
        if to_float:
            remaining_tensor = remaining_tensor.float()
        else:
            cur_tensor = cur_tensor.long()
            remaining_tensor = remaining_tensor.long()
        cur_tensor = torch.cat((cur_tensor, remaining_tensor), dim=0)
    else:
        cur_tensor = cur_tensor[:max_points]
        cur_points = max_points
    return cur_tensor, cur_points


def concatenate_tensors(read_file, input_keys, tile, max_points=0, pad=True):
    cat_tensor = []

    cur_tensor = torch.tensor(
        read_file["xyz"][tile[1] : tile[1] + tile[2]].astype(np.float32)
    )
    if len(cur_tensor.shape) < 2:
        cur_tensor = cur_tensor[:, None]

    if pad:
        cur_tensor, cur_points = pad_tensor(cur_tensor, max_points)
    cat_tensor.append(cur_tensor)

    for key, min_max in input_keys.items():
        if key not in ["xyz"]:
            cur_tensor = torch.tensor(
                read_file[key][tile[1] : tile[1] + tile[2]].astype(np.float32)
            )
            if len(cur_tensor.shape) < 2:
                cur_tensor = cur_tensor[:, None]

            max_val = cur_tensor.new_tensor(min_max["max"])
            min_val = cur_tensor.new_tensor(min_max["min"])
            cur_tensor = (cur_tensor - min_val) / (
                max_val - min_val
            )  ## Test with one_hot
            if pad:
                cur_tensor, cur_points = pad_tensor(cur_tensor, max_points)
            cat_tensor.append(cur_tensor)
    if not pad:
        return torch.cat(cat_tensor, dim=1)

    return torch.cat(cat_tensor, dim=1), cur_tensor.new_tensor(cur_points).long()


def to_remap_classes(class_values):
    flag = False
    if class_values[0] != 0:
        return True
    for i in range(len(class_values) - 1):
        if class_values[i] + 1 != class_values[i + 1]:
            return True
    return flag


def get_class_from_bit_mask(mask):
    classes_present = set()
    mask = np.array(mask)
    i = 0
    for b in mask:
        for k in range(8):
            if b & (1 << k):
                classes_present.add(i)
            i += 1

    return classes_present


def get_filter_index(
    masks, tiles, min_points=None, classes_of_interest=None, return_file_index=False
):
    """
    tiles = N, [4000, 5000, 1000]
    masks = N, [[1,2], [2,5], [1,5]]
    classes_of_interest=[2]
    """
    file_indexes = set([])
    if min_points is not None:
        indexes_min_points = set()
        for i, t in enumerate(tiles):
            if t[2] > min_points:
                indexes_min_points.add(i)
                file_indexes.add(t[0])
    else:
        indexes_min_points = set(range(len(tiles)))

    skipped_blocks_min_points = len(tiles) - len(indexes_min_points)

    if classes_of_interest is not None:
        classes_of_interest = set(classes_of_interest)
        indexes_classes_of_interest = set()
        for i, m in enumerate(masks):
            classes_present = get_class_from_bit_mask(m)
            # print(classes_present, classes_of_interest, classes_present & classes_of_interest)
            if classes_present & classes_of_interest:
                indexes_classes_of_interest.add(i)
                file_indexes.add(tiles[i][0])
    else:
        indexes_classes_of_interest = set(range(len(tiles)))

    skipped_blocks_COI = len(tiles) - len(indexes_classes_of_interest)

    # intersection of two sets is our list of indexes.
    indexes = list(indexes_min_points & indexes_classes_of_interest)

    if return_file_index:
        return (
            indexes,
            skipped_blocks_min_points,
            skipped_blocks_COI,
            list(file_indexes),
        )
    else:
        return indexes


def get_background_classes(class_mapping, classes_of_interests):
    all_classes = set(class_mapping.keys())
    classes_of_interests = set(classes_of_interests)
    bg_classes = all_classes - classes_of_interests
    return list(bg_classes)


def compute_inverse_remap(remap_classes):
    inverse = {}
    for k, v in remap_classes.items():
        inverse.setdefault(v, []).append(k)
    return inverse


def expand_classes_of_interest(classes_of_interest, inverse_remap_classes):
    expanded = []
    for c in classes_of_interest:
        expanded.extend(inverse_remap_classes.get(c, [c]))
    return expanded


def get_random_cluster_indexes(block_centers, block_size):
    """
    This clustering method will operate on block centers
    as this will be less compute intesive as compared to the
    points.
    DBScan algo is used for this where we set the eps value
    to be 3/2 times the block size. This makes sure not extra
    cluster is for
    returns: a bool index mask
    """
    block_centers = np.array(block_centers)
    from sklearn.cluster import DBSCAN

    # clustering is required because some h5 files have many las files
    # because of which the aspect ratio of the show batch/ show result
    # plot was getting messed up.
    # The nearest neighbour to consider is 1.5 times the block size
    # based on experiments.
    # slice :2 because we want this clustering in spatial space only X and Y.
    clustering = DBSCAN(eps=3 * block_size / 2, min_samples=2).fit(block_centers[:, :2])
    clustered_labels = clustering.labels_
    unique_labels = np.unique(clustering.labels_)
    # clusters with fewer than 5 blocks cannot be selected.
    fewer_than = 5
    unique_label_masks = [clustered_labels == i for i in unique_labels]
    unique_label_masks_filtered = [
        i for i in unique_label_masks if i.sum() > fewer_than
    ]
    if len(unique_label_masks_filtered) != 0:
        return unique_label_masks_filtered[
            np.random.randint(0, len(unique_label_masks_filtered))
        ]
    else:
        return unique_label_masks[np.random.randint(0, len(unique_label_masks))]


class PointCloudDataset(Dataset):
    def __init__(self, path, class_mapping, json_file, folder="", **kwargs):
        try_import("h5py")
        import h5py

        self.init_kwargs = kwargs
        self.path = Path(path)
        if json_file is None:
            with open(self.path / folder / "Statistics.json", "r") as f:
                json_file = json.load(f)
        self.max_point = json_file["parameters"][
            "maxPoints"
        ]  ## maximum number of points
        self.statistics = json_file
        self.folder = folder
        self.features_to_keep = copy.copy(kwargs.get("extra_features", []))
        self.classification_key = kwargs.get(
            "classification_key", "classification"
        )  ## Key which contain labels
        self.min_points = kwargs.get("min_points", None)
        self.remap_classes = kwargs.get("remap_classes", {})
        self.classes_of_interest = kwargs.get("classes_of_interest", [])
        self.background_classcode = kwargs.get("background_classcode", None)
        self.block_size = self.statistics["parameters"]["tileSize"]
        self.input_keys = kwargs.get(
            "attributes", self.statistics["attributes"]
        )  # Keys to include in training
        self.input_keys.pop("rgbType", None)
        features_present = list(self.input_keys.keys())
        if "xyz" in features_present:
            features_present.remove("xyz")
        if self.features_to_keep != []:
            if not all([c in self.input_keys.keys() for c in self.features_to_keep]):
                raise Exception(
                    f"extra_features {self.features_to_keep} must be a subset of {features_present}"
                )
        self.features_to_keep += ["xyz"]
        self.input_keys = {
            k: v for k, v in self.input_keys.items() if k in self.features_to_keep
        }
        ## It is assumed that the pointcloud will have only X,Y & Z.
        extra_dim = (
            sum(
                [
                    len(v["max"]) if isinstance(v.get("max"), list) else 1
                    for k, v in self.input_keys.items()
                ]
            )
            - 3
        )
        self.extra_dim = extra_dim
        self.total_dim = 3 + extra_dim  # XYZ + extra dimensions
        self.extra_features = self.input_keys
        self.remap = False
        if self.min_points is not None:
            assert (
                self.min_points < self.max_point
            ), f"min_points({self.min_points}) cannot be greater than max_points({self.max_point}) set during export"

        if self.classes_of_interest == [] and self.background_classcode is not None:
            raise Exception(
                "background_classcode can only be used when `classes_of_interest` is passed."
            )

        # dummy class mapping for predict case.
        full_class_mapping = {0: 0}
        # will overwrite classmapping when classification key is present.
        if "classification" in self.statistics:
            full_class_mapping = {
                int(v["classCode"]): str(v["classCode"])
                for v in self.statistics["classification"]["table"]
            }

        orig_classes = list(full_class_mapping.keys())
        # account for remapping here.
        if self.remap_classes != {}:
            self.remap = True
            mapped_classes = set(self.remap_classes.values())
            unmapped_classes = set(
                [
                    c
                    for c in full_class_mapping.keys()
                    if c not in self.remap_classes.keys()
                ]
            )
            all_classes = mapped_classes.union(unmapped_classes)
            full_class_mapping = {int(c): str(c) for c in all_classes}

        if self.classes_of_interest != []:
            filter_classes = True
            classes_of_interest = self.classes_of_interest
        else:
            filter_classes = False
            classes_of_interest = None

        unexpanded_COI = classes_of_interest
        if filter_classes:
            if not all([c in full_class_mapping.keys() for c in unexpanded_COI]):
                raise Exception(
                    f"classes_of_interest {classes_of_interest} must be a subset of {list(full_class_mapping.keys())}"
                )

        if class_mapping is None:
            class_mapping = full_class_mapping
        else:
            class_mapping = {
                k: class_mapping.get(k, v) for k, v in full_class_mapping.items()
            }

        if self.classes_of_interest != [] and self.background_classcode is not None:
            # filter class mapping
            class_mapping = {
                k: v for k, v in class_mapping.items() if k in self.classes_of_interest
            }
            class_mapping[self.background_classcode] = "background"

        class_mapping = {int(k): str(v) for k, v in class_mapping.items()}

        self.class_mapping = class_mapping

        # self.subset_classes = subset_classes
        # print(self.class_mapping.keys())
        important_classes = list(self.class_mapping.keys())
        self.important_classes = important_classes
        self.remap_dict = self.remap_classes
        self.remap_bool = self.remap_classes is not {}

        self.class2idx = {
            value: idx
            for idx, value in enumerate(sorted(list(self.class_mapping.keys())))
        }
        ## Helper attributes for remapping
        self.c = len(self.class2idx)
        if self.remap is False:
            self.remap = to_remap_classes(list(self.class2idx.keys()))
        self.classes = list(self.class2idx.keys())

        if self.background_classcode is not None:
            self.remap = True
            ## get classes that belong to background and remap to bg code.
            bg_classes = get_background_classes(full_class_mapping, classes_of_interest)
            # map original classes to the index to which bg would have been mapped
            bg_mapping = {
                k: self.class2idx[self.background_classcode] for k in bg_classes
            }
            self._bg_mapping = bg_mapping
            # remove bg code from class2idx
            self.class2idx.pop(self.background_classcode)
            # merge the original with the bgmapped
            self.class2idx = {**self.class2idx, **bg_mapping}

        if self.remap_classes != {}:
            # if remap_classes is {7:25, 3:25}, the inverse will be
            # {25:[7,3]}
            inverse_remap_classes = compute_inverse_remap(self.remap_classes)
            if filter_classes:
                classes_of_interest = expand_classes_of_interest(
                    classes_of_interest, inverse_remap_classes
                )
            # print(inverse_remap_classes)
            self.repeated_mapping = {}
            for k, v in inverse_remap_classes.items():
                for c in v:
                    self.repeated_mapping[c] = self.class2idx[k]
            self.class2idx = {**self.class2idx, **self.repeated_mapping}

        # Color mapping.
        self.color_mapping = kwargs.get(
            "color_mapping",
            {
                k: [random.choice(range(256)) for _ in range(3)]
                for k, _ in self.class_mapping.items()
            },
        )
        self.color_mapping = {int(k): v for k, v in self.color_mapping.items()}
        # filter if classes are less.

        # filter out mapping of classes i.e not in the data.
        # in case of remap sometimes we get {remapped_value (out of classes): some value}
        self.class2idx = {k: v for k, v in self.class2idx.items() if k in orig_classes}
        # inverse mapping for visualization
        self.idx2class = {i: c for i, c in enumerate(sorted(self.classes))}
        self.classes_of_interest = classes_of_interest
        self.important_classes = important_classes
        with h5py.File(self.path / folder / "ListTable.h5", "r") as f:
            files = f["Files"][:]
            self.tiles = f["Tiles"][:]
            orig_num_tiles = len(self.tiles)
            self.masks = f["Masks"][:]
            # centers and scales will be required in viz.
            self.centers = f["Centers"][:]
            self._file_indexes = list(range(len(files)))

            if self.min_points is not None or filter_classes:
                # We should not filter on valid blocks, currently its happening on both.
                if folder == "val":
                    self.min_points = None
                    classes_of_interest = None
                (
                    indexes,
                    skip_block_min_points,
                    skip_block_COI,
                    file_indexes,
                ) = get_filter_index(
                    self.masks, self.tiles, self.min_points, classes_of_interest, True
                )

                self._skip_block_min_points = skip_block_min_points
                self._skip_block_COI = skip_block_COI
                self._total_blocks = len(self.tiles)
                if folder != "val":
                    self._file_indexes = file_indexes

                self.tiles = self.tiles[indexes]
                if len(self.tiles) == 0:
                    raise Exception(
                        f"The {folder} set is empty because everything "
                        "got filtered out."
                    )
                self._frac_remaining = len(self.tiles) / orig_num_tiles
                self.masks = self.masks[indexes]
                self.centers = self.centers[indexes]

        self.relative_files = files
        self.filenames = [self.path / self.folder / file.decode() for file in files]
        self.h5files = [(h5py.File(filename, "r")) for filename in self.filenames]
        self.classes_of_interest = classes_of_interest

    def __len__(self):
        return len(self.tiles)

    def _get_file_blocks(self, i):
        """Gives the xyz, labels by files.
        used in show_batch for tool.

        Args:
            i: index of file
        """
        assert i < len(self.h5files)
        indexes = np.where(self.tiles[:, 0] == i)
        read_file = self.h5files[i]
        labels = []
        xyzs = []
        xyzs_scaled = []
        centers = []
        scale = self.block_size / 2
        block_centers = []
        for idx in indexes[0]:
            tile = self.tiles[idx]
            center = self.centers[idx]
            xyz = read_file["xyz"][tile[1] : tile[1] + tile[2]]
            xyz_scaled = xyz * scale + center
            xyzs.append(xyz)
            labels.append(read_file["classification"][tile[1] : tile[1] + tile[2]])
            xyzs_scaled.append(xyz_scaled)
            block_centers.append(xyz_scaled.mean(axis=0))

        index_mask = get_random_cluster_indexes(block_centers, self.block_size)
        xyzs = np.concatenate(np.array(xyzs)[index_mask], axis=0)
        labels = np.concatenate(np.array(labels)[index_mask], axis=0)
        xyzs_scaled = np.concatenate(np.array(xyzs_scaled)[index_mask], axis=0)

        return xyzs, labels, xyzs_scaled

    def __getitem__(self, i, return_scaled=False, add_centers=False):
        tile_index = i
        tile = self.tiles[i]
        read_file = self.h5files[tile[0]]

        # we need this in show_results of tool.
        rescaled_xyz = read_file["xyz"][tile[1] : tile[1] + tile[2]].astype(
            np.float32
        ) * (self.block_size / 2)
        if add_centers:
            rescaled_xyz += self.centers[i]
        if self.classification_key in read_file.keys():
            classification, _ = pad_tensor(
                torch.tensor(
                    read_file[self.classification_key][
                        tile[1] : tile[1] + tile[2]
                    ].astype(int)
                ),
                self.max_point,
                to_float=False,
            )
            if not self.remap:
                retval = [
                    concatenate_tensors(
                        read_file, self.input_keys, tile, self.max_point
                    ),
                    classification.long(),
                ]
            else:
                retval = [
                    concatenate_tensors(
                        read_file, self.input_keys, tile, self.max_point
                    ),
                    remap_labels(classification, self.class2idx).long(),
                ]

        else:
            # removing warning as it is showing up in the tool.
            # logger.warning(f"key `{self.classification_key}` could not be found in the exported files.")
            retval = [
                concatenate_tensors(read_file, self.input_keys, tile, self.max_point),
                None,
            ]

        if return_scaled:
            # indexed zero because pad tensor returns two things and we only want the first one.
            retval += [
                pad_tensor(
                    torch.tensor(rescaled_xyz).float(), self.max_point, to_float=True
                )[0]
            ]

        if getattr(self, "_get_metainfo_h5", False):
            if self._api_model_h5:
                point_feature, point_num = pad_tensor(
                    torch.tensor(rescaled_xyz).float(), self.max_point, to_float=True
                )

                xmin, ymin, zmin = point_feature[:point_num].min(dim=0)[0]
                xmax, ymax, _ = point_feature[:point_num].max(dim=0)[0]
                point_feature = point_feature - torch.tensor(
                    [(xmin + xmax) / 2, (ymin + ymax) / 2, zmin]
                )
                # z in center
                point_feature[:, :3] = point_feature[:, [0, 2, 1]]
                point_feature = torch.cat((point_feature, retval[0][0][:, 3:]), axis=1)

                retval[0] = (point_feature, point_num)

            retval[1] = tile_index

        return retval

    def close(self):
        [file.close() for file in self.h5files]


def minmax_scale(pc):
    min_val = np.amin(pc, axis=0)
    max_val = np.amax(pc, axis=0)
    return (pc - min_val[None]) / max(max_val - min_val)


def recompute_color_mapping(color_mapping, all_classes):
    color_mapping = {int(k): v for k, v in color_mapping.items()}
    try:
        color_mapping = {k: color_mapping[k] for k in all_classes}
    except KeyError:
        raise Exception(
            f"Keys of your classes in your color_mapping do not match with classes present in data i.e {all_classes}"
        )
    return color_mapping


def class_string(label_array, prefix="", class_mapping=None):
    label_text = np.vectorize(class_mapping.get)(label_array)
    return [f"{prefix}class: {k}" for i, k in enumerate(label_text)]


def mask_classes(labels, mask_class, classes, class2idx=None, remap_classes=None):
    if not set(mask_class).issubset(set(classes)):
        raise Exception(f"`mask_class` {mask_class} must be a subset of {classes}")
    if remap_classes is not None:
        inverse_remap_classes = {v: k for k, v in remap_classes.items()}
        mask_class = [inverse_remap_classes.get(m, m) for m in mask_class]
    if class2idx is not None:
        mask_class = [class2idx[x] for x in mask_class]
    if mask_class == []:
        ## return complete mask
        return labels != None
    else:
        sample_idxs = np.concatenate([(labels[None] != mask) for mask in mask_class])
        sample_idxs = sample_idxs.all(axis=0)
        return sample_idxs


def get_max_display_points(self, kwargs):
    if "max_display_point" in kwargs.keys():
        max_display_point = kwargs["max_display_point"]
        self.max_display_point = max_display_point
    else:
        if hasattr(self, "max_display_point"):
            max_display_point = self.max_display_point
        else:
            max_display_point = 20000
    return max_display_point


def show_point_cloud_batch(self, rows=2, figsize=(6, 12), color_mapping=None, **kwargs):
    """
    It will plot 3d point cloud data you exported in the notebook.
    Visualization of data, exported in a geographic coordinate system
    is not yet supported.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional rows. Number of rows to show. Default
                            value is 2 and maximum value is the `batch_size`
                            passed in :meth:`~arcgis.learn.prepare_data` .
    ---------------------   -------------------------------------------
    color_mapping           Optional dictionary. Mapping from class value
                            to RGB values. Default value example:
                            {0:[220,220,220], 2:[255,0,0], 6:[0,255,0]}.
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    mask_class              Optional list of integers. Array containing
                            class values to mask. Use this parameter to
                            display the classes of interest.
                            Default value is [].
                            Example: All the classes are in [0, 1, 2]
                            to display only class `0` set the mask class
                            parameter to be [1, 2]. List of all classes
                            can be accessed from `data.classes` attribute
                            where `data` is the `Databunch` object returned
                            by :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    width                   Optional integer. Width of the plot. Default
                            value is 750.
    ---------------------   -------------------------------------------
    height                  Optional integer. Height of the plot. Default
                            value is 512.
    ---------------------   -------------------------------------------
    max_display_point       Optional integer. Maximum number of points
                            to display. Default is 20000. A warning will
                            be raised if the total points to display exceeds
                            this parameter. Setting this parameter will
                            randomly sample the specified number of points
                            and once set, it will be used for future uses.
    =====================   ===========================================
    """

    filter_outliers = False
    try_import("h5py")
    import h5py

    try_import("plotly")
    import plotly.graph_objects as go

    mask_class = kwargs.get("mask_class", [])
    apply_tfms = kwargs.get("apply_tfms", False)
    save_txt = kwargs.get("save_txt", False)
    max_display_point = get_max_display_points(self, kwargs)
    color_mapping = self.color_mapping if color_mapping is None else color_mapping
    color_mapping = recompute_color_mapping(color_mapping, self.classes)
    color_mapping = np.array(list(color_mapping.values())) / 255

    h5_files = self.h5files.copy()
    random.shuffle(h5_files)

    idx = 0
    file_idx = self._file_indexes[0]
    f_idx = 1
    while idx < rows:
        # file = h5_files[file_idx]
        _pc, labels, pc = self._get_file_blocks(file_idx)
        if self.remap:
            labels = remap_labels(labels, self.class2idx)
            unmapped_labels = remap_labels(labels.copy(), self.idx2class)
        else:
            unmapped_labels = labels.copy()
        sample_idxs = mask_classes(
            labels=labels,
            mask_class=mask_class,
            classes=self.classes,
            class2idx=self.class2idx if self.remap else None,
            remap_classes=self.remap_classes if self.remap else None,
        )
        sampled_pc = pc[sample_idxs]

        if sampled_pc.shape[0] == 0:
            file_idx = self._file_indexes[f_idx]
            f_idx = (f_idx + 1) % len(self._file_indexes)
            continue

        if apply_tfms:
            sampled_pc = self.transform_fn._transform_tool(sampled_pc[None])[0]
        x, y, z = recenter(sampled_pc).transpose(
            1, 0
        )  # convert to 3,N so that upacking works

        if save_txt:
            import pandas as pd

            x, y, z = sampled_pc.transpose(1, 0)
            pd.DataFrame(
                data={"x": x, "y": y, "z": z, "classification": unmapped_labels}
            ).to_csv("temp.txt", index=False, header=False)
            print("saved at temp.txt in current directory.")
            return

        if filter_outliers:
            # Filter on the basis of std.
            mask = filter_pc(pc)
        else:
            # all points
            mask = x > -999999999

        if sample_idxs.sum() > max_display_point:
            raise_maxpoint_warning(idx, kwargs, logger, max_display_point)
            mask = np.random.randint(0, sample_idxs.sum(), max_display_point)
        else:
            mask = np.arange(0, sample_idxs.sum())

        color_list = color_mapping[labels[sample_idxs]][mask].tolist()
        scene = dict(aspectmode="data")

        layout = go.Layout(
            width=kwargs.get("width", 750),
            height=kwargs.get("height", 512),
            scene=scene,
        )

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x[mask],
                    y=y[mask],
                    z=z[mask],
                    mode="markers",
                    marker=dict(size=1, color=color_list),
                    text=class_string(
                        unmapped_labels[sample_idxs][mask], "", self.class_mapping
                    ),
                )
            ],
            layout=layout,
        )
        fig.show()

        if idx == rows - 1:
            break
        idx += 1
        file_idx = self._file_indexes[f_idx]
        f_idx = (f_idx + 1) % len(self._file_indexes)


def filter_pc(pc):
    mean = pc.mean(0)
    std = pc.std(0)
    mask = (
        (pc[:, 0] < (mean[0] + 2 * std[0]))
        & (pc[:, 1] < (mean[1] + 2 * std[1]))
        & (pc[:, 2] < (mean[2] + 2 * std[2]))
    )
    return mask


def recenter(pc):
    min_val = np.amin(pc, axis=0)
    max_val = np.amax(pc, axis=0)
    return pc - min_val[None]


def show_point_cloud_batch_TF(self, rows=2, color_mapping=None, **kwargs):
    """
    It will plot 3d point cloud data you exported in the notebook.
    Visualization of data, exported in a geographic coordinate system
    is not yet supported.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional rows. Number of rows to show. Default
                            value is 2 and maximum value is the `batch_size`
                            passed in :meth:`~arcgis.learn.prepare_data` .
    ---------------------   -------------------------------------------
    color_mapping           Optional dictionary. Mapping from class value
                            to RGB values. Default value example:
                            {0:[220,220,220], 2:[255,0,0], 6:[0,255,0]}.
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    mask_class              Optional list of integers. Array containing
                            class values to mask. Use this parameter to
                            display the classes of interest.
                            Default value is [].
                            Example: All the classes are in [0, 1, 2]
                            to display only class `0` set the mask class
                            parameter to be [1, 2]. List of all classes
                            can be accessed from `data.classes` attribute
                            where `data` is the `Databunch` object returned
                            by :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    width                   Optional integer. Width of the plot. Default
                            value is 750.
    ---------------------   -------------------------------------------
    height                  Optional integer. Height of the plot. Default
                            value is 512.
    ---------------------   -------------------------------------------
    max_display_point       Optional integer. Maximum number of points
                            to display. Default is 20000. A warning will
                            be raised if the total points to display exceeds
                            this parameter. Setting this parameter will
                            randomly sample the specified number of points
                            and once set, it will be used for future uses.
    =====================   ===========================================
    """

    filter_outliers = False
    try_import("h5py")
    import h5py

    try_import("plotly")
    import plotly.graph_objects as go

    mask_class = kwargs.get("mask_class", [])
    apply_tfms = kwargs.get("apply_tfms", False)
    save_txt = kwargs.get("save_txt", False)
    rows = min(rows, self.batch_size)
    max_display_point = get_max_display_points(self, kwargs)
    color_mapping = self.color_mapping if color_mapping is None else color_mapping
    color_mapping = recompute_color_mapping(color_mapping, self.classes)
    color_mapping = np.array(list(color_mapping.values())) / 255

    def is_class_present(classes_of_interest, k, meta):
        classes_present = set()
        for u in meta["files"][k]["unique_classes"]:
            classes_present = classes_present | set(u)
        classes_present = list(classes_present)
        return any([p in classes_present for p in classes_of_interest])

    idx = 0
    import random

    keys = list(self.meta["files"].keys()).copy()

    if self.classes_of_interest != []:
        classes_of_interest = self.important_classes
        keys = [k for k in keys if is_class_present(classes_of_interest, k, self.meta)]

    if len(keys) == 0:
        warnings.warn(
            "No blocks remains after filtering based on `classes_of_interest`"
        )
    keys = [k for k in keys if "train" in Path(k).parts]
    random.shuffle(keys)

    for idx_file, fn in enumerate(keys):
        num_files = self.meta["files"][fn]["idxs"]
        block_center = self.meta["files"][fn]["block_center"]
        block_center = np.array(block_center)
        block_center[0][2], block_center[0][1] = block_center[0][1], block_center[0][2]
        if num_files == []:
            continue
        if not Path(fn).is_absolute():
            fn = str(self._data_path / fn)
        idxs = [h5py.File(fn[:-3] + f"_{i}.h5", "r") for i in num_files]
        pc = []
        labels = []
        for i in idxs:
            current_block = i["unnormalized_data"][:, :3]
            data_num = i["data_num"][()]
            pc.append(current_block[:data_num])
            labels.append(i["label_seg"][:data_num])
            i.close()

        if pc == []:
            continue

        pc = np.concatenate(pc, axis=0)
        labels = np.concatenate(labels, axis=0)
        if self.remap:
            labels = remap_labels(labels, self.class2idx)
            unmapped_labels = remap_labels(labels.copy(), self.idx2class)
        else:
            unmapped_labels = labels.copy()

        sample_idxs = mask_classes(
            labels=labels,
            mask_class=mask_class,
            classes=self.classes,
            class2idx=self.class2idx if self.remap else None,
            remap_classes=self.remap_classes if self.remap else None,
        )
        sampled_pc = pc[sample_idxs]
        if sampled_pc.shape[0] == 0:
            continue

        if apply_tfms:
            sampled_pc = self.transform_fn(sampled_pc[None])[0]

        x, y, z = recenter(sampled_pc).transpose(1, 0)

        if save_txt:
            import pandas as pd

            x, y, z = sampled_pc.transpose(1, 0)
            pd.DataFrame(
                data={"x": x, "y": y, "z": z, "classification": unmapped_labels}
            ).to_csv("temp.txt", index=False, header=False)
            print("saved at temp.txt in current directory.")
            return

        if filter_outliers:
            ## Filter on the basis of std.
            mask = filter_pc(pc)
        else:
            ## all points
            mask = [True] * len(x)

        if sample_idxs.sum() > max_display_point:
            raise_maxpoint_warning(idx_file, kwargs, logger, max_display_point)
            mask = np.random.randint(0, sample_idxs.sum(), max_display_point)
        else:
            mask = np.arange(0, sample_idxs.sum())

        color_list = color_mapping[labels[sample_idxs]][mask].tolist()

        scene = dict(aspectmode="data")
        layout = go.Layout(
            width=kwargs.get("width", 750),
            height=kwargs.get("height", 512),
            scene=scene,
        )

        figww = go.Figure(
            data=[
                go.Scatter3d(
                    x=x[mask],
                    y=z[mask],
                    z=y[mask],
                    mode="markers",
                    marker=dict(size=1, color=color_list),
                    text=class_string(
                        unmapped_labels[sample_idxs][mask], "", self.class_mapping
                    ),
                )
            ],
            layout=layout,
        )
        figww.show()

        if idx == rows - 1:
            break
        idx += 1


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


def read_xyzinumr_label_from_las(filename_las, extra_features):
    try_import("laspy")
    import laspy

    file = laspy.file.File(filename_las, mode="r")
    h = file.header
    xyzirgb_num = h.point_records_count
    labels = file.Classification.copy()

    xyz = np.concatenate(
        [file.x[:, None], file.y[:, None], file.z[:, None]]
        + [
            (np.clip(getattr(file, f[0]), None, f[1])[:, None] - f[2]) / (f[1] - f[2])
            for f in extra_features
        ],
        axis=1,
    )

    xyzirgb_num = len(xyz)
    file.close()
    return xyz, labels, xyzirgb_num


def prepare_las_data(
    root,
    block_size,
    max_point_num,
    output_path,
    extra_features=[("intensity", 5000, 0), ("num_returns", 5, 0)],
    grid_size=1.0,
    blocks_per_file=2048,
    folder_names=["train", "val"],
    segregate=True,
    **kwargs,
):
    try_import("h5py")
    import h5py

    drop_classes = kwargs.get("drop_points", [])
    block_size_ = block_size
    batch_size = blocks_per_file
    data = np.zeros(
        (batch_size, max_point_num, 3 + len(extra_features))
    )  # XYZ, Intensity, NumReturns
    unnormalized_data = np.zeros((batch_size, max_point_num, 3 + len(extra_features)))
    data_num = np.zeros((batch_size), dtype=np.int32)
    label = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)
    indices_split_to_full = np.zeros((batch_size, max_point_num), dtype=np.int32)
    LOAD_FROM_EXT = ".las"
    os.makedirs(output_path, exist_ok=True)

    if (Path(output_path) / "meta.json").exists() or (
        Path(output_path) / "Statistics.json"
    ).exists():
        raise Exception(
            f"The given output path({output_path}) already contains exported data. Either delete those files or pass in a new output path."
        )

    folders = [
        os.path.join(root, folder) for folder in folder_names
    ]  ## Folders are named train and val
    mb = master_bar(range(len(folders)))
    for itn in mb:
        folder = folders[itn]
        os.makedirs(os.path.join(output_path, Path(folder).stem), exist_ok=True)
        datasets = [
            filename[:-4]
            for filename in os.listdir(folder)
            if filename.endswith(LOAD_FROM_EXT)
        ]
        # mb.write(f'{itn + 1}. Exporting {Path(folder).stem} folder')
        for dataset_idx, dataset in enumerate(progress_bar(datasets, parent=mb)):
            filename_ext = os.path.join(folder, dataset + LOAD_FROM_EXT)
            if LOAD_FROM_EXT == ".las":
                xyzinumr, labels, xyz_num = read_xyzinumr_label_from_las(
                    filename_ext, extra_features
                )
                xyz, other_features = np.split(xyzinumr, (3,), axis=-1)
                if len(other_features.shape) < 2:
                    other_features = other_features[:, None]
            else:
                xyz, labels, xyz_num = read_xyz_label_from_txt(filename_ext)

            offsets = [("zero", 0.0), ("half", block_size_ / 2)]

            for offset_name, offset in offsets:
                idx_h5 = 0
                idx = 0

                xyz_min = np.amin(xyz, axis=0, keepdims=True) - offset
                xyz_max = np.amax(xyz, axis=0, keepdims=True)
                block_size = (
                    block_size_,
                    block_size_,
                    2 * (xyz_max[0, -1] - xyz_min[0, -1]),
                )
                xyz_blocks = np.floor((xyz - xyz_min) / block_size).astype(np.int)

                blocks, point_block_indices, block_point_counts = np.unique(
                    xyz_blocks, return_inverse=True, return_counts=True, axis=0
                )
                block_point_indices = np.split(
                    np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1])
                )

                block_to_block_idx_map = dict()
                for block_idx in range(blocks.shape[0]):
                    block = (blocks[block_idx][0], blocks[block_idx][1])
                    block_to_block_idx_map[(block[0], block[1])] = block_idx

                # merge small blocks into one of their big neighbors
                block_point_count_threshold = max_point_num / 10
                nbr_block_offsets = [
                    (0, 1),
                    (1, 0),
                    (0, -1),
                    (-1, 0),
                    (-1, 1),
                    (1, 1),
                    (1, -1),
                    (-1, -1),
                ]
                block_merge_count = 0
                for block_idx in range(blocks.shape[0]):
                    if block_point_counts[block_idx] >= block_point_count_threshold:
                        continue

                    block = (blocks[block_idx][0], blocks[block_idx][1])
                    for x, y in nbr_block_offsets:
                        nbr_block = (block[0] + x, block[1] + y)
                        if nbr_block not in block_to_block_idx_map:
                            continue

                        nbr_block_idx = block_to_block_idx_map[nbr_block]
                        if (
                            block_point_counts[nbr_block_idx]
                            < block_point_count_threshold
                        ):
                            continue

                        block_point_indices[nbr_block_idx] = np.concatenate(
                            [
                                block_point_indices[nbr_block_idx],
                                block_point_indices[block_idx],
                            ],
                            axis=-1,
                        )
                        block_point_indices[block_idx] = np.array([], dtype=np.int)
                        block_merge_count = block_merge_count + 1
                        break

                idx_last_non_empty_block = 0
                for block_idx in reversed(range(blocks.shape[0])):
                    if block_point_indices[block_idx].shape[0] != 0:
                        idx_last_non_empty_block = block_idx
                        break

                # uniformly sample each block
                for block_idx in range(idx_last_non_empty_block + 1):
                    point_indices = block_point_indices[block_idx]
                    if point_indices.shape[0] == 0:
                        continue
                    block_points = xyz[point_indices]
                    block_min = np.amin(block_points, axis=0, keepdims=True)
                    xyz_grids = np.floor((block_points - block_min) / grid_size).astype(
                        np.int
                    )
                    grids, point_grid_indices, grid_point_counts = np.unique(
                        xyz_grids, return_inverse=True, return_counts=True, axis=0
                    )
                    grid_point_indices = np.split(
                        np.argsort(point_grid_indices),
                        np.cumsum(grid_point_counts[:-1]),
                    )
                    grid_point_count_avg = int(np.average(grid_point_counts))
                    point_indices_repeated = []
                    for grid_idx in range(grids.shape[0]):
                        point_indices_in_block = grid_point_indices[grid_idx]
                        repeat_num = math.ceil(
                            grid_point_count_avg / point_indices_in_block.shape[0]
                        )
                        if repeat_num > 1:
                            point_indices_in_block = np.repeat(
                                point_indices_in_block, repeat_num
                            )
                            np.random.shuffle(point_indices_in_block)
                            point_indices_in_block = point_indices_in_block[
                                :grid_point_count_avg
                            ]
                        point_indices_repeated.extend(
                            list(point_indices[point_indices_in_block])
                        )
                    block_point_indices[block_idx] = np.array(point_indices_repeated)
                    block_point_counts[block_idx] = len(point_indices_repeated)
                for block_idx in range(idx_last_non_empty_block + 1):
                    point_indices = block_point_indices[block_idx]
                    if point_indices.shape[0] == 0:
                        continue

                    block_point_num = point_indices.shape[0]
                    block_split_num = int(
                        math.ceil(block_point_num * 1.0 / max_point_num)
                    )
                    point_num_avg = int(
                        math.ceil(block_point_num * 1.0 / block_split_num)
                    )
                    point_nums = [point_num_avg] * block_split_num
                    point_nums[-1] = block_point_num - (
                        point_num_avg * (block_split_num - 1)
                    )
                    starts = [0] + list(np.cumsum(point_nums))

                    np.random.shuffle(point_indices)
                    block_points = xyz[point_indices]
                    block_min = np.amin(block_points, axis=0, keepdims=True)
                    block_max = np.amax(block_points, axis=0, keepdims=True)
                    block_center = (block_min + block_max) / 2
                    block_center[0][-1] = block_min[0][-1]
                    unnormalized_block_points = block_points.copy()
                    block_points = (
                        block_points - block_center
                    )  # align to block bottom center
                    x, y, z = np.split(block_points, (1, 2), axis=-1)

                    block_xzyrgbi = np.concatenate(
                        [x, z, y]
                        + [
                            i[point_indices][:, None]
                            for i in other_features.transpose(1, 0)
                        ],
                        axis=-1,
                    )  # XYZ, Intensity, NumReturns, RGB
                    block_labels = labels[point_indices]

                    ## unormalized points
                    x_u, y_u, z_u = np.split(unnormalized_block_points, (1, 2), axis=-1)
                    unnormalized_block_xzyrgbi = np.concatenate(
                        [x_u, z_u, y_u]
                        + [
                            i[point_indices][:, None]
                            for i in other_features.transpose(1, 0)
                        ],
                        axis=-1,
                    )

                    for block_split_idx in range(block_split_num):
                        start = starts[block_split_idx]
                        point_num = point_nums[block_split_idx]
                        end = start + point_num
                        idx_in_batch = idx % batch_size
                        data[idx_in_batch, 0:point_num, ...] = block_xzyrgbi[
                            start:end, :
                        ]
                        unnormalized_data[
                            idx_in_batch, 0:point_num, ...
                        ] = unnormalized_block_xzyrgbi[start:end, :]
                        data_num[idx_in_batch] = point_num
                        label[idx_in_batch] = dataset_idx  # won't be used...
                        label_seg[idx_in_batch, 0:point_num] = block_labels[start:end]
                        indices_split_to_full[
                            idx_in_batch, 0:point_num
                        ] = point_indices[start:end]

                        if ((idx + 1) % batch_size == 0) or (
                            block_idx == idx_last_non_empty_block
                            and block_split_idx == block_split_num - 1
                        ):
                            item_num = idx_in_batch + 1
                            filename_h5 = os.path.join(
                                output_path,
                                Path(folder).stem,
                                dataset + "_%s_%d.h5" % (offset_name, idx_h5),
                            )
                            file = h5py.File(filename_h5, "w")
                            file.create_dataset(
                                "unnormalized_data",
                                data=unnormalized_data[0:item_num, ...],
                            )
                            file.create_dataset("data", data=data[0:item_num, ...])
                            file.create_dataset(
                                "data_num", data=data_num[0:item_num, ...]
                            )
                            file.create_dataset("label", data=label[0:item_num, ...])
                            file.create_dataset(
                                "label_seg", data=label_seg[0:item_num, ...]
                            )
                            file.create_dataset(
                                "indices_split_to_full",
                                data=indices_split_to_full[0:item_num, ...],
                            )
                            file.create_dataset("block_center", data=block_center)
                            file.close()
                            idx_h5 = idx_h5 + 1
                        idx = idx + 1
    if segregate:
        ## Segregate data
        output_path = Path(output_path)
        path_convert = output_path

        GROUND_CLASS = 0
        mb = master_bar(range(len(folders)))
        meta_file = {}
        meta_file["files"] = {}
        all_classes = set()
        folder_classes = {}
        for itn in mb:
            current_folder_classes = set()
            folder = folders[itn]
            path = output_path / Path(folder).stem
            total = 0
            # mb.write(f'{itn + 1}. Segregating {Path(folder).stem} folder')
            all_files = list(path.glob("*.h5"))
            file_id = 0
            for idx, fn in enumerate(progress_bar(all_files, parent=mb)):
                file = h5py.File(fn, "r")
                data = file["data"]
                total += data.shape[0]
                label_seg = file["label_seg"]
                data_num = file["data_num"]
                unnormalized_data = file["unnormalized_data"]
                block_center = file["block_center"][:]
                file_idxs = []
                all_unique_classes = []
                total_points = []
                for i in range(file["data_num"][:].shape[0]):
                    save_file = (
                        path_convert / Path(folder).stem / (fn.stem + f"_{i}" + ".h5")
                    )
                    new_file = h5py.File(save_file, mode="w")
                    new_file.create_dataset(
                        "unnormalized_data", data=unnormalized_data[i]
                    )
                    new_file.create_dataset("data", data=data[i])
                    new_file.create_dataset("label_seg", data=label_seg[i])
                    new_file.create_dataset("data_num", data=data_num[i])
                    new_file.close()
                    unique_classes = np.unique(label_seg[i][: data_num[i]]).tolist()
                    all_classes = all_classes.union(unique_classes)
                    current_folder_classes = current_folder_classes.union(
                        unique_classes
                    )
                    all_unique_classes.append(unique_classes)
                    total_points.append(int(data_num[i]))
                    file_idxs.append(i)
                meta_file["files"][os.path.join(*fn.parts[-2:])] = {
                    "idxs": file_idxs,
                    "block_center": block_center.tolist(),
                    "unique_classes": all_unique_classes,
                    "total_points": total_points,
                }
                folder_classes[Path(folder).name] = list(current_folder_classes)
                file.close()
                os.remove(fn)
        meta_file["num_classes"] = len(all_classes)
        meta_file["classes"] = list(all_classes)
        meta_file["max_point"] = max_point_num
        meta_file["num_extra_dim"] = len(extra_features)
        meta_file["extra_features"] = extra_features
        meta_file["block_size"] = block_size
        meta_file["folder_classes"] = folder_classes
        with open(output_path / "meta.json", "w") as f:
            json.dump(meta_file, f)

    if kwargs.get("print_it", True):
        print("Export finished.")

    return output_path


## Segregated data ItemList


def open_h5py_tensor(fn, keys=["data"]):
    try_import("h5py")
    import h5py

    data_label = []
    file = h5py.File(fn, "r")
    for key in keys:
        tensor = torch.tensor(file[key][...]).float()
        data_label.append(tensor)

    file.close()
    return data_label  ## While getting a specific index from the file


## It also stores the label so that we don't have to open the file twice.
class DataStore:
    indexes = False
    pass


class PointCloudItemList(ItemList):
    def __init__(self, items, **kwargs):
        if DataStore.indexes is False and (not isinstance(DataStore.indexes, list)):
            DataStore.indexes = kwargs.get("extra_feat_indexes")
        elif kwargs.get(
            "extra_feat_indexes"
        ) is not None and DataStore.indexes != kwargs.get("extra_feat_indexes"):
            DataStore.indexes = kwargs.get("extra_feat_indexes", None)
        kwargs.pop("extra_feat_indexes", None)
        super().__init__(items, **kwargs)
        self.keys = ["data", "label_seg", "data_num"]

    def get(self, i):
        indexes = DataStore.indexes
        data = self.open(self.items[i])
        DataStore.i = i
        DataStore.data = data
        if indexes is None:
            return (data[0], data[2])
        else:
            return (data[0][:, indexes], data[2])

    def open(self, fn):
        return open_h5py_tensor(fn, keys=self.keys)


def remap_labels(labels, class2idx):
    """
    Remaps labels from non-contigous space to contiguous.
    """
    if isinstance(labels, torch.Tensor):
        remapped_label = torch.zeros_like(labels)
    else:
        remapped_label = np.zeros_like(labels)
    for k, v in class2idx.items():
        remapped_label[labels == k] = v
    return remapped_label


class PointCloudLabelList(ItemList):
    def __init__(self, items, remap=False, class2idx={}, **kwargs):
        super().__init__(items, **kwargs)
        self.key = "label_seg"
        self.remap = remap
        self.class2idx = class2idx

    def get(self, i):
        labels = DataStore.data[1].long()
        if self.remap:
            return remap_labels(labels, self.class2idx)
        else:
            return labels

    def analyze_pred(self, pred):
        return pred.argmax(dim=1)


PointCloudItemList._label_cls = PointCloudLabelList


def filter_files(fname, meta, classes_to_check, min_points):
    idx = int(str(fname)[-4])
    fname = Path(fname)
    # Do not filter validation files.
    if fname.parent.name != "train":
        return True

    key = os.path.join(fname.parent.name, re.sub("_[0-9]+\.h5", "", fname.name) + ".h5")
    # print(key, fname.name, idx)
    classes_present = meta["files"][key]["unique_classes"][idx]
    is_present_COI = any([k in classes_present for k in classes_to_check])
    if classes_to_check == []:
        is_present_COI = True
    is_present_min_points = meta["files"][key]["total_points"][idx] > min_points
    is_present = is_present_COI and is_present_min_points
    return is_present


def raise_class_mismatch_warning(train_classes, valid_classes, remap_classes):
    train_classes_mapped = list(set([remap_classes.get(c, c) for c in train_classes]))
    valid_classes_mapped = list(set([remap_classes.get(c, c) for c in valid_classes]))

    if sorted(train_classes) != sorted(valid_classes) and remap_classes == {}:
        warnings.warn(
            "Classes in your training and validation datasets are not same. "
            "This will not affect training but will result in poor validation metrics. "
            f"Got class codes {train_classes} in training dataset and {valid_classes} in validation dataset. "
            "The model will be trained on union of the two class lists."
            "If required, use the `remap_classes` parameter to map the extra class to one of the other classes."
        )

    elif sorted(train_classes_mapped) != sorted(valid_classes_mapped):
        warnings.warn(
            "Classes in your training and validation datasets are not same. "
            "Remapped classes do not match. "
            f"Got mapped class codes {train_classes_mapped} in training dataset and "
            f"{valid_classes_mapped} in validation dataset. "
        )


def merge_classes(json_train, json_val, remap_classes):
    class_train = [c["classCode"] for c in json_train["classification"]["table"]]
    class_val = [c["classCode"] for c in json_val["classification"]["table"]]
    if sorted(class_train) != sorted(class_val):
        raise_class_mismatch_warning(class_train, class_val, remap_classes)
    classification = json_train["classification"]

    for c in json_val["classification"]["table"]:
        if c["classCode"] not in class_train:
            classification["table"].append(c)

    classification["table"] = sorted(
        classification["table"], key=lambda x: x["classCode"]
    )
    classification["max"] = max(
        json_train["classification"]["max"], json_val["classification"]["max"]
    )
    classification["min"] = min(
        json_train["classification"]["min"], json_val["classification"]["min"]
    )
    return classification


# Prepare data called in _data.py
def pointcloud_prepare_data(
    path,
    class_mapping,
    batch_size,
    val_split_pct,
    dataset_type="PointCloud",
    transform_fn=None,
    **kwargs,
):
    try_imports(["h5py", "plotly", "laspy"])
    databunch_kwargs = {"num_workers": 0} if sys.platform == "win32" else {}
    if (path / "Statistics.json").exists():
        dataset_type = "PointCloud"
        already_split = False
    if (path / "train" / "Statistics.json").exists() and (
        path / "val" / "Statistics.json"
    ).exists():
        dataset_type = "PointCloud"
        already_split = True
    elif (path / "meta.json").exists():
        dataset_type = "PointCloud_TF"
    else:
        dataset_type = "Unknown"

    extra_features = kwargs.get("extra_features", None)
    if class_mapping is not None:
        class_mapping = {int(k): v for k, v in class_mapping.items()}

    if dataset_type == "PointCloud":
        if already_split:
            # write code to merge json.
            with open(path / "train" / "Statistics.json") as f:
                json_train = json.load(f)

            with open(path / "val" / "Statistics.json") as f:
                json_val = json.load(f)

            classification = merge_classes(
                json_train, json_val, kwargs.get("remap_classes", {})
            )
            json_train["classification"] = classification

            pointcloud_dataset_train = PointCloudDataset(
                path, class_mapping, json_train, folder="train", **kwargs
            )
            pointcloud_dataset_val = PointCloudDataset(
                path, class_mapping, json_train, folder="val", **kwargs
            )
            train_dl = DataLoader(
                pointcloud_dataset_train, batch_size=batch_size, **databunch_kwargs
            )
            valid_dl = DataLoader(
                pointcloud_dataset_val, batch_size=batch_size, **databunch_kwargs
            )
            device = get_device()
            data = DataBunch(train_dl, valid_dl, device=device)
            data.show_batch = types.MethodType(show_point_cloud_batch, data)
            data.path = data.train_ds.path

            data.subset_classes = []
            data.remap = data.train_ds.remap

            # data.val_files = val_files
        else:
            with open(path / "Statistics.json") as f:
                json_file = json.load(f)

            max_points = json_file["parameters"]["maxPoints"]
            pointcloud_dataset = PointCloudDataset(
                path, class_mapping, json_file, **kwargs
            )
            # Splitting in train and test based on files.
            total_files = len(pointcloud_dataset.filenames)
            total_files_idxs = list(range(total_files))
            random.shuffle(total_files_idxs)
            total_val_files = int(val_split_pct * total_files)
            val_files = total_files_idxs[-total_val_files:]
            if total_val_files == 0:
                raise Exception(
                    "No files could be added to validation dataset. Please increase the value of `val_split_pct`"
                )
            tile_file_indices = pointcloud_dataset.tiles[:, 0]
            val_indices = torch.from_numpy(
                np.isin(tile_file_indices, val_files)
            ).nonzero()
            train_indices = torch.from_numpy(
                np.logical_not(np.isin(tile_file_indices, val_files))
            ).nonzero()
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            train_dl = DataLoader(
                pointcloud_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                **databunch_kwargs,
            )
            valid_dl = DataLoader(
                pointcloud_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                **databunch_kwargs,
            )
            device = get_device()
            data = DataBunch(train_dl, valid_dl, device=device)
            data.show_batch = types.MethodType(show_point_cloud_batch, data)
            data.path = data.train_ds.path
            data.val_files = val_files

    elif dataset_type == "PointCloud_TF":
        with open(Path(path) / "meta.json", "r") as f:
            meta = json.load(f)
        classes = sorted(meta["classes"])
        remap_classes = kwargs.get("remap_classes", {})
        min_points = kwargs.get("min_points", 0)
        classes_of_interest = kwargs.get("classes_of_interest", [])
        background_classcode = kwargs.get("background_classcode", None)
        if background_classcode is not None and classes_of_interest == []:
            raise Exception(
                "`background_classcode can only be used when `classes_of_interest` is passed."
            )

        if "folder_classes" in meta.keys():
            folder_classes = meta["folder_classes"]
            if sorted(folder_classes["train"]) != sorted(folder_classes["val"]):
                raise_class_mismatch_warning(
                    folder_classes["train"], folder_classes["val"], remap_classes
                )

        if remap_classes != {}:
            to_be_remapped = list(remap_classes.keys())
            if not all([c in classes for c in to_be_remapped]):
                raise Exception(
                    f"Remapping keys {to_be_remapped} must be a subset of classes available i.e {list(classes)}"
                )

        remap = False

        keys = list(meta["files"].keys())
        prepare_data_feature_warning = False
        if "unique_classes" not in meta["files"][keys[0]].keys():
            old_exported_data_warning = True
        else:
            old_exported_data_warning = False

        full_class_mapping = {int(v): str(v) for v in classes}
        orig_classes = list(full_class_mapping.keys())
        # account for remapping here.
        if remap_classes != {}:
            remap = True
            mapped_classes = set(remap_classes.values())
            unmapped_classes = set(
                [c for c in full_class_mapping.keys() if c not in remap_classes.keys()]
            )
            all_classes = mapped_classes.union(unmapped_classes)
            full_class_mapping = {int(c): str(c) for c in all_classes}

        filter_classes = classes_of_interest != []

        unexpanded_COI = classes_of_interest
        if filter_classes:
            if not all([c in full_class_mapping.keys() for c in unexpanded_COI]):
                raise Exception(
                    f"classes_of_interest {classes_of_interest} must be a subset of {list(full_class_mapping.keys())}"
                )

        # class mapping is string to integer.
        if class_mapping is None:
            class_mapping = full_class_mapping
        else:
            class_mapping = {
                int(k): class_mapping.get(k, v) for k, v in full_class_mapping.items()
            }

        if classes_of_interest != [] and background_classcode is not None:
            # filter class mapping
            class_mapping = {
                k: v for k, v in class_mapping.items() if k in classes_of_interest
            }
            class_mapping[background_classcode] = "background"

        class_mapping = {int(k): str(v) for k, v in class_mapping.items()}

        important_classes = list(class_mapping.keys())
        important_classes = important_classes
        remap_dict = remap_classes
        remap_bool = remap_classes is not {}

        class2idx = {
            value: idx for idx, value in enumerate(sorted(list(class_mapping.keys())))
        }
        ## Helper attributes for remapping
        c = len(class2idx)
        if remap is False:
            remap = to_remap_classes(list(class2idx.keys()))
        classes = list(class2idx.keys())

        if background_classcode is not None:
            remap = True
            ## get classes that belong to background and remap to bg code.
            bg_classes = get_background_classes(full_class_mapping, classes_of_interest)
            # map original classes to the index to which bg would have been mapped
            bg_mapping = {k: class2idx[background_classcode] for k in bg_classes}
            _bg_mapping = bg_mapping
            # remove bg code from class2idx
            class2idx.pop(background_classcode)
            # merge the original with the bgmapped
            class2idx = {**class2idx, **bg_mapping}

        if remap_classes != {}:
            # if remap_classes is {7:25, 3:25}, the inverse will be
            # {25:[7,3]}
            inverse_remap_classes = compute_inverse_remap(remap_classes)
            if filter_classes:
                classes_of_interest = expand_classes_of_interest(
                    classes_of_interest, inverse_remap_classes
                )
            # print(inverse_remap_classes)
            repeated_mapping = {}
            for k, v in inverse_remap_classes.items():
                for c in v:
                    repeated_mapping[c] = class2idx[k]
            class2idx = {**class2idx, **repeated_mapping}

        # Color mapping.
        color_mapping = kwargs.get(
            "color_mapping",
            {
                k: [random.choice(range(256)) for _ in range(3)]
                for k, _ in class_mapping.items()
            },
        )
        color_mapping = {int(k): v for k, v in color_mapping.items()}
        idx2class = {i: c for i, c in enumerate(sorted(classes))}
        class2idx = {k: v for k, v in class2idx.items() if k in orig_classes}
        extra_features = meta["extra_features"]
        string_mapped_features = {
            "numberOfReturns": "num_returns",
            "returnNumber": "return_num",
            "nearInfrared": "nir",
        }
        inverse_string_mapped_features = {
            v: k for k, v in string_mapped_features.items()
        }
        extra_features_users = kwargs.get("extra_features", [])
        extra_features_users_mapped = [
            inverse_string_mapped_features.get(f, f) for f in extra_features_users
        ]
        extra_feature_keys = [f[0] for f in extra_features]
        extra_features_keys_mapped = [
            inverse_string_mapped_features.get(f, f) for f in extra_feature_keys
        ]
        # +3 to adjust for xyz.
        extra_feat_indexes = [
            i + 3
            for i, f in enumerate(extra_features)
            if inverse_string_mapped_features.get(f[0], f[0])
            in extra_features_users_mapped
        ]
        if not all(
            [c in extra_features_keys_mapped for c in extra_features_users_mapped]
        ):
            raise Exception(
                f"extra_features {extra_features_users} must be a subset of {extra_features_keys_mapped}"
            )

        # filter features to keep
        extra_features_keys_mapped = [
            k for k in extra_features_keys_mapped if k in extra_features_users_mapped
        ]
        extra_features = [
            e
            for e in extra_features
            if inverse_string_mapped_features.get(e[0], e[0])
            in extra_features_users_mapped
        ]

        extra_feat_indexes = [0, 1, 2] + extra_feat_indexes
        src = PointCloudItemList.from_folder(
            path, [".h5"], extra_feat_indexes=extra_feat_indexes
        )
        if classes_of_interest != [] or min_points > 0:
            if old_exported_data_warning:
                warnings.warn(
                    "You are using exported data from an older version of the library. "
                    "Ignoring `classes_of_interest` and `min_points` parameters. "
                    "To use these features, please export your data again."
                )
            else:
                src = src.filter_by_func(
                    partial(
                        filter_files,
                        meta=meta,
                        classes_to_check=classes_of_interest,
                        min_points=min_points,
                    )
                )

        train_idxs = [i for i, p in enumerate(src.items) if p.parent.name == "train"]
        val_idxs = [i for i, p in enumerate(src.items) if p.parent.name == "val"]
        src = src.split_by_idxs(train_idxs, val_idxs).label_from_func(
            lambda x: x, remap=remap, class2idx=class2idx
        )
        device = get_device()
        data = src.databunch(bs=batch_size, device=device, **databunch_kwargs)
        data.meta = meta
        data.subset_classes = [classes_of_interest, background_classcode]
        data.remap_dict = remap_classes
        data.remap_classes = remap_classes
        data.remap_bool = remap_bool
        data.classes_of_interest = classes_of_interest
        data.background_classcode = background_classcode
        data.important_classes = classes_of_interest
        data.remap = remap
        data.classes = classes
        data.c = len(class_mapping)
        data.show_batch = types.MethodType(show_point_cloud_batch_TF, data)
        data.color_mapping = color_mapping
        data.color_mapping = {int(k): v for k, v in data.color_mapping.items()}
        data.color_mapping = recompute_color_mapping(data.color_mapping, data.classes)
        data.class2idx = class2idx
        data.idx2class = idx2class
        data.max_point = data.meta["max_point"]
        data.extra_dim = len(extra_features_users)
        data.extra_features = extra_features
        data.extra_feat_indexes = extra_feat_indexes
        data.features_to_keep = extra_features_keys_mapped
        data.block_size = data.meta["block_size"]
        data.class_mapping = class_mapping
        ## To accomodate save function to save in correct directory
        data.path = data.path / "train"
    else:
        raise Exception("Could not infer dataset type.")

    data.pc_type = dataset_type
    data.path = data.train_ds.path
    ## Below are the lines to make save function work
    data.chip_size = None
    data._image_space_used = None
    data.dataset_type = dataset_type
    data.transform_fn = transform_fn
    data.init_kwargs = kwargs
    return data


def prediction_remap_classes(labels, reclassify_classes, inverse_class2idx):
    labels = np.vectorize(inverse_class2idx.get)(labels)


def read_xyz_label_from_las(filename_las):
    try_import("laspy")
    import laspy

    msg = "Loading {}...".format(filename_las)
    f = laspy.file.File(filename_las, mode="r")
    h = f.header
    xyzirgb_num = h.point_records_count
    xyz_offset = h.offset
    encoding = h.encoding
    xyz = np.ndarray((xyzirgb_num, 3))
    labels = np.ndarray(xyzirgb_num, np.int16)
    i = 0
    for p in f:
        xyz[i] = [p.x, p.y, p.z]
        labels[i] = p.classification
        i += 1
    f.close()
    return xyz, labels, xyzirgb_num, xyz_offset, encoding


def save_xyz_label_to_las(filename_las, xyz, xyz_offset, encoding, labels):
    try_import("laspy")
    import laspy

    msg = "Saving {}...".format(filename_las)
    h = laspy.header.Header()
    h.dataformat_id = 1
    h.major = 1
    h.minor = 2
    h.min = np.min(xyz, axis=0)
    h.max = np.max(xyz, axis=0)
    h.scale = [1e-3, 1e-3, 1e-3]
    h.offset = xyz_offset
    h.encoding = encoding

    f = laspy.file.File(filename_las, mode="w", header=h)
    for i in range(xyz.shape[0]):
        p = laspy.point.Point()
        p.x = xyz[i, 0] / h.scale[0]
        p.y = xyz[i, 1] / h.scale[1]
        p.z = xyz[i, 2] / h.scale[2]
        p.classification = labels[i]
        p.color = laspy.color.Color()
        p.intensity = 100
        p.return_number = 1
        p.number_of_returns = 1
        p.scan_direction = 1
        p.scan_angle = 0
        f.write(p)

    f.close()


def prediction_remap_classes(labels, reclassify_classes, inverse_class_mapping):
    labels = np.vectorize(inverse_class_mapping.get)(labels)
    if reclassify_classes == {}:
        return labels
    else:
        labels = np.vectorize(reclassify_classes.get)(labels)
        return labels


def prediction_selective_classify(labels, classification, selective_classify):
    all_indexes = list(range(len(labels)))
    return np.vectorize(
        lambda i: labels[i] if labels[i] in selective_classify else classification[i]
    )(all_indexes)


def preserved_overwrite(orig_classf, labels, preserve_classes):
    """
    Does not write class code which is specified
    in preserve_classes param
    """
    bool_mat = np.concatenate(
        [orig_classf[:, None] == c for c in preserve_classes], axis=1
    )
    mask = np.any(bool_mat, axis=1)
    orig_classf[~mask] = labels[~mask]
    return orig_classf


def write_resulting_las(
    in_las_filename,
    out_las_filename,
    labels,
    num_classes,
    data,
    print_metrics,
    reclassify_classes={},
    selective_classify=[],
    preserve_classes=[],
):
    try_import("laspy")
    import laspy

    false_positives = [0] * num_classes
    true_positives = [0] * num_classes
    false_negatives = [0] * num_classes
    if hasattr(data, "idx2class"):
        inverse_class2idx = data.idx2class
    else:
        inverse_class2idx = {v: k for k, v in data.class2idx.items()}
    shutil.copy(in_las_filename, out_las_filename)

    with laspy.file.File(in_las_filename, mode="r") as f:
        gt_classification = f.classification.copy()

    with laspy.file.File(out_las_filename, mode="rw") as f_out:
        classification = []
        warn_flag = False

        ## remap classes
        old_labels = labels.copy()
        labels = prediction_remap_classes(labels, reclassify_classes, inverse_class2idx)

        if print_metrics:
            for i in range(len(gt_classification)):
                p_classification = gt_classification[i]
                current_class = inverse_class2idx[old_labels[i]]
                if reclassify_classes != {}:
                    current_class = reclassify_classes[current_class]
                try:
                    false_positives[old_labels[i]] += int(
                        p_classification != current_class
                    )
                    true_positives[old_labels[i]] += int(
                        p_classification == current_class
                    )
                    false_negatives[data.class2idx[p_classification]] += int(
                        p_classification != current_class
                    )
                except (IndexError, KeyError) as _:
                    warn_flag = True

                i += 1

        if selective_classify != []:
            # current_class if current_class in selective_classify else p.classification
            labels = prediction_selective_classify(
                labels, gt_classification, selective_classify
            )

        if preserve_classes != []:
            labels = preserved_overwrite(gt_classification, labels, preserve_classes)

        f_out.classification = labels.tolist()

    # if print_metrics and warn_flag:
    #     logger.warning(f"Some classes in your las file {in_las_filename} do not match the classes the model is trained on")
    #     print_metrics = False
    return false_positives, true_positives, false_negatives


def calculate_metrics(false_positives, true_positives, false_negatives):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = np.divide(true_positives, np.add(true_positives, false_positives))
        recall = np.divide(true_positives, np.add(true_positives, false_negatives))
        f_1 = np.multiply(
            2.0, np.divide(np.multiply(precision, recall), np.add(precision, recall))
        )
    return precision, recall, f_1


def get_pred_prefixes(datafolder):
    fs = os.listdir(datafolder)
    preds = []
    for f in fs:
        if f[-8:] == "_pred.h5":
            preds += [f]
    pred_pfx = []
    for p in preds:
        to_check = "_half"  # "_zero"
        if to_check in p:
            pred_pfx += [p.rsplit(to_check, 1)[0]]
    return np.unique(pred_pfx)


def get_predictions(
    pointcnn_model, data, batch_idx, points_batch, sample_num, batch_size, point_num
):
    ## Getting sampling indices
    tile_num = math.ceil((sample_num * batch_size) / point_num)
    indices_shuffle = np.tile(np.arange(point_num), tile_num)[
        0 : sample_num * batch_size
    ]
    np.random.shuffle(indices_shuffle)
    indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))

    model_input = torch.cat(
        [points_batch[i, s[:, 0]][None] for i, s in enumerate(indices_batch_shuffle)],
        dim=0,
    )

    seg_probs = model_predictions(pointcnn_model, model_input, point_num)

    probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))  ## Complete probs
    predictions = [(-1, 0.0, None)] * point_num  ## predictions

    ## Assigning the confidences and labels to the appropriate index.
    for idx in range(sample_num * batch_size):
        point_idx = indices_shuffle[idx]
        probs = probs_2d[idx, :]
        confidence = np.amax(probs)
        label = np.argmax(probs)
        if confidence > predictions[point_idx][1]:
            predictions[point_idx] = [label, confidence, probs]

    return predictions


def inference_las(
    path,
    pointcnn_model,
    out_path=None,
    print_metrics=False,
    remap_classes={},
    selective_classify=[],
    preserve_classes=[],
):
    try_import("h5py")
    import h5py
    import pandas as pd

    # check if the model is trained on new exported data and raise Exception.
    if hasattr(pointcnn_model._data, "pc_type"):
        if pointcnn_model._data.pc_type == "PointCloud":
            raise Exception(
                "Models trained on exported data from ArcGIS Pro 2.8 onwards are not supported. "
                "Use `Classify Points Using Trained Model` tool available in 3D Analyst "
                "extension in ArcGIS Pro 2.8 onwards."
            )

    try:
        ## Export data
        path = Path(path)

        if len(list(path.glob("*.las"))) == 0:
            raise Exception(f"The given path({path}) contains no las files.")

        reclassify_classes = remap_classes
        if reclassify_classes != {}:
            if not all(
                [k in pointcnn_model._data.classes for k in reclassify_classes.keys()]
            ):
                raise Exception(
                    f"`remap_classes` dictionary keys are not present in dataset with classes {pointcnn_model._data.classes}."
                )
            reclassify_classes = {
                k: reclassify_classes.get(k, k) for k in pointcnn_model._data.class2idx
            }

        if out_path is None:
            out_path = path / "results"
        else:
            out_path = Path(out_path)

        if selective_classify != []:
            if reclassify_classes != {}:
                values_to_check = np.unique(
                    np.array(list(reclassify_classes.values()))
                ).tolist()
            else:
                values_to_check = list(pointcnn_model._data.classes)

            if not all([k in values_to_check for k in selective_classify]):
                raise Exception(
                    f"`selective_classify` can only contain values from these class values {values_to_check}."
                )

        prepare_las_data(
            path.parent,
            block_size=pointcnn_model._data.block_size[0],
            max_point_num=pointcnn_model._data.max_point,
            output_path=path.parent,
            extra_features=pointcnn_model._data.extra_features,
            folder_names=[path.stem],
            segregate=False,
            print_it=False,
        )
        ## Predict and postprocess
        max_point_num = pointcnn_model._data.max_point
        sample_num = pointcnn_model.sample_point_num
        batch_size = 1 * math.ceil(max_point_num / sample_num)
        filenames = list(glob.glob(str(path / "*.h5")))

        mb = master_bar(range(len(filenames)))
        for itn in mb:
            filename = filenames[itn]
            with h5py.File(filename, "r") as data_h5:
                has_indices = "indices_split_to_full" in data_h5
                data = data_h5["data"][...].astype(np.float32)
                data_num = data_h5["data_num"][...].astype(np.int32)
                indices_split_to_full = data_h5["indices_split_to_full"][...]
            batch_num = data.shape[0]
            labels_pred = np.full((batch_num, max_point_num), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, max_point_num), dtype=np.float32)

            for batch_idx in progress_bar(range(batch_num), parent=mb):
                points_batch = data[[batch_idx] * batch_size, ...]
                point_num = data_num[batch_idx]
                predictions = get_predictions(
                    pointcnn_model,
                    data,
                    batch_idx,
                    points_batch,
                    sample_num,
                    batch_size,
                    point_num,
                )
                labels_pred[batch_idx, 0:point_num] = np.array(
                    [label for label, _, _ in predictions]
                )
                confidences_pred[batch_idx, 0:point_num] = np.array(
                    [confidence for _, confidence, _ in predictions]
                )

            ## Saving h5 predictions file
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            filename_pred = os.path.join(out_path, Path(filename).stem + "_pred.h5")
            with h5py.File(filename_pred, "w") as file:
                file.create_dataset("data_num", data=data_num)
                file.create_dataset("label_seg", data=labels_pred)
                file.create_dataset("confidence", data=confidences_pred)
                if has_indices:
                    file.create_dataset(
                        "indices_split_to_full", data=indices_split_to_full
                    )

        ## Merge H5 files and write las files
        SAVE_TO_EXT = ".las"
        LOAD_FROM_EXT = ".las"

        categories_list = get_pred_prefixes(out_path)

        global_false_positives = [0] * pointcnn_model._data.c
        global_true_positives = [0] * pointcnn_model._data.c
        global_false_negatives = [0] * pointcnn_model._data.c

        for category in categories_list:
            output_path = os.path.join(out_path, category + "_pred" + SAVE_TO_EXT)
            if not os.path.exists(os.path.join(out_path)):
                os.makedirs(os.path.join(out_path))
            pred_list = [
                pred
                for pred in os.listdir(out_path)
                if category in pred
                and pred.rsplit(".", 1)[0].split("_")[-1] == "pred"
                and pred[-3:] == ".h5"
            ]
            merged_label = None
            merged_confidence = None

            for pred_file in pred_list:
                with h5py.File(os.path.join(out_path, pred_file), mode="r") as data:
                    labels_seg = data["label_seg"][...].astype(np.int64)
                    indices = data["indices_split_to_full"][...].astype(np.int64)
                    confidence = data["confidence"][...].astype(np.float32)
                    data_num = data["data_num"][...].astype(np.int64)

                if merged_label is None:
                    # calculating how many labels need to be there in the output
                    label_length = 0
                    for i in range(indices.shape[0]):
                        label_length = np.max(
                            [label_length, np.max(indices[i][: data_num[i]])]
                        )
                    label_length += 1
                    merged_label = np.zeros((label_length), dtype=int)
                    merged_confidence = np.zeros((label_length), dtype=float)
                else:
                    label_length2 = 0
                    for i in range(indices.shape[0]):
                        label_length2 = np.max(
                            [label_length2, np.max(indices[i][: data_num[i]])]
                        )
                    label_length2 += 1
                    if label_length < label_length2:
                        # expanding labels and confidence arrays, as the new file appears having more of them.
                        labels_more = np.zeros(
                            (label_length2 - label_length), dtype=merged_label.dtype
                        )
                        conf_more = np.zeros(
                            (label_length2 - label_length),
                            dtype=merged_confidence.dtype,
                        )
                        merged_label = np.append(merged_label, labels_more)
                        merged_confidence = np.append(merged_confidence, conf_more)
                        label_length = label_length2

                for i in range(labels_seg.shape[0]):
                    temp_label = np.zeros((data_num[i]), dtype=int)
                    pred_confidence = confidence[i][: data_num[i]]
                    temp_confidence = merged_confidence[indices[i][: data_num[i]]]

                    temp_label[temp_confidence >= pred_confidence] = merged_label[
                        indices[i][: data_num[i]]
                    ][temp_confidence >= pred_confidence]
                    temp_label[pred_confidence > temp_confidence] = labels_seg[i][
                        : data_num[i]
                    ][pred_confidence > temp_confidence]

                    merged_confidence[
                        indices[i][: data_num[i]][pred_confidence > temp_confidence]
                    ] = pred_confidence[pred_confidence > temp_confidence]
                    merged_label[indices[i][: data_num[i]]] = temp_label

            if len(pred_list) > 0:
                # concatenating source points with the final labels and writing out resulting file
                points_path = os.path.join(path, category + LOAD_FROM_EXT)

                false_positives, true_positives, false_negatives = write_resulting_las(
                    points_path,
                    output_path,
                    merged_label,
                    pointcnn_model._data.c,
                    pointcnn_model._data,
                    print_metrics,
                    reclassify_classes,
                    selective_classify,
                    preserve_classes,
                )
                global_false_positives = np.add(global_false_positives, false_positives)
                global_true_positives = np.add(global_true_positives, true_positives)
                global_false_negatives = np.add(global_false_negatives, false_negatives)

        if print_metrics:
            index = ["precision", "recall", "f1_score"]
            if hasattr(pointcnn_model._data, "idx2class"):
                inverse_class2idx = pointcnn_model._data.idx2class
            else:
                inverse_class2idx = {
                    v: k for k, v in pointcnn_model._data.class2idx.items()
                }
            unique_mapped_classes = np.unique(
                np.array(list(reclassify_classes.values()))
            )
            if (
                len(unique_mapped_classes) == len(pointcnn_model._data.classes)
                or remap_classes == {}
            ):
                precision, recall, f_1 = calculate_metrics(
                    global_false_positives,
                    global_true_positives,
                    global_false_negatives,
                )
                data = [precision, recall, f_1]
                column_names = [
                    inverse_class2idx[cval] for cval in range(pointcnn_model._data.c)
                ]
                if reclassify_classes != {}:
                    remapping_class2idx = {
                        v: reclassify_classes[k]
                        for k, v in pointcnn_model._data.class2idx.items()
                    }
                    column_names = [
                        remapping_class2idx[cval]
                        for cval in range(pointcnn_model._data.c)
                    ]
                df = pd.DataFrame(data, columns=column_names, index=index)
            else:
                inverse_reclassify_classes = {}
                for k, v in reclassify_classes.items():
                    current_value = inverse_reclassify_classes.get(v, [])
                    current_value.append(k)
                    inverse_reclassify_classes[v] = current_value
                map_dict = {
                    u: [
                        pointcnn_model._data.class2idx[k]
                        for k in inverse_reclassify_classes[u]
                    ]
                    for u in unique_mapped_classes
                }
                global_false_positives = recompute_globals(
                    global_false_positives, map_dict
                )
                global_true_positives = recompute_globals(
                    global_true_positives, map_dict
                )
                global_false_negatives = recompute_globals(
                    global_false_negatives, map_dict
                )
                precision, recall, f_1 = calculate_metrics(
                    global_false_positives,
                    global_true_positives,
                    global_false_negatives,
                )
                data = [precision, recall, f_1]
                column_names = list(map_dict.keys())
                df = pd.DataFrame(data, columns=column_names, index=index)

            from IPython.display import display

            display(df)

    except KeyboardInterrupt:
        remove_temp_files(path, out_path)
        raise

    remove_temp_files(path, out_path)

    return out_path


def remove_temp_files(path, out_path):
    for fn in glob.glob(
        str(path / "*.h5"), recursive=True
    ):  ## Remove h5 files in val directory.
        os.remove(fn)

    for fn in glob.glob(
        str(out_path / "*.h5"), recursive=True
    ):  ## Remove h5 files in results directory.
        os.remove(fn)


def recompute_globals(global_count, map_dict):
    return [sum([global_count[ci] for ci in v]) for k, v in map_dict.items()]


def raise_maxpoint_warning(
    idx_file, kwargs, logger, max_display_point, save_html=False
):
    if not save_html:
        if idx_file == 0:
            if "max_display_point" not in kwargs.keys():
                warnings.warn(
                    f"Randomly sampling {max_display_point} points for visualization. You can adjust this using the `max_display_point` parameter."
                )


def get_title_text(idx, save_html, max_display_point):
    title_text = "Ground Truth / Predictions" if idx == 0 else ""
    if save_html:
        title_text = (
            f"Ground Truth / Predictions (Displaying randomly sampled {max_display_point} points.)"
            if idx == 0
            else ""
        )
    return title_text


def show_results(self, rows, color_mapping=None, **kwargs):
    """
    It will plot results from your trained model with ground truth on the
    left and predictions on the right.
    Visualization of data, exported in a geographic coordinate system
    is not yet supported.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional rows. Number of rows to show. Deafults
                            value is 2.
    ---------------------   -------------------------------------------
    color_mapping           Optional dictionary. Mapping from class value
                            to RGB values. Default value example:
                            {0:[220,220,220], 2:[255,0,0], 6:[0,255,0]}.
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    mask_class              Optional array of integers. Array containing
                            class values to mask. Default value is [].
    ---------------------   -------------------------------------------
    width                   Optional integer. Width of the plot. Default
                            value is 750.
    ---------------------   -------------------------------------------
    height                  Optional integer. Height of the plot. Default
                            value is 512
    ---------------------   -------------------------------------------
    max_display_point       Optional integer. Maximum number of points
                            to display. Default is 20000.
    ---------------------   -------------------------------------------
    return_fig              Optional bool. Flag set as True when the matplotlib
                            figure is needed as a return object. Used for TB integration
    ---------------------   -------------------------------------------
    =====================   ===========================================
    """

    filter_outliers = False
    try_import("h5py")
    try_import("plotly")
    import h5py
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import random

    mask_class = kwargs.get("mask_class", [])
    save_html = kwargs.get("save_html", False)
    save_path = kwargs.get("save_path", ".")
    return_fig = kwargs.get("return_fig", False)
    max_display_point = get_max_display_points(self._data, kwargs)
    rows = min(rows, self._data.batch_size)
    color_mapping = self._data.color_mapping if color_mapping is None else color_mapping
    color_mapping = recompute_color_mapping(color_mapping, self._data.classes)
    color_mapping = np.array(list(color_mapping.values())) / 255
    if save_html:
        max_display_point = 20000

    idx = 0
    keys = list(self._data.meta["files"].keys()).copy()
    keys = [f for f in keys if Path(f).parent.stem == "val"]
    random.shuffle(keys)

    for idx_file, fn in enumerate(keys):
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]]
        )
        num_files = self._data.meta["files"][fn]["idxs"]
        block_center = self._data.meta["files"][fn]["block_center"]
        block_center = np.array(block_center)
        block_center[0][2], block_center[0][1] = block_center[0][1], block_center[0][2]
        if num_files == []:
            continue
        if not Path(fn).is_absolute():
            fn = str(self._data._data_path / fn)
        idxs = [h5py.File(fn[:-3] + f"_{i}.h5", "r") for i in num_files]
        pc = []
        labels = []
        pred_class = []
        pred_confidence = []
        for i in idxs:
            # print(f'Running Show Results: Processing {nn+1} of {len(idxs)} blocks.', end='\r')
            current_block = i["unnormalized_data"][:, :3]
            data_num = i["data_num"][()]
            data = i["data"][:]
            pc.append(current_block[:data_num])
            labels.append(i["label_seg"][:data_num])

            max_point_num = self._data.max_point
            sample_num = self.sample_point_num
            batch_size = 1 * math.ceil(max_point_num / sample_num)
            data = data[:, self._data.extra_feat_indexes][None]
            batch_idx = 0
            points_batch = data[[batch_idx] * batch_size, ...]
            point_num = data_num
            predictions = np.array(
                get_predictions(
                    self,
                    data,
                    batch_idx,
                    points_batch,
                    sample_num,
                    batch_size,
                    point_num,
                )
            )
            pred_class.append(predictions[:, 0])
            pred_confidence.append(predictions[:, 1])
            i.close()

        if pc == []:
            continue

        pc = np.concatenate(pc, axis=0)
        labels = np.concatenate(labels, axis=0)
        unmapped_labels = labels.copy().astype(int)
        if self._data.remap:
            labels = remap_labels(labels, self._data.class2idx)
            unmapped_labels = remap_labels(
                labels.copy().astype(int), self._data.idx2class
            )
        pred_class = np.concatenate(pred_class, axis=0).astype(int)
        unmapped_pred_class = remap_labels(
            pred_class.copy().astype(int), self._data.idx2class
        )
        sample_idxs = mask_classes(
            labels=labels,
            mask_class=mask_class,
            classes=self._data.classes,
            class2idx=self._data.class2idx if self._data.remap else None,
            remap_classes=self._data.remap_classes if self._data.remap else None,
        )
        sampled_pc = pc[sample_idxs]
        if sampled_pc.shape[0] == 0:
            continue
        x, y, z = recenter(sampled_pc).transpose(1, 0)
        if filter_outliers:
            ## Filter on the basis of std.
            mask = filter_pc(pc)
        else:
            ## all points
            mask = x != None

        if sample_idxs.sum() > max_display_point:
            raise_maxpoint_warning(
                idx_file, kwargs, logger, max_display_point, save_html
            )
            mask = np.random.randint(0, sample_idxs.sum(), max_display_point)
        else:
            mask = np.arange(0, sample_idxs.sum())

        color_list_true = color_mapping[labels[sample_idxs]][mask].tolist()
        color_list_pred = color_mapping[pred_class[sample_idxs]][mask].tolist()

        scene = dict(aspectmode="data")

        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=z[mask],
                z=y[mask],
                mode="markers",
                marker=dict(size=1, color=color_list_true),
                text=class_string(
                    unmapped_labels[sample_idxs][mask],
                    class_mapping=self._data.class_mapping,
                ),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=z[mask],
                z=y[mask],
                mode="markers",
                marker=dict(size=1, color=color_list_pred),
                text=class_string(
                    unmapped_pred_class[sample_idxs][mask],
                    prefix="pred_",
                    class_mapping=self._data.class_mapping,
                ),
            ),
            row=1,
            col=2,
        )

        title_text = get_title_text(idx, save_html, max_display_point)

        fig.update_layout(
            scene=scene,
            scene2=scene,
            title_text=title_text,
            width=kwargs.get("width", 750),
            height=kwargs.get("width", 512),
            showlegend=False,
            title_x=0.5,
        )

        fig2 = go.FigureWidget(fig)

        def cam_change(layout, camera):
            if fig2.layout.scene2.camera == camera:
                return
            else:
                fig2.layout.scene2.camera = camera

        def cam_change2(layout, camera):
            if fig2.layout.scene.camera == camera:
                return
            else:
                fig2.layout.scene.camera = camera

        fig2.layout.scene.on_change(cam_change, "camera")
        fig2.layout.scene2.on_change(cam_change2, "camera")

        if save_html:
            save_path = Path(save_path)
            plotly.io.write_html(fig, str(save_path / "show_results.html"))
            # remove plotly orca dep
            # fig.write_image(str(save_path / 'show_results.png'))
            return
        else:
            from IPython.display import display

            display(fig2)

        if idx == rows - 1:
            break
        idx += 1
    if return_fig:
        return fig


def compute_precision_recall(self):
    from ..models._pointcnn_utils import get_indices
    import pandas as pd

    valid_dl = self._data.valid_dl
    model = self.learn.model.eval()

    false_positives = [0] * self._data.c
    true_positives = [0] * self._data.c
    false_negatives = [0] * self._data.c
    class_count = [0] * self._data.c

    all_y = []
    all_pred = []
    for x_in, y_in in iter(valid_dl):
        if not getattr(self, "_is_ModelInputDict", False):
            x_in, point_nums = x_in  ## (batch, total_points, num_features), (batch,)
            batch, _, num_features = x_in.shape
            indices = torch.tensor(
                get_indices(batch, self.sample_point_num, point_nums.long())
            ).to(x_in.device)
            indices = indices.view(-1, 2).long()
            x_in = (
                x_in[indices[:, 0], indices[:, 1]]
                .view(batch, self.sample_point_num, num_features)
                .contiguous()
            )  ## batch, self.sample_point_num, num_features
            y_in = (
                y_in[indices[:, 0], indices[:, 1]]
                .view(batch, self.sample_point_num)
                .contiguous()
                .cpu()
                .numpy()
            )  ## batch, self.sample_point_num
        else:
            y_in = y_in.cpu().numpy()
        with torch.no_grad():
            preds = model(x_in).detach().cpu().numpy()
        predicted_labels = preds.argmax(axis=-1)
        all_y.append(y_in.reshape(-1))
        all_pred.append(predicted_labels.reshape(-1))

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)

    for i in range(len(all_y)):
        class_count[all_y[i]] += 1
        false_positives[all_pred[i]] += int(all_y[i] != all_pred[i])
        true_positives[all_pred[i]] += int(all_y[i] == all_pred[i])
        false_negatives[all_y[i]] += int(all_y[i] != all_pred[i])

    precision, recall, f_1 = calculate_metrics(
        false_positives, true_positives, false_negatives
    )
    data = [precision, recall, f_1]
    index = ["precision", "recall", "f1_score"]
    if hasattr(self._data, "idx2class"):
        inverse_class2idx = self._data.idx2class
    else:
        inverse_class2idx = {v: k for k, v in self._data.class2idx.items()}
    class_mapping = self._data.class_mapping
    # columns = [f'{inverse_class2idx[cval]} ({class_mapping[inverse_class2idx[cval]]})' for cval in range(self._data.c)]
    # # check whether the class mapping was specified by user.
    # for x, c in class_mapping.items():
    #     x = int(x)
    #     try:
    #         c = int(c)
    #         if x != c:
    #             none_cm = False
    #         else:
    #             none_cm = True
    #     except ValueError:
    #         none_cm = False

    # if none_cm:
    #     columns = [f'{inverse_class2idx[cval]}' for cval in range(self._data.c)]
    columns = [f"{inverse_class2idx[cval]}" for cval in range(self._data.c)]
    columns = [f"{self._data.class_mapping[int(cval)]}" for cval in columns]
    df = pd.DataFrame(data, columns=columns, index=index)
    return df


def gauss_clip(mu, sigma, clip):
    v = random.gauss(mu, sigma)
    v = max(min(v, mu + clip * sigma), mu - clip * sigma)
    return v


def uniform(bound):
    return bound * (2 * random.random() - 1)


def scaling_factor(scaling_param, method):
    try:
        scaling_list = list(scaling_param)
        return random.choice(scaling_list)
    except:
        if method == "g":
            return gauss_clip(1.0, scaling_param, 3)
        elif method == "u":
            return 1.0 + uniform(scaling_param)


def rotation_angle(rotation_param, method):
    try:
        rotation_list = list(rotation_param)
        return random.choice(rotation_list)
    except:
        if method == "g":
            return gauss_clip(0.0, rotation_param, 3)
        elif method == "u":
            return uniform(rotation_param)


def get_uniform_distribution(range):
    return np.random.uniform(-range, range)


def get_xforms(
    xform_num,
    rotation_range=(0, 0, 0, "u"),
    scaling_range=(0.0, 0.0, 0.0, "u"),
    order="XYZ",
):
    xforms = np.empty(shape=(xform_num, 3, 3))
    rotations = np.empty(shape=(xform_num, 3, 3))
    for i in range(xform_num):
        rx = get_uniform_distribution(rotation_range[0])
        ry = get_uniform_distribution(rotation_range[1])
        rz = get_uniform_distribution(rotation_range[2])
        rotation = R.from_euler(order, [rx, ry, rz]).as_matrix()

        sx = 1.0 + get_uniform_distribution(scaling_range[0])
        sy = 1.0 + get_uniform_distribution(scaling_range[1])
        sz = 1.0 + get_uniform_distribution(scaling_range[2])

        scaling = np.diag([sx, sy, sz])
        xforms[i, :] = np.dot(scaling, rotation)
        rotations[i, :] = rotation
    return xforms, rotations


def augment(points, xforms, range=None):
    points_xformed = points @ xforms
    if range is None:
        return points_xformed

    if isinstance(points, torch.Tensor):
        jitter_data = range * points.new(np.random.randn(*points_xformed.shape))
        jitter_clipped = torch.clamp(jitter_data, -range, range)
    else:
        jitter_data = range * np.random.randn(*points_xformed.shape)
        jitter_clipped = np.clip(jitter_data, -range, range)
    return points_xformed + jitter_clipped


class Transform3d(object):

    """
    Create transformations for 3D datasets, that can be used in
    :meth:`~arcgis.learn.prepare_data` to apply data augmentation
    with a 50% probability. Applicable for dataset_type='PointCloud'
    and dataset_type='PointCloudOD'.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rotation                An optional list of float. It defines a value in
                            degrees for each X, Y, and Z, dimensions which will
                            be used to rotate a block around the X, Y, and Z, axes.

                            Example:
                            A value of [2, 3, 180] means a random value for each
                            X, Y, and Z will be selected between, [-2, 2], [-3, 3],
                            and [-180, 180], respectively. The block will rotate
                            around the respective axis as per the selected random
                            value.

                            Note: For dataset_type=’PointCloudOD’, rotation around
                            the X and Y axes will not be considered.
                            Default: [2.5, 2.5, 45]
    ---------------------   -------------------------------------------
    scaling                 An optional float. It defines a percentage value, that
                            will be used to apply scaling transformation to a block.

                            Example:
                            A value of 5 means, for each X, Y, and Z, dimensions a
                            random value will be selected within the range of [0, 5],
                            where the block might be scaled up or scaled down randomly,
                            in the respective dimension.

                            Note: For dataset_type=’PointCloudOD’, the same scale
                            percentage in all three directions is considered.
                            Default: 5
    ---------------------   -------------------------------------------
    jitter                  Optional float within [0, 1]. It defines a value in
                            meters, which is used to add random variations in
                            X, Y, and Z of all points.

                            Example:
                            if the value provided is 0.1 then within the range
                            of [-0.1, 0.1] a random value is selected, The
                            selected value is then added to the point's X coordinate.
                            Similarly, it is applied for Y and Z coordinates.

                            Note: Only applicable for dataset_type=’PointCloud’.
                            Default: 0.0.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.Transform3d` object
    """

    def __init__(self, rotation=[2.5, 2.5, 45], scaling=5, jitter=0.0, **kwargs):
        rotation = kwargs.get("rotation_range", rotation)
        scaling = kwargs.get("scaling_range", scaling)
        if len(rotation) not in [3, 4]:
            raise Exception("The syntax of Rotation is not correct.")
        if min(rotation[:3]) < 0 or max(rotation[:3]) > 180:
            raise Exception("Rotation values should be in the range of [0, 180].")
        if (
            isinstance(scaling, (list, tuple))
            and len(scaling) != 4
            or isinstance(scaling, (int, float))
            and scaling < 0
        ):
            raise Exception(
                "Scaling parameter's syntax is not correct or it is not a positive number."
            )
        if jitter < 0 or jitter > 1:
            raise Exception("Jitter value should be in the range of [0,1].")

        if len(rotation) == 3:
            degree_to_redian = np.pi / 180
            rotation = (np.array(rotation) * degree_to_redian).tolist()
            rotation[1], rotation[2] = rotation[2], rotation[1]
        if not isinstance(scaling, (list, tuple)):
            scaling = [scaling / 100] * 3
        self.rotation_range = rotation
        self.scaling_range = scaling
        self.order = "XYZ"
        self.jitter = jitter

    def _detection_transforms(self):
        from .pointcloud_od import ODTransform3D

        rotation_range = [-self.rotation_range[1], self.rotation_range[1]]
        scaling_range = [1 - self.scaling_range[1], 1 + self.scaling_range[1]]

        return ODTransform3D(rotation_range, scaling_range)

    @property
    def _is_Transform3d(self):
        return True

    def __call__(self, x_in):
        xforms, _ = get_xforms(
            x_in.shape[0],
            rotation_range=self.rotation_range,
            scaling_range=self.scaling_range,
            order=self.order,
        )

        if isinstance(x_in, torch.Tensor):
            return augment(
                x_in[:, :, :3], x_in.new(xforms), x_in.new(np.array(self.jitter))
            )
        else:
            return augment(x_in[:, :, :3], xforms, np.array(self.jitter))

    def _transform_tool(self, x_in):
        if isinstance(x_in, torch.Tensor):
            inp = x_in.clone()
        else:
            inp = x_in.copy()
        inp = inp[:, :, :3]
        # convert to xzy
        inp = inp[:, :, [0, 2, 1]]
        return self(inp)[:, :, [0, 2, 1]]


def save_h5(filename, labels_pred, confidences_pred, class_confidence):
    try_import("h5py")
    import h5py

    filename = Path(filename)
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    filename_pred = filename.parent / (filename.stem + "_pred.h5")
    with h5py.File(filename_pred, "w") as file:
        file.create_dataset("per_class_confidence", data=class_confidence)
        file.create_dataset("label_seg", data=labels_pred)
        file.create_dataset("confidence", data=confidences_pred)


def convert_extra_features(attributes, features_to_keep):
    attributes_dict = {}

    string_mapped_features = {
        "numberOfReturns": "num_returns",
        "returnNumber": "return_num",
        "nearInfrared": "nir",
    }
    inverse_string_mapped_features = {v: k for k, v in string_mapped_features.items()}

    for a in attributes:
        attributes_dict[inverse_string_mapped_features.get(a[0], a[0])] = {
            "max": a[1],
            "min": a[2],
        }

    features_to_keep = [
        inverse_string_mapped_features.get(f, f) for f in features_to_keep
    ]

    return attributes_dict, features_to_keep


def model_predictions(model, data, point_nums):
    model.learn.model.eval()
    with torch.no_grad():
        if getattr(model, "_is_ModelInputDict", False):
            if isinstance(point_nums, int):
                point_nums = [point_nums]
            for batch_idx, p_num in enumerate(point_nums):
                if data.shape[1] != p_num:
                    # shifted xyz only not other fetures for knn
                    min_point = data[batch_idx, :p_num, :3].min(dim=0)[0][None]
                    max_point = data[batch_idx, :p_num, :3].max(dim=0)[0][None]
                    diameter = torch.cdist(min_point, max_point, p=2).max()
                    shift_point = 100 * diameter * max_point
                    data[batch_idx, p_num:, :3] += (
                        torch.rand(data.shape[1] - p_num, 3) + shift_point
                    )
            data = batch_preprocess_dict(
                data, model.encoder_params, model.__str__() == "<SQNSeg>"
            )
            for key in data:
                if type(data[key]) is list:
                    for i in range(len(data[key])):
                        data[key][i] = data[key][i].to(model._device)
                else:
                    data[key] = data[key].to(model._device)
            probs = model.learn.model(data).softmax(dim=-1).cpu()
        else:
            probs = (
                model.learn.model(data.to(model._device).float()).softmax(dim=-1).cpu()
            )

    return probs.numpy()


def get_batch_predictions(model, data, point_nums, point_batch_size):
    if model._data.max_point == model.sample_point_num:
        return model_predictions(model, data, point_nums)

    # handle case if max point in the block is greter than model.sample_point_num
    indices = []
    model_input = []
    for batch_idx, p_num in enumerate(point_nums):
        ## Getting sampling indices
        tile_num = math.ceil((model.sample_point_num * point_batch_size) / p_num)
        indices_shuffle = np.tile(np.arange(p_num), tile_num)[
            0 : model.sample_point_num * point_batch_size
        ]
        np.random.shuffle(indices_shuffle)
        indices_batch_shuffle = np.reshape(
            indices_shuffle, (point_batch_size, model.sample_point_num, 1)
        )

        input_point = torch.cat(
            [data[batch_idx, s[:, 0]][None] for s in indices_batch_shuffle],
            dim=0,
        )

        indices.append(indices_shuffle)
        model_input.append(input_point)

    model_input = torch.cat(model_input, dim=0)
    seg_probs = model_predictions(model, model_input, point_nums)

    # for each point of batch
    model_output = []
    for i in range(data.shape[0]):
        low = i * point_batch_size
        pred = seg_probs[low : low + point_batch_size]
        probs_2d = np.reshape(pred, (model.sample_point_num * point_batch_size, -1))
        predictions = np.ones((data.shape[1], probs_2d.shape[1]))
        for idx in range(model.sample_point_num * point_batch_size):
            point_idx = indices[i][idx]
            probs = probs_2d[idx, :]
            predictions[point_idx] = probs
        model_output.append(predictions[None])

    return np.concatenate(model_output, axis=0)


def split_prediction(model, predictions, point_nums):
    label = []
    confidance = []
    per_cls_conf = []
    for i, point_num in enumerate(point_nums):
        lbl = np.argmax(predictions[i][:point_num], axis=1)
        conf = np.amax(predictions[i][:point_num], axis=1)
        per_cls_conf.extend(predictions[i][:point_num])
        confidance.extend(conf)
        label.extend(np.array(list(model._data.idx2class.values()))[lbl])

    return np.array(per_cls_conf), np.array(confidance), np.array(label)


def predict_batch_h5(self, dl, output_path, progressor):
    current_file_name = ""
    point_batch_size = 1 * math.ceil(self._data.max_point / self.sample_point_num)

    for (data, point_num), tile_index in progress_bar(dl):
        pred = get_batch_predictions(self, data, point_num, point_batch_size)

        tile = dl.dataset.tiles[tile_index]
        if len(tile.shape) < 2:
            tile = tile[None]

        fname = np.array(dl.dataset.filenames)[tile[:, 0]]
        _, unique_index = np.unique(fname, return_index=True)
        # get unique name form sorted index in actual array
        fname = [fname[i] for i in np.sort(unique_index)]
        # add batch_size for spliting prediction till last batch number
        unique_index = list(np.sort(unique_index)) + [dl.batch_size]
        for i, ufname in enumerate(fname):
            if ufname != current_file_name:
                current_file_name = ufname
                h5_file = dl.dataset.h5files[tile[unique_index[i]][0]]
                batch_num, _ = h5_file["xyz"].shape
                labels_pred = np.full(batch_num, -1, dtype=np.int8)
                confidences_pred = np.zeros(batch_num, dtype=np.float32)
                class_confidence = np.zeros(
                    (batch_num + 1, self._data.c), dtype=np.float32
                )
                class_confidence[0] = np.array(self._data.classes)
                low = high = 0

            predictions = split_prediction(
                self,
                pred[unique_index[i] : unique_index[i + 1]],
                point_num[unique_index[i] : unique_index[i + 1]],
            )

            high = low + predictions[0].shape[0]
            labels_pred[low:high] = predictions[2]
            confidences_pred[low:high] = predictions[1]
            class_confidence[low + 1 : high + 1] = predictions[0]
            if high == batch_num:
                save_h5(
                    output_path
                    / dl.dataset.relative_files[
                        int(tile[unique_index[i]][0])
                    ].decode(),  # need to see tile number for -1
                    labels_pred,
                    confidences_pred,
                    class_confidence,
                )
            low = high

        if progressor is not None:
            progressor.current_block(tile_index[0].item())


def predict_h5(self, path, output_path, **kwargs):
    """
    self: PointCNN object
    path: path/to/h5/files/exported/by/tool
    """
    path = Path(path)
    if output_path is None:
        output_path = path.parent / "results"
    else:
        output_path = Path(output_path)

    progressor = kwargs.get("progressor", None)
    batch_size = kwargs.get("batch_size", 1)

    data = self._data
    # features to keep will be present always except for older models.
    features_to_keep = copy.copy(
        getattr(data, "features_to_keep", [f[0] for f in data.extra_features])
    )
    attributes = copy.copy(data.extra_features)
    if isinstance(attributes, list):
        api_model = True
        attributes, features_to_keep = convert_extra_features(
            attributes, features_to_keep
        )
        data.pc_type = "PointCloud"
        if not hasattr(data, "idx2class"):
            data.idx2class = {v: k for k, v in data.class2idx.items()}
    else:
        api_model = False
    if "xyz" in features_to_keep:
        features_to_keep.remove("xyz")
    point_cloud_dataset = PointCloudDataset(
        path,
        None,
        None,
        "",
        extra_features=features_to_keep,
        attributes=attributes,
    )
    if progressor is not None:
        progressor.set_total_blocks(len(point_cloud_dataset))

    output_path.mkdir(parents=True, exist_ok=True)

    point_cloud_dataset._get_metainfo_h5 = True
    point_cloud_dataset._api_model_h5 = api_model
    sampler = SequentialSampler(point_cloud_dataset)
    dataloader = DataLoader(point_cloud_dataset, batch_size=batch_size, sampler=sampler)
    predict_batch_h5(self, dataloader, output_path, progressor)

    return output_path


def calculate_per_class_stats(all_pred, all_y, total_classes):
    true_positives = [0] * total_classes
    false_positives = [0] * total_classes
    false_negatives = [0] * total_classes
    class_count = [0] * total_classes

    for i in range(len(all_y)):
        class_count[all_y[i]] += 1
        false_positives[all_pred[i]] += int(all_y[i] != all_pred[i])
        true_positives[all_pred[i]] += int(all_y[i] == all_pred[i])
        false_negatives[all_y[i]] += int(all_y[i] != all_pred[i])

    return true_positives, false_positives, false_negatives


def show_results_tool(self, rows, color_mapping=None, **kwargs):
    """
    It will plot results from your trained model with ground truth on the
    left and predictions on the right.
    Visualization of data, exported in a geographic coordinate system
    is not yet supported.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional rows. Number of rows to show. Deafults
                            value is 2.
    ---------------------   -------------------------------------------
    color_mapping           Optional dictionary. Mapping from class value
                            to RGB values. Default value example:
                            {0:[220,220,220], 2:[255,0,0], 6:[0,255,0]}.
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    mask_class              Optional array of integers. Array containing
                            class values to mask. Default value is [].
    ---------------------   -------------------------------------------
    width                   Optional integer. Width of the plot. Default
                            value is 750.
    ---------------------   -------------------------------------------
    height                  Optional integer. Height of the plot. Default
                            value is 512
    ---------------------   -------------------------------------------
    max_display_point       Optional integer. Maximum number of points
                            to display. Default is 20000.
    =====================   ===========================================
    """
    filter_outliers = False
    try_import("h5py")
    try_import("plotly")
    import h5py
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import random

    mask_class = kwargs.get("mask_class", [])
    save_html = kwargs.get("save_html", False)
    save_path = kwargs.get("save_path", ".")
    max_display_point = get_max_display_points(self._data, kwargs)
    data = self._data
    rows = min(rows, data.batch_size)
    color_mapping = data.color_mapping if color_mapping is None else color_mapping
    color_mapping = recompute_color_mapping(color_mapping, self._data.classes)
    color_mapping = np.array(list(color_mapping.values())) / 255
    if save_html:
        max_display_point = 20000
    ## dataset tiles Get all files from the tiles
    tile_file_indices = data.valid_ds.tiles[:, 0]
    ## iterate: on files
    for idx, _ in enumerate(data.h5files):
        ## Create subplot
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]]
        )

        # read that file
        indices = (tile_file_indices == idx).nonzero()[0]

        ## predict on each block by iterating.
        pred_batch_size = 1 * math.ceil(self._data.max_point / self.sample_point_num)
        labels = []
        pc = []
        pred_class = []
        block_centers = []
        blocks = []
        for block_idx in indices:
            (
                (block, point_num),
                classification,
                scaled_block,
            ) = data.valid_ds.__getitem__(
                block_idx, return_scaled=True, add_centers=True
            )

            blocks.append((block, point_num))
            pc.append(scaled_block[:point_num].cpu().numpy())
            block_centers.append(
                scaled_block[:point_num, :3].cpu().numpy().mean(axis=0)
            )
            labels.append(classification[:point_num].cpu().numpy())

        # clustered indexes is a boolean mask
        clustered_index_bool_mask = get_random_cluster_indexes(
            block_centers, self._data.block_size
        )
        labels = np.concatenate(np.array(labels)[clustered_index_bool_mask])
        pc = np.concatenate(np.array(pc)[clustered_index_bool_mask], axis=0)
        blocks = [blocks[i] for i, mask in enumerate(clustered_index_bool_mask) if mask]

        # prediction step
        for block, point_num in blocks:
            block = block[None]
            points_batch = block[[0] * 1]
            predictions = np.array(
                get_predictions(
                    self,
                    block,
                    0,
                    points_batch,
                    self.sample_point_num,
                    pred_batch_size,
                    point_num.cpu().item(),
                )
            )
            pred_class.append(predictions[:point_num, 0])

        pred_class = np.concatenate(pred_class, axis=0)
        unmapped_labels = remap_labels(labels.copy().astype(int), data.idx2class)
        unmapped_predictions = remap_labels(pred_class, data.idx2class)

        ## sample points
        sample_idxs = mask_classes(
            labels=labels,
            mask_class=mask_class,
            classes=self._data.classes,
            class2idx=self._data.class2idx if self._data.remap else None,
            remap_classes=self._data.remap_classes if self._data.remap else None,
        )
        sampled_pc = pc[sample_idxs]
        if sampled_pc.shape[0] == 0:
            continue
        sampled_pc = sampled_pc[:, :3]
        x, y, z = recenter(sampled_pc).transpose(1, 0)

        ## resample points if exeeds limits.
        if sample_idxs.sum() > max_display_point:
            raise_maxpoint_warning(idx, kwargs, logger, max_display_point, save_html)
            mask = np.random.randint(0, sample_idxs.sum(), max_display_point)
        else:
            mask = np.arange(0, sample_idxs.sum())

        ## Apply cmap
        color_list_true = color_mapping[labels[sample_idxs]][mask].tolist()
        color_list_pred = color_mapping[pred_class[sample_idxs].astype(int)][
            mask
        ].tolist()
        # print(x[mask].shape, y[mask].shape)
        ## Plot
        scene = dict(aspectmode="data")
        scene2 = dict(aspectmode="data")

        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode="markers",
                marker=dict(size=1, color=color_list_true),
                text=class_string(
                    unmapped_labels[sample_idxs][mask], "", data.class_mapping
                ),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode="markers",
                marker=dict(size=1, color=color_list_pred),
                text=class_string(
                    unmapped_predictions[sample_idxs][mask], "pred_", data.class_mapping
                ),
            ),
            row=1,
            col=2,
        )

        title_text = get_title_text(idx, save_html, max_display_point)
        fig.update_layout(
            scene=scene,
            scene2=scene2,
            title_text=title_text,
            width=kwargs.get("width", 750),
            height=kwargs.get("width", 512),
            showlegend=False,
            title_x=0.5,
        )

        fig2 = go.FigureWidget(fig)

        def cam_change(layout, camera):
            if fig2.layout.scene2.camera == camera:
                return
            else:
                fig2.layout.scene2.camera = camera

        def cam_change2(layout, camera):
            if fig2.layout.scene.camera == camera:
                return
            else:
                fig2.layout.scene.camera = camera

        fig2.layout.scene.on_change(cam_change, "camera")
        fig2.layout.scene2.on_change(cam_change2, "camera")

        if save_html:
            save_path = Path(save_path)
            plotly.io.write_html(fig, str(save_path / "show_results.html"))
            # remove plotly orca dep
            # fig.write_image(str(save_path / 'show_results.png'))
            return
        else:
            from IPython.display import display

            display(fig2)

        if idx == rows - 1:
            break
