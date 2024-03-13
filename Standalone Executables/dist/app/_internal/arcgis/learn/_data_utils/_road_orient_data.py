import glob
import math
import os
import random
import traceback
import types
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from .._utils.env import raise_fastai_import_error

HAS_FASTAI = True

try:
    import torch
    from .._utils.common import ArcGISMSImage
    from ._base_data import ArcgisData
    from .._utils.pointcloud_data import get_device
    from fastai.basic_data import DatasetType
    from fastai.vision.data import imagenet_stats
    from fastai.core import subplots
    from fastai.data_block import DataBunch
    from fastai.vision.data import ImageDataBunch
    from fastai.torch_core import grab_idx, to_detach
    import torchvision.transforms.functional as TF
    from typing import Sequence
    from fastai.vision.image import Image, pil2tensor
    from numpy import ma
    from PIL import Image as PILImage
    from scipy.ndimage.morphology import distance_transform_edt
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as pytorch_tfms
except ImportError as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


def _to_np(x: Tensor) -> np.ndarray:
    x = x.numpy()
    if x.ndim > 2:
        x = x * 255.0
        return x.transpose(1, 2, 0).astype(np.uint8)
    return x


def _show_batch(
    self,
    rows: int = 5,
    ds_type: DatasetType = DatasetType.Train,
    cpu: bool = True,
    **kwargs,
):
    "Show a batch of data in `ds_type` on a few `rows`."
    if self is None:
        return

    dl = self.dl(ds_type)
    w = dl.num_workers
    dl.num_workers = 0
    try:
        x, labels = next(iter(dl))
    finally:
        dl.num_workers = w

    y, o = labels
    x, y, o = to_detach(x, cpu=cpu), to_detach(y, cpu=cpu), to_detach(o, cpu=cpu)
    y, o = y.squeeze(1), o.squeeze(1)

    n_items = min(self.dl(ds_type).batch_size, rows, x.size(0))
    xs = [_to_np(grab_idx(x, i)) for i in range(n_items)]
    ys = [_to_np(grab_idx(y, i)) for i in range(n_items)]
    orients = [_to_np(grab_idx(o, i)) for i in range(n_items)]
    _show_pairs(xs, ys, orients, bin_size=self.orient_bin_size)


def _show_pairs(
    xs,
    ys,
    orients,
    imgsize: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    bin_size: int = 20,
):
    """
    Show Image - Road Label - Orientation Vector label pairs
    """
    rows = len(xs)
    main_title = "Image / Binary Road Maks"
    axs = subplots(rows, 2, imgsize=imgsize, figsize=figsize, title=main_title)
    for x, y, o, ax in zip(xs, ys, orients, axs):
        ax[0].imshow(x)
        ax[1].imshow(y)
    for ax in axs.flatten():
        ax.axis("off")


def _plotOrientationOnImage(ax, orientMap, image, bin_size=20):
    """
    Plot Orientation Vectors overlay on Image
    """
    from .._utils.road_orient_utils.affinity_utils import convertAngles2VecMap

    ax.imshow(image)
    orientmap_xy = convertAngles2VecMap(orientMap.shape, orientMap, bin_size)
    U = orientmap_xy[:, :, 0] * -1
    V = orientmap_xy[:, :, 1]
    X, Y = np.meshgrid(np.arange(U.shape[0]), np.arange(U.shape[1]))
    M = np.zeros(U.shape, dtype="bool")
    M[U**2 + V**2 < 0.5 * 0.5] = True
    U = ma.masked_array(U, mask=M)
    V = ma.masked_array(V, mask=M)

    s = 15
    ax.quiver(
        X[::s, ::s],
        Y[::s, ::s],
        U[::s, ::s],
        V[::s, ::s],
        scale=50,
        headaxislength=3,
        headwidth=4,
        width=0.01,
        alpha=0.8,
        color="r",
    )


class DiscreteAffine:
    """
    Custom PyTorch Rotation transform.
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    angles                   Required List of int ranging from 0..360
    ---------------------   -------------------------------------------
    """

    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.affine(x, angle, (0, 0), 1.0, 0.0, resample=False, fillcolor=0)


class RoadOrientation:
    def __init__(self, orig_data, classified_tiles_data: ArcgisData, **kwargs):
        """
        The class is used to create Fast AI Databunch for Multi-Task
        Road - Orientation learning. It internally utilizes RoadOrientation
        Dataset.

        Reference Paper: CVPR 2019 - Improved Road Connectivity by Joint
                        Learning of Orientation and Segmentation

        Parameters
        --------------------------------------------------------------------
        @param classified_tiles_data: instance of
            arcgis.learn._data_utils._pixel_classifier_data.ClassifiedTilesData
        @param kwargs:
        |--> orient_bin_size: float, Bin size used to quantize Orientation angles
        |--> orient_theta: float, number used as width of orientation angles
        |--> multi_scale: list of floats [0..1], used to create multi-scale label masks
        |                    e.g. [0.25,0.5, 1] will return 3 masks with dimensions
        |                    256x256, 512x512, 1024x1024 if the image size is 1024x1024
        """
        if not HAS_FASTAI:
            raise_fastai_import_error(import_exception=import_exception)

        self.base = classified_tiles_data
        assert (
            self.base
        ), "Road Orientation requires instance of base Classified Tiles Dataset!"
        # TODO: Need to fix this Hack
        if "0" not in self.base.class_mapping:
            self.base.class_mapping[0] = "0"
        self.base.class_mapping = dict(sorted(self.base.class_mapping.items()))

        self.class_mapping = (
            {"0": 0, "1": 1}
            if not bool(self.base.class_mapping) is None
            else self.base.class_mapping
        )
        # Dataset parameters
        default_road_params = {
            "orient_bin_size": 20,
            "orient_theta": 8.0,
            "multi_scale": None,
        }
        # self.road_extractor_params = kwargs.get("road_extractor_params", default_road_params)
        self.orient_bin_size = kwargs.get(
            "orient_bin_size", default_road_params["orient_bin_size"]
        )
        self.orient_theta = kwargs.get(
            "orient_theta", default_road_params["orient_theta"]
        )
        self.multi_scale = kwargs.get("multi_scale", default_road_params["multi_scale"])

        self.files = []
        for cnt, image in enumerate(orig_data.train_ds.x.items):
            file_pair = {
                "image": image,
                "label": orig_data.train_ds.y.items[cnt],
            }
            self.files.append(file_pair)

        # self.valid_files = []
        for cnt, image in enumerate(orig_data.valid_ds.x.items):
            file_pair = {
                "image": image,
                "label": orig_data.valid_ds.y.items[cnt],
            }
            self.files.append(file_pair)

        random.shuffle(self.files)
        if self.base.data_count:
            print(
                f"Taking only {self.base.data_count} samples from total files {len(self.files)}"
            )
            self.files = self.files[: self.base.data_count]
            print(f"New Data Pairs => {len(self.files)}")
        files_iterator = iter(self.files)
        val_split_count = int(len(self.files) * self.base.val_split_pct)
        self.valid_files = list(islice(files_iterator, val_split_count))
        self.train_files = list(files_iterator)
        self.transforms = (
            [
                # Training Transforms
                [
                    # Pairwise Transforms
                    [
                        pytorch_tfms.RandomCrop(
                            size=self.base.chip_size,
                            padding_mode="constant",
                            pad_if_needed=True,
                        )
                        if self.base.chip_size
                        else None,
                        pytorch_tfms.Resize(size=self.base.resize_to)
                        if self.base.resize_to
                        else None,
                        pytorch_tfms.RandomHorizontalFlip(),
                        pytorch_tfms.RandomVerticalFlip(),
                        pytorch_tfms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ],
                    # Image Transforms
                    [
                        pytorch_tfms.ColorJitter(brightness=0.1, contrast=0.1),
                        pytorch_tfms.ToTensor(),
                    ],
                ],
                # Validation Transforms
                [
                    # Pairwise Transforms
                    [
                        pytorch_tfms.Resize(size=self.base.resize_to)
                        if self.base.resize_to
                        else None
                    ],
                    # Image Transforms
                    [pytorch_tfms.ToTensor()],
                    [
                        pytorch_tfms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        )
                    ],
                ],
            ]
            if self.base.transforms is None
            else self.base.transforms
        )

    def get_databunch(self, orig_data, **kwargs) -> ImageDataBunch:
        """
        Method to create Databunch
        """
        train_dataset = RoadOrientDataset(
            orig_data,
            self.train_files,
            self.transforms[0],
            **kwargs,
        )
        valid_dataset = RoadOrientDataset(
            orig_data,
            self.valid_files,
            self.transforms[1],
            **kwargs,
        )
        train_dl = DataLoader(
            train_dataset,
            batch_size=self.base.train_batch_size,
            **self.base.databunch_kwargs,
        )
        valid_dl = DataLoader(
            valid_dataset,
            batch_size=self.base.valid_batch_size,
            **self.base.databunch_kwargs,
        )

        device = get_device()
        data = ImageDataBunch(train_dl, valid_dl, device=device)

        data.chip_size = data.train_ds[0][0].shape[-1]
        data.c = len(self.class_mapping) if self.class_mapping else 2

        data.orient_c = int(360.0 / self.orient_bin_size) + 1
        data.orient_bin_size = self.orient_bin_size
        data.orient_theta = self.orient_theta
        data.multi_scale = self.multi_scale
        data.classes = []
        for _, value in self.class_mapping.items():
            if value == 0:
                data.classes.append("NoData")
                continue
            data.classes.append(value)
        data.extension = self.base.extension
        data.val_files = self.valid_files
        data.show_batch = types.MethodType(_show_batch, data)
        data.parent_obj = self
        data.sub_dataset_type = "RoadOrientation"
        data.transform = self.transforms
        x_shape = data.train_ds[0][0].shape

        return data


class RoadOrientDataset(Dataset):
    def __init__(
        self,
        orig_data,
        data_files: Dict,
        transforms: List = None,
        multi_scale: List = None,
        orient_theta: int = 5,
        orient_bin_size: int = 20,
        **kwargs,
    ):
        """
        PyTorch Dataset class to create Road-Orientaion pair for training.

        @param data_files: Dict, contain image-label pair file paths
        @param transforms: list of Fastai transforms
        @param multi_scale: list of floats [0..1], used to create multi-scale label masks
                            e.g. [0.25,0.5, 1] will return 3 masks with dimensions
                            256x256, 512x512, 1024x1024 if the image size is 1024x1024
        @param orient_theta: float, number used as width of orientation angles
        @param orient_bin_size: float, Bin size used to quantize Orientation angles
        @param kwargs:
            |--> is_gaussian_mask: bool, if True code assumes that given road labels are
            |                      gaussian and do not convert binary road mask to gaussian roads
            |--> gaussian_thresh: float, used to change the road width using gaussian mask
            |--> generate_orient: bool, flag will allow to create Orientation Label. It is
            |                     helpful at inference time for faster data prepration.
        """
        self.data = kwargs.get("data", None)
        self.orig_data = orig_data
        self.files = data_files
        self.pair_tfms = [tfm for tfm in transforms[0] if tfm]
        self.image_tfms = [tfm for tfm in transforms[1] if tfm]
        self.pair_tfms = (
            pytorch_tfms.Compose(self.pair_tfms) if self.pair_tfms else None
        )
        self.image_tfms = (
            pytorch_tfms.Compose(self.image_tfms) if self.image_tfms else None
        )

        self.gaussian_thresh = kwargs.get("gaussian_thresh", 0.76)
        self.is_gaussian_mask = kwargs.get("is_gaussian_mask", False)
        self.generate_orient = kwargs.get("generate_orient", True)

        # Angle Mask Buffers
        self.angle_theta = orient_theta
        self.bin_size = orient_bin_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label = self._getRoadData(index)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        label = np.asarray(label)
        if isinstance(image, PILImage.Image):
            image = torch.from_numpy(
                np.asarray(image).astype(np.float32).transpose(2, 0, 1)
            )
        if self.generate_orient:
            orient_label = self._getOrientationGT(np.copy(label.astype(np.uint8)))
            # orient_label = self._get_fastai_image(orient_label, dtype=np.float32)
            return (
                image,
                [
                    torch.from_numpy(label.copy()),
                    torch.from_numpy(orient_label.copy()),
                ],
            )
        else:
            return (
                image,
                [
                    torch.from_numpy(label.copy()),
                    torch.from_numpy(label.copy()),
                ],
            )

    def _get_fastai_image(self, x, dtype) -> Image:
        """
        Convert PILImage or np array to Fastai.Vision.Image
        """
        if isinstance(x, PILImage.Image):
            tensor = pil2tensor(x, dtype=dtype)
        elif isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x.astype(dtype))
        return Image(tensor)

    def _getRoadData(self, index: int) -> Tuple:
        """
        Get Sattellite Image and Road Label Mask
        """
        image_dict = self.files[index]
        # read each image in list
        image = ArcGISMSImage.open(image_dict["image"])
        if (image.shape[0] > 3) and (self.orig_data._imagery_type == "RGB"):
            image = image.data[[0, 1, 2]]
        label = self._get_mask(ArcGISMSImage.read_image(image_dict["label"]))

        return image, PILImage.fromarray(label).convert("L")

    def _getOrientationGT(self, road_mask: np.ndarray) -> np.ndarray:
        """
        Create Orientation Label for given road label
        """
        from .._utils.road_orient_utils.affinity_utils import (
            getKeypoints,
            getVectorMapsAngles,
        )

        height, width = (
            road_mask.shape if isinstance(road_mask, np.ndarray) else road_mask.size
        )
        smooth_dist_dict = {6: 1, 7: 1, 8: 1, 9: 2, 10: 4, 11: 4}
        keypoints = getKeypoints(
            road_mask.astype(np.float32),
            is_gaussian=False,
            thresh=0.98,
            smooth_dist=smooth_dist_dict[round(math.log(height) / math.log(2))],
        )
        _, vecmap_angles = getVectorMapsAngles(
            (height, width), keypoints, theta=self.angle_theta, bin_size=self.bin_size
        )

        return vecmap_angles  # np.expand_dims(vecmap_angles, 0)

    def _createGaussianMask(self, gt_array):
        """
        Create Gaussian road mask for given road label.
            1 - Perform Skeletonization
            2 - Compute Distance Transform
            3 - Convert Gaussian Road Mask using standard Deviation of 15
        """
        from skimage.morphology import skeletonize

        gt_array = skeletonize(gt_array)
        distance_array = distance_transform_edt(1 - (gt_array))
        std = 15
        distance_array = np.exp(-0.5 * (distance_array * distance_array) / (std * std))

        return distance_array

    def _get_mask(self, label: np.ndarray) -> np.ndarray:
        """
        Threshold the gaussian mask.
        """
        from skimage import filters

        if self.is_gaussian_mask:
            new_label = (np.array(label).astype(float)) / 255.0
        else:
            try:
                new_label = self._createGaussianMask(label)
            except:
                new_label = self._createGaussianMask(
                    label > filters.threshold_otsu(label)
                )
        new_label[new_label >= self.gaussian_thresh] = 1
        new_label[new_label < self.gaussian_thresh] = 0

        return new_label.astype(np.uint8)
