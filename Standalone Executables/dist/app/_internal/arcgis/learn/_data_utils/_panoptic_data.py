# necessary import

import os
from pathlib import Path
import traceback
import warnings
import glob

try:
    import numpy as np
    import torch
    from .._utils.common import ArcGISMSImage
    from fastai.vision.data import ImageList, ItemBase
    from ..models._maskrcnn_utils import ArcGISImageSegment as ImageMask
    from ..models._unet_utils import ArcGISImageSegment as ImageSegment
    from ..models._unet_utils import is_contiguous, map_to_contiguous

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class MaskLabelSemanticItem(ItemBase):
    "ItemBase class suitable for Panoptic(Mask + Semantic)"

    def __init__(self, mask, labels, semantic):
        self.obj = (mask, labels, semantic)
        self.data = [mask.data, labels, semantic.data]
        self.shape = semantic.shape

    def apply_tfms(self, tfms, **kwargs):
        self.obj = [
            self.obj[0].apply_tfms(tfms, **kwargs),
            self.obj[1],
            self.obj[2].apply_tfms(tfms, **kwargs),
        ]
        self.data = [self.obj[0].data, self.obj[1], self.obj[2].data]
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}{(self.obj[0].shape, self.obj[1].shape, self.obj[2].shape)}"


# Class to read Panoptic Segmentation Labels
class PanopticSegmentationLabelList(ImageList):
    "ItemList class suitable for Panoptic Targets (Labels)."

    def __init__(
        self,
        items,
        chip_size,
        classes=None,
        class_mapping=None,
        color_mapping=None,
        **kwargs,
    ):
        # max number of masks (K in ground truth, N  in predictions)
        self.K = kwargs.pop("n_masks")
        self.inst_class_mapping = kwargs.pop("inst_class_mapping")
        self.inv_inst_class_mapping = {v: k for k, v in self.inst_class_mapping.items()}

        super().__init__(items, **kwargs)
        self.class_mapping = class_mapping
        self.color_mapping = color_mapping
        self.copy_new.append("classes")
        self.classes = classes
        self.chip_size = chip_size

        # self.items[i] - WindowsPath('data/panoptic_kent_100/labels/000000102.tif')
        fn, ext = os.path.splitext(os.path.split(self.items[0])[1])
        self.inst_lbl_path = self.items[0].parent.parent / "labels2"

        # Create a list of instance class values
        self.instance_classes = list(self.inst_class_mapping.keys())

        ## Check whether the class values are contiguous and create contiguous mapping
        self.is_contiguous = is_contiguous(
            sorted([0] + list(self.class_mapping.keys()))
        )
        if not self.is_contiguous:
            self.pixel_mapping = [0] + list(self.class_mapping.keys())
            self.indexed_inst_classes = map_to_contiguous(
                torch.tensor(self.instance_classes), self.pixel_mapping
            ).tolist()
        else:
            self.indexed_inst_classes = self.instance_classes

        # Create a list of files in all class folders in labels2
        self.inst_lbl_files = list()
        for root, dirs, files in os.walk(self.inst_lbl_path):
            for file in files:
                if file.endswith(ext):
                    self.inst_lbl_files.append(os.path.join(root, file))

    def get(self, i):
        # Use the default method to open semantic labels
        semantics = super().get(i)
        # Create individual masks of all semantic segments
        masks, labels = self.create_semantic_masks(semantics.data)

        # self.items[i] - WindowsPath('data/panoptic_kent_100/labels/000000102.tif')
        fn, ext = os.path.splitext(os.path.split(self.items[i])[1])

        # Create a list of instance masks for the current image
        mask_files = list()
        for file in self.inst_lbl_files:
            if fn in file:
                mask_files.append(file)

        if len(mask_files) != 0:
            inst_masks, inst_labels = self.create_instance_masks(mask_files)
            masks = torch.cat((masks, inst_masks))
            labels = torch.cat((labels, inst_labels))

        n_labels = labels.shape[0]
        if n_labels > self.K:
            warnings.warn(
                "{} classes and instance labels present in one of the chips but only {} masks created. Please rerun 'prepare_data' with a higher 'n_masks' value.".format(
                    n_labels, self.K
                ),
                stacklevel=1,
            )
            masks = masks[: self.K]
            labels = labels[: self.K]

        # Add padding to masks and labels
        if n_labels < self.K:
            pad_size = self.K - n_labels
            mask_pad = torch.zeros(
                (pad_size, masks.shape[1], masks.shape[2]), dtype=torch.uint8
            )
            masks = torch.cat((masks, mask_pad), dim=0)
            labels_pad = torch.zeros(pad_size, dtype=torch.long)
            labels = torch.cat((labels, labels_pad), dim=0)

        masks = ArcGISMSImage(masks.to(torch.float))
        return MaskLabelSemanticItem(masks, labels, semantics)

    def open(self, fn):
        x = ArcGISMSImage.open(fn).data
        if not self.is_contiguous:
            x = map_to_contiguous(x, self.pixel_mapping)
        return ImageSegment(x, color_mapping=self.color_mapping)

    # Method to convert the semantics to individual masks and labels
    def create_semantic_masks(self, semantic):
        # Change the instance class pixels to -1
        for inst_cls in self.indexed_inst_classes:
            semantic = torch.where(
                semantic == int(inst_cls), torch.tensor(-1), semantic
            )

        semantic_np = np.asarray(semantic.cpu())
        labels = np.unique(semantic.cpu())

        ## TODO: investigate this
        # Remove the label -1 (instance classes)
        if labels[0] == -1:
            labels = labels[1:]

        semantic_labels = torch.tensor(labels, dtype=torch.long)

        if len(labels) == 0:
            return (
                torch.empty(size=(0, semantic.shape[1], semantic.shape[2])),
                semantic_labels,
            )

        semantic_masks = semantic_np == labels[:, None, None]
        semantic_masks = [mask.astype(np.uint8) for mask in semantic_masks]
        semantic_masks = torch.from_numpy(np.stack(semantic_masks, axis=0))
        return semantic_masks, semantic_labels

    # Method to convert the instance label files to masks and labels
    def create_instance_masks(self, fn):
        masks = []
        labels = []

        for mask_file in fn:
            # Read the mask file and grab the classname of the mask
            mask_img = torch.from_numpy(
                ArcGISMSImage.read_image(Path(mask_file)).astype("int16")
            )
            lbl_name = Path(mask_file).parent.name
            lbl_value = self.inv_inst_class_mapping[lbl_name]

            # Ensure a channel dimension exists
            if (
                len(mask_img.shape) == 3
            ):  # if mask shape has a channels dim (i.e. #channels > 1)
                mask_img = mask_img.permute(2, 0, 1)
            else:
                mask_img = torch.unsqueeze(mask_img, 0)

            for ch in range(mask_img.shape[0]):
                unique_instances = np.unique(mask_img[ch]).max()

                # For each unique mask id, starting 1, create an individual mask
                for instance in range(1, unique_instances + 1):
                    instance_mask = (mask_img[ch] == instance).to(torch.uint8)
                    masks.append(instance_mask)
                    labels.append(lbl_value)

        masks = torch.stack(masks)
        labels = torch.tensor(labels)

        if not self.is_contiguous:
            labels = map_to_contiguous(labels, self.pixel_mapping)

        return masks, labels

    def analyze_pred(self, pred, thresh=0.5, model=None):
        pred = model._model_conf.post_process(pred, thresh)
        return pred


class PanopticSegmentationItemList(ImageList):
    "`ItemList` suitable for Panoptic tasks."
    _label_cls = PanopticSegmentationLabelList
    _square_show_res = False
    _div = None
    _imagery_type = None

    def open(self, fn):
        return ArcGISMSImage.open(fn, div=self._div, imagery_type=self._imagery_type)


def compute_n_masks(path):
    n_masks = 0

    with open(path / "map.txt") as f:
        line = f.readline()
        ext = line.split()[1].split(".")[-1].lower()

    imgs = glob.glob(str(path / "images") + "/*.{}".format(ext))
    inst_labels = glob.glob(str(path / "labels2") + "/*/*.{}".format(ext))

    # Read labels and labels2 for each image
    for i, img in enumerate(imgs):
        label_count = 0
        img_name, extn = os.path.splitext(os.path.split(img)[1])
        sem = path / "labels" / "{}.{}".format(img_name, ext)
        insts = [i for i in inst_labels if img_name in i]

        # Count number of classes in semantics
        if not os.path.exists(sem):
            continue
        semantic = ArcGISMSImage.open(sem).data
        label_count = len(np.unique(semantic.cpu()))

        # Count number of instances
        for inst in insts:
            mask_img = torch.from_numpy(
                ArcGISMSImage.read_image(Path(inst)).astype("int16")
            )
            # Ensure a channel dimension exists
            if len(mask_img.shape) == 3:
                # if mask shape has a channels dim (i.e. #channels > 1)
                mask_img = mask_img.permute(2, 0, 1)
            else:
                mask_img = torch.unsqueeze(mask_img, 0)

            for ch in range(mask_img.shape[0]):
                unique_instances = np.unique(mask_img[ch]).max().item()
                label_count += unique_instances

        n_masks = max(n_masks, label_count)

    # Return max n_mask
    return n_masks
