try:
    import os, json
    import numpy as np
    import torch
    from fastai.core import split_kwargs_by_func
    import math
    from . import util
    from .util import (
        variable_tile_size_check,
        normalize_batch,
        fold_tensor,
        unfold_tensor,
        dihedral_transform,
        create_interpolation_mask,
    )
    from pathlib import Path
    import torch.nn.functional as F
    from ...models._unet_utils import is_contiguous
    from .._inferencing.util import remap

    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False

import arcgis
from arcgis.learn import ModelExtension
from skimage.measure import find_contours


try:
    import arcpy
except:
    pass


def normalize_batch_imagenetstats(batch):
    imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    mean = 255 * np.array(imagenet_stats[0], dtype=np.float32)
    std = 255 * np.array(imagenet_stats[1], dtype=np.float32)

    return (batch - mean) / std


def calculate_rectangle_size_from_batch_size(batch_size):
    """
    calculate number of rows and cols to composite a rectangle given a batch size
    :param batch_size:
    :return: number of cols and number of rows
    """
    rectangle_height = int(math.sqrt(batch_size) + 0.5)
    rectangle_width = int(batch_size / rectangle_height)

    if rectangle_height * rectangle_width > batch_size:
        if rectangle_height >= rectangle_width:
            rectangle_height = rectangle_height - 1
        else:
            rectangle_width = rectangle_width - 1

    if (rectangle_height + 1) * rectangle_width <= batch_size:
        rectangle_height = rectangle_height + 1
    if (rectangle_width + 1) * rectangle_height <= batch_size:
        rectangle_width = rectangle_width + 1

    # swap col and row to make a horizontal rect
    if rectangle_height > rectangle_width:
        rectangle_height, rectangle_width = rectangle_width, rectangle_height

    if rectangle_height * rectangle_width != batch_size:
        return batch_size, 1

    return rectangle_height, rectangle_width


def convert_bounding_boxes_to_coord_list(bounding_boxes):
    """
    convert bounding box numpy array to python list of point arrays
    :param bounding_boxes: numpy array of shape [n, 4]
    :return: python array of point numpy arrays, each point array is in shape [4,2]
    """
    num_bounding_boxes = bounding_boxes.shape[0]
    bounding_box_coord_list = []
    for i in range(num_bounding_boxes):
        coord_array = np.empty(shape=(4, 2), dtype=float)
        coord_array[0][0] = bounding_boxes[i][0]
        coord_array[0][1] = bounding_boxes[i][1]

        coord_array[1][0] = bounding_boxes[i][0]
        coord_array[1][1] = bounding_boxes[i][3]

        coord_array[2][0] = bounding_boxes[i][2]
        coord_array[2][1] = bounding_boxes[i][3]

        coord_array[3][0] = bounding_boxes[i][2]
        coord_array[3][1] = bounding_boxes[i][1]

        bounding_box_coord_list.append(coord_array)

    return bounding_box_coord_list


def get_tile_size(model_height, model_width, padding, batch_height, batch_width):
    """
    Calculate request tile size given model and batch dimensions
    :param model_height:
    :param model_width:
    :param padding:
    :param batch_width:
    :param batch_height:
    :return: tile height and tile width
    """
    tile_height = (model_height - 2 * padding) * batch_height
    tile_width = (model_width - 2 * padding) * batch_width

    return tile_height, tile_width


def tile_to_batch(
    pixel_block, model_height, model_width, padding, fixed_tile_size=True, **kwargs
):
    inner_width = model_width - 2 * padding
    inner_height = model_height - 2 * padding

    band_count, pb_height, pb_width = pixel_block.shape
    pixel_type = pixel_block.dtype

    if fixed_tile_size is True:
        batch_height = kwargs["batch_height"]
        batch_width = kwargs["batch_width"]
    else:
        batch_height = math.ceil((pb_height - 2 * padding) / inner_height)
        batch_width = math.ceil((pb_width - 2 * padding) / inner_width)

    batch = np.zeros(
        shape=(batch_width * batch_height, band_count, model_height, model_width),
        dtype=pixel_type,
    )
    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        # pixel block might not be the shape (band_count, model_height, model_width)
        sub_pixel_block = pixel_block[
            :,
            y * inner_height : y * inner_height + model_height,
            x * inner_width : x * inner_width + model_width,
        ]
        sub_pixel_block_shape = sub_pixel_block.shape
        batch[
            b, :, : sub_pixel_block_shape[1], : sub_pixel_block_shape[2]
        ] = sub_pixel_block

    return batch, batch_height, batch_width


def batch_to_tile(batch, batch_height, batch_width):
    batch_size, bands, inner_height, inner_width = batch.shape
    tile = np.zeros(
        shape=(bands, inner_height * batch_height, inner_width * batch_width),
        dtype=batch.dtype,
    )

    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        tile[
            :,
            y * inner_height : (y + 1) * inner_height,
            x * inner_width : (x + 1) * inner_width,
        ] = batch[b]

    return tile


def remove_bounding_boxes_in_padding(
    bounding_boxes,
    scores,
    classes,
    image_height,
    image_width,
    padding,
    batch_height=1,
    batch_width=1,
):
    """

    :param bounding_boxes: the batch of bounding boxes, shape=[B,N,4]
    :param scores: the batch of box scores, shape=[B,N]
    :param classes: the batch of labels, shape=[B,N]
    :param image_height: model height
    :param image_width: model width
    :param padding:
    :param batch_height:
    :param batch_width:
    :return:
    """
    keep_indices = np.where(
        (bounding_boxes[:, :, 0] < image_height - padding)
        & (bounding_boxes[:, :, 1] < image_width - padding)
        & (bounding_boxes[:, :, 2] > padding)
        & (bounding_boxes[:, :, 3] > padding)
    )

    inner_width = image_width - 2 * padding
    inner_height = image_height - 2 * padding

    # convert coordinates in the batch to super tile and then filter by the keep_indices
    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        bounding_boxes[b, :, [0, 2]] = bounding_boxes[b, :, [0, 2]] + y * inner_height
        bounding_boxes[b, :, [1, 3]] = bounding_boxes[b, :, [1, 3]] + x * inner_width

    bounding_boxes = bounding_boxes[keep_indices]
    scores = scores[keep_indices]
    classes = classes[keep_indices]

    return bounding_boxes, scores, classes


class ChildPanopticSegmenter:
    def initialize(self, model, model_as_file):
        if not HAS_TORCH:
            raise Exception(
                "Could not find the required deep learning dependencies. Ensure you have installed the required dependent libraries. See https://developers.arcgis.com/python/guide/deep-learning/"
            )

        if arcpy.env.processorType == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            arcgis.env._processorType = "GPU"
        else:
            self.device = torch.device("cpu")
            arcgis.env._processorType = "CPU"

        if model_as_file:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        else:
            self.json_info = json.load(model)

        model_path = self.json_info["ModelFile"]
        if model_as_file and not os.path.isabs(model_path):
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(model), model_path)
            )

        self.json_emd_file = Path(model).parent
        self.model_extension = ModelExtension.from_model(emd_path=model)
        self._learnmodel = self.model_extension
        self.model = self.model_extension.learn.model.to(self.device)
        self.model.eval()

        class_values = [clas["Value"] for clas in self.json_info["Classes"]]
        self.instance_classes = self.json_info["Kwargs"]["instance_classes"]
        self.is_contig = is_contiguous([0] + class_values)
        self.idx2pixel = None
        if not self.is_contig:
            pixel_mapping = [0] + class_values
            self.idx2pixel = {i: d for i, d in enumerate(pixel_mapping)}
            pixel2idx = {v: k for k, v in self.idx2pixel.items()}
            mapped_instcls = [pixel2idx[i] for i in self.instance_classes]
            self.instance_classes = mapped_instcls

        self.activations = None

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                {
                    "name": "padding",
                    "dataType": "numeric",
                    "value": self.json_info["ImageHeight"] // 4,
                    "required": False,
                    "displayName": "Padding",
                    "description": "Padding",
                },
                {
                    "name": "threshold",
                    "dataType": "numeric",
                    "value": 0.5,
                    "required": False,
                    "displayName": "Confidence Score Threshold [0.0, 1.0]",
                    "description": "Confidence score threshold value [0.0, 1.0]",
                },
                {
                    "name": "nms_overlap",
                    "dataType": "numeric",
                    "value": 0.1,
                    "required": False,
                    "displayName": "NMS Overlap",
                    "description": "Maximum allowed overlap within each chip",
                },
                {
                    "name": "batch_size",
                    "dataType": "numeric",
                    "required": False,
                    "value": 4,
                    "displayName": "Batch Size",
                    "description": "Batch Size",
                },
                {
                    "name": "test_time_augmentation",
                    "dataType": "string",
                    "required": False,
                    "value": "False"
                    if "test_time_augmentation" not in self.json_info
                    else str(self.json_info["test_time_augmentation"]),
                    "displayName": "Perform test time augmentation while predicting",
                    "description": "If True, will merge predictions from flipped and rotated images.",
                },
            ]
        )
        return required_parameters

    def getConfiguration(self, **scalars):
        # self.chip_size = self.json_info["ImageHeight"]
        self.tytx = int(scalars.get("tile_size", self.json_info["ImageHeight"]))
        self.padding = int(scalars.get("padding", self.tytx // 4))
        self.nms_overlap = float(
            scalars.get("nms_overlap", 0.1)
        )  ## Default 0.1 NMS Overlap.
        self.thres = float(scalars.get("threshold", 0.5))  ## Default 0.5 threshold.
        self.batch_size = int(math.sqrt(int(scalars.get("batch_size", 4)))) ** 2
        self.filter_outer_padding_detections = True
        self.thinning = None
        self.probability_raster = None
        self.merge_policy = "mean"
        self.predict_background = False

        (
            self.rectangle_height,
            self.rectangle_width,
        ) = calculate_rectangle_size_from_batch_size(self.batch_size)
        ty, tx = get_tile_size(
            self.json_info["ImageHeight"],
            self.json_info["ImageWidth"],
            self.padding,
            self.rectangle_height,
            self.rectangle_width,
        )

        self.use_tta = scalars.get("test_time_augmentation", "false").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  # Default value False

        return {
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "threshold": self.thres,
            "nms_overlap": self.nms_overlap,
            "tx": tx,
            "ty": ty,
            "fixedTileSize": 1,
            "test_time_augmentation": self.use_tta,
        }

    def vectorize(self, **pixelBlocks):  # 8 x 3 x 224 x 224
        input_image = pixelBlocks["raster_pixels"].astype(np.float32)
        batch, batch_height, batch_width = tile_to_batch(
            input_image,
            self.json_info["ImageHeight"],
            self.json_info["ImageWidth"],
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )

        if "NormalizationStats" in self.json_info:
            img_normed = normalize_batch(batch, self.json_info)
        else:
            img_normed = normalize_batch_imagenetstats(
                batch.transpose(0, 2, 3, 1)
            ).transpose(0, 3, 1, 2)

        predictions = detect_object_mask(
            self.model_extension._model_conf,
            self.model,
            img_normed,
            self.device,
            self.tytx,
            batch_size=self.batch_size,
            model_info=self.json_info,
            threshold=self.thres,
            instance_classes=self.instance_classes,
            is_contig=self.is_contig,
            pred_batch=self.activations,
        )
        return predictions

    def split_predict_interpolate(self, normalized_image_tensor):
        kernel_size = self.tytx
        stride = kernel_size - (2 * self.padding)

        # Split image into overlapping tiles
        masks, t_size, patches = unfold_tensor(
            normalized_image_tensor, kernel_size, stride
        )

        output, self.activations = classify_image(
            self.model_extension._model_conf,
            self.model,
            patches,
            self.device,
            self.predict_background,
            self.json_info,
            self.json_emd_file,
            thinning=False,
            threshold=self.thres,
            prob_raster=True,
        )

        interpolation_mask = create_interpolation_mask(
            kernel_size, 0, self.device, "hann"
        )

        output = output * interpolation_mask
        masks = masks * interpolation_mask

        # merge predictions from overlapping chips
        int_surface = fold_tensor(output, masks, t_size, kernel_size, stride)

        return int_surface

    def tta_predict(self, normalized_image_tensor, test_time_aug=True):
        all_activations = []

        transforms = [0]
        if test_time_aug:
            if self.json_info["ImageSpaceUsed"] == "MAP_SPACE":
                transforms = list(range(8))
            else:
                transforms = [
                    0,
                    2,
                ]  # no vertical flips for pixel space (oriented imagery)

        for k in transforms:
            flipped_image_tensor = dihedral_transform(normalized_image_tensor[0], k)
            int_surface = self.split_predict_interpolate(
                flipped_image_tensor.unsqueeze(0)
            )
            corrected_activation = dihedral_transform(int_surface[0], k)

            if k in [5, 6]:
                corrected_activation = dihedral_transform(int_surface[0], k).rot90(
                    2, [1, 2]
                )

            all_activations.append(corrected_activation)

        all_activations = torch.stack(all_activations)

        return all_activations

    def updatePixelsTTA(self, tlc, shape, props, **pixelBlocks):  # 8 x 224 x 224 x 3
        model_info = self.json_info

        input_image = pixelBlocks["raster_pixels"].astype(np.float32)
        input_image_tensor = torch.tensor(input_image).to(self.device).float()

        if "NormalizationStats" in model_info:
            normalized_image_tensor = normalize_batch(
                input_image_tensor.cpu(), model_info
            )
            normalized_image_tensor = normalized_image_tensor.float().to(
                input_image_tensor.device
            )
        else:
            from torchvision import transforms

            normalize = transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            normalized_image_tensor = normalize(input_image_tensor / 255.0).unsqueeze(0)

        all_activations = self.tta_predict(
            normalized_image_tensor,
            test_time_aug=self.use_tta,
        )

        if self.merge_policy == "max":
            activations = all_activations.max(dim=0)[0]
        elif self.merge_policy == "min":
            activations = all_activations.min(dim=0)[0]
        else:
            activations = all_activations.mean(dim=0)

        if self.probability_raster:
            predictions = activations.unsqueeze(0)
        else:
            predictions = self.model_extension._model_conf.post_process(
                activations.unsqueeze(0), thres=self.thres, thinning=self.thinning
            )

            if not self.is_contig:
                predictions = remap(predictions, self.idx2pixel)

        pad = self.padding

        return predictions[0].cpu().numpy()[:, pad : -pad or None, pad : -pad or None]


def detect_object_mask(
    model_configuration,
    model,
    images,
    device,
    chip_size,
    batch_size,
    model_info,
    threshold,
    instance_classes,
    is_contig,
    pred_batch,
):
    tile_height, tile_width = images.shape[2], images.shape[3]
    side = math.sqrt(batch_size)
    N = model_info["Kwargs"]["n_masks"]

    if pred_batch is None:
        if "NormalizationStats" in model_info:
            batch_input = model_configuration.transform_input_multispectral(
                torch.tensor(images).to(device).float()
            )
        else:
            batch_input = model_configuration.transform_input(
                torch.tensor(images).to(device).float()
            )
        with torch.no_grad():
            pred_batch = model(batch_input)

    preds = model_configuration.post_process(pred_batch, thres=threshold, detector=True)

    ## TODO: check if this block can be moved to post_process
    instance_probs = F.softmax(preds[0], dim=1)
    instances = instance_probs.argmax(dim=1)
    instances = F.one_hot(instances, num_classes=N).permute(0, 3, 1, 2)
    class_confidence, classes = F.softmax(preds[1], dim=-1).max(-1)

    # Filter predictions for instances
    inst_cls = classes.detach().clone()
    for i in instance_classes:
        inst_cls = torch.where(
            inst_cls == i, torch.tensor(-1).to(classes.device), inst_cls
        )
    keep_pred_instances = torch.where(
        torch.logical_and(inst_cls == -1, class_confidence > threshold)
    )

    pred_instances = []
    pred_classes = []
    pred_scores = []

    for index in range(batch_size):
        keep_pred = keep_pred_instances[1][keep_pred_instances[0] == index]
        pred_instances.append(instances.detach()[index, keep_pred].cpu().numpy())
        pred_classes.append(classes.detach()[index, keep_pred].cpu().numpy())
        pred_scores.append(class_confidence.detach()[index, keep_pred].cpu().numpy())

    all_contour_list = []
    pred_class = []
    pred_score = []

    for batch_idx in range(batch_size):
        i, j = batch_idx // side, batch_idx % side
        masks = pred_instances[batch_idx]

        # handle for prediction with n masks
        if masks.shape[0] != 0:
            # for mask dimension hxw (in case of only one predicted mask)
            if len(masks.shape) == 2:
                masks = masks[None]

            for n, mask in enumerate(masks):
                contours = find_contours(mask, 0.5, fully_connected="high")
                if len(contours) == 0:
                    continue
                contour = contours[0]
                contour[:, 0] = contour[:, 0] + (i * chip_size)
                contour[:, 1] = contour[:, 1] + (j * chip_size)

                # coord_list = []
                # for c_idx, contour in enumerate(contours):

                #     contour[:, 0] = contour[:, 0] + (i * chip_size)
                #     contour[:, 1] = contour[:, 1] + (j * chip_size)
                #     if c_idx == 0:
                #         coord_list.append(contour[:, [1, 0]].tolist())
                #     else:
                #         coord_list.append(list(reversed(contour[:, [1, 0]].tolist())))
                # all_contour_list.append(coord_list)

                all_contour_list.append(contour)
                pred_class.append(pred_classes[batch_idx][n].tolist())
                pred_score.append(pred_scores[batch_idx][n].tolist() * 100)

    return np.array(all_contour_list), pred_score, pred_class


def classify_image(
    model_configuration,
    model,
    images,
    device,
    predict_bg,
    model_info,
    emd_path,
    thinning,
    threshold,
    prob_raster=False,
):
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images).to(device).float()

    if "NormalizationStats" in model_info:
        batch_input = model_configuration.transform_input_multispectral(images)
    else:
        batch_input = model_configuration.transform_input(images)

    with torch.no_grad():
        pred_batch = model(batch_input)

    preds = model_configuration.post_process(
        pred_batch, thres=threshold, thinning=thinning, prob_raster=prob_raster
    )

    return preds, pred_batch
