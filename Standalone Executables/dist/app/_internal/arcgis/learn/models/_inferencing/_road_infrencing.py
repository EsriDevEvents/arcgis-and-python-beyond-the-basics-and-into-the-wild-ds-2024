try:
    import os, json
    import numpy as np
    import torch
    import torch.nn as nn
    import math
    from torch import tensor
    from .util import (
        variable_tile_size_check,
        fold_tensor,
        unfold_tensor,
        dihedral_transform,
        create_interpolation_mask,
    )

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

import arcgis
from arcgis.learn.models._multi_task_road_extractor import MultiTaskRoadExtractor

try:
    import arcpy
except Exception:
    pass


def split_tensor(tensor, tile_size, stride):
    # based on # https://discuss.pytorch.org/t/seemlessly-blending-tensors-together/65235/9
    mask = torch.ones_like(tensor)
    number_of_bands = tensor.shape[1]
    softmax_mask = torch.ones_like(tensor[0][0].unsqueeze(0).unsqueeze(0))

    unfold = nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    # Apply to mask and original image
    mask_p = unfold(softmax_mask)
    patches = unfold(tensor)

    patches = patches.reshape(number_of_bands, tile_size, tile_size, -1).permute(
        3, 0, 1, 2
    )
    if tensor.is_cuda:
        patches_base = torch.zeros(patches.size(), device=tensor.get_device())
    else:
        patches_base = torch.zeros(patches.size())

    return mask_p, patches_base, (tensor.size(2), tensor.size(3)), patches


def rebuild_tensor(input_tensor, mask_t, t_size, tile_size, stride):
    input_tensor_permuted = (
        input_tensor.permute(1, 2, 3, 0).reshape(-1, input_tensor.size(0)).unsqueeze(0)
    )
    fold = nn.Fold(
        output_size=(t_size[0], t_size[1]),
        kernel_size=(tile_size, tile_size),
        stride=stride,
    )
    output_tensor = fold(input_tensor_permuted) / fold(mask_t)
    return output_tensor


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


class ChildImageClassifier:
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

        self.road_extractor = MultiTaskRoadExtractor.from_model(
            data=None, emd_path=model
        )
        self._learnmodel = self.road_extractor
        self.model = self.road_extractor.learn.model.to(self.device)
        self.model.eval()

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                {
                    "name": "padding",
                    "dataType": "numeric",
                    "value": int(self.json_info["ImageHeight"]) // 4,
                    "required": False,
                    "displayName": "Padding",
                    "description": "Padding",
                },
                {
                    "name": "batch_size",
                    "dataType": "numeric",
                    "required": False,
                    "value": 4,
                    "displayName": "Batch Size",
                    "description": "Batch Size",
                },
            ]
        )
        if (
            self.json_info.get("ArcGISLearnVersion", False)
            and self.json_info["ArcGISLearnVersion"] > "1.8.4"
        ):
            required_parameters.extend(
                [
                    {
                        "name": "return_probability_raster",
                        "dataType": "string",
                        "required": False,
                        "value": "False",
                        "displayName": "Return Probability Raster",
                        "description": "If True, will return the probability surface of the result.",
                    },
                    {
                        "name": "threshold",
                        "dataType": "numeric",
                        "value": 0.5,
                        "required": False,
                        "displayName": "Confidence Score Threshold [0.0, 1.0]",
                        "description": "Confidence score threshold value [0.0, 1.0]",
                    },
                ]
            )

        required_parameters.extend(
            [
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
                {
                    "name": "merge_policy",
                    "dataType": "string",
                    "required": False,
                    "value": "max",
                    "displayName": "Policy for merging augmented predictions",
                    "description": "Policy for merging predictions('mean', 'max' or 'min'). Applicable when test_time_augmentation is True.",
                },
            ]
        )

        required_parameters = variable_tile_size_check(
            self.json_info, required_parameters
        )
        return required_parameters

    def getConfiguration(self, **scalars):
        self.tytx = int(scalars.get("tile_size", self.json_info["ImageHeight"]))
        self.padding = int(
            scalars.get("padding", self.tytx // 4)
        )  ## Default padding Imageheight//4.
        self.batch_size = (
            int(math.sqrt(int(scalars.get("batch_size", 4)))) ** 2
        )  # Default 4 batch_size
        self.predict_background = scalars.get("predict_background", "true").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  # Default value True
        self.probability_raster = scalars.get(
            "return_probability_raster", "false"
        ).lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  # Default value False
        (
            self.rectangle_height,
            self.rectangle_width,
        ) = calculate_rectangle_size_from_batch_size(self.batch_size)
        ty, tx = get_tile_size(
            self.tytx,
            self.tytx,
            self.padding,
            self.rectangle_height,
            self.rectangle_width,
        )

        self.thres = float(scalars.get("threshold", 0.5))  ## Default 0.5 threshold.

        self.use_tta = scalars.get("test_time_augmentation", "false").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  # Default value True

        self.merge_policy = scalars.get("merge_policy", "mean").lower()

        return {
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "tx": tx,
            "ty": ty,
            "threshold": self.thres,
            "fixedTileSize": 1,
            "test_time_augmentation": self.use_tta,
            "merge_policy": self.merge_policy,
        }

    def pixel_classify_image(
        self, model, tiles, device, classes, predict_bg, model_info
    ):
        model = model.to(device).eval()
        # logger.info("Tiles length is ", len(tiles))
        normed_batch_tensor = tensor(tiles).to(device).float()
        # logger.info("Normed length is ", len(normed_batch_tensor))
        with torch.no_grad():
            output, _ = model(normed_batch_tensor)

        if predict_bg:
            return output.max(dim=1)[1]
        else:
            output[:, 0] = -1
            return output.max(dim=1)[1]

    def updatePixels(self, tlc, shape, props, **pixelBlocks):  # 8 x 224 x 224 x 3
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

        semantic_predictions = self.pixel_classify_image(
            self.model,
            batch,
            self.device,
            classes=[clas["Name"] for clas in self.json_info["Classes"]],
            predict_bg=self.predict_background,
            model_info=self.json_info,
        )

        semantic_predictions = batch_to_tile(
            semantic_predictions.unsqueeze(dim=1).cpu().numpy(),
            batch_height,
            batch_width,
        )

        return semantic_predictions

    def split_predict_interpolate(self, normalized_image_tensor):
        model_arch = self.json_info["ModelParameters"]["mtl_model"]
        kernel_size = self.tytx
        stride = kernel_size - (2 * self.padding)

        if model_arch == "hourglass":
            kernel_size = (math.ceil(kernel_size / 2)) * 2

        # Split image into overlapping tiles
        masks, t_size, patches = unfold_tensor(
            normalized_image_tensor, kernel_size, stride
        )

        with torch.no_grad():
            output, _ = self.model(patches)

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

    def detectRoads(self, tlc, shape, props, **pixelBlocks):  # 8 x 224 x 224 x 3
        input_image = pixelBlocks["raster_pixels"].astype(np.float32)
        input_image_tensor = torch.tensor(input_image).to(self.device).float()

        normalized_image_tensor = input_image_tensor.unsqueeze(0)

        all_activations = self.tta_predict(
            normalized_image_tensor, test_time_aug=self.use_tta
        )

        softmax_surface = all_activations.softmax(dim=1)[:, [1], :, :]
        if self.merge_policy == "max":
            activations = softmax_surface.max(dim=0)[0]
        elif self.merge_policy == "min":
            activations = softmax_surface.min(dim=0)[0]
        else:
            activations = softmax_surface.mean(dim=0)

        if self.probability_raster:
            predictions = activations
        else:
            predictions = activations.gt(self.thres) * torch.tensor(1)

        pad = self.padding
        return (
            predictions[0][pad : -pad or None, pad : -pad or None]
            .unsqueeze(0)
            .cpu()
            .numpy()
        )
