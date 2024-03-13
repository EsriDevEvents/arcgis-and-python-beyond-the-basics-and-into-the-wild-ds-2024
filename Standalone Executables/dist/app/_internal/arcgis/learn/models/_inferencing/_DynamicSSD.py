try:
    import os, sys, json
    import numpy as np
    import torch
    import torch.nn as nn
    import math
    from . import util

    HAS_TORCH = True

except Exception as e:
    HAS_TORCH = False

try:
    import arcpy
except:
    pass


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


class ChildObjectDetector:
    def initialize(self, model, model_as_file):
        if not HAS_TORCH:
            raise Exception(
                "Could not find the required deep learning dependencies. Ensure you have installed the required dependent libraries. See https://developers.arcgis.com/python/guide/deep-learning/"
            )

        from arcgis.learn.models import SingleShotDetector
        import arcgis

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

        self.ssd = SingleShotDetector.from_emd(data=None, emd_path=model)
        self._learnmodel = self.ssd
        self.model = self.ssd.learn.model.to(self.device)
        self.model.eval()

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
                    "value": 64,
                    "displayName": "Batch Size",
                    "description": "Batch Size",
                },
                {
                    "name": "exclude_pad_detections",
                    "dataType": "string",
                    "required": False,
                    "domain": ("True", "False"),
                    "value": "True",
                    "displayName": "Filter Outer Padding Detections",
                    "description": "Filter detections which are outside the specified padding",
                },
            ]
        )
        return required_parameters

    def getConfiguration(self, **scalars):
        self.padding = int(
            scalars.get("padding", self.json_info["ImageHeight"] // 4)
        )  ## Default padding Imageheight//4.
        self.nms_overlap = float(
            scalars.get("nms_overlap", 0.1)
        )  ## Default 0.1 NMS Overlap.
        self.thres = float(scalars.get("threshold", 0.5))  ## Default 0.5 threshold.
        self.batch_size = (
            int(math.sqrt(int(scalars.get("batch_size", 64)))) ** 2
        )  ## Default 64 batch_size
        self.filter_outer_padding_detections = scalars.get(
            "exclude_pad_detections", "True"
        ).lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  ## Default value True

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

        return {
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "threshold": self.thres,
            "nms_overlap": self.nms_overlap,
            "tx": tx,
            "ty": ty,
            "fixedTileSize": 1,
        }

    def vectorize(self, **pixelBlocks):  # 8 x 224 x 224 x 3
        input_image = pixelBlocks["raster_pixels"]
        batch, batch_height, batch_width = tile_to_batch(
            input_image,
            self.json_info["ImageHeight"],
            self.json_info["ImageWidth"],
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )

        bounding_boxes, scores, classes = util.detect_objects_image_space(
            self.model,
            batch,
            self.ssd._anchors,
            self.ssd._grid_sizes,
            self.device,
            classes=[clas["Name"] for clas in self.json_info["Classes"]],
            nms_overlap=self.nms_overlap,
            thres=self.thres,
            model_info=self.json_info,
        )

        return convert_bounding_boxes_to_coord_list(bounding_boxes), scores, classes
