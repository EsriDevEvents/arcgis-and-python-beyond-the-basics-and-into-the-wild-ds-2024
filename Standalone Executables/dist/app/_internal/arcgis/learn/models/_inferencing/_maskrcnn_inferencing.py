try:
    import os, sys, json
    import numpy as np
    import torch
    import torch.nn as nn
    import math
    from .util import scale_batch, variable_tile_size_check
    import cv2

    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False

import arcgis
from arcgis.learn import MaskRCNN
from arcgis.learn.models._maskrcnn_utils import predict_tta

try:
    import arcpy
except:
    pass


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

        bounding_box_coord_list.append([coord_array.tolist()])

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


class ChildInstanceDetector:
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

        self.model_emd = model

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
                {
                    "name": "threshold",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.9,
                    "displayName": "Threshold",
                    "description": "Threshold",
                },
                {
                    "name": "return_bboxes",
                    "dataType": "string",
                    "required": False,
                    "domain": ("True", "False"),
                    "value": "False",
                    "displayName": "return_bboxes",
                    "description": "return_bboxes",
                },
                {
                    "name": "test_time_augmentation",
                    "dataType": "string",
                    "required": False,
                    "value": "False",
                    "displayName": "Perform test time augmentation while predicting",
                    "description": "If True, will merge predictions from flipped and rotated images.",
                },
                {
                    "name": "merge_policy",
                    "dataType": "string",
                    "required": False,
                    "value": "mean",
                    "displayName": "Policy for merging augmented predictions",
                    "description": "Policy for merging predictions('mean' or 'nms'). Applicable when test_time_augmentation is True.",
                },
            ]
        )
        required_parameters = variable_tile_size_check(
            self.json_info, required_parameters
        )
        return required_parameters

    def getConfiguration(self, **scalars):
        self.tytx = int(scalars.get("tile_size", self.json_info["ImageHeight"]))
        self.mask_rcnn = MaskRCNN.from_model(
            emd_path=self.model_emd, chip_size=self.tytx
        )
        self._learnmodel = self.mask_rcnn
        self.model = self.mask_rcnn.learn.model.to(self.device)
        self.model.eval()

        #
        self.padding = int(
            scalars.get("padding", self.tytx // 4)
        )  ## Default padding Imageheight//4.
        self.batch_size = (
            int(math.sqrt(int(scalars.get("batch_size", 4)))) ** 2
        )  ## Default 4 batch_size
        self.threshold = float(scalars.get("threshold", 0.9))  ## Default 0.9 threshold.
        self.return_bboxes = eval(scalars.get("return_bboxes", "False"))
        self.merge_policy = scalars.get("merge_policy", "mean").lower()
        self.use_tta = scalars.get("test_time_augmentation", "false").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]

        if self.use_tta:
            if self.json_info["ImageSpaceUsed"] == "MAP_SPACE":
                self.model.arcgis_tta = list(range(8))
            else:
                self.model.arcgis_tta = [0, 2]

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

        return {
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "tx": tx,
            "ty": ty,
            "fixedTileSize": 1,
        }

    def vectorize(self, **pixelBlocks):  # 8 x 3 x 224 x 224
        input_image = pixelBlocks["raster_pixels"].astype(np.float32)
        batch, batch_height, batch_width = tile_to_batch(
            input_image,
            self.tytx,
            self.tytx,
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )

        if "NormalizationStats" in self.json_info:
            img_normed = scale_batch(batch, self.json_info)
        else:
            img_normed = batch / 255

        predictions = pixel_mask_image(
            self.model,
            img_normed,
            self.device,
            self.tytx,
            threshold=self.threshold,
            batch_size=self.batch_size,
            return_bboxes=self.return_bboxes,
            use_tta=self.use_tta,
            merge_policy=self.merge_policy,
        )

        return predictions


def predict_mask_rcnn(
    model, images, device, chip_size, threshold=0.5, use_tta=False, merge_policy="mean"
):
    model = model.to(device)
    normed_batch_tensor = torch.tensor(images).to(device).float()
    if use_tta:
        predictions = predict_tta(model, normed_batch_tensor, threshold, merge_policy)
    else:
        temp = model.roi_heads.score_thresh
        model.roi_heads.score_thresh = threshold
        predictions = model(list(normed_batch_tensor))
        model.roi_heads.score_thresh = temp

    return predictions


def pixel_mask_image(
    model,
    img_normed,
    device,
    chip_size,
    threshold=0.5,
    batch_size=4,
    return_bboxes=False,
    use_tta=False,
    merge_policy="mean",
):
    side = int(math.sqrt(batch_size))

    predictions = predict_mask_rcnn(
        model,
        img_normed,
        device,
        chip_size,
        threshold=threshold,
        use_tta=use_tta,
        merge_policy=merge_policy,
    )

    all_contour_list = []
    pred_box = []
    pred_class = []
    pred_score = []

    for batch_idx in range(len(predictions)):
        i, j = batch_idx // side, batch_idx % side
        masks = predictions[batch_idx]["masks"].squeeze().detach().cpu().numpy()
        if masks.shape[0] != 0:  # handle for prediction with n masks
            if (
                len(masks.shape) == 2
            ):  # for mask dimension hxw (in case of only one predicted mask)
                masks = masks[None]
            for n, mask in enumerate(masks):
                if predictions[batch_idx]["scores"][n].tolist() >= threshold:
                    contours, hierarchy = cv2.findContours(
                        (mask >= 0.5).astype(np.uint8),
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_NONE,
                    )
                    if len(contours) > 0:
                        hierarchy = hierarchy[0]
                        for c_idx, contour in enumerate(contours):
                            contour = contours[c_idx] = contour.squeeze(1)
                            contours[c_idx][:, 0] = contour[:, 0] + (j * chip_size)
                            contours[c_idx][:, 1] = contour[:, 1] + (i * chip_size)
                        for (
                            contour_idx,
                            (next_contour, prev_contour, child_contour, parent_contour),
                        ) in enumerate(hierarchy):
                            if parent_contour == -1:
                                coord_list = []
                                # check if it is a state line
                                closed_contour = (
                                    all(
                                        contours[contour_idx].max(axis=0)
                                        - contours[contour_idx].min(axis=0)
                                    )
                                    and contours[contour_idx].shape[0] > 2
                                )
                                if closed_contour:
                                    coord_list.append(contours[contour_idx].tolist())
                                while child_contour != -1:
                                    closed_contour = (
                                        all(
                                            contours[child_contour].max(axis=0)
                                            - contours[child_contour].min(axis=0)
                                        )
                                        and contours[child_contour].shape[0] > 2
                                    )
                                    if closed_contour:
                                        coord_list.append(
                                            contours[child_contour].tolist()
                                        )
                                    child_contour = hierarchy[child_contour][0]
                                #
                                if coord_list != []:
                                    all_contour_list.append(coord_list)
                                    pred_class.append(
                                        predictions[batch_idx]["labels"][n].tolist()
                                    )
                                    pred_score.append(
                                        predictions[batch_idx]["scores"][n].tolist()
                                        * 100
                                    )
                                    box = (
                                        predictions[batch_idx]["boxes"][n]
                                        .cpu()
                                        .detach()
                                        .numpy()
                                    )
                                    box[0] += j * chip_size
                                    box[2] += j * chip_size
                                    box[1] += i * chip_size
                                    box[3] += i * chip_size
                                    pred_box.append(box)

    if return_bboxes:
        pred_box = np.array(pred_box)
        bbox = convert_bounding_boxes_to_coord_list(pred_box)
        return bbox, pred_class, pred_score

    else:
        return all_contour_list, pred_class, pred_score
