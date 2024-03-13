try:
    import os, sys, json
    import numpy as np
    import torch
    import torch.nn as nn
    import math
    from . import util
    from ._pix2pix import (
        calculate_rectangle_size_from_batch_size,
        get_tile_size,
        tile_to_batch,
        batch_to_tile,
    )

    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False
import arcgis

from arcgis.learn import PSETAE

try:
    import arcpy
except:
    pass


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

        self.psetae = PSETAE.from_model(data=None, emd_path=model)
        self._learnmodel = self.psetae
        self.model = self.psetae.learn.model.to(self.device)
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
        return required_parameters

    def getConfiguration(self, **scalars):
        self.padding = int(
            scalars.get("padding", self.json_info["ImageHeight"] // 4)
        )  ## Default padding Imageheight//4.
        self.batch_size = (
            int(math.sqrt(int(scalars.get("batch_size", 4)))) ** 2
        )  ## Default 4 batch_size

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

        return {"padding": self.padding, "tx": tx, "ty": ty, "fixedTileSize": 1}

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
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

        psetae_ts_prediction = util.pixel_classify_ts_image(
            self.model, batch, self.device, model_info=self.json_info
        )
        psetae_ts_prediction = batch_to_tile(
            psetae_ts_prediction.unsqueeze(dim=1).detach().cpu().numpy(),
            batch_height,
            batch_width,
        )

        return psetae_ts_prediction
