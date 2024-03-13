try:
    import os, sys, json
    import numpy as np
    import pandas as pd
    import torch
    from torch import tensor
    import torch.nn as nn
    import math
    from . import util

    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False
import arcgis
from arcgis.learn import AutoML

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

        model_path = self.json_info["ResultsPath"]
        if model_as_file and not os.path.isabs(model_path):
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(model), model_path)
            )

        self.automl = AutoML.from_model(emd_path=model)
        self._learnmodel = self.automl
        # self.model = self.pix2pix_hd.learn.model.to(self.device)
        # self.model.eval()

    def getParameterInfo(self, required_parameters):
        band_cnt = 0
        for col in self.json_info["_raster_field_variables"]:
            required_parameters.extend(
                [
                    {
                        "name": f"{col}",
                        "dataType": "numeric",
                        "value": band_cnt,
                        "required": True,
                        "displayName": "Band Value",
                        "description": "mapping of band",
                    },
                ]
            )
            band_cnt = band_cnt + 1

        return required_parameters

    def getConfiguration(self, **scalars):
        self.scalars = scalars

        return {"fixedTileSize": 1}

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        image = pixelBlocks["raster_pixels"].astype(np.float32)  # change to int

        cols = {}

        for col in self.json_info["_raster_field_variables"]:
            cols[f"{col}"] = image[int(self.scalars.get("{}".format(col)))].flatten()

        raster_data = cols

        processed_data = []

        length_values = len(raster_data[list(raster_data.keys())[0]])
        for i in range(length_values):
            processed_row = []
            for raster_name in sorted(raster_data.keys()):
                processed_row.append(raster_data[raster_name][i])
            processed_data.append(processed_row)

        processed_df = pd.DataFrame(
            data=np.array(processed_data), columns=sorted(raster_data)
        )
        processed_numpy = self.automl._data._process_data(processed_df, fit=False)
        predictions = self.automl._predict(processed_numpy)

        predictions = np.array(
            predictions.reshape([image.shape[1], image.shape[2]]),
            dtype="float64",
        )

        predictions = np.expand_dims(predictions, axis=0)

        return predictions
