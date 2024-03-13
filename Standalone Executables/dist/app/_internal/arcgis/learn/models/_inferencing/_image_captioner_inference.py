import os
import sys
import json
import arcgis
from arcgis.learn import ImageCaptioner

import numpy as np
from .util import normalize_batch

try:
    from fastai.vision import *
    import torch

    HAS_PYTORCH_FA = True

except Exception as e:
    HAS_PYTORCH_FA = False

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

imagenet_mean = 255 * np.array(imagenet_stats[0], dtype=np.float32)
imagenet_std = 255 * np.array(imagenet_stats[1], dtype=np.float32)


def norm(x, mean=imagenet_mean, std=imagenet_std):
    return (x - mean) / std


def denorm(x, mean=imagenet_mean, std=imagenet_std):
    return x * std + mean


class ChildObjectDetector:
    def initialize(self, model, model_as_file):
        if not HAS_PYTORCH_FA:
            raise Exception(
                'PyTorch (version 1.1.0 or above) and fast.ai (version 1.0.54 or above) libraries are not installed. Install PyTorch using "conda install pytorch=1.1.0 fastai=1.0.54".'
            )

        if model_as_file:
            with open(model, "r") as f:
                self.emd = json.load(f)
        else:
            self.emd = json.loads(model)
        import arcpy

        if arcpy.env.processorType == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            arcgis.env._processorType = "GPU"
        else:
            self.device = torch.device("cpu")
            arcgis.env._processorType = "CPU"

        # Using arcgis.learn FeatureClassifer from_model function.
        self.cf = ImageCaptioner.from_model(emd_path=model)
        self._learnmodel = self.cf
        self.model = self.cf.learn.model.to(self.device)
        self.model.eval()

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [
                {
                    "name": "beam_width",
                    "dataType": "numeric",
                    "value": 5,
                    "required": False,
                    "displayName": "The number of captions to consider.",
                    "description": "The number of captions to consider.",
                },
                {
                    "name": "max_length",
                    "dataType": "numeric",
                    "value": 20,
                    "required": False,
                    "displayName": "Maximum length of the caption.",
                    "description": "Maximum length of the caption.",
                },
            ]
        )
        return required_parameters

    def getConfiguration(self, **scalars):
        if "BatchSize" not in self.emd and "batch_size" not in scalars:
            self.batch_size = 1
        elif "BatchSize" not in self.emd and "batch_size" in scalars:
            self.batch_size = int(scalars["batch_size"])
        else:
            self.batch_size = int(self.emd["BatchSize"])

        self.beam_width = int(scalars.get("beam_width", 5))

        self.max_length = int(scalars.get("max_length", 20))

        return {
            # CropSizeFixed is a boolean value parameter (1 or 0) in the emd file, representing whether the size of
            # tile cropped around the feature is fixed or not.
            # 1 -- fixed tile size, crop fixed size tiles centered on the feature. The tile can be bigger or smaller
            # than the feature;
            # 0 -- Variable tile size, crop out the feature using the smallest fitting rectangle. This results in tiles
            # of varying size, both in x and y. the ImageWidth and ImageHeight in the emd file are still passed and used
            # as a maximum size. If the feature is bigger than the defined ImageWidth/ImageHeight, the tiles are cropped
            # the same way as in the fixed tile size option using the maximum size.
            "CropSizeFixed": int(self.emd["CropSizeFixed"]),
            # BlackenAroundFeature is a boolean value paramater (1 or 0) in the emd file, representing whether blacken
            # the pixels outside the feature in each image tile.
            # 1 -- Blacken
            # 0 -- Not blacken
            "BlackenAroundFeature": int(self.emd["BlackenAroundFeature"]),
            "extractBands": tuple(self.emd["ExtractBands"]),
            "tx": self.emd["ImageWidth"],
            "ty": self.emd["ImageHeight"],
            "batch_size": self.batch_size,
        }

    def vectorize(self, **pixelBlocks):
        # Get pixel blocks - tuple of 3-d rasters: ([bands,height,width],[bands,height.width],...)
        # Convert tuple to 4-d numpy array
        batch_images = np.asarray(pixelBlocks["rasters_pixels"])

        # Get the shape of the 4-d numpy array
        batch, bands, height, width = batch_images.shape

        rings = []
        labels, confidences = [], []

        # Normalize Image
        if "NormalizationStats" in self.emd:
            batch_images = normalize_batch(batch_images, self.emd)
        else:
            # Transpose the image dimensions to [batch, height, width, bands],
            # normalize and transpose back to [batch, bands, height, width]
            batch_images = norm(batch_images.transpose(0, 2, 3, 1)).transpose(
                0, 3, 1, 2
            )

        # Convert to torch tensor, set device and convert to float
        batch_images = torch.tensor(batch_images).to(self.device).float()

        _, labels, _ = self.cf.learn.model.sample(
            batch_images.to(self.device), self.beam_width, self.max_length
        )

        # Appending this ring for all the features in the batch
        rings = [
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            for i in range(batch)
        ]

        return rings, labels
