import os
import sys
import json
import arcgis
from arcgis.learn import FeatureClassifier
import arcpy


import numpy as np
from .util import normalize_batch

try:
    from fastai.vision import *
    import torch
    from fastai.vision.transform import dihedral

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

        if arcpy.env.processorType == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            arcgis.env._processorType = "GPU"
        else:
            self.device = torch.device("cpu")
            arcgis.env._processorType = "CPU"

        # Using arcgis.learn FeatureClassifer from_model function.
        self.cf = FeatureClassifier.from_model(emd_path=model)
        self._learnmodel = self.cf
        self.model = self.cf.learn.model
        self.model = self.cf.learn.model.to(self.device)
        self.model.eval()

    def getParameterInfo(self, required_parameters):
        if (
            "MetaDataMode" in self.emd
            and self.emd["MetaDataMode"] == "MultiLabeled_Tiles"
        ):
            required_parameters.append(
                {
                    "name": "score_threshold",
                    "dataType": "numeric",
                    "value": 0.5,
                    "required": False,
                    "displayName": "Confidence Score Threshold [0.0, 1.0]",
                    "description": "Confidence score threshold value [0.0, 1.0]",
                }
            )
        # add tta in the parameters
        required_parameters.append(
            {
                "name": "test_time_augmentation",
                "dataType": "string",
                "required": False,
                "value": "False"
                if "test_time_augmentation" not in self.emd
                else str(self.emd["test_time_augmentation"]),
                "displayName": "Perform test time augmentation while predicting",
                "description": "If True, will merge predictions from flipped and rotated images.",
            }
        )
        return required_parameters

    def getConfiguration(self, **scalars):
        if "BatchSize" not in self.emd and "batch_size" not in scalars:
            self.batch_size = 1
        elif "BatchSize" not in self.emd and "batch_size" in scalars:
            self.batch_size = int(scalars["batch_size"])
        else:
            self.batch_size = int(self.emd["BatchSize"])

        self.thresh = float(
            scalars.get("score_threshold", 0.5)
        )  # Default 0.5 threshold

        self.use_tta = scalars.get("test_time_augmentation", "false").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  # Default value True

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
            "test_time_augmentation": self.use_tta,
        }

    def tta_predict(self, normalized_image_tensor):
        # Get normalized image and apply test time augmentation on image
        if self.emd["ImageSpaceUsed"] == "MAP_SPACE":
            aug_tfms = list(range(8))
        else:
            aug_tfms = [
                0,
                2,
            ]  # no vertical flips for pixel space (oriented imagery)
        tta_pred_combined = []
        for tfm in aug_tfms:
            tta_batch_images = []
            for tensorimage in normalized_image_tensor:
                out = dihedral(Image(tensorimage), tfm)
                tta_batch_images.append(out.data)
            tta_batch = torch.stack(tta_batch_images)
            tta_pred = self.cf.learn.pred_batch(
                batch=(
                    tta_batch.to(self.device),
                    torch.tensor([40]).to(self.device),
                )
            )
            tta_pred_combined.append(tta_pred)

        return torch.stack(tta_pred_combined).mean(0)

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

        if self.use_tta:
            # Convert to torch tensor, set device and convert to float
            batch_images = torch.tensor(batch_images).float()

            predictions = self.tta_predict(batch_images)
            # predictions: torch.tensor(B,C), where B is the batch size and C is the number of classe
        else:
            # Convert to torch tensor, set device and convert to float
            batch_images = torch.tensor(batch_images).to(self.device).float()

            # the second element in the passed tuple is hardcoded to make fastai's pred_batch work
            predictions = self.cf.learn.pred_batch(
                batch=(batch_images, torch.tensor([40]).to(self.device))
            )
            # predictions: torch.tensor(B,C), where B is the batch size and C is the number of classes

        # Using emd to map the class
        class_map = [c["Name"] for c in self.emd["Classes"]]

        # For Multi Label Classification
        if (
            "MetaDataMode" in self.emd
            and self.emd["MetaDataMode"] == "MultiLabeled_Tiles"
        ):
            for pred in predictions:
                # Select the class labels >= threshold and convert them to a comma separated string
                class_idxs = np.where(pred >= self.thresh)[0]
                lbls = [class_map[idx] for idx in class_idxs]
                lbls_string = ";".join(lbls)
                labels.append(lbls_string)

                # Select all confidences and convert them to a comma separated string
                scores = [str(p.item()) for p in pred]
                # scores = [str(pred[idx].tolist()) for idx in class_idxs]
                scores_string = ";".join(scores)
                confidences.append(scores_string)

        # For Single Label Classification
        else:
            # torch.max returns the max value and the index of the max as a tuple
            confidences, class_idxs = torch.max(predictions, dim=1)
            confidences = confidences.tolist()
            labels = [class_map[c] for c in class_idxs]

        # Appending this ring for all the features in the batch
        rings = [
            [[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]]
            for i in range(batch)
        ]

        return rings, confidences, labels
