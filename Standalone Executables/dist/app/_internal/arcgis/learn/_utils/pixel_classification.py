import math
from .env import ARCGIS_ENABLE_TF_BACKEND

if ARCGIS_ENABLE_TF_BACKEND:
    import tensorflow as tf
    from .common_tf import get_channel_axis

try:
    import torch
    import fastai
    import numpy as np
    import matplotlib.pyplot as plt
    from fastai.vision import imagenet_stats

    HAS_FASTAI = True
except:
    HAS_FASTAI = False


def segmentation_mask_to_one_hot(segmentation_mask, n_classes: int):
    """
    returns a one hot encoding of the supplied segmentation mask

    segmentation_mask: torch.tensor with shape (batch, 1, rows, cols) or (batch, rows, cols) or (rows, cols)
    """
    class_embeddings = torch.eye(n_classes, dtype=torch.uint8)
    shp = segmentation_mask.shape
    if len(shp) == 2:
        target_reshape_shape = (shp[0], shp[1], n_classes)
        mutation = (2, 0, 1)
    else:
        target_reshape_shape = (shp[0], shp[-2], shp[-1], n_classes)
        mutation = (0, 3, 1, 2)

    return (
        class_embeddings[segmentation_mask.flatten()]
        .reshape(target_reshape_shape)
        .permute(*mutation)
    )


def analyze_pred_pixel_classification(self, activations):
    """
    Analyzes predictions to be ready to plot
    returns numpy array.
    """
    if self._backend == "pytorch":
        if type(activations) == list:
            activations = torch.cat(activations)

        if not getattr(self, "_is_model_extension", False):
            if self._ignore_mapped_class != []:
                for k in self._ignore_mapped_class:
                    activations[:, k] = activations.min() - 1
        return activations.max(dim=1)[1].cpu().numpy()
    elif self._backend == "tensorflow":
        if type(activations) == list:
            activations = tf.concat(activations, 0)
        return analyze_pred_TFPC(activations).numpy()


def analyze_pred_TFPC(activations):
    return tf.argmax(activations, axis=get_channel_axis())
