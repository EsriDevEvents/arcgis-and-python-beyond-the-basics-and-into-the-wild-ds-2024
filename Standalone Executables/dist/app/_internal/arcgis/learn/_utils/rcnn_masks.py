import os
import math
import torch
import matplotlib.pyplot as plt
import matplotlib
from ..models._maskrcnn_utils import ArcGISImageSegment
from .common import get_nbatches, kwarg_fill_none, image_batch_stretcher
from .._utils.env import is_arcgispronotebook


def show_batch_rcnn_masks(
    self, rows=3, alpha=0.5, **kwargs
):  # parameters adjusted in kwargs
    nrows = rows
    ncols = kwarg_fill_none(kwargs, "ncols", 3)

    n_items = None
    if kwargs.get("n_items", None) is not None:
        n_items = kwargs.get("n_items")

    type_data_loader = kwargs.get(
        "data_loader", "training"
    )  # options : traininig, validation, testing
    if type_data_loader == "training":
        data_loader = self.train_dl
    elif type_data_loader == "validation":
        data_loader = self.valid_dl
    elif type_data_loader == "testing":
        data_loader = self.test_dl
    else:
        e = Exception(
            f"could not find {type_data_loader} in data. Please ensure that the data loader type is traininig, validation or testing "
        )
        raise (e)

    imsize = 5
    if kwargs.get("imsize", None) is not None:
        imsize = kwargs.get("imsize")

    cmap = kwargs.get("cmap", "tab20")

    statistics_type = kwargs.get(
        "statistics_type", "dataset"
    )  # Accepted Values `dataset`, `DRA`
    stretch_type = kwargs.get(
        "stretch_type", "minmax"
    )  # Accepted Values `minmax`, `percentclip`

    # Get Batch
    if n_items is None:
        n_items = nrows * ncols
    else:
        nrows = math.ceil(n_items / ncols)
    n_items = min(n_items, len(self.x))

    x_batch, y_batch = get_nbatches(data_loader, n_items)
    x_batch = torch.cat(x_batch)
    y_batch = torch.cat(y_batch)

    if self._is_multispectral:
        # Denormalize X
        if self._do_normalize:
            x_batch = (
                self._scaled_std_values[self._extract_bands]
                .view(1, -1, 1, 1)
                .to(x_batch)
                * x_batch
            ) + self._scaled_mean_values[self._extract_bands].view(1, -1, 1, 1).to(
                x_batch
            )

        rgb_bands = self._symbology_rgb_bands
        if kwargs.get("rgb_bands", None) is not None:
            rgb_bands = kwargs.get("rgb_bands")

        nodata = 0
        if kwargs.get("nodata", None) is not None:
            nodata = kwargs.get("nodata")

        e = Exception(
            "`rgb_bands` should be a valid band_order, list or tuple of length 3 or 1."
        )
        symbology_bands = []
        if not (len(rgb_bands) == 3 or len(rgb_bands) == 1):
            raise (e)
        for b in rgb_bands:
            if type(b) == str:
                b_index = self._bands.index(b)
            elif type(b) == int:
                self._bands[
                    b
                ]  # To check if the band index specified by the user really exists.
                b_index = b
            else:
                raise (e)
            b_index = self._extract_bands.index(b_index)
            symbology_bands.append(b_index)

        # Extract RGB Bands
        symbology_x_batch = x_batch[:, symbology_bands]
        if stretch_type is not None:
            symbology_x_batch = image_batch_stretcher(
                symbology_x_batch, stretch_type, statistics_type
            )

        # Channel first to channel last and clamp float values to range 0 - 1 for plotting
        symbology_x_batch = symbology_x_batch.permute(0, 2, 3, 1)
        # Clamp float values to range 0 - 1
        if symbology_x_batch.mean() < 1:
            symbology_x_batch = symbology_x_batch.clamp(0, 1)

        # Get color Array
        color_array = self._multispectral_color_array
        color_array[1:, 3] = alpha
    else:
        symbology_x_batch = x_batch.permute(0, 2, 3, 1)

    # Squeeze channels if single channel (1, 224, 224) -> (224, 224)
    if symbology_x_batch.shape[-1] == 1:
        symbology_x_batch = symbology_x_batch.squeeze()

    # Size for plotting
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * imsize, nrows * imsize)
    )
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            axi = axs
            if nrows == 1:
                axi = axi
            else:
                axi = axi[r]
            if ncols == 1:
                axi = axi
            else:
                axi = axi[c]
            axi.axis("off")
            if idx < symbology_x_batch.shape[0]:
                axi.imshow(symbology_x_batch[idx].cpu().numpy())
                n_instance = y_batch[idx].unique().shape[0]
                y_merged = y_batch[idx].max(dim=0)[0].cpu().numpy()
                cmap_fn = getattr(matplotlib.cm, cmap)
                try:
                    y_rgba = cmap_fn.resampled(n_instance)(y_merged)
                except:
                    y_rgba = cmap_fn._resample(n_instance)(y_merged)
                y_rgba[y_merged == 0] = 0
                y_rgba[:, :, -1] = alpha
                axi.imshow(y_rgba)
                axi.axis("off")
            idx += 1
    if is_arcgispronotebook():
        plt.show()
