import torch
import matplotlib.pyplot as plt
import math
import numpy as np
from .common import get_nbatches, image_batch_stretcher
from .._utils.env import is_arcgispronotebook


def show_batch_labeled_tiles(self, rows=3, **kwargs):  # parameters adjusted in kwargs
    """
    This function randomly picks a few training chips and visualizes them.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional Integer.
                            Number of rows to display.
                            Default: 3.
    ---------------------   -------------------------------------------
    alpha                   Optional Float.
                            Opacity of the lables for the corresponding
                            images. Values range between 0 and 1, where
                            1 means opaque.
    -------------------------------------------------------------------

    """
    from .._utils.common import denorm_x

    nrows = rows
    ncols = kwargs.get("ncols", nrows)
    # start_index = kwargs.get('start_index', 0) # Does not work with dataloader

    # Modify nrows and ncols according to the dataset
    n_items = kwargs.get("n_items", nrows * ncols)
    n_items = min(n_items, len(self.x))
    nrows = math.ceil(n_items / ncols)
    nbatches = math.ceil(n_items / self.batch_size)

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

    rgb_bands = kwargs.get("rgb_bands", self._symbology_rgb_bands)
    nodata = kwargs.get("nodata", 0)
    imsize = kwargs.get("imsize", 5)
    statistics_type = kwargs.get(
        "statistics_type", "dataset"
    )  # Accepted Values `dataset`, `DRA`
    stretch_type = kwargs.get(
        "stretch_type", "minmax"
    )  # Accepted Values `minmax`, `percentclip`

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

    # Get Batch
    x_batch, y_batch = get_nbatches(data_loader, nbatches)
    x_batch = torch.cat(x_batch)
    y_batch = torch.cat(y_batch)

    # Denormalize X
    x_batch = denorm_x(x_batch, self)

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

    # Squeeze channels if single channel (1, 224, 224) -> (224, 224)
    if symbology_x_batch.shape[-1] == 1:
        symbology_x_batch = symbology_x_batch.squeeze()

    # Get color Array
    color_array = self._multispectral_color_array

    # Size for plotting
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * imsize, nrows * imsize)
    )
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx < symbology_x_batch.shape[0]:
                axi = axs
                if nrows == 1:
                    axi = axi
                else:
                    axi = axi[r]
                if ncols == 1:
                    axi = axi
                else:
                    axi = axi[c]
                axi.imshow(symbology_x_batch[idx].cpu().numpy())

                if self.dataset_type == "MultiLabeled_Tiles":
                    one_hot_labels = y_batch[idx].tolist()
                    from itertools import compress

                    labels = compress(self.classes, one_hot_labels)
                    title = ";".join(labels)
                else:
                    title = f"{self.classes[y_batch[idx].item()]}"

                axi.set_title(title)
                axi.axis("off")
            else:
                axs[r][c].axis("off")
            idx += 1
    if is_arcgispronotebook():
        plt.show()


# Function to plot hard examples for multilabel classification
# This function has been taken from fastai and modified to work with multispectral and rgb (ArcGISMSImage).
def plot_multi_top_losses_modified(
    self, samples=3, figsize=(8, 8), save_misclassified=False
):
    "Show images in `top_losses` along with their prediction, actual, loss, and probability of predicted class in a multilabeled dataset."
    if samples > 20:
        print("Max 20 samples")
        return
    losses, idxs = self.top_losses(self.data.c)
    l_dim = len(losses.size())
    if l_dim == 1:
        losses, idxs = self.top_losses()
    (
        infolist,
        ordlosses_idxs,
        mismatches_idxs,
        mismatches,
        losses_mismatches,
        mismatchescontainer,
    ) = ([], [], [], [], [], [])
    truthlabels = np.asarray(self.y_true, dtype=int)
    classes_ids = [k for k in enumerate(self.data.classes)]
    predclass = np.asarray(self.pred_class)
    for i, pred in enumerate(predclass):
        where_truth = np.nonzero((truthlabels[i] > 0))[0]
        mismatch = np.all(pred != where_truth)
        if mismatch:
            mismatches_idxs.append(i)
            if l_dim > 1:
                losses_mismatches.append((losses[i][pred], i))
            else:
                losses_mismatches.append((losses[i], i))
        if l_dim > 1:
            infotup = (
                i,
                pred,
                where_truth,
                losses[i][pred],
                np.round(self.preds[i], decimals=3)[pred],
                mismatch,
            )
        else:
            infotup = (
                i,
                pred,
                where_truth,
                losses[i],
                np.round(self.preds[i], decimals=3)[pred],
                mismatch,
            )
        infolist.append(infotup)
    ds = self.data.dl(self.ds_type).dataset
    mismatches = ds[mismatches_idxs]
    ordlosses = sorted(losses_mismatches, key=lambda x: x[0], reverse=True)
    for w in ordlosses:
        ordlosses_idxs.append(w[1])
    mismatches_ordered_byloss = ds[ordlosses_idxs]
    print(
        f"{str(len(mismatches))} misclassified samples over {str(len(self.data.valid_ds))} samples in the validation set."
    )
    samples = min(samples, len(mismatches))
    from arcgis.learn._utils.common import ArcGISMSImage

    for ima in range(len(mismatches_ordered_byloss)):
        mismatchescontainer.append(mismatches_ordered_byloss[ima][0])
    for sampleN in range(samples):
        actualclasses = ""
        for clas in infolist[ordlosses_idxs[sampleN]][2]:
            actualclasses = f"{actualclasses} -- {str(classes_ids[clas][1])}"
        imag = mismatches_ordered_byloss[sampleN][0]
        imag = ArcGISMSImage.show(imag, return_ax=True)
        imag.set_title(
            f"""Predicted: {classes_ids[infolist[ordlosses_idxs[sampleN]][1]][1]} \nActual: {actualclasses}\nLoss: {infolist[ordlosses_idxs[sampleN]][3]}\nProbability: {infolist[ordlosses_idxs[sampleN]][4]}""",
            loc="left",
        )
        plt.show()
        if save_misclassified:
            return mismatchescontainer
