import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt

HAS_FASTAI = True

# Try to import the necessary modules
# Exception will turn the HAS_FASTAI flag to false
# so that relevant exception can be raised
try:
    import torch
    import numpy as np
    from fastai.basic_train import Learner, RecordOnCPU
    from fastai.vision import Image
    from fastai.torch_core import (
        grab_idx,
        subplots,
        to_detach,
    )
    from fastai.basic_data import DatasetType
    from arcgis.learn.models._unet_utils import show_results_multispectral
    from arcgis.learn._utils.common import dynamic_range_adjustment, kwarg_fill_none
    from arcgis.learn.models._unet_utils import ArcGISSegmentationLabelList
    from arcgis.learn._data_utils._road_orient_data import (
        _plotOrientationOnImage,
        _to_np,
    )
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


@dataclass
class MultiTaskRoadLearner(Learner):
    """show_results
    Custom Multi-Task Learner for Road Orientation training flow.
    """

    def show_results(self, ds_type=DatasetType.Valid, rows: int = 5, **kwargs):
        """
        Show `rows` result of predictions on `ds_type` dataset.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    number of rows of data to be displayed, if
                                batch size is smaller than the rows will
                                display the number provided for batch size.
        =====================   ===========================================
        **Keyword Arguments**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        alpha                   Opacity parameter for label overlay on image
                                float [0..1]
        ---------------------   -------------------------------------------
        """

        total_len = len(self.dl(ds_type))
        batch_size = self.dl(ds_type).batch_size
        self.return_fig = kwargs.get("return_fig", False)
        rows = min(total_len * batch_size, rows)
        data_iterator = iter(self.dl(ds_type))
        ds = (
            self.data.dl(DatasetType.Valid)
            .dataset.orig_data.dl(DatasetType.Valid)
            .dataset
        )
        norm = getattr(self.data.dl(DatasetType.Valid).dataset.orig_data, "norm", False)

        # for index in range(1, rows+1, batch_size):
        # n_items = (rows + 1 - index) if (index + batch_size) > (rows + 1) else batch_size
        # self.callbacks.append(RecordOnCPU())
        # batch = next(data_iterator)
        # preds = self.pred_batch(ds_type, batch=batch)
        # *self.callbacks, rec_cpu = self.callbacks
        # x, labels = rec_cpu.input, rec_cpu.target
        # x = to_detach(x)
        # labels = [to_detach(y) for y in labels]
        # if norm:
        #    x = self.data.dl(DatasetType.Valid).dataset.orig_data.denorm(x)
        #    if norm.keywords.get('do_y', False):
        #        labels = self.data.dl(DatasetType.Valid).dataset.orig_data.denorm(labels, do_x=True)
        #        preds = self.data.dl(DatasetType.Valid).dataset.orig_data.denorm(preds, do_x=True)
        # xs = [Image(x[i]) for i in range(n_items)]
        # ys = [ds.x.reconstruct(grab_idx(labels[0].unsqueeze(1), i)) for i in range(n_items)]
        # zs = [ds.x.reconstruct(grab_idx(torch.max(preds[0],axis=0)[1].unsqueeze(1),i)) for i in range(n_items)]
        # self._show_xyzs(xs, ys, zs, **kwargs)
        # ds.x.show_xyzs(xs, ys, zs, **kwargs)
        # show_results_multispectral(4,alpha=0.1)
        # self._show_pairs(xs, ys, zs, bin_size=self.data.orient_bin_size, **kwargs)
        total_len = len(self.dl(ds_type))
        batch_size = self.dl(ds_type).batch_size
        rows = min(total_len * batch_size, rows)
        if batch_size == 1:
            rows = 1
        data_iterator = iter(self.dl(ds_type))
        for index in range(1, rows + 1, batch_size):
            n_items = (
                (rows + 1 - index) if (index + batch_size) > (rows + 1) else batch_size
            )
            self.callbacks.append(RecordOnCPU())
            batch = next(data_iterator)
            preds = self.pred_batch(ds_type, batch=batch)
            *self.callbacks, rec_cpu = self.callbacks
            x, labels = rec_cpu.input, rec_cpu.target

            x = to_detach(x)
            labels = [to_detach(y).squeeze(1) for y in labels]

            xs = [
                grab_idx(x, i).numpy().transpose(1, 2, 0).astype(np.uint8)
                for i in range(n_items)
            ]
            ys = []
            for i in range(n_items):
                ys.append([_to_np(grab_idx(y, i)) for y in labels])

            zs = []
            for i in range(n_items):
                sub_zs = []
                for pred in preds:
                    _, pred_y = torch.max(pred, axis=1)
                    pred_y = pred_y.squeeze(1)
                    sub_zs.append(_to_np(grab_idx(pred_y, i)))
                zs.append(sub_zs)

            fig, axs = self._show_pairs(
                xs, ys, zs, bin_size=self.data.orient_bin_size, **kwargs
            )
            if self.return_fig:
                return fig

    def _show_pairs(
        self,
        xs,
        ys,
        zs,
        imgsize: int = 4,
        figsize: Optional[Tuple[int, int]] = None,
        bin_size: int = 10,
        **kwargs
    ):
        self.alpha = kwargs.get("alpha", 0.6)
        rows = len(xs)
        main_title = "Ground Truth  / Predictions"
        # axs = subplots(rows, 2, imgsize=imgsize, figsize=figsize, title=main_title)
        fig, axs = plt.subplots(
            nrows=rows, ncols=2, figsize=(2 * imgsize, rows * imgsize)
        )
        fig.suptitle(main_title)
        for x, y, z, ax in zip(xs, ys, zs, axs):
            if rows == 1:
                ax = axs
            ax[0].imshow(x)
            ax[0].imshow(y[0], alpha=self.alpha, cmap="binary")
            # _plotOrientationOnImage(ax[0], y[0], x, bin_size)

            ax[1].imshow(x)
            ax[1].imshow(z[0], alpha=self.alpha, cmap="binary")
            # _plotOrientationOnImage(ax[3], z[1], x, bin_size)
        for ax in axs.flatten():
            ax.axis("off")
        # plt.tight_layout()
        # self.display = kwargs.get("for_display", True)
        return fig, axs
