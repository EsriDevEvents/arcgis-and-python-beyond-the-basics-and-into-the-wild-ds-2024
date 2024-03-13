try:
    import tensorboardX
    from fastai.basics import *
    from fastai.vision import Learner
    from fastai.callbacks.tensorboard import LearnerTensorboardWriter
    from ..models._unet_utils import ArcGISSegmentationItemList
    from fastai.vision.data import ImageList, ImageImageList
    from torch.utils.tensorboard import SummaryWriter
    from fastai.callbacks.tensorboard import *
    from fastai.core import split_kwargs_by_func
    from PIL import Image
    from torchvision.transforms import ToTensor
except:
    pass


class ArcGISTBCallback(
    LearnerTensorboardWriter,
    Learner,
    ImageImageList,
    ArcGISSegmentationItemList,
    ImageList,
):
    def __init__(self, learn, base_dir, name, arcgis_model):
        self._base_dir = base_dir
        self._name = name
        self._arcgis_model = arcgis_model
        self._current_epoch = 0
        self._current_run = name + str(self._current_epoch)
        super(ArcGISTBCallback, self).__init__(learn, base_dir, self._current_run)

    def on_train_begin(self, **kwargs: Any):
        pass

    # Override the on_train_begin method of the parent class as it causes graph related errors and warnings.#5244

    def on_epoch_end(self, last_metrics: MetricsList, iteration: int, **kwargs) -> None:
        self._current_epoch = self._current_epoch + 1
        self._current_run = self._name + "-Epoch-" + str(self._current_epoch)
        log_dir = self._base_dir / self._current_run
        self.tbwriter = SummaryWriter(str(log_dir))
        "Callback function that writes epoch end appropriate data to Tensorboard."
        if not all(x is None for x in last_metrics):
            self._write_metrics(iteration=iteration, last_metrics=last_metrics)
        # self._write_embedding(iteration=iteration)
        self._write_figure(iteration=iteration)

    def _write_figure(self, iteration: int) -> None:
        tag = "predictions vs. actuals"
        rows = 4
        thresh = 0.1
        nms_overlap = 0.1
        obj_det_models = ["FeatureClassifier", "SingleShotDetector", "RetinaNet"]
        img_to_img_models = [
            "UnetClassifier",
            "SuperResolution",
            "PSPNetClassifier",
            "DeepLab",
        ]
        other_models = [
            "ImageCaptioner",
            "MaskRCNN",
            "MultiTaskRoadExtractor",
            "ConnectNet",
        ]
        text_models = ["TextClassifier"]
        if (type(self._arcgis_model).__name__) in obj_det_models:
            fig1 = self.show_results(
                rows=rows,
                thresh=thresh,
                nms_overlap=nms_overlap,
                model=self._arcgis_model,
            )
        elif (type(self._arcgis_model).__name__) in img_to_img_models:
            if (type(self._arcgis_model).__name__) == "SuperResolution":
                if self.data.train_ds.__class__.__name__ == "SR3Dataset":
                    rows = 1
                else:
                    rows = rows
                fig1 = self.show_results_superres(rows=rows)
            else:
                fig1 = self.show_results(rows=rows)  # Segmentation
        elif (type(self._arcgis_model).__name__) in other_models:
            fig1 = self._arcgis_model.show_results(2, return_fig=True)
        # elif (type(self._arcgis_model).__name__) in text_models:
        # txt= self._arcgis_model.show_results(return_text=True)
        elif (type(self._arcgis_model).__name__) == "FasterRCNN":
            if self.data._is_multispectral:
                fig1 = self._arcgis_model._show_results_multispectral(return_fig=True)
            else:
                fig1 = self._show_results_modified(2, return_fig=True)
        elif (type(self._arcgis_model).__name__) == "CycleGAN":
            self._arcgis_model.learn.model.arcgis_results = True
            fig1 = self.show_results(rows=rows)
            self._arcgis_model.learn.model.arcgis_results = False
        elif (type(self._arcgis_model).__name__) == "Pix2Pix" or (
            type(self._arcgis_model).__name__
        ) == "Pix2PixHD":
            fig1 = self.show_results_pix2pix(rows=rows)
        else:
            return

        # Removing the support for Point CNN due to plotly orca dependency
        # if (type(self._arcgis_model).__name__) == 'PointCNN':
        #    fig1.write_image("fig1_pcnn.jpeg")
        #    image = Image.open("fig1_pcnn.jpeg")
        #    image = ToTensor()(image)
        #    self.tbwriter.add_image(tag=tag, img_tensor=image, global_step=iteration)
        # elif (type(self._arcgis_model).__name__) in text_models:
        # self.tbwriter.add_text(tag=tag, text_string =txt, global_step=iteration)
        # else:
        self.tbwriter.add_figure(
            tag=tag, figure=fig1, global_step=iteration, close=True
        )

    def show_results(self, ds_type=DatasetType.Valid, rows: int = 5, **kwargs):
        "Show `rows` result of predictions on `ds_type` dataset."
        n_items = rows**2 if self.data.train_ds.x._square_show_res else rows
        if self.dl(ds_type).batch_size < n_items:
            n_items = self.dl(ds_type).batch_size
        ds = self.dl(ds_type).dataset
        self.callbacks.append(RecordOnCPU())
        preds = self.pred_batch(ds_type)
        *self.callbacks, rec_cpu = self.callbacks
        x, y = rec_cpu.input, rec_cpu.target
        norm = getattr(self.data, "norm", False)
        if norm:
            x = self.data.denorm(x)
            if norm.keywords.get("do_y", False):
                y = self.data.denorm(y, do_x=True)
                preds = self.data.denorm(preds, do_x=True)
        analyze_kwargs, kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
        if n_items > len(preds):
            n_items = len(preds)
        preds = [
            ds.y.analyze_pred(grab_idx(preds, i), **analyze_kwargs)
            for i in range(n_items)
        ]
        xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
        if has_arg(ds.y.reconstruct, "x"):
            ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i, x in enumerate(xs)]
            zs = [ds.y.reconstruct(z, x=x) for z, x in zip(preds, xs)]
        else:
            ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
            zs = [ds.y.reconstruct(z) for z in preds]
        if (
            (type(self._arcgis_model).__name__) == "FeatureClassifier"
            or (type(self._arcgis_model).__name__) == "SingleShotDetector"
            or (type(self._arcgis_model).__name__) == "RetinaNet"
        ):
            if self.data._is_multispectral:
                fig1 = self._arcgis_model._show_results_multispectral(return_fig=True)
            else:
                fig1 = self.show_xyzs(xs, ys, zs, **kwargs)
        elif (
            (type(self._arcgis_model).__name__) == "UnetClassifier"
            or (type(self._arcgis_model).__name__) == "PSPNetClassifier"
            or (type(self._arcgis_model).__name__) == "DeepLab"
        ):
            if self.data._is_multispectral:
                fig1 = self._arcgis_model._show_results_multispectral(return_fig=True)
            else:
                fig1 = self.segment_show_xyzs(xs, ys, zs)
        elif (type(self._arcgis_model).__name__) == "SuperResolution":
            fig1 = self.img_img_show_xyzs(xs, ys, zs)
        elif (type(self._arcgis_model).__name__) == "CycleGAN":
            fig1 = self.img_tuple_show_xyzs(xs, ys, zs)
        return fig1

    def show_xyzs(
        self,
        xs,
        ys,
        zs,
        imgsize: int = 4,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        if (
            (type(self._arcgis_model).__name__) == "SingleShotDetector"
            or (type(self._arcgis_model).__name__) == "RetinaNet"
            or (type(self._arcgis_model).__name__) == "FasterRCNN"
        ):
            self._square_show_res = False
        else:
            self._square_show_res = True
        if self._square_show_res:
            title = "Ground truth\nPredictions"
            rows = int(np.ceil(math.sqrt(len(xs))))
            fig, axs = self.subplots(
                rows,
                rows,
                imgsize=imgsize,
                figsize=figsize,
                title=title,
                weight="bold",
                size=12,
            )
            for x, y, z, ax in zip(xs, ys, zs, axs.flatten()):
                x.show(ax=ax, title=f"{str(y)}\n{str(z)}", **kwargs)
            for ax in axs.flatten()[len(xs) :]:
                ax.axis("off")
        else:
            title = "Ground truth/Predictions"
            fig, axs = self.subplots(
                len(xs),
                2,
                imgsize=imgsize,
                figsize=figsize,
                title=title,
                weight="bold",
                size=14,
            )
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                x.show(ax=axs[i, 0], y=y, **kwargs)
                x.show(ax=axs[i, 1], y=z, **kwargs)

        return fig

    def show_xys(
        self,
        xs,
        ys,
        imgsize: int = 4,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        "Show the `xs` (inputs) and `ys`(targets)  on a figure of `figsize`."
        fig, axs = self.subplots(len(xs), 2, imgsize=imgsize, figsize=figsize)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x.show(ax=axs[i, 0], **kwargs)
            y.show(ax=axs[i, 1], **kwargs)
        plt.tight_layout()
        return fig

    def subplots(
        self,
        rows: int,
        cols: int,
        imgsize: int = 4,
        figsize: Optional[Tuple[int, int]] = None,
        title=None,
        **kwargs,
    ):
        "Like `plt.subplots` but with consistent axs shape, `kwargs` passed to `fig.suptitle` with `title`"
        figsize = ifnone(figsize, (imgsize * cols, imgsize * rows))
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        if rows == cols == 1:
            axs = [[axs]]  # subplots(1,1) returns Axes, not [Axes]
        elif (rows == 1 and cols != 1) or (cols == 1 and rows != 1):
            axs = [axs]
        if title is not None:
            fig.suptitle(title, **kwargs)
        return fig, array(axs)

    def segment_show_xyzs(
        self,
        xs,
        ys,
        zs,
        imgsize: int = 4,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        if self._square_show_res:
            title = "Ground truth\nPredictions"
            rows = int(np.ceil(math.sqrt(len(xs))))
            fig, axs = self.subplots(
                rows,
                rows,
                imgsize=imgsize,
                figsize=figsize,
                title=title,
                weight="bold",
                size=12,
            )
            for x, y, z, ax in zip(xs, ys, zs, axs.flatten()):
                x.show(ax=ax, title=f"{str(y)}\n{str(z)}", **kwargs)
            for ax in axs.flatten()[len(xs) :]:
                ax.axis("off")
        else:
            title = "Ground truth/Predictions"
            fig, axs = self.subplots(
                len(xs),
                2,
                imgsize=imgsize,
                figsize=figsize,
                title=title,
                weight="bold",
                size=14,
            )
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                x.show(ax=axs[i, 0], y=y, **kwargs)
                x.show(ax=axs[i, 1], y=z, **kwargs)
        return fig

    def show_results_superres(self, rows, **kwargs):
        from .._data_utils.pix2pix_data import display_row, denormalize
        from .._utils.common import get_nbatches, get_top_padding
        from .common import ArcGISMSImage
        from math import ceil

        sampling = kwargs.get("sampling_type", "ddim")
        ntimestep = kwargs.get("n_timestep", None)
        device = next(self.learn.model.parameters()).device.type

        self.learn.model.eval()
        activ = []
        top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
        denormfunc = lambda img, max, min: (img + 1) * (max - min) / 2 + min

        if self._arcgis_model.model_type == "UNet":
            x_batch, y_batch = get_nbatches(
                self._arcgis_model._data.valid_dl,
                ceil(rows / self._arcgis_model._data.batch_size),
            )
            if isinstance(x_batch[0], list):
                x_batch = x_batch[0]

            x_A, x_B = torch.cat(x_batch), torch.cat(y_batch)

            for i in range(0, x_A.shape[0], self._arcgis_model._data.batch_size):
                preds = self._arcgis_model.learn.model(
                    x_A[i : i + self._arcgis_model._data.batch_size].detach()
                )
                activ.append(preds)
            activations = torch.cat(activ)

            x_A = denormalize(x_A.cpu(), *self._arcgis_model._data._image_stats)
            x_B = denormalize(x_B.cpu(), *self._arcgis_model._data._image_stats)
            activations = denormalize(
                activations.cpu(), *self._arcgis_model._data._image_stats
            )
            rows = min(rows, x_A.shape[0])
        else:
            x_A = torch.cat([i[0] for i, _ in self._arcgis_model.learn.data.valid_dl])[
                :rows
            ]
            x_B = torch.cat([j for _, j in self._arcgis_model.learn.data.valid_dl])[
                :rows
            ]

            x_A_batch = x_A.detach()

            if sampling == "ddim":
                n_timestep = ntimestep if ntimestep else 200
                nstp = {"n_timestep": self._arcgis_model.kwargs.get("n_timestep", 1000)}
            else:
                n_timestep = (
                    ntimestep
                    if ntimestep
                    else self._arcgis_model.kwargs.get("n_timestep", 1000)
                )
                nstp = {"n_timestep": n_timestep}
            combkwargs = {**kwargs, **self._arcgis_model.kwargs, **nstp}
            self._arcgis_model.learn.model.set_new_noise_schedule(device, **combkwargs)

            preds = []
            for k in range(x_A_batch.shape[0]):
                if sampling == "ddim":
                    preds.append(
                        self._arcgis_model.learn.model.super_resolution(
                            x_A_batch[k, None],
                            continous=False,
                            sampling_timesteps=n_timestep,
                            ddim_sampling_eta=1,
                            sampling="ddim",
                        )
                    )
                else:
                    preds.append(
                        self._arcgis_model.learn.model.super_resolution(
                            x_A_batch[k, None], continous=False
                        )
                    )
            activations = torch.cat(preds)

            maxvals_a = self._arcgis_model._data.batch_stats_a["band_max_values"][
                ..., None, None
            ]
            minvals_a = self._arcgis_model._data.batch_stats_a["band_min_values"][
                ..., None, None
            ]
            maxvals_b = self._arcgis_model._data.batch_stats_b["band_max_values"][
                ..., None, None
            ]
            minvals_b = self._arcgis_model._data.batch_stats_b["band_min_values"][
                ..., None, None
            ]
            x_A = denormfunc(x_A.cpu(), maxvals_a, minvals_a)
            x_B = denormfunc(x_B.cpu(), maxvals_b, minvals_b)
            activations = denormfunc(activations.cpu(), maxvals_b, minvals_b)

        fig, axs = plt.subplots(
            nrows=rows, ncols=3, figsize=(4 * 5, rows * 5), squeeze=False
        )
        plt.subplots_adjust(top=top)
        axs[0, 0].title.set_text("Input")
        axs[0, 1].title.set_text("Target")
        axs[0, 2].title.set_text("Prediction")
        for r in range(rows):
            display_row(
                axs[r],
                (
                    ArcGISMSImage(x_A[r].cpu()),
                    ArcGISMSImage(x_B[r].cpu()),
                    ArcGISMSImage(activations[r].detach().cpu()),
                ),
                kwargs.get("rgb_bands", None),
            )
        return fig

    def img_img_show_xyzs(
        self,
        xs,
        ys,
        zs,
        imgsize: int = 4,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        if (type(self._arcgis_model).__name__) == "SuperResolution":
            self._square_show_res = False
            self._square_show = False
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        title = "Input / Prediction / Target"
        fig, axs = self.subplots(
            len(xs),
            3,
            imgsize=imgsize,
            figsize=figsize,
            title=title,
            weight="bold",
            size=14,
        )
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            x.show(ax=axs[i, 0], **kwargs)
            y.show(ax=axs[i, 2], **kwargs)
            z.show(ax=axs[i, 1], **kwargs)
        return fig

    def img_tuple_show_xyzs(
        self, xs, ys, zs, figsize: Tuple[int, int] = None, **kwargs
    ):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""
        figsize = ifnone(figsize, (12, 3 * len(xs)))
        fig, axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle("Ground truth / Predictions", weight="bold", size=14)
        for i, (x, z) in enumerate(zip(xs, zs)):
            x.to_one().show(ax=axs[i, 0], **kwargs)
            z.to_one_pred().show(ax=axs[i, 1], **kwargs)
        return fig

    def _show_results_modified(self, rows=5, **kwargs):
        if rows > len(self._arcgis_model._data.valid_ds):
            rows = len(self._arcgis_model._data.valid_ds)

        ds_type = DatasetType.Valid
        xb, yb = self._arcgis_model.learn.data.one_batch(
            ds_type, detach=False, denorm=False
        )
        ds = self._arcgis_model.learn.dl(ds_type).dataset
        if xb.shape[0] < rows**2:
            ds_type = DatasetType.Train
            xb, yb = self._arcgis_model.learn.data.one_batch(
                ds_type, detach=False, denorm=False
            )
            ds = self._arcgis_model.learn.dl(ds_type).dataset
        n_items = (
            rows**2
            if self._arcgis_model.learn.data.train_ds.x._square_show_res
            else rows
        )
        if self._arcgis_model.learn.dl(ds_type).batch_size < n_items:
            n_items = self._arcgis_model.learn.dl(ds_type).batch_size
        self._arcgis_model.learn.model.eval()
        transform_kwargs, kwargs = split_kwargs_by_func(
            kwargs, self._arcgis_model._model_conf.transform_input
        )
        try:
            preds = self._arcgis_model.learn.model(
                self._arcgis_model._model_conf.transform_input(xb, transform_kwargs)
            )
        except Exception as e:
            if getattr(self._arcgis_model, "_is_fasterrcnn", False):
                preds = []
                for _ in range(xb.shape[0]):
                    res = {}
                    res["boxes"] = torch.empty(0, 4)
                    res["scores"] = torch.tensor([])
                    res["labels"] = torch.tensor([])
                    preds.append(res)
            else:
                raise e

        x, y = to_cpu(xb), to_cpu(yb)
        norm = getattr(self._arcgis_model.learn.data, "norm", False)
        if norm:
            x = self._arcgis_model.learn.data.denorm(x)
            if norm.keywords.get("do_y", False):
                y = self._arcgis_model.learn.data.denorm(y, do_x=True)
                preds = self._arcgis_model.learn.data.denorm(preds, do_x=True)
        analyze_kwargs, kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
        preds = ds.y.analyze_pred(preds, self._arcgis_model, **analyze_kwargs)
        xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
        if has_arg(ds.y.reconstruct, "x"):
            ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i, x in enumerate(xs)]
            zs = [ds.y.reconstruct(z, x=x) for z, x in zip(preds, xs)]
        else:
            ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
            zs = [ds.y.reconstruct(z) for z in preds]
        fig1 = self.show_xyzs(xs, ys, zs, **kwargs)
        return fig1

    def show_results_pix2pix(self, rows, **kwargs):
        from .common import get_nbatches, get_top_padding, ArcGISMSImage
        from .._data_utils.pix2pix_data import denormalize, display_row

        self._arcgis_model.learn.model.eval()
        x_batch, y_batch = get_nbatches(
            self._arcgis_model._data.valid_dl,
            math.ceil(rows / self._arcgis_model._data.batch_size),
        )
        x_A, x_B = [x[0] for x in x_batch], [x[1] for x in x_batch]

        x_A = torch.cat(x_A)
        x_B = torch.cat(x_B)

        top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
        activ = []

        for i in range(0, x_B.shape[0], self._arcgis_model._data.batch_size):
            with torch.no_grad():
                preds = self._arcgis_model.learn.model(
                    x_A[i : i + self._arcgis_model._data.batch_size].detach(),
                    x_B[i : i + self._arcgis_model._data.batch_size].detach(),
                )
            activ.append(preds[0])

        activations = torch.cat(activ)
        if self._arcgis_model._data.label_nc == 0:
            x_A = denormalize(x_A.cpu(), *self._arcgis_model._data.norm_stats)
        x_B = denormalize(x_B.cpu(), *self._arcgis_model._data.norm_stats)
        activations = denormalize(
            activations.cpu(), *self._arcgis_model._data.norm_stats
        )
        rows = min(rows, x_A.shape[0])

        fig, axs = plt.subplots(
            nrows=rows, ncols=3, figsize=(4 * 5, rows * 5), squeeze=False
        )
        plt.subplots_adjust(top=top)
        axs[0, 0].title.set_text("Input")
        axs[0, 1].title.set_text("Ground Truth")
        axs[0, 2].title.set_text("Prediction")
        from fastai.vision import image2np

        for r in range(rows):
            if self._arcgis_model._data._is_multispectral:
                display_row(
                    axs[r],
                    (
                        ArcGISMSImage(x_A[r]),
                        ArcGISMSImage(x_B[r]),
                        ArcGISMSImage(activations[r]),
                    ),
                )
            else:
                display_row(
                    axs[r],
                    (image2np(x_A[r]), image2np(x_B[r]), image2np(activations[r])),
                )
        return fig
