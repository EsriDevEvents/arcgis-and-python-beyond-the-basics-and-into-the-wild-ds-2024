from ._codetemplate import super_resolution
import torch, json, traceback
from ... import __version__ as ArcGISLearnVersion
from .._data import prepare_data, _raise_fastai_import_error

try:
    from ._arcgis_model import ArcGISModel, _resnet_family, _EmptyData, _get_device
    from ._superres_utils import (
        compute_metrics,
        create_loss,
        UNetSR,
    )
    from ._SR3_utils import (
        UNet,
        GaussianDiffusion,
        init_weights,
        l1Loss,
    )
    from fastai.vision.learner import unet_learner, cnn_config
    from fastai.vision import nn, NormType, Learner, optim
    from fastai.callbacks import LossMetrics
    from fastai.utils.mem import Path
    from .._utils.common import _get_emd_path, ArcGISMSImage
    from .._utils.env import is_arcgispronotebook
    from fastai.core import ifnone
    from .._utils.superres import show_results
    from .._data_utils.pix2pix_data import normalize, denormalize, prepare_pix2pix_data

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class SuperResolution(ArcGISModel):

    """
    Creates a model object which increases the resolution and improves the quality of images.
    Based on Fast.ai MOOC Lesson 7 and https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    backbone                Optional string. Backbone CNN model to be used for
                            creating the base of the
                            :class:`~arcgis.learn.SuperResolution`, which
                            is `resnet34` by default.
                            Compatible backbones: 'SR3', 'resnet18', 'resnet34',
                            'resnet50', 'resnet101', 'resnet152'.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    In addition to explicitly named parameters, the SuperResolution model with 'SR3' backbone supports the optional key word arguments:

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    inner_channel           Optional int. Channel dimension.
                            Default: 64.
    ---------------------   -------------------------------------------
    norm_groups             Optional int. Group normalization.
                            Default: 32
    ---------------------   -------------------------------------------
    channel_mults           Optional int. Depth or channel multipliers.
                            Default: [1, 2, 4, 4, 8, 8]
    ---------------------   -------------------------------------------
    attn_res                Optional int. Number of attention in residual blocks.
                            Default: 16
    ---------------------   -------------------------------------------
    res_blocks              Optional int. Number of resnet block.
                            Default: 3
    ---------------------   -------------------------------------------
    dropout                 Optional bool. Dropout.
                            Default: 0
    ---------------------   -------------------------------------------
    schedule                Optional int. Type of noise schedule. available types
                            are "linear", 'warmup10', 'warmup50', 'const', 'jsd',
                            'cosine'. Default: 'linear'
    ---------------------   -------------------------------------------
    n_timestep              Optional int. Number of time-steps.
                            Default: 1000
    ---------------------   -------------------------------------------
    linear_start            Optional bool. Schedule start.
                            Default: 1e-06
    ---------------------   -------------------------------------------
    linear_end              Optional bool. Schedule end.
                            Default: 1e-02
    =====================   ===========================================

    :return: :class:`~arcgis.learn.SuperResolution` Object
    """

    def __init__(self, data, backbone=None, pretrained_path=None, *args, **kwargs):
        self._learn_version = kwargs.get("ArcGISLearnVersion", ArcGISLearnVersion)
        if backbone == "SR3":
            data_bunch = None
            if data.train_ds.__class__.__name__ == "Pix2PixHDDataset":
                downsampling_factor = data._downsampling_factor
                val_split_pct = data.val_split_pct
            else:
                downsampling_factor = data.downsample_factor
                val_split_pct = data._val_split_pct

            if isinstance(data.train_ds, list):
                data_bunch = data
            else:
                data_bunch = prepare_data(
                    path=data.train_ds.path,
                    batch_size=data.batch_size,
                    downsample_factor=downsampling_factor,
                    val_split_pct=val_split_pct,
                    seed=data.seed,
                    dataset_type="SR3",
                )
            super().__init__(data_bunch, backbone, **kwargs)
            self.kwargs = kwargs
            self._data = data_bunch if data_bunch else data
            denoiseUnet = UNet(
                in_channel=(self._data._n_channel) * 2,
                out_channel=self._data._n_channel,
                image_size=self._data.chip_size,
                with_noise_level_emb=True,
                **kwargs
            )
            sr3model = GaussianDiffusion(
                denoiseUnet,
                image_size=self._data.chip_size,
                channels=self._data._n_channel,
                device=self._device.type,
            )
            init_weights(sr3model, init_type="orthogonal")
            sr3model.set_new_noise_schedule(self._device.type, **kwargs)
            self.learn = Learner(
                self._data,
                sr3model,
                loss_func=l1Loss(self._device.type),
                opt_func=optim.Adam,
            )
            self.model_type = "SR3"
        else:
            data_bunch = None
            if data.train_ds.__class__.__name__ == "Pix2PixHDDataset":
                data_bunch = prepare_data(
                    path=data.path,
                    batch_size=data.batch_size,
                    downsample_factor=data._downsampling_factor,
                    val_split_pct=data.val_split_pct,
                    seed=data.seed,
                    dataset_type="superres",
                )
            super().__init__(data, backbone, **kwargs)
            self._data = data_bunch if data_bunch else data
            self._data._extract_bands = list(range(self._data._n_channel))
            feat_loss = create_loss(self._data._n_channel, self._device.type)
            self._check_dataset_support(self._data)
            if self._data._is_multispec:
                model = UNetSR(
                    self._data,
                    arch=self._backbone,
                    norm_type=NormType.Weight,
                )
                self.learn = Learner(
                    self._data,
                    model,
                    wd=1e-3,
                    loss_func=feat_loss,
                    callback_fns=LossMetrics,
                )
                self.learn.split(ifnone(None, cnn_config(self._backbone)["split"]))
            else:
                attention = True if self._learn_version > "2.1.0.3" else False
                self._data.c = self._data._n_channel
                self.learn = unet_learner(
                    self._data,
                    arch=self._backbone,
                    wd=1e-3,
                    loss_func=feat_loss,
                    callback_fns=LossMetrics,
                    blur=True,
                    self_attention=attention,
                    norm_type=NormType.Weight,
                )
            self.model_type = "UNet"
        self.learn.data = self._data
        self.learn.model = self.learn.model.to(self._device)
        if pretrained_path is not None:
            self.load(pretrained_path)

        self._code = super_resolution

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @staticmethod
    def _available_metrics():
        return ["valid_loss"]

    @property
    def supported_backbones(self):
        """
        Supported torchvision backbones for this model.
        """
        return SuperResolution._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return ["SR3", *_resnet_family]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a :class:`~arcgis.learn.SuperResolution` object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data` function or None for
                                inferencing.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.SuperResolution` Object
        """
        return cls.from_emd(data, emd_path)

    @classmethod
    def from_emd(cls, data, emd_path):
        """
        Creates a SuperResolution object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data` function or None for
                                inferencing.
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Esri Model Definition
                                file.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.SuperResolution` Object
        """

        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)
        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)
        model_file = Path(emd["ModelFile"])
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file
        modtype = emd.get("ModelArch", "UNet")
        model_params = emd["ModelParameters"]
        downsample_factor = emd.get("downsample_factor")
        n_channel = emd.get("n_channel", 3)
        resize_to = emd.get("resize_to")
        chip_size = emd["ImageHeight"]
        kwargs = emd.get("Kwargs", {})
        kwargs["ArcGISLearnVersion"] = emd.get("ArcGISLearnVersion", "1.0.0")

        if data is None:
            if modtype == "SR3":
                data = _EmptyData(
                    path=emd_path.parent, loss_func=None, c=2, chip_size=chip_size
                )
                data._val_split_pct = 0.1
                data._image_stats = emd.get("image_stats")
                data._image_stats2 = emd.get("image_stats2", None)
            else:
                if emd.get("is_multispec", False):
                    data = _EmptyData(
                        path=emd_path.parent, loss_func=None, c=2, chip_size=chip_size
                    )
                    data._train_tail = False
                    data._image_stats = emd.get("image_stats")
                    data._image_stats2 = emd.get("image_stats2", None)
                else:
                    data = _EmptyData(
                        path=emd_path.parent, loss_func=None, c=2, chip_size=chip_size
                    )
            data._is_multispec = emd.get("is_multispec", False)
            data._n_channel = n_channel
            data._is_empty = True
            data.emd_path = emd_path
            data.downsample_factor = downsample_factor
            data.emd = emd
            data._extract_bands = emd.get("extract_bands", None)
            data._bands = emd.get("bands", None)
            data.device = _get_device()
        backbone = "SR3" if modtype == "SR3" else model_params.get("backbone")
        data.resize_to = resize_to

        return cls(data, backbone=backbone, pretrained_path=str(model_file), **kwargs)

    @property
    def _model_metrics(self):
        return self.compute_metrics(show_progress=True)

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_superres"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISSuperResolution.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISSuperResolution.py"
        _emd_template["downsample_factor"] = self._data.downsample_factor
        _emd_template["n_channel"] = self._data._n_channel
        _emd_template["is_multispec"] = self._data._is_multispec
        _emd_template["ModelType"] = "SuperResolution"

        if self._data.train_ds.__class__.__name__ == "SR3Dataset":
            _emd_template["ModelArch"] = "SR3"
            _emd_template["image_stats"] = {
                i: j.tolist() for i, j in self._data.batch_stats_a.items() if j != None
            }
            _emd_template["image_stats2"] = {
                i: j.tolist() for i, j in self._data.batch_stats_b.items() if j != None
            }
            _emd_template["Kwargs"] = self.kwargs
        else:
            _emd_template["ModelArch"] = "UNet"
            _emd_template["image_stats"] = self._data._image_stats
            _emd_template["image_stats2"] = self._data._image_stats2
            _emd_template["extract_bands"] = self._data._extract_bands
            _emd_template["bands"] = self._data._bands
        return _emd_template

    def compute_metrics(self, accuracy=True, show_progress=True, **kwargs):
        """
        Computes Peak Signal-to-Noise Ratio (PSNR) and
        Structural Similarity Index Measure (SSIM) on validation set.

        """
        self._check_requisites()
        psnr, ssim = compute_metrics(self, self._data.valid_dl, show_progress, **kwargs)
        return {"PSNR": "{0:1.4e}".format(psnr), "SSIM": "{0:1.4e}".format(ssim)}

    def show_results(self, rows=None, **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        sampling_type           Optional string. Type of sampling.
                                Default: 'ddim'. keyword arguments applicable for
                                SR3 model type only.
        ---------------------   -------------------------------------------
        n_timestep              Optional int. Number of time-steps for the sampling process.
                                Default: 200
        =====================   ===========================================

        """
        if not rows:
            if self.model_type == "UNet":
                rows = 5
            else:
                rows = 1
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)

        self._check_requisites()
        show_results(self, rows, **kwargs)
        if is_arcgispronotebook():
            from matplotlib import pyplot as plt

            plt.show()

    def predict(self, img_path):
        """
        Predicts and display the image.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        img_path                Required path of an image.
        =====================   ===========================================

        """
        from ..models._inferencing.util import mean, std

        img_path = Path(img_path)
        raw_img = ArcGISMSImage.open(img_path)
        raw_img = raw_img.resize(self._data.chip_size)
        d_mean, d_std = mean, std

        if self._data._is_multispec:
            mean, std = self._data._image_stats[0], self._data._image_stats[1]
            d_mean, d_std = (
                self._data._image_stats2[0],
                self._data._image_stats2[1],
            )

        raw_img_tensor = normalize(raw_img.px, mean, std)
        raw_img_tensor = raw_img_tensor[None].to(self._device)

        self.learn.model.eval()
        with torch.no_grad():
            prediction = self.learn.model(raw_img_tensor)[0].detach()[0].cpu()

        pred_denorm = ArcGISMSImage(denormalize(prediction, d_mean, d_std))
        # pred_denorm = pred_denorm.show()
        return pred_denorm

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return SuperResolution._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Export_Tiles", "superres"]
