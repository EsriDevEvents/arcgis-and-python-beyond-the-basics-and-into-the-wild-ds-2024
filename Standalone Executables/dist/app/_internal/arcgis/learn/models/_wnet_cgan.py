from ._codetemplate import image_translation_prf
import json
import traceback
from .._data import _raise_fastai_import_error
from ._arcgis_model import ArcGISModel, _EmptyData

try:
    from ._wnet_cgan_utils import WNetcGANLoss, WNetcGANTrainer, optim, compute_metrics
    from ._wnet_cgan_utils import WNet_cGAN as wnet_cgan_model
    from .._utils.wnet_cgan import ImageTupleListMS2, _tensor_scaler_tfm, ImageTuple
    from .._utils.common import (
        get_multispectral_data_params_from_emd,
        _get_emd_path,
        ArcGISMSImage,
    )
    import torchvision
    from .._utils.wnet_cgan import show_results
    from pathlib import Path
    from fastai.vision import *
    from fastai.vision import DatasetType, Learner, partial, open_image
    import torch

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class WNet_cGAN(ArcGISModel):

    """
    Creates a model object which generates images of type C from type A and type B.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    :return:  :class:`~arcgis.learn.WNet_cGAN` Object
    """

    def __init__(self, data, pretrained_path=None, *args, **kwargs):
        super().__init__(data)
        self._check_dataset_support(data)
        if self._data.chip_size % 256 == 0:
            wnet_cgan_gan = wnet_cgan_model(self._data.n_channel, self._data.n_channel)
            self.learn = Learner(
                data,
                wnet_cgan_gan,
                loss_func=WNetcGANLoss(wnet_cgan_gan),
                opt_func=partial(optim.Adam, betas=(0.5, 0.99)),
                callback_fns=[WNetcGANTrainer],
            )
            self.learn.model = self.learn.model.to(self._device)
            self._slice_lr = False
            if pretrained_path is not None:
                self.load(pretrained_path)
            self._code = image_translation_prf

            def __str__(self):
                return self.__repr__()

            def __repr__(self):
                return "<%s>" % (type(self).__name__)

        else:
            raise Exception("Image chip sizes should be in multiples of 256")

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a :class:`~arcgis.learn.WNet_cGAN` object from an Esri Model Definition (EMD) file.

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

        :return:  :class:`~arcgis.learn.WNet_cGAN` Object
        """

        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd["ModelParameters"]
        resize_to = emd.get("resize_to")
        chip_size = emd["ImageHeight"]
        if data is None:
            data = _EmptyData(
                path=emd_path.parent, loss_func=None, c=2, chip_size=chip_size
            )
            data = get_multispectral_data_params_from_emd(data, emd)
            data._is_multispectral = emd.get("IsMultispectral", False)
            normalization_stats = dict(
                emd.get("NormalizationStats")
            )  # Copy the normalization stats so that self._data.emd has no tensors other wise it will raise error while creating emd
            for _stat in normalization_stats:
                if normalization_stats[_stat] is not None:
                    normalization_stats[_stat] = torch.tensor(
                        normalization_stats[_stat]
                    )
                setattr(data, ("_" + _stat), normalization_stats[_stat])
            normalization_stats_b = dict(emd.get("NormalizationStats_b"))
            for _stat in normalization_stats_b:
                if normalization_stats_b[_stat] is not None:
                    normalization_stats_b[_stat] = torch.tensor(
                        normalization_stats_b[_stat]
                    )
                setattr(data, ("_" + _stat + "_b"), normalization_stats_b[_stat])
            normalization_stats_c = dict(emd.get("NormalizationStats_c"))
            for _stat in normalization_stats_c:
                if normalization_stats_c[_stat] is not None:
                    normalization_stats_c[_stat] = torch.tensor(
                        normalization_stats_c[_stat]
                    )
                setattr(data, ("_" + _stat + "_c"), normalization_stats_c[_stat])
            data.n_channel = emd["n_channel"]
            data.nband_c = emd["n_band_c"]
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            data.chip_size = chip_size
        data.resize_to = chip_size
        return cls(data, **model_params, pretrained_path=str(model_file))

    @property
    def _model_metrics(self):
        return self.compute_metrics(show_progress=True)

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_wnet_cgan"
        _emd_template["InferenceFunction"] = "ArcGISImageTranslation.py"
        _emd_template["ModelType"] = "WNet_cGAN"
        _emd_template["n_channel"] = self._data.n_channel
        _emd_template["n_band_a"] = self._data.nband_a
        _emd_template["n_band_b"] = self._data.nband_b
        _emd_template["n_band_c"] = self._data.nband_c
        _emd_template["NormalizationStats_c"] = {
            "band_min_values": self._data._band_min_values_c,
            "band_max_values": self._data._band_max_values_c,
            "band_mean_values": self._data._band_mean_values_c,
            "band_std_values": self._data._band_std_values_c,
            "scaled_min_values": self._data._scaled_min_values_c,
            "scaled_max_values": self._data._scaled_max_values_c,
            "scaled_mean_values": self._data._scaled_mean_values_c,
            "scaled_std_values": self._data._scaled_std_values_c,
        }
        for _stat in _emd_template["NormalizationStats_c"]:
            if _emd_template["NormalizationStats_c"][_stat] is not None:
                _emd_template["NormalizationStats_c"][_stat] = _emd_template[
                    "NormalizationStats_c"
                ][_stat].tolist()
        return _emd_template

    def show_results(self, rows=2, **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================

        """
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)
        show_results(self, rows, **kwargs)

    def predict(self, img_path1, img_path2):
        """
        Predicts and display the image.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        img_path1                Required path of an image 1.
        img_path2                Required path of an image 2.
        =====================   ===========================================

        """

        self.learn.model.arcgis_results = True
        img_path1, img_path2 = Path(img_path1), Path(img_path2)
        n_band = self._data.n_channel

        raw_img1, raw_img2 = (
            ArcGISMSImage.open(img_path1),
            ArcGISMSImage.open(img_path2),
        )
        if n_band > raw_img1.shape[0]:
            last_tile = np.expand_dims(raw_img1.data[raw_img1.shape[0] - 1, :, :], 0)
            res = abs(n_band - raw_img1.shape[0])
            for i in range(res):
                raw_img1 = Image(
                    torch.tensor(np.concatenate((raw_img1.data, last_tile), axis=0))
                )
        if n_band > raw_img2.shape[0]:
            last_tile = np.expand_dims(raw_img2.data[raw_img2.shape[0] - 1, :, :], 0)
            res = abs(n_band - raw_img2.shape[0])
            for i in range(res):
                raw_img2 = Image(
                    torch.tensor(np.concatenate((raw_img2.data, last_tile), axis=0))
                )
        raw_img1_scaled = _tensor_scaler_tfm(
            raw_img1.data,
            min_values=self._data._band_min_values,
            max_values=self._data._band_max_values,
            mode="minmax",
        )
        raw_img2_scaled = _tensor_scaler_tfm(
            raw_img2.data,
            min_values=self._data._band_min_values_b,
            max_values=self._data._band_max_values_b,
            mode="minmax",
        )
        raw_img1_scaled_tensor = raw_img1_scaled[None].to(self._device)
        raw_img2_scaled_tensor = raw_img2_scaled[None].to(self._device)
        self.learn.model.eval()
        with torch.no_grad():
            prediction = (
                self.learn.model(
                    raw_img1_scaled_tensor,
                    raw_img2_scaled_tensor,
                    raw_img2_scaled_tensor,
                )[0]
                .detach()[0]
                .cpu()
            )
        pred_img = prediction / 2 + 0.5
        if self._data.nband_c == 1:
            pred_np = np.array(pred_img[0, :, :])
            pred_img = ArcGISMSImage(torch.tensor([pred_np] * pred_img.shape[0]))
        else:
            pred_img = ArcGISMSImage(pred_img)
        pred_img = pred_img.show()
        self.learn.model.arcgis_results = False
        return pred_img

    def compute_metrics(self, accuracy=True, show_progress=True):
        """
        Computes Peak Signal-to-Noise Ratio (PSNR) and
        Structural Similarity Index Measure (SSIM) on validation set.

        """

        psnr, ssim, ncc = compute_metrics(self, self._data.valid_dl, show_progress)
        # return {"PSNR":'{0:1.4e}'.format(psnr), "SSIM":'{0:1.4e}'.format(ssim), "NCC":'{0:1.4e}'.format(ncc)}
        return {"PSNR": psnr, "SSIM": ssim, "NCC": ncc}

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return WNet_cGAN._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["WNet_cGAN"]
