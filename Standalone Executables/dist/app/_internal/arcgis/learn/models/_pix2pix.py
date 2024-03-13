from ._codetemplate import image_translation_prf
import json
import traceback
from .._data import _raise_fastai_import_error
from ._arcgis_model import ArcGISModel, _EmptyData

try:
    from ._pix2pix_utils import (
        pix2pixLoss,
        Pix2PixPerceptualLoss,
        pix2pixTrainer,
        Pix2PixPerceptualTrainer,
        optim,
        compute_metrics,
        compute_fid_metric,
    )
    from ._pix2pix_utils import pix2pix as pix2pix_model
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from .._data_utils.pix2pix_data import show_results, predict
    from pathlib import Path
    from fastai.vision import DatasetType, Learner, partial
    import torch

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class Pix2Pix(ArcGISModel):

    """
    Creates a model object which generates fake images of type B from type A.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch with image chip sizes
                            in multiples of 256. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used for
                            creating the base of the :class:`~arcgis.learn.Pix2Pix`, which
                            is UNet with vanilla encoder by default.
                            Compatible backbones as encoder: 'resnet18', 'resnet34',
                            'resnet50', "resnet101", "resnet152", 'resnext50', 'wide_resnet50'
    ---------------------   -------------------------------------------
    perceptual_loss         Optional boolean. True when Perceptual loss is used.
                            Default set to False.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.Pix2Pix` Object
    """

    def __init__(
        self,
        data,
        pretrained_path=None,
        backbone=None,
        perceptual_loss=False,
        *args,
        **kwargs
    ):
        super().__init__(data, backbone, **kwargs)
        self._check_dataset_support(data)
        if self._data.chip_size % 256 == 0:
            bnds = ["o" for i in range(self._data.n_channel)]
            if not hasattr(self._data, "_bands"):
                self._data._bands = bnds
            else:
                if not self._data._bands:
                    self._data._bands = bnds
            self._data._extract_bands = list(range(self._data.n_channel))
            pix2pix_gan = pix2pix_model(
                self._data,
                self._data.n_channel,
                self._data.n_channel,
                self._backbone if backbone else None,
                perceptual_loss,
                self._data.chip_size,
            )
            if perceptual_loss:
                self.learn = Learner(
                    data,
                    pix2pix_gan,
                    loss_func=Pix2PixPerceptualLoss(pix2pix_gan),
                    opt_func=partial(optim.Adam, betas=(0.5, 0.99)),
                    callback_fns=[Pix2PixPerceptualTrainer],
                )
            else:
                self.learn = Learner(
                    data,
                    pix2pix_gan,
                    loss_func=pix2pixLoss(pix2pix_gan),
                    opt_func=partial(optim.Adam, betas=(0.5, 0.99)),
                    callback_fns=[pix2pixTrainer],
                )

            self.learn.model = self.learn.model.to(self._device)
            self._slice_lr = False
            self.perceptual_loss = perceptual_loss
            self.backbone = backbone.lower() if backbone else backbone
            if pretrained_path is not None:
                self.load(pretrained_path)
            self._code = image_translation_prf

            def __str__(self):
                return self.__repr__()

            def __repr__(self):
                return "<%s>" % (type(self).__name__)

        else:
            raise Exception("Image chip sizes should be in multiples of 256")

    @staticmethod
    def _available_metrics():
        return ["valid_loss"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a :class:`~arcgis.learn.Pix2Pix` object from an Esri Model Definition (EMD) file.

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

        :return: :class:`~arcgis.learn.Pix2Pix` Object
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
            if emd.get("IsMultispectral", False):
                data = _EmptyData(
                    path=emd_path.parent, loss_func=None, c=2, chip_size=chip_size
                )
                data = get_multispectral_data_params_from_emd(data, emd)
                data._is_multispectral = emd.get("IsMultispectral", False)
                normalization_stats_b = dict(emd.get("NormalizationStats_b"))
                for _stat in normalization_stats_b:
                    if normalization_stats_b[_stat] is not None:
                        normalization_stats_b[_stat] = torch.tensor(
                            normalization_stats_b[_stat]
                        )
                    setattr(data, ("_" + _stat), normalization_stats_b[_stat])

            else:
                data = _EmptyData(
                    path=emd_path.parent, loss_func=None, c=2, chip_size=chip_size
                )

            data.n_channel = emd.get("n_intput_channel", None)
            if data.n_channel == None:
                data.n_channel = emd.get("n_channel", None)
            data.emd_path = emd_path
            data.emd = emd
            data._is_empty = True
            data.resize_to = chip_size

        model_params["backbone"] = emd.get("backbone", None)
        model_params["perceptual_loss"] = emd.get("perceptual_loss", False)
        data._extract_bands = emd.get("extract_bands", None)
        data._bands = emd.get("bands", None)
        return cls(data, **model_params, pretrained_path=str(model_file))

    @property
    def _model_metrics(self):
        return self.compute_metrics(show_progress=True)

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_pix2pix"
        _emd_template["perceptual_loss"] = self.perceptual_loss
        _emd_template["backbone"] = self.backbone
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISImageTranslation.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISImageTranslation.py"
        _emd_template["ModelType"] = "Pix2Pix"
        _emd_template["n_intput_channel"] = self._data.n_channel
        _emd_template["NormalizationStats_b"] = {
            "band_min_values": self._data._band_min_values_b,
            "band_max_values": self._data._band_max_values_b,
            "band_mean_values": self._data._band_mean_values_b,
            "band_std_values": self._data._band_std_values_b,
            "scaled_min_values": self._data._scaled_min_values_b,
            "scaled_max_values": self._data._scaled_max_values_b,
            "scaled_mean_values": self._data._scaled_mean_values_b,
            "scaled_std_values": self._data._scaled_std_values_b,
        }
        for _stat in _emd_template["NormalizationStats_b"]:
            if _emd_template["NormalizationStats_b"][_stat] is not None:
                _emd_template["NormalizationStats_b"][_stat] = _emd_template[
                    "NormalizationStats_b"
                ][_stat].tolist()
        _emd_template["n_channel"] = len(
            _emd_template["NormalizationStats_b"]["band_min_values"]
        )
        _emd_template["extract_bands"] = self._data._extract_bands
        _emd_template["bands"] = self._data._bands
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


        **kwargs**

        =====================   ===========================================
        rgb_bands               Optional list of integers (band numbers)
                                to be considered for rgb visualization.
        =====================   ===========================================

        """
        show_results(self, rows, **kwargs)

    def predict(self, path):
        """
        Predicts and display the image.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        img_path                Required path of an image.
        =====================   ===========================================

        """
        return predict(self, path)

    def compute_metrics(self, show_progress=True):
        """
        Computes Peak Signal-to-Noise Ratio (PSNR) and
        Structural Similarity Index Measure (SSIM) on validation set.
        Additionally, computes Frechet Inception Distance (FID) for
        RGB imagery only.

        """
        psnr, ssim = compute_metrics(self, self._data.valid_dl, show_progress)
        if self._data._imagery_type_b == "RGB" and self._data.n_channel == 3:
            fid = compute_fid_metric(self, self._data)
            return {
                "PSNR": "{0:1.4e}".format(psnr),
                "SSIM": "{0:1.4e}".format(ssim),
                "FID": "{0:1.4e}".format(fid),
            }
        else:
            fid = None
            return {"PSNR": "{0:1.4e}".format(psnr), "SSIM": "{0:1.4e}".format(ssim)}

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return Pix2Pix._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Pix2Pix", "Export_Tiles"]

    @property
    def supported_backbones(self):
        """
        Supported backbones for this model.
        """
        return Pix2Pix._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50",
            "wide_resnet50",
        ]
