# necessary import
import json
import traceback
from pathlib import Path

try:
    from ._model_extension import ModelExtension
    from ._arcgis_model import _EmptyData
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from .._data_utils._panoptic_data import compute_n_masks

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class MaXDeepLabConfig:
    "This class defines the required functions according to Model Extension."

    try:
        import torch
        from arcgis.learn.models import _max_deeplab_utils as maxdeeplab
    except:
        pass

    def on_batch_begin(self, learn, model_input_batch, model_target_batch, **kwargs):
        mask, label, semantic = model_target_batch
        semantic = semantic.squeeze(dim=1)
        model_target_batch = (mask, label, semantic)

        return model_input_batch, model_target_batch

    def transform_input(self, xb):
        return xb

    def transform_input_multispectral(self, xb):
        return xb

    def get_model(self, data, backbone, **kwargs):
        N = data.K  # Max num of masks in predictions #TODO: make K private variable
        n_bands = len(data._bands) if data._is_multispectral else 3
        model = self.maxdeeplab.MaXDeepLabS(
            im_size=data.chip_size,
            n_classes=data.c,
            n_masks=N,
            in_channels=n_bands,
        )
        return model

    def loss(self, model_output, *model_target):
        criterion = self.maxdeeplab.MaXDeepLabLoss()
        final_loss, loss_items = criterion(model_output, model_target)

        return final_loss

    def post_process(self, pred, thres=0.5, thinning=True, **kwargs):
        """
        In this function you have to return list with appended output for each image in the batch with shape [C=1,H,W]!

        """
        import torch.nn.functional as F

        if kwargs.get("detector", False):
            return pred
            # instance_probs = F.softmax(pred[0], dim=1)
            # instances = instance_probs.argmax(dim=1)
            # instances = F.one_hot(instances, num_classes=N).permute(0, 3, 1, 2)
            # class_confidence, classes = F.softmax(pred[1], dim=-1).max(-1)
        if kwargs.get("prob_raster", False):
            return pred[2]
        else:
            # pred = self.torch.unsqueeze(pred.argmax(dim=1), dim=1)

            pred = F.softmax(pred, dim=1).argmax(dim=1)
            pred = pred.unsqueeze(1)
        return pred


class MaXDeepLab(ModelExtension):
    """
    Creates a :class:`~arcgis.learn.MaXDeepLab` panoptic segmentation model.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data
                            object from :meth:`~arcgis.learn.prepare_data`  function.
                            MaXDeepLab only supports image sizes in
                            multiples of 16 (e.g. 256, 416, etc.).
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained
                            model is saved.
    =====================   ===========================================

    :return:  :class:`~arcgis.learn.MaXDeepLab` Object
    """

    def __init__(self, data, backbone=None, pretrained_path=None, **kwargs):
        self._check_dataset_support(data)
        super().__init__(
            data, MaXDeepLabConfig, pretrained_path=pretrained_path, **kwargs
        )
        self._backbone = None

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return MaXDeepLab._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Panoptic_Segmentation"]

    @property
    def supported_backbones(self):
        """Supported backbones for this model."""
        return MaXDeepLab._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return []

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a ``MaXDeepLab Panoptic Segmentation`` object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data`  function or None for
                                inferencing.

        =====================   ===========================================

        :return:  `MaXDeepLab Panoptic Segmentation` Object
        """
        emd_path = _get_emd_path(emd_path)

        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        backbone = emd["ModelParameters"]["backbone"]

        try:
            class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}
            color_mapping = {i["Value"]: i["Color"] for i in emd["Classes"]}
        except KeyError:
            class_mapping = {i["ClassValue"]: i["ClassName"] for i in emd["Classes"]}
            color_mapping = {i["ClassValue"]: i["Color"] for i in emd["Classes"]}

        if data is None:
            data = _EmptyData(
                path=emd_path.parent.parent,
                loss_func=None,
                c=len(class_mapping) + 1,
                chip_size=emd["ImageHeight"],
            )
            data.class_mapping = class_mapping
            data.color_mapping = color_mapping
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            data.classes = ["background"]
            for k, v in class_mapping.items():
                data.classes.append(v)
            data = get_multispectral_data_params_from_emd(data, emd)
            data.dataset_type = emd["DatasetType"]
            data.K = emd["Kwargs"]["n_masks"]
            data.instance_classes = emd["Kwargs"]["instance_classes"]

        return cls(data, backbone, pretrained_path=str(model_file))

    def compute_n_masks(self):
        """
        Computes the maximum number of class labels and masks in any chip in the entire dataset.
        Note: It might take long time for larger datasets.
        """
        return compute_n_masks(self._data.orig_path)
