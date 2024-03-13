from ._codetemplate import imagets_classifier_prf
import json
import traceback
import numpy as np
import pandas as pd
from IPython.display import display

from .._data import _raise_fastai_import_error
from ._arcgis_model import ArcGISModel, _EmptyData

try:
    from ._psetae_utils import FocalLoss, miou
    from ._psetae_utils import PseTae, weight_init, model_eval
    from .._data_utils.psetae_data import show_results
    from .._utils.common import _get_emd_path
    from pathlib import Path
    from fastai.vision import Learner, partial, optim

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class PSETAE(ArcGISModel):

    """
    Creates a Pixel-Set encoder + Temporal Attention Encoder sequence classifier.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            `prepare_data` function.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    **Keyword Arguments**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    mlp1                    Optional list. Dimensions of the
                            successive feature spaces of MLP1.
                            default set to [32, 64]
    ---------------------   -------------------------------------------
    pooling                 Optional string. Pixel-embedding
                            pooling strategy, can be chosen in
                            ('mean','std','max','min').
                            default set to 'mean'
    ---------------------   -------------------------------------------
    mlp2                    Optional list. Dimensions of the
                            successive feature spaces of MLP2.
                            default set to [128, 128]
    ---------------------   -------------------------------------------
    n_head                  Optional integer. Number of attention heads.
                            default set to 4
    ---------------------   -------------------------------------------
    d_k                     Optional integer. Dimension of the
                            key and query vectors. default set to 32
    ---------------------   -------------------------------------------
    dropout                 Optional float. dropout. default set to 0.2
    ---------------------   -------------------------------------------
    T                       Optional integer. Period to use for
                            the positional encoding.
                            default set to 1000
    ---------------------   -------------------------------------------
    mlp4                    Optional list. dimensions of decoder mlp
                            .default set to [64, 32]
    =====================   ===========================================

    :return: `PSETAE` Object
    """

    def __init__(self, data, pretrained_path=None, *args, **kwargs):
        super().__init__(data, pretrained_path=None, *args, **kwargs)

        self.kwargs = kwargs
        gamma = kwargs.get("gamma", 1)
        psetae = PseTae(
            input_dim=data._n_channel,
            len_max_seq=data._n_temp,
            positions=data._date_positions,
            out_class=len(data._class_map_dict),
            **kwargs,
        )

        psetae.apply(weight_init)

        self.learn = Learner(
            data,
            psetae,
            loss_func=FocalLoss(gamma),
            opt_func=partial(optim.Adam, betas=(0.5, 0.99)),
            metrics=[miou(len(data._class_map_dict))],
        )

        self.learn.model = self.learn.model.to(self._device)
        self.learn.model._device = self._device
        self._slice_lr = False
        if pretrained_path is not None:
            self.load(pretrained_path)
        self._code = imagets_classifier_prf

        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            return "<%s>" % (type(self).__name__)

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_psetae_inferencing"
        _emd_template["Kwargs"] = self.kwargs
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISImageTsClassifier.py"
        else:
            _emd_template[
                "InferenceFunction"
            ] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISImageTsClassifier.py"
        _emd_template["ModelType"] = "ImageClassification"
        _emd_template["Class_mapping"] = self._data._class_map_dict
        if self._data._num_class_map_dict:
            _emd_template["Num_class_mapping"] = self._data._num_class_map_dict
        _emd_template["n_channel"] = self._data._n_channel
        _emd_template["n_temporal"] = self._data._n_temp
        _emd_template["IsMultidimensional"] = self._data._is_multidimensional
        _emd_template["date_positions"] = self._data._date_positions
        _emd_template["ImageHeight"] = 256
        _emd_template["ImageWidth"] = 256
        _emd_template["ImageSpaceUsed"] = self._data._imagespace
        _emd_template["convertmap"] = (
            self._data._convertmap if self._data._convertmap else None
        )
        _emd_template["bandindex"] = (
            self._data._bandindex if self._data._bandindex else None
        )
        _emd_template["timeindex"] = (
            self._data._timeindex if self._data._timeindex else None
        )
        _emd_template["timestep_infer"] = self._data._timestep_infer
        _emd_template["channels_infer"] = self._data._channels_infer
        _emd_template["mean_norm_stats"] = {
            "mean_stats": [
                list(i) for i in (self._data._mean_norm_stats).astype(np.float64)
            ]
        }
        _emd_template["std_norm_stats"] = {
            "std_stats": [
                list(i) for i in (self._data._std_norm_stats).astype(np.float64)
            ]
        }

        return _emd_template

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a PSETAE object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from `prepare_data` function or None for
                                inferencing.
        =====================   ===========================================

        :return: `PSETAE` Object
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
        chip_size = emd["ImageHeight"]
        kwargs = emd.get("Kwargs", {})

        if "backbone" in kwargs.keys():
            kwargs.pop("backbone")

        if data is None:
            data = _EmptyData(
                path=emd_path.parent, loss_func=None, c=2, chip_size=chip_size
            )
            data._n_channel = emd.get("n_channel", None)
            data._n_temp = emd.get("n_temporal", None)
            data._mean_norm_stats = emd.get("mean_norm_stats", None)
            data._std_norm_stats = emd.get("std_norm_stats", None)
            data._class_map_dict = emd.get("Class_mapping", None)
            data._date_positions = emd.get("date_positions", None)
            data._convertmap = emd.get("convertmap", None)
            data.emd_path = emd_path
            data.emd = emd
            data._is_empty = True
        return cls(data, **model_params, pretrained_path=str(model_file), **kwargs)

    @property
    def _model_metrics(self):
        return self.compute_metrics()

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return PSETAE._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["RCNN_Masks"]

    def show_results(self, rows=20, **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================
        total_sample_size       Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================
        **kwargs**

        """
        show_results(self, rows, **kwargs)

    def compute_metrics(self):
        """
        Computes mean intersection over union (mIOU) and
        overall accuracy (OA) on validation set.

        """
        if not hasattr(self._data, "load_empty"):
            raise Exception("Dataset is required for compute metrics")
        class_dict = (
            self._data._num_class_map_dict
            if self._data._num_class_map_dict
            else self._data._class_map_dict
        )
        mats, miou = model_eval(
            self._data, self.learn.model, class_dict, self._data._convertmap
        )
        return {
            "mIOU": "{}".format(miou),
            "Accuracy (OA)": "{}".format(mats[1]["Accuracy"]),
        }

    def accuracy(self):
        """
        Computes overall accuracy (OA) on validation set.

        """
        if not hasattr(self._data, "load_empty"):
            raise Exception("Dataset is required for compute metrics")
        class_dict = (
            self._data._num_class_map_dict
            if self._data._num_class_map_dict
            else self._data._class_map_dict
        )
        mats, miou = model_eval(
            self._data, self.learn.model, class_dict, self._data._convertmap
        )
        return {"Accuracy (OA)": "{}".format(mats[1]["Accuracy"])}

    def mIOU(self):
        """
        Computes mean intersection over union (mIOU) on validation set.

        """
        if not hasattr(self._data, "load_empty"):
            raise Exception("Dataset is required for compute metrics")
        class_dict = (
            self._data._num_class_map_dict
            if self._data._num_class_map_dict
            else self._data._class_map_dict
        )
        mats, miou = model_eval(
            self._data, self.learn.model, class_dict, self._data._convertmap
        )
        return {"mIOU": "{}".format(miou)}

    def per_class_metrics(self):
        """
        Computes IoU, Precision, Recall, F1-score for all classes.

        """
        if not hasattr(self._data, "load_empty"):
            raise Exception("Dataset is required for compute metrics")
        class_dict = (
            self._data._num_class_map_dict
            if self._data._num_class_map_dict
            else self._data._class_map_dict
        )
        mats, _ = model_eval(
            self._data, self.learn.model, class_dict, self._data._convertmap
        )

        mat_types = ["IoU", "Precision", "Recall", "F1-score"]

        mat = []
        for i in class_dict.values():
            for j in mat_types:
                if str(i) in mats[0].keys():
                    mat.append(mats[0][str(i)][j])

        matrix_1 = np.reshape(np.array(mat), (len(mats[0].keys()), 4))

        display(
            pd.DataFrame(
                matrix_1,
                index=[
                    self._data._class_map_dict.get(i)
                    for i, j in class_dict.items()
                    if str(j) in mats[0].keys()
                ],
                columns=mat_types,
            )
        )
