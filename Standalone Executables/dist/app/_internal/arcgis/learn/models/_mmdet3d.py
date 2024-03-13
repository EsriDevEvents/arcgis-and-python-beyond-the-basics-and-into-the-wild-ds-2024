import traceback
from ._arcgis_model import ArcGISModel, _EmptyData
from .._data import _raise_fastai_import_error
import warnings

import_exception = None

try:
    import gc
    from fastai.basic_train import Learner
    import torch
    import numpy as np
    from .._utils.pointcloud_od import plot_results, confusion_matrix3d, predict_h5
    from .._utils.pointcloud_data import raise_maxpoint_warning
    from fastprogress.fastprogress import progress_bar
    from torch import LongTensor
    from ._ssd_utils import compute_ap_score, AveragePrecision
    import json
    from pathlib import Path
    from .._utils.common import _get_emd_path

    HAS_FASTAI = True
except Exception as e:
    import_exception = traceback.format_exc()
    HAS_FASTAI = False


def mmdet3d_model(data, **kwargs):
    from ._mmdet3d_utils import get_model

    return get_model(data, **kwargs)


def mmdet3d_loss(model_output, *model_target):
    return model_output[1]["loss"]


class MMDetection3D(ArcGISModel):
    """
    =============================   =============================================
    **Parameter**                    **Description**
    -----------------------------   ---------------------------------------------
    data                            Required fastai Databunch. Returned data object from
                                    :meth:`~arcgis.learn.prepare_data` function.
    -----------------------------   ---------------------------------------------
    model                           Required model name or path to the configuration file
                                    from :class:`~arcgis.learn.MMDetection3D` repository.
                                    The list of the supported models can be queried using
                                    :attr:`~arcgis.learn.MMDetection.supported_models`.
    -----------------------------   ---------------------------------------------
    pretrained_path                 Optional string. Path where pre-trained model is
                                    saved.
    =============================   =============================================

     **kwargs**

    =============================   =============================================
    **Parameter**                   **Description**
    -----------------------------   ---------------------------------------------
    voxel_parms                     Optional dictionary. The keys of the dictionary are
                                    `voxel_size`, `voxel_points`, and `max_voxels`. The
                                    default value of `voxel_size`,`voxel_points`, and
                                    `max_voxels` are automatically calculated based on
                                    the 'block size', 'object size' and 'average no.
                                    of points per block' of the exported data.

                                    Example:
                                        |    {'voxel_size': [0.05, 0.05, 0.1],
                                        |    'voxel_points': 10,
                                        |    'max_voxels':[20000, 40000],
                                        |    }

                                    Parameter Explanation:

                                    - 'voxel_size': List of voxel dimensions in meter
                                      [x,y,z],
                                    - 'voxel_points': An Int, that decides the maximum
                                      number of points per voxel,
                                    - 'max_voxels': List of maximum number of voxels in
                                      [training, validation].

                                    Default: None.
    =============================   =============================================

    :return: :class:`~arcgis.learn.MMDetection3D` Object
    """

    def __init__(self, data, model="SECOND", pretrained_path=None, **kwargs):
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        kwargs["voxel_parms"] = kwargs.get("voxel_parms", {})
        self._kwargs = kwargs
        self._kwargs["model"] = model
        self._check_dataset_support(data)
        super().__init__(data, None, **kwargs)
        self._backbone = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, self._config = mmdet3d_model(data, **kwargs)
        self.learn = Learner(data, model, loss_func=mmdet3d_loss)
        self.learn.metrics = [AveragePrecision(self, data.c, mode_3d=True)]
        self._reset_thresh()
        if pretrained_path is not None:
            self.load(pretrained_path)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    supported_models = ["SECOND"]
    """
    List of models supported by this class.
    """

    def lr_find(self, allow_plot=True):
        """
        Runs the Learning Rate Finder. Helps in choosing the
        optimum learning rate for training the model.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        allow_plot              Optional boolean. Display the plot of losses
                                against the learning rates and mark the optimal
                                value of the learning rate on the plot.
                                The default value is 'True'.
        =====================   ===========================================
        """
        lr = super().lr_find(allow_plot)
        lr = min(max(lr, 5e-05), 3e-03)
        return lr

    def _free_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _reset_thresh(self, detect_thresh=0.2, nms_overlap=0.2):
        self.learn.model.bbox_head.test_cfg.score_thr = detect_thresh
        self.learn.model.bbox_head.test_cfg.nms_thr = nms_overlap
        self._config.model.test_cfg.score_thr = detect_thresh
        self._config.model.test_cfg.nms_thr = nms_overlap

    def _pred_batch(self, data, detect_thresh=0.2, nms_overlap=0.5):
        self.learn.model.bbox_head.test_cfg.score_thr = detect_thresh
        self.learn.model.bbox_head.test_cfg.nms_thr = nms_overlap
        self.learn.model.eval()
        self.learn.model.prediction = True
        with torch.no_grad():
            pred = self.learn.model(data)
        self.learn.model.prediction = False
        return pred

    def show_results(self, rows=2, detect_thresh=0.3, nms_overlap=0.01, **kwargs):
        """
        Displays the results of the trained model on a part of validation/train set.
        Colors of the PointCloud are only used for better visualization, and it does
        not depict the actual classcode colors. Visualization of data, exported in a
        geographic coordinate system is not yet supported.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        ---------------------   -------------------------------------------
        detect_thresh           Optional float. The probability above which
                                a detection will be considered valid.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        color_mapping           Optional dictionary. Mapping from object id
                                to RGB values. Colors of the PointCloud via
                                color_mapping are only used for better
                                visualization, and it does not depict the
                                actual classcode colors. Default value example:
                                {0:[220,220,220], 2:[255,0,0], 6:[0,255,0]}.
        ---------------------   -------------------------------------------
        max_display_point       Optional integer. Maximum number of points
                                to display. Default is 20000. A warning will
                                be raised if the total points to display exceeds
                                this parameter. Setting this parameter will
                                randomly sample the specified number of points
                                and once set, it will be used for future uses.
        ---------------------   -------------------------------------------
        view_type               Optional string. Dataset type to display the
                                results.
                                    * ``valid`` - For validation set.
                                    * ``train`` - For training set.
                                Default: 'valid'.
        =====================   ===========================================
        """
        ds_type = kwargs.get("view_type", "valid")
        max_display_point = kwargs.get("max_display_point", 20000)
        save_html = kwargs.get("save_html", False)
        kwargs["color_mapping"] = kwargs.get("color_mapping", self._data.color_mapping)
        kwargs["class_mapping"] = self._data.class_mapping
        batch_size = self._data.valid_dl.batch_size
        row_idx = 0
        while row_idx < rows:
            if ds_type == "valid":
                data = self._data.valid_ds.get_batch(batch_size)
            else:
                data = self._data.train_ds.get_batch(batch_size)
            preds = self._pred_batch(data, detect_thresh, nms_overlap)

            for idx in range(batch_size):
                points = data["points"][idx][:, :3]
                if points.shape[0] > max_display_point:
                    raise_maxpoint_warning(
                        row_idx, kwargs, None, max_display_point, save_html
                    )
                    mask = torch.from_numpy(
                        np.random.randint(
                            0, points.shape[0], max_display_point, dtype=np.int64
                        )
                    )
                else:
                    mask = torch.arange(0, points.shape[0])
                points = points[mask]
                gt_boxes = data["gt_bboxes_3d"][idx]
                gt_boxes_labels = data["gt_labels_3d"][idx].detach().cpu()
                gt_points_labels = gt_boxes.points_in_boxes_part(points).detach().cpu()
                pred_boxes = preds[idx]["boxes_3d"]
                pred_boxes_labels = preds[idx]["labels_3d"]
                pred_points_labels = (
                    pred_boxes.points_in_boxes_part(points).detach().cpu()
                )
                points = points.detach().cpu()
                plot_results(
                    row_idx,
                    points,
                    gt_boxes,
                    gt_boxes_labels,
                    gt_points_labels,
                    pred_boxes,
                    pred_boxes_labels,
                    pred_points_labels,
                    idx2class=self._data.idx2class,
                    **kwargs,
                )
                row_idx += 1
                if row_idx == rows:
                    break
        self._free_memory()
        self._reset_thresh()

    def average_precision_score(
        self, detect_thresh=0.3, iou_thresh=0.1, nms_overlap=0.01, mean=False, **kwargs
    ):
        """
        Computes average precision on the validation/train set for each class.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        detect_thresh           Optional float. The probability above which
                                a detection will be considered for computing
                                average precision.
                                Default: 0.3.
        ---------------------   -------------------------------------------
        iou_thresh              Optional float. The intersection over union
                                threshold with the ground truth labels, above
                                which a predicted bounding box will be
                                considered a true positive.
                                Default: 0.1.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
                                Default: 0.01.
        ---------------------   -------------------------------------------
        mean                    Optional bool. If False returns class-wise
                                average precision otherwise returns mean
                                average precision.
                                Default: False.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        view_type               Optional string. Dataset type to display the
                                results.
                                    * ``valid`` - For validation set.
                                    * ``train`` - For training set.
                                Default: 'valid'.
        =====================   ===========================================

        :return: `dict` if mean is False otherwise `float`
        """
        ds_type = kwargs.get("view_type", "valid")
        show_progress = kwargs.get("show_progress", True)
        if ds_type == "valid":
            dl = self._data.valid_dl
        else:
            dl = self._data.train_dl
        classes, n_gts = (
            LongTensor(range(self._data.c)),
            torch.zeros(self._data.c).long(),
        )
        tps, pred_scores, pred_clas = [], [], []
        with torch.no_grad():
            for input, target in progress_bar(dl, display=show_progress):
                pred = self._pred_batch(input, detect_thresh, nms_overlap)
                batch_tps, batch_score, batch_clas, n_gts = confusion_matrix3d(
                    pred, target, n_gts, classes, iou_thresh
                )
                tps.extend(batch_tps)
                pred_scores.extend(batch_score)
                pred_clas.extend(batch_clas)
        aps = compute_ap_score(
            tps, pred_scores, pred_clas, n_gts, self._data.c, mode_3d=True
        )
        self._free_memory()
        self._reset_thresh()
        if mean:
            return np.mean(aps)
        else:
            return dict(zip(self._data.classes, aps))

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "average_precision"]

    @property
    def _model_metrics(self):
        return {"average_precision_score": self._get_model_metrics()}

    def _get_model_metrics(self, **kwargs):
        checkpoint = getattr(self, "_is_checkpointed", False)
        if not hasattr(self.learn, "recorder"):
            return self.average_precision_score()

        model_accuracy = self.learn.recorder.metrics[-1][0]
        if checkpoint:
            model_accuracy = self.learn.recorder.metrics[
                self.learn._best_epoch  # index using best epoch.
            ][0]

        return float(model_accuracy)

    @property
    def _is_mmsegdet(self):
        return True

    @property
    def _is_mm3d(self):
        return True

    def _get_emd_params(self, save_inference_file):
        emd_template = {"DataAttributes": {}, "ModelParameters": {}}
        emd_template["ModelType"] = "PointCloudDetection"
        emd_template["ModelParameters"]["kwargs"] = self._kwargs
        emd_template["DataAttributes"]["block_size"] = self._data.block_size
        emd_template["DataAttributes"]["max_point"] = self._data.max_point
        emd_template["DataAttributes"]["extra_features"] = self._data.extra_features
        emd_template["DataAttributes"]["pc_type"] = self._data.pc_type
        emd_template["DataAttributes"]["scale_factor"] = self._data.scale_factor
        emd_template["DataAttributes"]["color_mapping"] = self._data.color_mapping
        emd_template["DataAttributes"]["class_mapping"] = self._data.class_mapping
        emd_template["DataAttributes"]["classes"] = self._data.classes
        emd_template["DataAttributes"]["class2idx"] = self._data.class2idx
        emd_template["DataAttributes"]["idx2class"] = self._data.idx2class
        emd_template["DataAttributes"]["remap"] = self._data.remap
        emd_template["DataAttributes"]["z_range"] = self._data.z_range
        emd_template["DataAttributes"]["range"] = self._data.range
        emd_template["DataAttributes"]["average_box_size"] = self._data.average_box_size
        emd_template["DataAttributes"]["anchor_range"] = self._data.anchor_range
        emd_template["DataAttributes"]["num_features"] = self._data.num_features
        emd_template["DataAttributes"]["features_to_keep"] = self._data.features_to_keep
        emd_template["DataAttributes"]["voxel_size"] = self._data.voxel_size
        emd_template["DataAttributes"][
            "no_of_points_per_tile"
        ] = self._data.no_of_points_per_tile

        emd_template["Classes"] = []
        class_data = {}
        for class_name in self._data.classes:
            class_data["Value"] = class_name
            class_data["Name"] = self._data.class_mapping[class_name]
            class_data["Color"] = self._data.color_mapping[class_name]
            emd_template["Classes"].append(class_data.copy())

        if hasattr(self._data, "statistics") and self._data.statistics[
            "parameters"
        ].get("excludedClasses", False):
            emd_template["excludedClasses"] = self._data.statistics["parameters"][
                "excludedClasses"
            ]

        if hasattr(self._data, "statistics") and self._data.statistics.get(
            "blockShape", False
        ):
            emd_template["blockShape"] = self._data.statistics.get("blockShape")

        return emd_template

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a :class:`~arcgis.learn.MMDetection` object from an Esri Model Definition (EMD) file.

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

        :return: :class:`~arcgis.learn.MMDetection3D` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}
        color_mapping = {i["Value"]: i["Color"] for i in emd["Classes"]}

        if data is None:
            data = _EmptyData(
                path=emd_path.parent.parent,
                loss_func=None,
                c=len(class_mapping),
                chip_size=emd["ImageHeight"],
            )
            data.emd_path = emd_path
            data.emd = emd
            for key, value in emd["DataAttributes"].items():
                setattr(data, key, value)
            data.class_mapping = class_mapping
            data.color_mapping = color_mapping

        return cls(
            data, **emd["ModelParameters"]["kwargs"], pretrained_path=str(model_file)
        )

    def predict_h5(self, path, output_path=None, **kwargs):
        """
        This method is used for infrencing using HDF file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        path                    Required string. The path to folder where the HDF
                                files which needs to be predicted are present.
        ---------------------   -------------------------------------------
        output_path             Optional string. The path to folder where to dump
                                the resulting HDF files. Defaults to `results`
                                folder in input path.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        batch_size              Optional integer. The number of blocks to process
                                in one batch. Default is set to 1.
        ---------------------   -------------------------------------------
        detect_thresh           Optional float. The probability above which
                                a detection will be considered valid.
                                Default: 0.1.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
                                Default: 0.6.
        =====================   ===========================================

        :return: Path where files are dumped.
        """
        return predict_h5(self, path, output_path, **kwargs)

    def unfreeze(self):
        """
        Not implemented for this model as
        none of the layers are frozen by default.
        """
