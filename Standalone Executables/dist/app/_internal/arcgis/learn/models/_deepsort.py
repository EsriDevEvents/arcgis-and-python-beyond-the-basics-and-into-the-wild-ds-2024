import os
import glob
import json
import random
import warnings
from shutil import ExecError
import traceback
from pathlib import Path

try:
    # TODO: remove unused imports
    import torch
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import mod
    from torch import resize_as_
    from fastai.vision import flatten_model
    from fastai.vision import ImageList
    from fastai.vision import Image
    from fastai.vision.image import open_image
    from fastai.vision.data import ImageDataBunch, ImageList
    from fastai.vision import imagenet_stats, normalize
    from fastai.vision.transform import crop, rotate, get_transforms, ResizeMethod

    from .._utils.common import _get_emd_path
    from ._arcgis_model import _EmptyData, _get_device
    from .._utils.env import raise_fastai_import_error
    from .._utils.pascal_voc_rectangles import ObjectDetectionCategoryList
    from ._siammask import Track  # TODO: need to move from Siammask to _tracking_utils
    from ._deepsort_train_utils import (
        get_learner,
        check_data_sanity,
        get_default_backbone,
        get_default_imgsize,
        load_for_prediction,
        get_fake_data,
    )
    from ._deepsort_predict_utils import DeepSortPredictor, get_corrected_labels_scores
    from ._tracker_util import TrackStatus

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

from ._arcgis_model import ArcGISModel
from .._utils.env import is_arcgispronotebook


class DeepSort(ArcGISModel):
    """
    Creates a :class:`~arcgis.learn.DeepSort` object.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function with `dataset_type=Imagenet`.
                            Default value is None.
                            DeepSort only supports image size of (3, 128, 64)
    =====================   ===========================================

    :return: :class:`~arcgis.learn.DeepSort` Object
    """

    # TODO: kwargs description

    def __init__(self, data, **kwargs):
        if not HAS_FASTAI:
            raise_fastai_import_error(import_exception=import_exception)

        if not check_data_sanity(data, self.supported_datasets):
            raise Exception("\nInvalid data format\n")

        if "backbone" not in kwargs:
            kwargs["backbone"] = None
        if data is None:
            data = get_fake_data()

        super().__init__(data, **kwargs)

        self._backend = "pytorch"
        self._is_multispectral = False
        self._num_classes = kwargs.get(
            "num_classes", len(self._data.class_mapping.items())
        )
        self._img_size = kwargs.get("img_size", get_default_imgsize())
        self._pretrained_path = kwargs.get("pretrained_path")
        self._infer_config = kwargs.get("infer_config", DeepSort._get_default_config())

        if not hasattr(self._data, "resize_to") or not self._data.resize_to:
            self._data.resize_to = get_default_imgsize()

        class ReID:
            def __init__(self, name=None):
                if name is None:
                    name = get_default_backbone()
                self.__name__ = name

            def get_name(self):
                return self.__name__

        self._backbone = ReID(kwargs.get("backbone"))
        self._device = _get_device()
        self.learn = get_learner(
            data, self._num_classes, self._backbone.get_name(), self._device
        )

        self._arcgis_init_callback()

        self.track_list = []
        self._predictor = None
        self._update_interval = 1
        if self._pretrained_path is not None:
            try:
                self.load(self._pretrained_path)
            except Exception:
                pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @classmethod
    def _get_default_label(cls):
        return "Object"

    @classmethod
    def get_model_type(cls):
        return "ObjectTracking"

    @classmethod
    def _get_default_config(cls):
        """
        Returns default config for inference.
        """
        infer_config = dict()
        infer_config["max_dist"] = 0.2
        infer_config["min_confidence"] = 0.1
        infer_config["nms_max_overlap"] = 0.9
        infer_config["max_iou_distance"] = 0.7
        infer_config["max_age"] = 70
        infer_config["n_init"] = 0
        infer_config["nn_budget"] = 100

        return infer_config

    @staticmethod
    def _supported_datasets():
        return ["Imagenet"]

    @staticmethod
    def _supported_backbones():
        return ["reid_v1", "reid_v2"]

    @property
    def supported_backbones(self):
        """Supported torchvision backbones for this model."""
        return DeepSort._supported_backbones()

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return DeepSort._supported_datasets()

    # TODO: bug crash on larger dataset, write custom show_results
    def show_results(self, rows=5):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================

        """
        self._check_requisites()
        from fastai.basic_data import DatasetType

        if self.learn.dl(DatasetType.Valid).batch_size > len(self.learn.data.valid_ds):
            rows = min(rows, len(self.learn.data.valid_ds))

        self.learn.show_results(rows=rows)
        if is_arcgispronotebook():
            plt.show()

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelType"] = DeepSort.get_model_type()
        _emd_template["BatchSize"] = self._data.batch_size
        _emd_template["NumClasses"] = len(self._data.classes)
        _emd_template["InferConfig"] = dict()

        for k, v in self._infer_config.items():
            _emd_template["InferConfig"][k] = v

        _emd_template["Classes"] = []
        inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
        class_data = {}
        for i, class_name in enumerate(self._data.classes):
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)]
            class_data["Color"] = color
            _emd_template["Classes"].append(class_data.copy())

        return _emd_template

    def plot_confusion_matrix(self, **kwargs):
        self._check_requisites()
        from fastai.vision.learner import _cl_int_from_learner
        from fastai.vision.learner import ClassificationInterpretation

        import copy

        learn_temp = copy.copy(self.learn)

        # Reassigning the function from vision.learner because fastai sets it from tabular.learner
        ClassificationInterpretation.from_learner = _cl_int_from_learner
        interp = ClassificationInterpretation.from_learner(learn_temp)

        nrows = self._data.c
        # figsize range: 4 <= (no. of classes + 15)/4 <=20
        fs = min(max(4, (nrows + 15) / 4), 20)
        interp.plot_confusion_matrix(figsize=(fs, fs))

    def _save_confusion_matrix(self, path):
        from IPython.utils import io

        with io.capture_output() as captured:
            self.plot_confusion_matrix()
            plt.savefig(os.path.join(path, "confusion_matrix.png"))
            plt.close()

    @property
    def _model_metrics(self):
        return {
            # "accuracy": self.batch_accuracy(show_progress=True)
        }

    # TODO: option to pass infer-config
    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a DeepSort Object tracker from an Esri Model Definition (EMD) file.

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

        :return: :class:`~arcgis.learn.DeepSort` Object
        """

        if not HAS_FASTAI:
            raise_fastai_import_error(import_exception=import_exception)

        emd_path = _get_emd_path(emd_path)
        emd_path = Path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd["ModelParameters"]
        num_classes = int(emd["NumClasses"])
        pretrained_path = str(model_file)
        chip_size = emd["ImageWidth"]
        infer_config = dict()
        for k, v in emd["InferConfig"].items():
            infer_config[k] = v

        try:
            class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}
        except KeyError:
            class_mapping = {i["ClassValue"]: i["ClassName"] for i in emd["Classes"]}

        if data is None:
            train_tfms = [rotate(degrees=30, p=0.5)]
            val_tfms = []
            transforms = (train_tfms, val_tfms)

            # TODO: possible duplication of code - see get_fake_data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                kwargs_transforms = {}
                resize_to = emd.get("resize_to")
                kwargs_transforms["size"] = (resize_to[0], resize_to[1])
                kwargs_transforms["resize_method"] = ResizeMethod.SQUISH
                data = ImageDataBunch.single_from_classes(
                    emd_path.parent.parent,
                    sorted(list(class_mapping.values())),
                    ds_tfms=transforms,
                ).normalize(imagenet_stats)

            data.chip_size = chip_size
            data.class_mapping = class_mapping
            data.classes = list(class_mapping.values())
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            data._dataset_type = "Imagenet"
        return cls(
            data,
            **model_params,
            pretrained_path=pretrained_path,
            num_classes=num_classes,
            infer_config=infer_config
        )

    def update(self, frame, detections=None, labels=None, scores=None, **kwargs):
        """
        Updates the :class:`~arcgis.learn.DeepSort` tracker.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        frame                   Required numpy array. Frame is used to
                                update the tracker.
        ---------------------   -------------------------------------------
        detections              Required list. A list of bounding boxes
                                corresponding to the detections.
                                bounding box = [xmin, ymin, width, height]
        ---------------------   -------------------------------------------
        labels                  Optional list. A list of labels
                                corresponding to the detections.
        ---------------------   -------------------------------------------
        scores                  Optional list.  A list of scores
                                corresponding to the detections.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.Track` list
        """
        if detections is None:
            detections = []
        label = DeepSort._get_default_label()
        labels, scores = get_corrected_labels_scores(
            labels, scores, len(detections), label
        )
        bbox_xywh = np.asarray(detections)
        cls_names = np.asarray(labels)
        cls_conf = np.asarray(scores)

        if bbox_xywh is None or bbox_xywh.size == 0:
            bbox_xywh = np.array([]).reshape(0, 4)
            cls_conf = np.array([])
            labels = np.array([])

        outputs = np.array([])
        if self._predictor is not None and frame is not None:
            outputs, track_labels = self._predictor.update(
                bbox_xywh, cls_conf, cls_names, frame
            )
        else:
            print(
                "\nDeepSort predictor Not initialized. Please call init() before update()\n"
            )

        # TODO: if detect_interval == 1, check for track_status
        if np.size(outputs) > 0:
            bbox_xywh = outputs[:, :4].astype(np.int)
            track_identities = outputs[:, -3].astype(np.int)
            track_scores = outputs[:, -2]
            track_ages = outputs[:, -1].astype(np.int)
            for i in range(0, len(self.track_list)):
                self.track_list[i].status = TrackStatus.lost.value

            for idt in range(0, len(track_identities)):
                create_new_track = True
                current_track_idx = -1
                for i in range(0, len(self.track_list)):
                    track = self.track_list[i]
                    if track_identities[idt] == track.id:
                        current_track_idx = i
                        create_new_track = False
                        break

                if create_new_track is True:
                    track = Track(track_identities[idt], None, None, None)
                    self.track_list.append(track)
                    current_track_idx = len(self.track_list) - 1

                self.track_list[current_track_idx].bbox = bbox_xywh[idt]
                self.track_list[current_track_idx].label = track_labels[idt]
                self.track_list[current_track_idx].score = track_scores[idt]
                self.track_list[current_track_idx].age = track_ages[idt]
                self.track_list[current_track_idx].status = TrackStatus.tracking.value

        delete_indices = []
        for i in range(0, len(self.track_list)):
            if self.track_list[i].status == TrackStatus.lost.value:
                delete_indices.append(i)

        delete_indices = sorted(delete_indices, reverse=True)
        for idx in delete_indices:
            del self.track_list[idx]
        return self.track_list

    def init(self, frame, detections=None, labels=None, scores=None, **kwargs):
        """
        Initializes the :class:`~arcgis.learn.DeepSort` tracker for inference.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        frame                   Required numpy array. Frame is used to
                                initialize the tracker.
        ---------------------   -------------------------------------------
        detections              Required list. A list of bounding boxes
                                corresponding to the detections.
        ---------------------   -------------------------------------------
        labels                  Optional list. A list of labels
                                corresponding to the detections.
        ---------------------   -------------------------------------------
        scores                  Optional list.  A list of scores
                                corresponding to the detections.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.Track` list
        """
        self.track_list = []
        self._update_interval = kwargs.get("update_interval", self._update_interval)
        deepsort = None
        learn_model = None
        if self._pretrained_path is None or not os.path.isfile(self._pretrained_path):
            if self.learn is not None:
                learn_model = self.learn.model
            else:
                raise Exception("\nInvalid initialization\n")

        try:
            deepsort = load_for_prediction(
                self._pretrained_path,
                self._num_classes,
                self._backbone.get_name(),
                learn_model,
            )
        except Exception:
            raise Exception("\nError loading model\n")

        if deepsort is not None:
            # TODO: separate predictor for each label
            self._predictor = DeepSortPredictor(
                deepsort, self._infer_config, self._device, self._update_interval
            )
        else:
            raise Exception("\nFailed Initialization\n")

        return self.update(frame, detections, labels, scores)

    # TODO: fix unchecked increment
    def remove(self, track_ids):
        """
        Removes the tracks from the track list using track_ids.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        track_ids               Required list. list of track ids to be removed
                                from the track list.
        =====================   ===========================================

        :return: Updated track list
        """
        try:
            for track_id in track_ids:
                for i in range(0, len(self.track_list)):
                    track = self.track_list[i]
                    if track.id == track_id:
                        del self.track_list[i]
                        break
            if self._predictor is not None:
                self._predictor.remove(track_ids)
        except Exception as e:
            print(e)

        return self.track_list
