import traceback
from .._utils.env import raise_fastai_import_error

import_exception = None
try:
    from .._data import _raise_fastai_import_error
    from ._arcgis_model import ArcGISModel, SaveModelCallback, _set_multigpu_callback
    from ._pointcnn_utils import (
        PointCNNSeg,
        SamplePointsCallback,
        CrossEntropyPC,
        accuracy,
        CalculateClassificationReport,
        accuracy_non_zero,
        AverageMetric,
        precision,
        recall,
        f1,
    )
    from .._utils.pointcloud_data import (
        get_device,
        inference_las,
        show_results,
        compute_precision_recall,
        predict_h5,
        show_results_tool,
    )
    from ._unet_utils import is_no_color
    from fastai.basic_train import Learner
    import torch
    import numpy as np
    from fastai.callbacks import EarlyStoppingCallback
    from functools import partial
    from ._arcgis_model import _EmptyData
    import json
    from pathlib import Path
    from .._utils.common import _get_emd_path

    HAS_FASTAI = True
except Exception as e:
    import_exception = traceback.format_exc()
    HAS_FASTAI = False


class PointCNN(ArcGISModel):

    """
    Model architecture from https://arxiv.org/abs/1801.07791.
    Creates a Point Cloud classification model.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data` function.
    ---------------------   -------------------------------------------
    pretrained_path         Optional String. Path where pre-trained model
                            is saved.
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    encoder_params          Optional dictionary. The keys of the dictionary are
                            `out_channels`, `P`, `K`, `D` and `m`.

                            Examples:

                                |    {'out_channels':[16, 32, 64, 96],
                                |    'P':[-1, 768, 384, 128],
                                |    'K':[12, 16, 16, 16],
                                |    'D':[1, 1, 2, 2],
                                |    'm':8
                                |    }

                            Length of `out_channels`, `P`, `K`, `D` should be same.
                            The length denotes the number of layers in encoder.

                            Parameter Explanation

                            - 'out_channels': Number of channels produced by each layer,
                            - 'P': Number of points in each layer,
                            - 'K': Number of K-nearest neighbor in each layer,
                            - 'D': Dilation in each layer,
                            - 'm': Multiplier which is multiplied by each element of out_channel.
    ---------------------   -------------------------------------------
    dropout                 Optional float. This parameter will control overfitting.
                            The range of this parameter is [0,1).
    ---------------------   -------------------------------------------
    sample_point_num        Optional integer. The number of points that the model
                            will actually process.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.PointCNN`  Object
    """

    def __init__(self, data, pretrained_path=None, *args, **kwargs):
        super().__init__(data, None)

        if not HAS_FASTAI:
            raise_fastai_import_error(
                import_exception=import_exception,
                message="This model requires module 'torch_geometric' to be installed.",
                installation_steps=" ",
            )

        self._backbone = None
        self.sample_point_num = kwargs.get("sample_point_num", data.max_point)
        self.learn = Learner(
            data,
            PointCNNSeg(
                self.sample_point_num,
                data.c,
                data.extra_dim,
                kwargs.get("encoder_params", None),
                kwargs.get("dropout", None),
            ),
            loss_func=CrossEntropyPC(data.c),
            metrics=[
                AverageMetric(accuracy),
                AverageMetric(precision),
                AverageMetric(recall),
                AverageMetric(f1),
            ],
            callback_fns=[
                partial(SamplePointsCallback, sample_point_num=self.sample_point_num),
                CalculateClassificationReport,
            ],
        )
        self.encoder_params = self.learn.model.encoder_params

        self.learn.model = self.learn.model.to(self._device)

        if pretrained_path is not None:
            self.load(pretrained_path)

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy", "precision", "recall", "f1"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates an PointCNN model object from a Deep Learning Package(DLPK)
        or Esri Model Definition (EMD) file.

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

        :return: :class:`~arcgis.learn.PointCNN`  Object
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
                c=len(class_mapping),
                chip_size=emd["ImageHeight"],
            )
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            for key, value in emd["DataAttributes"].items():
                setattr(data, key, value)

            ## backward compatibility.
            if not hasattr(data, "class_mapping"):
                data.class_mapping = class_mapping
                if not hasattr(data, "class2idx"):
                    data.class2idx = class_mapping
            if not hasattr(data, "color_mapping"):
                data.color_mapping = color_mapping
            if not hasattr(data, "class2idx"):
                data.class2idx = data.class_mapping
            if not hasattr(data, "classes"):
                data.classes = list(data.class2idx.values())

            data.class2idx = {int(k): int(v) for k, v in data.class2idx.items()}
            if hasattr(data, "idx2class"):
                data.idx2class = {int(k): int(v) for k, v in data.idx2class.items()}

            data.color_mapping = {int(k): v for k, v in data.color_mapping.items()}

            ## Below are the lines to make save function work
            data.chip_size = None
            data._image_space_used = None
            data.dataset_type = "PointCloud"

        return cls(data, **model_params, pretrained_path=str(model_file))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    def fit(
        self,
        epochs=10,
        lr=None,
        one_cycle=True,
        early_stopping=False,
        checkpoint=True,
        tensorboard=False,
        **kwargs,
    ):
        """
        Train the model for the specified number of epochs and using the
        specified learning rates. The precision, recall and f1 scores
        shown in the training table are macro averaged over all classes.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        epochs                  Required integer. Number of cycles of training
                                on the data. Increase it if underfitting.
        ---------------------   -------------------------------------------
        lr                      Optional float or slice of floats. Learning rate
                                to be used for training the model. If ``lr=None``,
                                an optimal learning rate is automatically deduced
                                for training the model.
        ---------------------   -------------------------------------------
        one_cycle               Optional boolean. Parameter to select 1cycle
                                learning rate schedule. If set to `False` no
                                learning rate schedule is used.
        ---------------------   -------------------------------------------
        early_stopping          Optional boolean. Parameter to add early stopping.
                                If set to 'True' training will stop if parameter
                                `monitor` value stops improving for 5 epochs.
                                A minimum difference of 0.001 is required for
                                it to be considered an improvement.
        ---------------------   -------------------------------------------
        checkpoint              Optional boolean or string.
                                Parameter to save checkpoint during training.
                                If set to `True` the best model
                                based on `monitor` will be saved during
                                training. If set to 'all', all checkpoints
                                are saved. If set to False, checkpointing will
                                be off. Setting this parameter loads the best
                                model at the end of training.
        ---------------------   -------------------------------------------
        tensorboard             Optional boolean. Parameter to write the training log.
                                If set to 'True' the log will be saved at
                                <dataset-path>/training_log which can be visualized in
                                tensorboard. Required tensorboardx version=2.1

                                The default value is 'False'.

                                .. note::
                                        Not applicable for Text Models
        ---------------------   -------------------------------------------
        monitor                 Optional string. Parameter specifies
                                which metric to monitor while checkpointing
                                and early stopping. Defaults to 'valid_loss'. Value
                                should be one of the metric that is displayed in
                                the training table. Use `{model_name}.available_metrics`
                                to list the available metrics to set here.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        iters_per_epoch         Optional integer. The number of iterations
                                to run during the training phase.
        =====================   ===========================================

        """
        iterations = kwargs.get("iters_per_epoch", None)
        from ._pointcnn_utils import IterationStop

        callbacks = kwargs["callbacks"] if "callbacks" in kwargs.keys() else []
        if iterations is not None:
            del kwargs["iters_per_epoch"]
            stop_iteration_cb = IterationStop(self.learn, iterations)
            callbacks.append(stop_iteration_cb)
            kwargs["callbacks"] = callbacks
        self._check_requisites()

        if lr is None:
            print("Finding optimum learning rate.")
            lr = self.lr_find(allow_plot=False)

        if isinstance(lr, slice):
            lr = lr.stop

        super().fit(
            epochs, lr, one_cycle, early_stopping, checkpoint, tensorboard, **kwargs
        )

    @property
    def _model_metrics(self):
        return {"accuracy": self._get_model_metrics()}

    def _get_model_metrics(self, **kwargs):
        checkpoint = getattr(self, "_is_checkpointed", False)
        if not hasattr(self.learn, "recorder"):
            return 0.0

        if len(self.learn.recorder.metrics) == 0:
            return 0.0

        model_accuracy = self.learn.recorder.metrics[-1][0]
        if checkpoint:
            val_losses = self.learn.recorder.val_losses
            model_accuracy = self.learn.recorder.metrics[
                self.learn._best_epoch  # index using best epoch.
            ][0]

        return float(model_accuracy)

    def unfreeze(self):
        """
        Not implemented for this model as
        none of the layers are frozen by default.
        """

    def _get_emd_params(self, save_inference_file):
        import random

        _emd_template = {"DataAttributes": {}, "ModelParameters": {}}
        _emd_template["ModelType"] = "PointCloudClassification"
        _emd_template["Framework"] = "N/A"
        _emd_template["ModelConfiguration"] = "N/A"
        _emd_template["ExtractBands"] = "N/A"
        _emd_template["ModelParameters"]["encoder_params"] = self.encoder_params
        _emd_template["ModelParameters"]["sample_point_num"] = self.sample_point_num

        _emd_template["DataAttributes"]["block_size"] = self._data.block_size
        _emd_template["DataAttributes"]["max_point"] = self._data.max_point
        _emd_template["DataAttributes"]["extra_features"] = self._data.extra_features
        _emd_template["DataAttributes"]["extra_dim"] = self._data.extra_dim
        _emd_template["DataAttributes"]["class2idx"] = self._data.class2idx
        _emd_template["DataAttributes"]["color_mapping"] = self._data.color_mapping
        _emd_template["DataAttributes"]["remap"] = self._data.remap
        _emd_template["DataAttributes"]["classes"] = self._data.classes
        _emd_template["DataAttributes"]["subset_classes"] = self._data.subset_classes
        _emd_template["DataAttributes"]["class_mapping"] = self._data.class_mapping
        _emd_template["DataAttributes"]["remap_dict"] = self._data.remap_dict
        _emd_template["DataAttributes"]["remap_bool"] = self._data.remap_bool
        _emd_template["DataAttributes"][
            "important_classes"
        ] = self._data.important_classes
        _emd_template["DataAttributes"]["idx2class"] = self._data.idx2class
        _emd_template["DataAttributes"][
            "features_to_keep"
        ] = self._data.features_to_keep
        _emd_template["DataAttributes"]["pc_type"] = self._data.pc_type
        _emd_template["DataAttributes"][
            "background_classcode"
        ] = self._data.background_classcode

        if hasattr(self.learn.data, "statistics") and self.learn.data.statistics[
            "parameters"
        ].get("excludedClasses", False):
            _emd_template["excludedClasses"] = self.learn.data.statistics["parameters"][
                "excludedClasses"
            ]

        if self._data.pc_type == "PointCloud_TF":
            _emd_template["DataAttributes"][
                "extra_feat_indexes"
            ] = self._data.extra_feat_indexes

        _emd_template["Classes"] = []
        class_data = {}
        for i, class_name in enumerate(self._data.classes):
            inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
            class_data["Value"] = class_name
            class_data["Name"] = self._data.class_mapping[class_name]
            color = (
                [random.choice(range(256)) for i in range(3)]
                if is_no_color(self._data.color_mapping)
                else self._data.color_mapping[class_name]
            )
            class_data["Color"] = np.array(color).astype(int).tolist()
            _emd_template["Classes"].append(class_data.copy())

        if hasattr(self.learn.data, "statistics") and self.learn.data.statistics.get(
            "blockShape", False
        ):
            _emd_template["blockShape"] = self.learn.data.statistics.get("blockShape")

        return _emd_template

    def show_results(self, rows=2, **kwargs):
        """
        Displays the results from your model on the validation set
        with ground truth on the left and predictions on the right.
        Visualization of data, exported in a geographic coordinate system
        is not yet supported.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional rows. Number of rows to show. Default
                                value is 2 and maximum value is the `batch_size`
                                passed in :meth:`~arcgis.learn.prepare_data`.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        color_mapping           Optional dictionary. Mapping from class value
                                to RGB values. Default value example:
                                {0:[220,220,220], 2:[255,0,0], 6:[0,255,0]}.
        ---------------------   -------------------------------------------
        mask_class              Optional list of integers. Array containing
                                class values to mask. Use this parameter to
                                display the classes of interest.
                                Default value is [].
                                Example: All the classes are in [0, 1, 2]
                                to display only class `0` set the mask class
                                parameter to be [1, 2]. List of all classes
                                can be accessed from `data.classes` attribute
                                where `data` is the `Databunch` object returned
                                by :meth:`~arcgis.learn.prepare_data` function.
        ---------------------   -------------------------------------------
        width                   Optional integer. Width of the plot. Default
                                value is 750.
        ---------------------   -------------------------------------------
        height                  Optional integer. Height of the plot. Default
                                value is 512.
        ---------------------   -------------------------------------------
        max_display_point       Optional integer. Maximum number of points
                                to display. Default is 20000. A warning will
                                be raised if the total points to display exceeds
                                this parameter. Setting this parameter will
                                randomly sample the specified number of points
                                and once set, it will be used for future uses.
        =====================   ===========================================
        """

        if self._data.pc_type == "PointCloud_TF":
            return show_results(self, rows, **kwargs)
        elif self._data.pc_type == "PointCloud":
            return show_results_tool(self, rows, **kwargs)

    def predict_las(self, path, output_path=None, print_metrics=False, **kwargs):
        """
        Note: This method has been deprecated starting from `ArcGIS API for
        Python` version 1.9.0.
        Use `Classify Points Using Trained Model` tool  available in 3D Analyst
        extension from ArcGIS Pro 2.8 onwards.
        """

        return inference_las(path, self, output_path, print_metrics, **kwargs)

    def compute_precision_recall(self):
        """
        Computes precision, recall and f1-score on the validation sets.
        """

        return compute_precision_recall(self)

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
        =====================   ===========================================

        :return: Path where files are dumped.
        """

        return predict_h5(self, path, output_path, **kwargs)

    def _get_name_path(self, name_or_path):
        ispath_like = "\\" in name_or_path or "/" in name_or_path
        if ispath_like:
            path = Path(name_or_path)
            # to make fastai from both path and with name
            if path.is_file():
                name = path.stem
            else:
                name = path.parts[-1]
        else:
            name = name_or_path

        return name, ispath_like

    def load(self, name_or_path):
        """
        Loads a compatible saved model for inferencing or fine tuning from the disk.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Name or Path to
                                Deep Learning Package (DLPK) or
                                Esri Model Definition(EMD) file.
        =====================   ===========================================
        """
        name, ispath_like = self._get_name_path(name_or_path)

        if not Path(name_or_path).is_absolute() and ispath_like:
            name_or_path = str(Path(name_or_path).absolute())

        if Path(name_or_path).is_absolute() and name_or_path.endswith(".emd"):
            emd_path = Path(name_or_path)
        elif Path(name_or_path).is_absolute() and name_or_path.endswith(".pth"):
            emd_path = Path(name_or_path).with_suffix(".emd")
        else:
            emd_path = self.learn.path / "models" / name / f"{name}.emd"

        # from_model
        if not emd_path.exists():
            super().load(name_or_path)
            return
        with open(emd_path) as f:
            emd = json.load(f)

        try:
            class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}
            color_mapping = {i["Value"]: i["Color"] for i in emd["Classes"]}
        except KeyError:
            class_mapping = {i["ClassValue"]: i["ClassName"] for i in emd["Classes"]}
            color_mapping = {i["ClassValue"]: i["Color"] for i in emd["Classes"]}

        dummy_data = _EmptyData(
            path=emd_path.parent,
            loss_func=None,
            c=len(class_mapping),
            chip_size=emd["ImageHeight"],
        )
        dummy_data.emd_path = emd_path
        dummy_data.emd = emd
        for key, value in emd["DataAttributes"].items():
            setattr(dummy_data, key, value)

        # old models
        if not hasattr(dummy_data, "classes"):
            dummy_data.classes = [k for k in list(class_mapping.keys())]

        if not hasattr(dummy_data, "pc_type"):
            dummy_data.pc_type = "PointCloud_TF"

        if not hasattr(self._data, "pc_type"):
            self._data.pc_type = "PointCloud_TF"

        string_mapped_features = {
            "numberOfReturns": "num_returns",
            "returnNumber": "return_num",
            "nearInfrared": "nir",
        }
        inverse_string_mapped_features = {
            v: k for k, v in string_mapped_features.items()
        }

        if not hasattr(dummy_data, "features_to_keep"):
            dummy_data.features_to_keep = [
                inverse_string_mapped_features.get(f[0], f[0])
                for f in dummy_data.extra_features
            ]

        if not hasattr(self._data, "features_to_keep"):
            self._data.features_to_keep = [
                inverse_string_mapped_features.get(f[0], f[0])
                for f in self._data.extra_features
            ]

        # for message
        api_fn_name = "`export_point_dataset` function"
        tool_name = "`Prepare Point Cloud Training Data` tool"
        exported_by = (
            api_fn_name if dummy_data.pc_type == "PointCloud_TF" else tool_name
        )
        training_on = (
            api_fn_name if self._data.pc_type == "PointCloud_TF" else tool_name
        )

        if dummy_data.pc_type != self._data.pc_type:
            raise Exception(
                "Models trained on one type of exported data cannot be trained on other. "
                f"Model was trained on exported data from {exported_by}. "
                f"Usage with data exported by {training_on} is not supported. "
                f"Export the data with {exported_by} for fine-tuning."
            )

        if dummy_data.max_point != self._data.max_point:
            raise Exception(
                "Max points do not match. Please export the data again "
                f"and set max points to same as loaded model i.e. {dummy_data.max_point}."
            )

        if dummy_data.features_to_keep != self._data.features_to_keep:
            if "xyz" in dummy_data.features_to_keep:
                dummy_data.features_to_keep.remove("xyz")
            raise Exception(
                f"Extra features of your data and the model to be loaded do not match. "
                f"Set `extra_features` attribute in `prepare_data` to be {dummy_data.features_to_keep}"
            )

        if dummy_data.classes != self._data.classes:
            raise Exception(
                f"Classes in your data and loaded model do not match. Got {self._data.classes}, required {dummy_data.classes}"
            )

        super().load(name_or_path)
