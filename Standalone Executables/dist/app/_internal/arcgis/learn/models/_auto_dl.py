from ._arcgis_model import ArcGISModel
import traceback
import json, time, datetime
from datetime import datetime as dt
import pandas as pd
from IPython.display import clear_output

try:
    import arcgis as ag
    import torch, sys
    from . import MMSegmentation, MMDetection
    import matplotlib.pyplot as plt
    from ._autodl_utils import (
        train_callback,
        ToolIsCancelled,
        _train_exhaust_mode,
        _get_emd_path,
    )
    from .._utils.evaluate_batchsize import estimate_batch_size
    from ._arcgis_model import ArcGISModel
    from .._data import prepare_data
    import numpy as np
    import time, gc
    import optuna, os
    from io import StringIO
    import os, json
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

self_obj = None


class RedirectedStdout:
    def __init__(self):
        self._stdout = None
        self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io = StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout

    def __str__(self):
        return self._string_io.getvalue()


class ImageryModel(ArcGISModel):
    """
    Imagery Model is used to fine tune models trained using AutoDL
    """

    def __init__(self):
        self._modeltype = None
        pass

    def load(self, path, data):
        """
        Loads a compatible saved model for inferencing or fine tuning from the disk,
        which can be used to further fine tune the models saved using AutoDL.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        path                    Required string. Path to
                                Esri Model Definition(EMD) or DLPK file.
        ---------------------   -------------------------------------------
        data                    Required ImageryDataObject. Returned data
                                object from :meth:`~arcgis.learn.prepare_data`  function.
        =====================   ===========================================
        """
        is_mm = False
        try:
            emd_path = _get_emd_path(path)
            f = open(emd_path)
            emd = json.load(f)
            f.close()
        except json.decoder.JSONDecodeError as E:
            raise Exception(E)
        except Exception as e:
            print(e)
            raise Exception("Not a supported Esri Model Definition(EMD) or DLPK file")
        try:
            modelname = emd["ModelName"]
            self._modeltype = emd["ModelType"]
            if "ModelFileConfigurationClass" in list(emd.keys()):
                self._modelconfig = emd["ModelFileConfigurationClass"]
                if self._modelconfig in ["MMDetectionConfig", "MMSegmentationConfig"]:
                    mm_model = emd["Kwargs"]["model"]
                    is_mm = True
        except Exception as e:
            print(e)
            raise Exception("Not a supported Esri Model Definition(EMD) or DLPK file")
        if is_mm:
            setattr(
                self,
                "imagery_model",
                getattr(ag.learn, modelname)(data, model=mm_model),
            )
        else:
            setattr(
                self,
                "imagery_model",
                getattr(ag.learn, modelname).from_model(path, data),
            )
        getattr(self, "imagery_model").load(path)

    def fit(
        self,
        epochs=10,
        lr=None,
        one_cycle=True,
        early_stopping=False,
        checkpoint=True,
        tensorboard=False,
        monitor="valid_loss",
        **kwargs
    ):
        """
        Train the model for the specified number of epochs while using the
        specified learning rates

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        epochs                  Optional integer. Number of cycles of training
                                on the data. Increase it if the model is underfitting.
                                Default value is 10.
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
                                the training table. Use ``{model_name}.available_metrics``
                                to list the available metrics to set here.
        =====================   ===========================================
        """
        try:
            getattr(self, "imagery_model").fit(
                epochs,
                lr,
                one_cycle,
                early_stopping,
                checkpoint,
                tensorboard,
                monitor,
                **kwargs
            )
        except Exception as E:
            print("Load the model first using load()")

    def show_results(self, rows=5, **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        rows                     Optional int. Number of rows of results
                                 to be displayed.
        =====================   ===========================================
        """
        getattr(self, "imagery_model").show_results(rows, **kwargs)

    def save(
        self,
        name_or_path,
        framework="PyTorch",
        publish=False,
        gis=None,
        compute_metrics=True,
        save_optimizer=False,
        save_inference_file=True,
        **kwargs
    ):
        """
        Saves the model weights, creates an Esri Model Definition and Deep
        Learning Package zip for deployment to Image Server or ArcGIS Pro.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Name of the model to save. It
                                stores it at the pre-defined location. If path
                                is passed then it stores at the specified path
                                with model name as directory name and creates
                                all the intermediate directories.
        ---------------------   -------------------------------------------
        framework               Optional string. Exports the model in the
                                specified framework format ('PyTorch', 'tflite'
                                'torchscript', and 'TF-ONXX' (deprecated)).
                                Only models saved with the default framework
                                (PyTorch) can be loaded using `from_model`.
                                ``tflite`` framework (experimental support) is
                                supported by :class:`~arcgis.learn.SingleShotDetector`,
                                :class:`~arcgis.learn.FeatureClassifier` and  :class:`~arcgis.learn.RetinaNet` .
                                ``torchscript`` format is supported by
                                :class:`~arcgis.learn.SiamMask` .
                                For usage of SiamMask model in ArcGIS Pro 2.8,
                                load the ``PyTorch`` framework saved model
                                and export it with ``torchscript`` framework
                                using ArcGIS API for Python v1.8.5.
                                For usage of SiamMask model in ArcGIS Pro 2.9,
                                set framework to ``torchscript`` and use the
                                model files additionally generated inside
                                'torch_scripts' folder.
                                If framework is ``TF-ONNX`` (Only supported for
                                :class:`~arcgis.learn.SingleShotDetector`), ``batch_size`` can
                                be passed as an optional keyword argument.
        ---------------------   -------------------------------------------
        publish                 Optional boolean. Publishes the DLPK as an item.
        ---------------------   -------------------------------------------
        gis                     Optional :class:`~arcgis.gis.GIS`  Object. Used for publishing the item.
                                If not specified then active gis user is taken.
        ---------------------   -------------------------------------------
        compute_metrics         Optional boolean. Used for computing model
                                metrics.
        ---------------------   -------------------------------------------
        save_optimizer          Optional boolean. Used for saving the model-optimizer
                                state along with the model. Default is set to False
        ---------------------   -------------------------------------------
        save_inference_file     Optional boolean. Used for saving the inference file
                                along with the model.
                                If False, the model will not work with ArcGIS Pro 2.6
                                or earlier. Default is set to True.
        ---------------------   -------------------------------------------
        kwargs                  Optional Parameters:
                                Boolean `overwrite` if True, it will overwrite
                                the item on ArcGIS Online/Enterprise, default False.
        =====================   ===========================================
        """
        getattr(self, "imagery_model").save(
            name_or_path,
            framework,
            publish,
            gis,
            compute_metrics,
            save_optimizer,
            save_inference_file,
            **kwargs
        )

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
        try:
            lr = getattr(self, "imagery_model").lr_find(allow_plot)
            return lr
        except Exception as E:
            print("Load the model first using load()")

    def available_metrics(self):
        """
        List of available metrics that are displayed in the training
        table. Set `monitor` value to be one of these while calling
        the :class:`~arcgis.learn.ImageryModel.fit` method.
        """
        try:
            return getattr(self, "imagery_model").available_metrics
        except Exception as E:
            print("Load the model first using load()")

    def plot_losses(self):
        """
        Plot validation and training losses after fitting the model.
        """
        try:
            return getattr(self, "imagery_model").plot_losses()
        except Exception as E:
            print("Load the model first using load()")

    def unfreeze(self):
        """
        Unfreezes the earlier layers of the model for fine-tuning.
        """
        try:
            if hasattr(getattr(self, "imagery_model"), "unfreeze"):
                return getattr(self, "imagery_model").unfreeze()
            else:
                print("This model does not support unfreeze method")
        except Exception as E:
            print("Load the model first using load()")

    def mIOU(self):
        """
        Computes mean IOU on the validation set for each class.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        mean                    Optional bool. If False returns class-wise
                                mean IOU, otherwise returns mean iou of all
                                classes combined.
        ---------------------   -------------------------------------------
        show_progress           Optional bool. Displays the progress bar if
                                True.
        =====================   ===========================================

        :return: `dict` if mean is False otherwise `float`
        """
        if self._modeltype is not None:
            if self._modeltype == "ObjectDetection":
                print("This method is not supported with Object Detection models")
                return
            else:
                try:
                    return getattr(self, "imagery_model").mIOU()
                except Exception as E:
                    print("Load the model first using load()")
        else:
            print("Train the model first using fit()")
            return

    def average_precision_score(self):
        """
        Computes average precision on the validation set for each class.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        detect_thresh           Optional float. The probability above which
                                a detection will be considered for computing
                                average precision.
        ---------------------   -------------------------------------------
        iou_thresh              Optional float. The intersection over union
                                threshold with the ground truth labels, above
                                which a predicted bounding box will be
                                considered a true positive.
        ---------------------   -------------------------------------------
        mean                    Optional bool. If False returns class-wise
                                average precision otherwise returns mean
                                average precision.
        =====================   ===========================================

        :return: `dict` if mean is False otherwise `float`
        """
        if self._modeltype is not None:
            if self._modeltype != "ObjectDetection":
                print("This method is not supported with pixel classification model")
                return
            else:
                try:
                    return getattr(self, "imagery_model").average_precision_score()
                except Exception as E:
                    print("Load the model first using load()")
        else:
            print("Train the model first using fit()")
            return


all_val_losses, all_train_losses, dice, temp_log_msg = [], [], [], []
BestPerformingModel, best_backbone, best_model = None, None, None

# Build neural network model


class AutoDL:
    """
    Automates the process of model selection, training and hyperparameter tuning of
    arcgis.learn supported deep learning models within a specified time limit.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required ImageryDataObject. Returned data object from
                            :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    total_time_limit        Optional Int. The total time limit in hours for
                            AutoDL training.
                            Default is 2 Hr.
    ---------------------   -------------------------------------------
    mode                    Optional String.
                            Can be "basic" or "advanced".

                            * basic : To be used when the user wants to train all selected networks.

                            * advanced : To be used when the user wants to tune hyper parameters of two
                            best performing models from basic mode.
    ---------------------   -------------------------------------------
    network                 Optional List of str.
                            The list of models that will be used in the training.
                            For eg:
                            Supported Object Detection models:
                            ["SingleShotDetector", "RetinaNet", "FasterRCNN", "YOLOv3", "MaskRCNN", "DETReg" ,"ATSS",
                            "CARAFE", "CascadeRCNN", "CascadeRPN", "DCN", 'Detectors',
                            'DoubleHeads', 'DynamicRCNN', 'EmpiricalAttention', 'FCOS', 'FoveaBox',
                            'FSAF', 'GHM', 'LibraRCNN', 'PaFPN', 'PISA', 'RegNet','RepPoints',
                            'Res2Net', 'SABL', 'VFNet']
                            Supported Pixel Classification models:
                            ["DeepLab", "UnetClassifier", "PSPNetClassifier",
                                "ANN", "APCNet", "CCNet", "CGNet", "HRNet", 'DeepLabV3Plus',
                                'DMNet', 'DNLNet', 'FastSCNN', 'FCN', 'GCNet', 'MobileNetV2',
                                'NonLocalNet','OCRNet', 'PSANet', 'SemFPN', 'UperNet']

    ---------------------   -------------------------------------------
    verbose                 Optional Boolean.
                            To be used to display logs while training the models.
                            Default is True.

    =====================   ===========================================

    :return:
        :class:`~arcgis.learn.AutoDL` Object
    """

    def __init__(
        self,
        data=None,
        total_time_limit=2,
        mode="basic",
        network=None,
        verbose=True,
        **kwargs
    ):
        if "save_evaluated_models" in kwargs:
            self._save_evaluated_models = kwargs["save_evaluated_models"]
        else:
            self._save_evaluated_models = True

        if "output_folder" in kwargs:
            self._output_path = kwargs["output_folder"]
            self._save_to_folder = True
        else:
            self._output_path = data.path
            self._save_to_folder = False

        if verbose:
            self._logger_dict = []

        prepare_data_args = data.arcgis_init_kwargs
        prepare_data_args["batch_size"] = None
        self._data = prepare_data(**prepare_data_args)

        self.verbose = verbose
        algorithms = network
        self._total_training_time = 0
        self._max_image_set = 500
        self._max_epochs = 20
        self._remaining_time = 0
        self._exhaustive_mode_studies = []
        self._epoch_obj = {}
        self._all_algorithms = [
            "DeepLab",
            "UnetClassifier",
            "PSPNetClassifier",
            "ANN",
            "APCNet",
            "CCNet",
            "CGNet",
            "HRNet",
            "DeepLabV3Plus",
            "DMNet",
            "DNLNet",
            "EMANet",
            "FastSCNN",
            "FCN",
            "GCNet",
            "MobileNetV2",
            "NonLocalNet",
            "OCRNet",
            "PSANet",
            "SemFPN",
            "UperNet",
            "SingleShotDetector",
            "MaskRCNN",
            "RetinaNet",
            "FasterRCNN",
            "YOLOv3",
            "DETReg",
            "ATSS",
            "CARAFE",
            "CascadeRPN",
            "CascadeRCNN",
            "DCN",
            "Detectors",
            "DoubleHeads",
            "DynamicRCNN",
            "EmpiricalAttention",
            "FCOS",
            "FoveaBox",
            "FSAF",
            "GHM",
            "LibraRCNN",
            "PaFPN",
            "PISA",
            "RegNet",
            "RepPoints",
            "Res2Net",
            "SABL",
            "VFNet",
            "MaskRCNN",
        ]
        self._all_mm_algorithms = [
            "ANN",
            "APCNet",
            "CCNet",
            "CGNet",
            "HRNet",
            "ATSS",
            "CARAFE",
            "CascadeRCNN",
            "CascadeRPN",
            "DCN",
            "Detectors",
            "DoubleHeads",
            "DynamicRCNN",
            "EmpiricalAttention",
            "FCOS",
            "FoveaBox",
            "FSAF",
            "GHM",
            "LibraRCNN",
            "PaFPN",
            "PISA",
            "RegNet",
            "RepPoints",
            "Res2Net",
            "SABL",
            "VFNet",
            "DeepLabV3Plus",
            "DMNet",
            "DNLNet",
            "EMANet",
            "FastSCNN",
            "FCN",
            "GCNet",
            "MobileNetV2",
            "NonLocalNet",
            "OCRNet",
            "PSANet",
            "SemFPN",
            "UperNet",
            "MaskRCNN",
        ]
        self._train_df = None
        self._average_precision_score_df = None
        self._mIOU_df = None
        self.best_model = None
        self._best_backbone = None
        self._all_losses = {}
        self._max_accuracy = 0
        self._train_callback = train_callback
        self._remaining_time = 0
        self._all_detection_data = [
            "PASCAL_VOC_rectangles",
            "KITTI_rectangles",
            "RCNN_Masks",
        ]
        if total_time_limit < 0.25:
            raise Exception(
                "Total time limit should be greater than or equal to 0.25 hr"
            )
        total_time_limit = total_time_limit * 60
        self._training_mode = mode.lower()
        if self._training_mode == "perform":
            self._training_mode = "advanced"

        if self._training_mode == "basic":
            self._time_in_sec = total_time_limit * 60
        elif self._training_mode == "advanced":
            self._time_in_sec = (total_time_limit * 60) // 2
        else:
            print("Please select a vaild mode for training..")
            return
        self._algos = []
        self._model_type = self._data._dataset_type

        if self._model_type == "Classified_Tiles":
            if algorithms is None:
                algorithms = self.supported_classification_models()
            all_algos = self.supported_classification_models()
            for algo in algorithms:
                if algo not in all_algos:
                    error = algo + " is not a supported classification model."
                    raise Exception(error)
                if algo == "MMSegmentation":
                    self._algos.extend(MMSegmentation.supported_models)
                else:
                    self._algos.append(algo)
        elif self._model_type in self._all_detection_data:
            if algorithms is None:
                if self._model_type != "RCNN_Masks":
                    algorithms = self.supported_detection_models()
                    algorithms.remove("MaskRCNN")
                else:
                    algorithms = ["MaskRCNN"]

            if self._model_type == "RCNN_Masks":
                self._algos.append("MaskRCNN")
                for algo in algorithms:
                    if algo != "MaskRCNN":
                        error = algo + " is not supported for RCNN_Masks dataset type."
                        raise Exception(error)
            else:
                all_algos = self.supported_detection_models()
                for algo in algorithms:
                    if algo not in all_algos:
                        error = algo + " is not a supported Object Detection model."
                        raise Exception(error)
                    if algo == "MMDetection":
                        self._algos.extend(MMDetection.supported_models)
                    if algo == "MaskRCNN":
                        error = (
                            algo + " is only compatible with RCNN_Masks dataset type."
                        )
                        raise Exception(error)
                    else:
                        self._algos.append(algo)
        else:
            raise Exception("Data must be in ESRI defined format")
        model_stats = self._model_stats()

        if "MaskRCNN" in algorithms and mode == "basic":
            raise Exception("MaskRCNN is only supported with advanced mode.")

        finetune_supported = [
            "SingleShotDetector",
            "MaskRCNN",
            "RetinaNet",
            "DETReg",
            "FasterRCNN",
            "PSPNetClassifier",
            "UnetClassifier",
            "DeepLab",
        ]
        all_mm_models = True
        for algo in self._algos:
            if algo in self._all_algorithms:
                self._total_training_time += int(model_stats[algo]["time"])
            if algo in finetune_supported:
                all_mm_models = False

        if all_mm_models and self._training_mode == "advanced":
            self._time_in_sec = self._time_in_sec * 2

        self._total_training_time //= 60
        if self._total_training_time == 0:
            self._total_training_time = 1

        # self._algos = self._sort_algos(self._algos)

        if total_time_limit is None:
            total_time_limit = self._total_training_time

        number_of_images = len(self._data.train_ds) + len(self._data.valid_ds)
        # print(self._total_training_time, number_of_images, self._max_image_set)
        required_time = (
            self._total_training_time * number_of_images
        ) // self._max_image_set

        if required_time == 0:
            required_time = 1

        self._tiles_required = (
            self._max_image_set * total_time_limit
        ) // self._total_training_time

        if self._tiles_required >= number_of_images:
            self._tiles_required = number_of_images

        ## Edit max time here
        if total_time_limit > required_time:
            self._max_epochs = self._get_max_epochs(total_time_limit, required_time)

        ## Max time edit ends
        # if self._tiles_required <= self.batch_size:
        #     self.batch_size = int(self._tiles_required // 2)
        if round(total_time_limit / 60, 2) == 1:
            unit = "hour"
        else:
            unit = "hours"
        print(
            "Given time to process the dataset is:",
            round(total_time_limit / 60, 2),
            unit,
        )
        print(
            "Number of images that can be processed in the given time:",
            self._tiles_required,
        )
        print(
            "Time required to process the entire dataset of",
            len(self._data.train_ds) + len(self._data.valid_ds),
            "images is",
            round((required_time / 60), 2),
            "hours",
        )

    def _get_max_epochs(self, total_time, required_time):
        self._remaining_time = total_time - required_time
        model_stats = self._model_stats()
        model_epochs = 0
        for algo in self._algos:
            mt = model_stats[algo]["time"]
            model_time_required = (
                (mt // 60) * self._tiles_required
            ) // self._max_image_set
            if model_time_required == 0:
                model_time_required = 1
            time_ratio = (model_time_required / required_time) * 100
            model_remaining_time = (time_ratio / 100) * self._remaining_time
            model_epochs = int((20 * model_remaining_time) // model_time_required)
            break

        self._max_epochs += model_epochs
        return self._max_epochs

    def _train_model(
        self,
        model,
        backbone=None,
        epochs=20,
        model_type="classification",
        model_time=1600,
    ):
        """
        Train the AutoDL models.
        """
        self._is_best = False
        start_time = time.time()
        if model_type == "classification":
            mm_model = "MMSegmentation"
        if model_type == "detection":
            mm_model = "MMDetection"
        if self.verbose:
            log_msg = "{date}: Initializing the {network} network.".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"), network=model
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        if backbone is None:
            if not self._model_stats()[model]["is_mm"]:
                setattr(self, model, getattr(ag.learn, model)(self._data))
                try:
                    estimated_batch_size = estimate_batch_size(getattr(self, model))
                except:
                    estimated_batch_size = (2, 2)
                callbacks = [
                    self._train_callback(
                        getattr(self, model).learn,
                        self._tiles_required // self._data.batch_size,
                    )
                ]
            else:
                model_with_underscore = [
                    "CascadeRCNN",
                    "CascadeRPN",
                    "EmpiricalAttention",
                    "MobileNetV2",
                    "NonLocalNet",
                    "SemFPN",
                    "DoubleHeads",
                    "DynamicRCNN",
                    "LibraRCNN",
                    "NasFcos",
                ]
                paired_model = [
                    "Cascade_RCNN",
                    "Cascade_RPN",
                    "Empirical_Attention",
                    "MobileNet_V2",
                    "NonLocal_Net",
                    "Sem_FPN",
                    "Double_Heads",
                    "Dynamic_RCNN",
                    "Libra_RCNN",
                    "Nas_Fcos",
                ]

                if model in model_with_underscore:
                    index = model_with_underscore.index(model)
                    setattr(
                        self,
                        model,
                        getattr(ag.learn, mm_model)(
                            self._data, model=paired_model[index]
                        ),
                    )
                else:
                    setattr(
                        self,
                        model,
                        getattr(ag.learn, mm_model)(self._data, model=model.lower()),
                    )
                try:
                    estimated_batch_size = estimate_batch_size(getattr(self, model))
                except:
                    estimated_batch_size = (2, 2)
                callbacks = [
                    self._train_callback(
                        getattr(self, model).learn,
                        self._tiles_required // self._data.batch_size,
                    )
                ]
            backbone = getattr(self, model)._backbone.__name__
        else:
            if not self._model_stats()[model]["is_mm"]:
                setattr(
                    self, model, getattr(ag.learn, model)(self._data, backbone=backbone)
                )
                try:
                    estimated_batch_size = estimate_batch_size(getattr(self, model))
                except:
                    estimated_batch_size = (2, 2)
                callbacks = [
                    self._train_callback(
                        getattr(self, model).learn,
                        self._tiles_required // self._data.batch_size,
                    )
                ]
            else:
                model_with_underscore = [
                    "CascadeRCNN",
                    "CascadeRPN",
                    "EmpiricalAttention",
                    "MobileNetV2",
                    "NonLocalNet",
                    "SemFPN",
                    "DoubleHeads",
                    "DynamicRCNN",
                    "LibraRCNN",
                    "NasFcos",
                ]
                paired_model = [
                    "Cascade_RCNN",
                    "Cascade_RPN",
                    "Empirical_Attention",
                    "MobileNet_V2",
                    "NonLocal_Net",
                    "Sem_FPN",
                    "Double_Heads",
                    "Dynamic_RCNN",
                    "Libra_RCNN",
                    "Nas_Fcos",
                ]

                if model in model_with_underscore:
                    index = model_with_underscore.index(model)
                    setattr(
                        self,
                        model,
                        getattr(ag.learn, mm_model)(
                            self._data, model=paired_model[index]
                        ),
                    )
                else:
                    setattr(
                        self,
                        model,
                        getattr(ag.learn, mm_model)(self._data, model=model.lower()),
                    )
                try:
                    estimated_batch_size = estimate_batch_size(getattr(self, model))
                except:
                    estimated_batch_size = (2, 2)
                callbacks = [
                    self._train_callback(
                        getattr(self, model).learn,
                        self._tiles_required // self._data.batch_size,
                    )
                ]

        if self.verbose:
            log_msg = "{date}: {network} initialized with {bk} backbone".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"), network=model, bk=backbone
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        if self.verbose:
            log_msg = "{date}: finding desired batch size for the data object.".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
            )
            print(log_msg)
            self._logger_dict.append(log_msg)
        if estimated_batch_size[0] > 2:
            getattr(self, model)._data.batch_size = estimated_batch_size[0] // 2
        else:
            getattr(self, model)._data.batch_size = 2
        if estimated_batch_size[0] > self._tiles_required:
            getattr(self, model)._data.batch_size = (
                2 ** (int(self._tiles_required) - 1).bit_length() // 2
            )

        if self.verbose:
            log_msg = "{date}: Optimized batch size for {network} with the selected backbone is {lr}".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                network=model,
                lr=estimated_batch_size[0],
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        lr_val = getattr(self, model).lr_find(allow_plot=False)
        if self.verbose:
            log_msg = "{date}: Best learning rate for {network} with the selected data is {lr}".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"), network=model, lr=lr_val
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

            log_msg = "{date}: Fitting {network}".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"), network=model
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        try:
            _training_time_ = 0
            while model_time > _training_time_:
                # clear_output(wait=True)
                _start_time = time.time()
                with RedirectedStdout() as out:
                    getattr(self, model).fit(
                        int(epochs),
                        lr=lr_val,
                        early_stopping=True,
                        checkpoint=False,
                        callbacks=callbacks,
                    )
                    init_log = str(out)
                if "early stopping" in init_log:
                    if self.verbose:
                        log_msg = "{date}: Early stopping the {model} ".format(
                            date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                            model=model,
                        )
                        print(log_msg)
                        self._logger_dict.append(log_msg)
                    break
                _end_time = time.time()
                _training_time_ = _end_time - _start_time
                model_time -= _training_time_
                epochs = int((model_time * epochs) // _training_time_)
                if model_time > _training_time_:
                    if epochs > 200:
                        epochs == 200
                    if self.verbose:
                        log_msg = "{date}: Time left for {epochs} more epochs, training the {model} again. ".format(
                            date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                            epochs=epochs,
                            model=model,
                        )
                        print(log_msg)

                else:
                    break

            if self.verbose:
                clear_output(wait=True)
                all_logs = "\n".join(self._logger_dict)
                print(all_logs)
        except ToolIsCancelled as e:
            raise Exception("Tool is cancelled")
            exit()
        except Exception as e:
            print("Error: ", e)
            end_time = time.time()
            tot_sec = int(end_time - start_time)
            delattr(self, model)
            gc.collect()
            torch.cuda.empty_cache()

            return tot_sec

        if self.verbose:
            log_msg = "{date}: Training completed".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"), network=model
            )
            print(log_msg)
            self._logger_dict.append(log_msg)
        end_time = time.time()
        # print(metrics)
        if self.verbose:
            log_msg = "{date}: Computing the network metrices".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S")
            )
            print(log_msg)
            self._logger_dict.append(log_msg)
        metrics = getattr(self, model).learn.recorder.get_state()
        self._all_losses[model] = metrics["losses"]
        tot_sec = int(end_time - start_time)
        # self.m = metrics
        #
        t = str(datetime.timedelta(seconds=tot_sec))
        gc.collect()
        torch.cuda.empty_cache()
        train_loss = np.array(metrics["losses"])[-1]
        valid_loss = np.array(metrics["val_losses"])[-1]
        name_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        if model_type == "classification":
            accuracy = np.array(metrics["metrics"])[-1][0]
            miou = getattr(self, model).mIOU()
            miou["Model"] = str(model)
            self._mIOU_df = pd.concat(
                [
                    self._mIOU_df,
                    pd.DataFrame({key: [val] for key, val in miou.items()}),
                ]
            )

            if accuracy >= self._max_accuracy:
                self._is_best = True
                self._max_accuracy = accuracy
            dice = np.array(metrics["metrics"])[-1][1]
            df = pd.DataFrame(
                {
                    "Model": [model],
                    "train_loss": [train_loss],
                    "valid_loss": [valid_loss],
                    "accuracy": [accuracy],
                    "dice": [dice],
                    "lr": [lr_val],
                    "training time": [t],
                    "backbone": [backbone],
                    "timing": [name_time],
                    "optuna_study": [False],
                }
            )
            self._train_df = pd.concat([self._train_df, df], ignore_index=True)
        else:
            avg_precision = getattr(self, model).average_precision_score()
            avg = sum(avg_precision.values()) / len(avg_precision.values())
            avg_precision["Model"] = str(model)
            self._average_precision_score_df = pd.concat(
                [
                    self._average_precision_score_df,
                    pd.DataFrame({key: [val] for key, val in avg_precision.items()}),
                ]
            )
            if avg >= self._max_accuracy:
                self._is_best = True
                self._max_accuracy = avg
            df = pd.DataFrame(
                {
                    "Model": [model],
                    "train_loss": [train_loss],
                    "valid_loss": [valid_loss],
                    "average_precision_score": [avg],
                    "lr": [lr_val],
                    "training time": [t],
                    "backbone": [backbone],
                    "timing": [name_time],
                    "optuna_study": [False],
                }
            )
            self._train_df = pd.concat([self._train_df, df], ignore_index=True)
        if self.verbose:
            log_msg = "{date}: Finished training {network}.".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"), network=model
            )
            print(log_msg)
            self._logger_dict.append(log_msg)
            log_msg = "{date}: Exiting.".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S")
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        ## Save the model
        if self.verbose:
            log_msg = "{date}: Saving the model".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S")
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        # print(self._save_evaluated_models)
        if self._save_evaluated_models:
            if self._save_to_folder:
                getattr(self, model).save(
                    self._output_path
                    + os.sep
                    + "models"
                    + os.sep
                    + "AutoDL_"
                    + str(model)
                    + "_"
                    + backbone
                    + "_"
                    + name_time
                )
                if self.verbose:
                    log_msg = "{date}: model saved at {path}".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                        path=os.path.join(
                            self._output_path,
                            "models",
                            "AutoDL_" + str(model) + "_" + backbone + "_" + name_time,
                        ),
                    )
                    print(log_msg)
                    self._logger_dict.append(log_msg)
            else:
                getattr(self, model).save(
                    "AutoDL_" + str(model) + "_" + backbone + "_" + name_time
                )
                if self.verbose:
                    log_msg = "{date}: model saved at {path}".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                        path=os.path.join(
                            self._data.path,
                            "models",
                            "AutoDL_" + str(model) + "_" + backbone + "_" + name_time,
                        ),
                    )
                    print(log_msg)
                    self._logger_dict.append(log_msg)

        if self._is_best:
            self.best_model = model
            self._best_backbone = backbone
            setattr(self, "BestPerformingModel", getattr(self, model))

        if not self._model_stats()[model]["is_mm"]:
            setattr(
                self, model + "_backbones", getattr(self, model).supported_backbones
            )
            delattr(self, model)
            gc.collect()
            torch.cuda.empty_cache()

        else:
            delattr(self, model)
            gc.collect()
            torch.cuda.empty_cache()
            if self.verbose:
                log_msg = "{date}: deleting {network} with {bk}".format(
                    date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                    network=model,
                    bk=backbone,
                )
                print(log_msg)
                self._logger_dict.append(log_msg)
        return tot_sec

    def fit(self, **kwargs):
        """
        Train the selected networks for the specified number of epochs and using the
        specified learning rates
        """
        self._logger_dict = []
        if self._model_type == "Classified_Tiles":
            m_type = "classification"
            self._train_df = pd.DataFrame(
                columns=[
                    "Model",
                    "train_loss",
                    "valid_loss",
                    "accuracy",
                    "dice",
                    "lr",
                    "training time",
                    "backbone",
                    "timing",
                    "optuna_study",
                ]
            )
        elif self._model_type in self._all_detection_data:
            m_type = "detection"
            self._train_df = pd.DataFrame(
                columns=[
                    "Model",
                    "train_loss",
                    "valid_loss",
                    "average_precision_score",
                    "lr",
                    "training time",
                    "backbone",
                    "timing",
                    "optuna_study",
                ]
            )

        compare_time = self._time_in_sec
        if self.verbose:
            log_msg = "{date}: Selected networks: {networks}".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                networks=" ".join(self._algos),
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        for model in self._algos:
            # self._max_epochs = int(self._epoch_obj[model])
            if self.verbose:
                log_msg = "{date}: Current network - {network}. ".format(
                    date=dt.now().strftime("%d-%m-%Y %H:%M:%S"), network=model
                )
                print(log_msg)
                self._logger_dict.append(log_msg)

            model_time = self._model_stats()[model]["time"]
            model_time = (model_time * self._tiles_required) // self._max_image_set
            mt = str(datetime.timedelta(seconds=model_time))
            if self.verbose:
                log_msg = "{date}: Total time alloted to train the {network} model is {network_time}".format(
                    date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                    network=model,
                    network_time=mt,
                )
                print(log_msg)
                self._logger_dict.append(log_msg)
            if model_time > compare_time:
                epochs = int((self._max_epochs * compare_time) // model_time)
                if epochs <= 0:
                    epochs = 0
                if self.verbose:
                    log_msg = """{date}: Insufficient time to train the {network} for 20 epochs. {net_epochs} epochs can only be trained in the remaining time.""".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                        network=model,
                        net_epochs=epochs,
                    )
                    print(log_msg)
                    self._logger_dict.append(log_msg)
            else:
                epochs = self._max_epochs
                if self.verbose:
                    log_msg = """{date}: Maximum number of epochs will be {net_epochs} to train {network}""".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                        network=model,
                        net_epochs=epochs,
                    )
                    self._logger_dict.append(log_msg)
                    print(log_msg)

            if epochs <= 0:
                if self.verbose:
                    log_msg = """{date}: The time left to train the {network} is not sufficent.""".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S"), network=model
                    )
                    print(log_msg)
                    self._logger_dict.append(log_msg)
                    log_msg = """{date}: Remaining networks will be skipped due to limited time, Stopping the training process.""".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S")
                    )
                    print(log_msg)
                    self._logger_dict.append(log_msg)
                break

            # tot_sec = self._train_model(
            #     model, epochs=epochs, model_type=m_type, model_time=model_time
            # )

            self.train_basic_model = self._train_model
            tot_sec = self.train_basic_model(
                model, epochs=epochs, model_type=m_type, model_time=model_time
            )
            self.train_basic_model = None
            del self.train_basic_model
            gc.collect()
            torch.cuda.empty_cache()

            compare_time -= tot_sec
        self._dataset_type = m_type
        if m_type == "classification":
            self._train_df = self._train_df.sort_values(
                "accuracy", ascending=False
            ).reset_index(drop=True)
        if m_type == "detection":
            self._train_df = self._train_df.sort_values(
                "average_precision_score", ascending=False
            ).reset_index(drop=True)

        if self._training_mode == "advanced":
            if self.verbose:
                log_msg = """{date}: Entering into exhaustive mode.""".format(
                    date=dt.now().strftime("%d-%m-%Y %H:%M:%S")
                )
                print(log_msg)
                self._logger_dict.append(log_msg)
            compare_time = self._time_in_sec

            # Edited
            top_models = list(self._train_df.head(2)["Model"])
            all_trained_models = list(self._train_df["Model"])

            # top_models = ["SingleShotDetector", "RetinaNet"]
            # all_trained_models = ["SingleShotDetector", "RetinaNet"]

            if self.verbose:
                log_msg = (
                    """{date}: Top two performing models are - {network}""".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                        network=" ".join(top_models),
                    )
                )
                print(log_msg)
                self._logger_dict.append(log_msg)

            # edit start here
            # compare_time = 2000
            time_for_each_model = compare_time // 2
            # best_models = top_models
            self._exhaustive_mode_studies = []
            global all_val_losses, all_train_losses, dice, BestPerformingModel, best_backbone, best_model
            global self_obj
            self_obj = self
            counter = 0
            for model in all_trained_models:
                if counter >= 2:
                    break

                all_val_losses = []
                all_train_losses = []
                dice = []
                if self._model_stats()[model]["is_mm"] or model == "YOLOv3":
                    if self.verbose:
                        log_msg = """{date}: {model} does not have additional parameters to tune, skipping.""".format(
                            date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                            model=model,
                        )
                        print(log_msg)
                        self._logger_dict.append(log_msg)
                    continue
                self._execute_optuna_study = _train_exhaust_mode
                (
                    current_study,
                    self,
                    all_val_losses,
                    all_train_losses,
                    dice,
                    timing,
                ) = self._execute_optuna_study(
                    self,
                    time_for_each_model,
                    model,
                    all_val_losses,
                    all_train_losses,
                    dice,
                )
                self._execute_optuna_study = None
                del self._execute_optuna_study
                gc.collect()
                torch.cuda.empty_cache()
                df = current_study.trials_dataframe()

                current_study._timing = timing
                df["valid_loss"] = all_val_losses
                df["train_loss"] = all_train_losses
                df["timing"] = timing
                df["optuna_study"] = [True] * len(all_val_losses)
                if self._dataset_type == "classification":
                    df["dice"] = dice
                sorted_df = df.sort_values(by=["value"], ascending=False)
                # sorted_df = sorted_df.drop_duplicates('params_backbones', keep='first')
                sorted_df["duration"] = sorted_df["duration"].apply(
                    lambda x: str(x).split(" ")[-1].split(".")[0]
                )

                if self._dataset_type == "classification":
                    updated_df = {
                        "Model": list([model] * sorted_df.shape[0]),
                        "train_loss": list(sorted_df["train_loss"]),
                        "valid_loss": list(sorted_df["valid_loss"]),
                        "accuracy": list(sorted_df["value"]),
                        "dice": list(sorted_df["dice"]),
                        "lr": list(sorted_df["params_lr"]),
                        "training time": list(sorted_df["duration"]),
                        "backbone": list(sorted_df["params_backbones"]),
                        "timing": list(sorted_df["timing"]),
                        "optuna_study": list(sorted_df["optuna_study"]),
                    }
                else:
                    updated_df = {
                        "Model": list([model] * sorted_df.shape[0]),
                        "train_loss": list(sorted_df["train_loss"]),
                        "valid_loss": list(sorted_df["valid_loss"]),
                        "average_precision_score": list(sorted_df["value"]),
                        "lr": list(sorted_df["params_lr"]),
                        "training time": list(sorted_df["duration"]),
                        "backbone": list(sorted_df["params_backbones"]),
                        "timing": list(sorted_df["timing"]),
                        "optuna_study": list(sorted_df["optuna_study"]),
                    }
                df_new = pd.DataFrame(updated_df)
                self._train_df = pd.concat([self._train_df, df_new], ignore_index=True)
                self._exhaustive_mode_studies.append(current_study)
                counter += 1

                # if BestPerformingModel is not None:
                #     self.BestPerformingModel = BestPerformingModel
                #     self._best_backbone = best_backbone
                #     self.best_model = best_model
                #     self.name_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        if self.verbose:
            log_msg = """{date}: Collating and evaluating model performances.""".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S")
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        self.name_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        if not self._save_evaluated_models:
            if self._save_to_folder:
                self.BestPerformingModel.save(
                    self._output_path
                    + os.sep
                    + "models"
                    + os.sep
                    + "AutoDL_"
                    + str(self.best_model)
                    + "_"
                    + self._best_backbone
                    + "_"
                    + self.name_time
                )
                if self.verbose:
                    log_msg = "{date}: Saving best performing model at {path}".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                        path=os.path.join(
                            self._output_path,
                            "models",
                            "AutoDL_"
                            + str(self.best_model)
                            + "_"
                            + self._best_backbone
                            + "_"
                            + self.name_time,
                        ),
                    )
                    print(log_msg)
                    self._logger_dict.append(log_msg)
            else:
                self.BestPerformingModel.save(
                    "AutoDL_"
                    + str(self.best_model)
                    + "_"
                    + self._best_backbone
                    + "_"
                    + self.name_time
                )
                if self.verbose:
                    log_msg = "{date}: model saved at {path}".format(
                        date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                        path=os.path.join(
                            self._data.path,
                            "models",
                            "AutoDL_"
                            + str(model)
                            + "_"
                            + self._best_backbone
                            + "_"
                            + self.name_time,
                        ),
                    )
                    print(log_msg)
                    self._logger_dict.append(log_msg)

        if m_type == "classification":
            self._train_df = self._train_df.sort_values(
                "accuracy", ascending=False
            ).reset_index(drop=True)
        if m_type == "detection":
            self._train_df = self._train_df.sort_values(
                "average_precision_score", ascending=False
            ).reset_index(drop=True)

        if self.verbose:
            log_msg = """{date}: Exiting.""".format(
                date=dt.now().strftime("%d-%m-%Y %H:%M:%S")
            )
            print(log_msg)
            self._logger_dict.append(log_msg)

        _all_executed_model = list(self._train_df["Model"])

        if self._save_to_folder:
            self.BestPerformingModel.save(self._output_path)

        for ex_models in _all_executed_model:
            try:
                delattr(self, ex_models)
                delattr(self, ex_models + "_backbones")
            except AttributeError:
                pass

    def show_results(self, rows=5, threshold=0.25, **kwargs):
        """
        Shows sample results for the model.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional number of rows. By default, 5 rows
                                are displayed.
        =====================   ===========================================

        :return: dataframe
        """
        print(
            "show_results will only show the output from the best performing model: "
            + self.best_model
        )
        plt.show(
            getattr(self, "BestPerformingModel").show_results(
                rows=rows, thresh=threshold, **kwargs
            )
        )
        pass

    def _display_plot(self, x, y):
        plt.bar(x, y)
        plt.xticks(rotation=60)
        plt.show()
        pass

    def score(self, allow_plot=False):
        """
        returns output from AutoDL's model.score(), "average precision score" in case of detection and accuracy in case of classification.
        """
        if self._train_df is None:
            raise Exception("Train a model using fit() before getting the scores.")
        else:
            if allow_plot:
                if self._model_type == "Classified_Tiles":
                    self._display_plot(
                        self._train_df["Model"], self._train_df["accuracy"]
                    )
                else:
                    self._display_plot(
                        self._train_df["Model"],
                        self._train_df["average_precision_score"],
                    )
            df = self._train_df[list(self._train_df.keys())[:-2]]
            return df

    def report(self, allow_plot=False):
        """
        returns a HTML report of the different models trained by AutoDL along with their performance.
        """
        from IPython.display import HTML
        from ._autodl_utils import generate_output_report

        if self._train_df is None:
            raise Exception("Train a model using fit() before getting the scores.")
        else:
            generate_output_report(
                self._train_df,
                self._output_path,
                self._exhaustive_mode_studies,
                self._training_mode,
                self._save_to_folder,
                self._save_evaluated_models,
            )
            # clear_output(wait=True)
            rel_report_path = os.path.join(self._output_path, "README.html")
            return HTML(rel_report_path)

    def average_precision_score(self):
        """
        Calculates the average of the "average precision score" of all classes for selected networks
        """
        if self._model_type == "Classified_Tiles":
            print("This method is not supported with the selected model type")
        elif self._model_type in self._all_detection_data:
            if self._average_precision_score_df is None:
                print("Please train the networks first using fit()")
                return
            cols = ["Model"] + list(self._average_precision_score_df.keys())[:-1]
            return self._average_precision_score_df[cols]
        else:
            print("Datatype not supported!!")

    def mIOU(self):
        """
        Calculates the mIOU of all classes for selected networks
        """
        if self._model_type == "Classified_Tiles":
            try:
                cols = ["Model"] + list(self._mIOU_df.keys())[:-1]
                return self._mIOU_df[cols]
            except Exception as E:
                print(E)
                print("Please train the networks first using fit()")

        elif self._model_type in self._all_detection_data:
            print("This method is not supported with the selected model type")
        else:
            print("Datatype not supported!!")

    def _model_stats(self):
        """
        Shows the model stats
        """
        details = {
            "DeepLab": {
                "time": 1600,
                "is_mm": False,
                "executed": False,
                "params": {
                    "type_list": {
                        "backbones": [
                            "resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                            "densenet121",
                            "densenet169",
                            "densenet161",
                            "densenet201",
                            "vgg11",
                            "vgg11_bn",
                            "vgg13",
                            "vgg13_bn",
                            "vgg16",
                            "vgg16_bn",
                            "vgg19",
                            "vgg19_bn",
                        ],
                        "dice_loss_average": ["micro", "macro"],
                        "pointrend": [True, False],
                        "class_balancing": [True, False],
                        "mixup": [True, False],
                        "focal_loss": [True, False],
                        "keep_dilation": [True, False],
                    },
                    "type_float": {"dice_loss_fraction": (0, 1)},
                },
            },
            "UnetClassifier": {
                "time": 1600,
                "is_mm": False,
                "executed": False,
                "params": {
                    "type_list": {
                        "backbones": [
                            "resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                        ],
                        # "backend": ["pytorch", "tensorflow"],
                        "use_unet": [True, False],
                        "pointrend": [True, False],
                        "class_balancing": [True, False],
                        "mixup": [True, False],
                        "focal_loss": [True, False],
                        "keep_dilation": [True, False],
                    },
                    "type_float": {"dice_loss_fraction": (0, 1)},
                },
            },
            "PSPNetClassifier": {
                "time": 1600,
                "is_mm": False,
                "executed": False,
                "params": {
                    "type_list": {
                        "backbones": [
                            "resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                            "densenet121",
                            "densenet169",
                            "densenet161",
                            "densenet201",
                            "vgg11",
                            "vgg11_bn",
                            "vgg13",
                            "vgg13_bn",
                            "vgg16",
                            "vgg16_bn",
                            "vgg19",
                            "vgg19_bn",
                        ],
                        "dice_loss_average": ["micro", "macro"],
                        "use_unet": [True, False],
                        "pointrend": [True, False],
                        "class_balancing": [True, False],
                        "mixup": [True, False],
                        "focal_loss": [True, False],
                        "keep_dilation": [True, False],
                    },
                    "type_float": {"dice_loss_fraction": (0, 1)},
                },
            },
            "ANN": {
                "time": 1600,
                "is_mm": True,
            },
            "APCNet": {
                "time": 1600,
                "is_mm": True,
            },
            "CCNet": {
                "time": 3500,
                "is_mm": True,
            },
            "CGNet": {
                "time": 700,
                "is_mm": True,
            },
            "HRNet": {
                "time": 4200,
                "is_mm": True,
            },
            "DeepLabV3Plus": {
                "time": 4200,
                "is_mm": True,
            },
            "DMNet": {
                "time": 4200,
                "is_mm": True,
            },
            "DNLNet": {
                "time": 4200,
                "is_mm": True,
            },
            "EMANet": {
                "time": 4200,
                "is_mm": True,
            },
            "FastSCNN": {
                "time": 4200,
                "is_mm": True,
            },
            "FCN": {
                "time": 4200,
                "is_mm": True,
            },
            "GCNet": {
                "time": 4200,
                "is_mm": True,
            },
            "MobileNetV2": {
                "time": 4200,
                "is_mm": True,
            },
            "NonLocalNet": {
                "time": 4200,
                "is_mm": True,
            },
            "OCRNet": {
                "time": 4200,
                "is_mm": True,
            },
            "PSANet": {
                "time": 4200,
                "is_mm": True,
            },
            "SemFPN": {
                "time": 4200,
                "is_mm": True,
            },
            "UperNet": {
                "time": 4200,
                "is_mm": True,
            },
            "SingleShotDetector": {
                "time": 1600,
                "is_mm": False,
                "executed": False,
                "params": {
                    "type_list": {
                        "backbones": [
                            "resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                            "densenet121",
                            "densenet169",
                            "densenet161",
                            "densenet201",
                            "vgg11",
                            "vgg11_bn",
                            "vgg13",
                            "vgg13_bn",
                            "vgg16",
                            "vgg16_bn",
                            "vgg19",
                            "vgg19_bn",
                            "mobilenet_v2",
                        ],
                        "bias": [True, False],
                        "mixup": [True, False],
                    },
                    "type_float": {
                        "dropout": (0.2, 0.8),
                        "location_loss_factor": (0, 1),
                    },
                },
            },
            "MaskRCNN": {
                "time": 1600,
                "is_mm": False,
                "executed": False,
                "params": {
                    "type_list": {
                        "backbones": [
                            "resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                        ],
                        "pointrend": [True, False],
                    },
                    "type_int": {
                        "rpn_pre_nms_top_n_train": (1000, 5000),
                        "rpn_pre_nms_top_n_test": (500, 1000),
                    },
                },
            },
            "RetinaNet": {
                "time": 3000,
                "is_mm": False,
                "executed": False,
                "params": {
                    "type_list": {
                        "backbones": [
                            "resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                        ]
                    }
                },
            },
            "DETReg": {
                "time": 1600,
                "is_mm": False,
                "executed": False,
                "params": {
                    "type_list": {
                        "backbones": [
                            "resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                        ]
                    }
                },
            },
            "FasterRCNN": {
                "time": 3000,
                "is_mm": False,
                "executed": False,
                "params": {
                    "type_list": {
                        "backbones": [
                            "resnet18",
                            "resnet34",
                            "resnet50",
                            "resnet101",
                            "resnet152",
                        ]
                    },
                    "type_int": {
                        "rpn_pre_nms_top_n_train": (1000, 5000),
                        "rpn_pre_nms_top_n_test": (500, 1000),
                    },
                },
            },
            "YOLOv3": {"time": 6500, "is_mm": False},
            "ATSS": {
                "time": 1650,
                "is_mm": True,
            },
            "CARAFE": {
                "time": 3500,
                "is_mm": True,
            },
            "CascadeRCNN": {
                "time": 700,
                "is_mm": True,
            },
            "CascadeRPN": {
                "time": 4200,
                "is_mm": True,
            },
            "DCN": {
                "time": 4200,
                "is_mm": True,
            },
            "Detectors": {
                "time": 4200,
                "is_mm": True,
            },
            "DoubleHeads": {
                "time": 4200,
                "is_mm": True,
            },
            "DynamicRCNN": {
                "time": 4200,
                "is_mm": True,
            },
            "EmpiricalAttention": {
                "time": 4200,
                "is_mm": True,
            },
            "FCOS": {
                "time": 4200,
                "is_mm": True,
            },
            "FoveaBox": {
                "time": 4200,
                "is_mm": True,
            },
            "FSAF": {
                "time": 4200,
                "is_mm": True,
            },
            "GHM": {
                "time": 4200,
                "is_mm": True,
            },
            "LibraRCNN": {
                "time": 4200,
                "is_mm": True,
            },
            "PaFPN": {
                "time": 4200,
                "is_mm": True,
            },
            "PISA": {
                "time": 4200,
                "is_mm": True,
            },
            "RegNet": {
                "time": 4200,
                "is_mm": True,
            },
            "RepPoints": {
                "time": 4200,
                "is_mm": True,
            },
            "Res2Net": {
                "time": 4200,
                "is_mm": True,
            },
            "SABL": {
                "time": 4200,
                "is_mm": True,
            },
            "VFNet": {
                "time": 4200,
                "is_mm": True,
            },
        }
        return details

    def _sort_algos(self, algorithms):
        """
        Sorts algorithms in a particular order
        """
        sorted_algos = []
        algos = self._all_algorithms
        for algo in algos:
            if algo in algorithms:
                if algo not in sorted_algos:
                    sorted_algos.append(algo)
        return list(sorted_algos)

    def supported_classification_models(self):
        """
        Supported classification models.
        """
        return [
            "DeepLab",
            "UnetClassifier",
            "PSPNetClassifier",
            "ANN",
            "APCNet",
            "CCNet",
            "CGNet",
            "HRNet",
            "DeepLabV3Plus",
            "DMNet",
            "DNLNet",
            "EMANet",
            "FastSCNN",
            "FCN",
            "GCNet",
            "MobileNetV2",
            "NonLocalNet",
            "OCRNet",
            "PSANet",
            "SemFPN",
            "UperNet",
        ]

    def supported_detection_models(self):
        """
        Supported detection models.
        """
        return [
            "SingleShotDetector",
            "RetinaNet",
            "FasterRCNN",
            "YOLOv3",
            "DETReg",
            "ATSS",
            "CARAFE",
            "CascadeRCNN",
            "CascadeRPN",
            "DCN",
            "Detectors",
            "DoubleHeads",
            "DynamicRCNN",
            "EmpiricalAttention",
            "FCOS",
            "FoveaBox",
            "FSAF",
            "GHM",
            "LibraRCNN",
            "PaFPN",
            "PISA",
            "RegNet",
            "RepPoints",
            "Res2Net",
            "SABL",
            "VFNet",
            "MaskRCNN",
        ]

    def lr_find(self):
        print("lr_find() is not supported in AutoDL")
