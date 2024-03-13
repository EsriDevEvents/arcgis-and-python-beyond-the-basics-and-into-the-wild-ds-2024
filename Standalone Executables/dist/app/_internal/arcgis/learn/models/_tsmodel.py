import os
import sys
import tempfile
import traceback
import json
import warnings
import pickle
import math
from pathlib import Path

HAS_FASTAI = True
import_exception = None

import arcgis
from arcgis.features import FeatureLayer

try:
    from ._arcgis_model import ArcGISModel, _raise_fastai_import_error
    from fastai.basic_train import Learner, load_learner
    from fastprogress.fastprogress import progress_bar
    from .._utils.tabular_data import TabularDataObject
    from fastai.torch_core import split_model_idx
    import torch
    from fastai.metrics import r2_score
    from sklearn.preprocessing import LabelEncoder
    from ._tsmodel_archs._InceptionTime import _TSInceptionTime
    from ._tsmodel_archs._Resnet import _TSResNet
    from ._tsmodel_archs._ResCNN import _TSResCNN
    from ._tsmodel_archs._FCN import _TSFCN
    from ._tsmodel_archs._LSTM import _TSLSTM
    from .._utils.TSData import To3dTensor, ToTensor
    from .._utils.common import _get_emd_path
    from ._tsmodel_archs._TST import TST

    _model_arch = {
        "inceptiontime": _TSInceptionTime,
        "resnet": _TSResNet,
        "rescnn": _TSResCNN,
        "fcn": _TSFCN,
        "lstm": _TSLSTM,
        "timeseriestransformer": TST,
    }

except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False
    _model_arch = {}

HAS_NUMPY = True
try:
    import numpy as np
except:
    HAS_NUMPY = False

_PROTOCOL_LEVEL = 2

HAS_PANDAS = True
try:
    import pandas as pd
except:
    HAS_PANDAS = False


def _get_model_from_path(pretrained_path):
    learn = load_learner(
        os.path.dirname(pretrained_path),
        os.path.basename(pretrained_path).rsplit(".", 1)[0] + "_exported.pth",
        no_check=True,
    )

    return learn


class TimeSeriesModel(ArcGISModel):
    """
    Creates a :class:`~arcgis.learn.TimeSeriesModel` Object.
    Based on the Fast.ai's https://github.com/timeseriesAI/timeseriesAI

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required TabularDataObject. Returned data object from
                            :class:`~arcgis.learn.prepare_tabulardata` function.
    ---------------------   -------------------------------------------
    seq_len                 Required Integer. Sequence Length for the series.
                            In case of raster only, seq_len = number of rasters,
                            any other passed value will be ignored.
    ---------------------   -------------------------------------------
    model_arch              Optional string. Model Architecture.
                            Allowed "InceptionTime", "ResCNN",
                            "Resnet", "FCN", "TimeSeriesTransformer", "LSTM". "LSTM"
                            supports both "LSTM" and "Bi-LSTM". "Bi-LSTM" is enabled by passing
                            `bidirectional=True` in kwargs.
    ---------------------   -------------------------------------------
    location_var            Optional string. Location variable in case of
                            NetCDF dataset.
    ---------------------   -------------------------------------------
    multistep               Optional string. It will set the model to generate
                            more than one time-step as output in multivariate scenario.
                            Compared to current auto-regressive fashion, it will generate
                            multi-step output in single pass.
                            This option is only  applicable in multivariate
                            scenario. Univariate implementation will ignore this flag.
                            Default value is `False`
    ---------------------   -------------------------------------------
    ``**kwargs``            Optional kwargs.
    ---------------------   -------------------------------------------
    =====================   ===========================================

    :return: :class:`~arcgis.learn.TimeSeriesModel` Object
    """

    def __init__(
        self,
        data,
        seq_len,
        model_arch="InceptionTime",
        location_var=None,
        multistep=False,
        **kwargs,
    ):
        data_bunch = None
        if not data._is_empty:
            data_bunch = data._time_series_bunch(
                seq_len, location_var, multistep=multistep
            )

        super().__init__(data, None)
        self.multistep = multistep
        # We need fix the number of steps based on whether it is univariate or multivariate. We are not supporting
        # multi-step training for univariate case.
        fields_needed = (
            data._categorical_variables
            + data._continuous_variables
            + data._dependent_variable
        )
        self.step = 1
        if self.multistep and len(list(fields_needed)) != 1:
            self.step = seq_len // 2

        if not data_bunch:
            self.learn = _get_model_from_path(kwargs.get("pretrained_path"))
        elif kwargs.get("pretrained_path"):
            self.learn = _get_model_from_path(kwargs.get("pretrained_path"))
            self.learn.data = data_bunch
        else:
            if not _model_arch.get(model_arch.lower()):
                raise Exception("Invalid model architecture")

            model_arch_ob = _model_arch.get(model_arch.lower())
            if model_arch.lower() == "lstm":
                model = model_arch_ob(
                    data_bunch.features, data_bunch.c, self._device, **kwargs
                ).to(self._device)
            elif model_arch.lower() == "timeseriestransformer":
                model = model_arch_ob(
                    data_bunch.features, data_bunch.c, seq_len, **kwargs
                ).to(self._device)
            else:
                if model_arch.lower() in ["resnet", "fcn"]:
                    kwargs["device"] = self._device
                model = model_arch_ob(data_bunch.features, data_bunch.c, **kwargs).to(
                    self._device
                )
                if model_arch.lower() in ["resnet", "fcn"]:
                    del kwargs["device"]
            self.learn = Learner(data_bunch, model, path=data.path)
            self.learn.data = data_bunch

        self.learn.layer_groups = split_model_idx(self.learn.model, [1])
        self._model_arch = model_arch.lower()
        if kwargs.get("pretrained_path"):
            del kwargs["pretrained_path"]
        self._kwargs = kwargs
        self._seq_len = seq_len

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s>" % (type(self).__name__)

    @staticmethod
    def _available_metrics():
        return ["valid_loss"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a :class:`~arcgis.learn.TimeSeriesModel` Object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :class:`~arcgis.learn.prepare_tabulardata` function or None for
                                inferencing.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.TimeSeriesModel` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        dependent_variable = emd["dependent_variable"]
        # added reverse support to the EMD params.
        if isinstance(dependent_variable, str):
            dependent_variable = [dependent_variable]
        categorical_variables = emd["categorical_variables"]
        continuous_variables = emd["continuous_variables"]

        _is_classification = False
        if emd["_is_classification"] == "classification":
            _is_classification = True

        model_params = emd["model_params"]
        model_arch = emd["model_arch"]
        seq_len = emd["seq_len"]
        index_field = emd.get("index_field", None)
        test_size = emd.get("test_size", None)
        step = emd.get("step", 1)
        multistep = emd.get("multistep", False)
        # encoder_path = os.path.join(os.path.dirname(emd_path),
        #                             os.path.basename(emd_path).split('.')[0] + '_encoders.pkl')

        scaler_path = os.path.join(
            os.path.dirname(emd_path),
            os.path.basename(emd_path).split(".")[0] + "_scaler.pkl",
        )

        encoder_mapping = None
        scaler = None

        # if os.path.exists(encoder_path):
        #     with open(encoder_path, 'rb') as f:
        #         encoder_mapping = pickle.loads(f.read())

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.loads(f.read())

        if data is None:
            data = TabularDataObject._empty(
                categorical_variables,
                continuous_variables,
                dependent_variable,
                encoder_mapping,
            )
            data._is_classification = _is_classification
            data._column_transforms_mapping = scaler

            # if index_field is not None:
            data._index_field = index_field
            data._test_size = test_size

            class_object = cls(
                data,
                seq_len,
                model_arch=model_arch,
                pretrained_path=emd_path,
                step=step,
                multistep=multistep,
                **model_params,
            )
            class_object._data.emd = emd
            class_object._data.emd_path = emd_path
            return class_object

        return cls(
            data,
            seq_len,
            model_arch=model_arch,
            pretrained_path=emd_path,
            step=step,
            multistep=multistep,
            **model_params,
        )

    def save(
        self,
        name_or_path,
        framework="PyTorch",
        publish=False,
        gis=None,
        save_optimizer=False,
        **kwargs,
    ):
        """
        Saves the model weights, creates an Esri Model Definition and Deep
        Learning Package zip for deployment to Image Server or ArcGIS Pro.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Folder path to save the model.
        ---------------------   -------------------------------------------
        framework               Optional string. Defines the framework of the
                                model. (Only supported by :class:`~arcgis.learn.SingleShotDetector`, currently.)
                                If framework used is ``TF-ONNX``, ``batch_size`` can be
                                passed as an optional keyword argument.

                                Framework choice: 'PyTorch' and 'TF-ONNX'
        ---------------------   -------------------------------------------
        publish                 Optional boolean. Publishes the DLPK as an item.
        ---------------------   -------------------------------------------
        gis                     Optional :class:`~arcgis.gis.GIS`  Object. Used for publishing the item.
                                If not specified then active gis user is taken.
        ---------------------   -------------------------------------------
        save_optimizer          Optional boolean. Used for saving the model-optimizer
                                state along with the model. Default is set to False
        ---------------------   -------------------------------------------
        kwargs                  Optional Parameters:
                                Boolean `overwrite` if True, it will overwrite
                                the item on ArcGIS Online/Enterprise, default False.
        =====================   ===========================================
        """

        if "\\" in name_or_path or "/" in name_or_path:
            path = os.path.abspath(name_or_path)
        else:
            path = os.path.join(self._data.path, "models", name_or_path)
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))

        if not os.path.exists(path):
            os.mkdir(path)

        base_file_name = os.path.basename(path)

        TimeSeriesModel._save_encoders(
            self._data._column_transforms_mapping, path, base_file_name
        )
        self.learn.export(os.path.join(path, os.path.basename(path) + "_exported.pth"))
        from IPython.utils import io

        with io.capture_output() as captured:
            saved_path = super().save(
                path, framework, False, gis, save_optimizer=save_optimizer, **kwargs
            )
        if publish:
            file_name = os.path.basename(saved_path) + ".dlpk"
            dlpk_path = Path(os.path.join(saved_path, file_name))
            self._publish_dlpk(
                dlpk_path,
                gis=gis,
                overwrite=kwargs.get("overwrite", False),
            )

        return Path(path)

    @staticmethod
    def _save_encoders(scaler, path, base_file_name):
        # if not encoder_mapping:
        #     return

        # encoder_file = os.path.join(path, base_file_name + '_encoders.pkl')
        # with open(encoder_file, 'wb') as f:
        #     f.write(pickle.dumps(encoder_mapping, protocol=_PROTOCOL_LEVEL))

        scaler_file = os.path.join(path, base_file_name + "_scaler.pkl")
        with open(scaler_file, "wb") as f:
            f.write(pickle.dumps(scaler, protocol=_PROTOCOL_LEVEL))

    @property
    def _model_metrics(self):
        # from IPython.utils import io
        # with io.capture_output() as captured:
        #     score = self.score()

        return {"score": self.score()}

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["ModelType"] = "TimeSeriesModel"
        _emd_template["model_arch"] = self._model_arch
        _emd_template["model_params"] = self._kwargs
        _emd_template["seq_len"] = self._seq_len
        # Added this for backward compatibility.
        # Added the logic to enable compatibility with current ArcgisPro
        if len(self._data._dependent_variable) == 1:
            _emd_template["dependent_variable"] = self._data._dependent_variable[0]
        else:
            _emd_template["dependent_variable"] = self._data._dependent_variable
        _emd_template["categorical_variables"] = self._data._categorical_variables
        _emd_template["continuous_variables"] = self._data._continuous_variables
        _emd_template["multistep"] = self.multistep
        _emd_template["step"] = self.step

        if self._data._index_field:
            _emd_template["index_field"] = self._data._index_field

        _emd_template["_is_classification"] = (
            "classification" if self._data._is_classification else "regression"
        )
        if hasattr(self._data, "_test_size"):
            _emd_template["test_size"] = self._data._test_size

        return _emd_template

    def _predict(self, orig_sequence):
        sequence = np.array(orig_sequence, dtype="float32")
        if sequence.shape[0] == 1:
            seq_arr = To3dTensor(orig_sequence).to(self._device)
        else:
            seq_arr = ToTensor(
                np.expand_dims(np.array(sequence, dtype="float32"), axis=0)
            ).to(self._device)
        model = self.learn.model
        model.eval()

        with torch.no_grad():
            output = model(seq_arr).cpu().numpy()

        return output

    def predict(
        self,
        input_features=None,
        explanatory_rasters=None,
        datefield=None,
        distance_features=None,
        output_layer_name="Prediction Layer",
        gis=None,
        prediction_type="features",
        output_raster_path=None,
        match_field_names=None,
        number_of_predictions=None,
    ):
        """

        Predict on data from feature layer and or raster data.

        =================================   =========================================================================
        **Parameter**                        **Description**
        ---------------------------------   -------------------------------------------------------------------------
        input_features                      Optional :class:`~arcgis.features.FeatureLayer` or spatially enabled dataframe.
                                            Contains features with location of the input data.
                                            Required if prediction_type is 'features' or 'dataframe'
        ---------------------------------   -------------------------------------------------------------------------
        explanatory_rasters                 Optional list of Raster Objects.
                                            Required if prediction_type is 'rasters'
        ---------------------------------   -------------------------------------------------------------------------
        datefield                           Optional field_name.
                                            This field contains the date in the input_features.
                                            The field type can be a string or date time field.
                                            If specified, the field will be split into
                                            Year, month, week, day, dayofweek, dayofyear,
                                            is_month_end, is_month_start, is_quarter_end,
                                            is_quarter_start, is_year_end, is_year_start,
                                            hour, minute, second, elapsed and these will be added
                                            to the prepared data as columns.
                                            All fields other than elapsed and dayofyear are treated
                                            as categorical.
        ---------------------------------   -------------------------------------------------------------------------
        distance_features                   Optional List of :class:`~arcgis.features.FeatureLayer` objects.
                                            These layers are used for calculation of field "NEAR_DIST_1",
                                            "NEAR_DIST_2" etc in the output dataframe.
                                            These fields contain the nearest feature distance
                                            from the input_features.
                                            Same as :meth:`~arcgis.learn.prepare_tabulardata`.
        ---------------------------------   -------------------------------------------------------------------------
        output_layer_name                   Optional string. Used for publishing the output layer.
        ---------------------------------   -------------------------------------------------------------------------
        gis                                 Optional :class:`~arcgis.gis.GIS`  Object. Used for publishing the item.
                                            If not specified then active gis user is taken.
        ---------------------------------   -------------------------------------------------------------------------
        prediction_type                     Optional String.
                                            Set 'features' or 'dataframe' to make output predictions.
        ---------------------------------   -------------------------------------------------------------------------
        output_raster_path                  Optional path. Required when prediction_type='raster', saves
                                            the output raster to this path.
        ---------------------------------   -------------------------------------------------------------------------
        match_field_names                   Optional string.
                                            Specify mapping of the original training set with prediction set.
        ---------------------------------   -------------------------------------------------------------------------
        number_of_predictions               Optional int for univariate time series.
                                            Specify the number of predictions to make, adds new rows to the dataframe.
                                            For multivariate or if None, it expects the dataframe to have empty rows.
                                            if multi-step is set to True during training then it does not need empty
                                            rows. If multi-step is set to False then dataframe needs to have rows with
                                            `NA` values in `variable predict` and non-NA values in `explnatory_varibles`
                                            For prediction_type='raster', a new raster is created.
        =================================   =========================================================================

        :return: :class:`~arcgis.features.FeatureLayer`/dataframe if prediction_type='features'/'dataframe', else returns True and saves output
        raster at the specified path.
        """

        rasters = explanatory_rasters if explanatory_rasters else []
        if prediction_type in ["features", "dataframe"]:
            if input_features is None:
                raise Exception("Feature Layer required for predict_features=True")

            gis = gis if gis else arcgis.env.active_gis
            return self._predict_features(
                input_features,
                rasters,
                datefield,
                distance_features,
                output_layer_name,
                gis,
                match_field_names,
                number_of_predictions,
                prediction_type,
            )
        else:
            if not rasters:
                raise Exception("Rasters required for predict_features=False")

            if not output_raster_path:
                raise Exception(
                    "Please specify output_raster_folder_path to save the output."
                )

            return self._predict_rasters(output_raster_path, rasters, match_field_names)

    def _predict_rasters(self, output_raster_path, rasters, match_field_names=None):
        if not os.path.exists(os.path.dirname(output_raster_path)):
            raise Exception("Output directory doesn't exist")

        if os.path.exists(output_raster_path):
            raise Exception("Output Folder already exists")

        try:
            import arcpy
        except:
            raise Exception("This function requires arcpy.")

        if not HAS_NUMPY:
            raise Exception("This function requires numpy.")

        try:
            import pandas as pd
        except:
            raise Exception("This function requires pandas.")

        fields_needed = (
            self._data._categorical_variables + self._data._continuous_variables
        )

        try:
            arcpy.env.outputCoordinateSystem = rasters[0].extent["spatialReference"][
                "wkt"
            ]
        except:
            arcpy.env.outputCoordinateSystem = rasters[0].extent["spatialReference"][
                "wkid"
            ]

        xmin = rasters[0].extent["xmin"]
        xmax = rasters[0].extent["xmax"]
        ymin = rasters[0].extent["ymin"]
        ymax = rasters[0].extent["ymax"]
        min_cell_size_x = rasters[0].mean_cell_width
        min_cell_size_y = rasters[0].mean_cell_height

        default_sr = rasters[0].extent["spatialReference"]

        for raster in rasters:
            point_upper = arcgis.geometry.Point(
                {
                    "x": raster.extent["xmin"],
                    "y": raster.extent["ymax"],
                    "sr": raster.extent["spatialReference"],
                }
            )
            point_lower = arcgis.geometry.Point(
                {
                    "x": raster.extent["xmax"],
                    "y": raster.extent["ymin"],
                    "sr": raster.extent["spatialReference"],
                }
            )
            cell_size = arcgis.geometry.Point(
                {
                    "x": raster.mean_cell_width,
                    "y": raster.mean_cell_height,
                    "sr": raster.extent["spatialReference"],
                }
            )

            points = arcgis.geometry.project(
                [point_upper, point_lower, cell_size],
                raster.extent["spatialReference"],
                default_sr,
            )
            point_upper = points[0]
            point_lower = points[1]
            cell_size = points[2]

            if xmin > point_upper.x:
                xmin = point_upper.x
            if ymax < point_upper.y:
                ymax = point_upper.y
            if xmax < point_lower.x:
                xmax = point_lower.x
            if ymin > point_lower.y:
                ymin = point_lower.y

            if min_cell_size_x > cell_size.x:
                min_cell_size_x = cell_size.x

            if min_cell_size_y > cell_size.y:
                min_cell_size_y = cell_size.y

        max_raster_columns = int(abs(math.ceil((xmax - xmin) / min_cell_size_x)))
        max_raster_rows = int(abs(math.ceil((ymax - ymin) / min_cell_size_y)))

        point_upper = arcgis.geometry.Point({"x": xmin, "y": ymax, "sr": default_sr})
        cell_size = arcgis.geometry.Point(
            {"x": min_cell_size_x, "y": min_cell_size_y, "sr": default_sr}
        )

        raster_data = {}
        for raster in rasters:
            field_name = raster.name
            point_upper_translated = arcgis.geometry.project(
                [point_upper], default_sr, raster.extent["spatialReference"]
            )[0]
            cell_size_translated = arcgis.geometry.project(
                [cell_size], default_sr, raster.extent["spatialReference"]
            )[0]
            if field_name in fields_needed:
                raster_read = raster.read(
                    origin_coordinate=(
                        point_upper_translated.x,
                        point_upper_translated.y,
                    ),
                    ncols=max_raster_columns,
                    nrows=max_raster_rows,
                    cell_size=(cell_size_translated.x, cell_size_translated.y),
                )
                for row in range(max_raster_rows):
                    for column in range(max_raster_columns):
                        values = raster_read[row][column]
                        index = 0
                        for value in values:
                            key = field_name
                            if index != 0:
                                key = key + f"_{index}"
                            if not raster_data.get(key):
                                raster_data[key] = []
                            index = index + 1
                            raster_data[key].append(value)
            elif match_field_names and match_field_names.get(raster.name):
                field_name = match_field_names.get(raster.name)
                raster_read = raster.read(
                    origin_coordinate=(
                        point_upper_translated.x,
                        point_upper_translated.y,
                    ),
                    ncols=max_raster_columns,
                    nrows=max_raster_rows,
                    cell_size=(cell_size_translated.x, cell_size_translated.y),
                )
                for row in range(max_raster_rows):
                    for column in range(max_raster_columns):
                        values = raster_read[row][column]
                        index = 0
                        for value in values:
                            key = field_name
                            if index != 0:
                                key = key + f"_{index}"
                            if not raster_data.get(key):
                                raster_data[key] = []

                            index = index + 1
                            raster_data[key].append(value)
            else:
                continue

        for field in fields_needed:
            if (
                field not in list(raster_data.keys())
                and match_field_names
                and match_field_names.get(field, None) is None
            ):
                raise Exception(f"Field missing {field}")

        length_values = len(raster_data[list(raster_data.keys())[0]])
        processed_output = []
        for i in progress_bar(range(length_values)):
            processed_row = []
            for raster_name in sorted(raster_data.keys()):
                processed_row.append(raster_data[raster_name][i])
            processed_output.append(self._predict([processed_row]))

        processed_numpy = np.array(processed_output, dtype="float64")
        processed_numpy = processed_numpy.reshape([max_raster_rows, max_raster_columns])
        processed_raster = arcpy.NumPyArrayToRaster(
            processed_numpy,
            arcpy.Point(xmin, ymin),
            x_cell_size=min_cell_size_x,
            y_cell_size=min_cell_size_y,
        )
        processed_raster.save(output_raster_path)

        return True

    def _predict_features(
        self,
        input_features,
        rasters=None,
        datefield=None,
        distance_features=None,
        output_layer_name="Prediction Layer",
        gis=None,
        match_field_names=None,
        number_of_predictions=None,
        prediction_type="features",
    ):
        if not HAS_PANDAS:
            raise Exception("This function requires pandas library")

        if isinstance(input_features, FeatureLayer):
            orig_dataframe = input_features.query().sdf
        else:
            orig_dataframe = input_features.copy()

        # Dtype conversion because native pandas format will break the plotting libraries
        for i in orig_dataframe.columns:
            if isinstance(orig_dataframe.loc[:, i].dtype, pd.Float64Dtype):
                orig_dataframe.loc[:, i] = orig_dataframe.loc[:, i].astype(np.float64)

        if match_field_names is None:
            match_field_names = {}
        fields_needed = (
            self._data._categorical_variables
            + self._data._continuous_variables
            + self._data._dependent_variable
        )

        (
            orig_dataframe,
            single_swap_pred,
            number_of_predictions,
        ) = self._infer_number_of_pred(
            orig_dataframe, number_of_predictions, match_field_names, fields_needed
        )

        dataframe = orig_dataframe.copy()
        distance_feature_layers = distance_features if distance_features else []

        continuous_variables = (
            self._data._continuous_variables + self._data._dependent_variable
        )

        feature_layer_columns = []
        for column in dataframe.columns:
            column_name = column
            categorical = False
            if column_name in fields_needed:
                if column_name not in continuous_variables:
                    categorical = True
            elif match_field_names and match_field_names.get(column_name):
                if match_field_names.get(column_name) not in continuous_variables:
                    categorical = True
            else:
                continue

            feature_layer_columns.append((column_name, categorical))

        raster_columns = []
        if rasters:
            for raster in rasters:
                column_name = raster.name
                categorical = False
                if column_name in fields_needed:
                    if column_name not in continuous_variables:
                        categorical = True
                elif match_field_names and match_field_names.get(column_name):
                    column_name = match_field_names.get(column_name)
                    if column_name not in continuous_variables:
                        categorical = True
                else:
                    continue

                raster_columns.append((raster, categorical))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            (
                processed_dataframe,
                fields_mapping,
            ) = TabularDataObject._prepare_dataframe_from_features(
                orig_dataframe,
                self._data._dependent_variable,
                feature_layer_columns,
                raster_columns,
                datefield,
                distance_feature_layers,
            )

        if match_field_names:
            processed_dataframe.rename(columns=match_field_names, inplace=True)

        for field in fields_needed:
            if field not in processed_dataframe.columns:
                raise Exception(f"Field missing {field}")

        for column in processed_dataframe.columns:
            if column not in fields_needed:
                processed_dataframe = processed_dataframe.drop(column, axis=1)

        # preserve the ordering of the column, It is required because sorting tends to change the order and ultimately
        # affects different transforms.

        order_columns = (
            self._data._dependent_variable
            + self._data._continuous_variables
            + self._data._categorical_variables
        )

        processed_dataframe = processed_dataframe.loc[:, order_columns]

        # processed_dataframe = processed_dataframe.reindex(
        #     sorted(processed_dataframe.columns), axis=1
        # )

        index = self._seq_len
        processed_dataframe[self._data._dependent_variable] = processed_dataframe[
            self._data._dependent_variable
        ].replace(r"^\s*$", np.nan, regex=True)

        processed_dataframe_transform = processed_dataframe.copy()

        processed_dataframe_transform = self._apply_transform(
            processed_dataframe, processed_dataframe_transform
        )

        big_bunch = []
        prediction_sequence_list = []
        processed_dataframe_transform = processed_dataframe_transform.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for col in range(len(processed_dataframe.columns.values)):
                if (
                    list(processed_dataframe.columns.values)[col]
                    in self._data._dependent_variable
                ):
                    prediction_sequence_list.append(
                        processed_dataframe_transform[:, col]
                    )
                    ind = len(prediction_sequence_list) - 1
                    big_bunch.append(prediction_sequence_list[ind][0 : self._seq_len])
                else:
                    big_bunch.append(
                        processed_dataframe_transform[:, col][0 : self._seq_len]
                    )
        # checking the first array because the changes would force this to be of type List[List]
        prediction_sequence_list = np.stack(prediction_sequence_list, axis=1)
        if prediction_sequence_list.shape[0] < self._seq_len:
            raise Exception("Basic Sequence not found!")
        # modified below column so that multivariates can be captured
        while index < len(prediction_sequence_list):
            if pd.isna(prediction_sequence_list[index]).any() or any(
                [
                    True
                    if i
                    in [
                        "",
                        None,
                        "null",
                        "None",
                    ]
                    else False
                    for i in prediction_sequence_list[index]
                ]
            ):
                value = self._predict(np.array(big_bunch))
                # Now the predicted value will have the shape of 1 * [self.step * len(variable_predict)
                if not single_swap_pred:
                    prediction_sequence_list[index] = value.reshape(self.step, -1).mean(
                        axis=0
                    )
                else:
                    prediction_sequence_list[index:] = value.reshape(self.step, -1)

            index = index + 1
            big_bunch = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for col in range(len(processed_dataframe.columns.values)):
                    if (
                        list(processed_dataframe.columns.values)[col]
                        in self._data._dependent_variable
                    ):
                        idx = self._data._dependent_variable.index(
                            list(processed_dataframe.columns.values)[col]
                        )
                        big_bunch.append(
                            prediction_sequence_list[:, idx][
                                index
                                - self._seq_len : (
                                    index - self._seq_len + self._seq_len
                                )
                            ]
                        )
                    else:
                        big_bunch.append(
                            processed_dataframe_transform[:, col][
                                index
                                - self._seq_len : (
                                    index - self._seq_len + self._seq_len
                                )
                            ]
                        )
        transformed_results = prediction_sequence_list
        transformed_results_col = self._apply_inverse_transform(transformed_results)

        for idx, col in enumerate(self._data._dependent_variable):
            orig_dataframe[col + "_results"] = transformed_results_col[:, idx]

        if prediction_type == "dataframe":
            return orig_dataframe

        if "SHAPE" in list(orig_dataframe.columns):
            orig_dataframe.spatial.to_featurelayer(output_layer_name, gis)
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                table_file = os.path.join(tmpdir, output_layer_name + ".xlsx")
                orig_dataframe.to_excel(table_file, index=False, header=True)
                online_table = gis.content.add(
                    {"type": "Microsoft Excel", "overwrite": True}, table_file
                )
                return online_table.publish(overwrite=True)

    def _apply_inverse_transform(self, transformed_results):
        # Apply inverse transform to generate the main data
        transformed_results_ret = []
        for idx, col in enumerate(self._data._dependent_variable):
            if self._data._column_transforms_mapping.get(col):
                for transform in self._data._column_transforms_mapping.get(col):
                    transformed_results_col = transform.inverse_transform(
                        np.array(transformed_results[:, idx]).reshape(-1, 1)
                    )
                    transformed_results_col = transformed_results_col.squeeze(1)
                transformed_results_ret.append(transformed_results_col)
        return np.stack(transformed_results_ret, axis=1)

    def _add_predict_rows(
        self, number_of_predictions, orig_dataframe, match_field_names, fields_needed
    ):
        # Changed to make code future ready as the previous method of adding
        # pandas series will be deprecated.
        #
        from pandas.api.types import is_datetime64_any_dtype as is_datetime

        datetime_dict = {}
        delta = None
        index_field_name = None
        end_value = None
        if self._data._index_field is not None:
            index_field_name = match_field_names.get(
                self._data._index_field, self._data._index_field
            )
            if index_field_name in list(orig_dataframe.columns):
                if is_datetime(orig_dataframe[index_field_name]):
                    delta = (
                        orig_dataframe[index_field_name].iloc[1]
                        - orig_dataframe[index_field_name].iloc[0]
                    )
                    end_value = None
                    if delta is not None:
                        end_value = orig_dataframe[index_field_name].iloc[
                            len(orig_dataframe) - 1
                        ]
                else:
                    orig_dataframe[index_field_name], sample = self._convert_datetime(
                        orig_dataframe[index_field_name]
                    )
                    if not sample:
                        delta = (
                            orig_dataframe[index_field_name].iloc[1]
                            - orig_dataframe[index_field_name].iloc[0]
                        )
                        end_value = None
                        if delta is not None:
                            end_value = orig_dataframe[index_field_name].iloc[
                                len(orig_dataframe) - 1
                            ]

        for i in orig_dataframe.columns:
            if i != index_field_name:
                if is_datetime(orig_dataframe[i]):
                    new_delta = orig_dataframe[i].iloc[1] - orig_dataframe[i].iloc[0]
                    end_value_temp = None
                    if new_delta is not None:
                        end_value_temp = orig_dataframe[i].iloc[len(orig_dataframe) - 1]
                    if new_delta is not None:
                        datetime_dict[i] = tuple([new_delta, end_value_temp])
                else:
                    if orig_dataframe[i].dtype == "object" and i in fields_needed:
                        # check whether it is time
                        orig_dataframe[i], sample = self._convert_datetime(
                            orig_dataframe[i]
                        )
                        if not sample:
                            new_delta = (
                                orig_dataframe[i].iloc[1] - orig_dataframe[i].iloc[0]
                            )
                            end_value_temp = None
                            if new_delta is not None:
                                end_value_temp = orig_dataframe[i].iloc[
                                    len(orig_dataframe) - 1
                                ]
                            if new_delta is not None:
                                datetime_dict[i] = tuple([new_delta, end_value_temp])

        pred_temp_df = pd.DataFrame(
            np.full([number_of_predictions, orig_dataframe.shape[1]], np.NAN)
        )
        pred_temp_df.columns = orig_dataframe.columns
        # preserve the indexes. Need to adjust 1 because new index will start from 0
        pred_temp_df.index += orig_dataframe.index[-1] + 1
        orig_dataframe = pd.concat([orig_dataframe, pred_temp_df])
        if delta is not None:
            tindex = pd.period_range(
                end_value, freq=delta, periods=number_of_predictions + 1
            )
            orig_dataframe.loc[
                orig_dataframe.tail(number_of_predictions).index, index_field_name
            ] = tindex[1:]
        if len(datetime_dict):
            for key, value in datetime_dict.items():
                new_delta, end_value_temp = value
                tindex = pd.period_range(
                    end_value_temp, freq=new_delta, periods=number_of_predictions + 1
                )
                orig_dataframe.loc[
                    orig_dataframe.tail(number_of_predictions).index, key
                ] = tindex[1:]

        return orig_dataframe

    def _infer_number_of_pred(
        self, orig_dataframe, number_of_predictions, match_field_names, fields_needed
    ):
        # Type of inference
        #     ├── Multivariate
        #     │   ├── MultiStep
        #     │   │   ├── NA rec in df>0 then num_of_pred= Number of NA rec/ otherwise warning
        #     │   │   └── Number_of_pred=0 or None and NA rec in df is 0 then seq_len//2
        #     │   └── Number of NA records in df will be honored
        #     │       └── Number_of_pred>0 then infer using NA in df warning
        #     └── Univariate
        #         └── Number_of_pred is honored
        ###

        single_swap_pred = False
        if self.multistep:
            df_na_count = len(orig_dataframe[orig_dataframe.isna().any(axis=1)])

            if df_na_count == 0:
                if not (number_of_predictions is None or number_of_predictions == 0):
                    warnings.warn(
                        f"Number of predictions is supplied in multivariate scenario. Overriding the value\
                                  with step value {self.step} "
                    )
                number_of_predictions = self.step
                single_swap_pred = True
                add_na_rec = True
            else:
                if not (number_of_predictions is None or number_of_predictions == 0):
                    warnings.warn(
                        f"Both Number of predictions and dataframe with NA is supplied.\
                                  Using dataframe NA count as number of predictions {df_na_count}"
                    )
                number_of_predictions = df_na_count
                add_na_rec = False
                warnings.warn(
                    "The model is trained with multistep objective. Setting NA values will result "
                    "in Auto-regressive mode of inference. This may lead to low accuracy."
                )

        else:
            if len(self._data._dependent_variable):
                number_of_predictions = number_of_predictions
                add_na_rec = True
            else:
                # multivariate case
                number_of_predictions = len(
                    orig_dataframe[orig_dataframe.isna().any(axis=1)]
                )
                add_na_rec = False

        if number_of_predictions is None:
            number_of_predictions = 0

        if add_na_rec:
            orig_dataframe = self._add_predict_rows(
                orig_dataframe=orig_dataframe,
                match_field_names=match_field_names,
                number_of_predictions=number_of_predictions,
                fields_needed=fields_needed,
            )
        return orig_dataframe, single_swap_pred, number_of_predictions

    def _apply_transform(self, processed_dataframe, processed_dataframe_transform):
        # changed to function because the transformation needs to handle the categorical variable case.
        # In such a scenario, the NaN is treated as label and label encoder fails because train data did not
        # have any NA label while the test data has because of nature of prediction.
        for col in list(processed_dataframe.columns):
            if col in self._data._categorical_variables:
                transformed_data = processed_dataframe[col].dropna()
            else:
                transformed_data = processed_dataframe[col]
            for transform in self._data._column_transforms_mapping.get(col, []):
                if isinstance(transform, LabelEncoder):
                    transformed_data = transform.transform(
                        np.array(
                            transformed_data,
                            dtype=type(processed_dataframe[col][0]),
                        )
                    )
                    transformed_data = transformed_data.reshape(-1, 1)
                else:
                    transformed_data = transform.transform(
                        np.array(
                            transformed_data,
                            dtype=type(processed_dataframe[col][0]),
                        ).reshape(-1, 1)
                    )

                transformed_data = transformed_data.squeeze(1)
            processed_dataframe_transform[col].head(len(transformed_data)).loc[
                :
            ] = np.array(transformed_data, dtype=type(processed_dataframe[col][0]))
        return processed_dataframe_transform

    def score(self):
        """
        :return: R2 score for regression model and Accuracy for classification model.
        """

        self._check_requisites()
        if not HAS_NUMPY:
            raise Exception("This function requires numpy.")

        model = self.learn.model

        model.eval()

        dl = self.learn.data.valid_dl

        targets = []
        predictions = []
        for i in range(len(dl.x.items)):
            prediction = self._predict(dl.x.items[i])[0]
            target = dl.y.items[i]
            # targets.append(target)
            if isinstance(target, (list, np.ndarray)):
                targets.append(target)
            else:
                targets.append([target])
            predictions.append(prediction)

        targets = np.array(targets, dtype="float64")
        predictions = np.array(predictions, dtype="float64")

        transformed_results = np.stack(targets, axis=0)
        transformed_results = self._apply_inverse_transform(transformed_results)
        targets_inversed = transformed_results

        transformed_results = np.stack(predictions, axis=0)

        transformed_results = self._apply_inverse_transform(transformed_results)
        predictions_inversed = transformed_results

        if self._data._is_classification:
            return (np.array(predictions) == np.array(targets)).mean()
        else:
            targets = torch.tensor(targets_inversed).to(self._device)
            predictions = torch.tensor(predictions_inversed).to(self._device)
            return float(r2_score(predictions, targets))

    def _safe_div(self, arr):
        len_arr = len(arr) - 1
        if len_arr % 10 == 0:
            return np.linspace(0, len_arr, 10).astype(int)
        elif len_arr < 10:
            return np.linspace(0, len_arr, len_arr + 1).astype(int)
        else:
            return np.linspace(0, len_arr, 6).astype(int)

    def _convert_datetime(self, index_data_copy):
        sample_ticks = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if not pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                index_data_copy
            ):
                try:
                    index_data_copy = pd.to_datetime(index_data_copy)
                except:
                    sample_ticks = True
        return index_data_copy, sample_ticks

    def show_results(self, rows=5):
        """
        Prints the graph with predictions.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional Integer.
                                Number of rows to print.
        =====================   ===========================================
        """
        model = self.learn.model

        model.eval()

        dl = self.learn.data.valid_dl

        targets = []
        predictions = []
        sequence = []
        for i in range(len(dl.x.items)):
            prediction = self._predict(dl.x.items[i])[0]
            target = dl.y.items[i]
            if isinstance(target, (list, np.ndarray)):
                targets.append(target)
            else:
                targets.append([target])
            predictions.append(prediction)
            sequence.append(dl.x.items[i])

        targets = np.array(targets, dtype="float64")
        predictions = np.array(predictions, dtype="float64")

        transformed_results = np.stack(targets, axis=0)
        transformed_results = self._apply_inverse_transform(transformed_results)
        targets_inversed = transformed_results

        transformed_results = np.stack(predictions, axis=0)
        transformed_results = self._apply_inverse_transform(transformed_results)
        predictions_inversed = transformed_results
        column_transforms_mapping = self._data._column_transforms_mapping.copy()
        # del column_transforms_mapping[self._data._dependent_variable]
        sequence_inversed = []
        keys_bk = list(column_transforms_mapping.keys())
        order_columns = (
            self._data._dependent_variable
            + self._data._continuous_variables
            + self._data._categorical_variables
        )
        keys = []
        for key in order_columns:
            if key in keys_bk:
                keys.append(key)

        for seq in sequence:
            seq_inverse = []
            index = 0
            for col_data in seq:
                transformed_data = col_data
                if len(keys) > index:
                    for transform in column_transforms_mapping.get(keys[index]):
                        if isinstance(transform, LabelEncoder):
                            transformed_data = transform.inverse_transform(
                                np.array(transformed_data, dtype=int)
                            )
                        # else: # Commenting it out. Because inverse transform tends to change the scale
                        #     transformed_data = transform.inverse_transform(
                        #         np.array(transformed_data).reshape(-1, 1)
                        #     )
                        #     transformed_data = transformed_data.squeeze(1)

                seq_inverse.append(transformed_data)
                index = index + 1

            sequence_inversed.append(seq_inverse)

        if self._data._index_seq is not None:
            validation_index_seq = self._data._index_seq.take(
                self._data._validation_indexes_ts, axis=0
            )
        else:
            validation_index_seq = None

        import matplotlib.pyplot as plt

        n_items = rows
        if n_items > len(targets_inversed):
            n_items = len(targets_inversed)

        rows = int(n_items)

        fig, axs = plt.subplots(rows, 2, figsize=(10, 10))
        fig.suptitle("Ground truth vs Predictions\n\n", fontsize=16)

        for i in range(rows):
            for idx, seq_plot in enumerate(sequence_inversed[i]):
                sample_ticks = False
                if self._data._index_seq is not None:
                    index_data, sample_ticks = self._convert_datetime(
                        validation_index_seq[i]
                    )
                    axs[i, 0].plot(index_data, seq_plot)
                    axs[i, 1].plot(index_data, seq_plot)
                else:
                    axs[i, 0].plot(seq_plot)
                    axs[i, 1].plot(seq_plot)

                if sample_ticks:
                    axs[i, 0].xaxis.set_major_locator(
                        plt.FixedLocator(
                            self._safe_div(range(len(validation_index_seq[i])))
                        )
                    )
                    axs[i, 1].xaxis.set_major_locator(
                        plt.FixedLocator(
                            self._safe_div(range(len(validation_index_seq[i])))
                        )
                    )
                axs[i, 0].tick_params(axis="x", labelrotation=60)
                axs[i, 1].tick_params(axis="x", labelrotation=60)
                axs[i, 0].set_title(
                    ",".join([f"{val :4f}" for val in targets_inversed[i]])
                )
                axs[i, 1].set_title(
                    ",".join([f"{val :4f}" for val in predictions_inversed[i]])
                )

        plt.tight_layout()
        plt.show()
