import os
import tempfile
import random
import traceback
import json
import warnings
import math
from pathlib import Path

import_exception = None

import arcgis
from arcgis.features import FeatureLayer

try:
    from ._arcgis_model import ArcGISModel, _raise_fastai_import_error
    from fastai.tabular import (
        tabular_learner,
        TabularDataBunch,
        TabularList,
        TabularModel,
    )
    from fastai.tabular.transform import FillMissing, Categorify, Normalize
    from fastai.basic_train import Learner, load_learner
    from fastprogress.fastprogress import progress_bar
    from .._utils.tabular_data import TabularDataObject, explain_prediction, add_h3
    from .._utils.common import _get_emd_path
    from fastai.torch_core import split_model_idx
    import torch
    from fastai.metrics import r2_score

    HAS_FASTAI = True

except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

try:
    import numpy as np

    HAS_NUMPY = True
except:
    HAS_NUMPY = False


def _get_learner_object(data, layers, emb_szs, ps, emb_drop, pretrained_path):
    if pretrained_path:
        learn = load_learner(
            os.path.dirname(pretrained_path),
            os.path.basename(pretrained_path).rsplit(".", 1)[0] + "_exported.pth",
        )
        learn.path = data.path
        if not data._is_empty:
            learn.data = data._databunch
    else:
        databunch = data._databunch
        if not emb_szs or isinstance(emb_szs, dict):
            emb_szs = databunch.get_emb_szs({} if not emb_szs else emb_szs)

        model = TabularModel(
            emb_szs,
            len(databunch.cont_names),
            out_sz=databunch.c,
            layers=layers,
            ps=ps,
            emb_drop=emb_drop,
            y_range=None,
            use_bn=False,
        )
        learn = Learner(databunch, model, path=data.path)

    return learn


class FullyConnectedNetwork(ArcGISModel):
    """
    Creates a :class:`~arcgis.learn.FullyConnectedNetwork` Object.
    Based on the Fast.ai's Tabular Learner

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required TabularDataObject. Returned data object from
                            :class:`~arcgis.learn.prepare_tabulardata` function.
    ---------------------   -------------------------------------------
    layers                  Optional list, specifying the number of nodes in each layer.
                            Default: [500, 100] is used.
                            2 layers each with nodes 500 and 100 respectively.
    ---------------------   -------------------------------------------
    emb_szs                 Optional dict, variable name with embedding size
                            for categorical variables.
                            If not specified, then calculated using fastai.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.FullyConnectedNetwork` Object
    """

    def __init__(self, data, layers=None, emb_szs=None, **kwargs):
        if data._is_unsupervised:
            raise Exception("Cannot train on unsupervised data")

        super().__init__(data, None)

        if isinstance(self._data._dependent_variable, list):
            self._data._dependent_variable = self._data._dependent_variable[0]

        self._backbone = None
        if layers is None:
            layers = [500, 100]

        ps = kwargs.get("ps", None)
        emb_drop = kwargs.get("emb_drop", 0)

        self.learn = _get_learner_object(
            data, layers, emb_szs, ps, emb_drop, kwargs.get("pretrained_path", None)
        )
        self._layers = layers
        self.learn.model = self.learn.model.to(self._device)
        idx = 1
        self.learn.layer_groups = split_model_idx(self.learn.model, [idx])
        self.learn.create_opt(lr=1e-03)

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
        Creates a :class:`~arcgis.learn.FullyConnectedNetwork` Object from an Esri Model Definition (EMD) file.

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

        :return: :class:`~arcgis.learn.FullyConnectedNetwork` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        dependent_variable = emd["dependent_variable"]
        categorical_variables = emd["categorical_variables"]
        continuous_variables = emd["continuous_variables"]
        _is_classification = False
        if emd["_is_classification"] == "classification":
            _is_classification = True

        layers = emd["layers"]
        cell_sizes = emd.get("cell_sizes", None)
        if data is None:
            data = TabularDataObject._empty(
                categorical_variables, continuous_variables, dependent_variable, None
            )
            data._is_classification = _is_classification
            data._cell_sizes = cell_sizes
            class_object = cls(data, pretrained_path=emd_path)
            class_object._data.emd = emd
            class_object._data.emd_path = emd_path
            return class_object

        return cls(data, layers=layers, pretrained_path=str(emd_path))

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

        self.learn.export(os.path.join(path, os.path.basename(path) + "_exported.pth"))
        # from IPython.utils import io

        # with io.capture_output() as captured:
        super().save(
            path, framework, publish, gis, save_optimizer=save_optimizer, **kwargs
        )
        # print(captured.stdout)
        return Path(path)

    @property
    def _model_metrics(self):
        from IPython.utils import io

        with io.capture_output() as captured:
            score = self.score()

        return {"score": score}

    @property
    def feature_importances_(self):
        """
        :return: the global feature importance summary plot from SHAP.Feature is temporarily disabled.
        """
        warnings.warn(
            "Feature importance for Fully Connected Network is currently disabled due to package incompatibility and is under review"
        )
        # processed_dataframe = None
        # explain_index = None
        # random_index = None
        # explain_prediction(
        #    self,
        #    processed_dataframe,
        #    index=explain_index,
        #    random_index=random_index,
        #    predictor=None,
        #    global_pred=True,
        # )
        return

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["ModelType"] = "FullyConnectedNetwork"
        _emd_template["layers"] = self._layers
        _emd_template["dependent_variable"] = self._data._dependent_variable
        _emd_template["categorical_variables"] = self._data._categorical_variables
        _emd_template["continuous_variables"] = self._data._continuous_variables
        _emd_template["_is_classification"] = (
            "classification" if self._data._is_classification else "regression"
        )
        _emd_template["cell_sizes"] = self._data._cell_sizes

        return _emd_template

    def _predict(self, data_frame_row):
        return self.learn.predict(data_frame_row)

    def _df_predict(self, dataframe):
        # from fastai.data_block import split_kwargs_by_func, grab_idx, DatasetType
        # ds_type = DatasetType.Valid
        # ds = self.learn.dl(ds_type).dataset
        #
        # current_databunch = self.learn.data
        # databunch_half, databunch_second_half = self._data._prepare_validation_databunch(dataframe)
        # return dataframe, databunch_half, databunch_second_half
        # try:
        #     self.learn.data = databunch
        #     preds = self.learn.get_preds(ds_type)[0]
        # except Exception as e:
        #     raise e
        # finally:
        #     self.learn.data = current_databunch
        #
        # analyze_kwargs, kwargs = split_kwargs_by_func({}, ds.y.analyze_pred)
        # preds = [ds.y.analyze_pred(grab_idx(preds, i), **analyze_kwargs) for i in range(len(preds))]
        # preds = [ds.y.reconstruct(z) for z in preds]
        if not HAS_NUMPY:
            raise Exception("This function requires numpy.")

        preds = []

        for i in progress_bar(range(len(dataframe))):
            prediction = self._predict(dataframe.iloc[i])[0].obj
            if isinstance(prediction, (list, np.ndarray)):
                prediction = prediction[0]
            if isinstance(prediction, np.float32):
                prediction = prediction.astype(np.float64)
            preds.append(prediction)

        return preds

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
        explain=False,
        explain_index=None,
    ):
        """

        Predict on data from feature layer, dataframe and or raster data.

        =================================   =========================================================================
        **Parameter**                        **Description**
        ---------------------------------   -------------------------------------------------------------------------
        input_features                      Optional :class:`~arcgis.features.FeatureLayer` or spatially enabled dataframe.
                                            Required if prediction_type='features'.
                                            Contains features with location and
                                            some or all fields required to infer the dependent variable value.
        ---------------------------------   -------------------------------------------------------------------------
        explanatory_rasters                 Optional list of Raster Objects.
                                            If prediction_type='raster', must contain all rasters
                                            required to make predictions.
        ---------------------------------   -------------------------------------------------------------------------
        datefield                           Optional string. Field name from feature layer
                                            that contains the date, time for the input features.
                                            Same as :meth:`~arcgis.learn.prepare_tabulardata` .
        ---------------------------------   -------------------------------------------------------------------------
        distance_features                   Optional List of :class:`~arcgis.features.FeatureLayer` objects.
                                            These layers are used for calculation of field "NEAR_DIST_1",
                                            "NEAR_DIST_2" etc in the output dataframe.
                                            These fields contain the nearest feature distance
                                            from the input_features.
                                            Same as :meth:`~arcgis.learn.prepare_tabulardata` .
        ---------------------------------   -------------------------------------------------------------------------
        output_layer_name                   Optional string. Used for publishing the output layer.
        ---------------------------------   -------------------------------------------------------------------------
        gis                                 Optional :class:`~arcgis.gis.GIS`  Object. Used for publishing the item.
                                            If not specified then active gis user is taken.
        ---------------------------------   -------------------------------------------------------------------------
        prediction_type                     Optional String.
                                            Set 'features' or 'dataframe' to make output feature layer predictions.
                                            With this feature_layer argument is required.

                                            Set 'raster', to make prediction raster.
                                            With this rasters must be specified.
        ---------------------------------   -------------------------------------------------------------------------
        output_raster_path                  Optional path.
                                            Required when prediction_type='raster', saves
                                            the output raster to this path.
        ---------------------------------   -------------------------------------------------------------------------
        match_field_names                   Optional dictionary.
                                            Specify mapping of field names from prediction set
                                            to training set.
                                            For example:

                                                | {
                                                |    "Field_Name_1": "Field_1",
                                                |    "Field_Name_2": "Field_2"
                                                | }
        ---------------------------------   -------------------------------------------------------------------------
        explain                             Optional Bool.
                                            Setting this parameter to true generates prediction explaination plot.
                                            Plot is generated using model interpretability library called SHAP.
                                            (https://github.com/slundberg/shap). Feature is temporarily disabled.
        ---------------------------------   -------------------------------------------------------------------------
        explain_index                       Optional Int.
                                            The index of the dataframe passed to the predict function for which model
                                            interpretability is desired. If the parameter is not passed and if the
                                            explain parameter is set to true, the SHAP plot will be generated for a
                                            random index of the dataframe.
        =================================   =========================================================================

        :return: Feature Layer if prediction_type='features', dataframe for prediction_type='dataframe' else creates an output raster.

        """

        rasters = explanatory_rasters if explanatory_rasters else []
        if explain:
            # try:
            #     import shap
            # except:
            #     warnings.warn(
            #         "Prediction cannot be explained as SHAP is not installed. Please install SHAP to get explainability working."
            #     )
            #     explain = False
            #     explain_index = None
            warnings.warn(
                "Model explainability feature for Fully Connected Network is currently disabled due to package incompatibility and is under review"
            )
            explain = False
            explain_index = None
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
                prediction_type,
                explain,
                explain_index,
            )
        else:
            if not rasters:
                raise Exception("Rasters required for predict_features=False")

            if not output_raster_path:
                raise Exception(
                    "Please specify output_raster_folder_path to save the output."
                )

            return self._predict_rasters(
                output_raster_path, rasters, match_field_names, explain, explain_index
            )

    def _predict_features(
        self,
        input_features,
        rasters=None,
        datefield=None,
        distance_feature_layers=None,
        output_name="Prediction Layer",
        gis=None,
        match_field_names=None,
        prediction_type="features",
        explain=False,
        explain_index=None,
    ):
        cell_sizes = self._data._cell_sizes
        if isinstance(input_features, FeatureLayer):
            if cell_sizes and not rasters:
                dataframe = input_features.query(out_sr=4326).sdf
                dataframe = add_h3(dataframe, cell_sizes)
            else:
                dataframe = input_features.query().sdf
        else:
            dataframe = input_features.copy()

        fields_needed = (
            self._data._categorical_variables + self._data._continuous_variables
        )
        distance_feature_layers = (
            distance_feature_layers if distance_feature_layers else []
        )

        continuous_variables = self._data._continuous_variables

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
                input_features,
                self._data._dependent_variable,
                feature_layer_columns,
                raster_columns,
                datefield,
                cell_sizes,
                distance_feature_layers,
            )

        if match_field_names:
            processed_dataframe.rename(columns=match_field_names, inplace=True)

        for field in fields_needed:
            if field not in processed_dataframe.columns:
                raise Exception(f"Field missing {field}")
        if explain:
            if explain_index is None:
                random_index = True
            else:
                random_index = False
            explain_prediction(
                self,
                processed_dataframe,
                index=explain_index,
                random_index=random_index,
                predictor=None,
            )

        dataframe["prediction_results"] = self._df_predict(processed_dataframe.copy())
        if prediction_type == "dataframe":
            return dataframe

        if "SHAPE" in list(dataframe.columns):
            return dataframe.spatial.to_featurelayer(output_name, gis)
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                table_file = os.path.join(tmpdir, output_name + ".xlsx")
                dataframe.to_excel(table_file, index=False, header=True)
                online_table = gis.content.add(
                    {"type": "Microsoft Excel", "overwrite": True}, table_file
                )
                return online_table.publish(overwrite=True)

    def _predict_rasters(
        self,
        output_folder_path,
        rasters,
        match_field_names=None,
        explain=False,
        explain_index=None,
    ):
        if not os.path.exists(os.path.dirname(output_folder_path)):
            raise Exception("Output directory doesn't exist")

        if os.path.exists(output_folder_path):
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
            self._data._continuous_variables + self._data._categorical_variables
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

        processed_data = []

        length_values = len(raster_data[list(raster_data.keys())[0]])
        for i in range(length_values):
            processed_row = []
            for raster_name in sorted(raster_data.keys()):
                processed_row.append(raster_data[raster_name][i])
            processed_data.append(processed_row)
        if explain:
            if explain_index is None:
                random_index = True
            else:
                random_index = False
            explain_prediction(
                self,
                pd.DataFrame(
                    data=np.array(processed_data), columns=sorted(raster_data)
                ),
                index=explain_index,
                random_index=random_index,
                predictor=None,
            )

        processed_numpy = np.array(
            self._df_predict(
                pd.DataFrame(data=np.array(processed_data), columns=sorted(raster_data))
            ),
            dtype="float64",
        )
        processed_numpy = processed_numpy.reshape([max_raster_rows, max_raster_columns])
        processed_raster = arcpy.NumPyArrayToRaster(
            processed_numpy,
            arcpy.Point(xmin, ymin),
            x_cell_size=min_cell_size_x,
            y_cell_size=min_cell_size_y,
        )
        processed_raster.save(output_folder_path)

        return True

    def show_results(self, rows=5):
        """
        Prints the rows of the dataframe with target and prediction columns.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional Integer.
                                Number of rows to print.
        =====================   ===========================================

        :return: dataframe
        """
        try:
            import pandas as pd
        except:
            raise Exception("This function requires pandas.")
        self._check_requisites()
        min_size = len(self._data._validation_indexes)
        if min_size > rows:
            min_size = rows

        sample_indexes = random.sample(self._data._validation_indexes, min_size)
        if self._data._is_classification:
            rows_df = self._data._dataframe.loc[sample_indexes]
        else:
            rows_df = self._data._dataframe.iloc[sample_indexes]
        predictions = self._df_predict(rows_df)
        pd.options.mode.chained_assignment = None
        rows_df["prediction_results"] = predictions

        return rows_df

    def score(self):
        """
        :return: R2 score for regression model and Accuracy for classification model.
        """

        self._check_requisites()
        if not HAS_NUMPY:
            raise Exception("This function requires numpy.")

        # using loc instead of iloc to get data when dataframe doesnt have continuous indexes
        if self._data._is_classification:
            validation_dataframe = self._data._dataframe.loc[
                self._data._validation_indexes
            ].reset_index(drop=True)
        else:
            validation_dataframe = self._data._dataframe.iloc[
                self._data._validation_indexes
            ].reset_index(drop=True)

        predictions = np.array(self._df_predict(validation_dataframe))
        labels = validation_dataframe[self._data._dependent_variable]

        if self._data._is_classification:
            return (np.array(predictions) == labels).mean()
        else:
            return float(
                r2_score(
                    torch.tensor(np.array(predictions, dtype="float64")),
                    torch.tensor(np.array(labels, dtype="float64")),
                )
            )
