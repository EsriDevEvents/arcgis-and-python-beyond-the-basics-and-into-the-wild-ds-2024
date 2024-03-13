from ._machine_learning import MLModel, raise_data_exception
import os
import shutil
import tempfile
import random
import json
import pickle
import warnings
import math
import shutil
import os
import time
from pathlib import Path
import traceback
import arcgis
from arcgis.features import FeatureLayer
from ._codetemplate import ml_raster_prf

HAS_AUTO_ML_DEPS = True
import_exception = None

try:
    from ._arcgis_model import ArcGISModel, _raise_fastai_import_error
    from arcgis.learn._fairlearn._fairlearn import calculate_metrics
    from arcgis.learn._utils.tabular_data import (
        TabularDataObject,
        add_h3,
        _extract_embeddings,
    )
    from arcgis.learn._utils.common import _get_emd_path
    from arcgis.learn._utils.utils import arcpy_localization_helper
    import pickle
    from sklearn.preprocessing import normalize

    HAS_FASTAI = True
except:
    import_exception = traceback.format_exc()
    HAS_FASTAI = False

try:
    import sklearn
    from sklearn import *
    from sklearn import preprocessing
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        Normalizer,
        LabelEncoder,
        MinMaxScaler,
        StandardScaler,
        OrdinalEncoder,
    )
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_AUTO_ML_DEPS = False

HAS_FAST_PROGRESS = True
try:
    from fastprogress.fastprogress import progress_bar
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FAST_PROGRESS = False

_PROTOCOL_LEVEL = 2


class AutoML(object):
    """
    Automates the process of model selection, training and hyperparameter tuning of
    machine learning models within a specified time limit. Based upon
    MLJar(https://github.com/mljar/mljar-supervised/) and scikit-learn.

    Note that automated machine learning support is provided only for supervised learning.
    Refer https://supervised.mljar.com/

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required TabularDataObject. Returned data object from
                            :meth:`~arcgis.learn.prepare_tabulardata` function.
    ---------------------   -------------------------------------------
    total_time_limit        Optional Int. The total time limit in seconds for
                            AutoML training.
                            Default is 3600 (1 Hr)
    ---------------------   -------------------------------------------
    mode                    Optional Str.
                            Can be {Basic, Intermediate, Advanced}. This parameter defines
                            the goal of AutoML and how intensive the AutoML search will be.

                            Basic : To to be used when the user wants to explain and understand the data. Uses 75%/25% train/test split. Uses the following models: Baseline, Linear, Decision Tree, Random Trees, XGBoost, Neural Network, and Ensemble. Has full explanations in reports: learning curves, importance  plots, and SHAP plots.
                            Intermediate : To be used when the user wants to train a model that will be used in real-life use cases. Uses 5-fold CV (Cross-Validation). Uses the following models: Linear, Random Trees, LightGBM, XGBoost, CatBoost, Neural Network, and Ensemble. Has learning curves and importance plots in reports.

                            Advanced : To be used for machine learning competitions (maximum performance). Uses 10-fold CV (Cross-Validation). Uses the following models: Decision Tree, Random Trees, Extra Trees, XGBoost, CatBoost, Neural Network, Nearest Neighbors, Ensemble, and Stacking.It has only learning curves in the reports. Default is Basic
    ---------------------   -------------------------------------------
    algorithms              Optional. List of str.
                            The list of algorithms that will be used in the training. The algorithms can be:
                            Linear, Decision Tree, Random Trees, Extra Trees, LightGBM, Xgboost, Neural Network
    ---------------------   -------------------------------------------
    eval_metric             Optional  Str. The metric to be used to compare models.
                            Possible values are:
                            For binary classification - logloss (default), auc, f1, average_precision,
                            accuracy.
                            For multiclass classification - logloss (default), f1, accuracy
                            For regression - rmse (default), mse, mae, r2, mape, spearman, pearson

                            Note - If there are only 2 unique values in the target, then
                            binary classification is performed,
                            If number of unique values in the target is between 2 and 20 (included), then
                            multiclass classification is performed,
                            In all other cases, regression is performed on the dataset.
    ---------------------   -------------------------------------------
    n_jobs                  Optional. Int.
                            Number of CPU cores to be used. By default, it is set to 1.Set it
                            to -1 to use all the cores.
    =====================   ===========================================

    **kwargs**

    =======================   ===========================================
    sensitive_variables       Optional. List of strings.
                              Variables in the feature class/dataframe which are sensitive and prone to model bias.
                              Ex - ['sex','race'] or ['nationality']
    -----------------------   -------------------------------------------
    fairness_metric           Optional. String.
                              Name of fairness metric based on which fairness optimization should be done on the evaluated models.
                              Available metrics for binary classification are 'demographic_parity_difference' , 'demographic_parity_ratio',
                              'equalized_odds_difference', 'equalized_odds_ratio'.
                              'demographic_parity_ratio' is the default.
                              Available metrics for regression are 'group_loss_ratio' (Default) and 'group_loss_difference'.
    -----------------------   -------------------------------------------
    fairness_threshold        Optional. Float.
                              Required when the chosen metric is group_loss_difference
                              The threshold value for fairness metric. Default values are as follows:
                              - for `demographic_parity_difference` the metric value should be below 0.25,
                              - for `demographic_parity_ratio` the metric value should be above 0.8,
                              - for `equalized_odds_difference` the metric value should be below 0.25,
                              - for `equalized_odds_ratio` the metric value should be above 0.8.
                              - for `group_loss_ratio` the metric value should be above 0.8.
                              - for `group_loss_difference` the metric value should be below 0.25,
    -----------------------   -------------------------------------------
    privileged_groups         Optional. List.
                              List of previleged groups in the sensitive attribute.
                              For example, in binary classification task, a privileged group is the one with the highest selection rate.
                              Example value: [{"sex": "Male"}]
    -----------------------   -------------------------------------------
    underprivileged_groups    Optional. List.
                              List of underprivileged groups in the sensitive attribute.
                              For example, in binary classification task, an underprivileged group
                              is the one with the lowest selection rate.
                              Example value: [{"sex": "Female"}]
    =======================   ===========================================

    :return: :class:`~arcgis.learn.AutoML` Object
    """

    def __init__(
        self,
        data=None,
        total_time_limit=3600,
        mode="Basic",
        algorithms=None,
        eval_metric="auto",
        n_jobs=1,
        ml_task="auto",
        **kwargs,
    ):
        try:
            import platform

            if platform.system() == "Linux":
                message = """
                        Please enable tensorflow by setting the required environment variable 'ARCGIS_ENABLE_TF_BACKEND' to '1' before importing arcgis
                        \n for example the following code block needs to be executed before importing arcgis
                        \n\n`import os; os.environ['ARCGIS_ENABLE_TF_BACKEND'] = '1'`
                        """
                print(message)
            from supervised.automl import AutoML as base_AutoML
        except Exception as e:
            import_exception = "\n".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            _raise_fastai_import_error(import_exception=import_exception)

        if not HAS_AUTO_ML_DEPS:
            _raise_fastai_import_error(import_exception=import_exception)

        self._data = data
        if isinstance(self._data._dependent_variable, list):
            self._data._dependent_variable = self._data._dependent_variable[0]

        if getattr(self._data, "_is_unsupervised", False):
            raise Exception(
                "Auto ML feature is currently only available for Supervised learning."
            )
        if getattr(self._data, "_is_not_empty", False):
            if (len(data._training_indexes) < 20) & (
                eval_metric in ["r2", "rmse", "mse", "mape", "spearman", "pearson"]
            ):
                warnings.warn(
                    "The eval metric you have passed, is not valid for a classification usecase. If the use case is regression, then ensure that your dataset has atleast 22 records"
                )
                return

        self._code = ml_raster_prf
        if algorithms:
            algorithms = algorithms
        else:
            algorithms = [
                "Linear",
                "Decision Tree",
                "Random Trees",
                "Extra Trees",
                "LightGBM",
                "Xgboost",
                "Neural Network",
            ]
        try:
            algorithms = [
                "Random Trees" if x == "Random Forest" else x for x in algorithms
            ]  # To handle backward compatibility with RF
        except:
            pass

        if getattr(self._data, "_is_not_empty", True):
            (
                self._training_data,
                self._training_labels,
                self._validation_data,
                self._validation_labels,
            ) = self._data._ml_data

            self._all_data_df = self._data._dataframe[
                self._data._continuous_variables
                + self._data._categorical_variables
                + self._data._embedding_variables
            ]
            self._all_data_df = self._impute_missing_values(data=self._all_data_df)
            self._all_labels = self._data._dataframe[
                self._data._dependent_variable
            ]  # .values
            self._validation_data_df = pd.DataFrame(
                self._validation_data,
                columns=self._data._continuous_variables
                + self._data._categorical_variables
                + self._data._embedding_variables,
            )
            if ml_task == "auto":
                ml_task = self.get_ml_task(self._all_labels)
            if ml_task == "text":
                msg = arcpy_localization_helper(
                    "Dependent variable has more than 200 unique values more than half of the total records are unique, hence there is not enough information to train a model",
                    260154,
                    "ERROR",
                )
                exit(260146)
            if (mode == "Explain") or (mode == "Basic"):
                explain_level = 2
                zone_list = ["zone3_id", "zone4_id", "zone5_id", "zone6_id", "zone7_id"]
                # try:
                #    self._all_data_df = self._all_data_df.drop(columns=zone_list)
                #    self._data._continuous_variables = [x for x in self._data._continuous_variables if x not in zone_list]
                #    self._data._categorical_variables = [x for x in self._data._categorical_variables if x not in zone_list]
                # except:
                #    pass
            else:
                explain_level = 0  # Setting explain level to 0 in case of Perform and Compete as EDA seems to be creating memory issues

            # Mapping and conversion of old mode names to new
            api_modes = ["Explain", "Perform", "Compete"]
            tool_modes = ["Basic", "Intermediate", "Advanced"]
            if mode in tool_modes:
                mode = api_modes[tool_modes.index(mode)]
            elif mode in api_modes:
                mode = mode
            else:
                mode = "Explain"

            if self._data._embedding_variables and mode != "Compete":
                warnings.warn(
                    "AutoML will be trained in Advanced/Compete mode when text or Image variables are used in model training."
                )
                mode = "Compete"

            try:
                import arcpy

                result_path = tempfile.mkdtemp(dir=arcpy.env.scratchFolder)
            except:
                result_path = tempfile.mkdtemp(dir=tempfile.gettempdir())

            self._sensitive_variables = kwargs.get("sensitive_variables", None)
            self._fairness_metric = kwargs.get("fairness_metric", "auto")
            self._fairness_threshold = kwargs.get("fairness_threshold", "auto")
            self._privileged_groups = kwargs.get("privileged_groups", [])
            self._underprivileged_groups = kwargs.get("underprivileged_groups", [])

            for grp in self._underprivileged_groups:
                for key in grp:
                    val = grp[key]
                    if val == "":
                        self._underprivileged_groups = []

            if (
                self._fairness_metric == "group_loss_difference"
                and self._fairness_threshold == "auto"
            ):
                warnings.warn(
                    "Fairness Threshold value is required to be passed when the chosen fairness metric is group_loss_difference."
                )
                # exit()

            if self._fairness_metric == "equalised_odds_ratio":
                self._fairness_metric = "equalized_odds_ratio"
            if self._fairness_metric == "equalised_odds_difference":
                self._fairness_metric = "equalized_odds_difference"

            self._model = base_AutoML(
                results_path=result_path,
                mode=mode,
                ml_task=ml_task,
                algorithms=algorithms,
                total_time_limit=total_time_limit,
                golden_features=False,
                explain_level=explain_level,
                eval_metric=eval_metric,
                n_jobs=n_jobs,
                kmeans_features=False,
                fairness_metric=self._fairness_metric,
                fairness_threshold=self._fairness_threshold,
                privileged_groups=self._privileged_groups,
                underprivileged_groups=self._underprivileged_groups,
            )
        else:
            result_path = self._data.path
            self._model = base_AutoML(results_path=result_path)
            self._model._results_path = self._data.path

    def get_ml_task(self, all_labels):
        try:
            if isinstance(all_labels[0], str):
                unique = np.unique(all_labels, return_counts=False)
                if len(unique) == 2:
                    return "binary_classification"
                elif len(unique) > 200 and len(unique) > int(0.5 * all_labels.shape[0]):
                    return "text"
                else:
                    return "multiclass_classification"
            else:
                return "auto"
        except:
            return "auto"

    def _impute_missing_values(self, data=None):
        original_dtype = data.dtypes
        numerical_transformer = make_pipeline(SimpleImputer(strategy="median"))

        categorical_transformer = make_pipeline(SimpleImputer(strategy="constant"))

        _procs = make_column_transformer(
            (numerical_transformer, self._data._continuous_variables),
            (categorical_transformer, self._data._categorical_variables),
            (numerical_transformer, self._data._embedding_variables),
        )
        if data is None:
            data = self._all_data_df
        try:
            processed_data = _procs.fit_transform(data)
            processed_data_df = pd.DataFrame(
                processed_data, columns=data.columns.values.tolist()
            )
            processed_data_df = processed_data_df.astype(original_dtype)
        except:
            processed_data_df = data
        return processed_data_df

    def fit(self, sample_weight=None):
        """
        Fits the AutoML model.
        """
        if getattr(self._data, "_is_not_empty", True):
            if isinstance(self._all_labels[0], int):
                self._all_labels = self._all_labels.astype(np.int32)
            elif isinstance(self._all_labels[0], float):
                self._all_labels = self._all_labels.astype(np.float)
            if self._sensitive_variables:
                sensitive_features = self._all_data_df[
                    self._sensitive_variables
                ].astype("category")
            else:
                sensitive_features = None
            try:
                self._model.fit(
                    self._all_data_df,
                    self._all_labels,
                    sample_weight=sample_weight,
                    sensitive_features=sensitive_features,
                )
            except:
                msg = arcpy_localization_helper(
                    "The desired models could not be trained using the input data provided.",
                    260150,
                    "ERROR",
                )
                exit()
        else:
            raise Exception("Fit can be called only with data.")
        # self.save()
        print(
            "All the evaluated models are saved in the path ",
            os.path.abspath(self._model._get_results_path()),
        )

    def show_results(self, rows=5):
        """
        Shows sample results for the model.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional number of rows. By default, 5 rows
                                are displayed.
        =====================   ===========================================
        :return:
            dataframe
        """
        if getattr(self._data, "is_not_empty", True) == False:
            raise Exception(
                "This method is not available when the model is initiated for prediction"
            )
        if (
            not self._data._is_unsupervised
            and (self._validation_data is None or self._validation_labels is None)
        ) or (self._data._is_unsupervised and self._validation_data is None):
            raise_data_exception()

        min_size = len(self._validation_data)

        if rows < min_size:
            min_size = rows

        # sample_batch = random.sample(self._data._validation_indexes, min_size)
        sample_batch = random.sample(range(len(self._validation_data)), min_size)
        validation_data_batch = self._validation_data.take(sample_batch, axis=0)
        # validation_data_batch_df = pd.DataFrame(validation_data_batch,
        # columns=self._data._continuous_variables + self._data._categorical_variables)
        sample_indexes = [self._data._validation_indexes[i] for i in sample_batch]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_labels = self._predict(validation_data_batch)
        pd.options.mode.chained_assignment = None
        if self._data._is_classification:
            df = self._data._dataframe.loc[sample_indexes]
        else:
            df = self._data._dataframe.iloc[
                sample_indexes
            ]  # .loc[sample_batch]#.reset_index(drop=True).loc[sample_batch].reset_index(drop=True)

        if self._data._dependent_variable:
            df[self._data._dependent_variable + "_results"] = output_labels
        else:
            df["prediction_results"] = output_labels

        return df.sort_index()

    def score(self):
        """
        :return:
            output from AutoML's model.score(), R2 score in case of regression and Accuracy in case of classification.
        """
        col_type = str(self._validation_labels.dtype)
        val_labels = self._validation_labels
        if col_type == "object":
            if isinstance(val_labels[0], float):
                val_labels = val_labels.astype(float)
            elif isinstance(val_labels[0], int):
                val_labels = val_labels.astype(int)
            else:
                val_labels = self._validation_labels
        val_labels = self._validation_labels.astype(int)
        if getattr(self._data, "_is_not_empty", True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return self._model.score(self._validation_data_df, val_labels)
        else:
            raise Exception(
                "This method is not available when the model is initiated for prediction"
            )

    def fairness_score(
        self,
        sensitive_feature,
        fairness_metrics=None,
        visualize=False,
    ):
        """
        Shows sample results for the model.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        sensitive_feature       Column name of the protected class.
        ---------------------   -------------------------------------------
        fairness_metrics        Allowed list of fairness metrics. List can
                                have any of the metrics from the list below.
                                Multiple metrics can be passed in the list.
                                 1. For Binary classification
                                    [
                                     "equalized_odds_difference",
                                     "demographic_parity_difference",
                                     "equalized_odds_ratio",
                                     "demographic_parity_ratio"
                                    ]
                                 2. for Regression
                                    [
                                    "MAE",
                                    "MSE",
                                    "RMSE",
                                    "MAPE"
                                    ]
                                 Metric should be one of the values mentioned in
                                 the list.
                                This method is not yet supported for multiclass
                                classification.
        ---------------------   -------------------------------------------
        visualize               A boolean value to visualize plot of metrics
        =====================   ===========================================
        :return: tuple/dataframe
        """
        if self._data._is_classification:
            validation_indexes = self._data._validation_indexes
        else:
            validation_indexes = self._data._dataframe.sample(
                n=round(0.1 * len(self._data._dataframe)),
                replace=False,
                random_state=42,
            ).index.to_list()
        self.sensitive_feature_series = self._validation_data_df.loc[
            :, [sensitive_feature]
        ]
        if self._sensitive_variables:
            return "Since AutoML was trained with fairness mitigation, the fairness score can be obtained by running the report() method."

        if not getattr(self._data, "_is_not_empty", True):
            raise Exception(
                "This method is not available when the model is initiated for prediction"
            )

        y_true = self._data._dataframe.loc[validation_indexes][
            self._data._dependent_variable
        ]
        y_true = y_true.reset_index(drop=True)
        y_pred = self.predict(
            self._data._dataframe.loc[validation_indexes], prediction_type="dataframe"
        )
        y_pred = y_pred["prediction_results"]
        y_pred = y_pred.reset_index(drop=True)

        if self._data._is_classification:
            le_1 = LabelEncoder()
            le_1.fit(y_true)
            y_true = le_1.transform(y_true)
            y_pred = le_1.transform(y_pred)

        return calculate_metrics(
            self._data._is_classification,
            self._data,
            y_true,
            y_pred,
            self.sensitive_feature_series,
            sensitive_feature,
            fairness_metrics,
            visualize,
        )

    def report(self):
        """
        :return:
            a report of the different models trained by AutoML along with their performance.
        """
        main_readme_html = os.path.join(self._model._results_path, "README.html")
        warnings.warn(
            "In case the report html is not rendered appropriately in the notebook, the same can be found in the path "
            "" + main_readme_html
        )
        return self._model.report()

    def predict_proba(self):
        """
        :return:
            output from AutoML's model.predict_proba() with prediction probability for the training data
        """
        if (self._data._is_classification == "classification") or (
            self._data._is_classification == True
        ):
            if getattr(self._data, "_is_not_empty", False):
                raise Exception(
                    "This method is not available when the model is initiated for prediction"
                )
            else:
                cols = (
                    self._data._continuous_variables + self._data._categorical_variables
                )
                data_df = pd.DataFrame(self._data._ml_data[0], columns=cols)
                return self._model.predict_proba(data_df)
        else:
            raise Exception("This method is applicable only for classification models.")

    def copy_and_overwrite(self, from_path, to_path):
        dest_dir = os.path.join(to_path, os.path.basename(from_path))
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(from_path, dest_dir)

    def _copy_reports(self, src_dir, dest):
        extensions = [".html", ".png", ".svg"]

        for root, dirs, files in os.walk(src_dir):
            for file in files:
                for extension in extensions:
                    if file.endswith(extension):
                        folder_name = os.path.basename(root)
                        src = os.path.join(root, file)
                        dest_folder = dest
                        if not folder_name.startswith("tmp"):
                            dest_folder = os.path.join(dest, folder_name)
                            if not os.path.isdir(dest_folder):
                                os.mkdir(dest_folder)
                        dest_folder = os.path.join(dest_folder, file)
                        shutil.copy2(src, dest_folder)

    def save(self, path):
        """
        Saves the model in the path specified. Creates an Esri Model and a dlpk.
        Uses pickle to save the model and transforms.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        path                    Path of the directory where the model should be saved.
        =====================   ===========================================
        :return:
            path
        """
        if getattr(self._data, "_is_not_empty", True) == False:
            raise Exception(
                "This method is not available when the model is initiated for prediction"
            )
        # Required files to be copied to new path
        files_required = [
            "data_info.json",
            "ldb_performance.png",
            "ldb_performance_boxplot.png",
            "params.json",
            "progress.json",
            "README.md",
            "drop_features.json",
            "model_explainer.sav",
        ]
        required_model_folders = []  # List of folders that are to be copied to new path
        base_file_name = os.path.basename(self._model._get_results_path())
        result_path = os.path.abspath(self._model._get_results_path())

        save_model_path = os.path.abspath(path)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        with open(Path(save_model_path) / "ArcGISImageClassifier.py", "w") as f:
            f.write(self._code)

        MLModel._save_encoders(
            self._data._encoder_mapping, save_model_path, base_file_name
        )

        if self._data._procs:
            MLModel._save_transforms(self._data._procs, save_model_path, base_file_name)

        try:
            self._save_explainer(save_model_path)
        except:
            print("Explainer could not be saved")

        self._write_emd(save_model_path, base_file_name)
        if (self._model._best_model._name == "Ensemble") or (
            self._model._best_model._name == "Ensemble_Stacked"
        ):
            model_map = self._model._best_model.models_map
            required_model_folders.append(os.path.join(result_path, "Ensemble"))
            for i in self._model._best_model.selected_models:
                # print(i['model'])
                sub_path = list(model_map.keys())[
                    list(model_map.values()).index(i["model"])
                ]
                final_path = os.path.join(result_path, sub_path)
                required_model_folders.append(final_path)
        else:
            final_path = os.path.join(result_path, self._model._best_model._name)
            required_model_folders.append(final_path)

        for folder in required_model_folders:
            # copyfolder(folder,dest)
            try:
                self.copy_and_overwrite(folder, save_model_path)
            except:
                print(
                    "It looks like the model has been already been saved once. Unable to save at a different location again"
                )
                return

        for file in files_required:
            abs_file_path = os.path.join(result_path, file)
            dest_file = os.path.join(save_model_path, os.path.basename(file))
            if os.path.isfile(abs_file_path):
                shutil.copyfile(abs_file_path, dest_file)

        # Copies reports if present
        try:
            self._copy_reports(result_path, save_model_path)
            copy_success = True
        except:
            copy_success = False
        if copy_success:
            shutil.rmtree(result_path)
        # Creates dlpk

        from ._arcgis_model import _create_zip

        _create_zip(Path(save_model_path).name, str(save_model_path))

        print("Model has been saved in the path", save_model_path)
        return save_model_path

    def _save_explainer(self, path):
        import shap

        if self._model._get_ml_task() == "regression":
            explainer = shap.KernelExplainer(
                self._shap_predict, shap.sample(self._all_data_df.values, 500)
            )
        else:
            explainer = shap.KernelExplainer(
                self._shap_predict,
                shap.sample(self._all_data_df.values, 500),
                link="logit",
            )
        filename = os.path.join(path, "model_explainer.sav")
        pickle.dump(explainer, open(filename, "wb"))

    def _write_emd(self, path, base_file_name):
        def convert(o):
            import numpy

            if isinstance(o, numpy.int64):
                return int(o)

        emd_file = os.path.join(path, base_file_name + ".emd")
        emd_params = {}
        emd_params["version"] = str(sklearn.__version__)
        # if not self._data._is_unsupervised:
        # emd_params["score"] = self.score()
        emd_params["_is_classification"] = (
            "classification" if self._data._is_classification else "regression"
        )
        emd_params["ModelName"] = "AutoML"
        emd_params["ResultsPath"] = self._model._results_path
        # emd_params['ModelFile'] = base_file_name + '.pkl'
        # emd_params['ModelParameters'] = self._model.get_params()
        emd_params["categorical_variables"] = self._data._categorical_variables

        if self._data._dependent_variable:
            emd_params["dependent_variable"] = self._data._dependent_variable

        emd_params["continuous_variables"] = self._data._continuous_variables
        emd_params["text_variables"] = self._data._text_variables
        if self._data._image_variables:
            emd_params["image_variables"] = self._data._image_variables
        emd_params["embedding_variables"] = self._data._embedding_variables
        if self._data._feature_field_variables:
            emd_params["_feature_field_variables"] = self._data._feature_field_variables
        if self._data._raster_field_variables:
            emd_params["_raster_field_variables"] = self._data._raster_field_variables

        emd_params["Framework"] = "arcgis.learn.models._inferencing"
        emd_params["ModelConfiguration"] = "_auto_ml"
        emd_params["InferenceFunction"] = "ArcGISImageClassifier.py"

        if self._model._get_ml_task() != "regression":
            try:
                emd_params["dependent_variable_unique"] = (
                    self._data._dataframe[self._data._dependent_variable]
                    .unique()
                    .tolist()
                )
            except:
                emd_params["dependent_variable_unique"] = []

        with open(emd_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(emd_params, indent=4, default=convert))

    @classmethod
    def from_model(cls, emd_path):
        """
        Creates an `AutoML Model` Object from an Esri Model Definition (EMD) file.
        The model object created can only be used for inference on a new dataset
        and cannot be retrained.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Esri Model Definition
                                file.
        =====================   ===========================================

        :return:
            :class:`~arcgis.learn.AutoML` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)
        emd_path_orig = Path(emd_path)
        emd_path = _get_emd_path(emd_path)
        if not HAS_AUTO_ML_DEPS:
            _raise_fastai_import_error(import_exception=import_exception)

        if not os.path.exists(emd_path):
            raise Exception("Invalid data path.")

        with open(emd_path, "r") as f:
            emd = json.loads(f.read())

        categorical_variables = emd["categorical_variables"]
        dependent_variable = emd.get("dependent_variable", None)
        continuous_variables = emd["continuous_variables"]
        text_variables = emd.get("text_variables", None)
        image_variables = emd.get("image_variables", None)
        embedding_variables = emd.get("embedding_variables", None)

        if emd["version"] != str(sklearn.__version__):
            warnings.warn(
                f"Sklearn version has changed. Model Trained using version {emd['version']}"
            )

        _is_classification = True
        if emd["_is_classification"] != "classification":
            _is_classification = False

        encoder_mapping = None
        if categorical_variables:
            encoder_path = os.path.join(
                os.path.dirname(emd_path),
                os.path.basename(emd_path).split(".")[0] + "_encoders.pkl",
            )
            if os.path.exists(encoder_path):
                with open(encoder_path, "rb") as f:
                    encoder_mapping = pickle.loads(f.read())

        column_transformer = None
        transforms_path = os.path.join(
            os.path.dirname(emd_path),
            os.path.basename(emd_path).split(".")[0] + "_transforms.pkl",
        )
        if os.path.exists(transforms_path):
            with open(transforms_path, "rb") as f:
                column_transformer = pickle.loads(f.read())

        empty_data = TabularDataObject._empty(
            categorical_variables,
            continuous_variables,
            dependent_variable,
            encoder_mapping,
            column_transformer,
            text_variables=text_variables,
            image_variables=image_variables,
            embedding_variables=embedding_variables,
        )
        empty_data._is_classification = _is_classification
        if _is_classification:
            try:
                empty_data.unique_var_list = emd["dependent_variable_unique"]
            except:
                empty_data.unique_var_list = None
        empty_data._is_not_empty = False
        # empty_data.path = emd["ResultsPath"]
        empty_data.path = emd_path.parent
        try:
            empty_data.explainer_path = os.path.join(
                os.path.dirname(emd_path),
                "model_explainer.sav",
            )
        except:
            empty_data.explainer_path = None
        return cls(data=empty_data)

    def _predict(self, data):
        data_df = pd.DataFrame(
            data,
            columns=self._data._continuous_variables
            + self._data._categorical_variables
            + self._data._embedding_variables,
        )
        data_df = self._impute_missing_values(data=data_df)
        return self._model.predict(data_df)

    def _shap_predict(self, data):
        data_df = pd.DataFrame(
            data,
            columns=self._data._continuous_variables
            + self._data._categorical_variables,
        )
        data_df = self._impute_missing_values(data=data_df)
        if self._model._get_ml_task() == "regression":
            return self._model.predict(data_df)
        else:
            return self._model.predict_proba(data_df)

    def _predict_all(self, data):
        data_df = pd.DataFrame(
            data,
            columns=self._data._continuous_variables
            + self._data._categorical_variables,
        )
        data_df = self._impute_missing_values(data=data_df)
        return self._model.predict_all(data_df)

    def _predict_proba(self, data):
        data_df = pd.DataFrame(
            data,
            columns=self._data._continuous_variables
            + self._data._categorical_variables,
        )
        data_df = self._impute_missing_values(data=data_df)
        return self._model.predict_proba(data_df)

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
        cell_sizes=[3, 4, 5, 6, 7],
        confidence=True,
        get_local_explanations=False,
        **kwargs,
    ):
        """

        Predict on data from feature layer, dataframe and or raster data.

        =================================   =========================================================================
        **Parameter**                        **Description**
        ---------------------------------   -------------------------------------------------------------------------
        input_features                      Optional :class:`~arcgis.features.FeatureLayer` or spatial dataframe. Required if prediction_type='features'.
                                            Contains features with location and
                                            some or all fields required to infer the dependent variable value.
        ---------------------------------   -------------------------------------------------------------------------
        explanatory_rasters                 Optional list. Required if prediction_type='raster'.
                                            Contains a list of raster objects containing
                                            some or all fields required to infer the dependent variable value.
        ---------------------------------   -------------------------------------------------------------------------
        datefield                           Optional string. Field name from feature layer
                                            that contains the date, time for the input features.
                                            Same as :meth:`~arcgis.learn.prepare_tabulardata`.
        ---------------------------------   -------------------------------------------------------------------------
        cell_sizes                          Size of H3 cells (specified as H3 resolution) for spatially
                                            aggregating input features and passing in the cell ids as additional
                                            explanatory variables to the model. If a spatial dataframe is passed
                                            as input_features, ensure that the spatial reference is 4326,
                                            and the geometry type is Point. Not applicable when explanatory_rasters
                                            are provided.
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

                                            |    {
                                            |        "Field_Name_1": "Field_1",
                                            |        "Field_Name_2": "Field_2"
                                            |   }
        ---------------------------------   -------------------------------------------------------------------------
        confidence                          Optional Bool.
                                            Set confidence to True to get prediction confidence for classification
                                            use cases.Default is True.
        =================================   =========================================================================

        :return:
            :class:`~arcgis.features.FeatureLayer` if prediction_type='features', dataframe for prediction_type='dataframe' else creates an output raster.

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
                cell_sizes,
                distance_features,
                output_layer_name,
                gis,
                match_field_names,
                prediction_type,
                confidence,
                get_local_explanations,
                **kwargs,
            )
        else:
            if not rasters:
                raise Exception("Rasters required for predict_features=False")

            if not output_raster_path:
                raise Exception(
                    "Please specify output_raster_folder_path to save the output."
                )

            return self._predict_rasters(
                output_raster_path, rasters, match_field_names, confidence
            )

    def _get_normalised_shap_values(self, processed_numpy):
        # try:
        filename_expl = self._data.explainer_path
        load_explainer = pickle.load(open(filename_expl, "rb"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = load_explainer.shap_values(processed_numpy, nsamples=100)
        shap_values_normalised = []
        if isinstance(shap_values, np.ndarray):
            shap_values_normalised = normalize(shap_values, axis=1, norm="l1")
        else:
            for val in shap_values:
                normed_matrix = normalize(val, axis=1, norm="l1")
                shap_values_normalised.append(normed_matrix)
        return shap_values_normalised

    def _predict_features(
        self,
        input_features,
        rasters=None,
        datefield=None,
        cell_sizes=[3, 4, 5, 6, 7],
        distance_feature_layers=None,
        output_name="Prediction Layer",
        gis=None,
        match_field_names=None,
        prediction_type="features",
        confidence=False,
        get_local_explanations=False,
        **kwargs,
    ):
        dataframe_complete = False
        attachment_list = kwargs.get("image_attach_list", None)
        if isinstance(input_features, FeatureLayer):
            try:
                import arcpy
                import spatial_reference_helper

                ex = input_features.properties["extent"]
                data_source_spatial_reference = arcpy.SpatialReference(
                    ex["spatialReference"].get(
                        "wkid", ex["spatialReference"].get("latestWkid")
                    )
                )
                extent = arcpy.Extent(
                    ex["xmin"],
                    ex["ymin"],
                    ex["xmax"],
                    ex["ymax"],
                    spatial_reference=data_source_spatial_reference,
                )
                transformation = spatial_reference_helper.get_datum_transformation(
                    data_source_spatial_reference, arcpy.SpatialReference(4326), extent
                )
            except:
                transformation = None
            if cell_sizes and not rasters:
                dataframe = input_features.query(
                    out_sr=4326, datum_transformation=transformation
                ).sdf
                dataframe = add_h3(dataframe, cell_sizes)
            else:
                dataframe = input_features.query().sdf

            if attachment_list:
                dataframe["Images"] = attachment_list
        elif (
            hasattr(input_features, "dataSource")
            or str(input_features).endswith(".shp")
            or isinstance(input_features, tuple)
        ):
            dataframe, index_data = TabularDataObject._sdf_gptool_workflow(
                input_features,
                distance_feature_layers,
                rasters,
                index_field=None,
                is_table_obj=False,
            )
            if cell_sizes and not rasters:
                dataframe = add_h3(dataframe, cell_sizes)
            if attachment_list:
                dataframe["Images"] = attachment_list
            dataframe_complete = True
            self._data._text_variables = self._data._text_variables or []
            self._data._image_variables = self._data._image_variables or []
            if len(self._data._text_variables + self._data._image_variables) > 0:
                dataframe, new_embd_cols = _extract_embeddings(
                    self._data._text_variables, self._data._image_variables, dataframe
                )
        elif hasattr(input_features, "value"):
            dataframe, index_data = TabularDataObject._sdf_gptool_workflow(
                input_features,
                distance_feature_layers,
                rasters,
                index_field=None,
                is_table_obj=True,
            )
            dataframe_complete = True
        else:
            dataframe = input_features.copy()

        self._data._text_variables = self._data._text_variables or []
        self._data._image_variables = self._data._image_variables or []

        fields_needed = (
            self._data._categorical_variables
            + self._data._continuous_variables
            + self._data._text_variables
            + self._data._image_variables
        )
        fields_needed_without_embeddings = (
            self._data._categorical_variables + self._data._continuous_variables
        )
        distance_feature_layers = (
            distance_feature_layers if distance_feature_layers else []
        )
        continuous_variables = self._data._continuous_variables
        non_categorical_variables = (
            self._data._continuous_variables
            + self._data._text_variables
            + self._data._image_variables
        )

        columns = dataframe.columns
        if dataframe_complete:
            processed_dataframe = dataframe
        else:
            feature_layer_columns = []
            for column in columns:
                column_name = column
                categorical = False

                if column_name in fields_needed:
                    if column_name in self._data._text_variables:
                        categorical = "text"
                    elif column_name in self._data._image_variables:
                        categorical = "image"
                    elif column_name not in continuous_variables:
                        categorical = True
                    else:
                        pass
                elif match_field_names and match_field_names.get(column_name):
                    if match_field_names.get(column_name) in self._data._text_variables:
                        categorical = "text"
                    elif (
                        match_field_names.get(column_name)
                        in self._data._image_variables
                    ):
                        categorical = "image"
                    elif match_field_names.get(column_name) not in continuous_variables:
                        categorical = True
                    else:
                        pass
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
                if not HAS_FASTAI:
                    _raise_fastai_import_error(import_exception=import_exception)
                warnings.simplefilter("ignore")
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
            try:
                list_of_train_fields = []
                for key, value in match_field_names.items():
                    if (not key == value) and (len(str(key)) > 0):
                        list_of_train_fields.append(value)
                processed_dataframe = processed_dataframe.drop(
                    list_of_train_fields, axis=1, errors="ignore"
                )
            except:
                pass
            processed_dataframe.rename(columns=match_field_names, inplace=True)

        for field in fields_needed_without_embeddings:
            if field not in processed_dataframe.columns:
                msg = arcpy_localization_helper(
                    "Data on which prediction in needed does not have the fields the model was trained on",
                    260152,
                    "ERROR",
                )
                exit()

        for column in processed_dataframe.columns:
            if column not in fields_needed:
                if "emb_" not in column:
                    processed_dataframe = processed_dataframe.drop(column, axis=1)

        processed_numpy = processed_dataframe[
            self._data._continuous_variables
            + self._data._categorical_variables
            + self._data._embedding_variables
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self._predict(processed_numpy)

        if get_local_explanations and self._data._embedding_variables:
            get_local_explanations = False
            warnings.warn(
                "Local explanations cannot be generated when text or image variables are used to train the model."
            )

        if get_local_explanations:
            shap_values_normalised = self._get_normalised_shap_values(processed_numpy)
        shap_df = pd.DataFrame()
        if confidence and self._model._ml_task in [
            "multiclass_classification",
            "binary_classification",
        ]:
            dataframe["prediction_results"] = predictions
            try:
                prediction_confidence = np.amax(
                    self._predict_proba(processed_numpy), axis=1
                )
                dataframe["prediction_confidence"] = prediction_confidence
            except:
                pass

            if get_local_explanations:
                try:
                    labels = self._data.unique_var_list
                    get_element_id = lambda x: labels.index(x)
                    index_id = dataframe["prediction_results"].map(get_element_id)

                    list_for_df = []
                    for index_df, row in dataframe.iterrows():
                        index = index_id[index_df]
                        shap_val_list = shap_values_normalised[index][index_df]
                        list_for_df.append(shap_val_list)
                    shap_df = pd.DataFrame(
                        list_for_df,
                        columns=[
                            i + "_imp"
                            for i in (
                                self._data._continuous_variables
                                + self._data._categorical_variables
                            )
                        ],
                    )
                except:
                    pass
        else:
            dataframe["prediction_results"] = predictions
            if get_local_explanations:
                try:
                    shap_df = pd.DataFrame(
                        shap_values_normalised,
                        columns=[
                            i + "_imp"
                            for i in (
                                self._data._continuous_variables
                                + self._data._categorical_variables
                            )
                        ],
                    )
                except:
                    pass
        dataframe_merged = pd.concat([dataframe, shap_df.abs()], axis=1)
        dataframe_merged = dataframe_merged.filter(regex="^(?!emb_)")

        if prediction_type == "dataframe":
            return dataframe_merged

        return dataframe_merged.spatial.to_featurelayer(output_name, gis)

    def _raster_sr(self, raster):
        try:
            return raster._engine_obj._raster.spatialReference
        except:
            try:
                import arcpy

                return arcpy.SpatialReference(raster.extent["spatialReference"]["wkid"])
            except:
                try:
                    import arcpy

                    return arcpy.SpatialReference(
                        raster.extent["spatialReference"]["wkt"]
                    )
                except:
                    msg = arcpy_localization_helper(
                        "One or more input rasters do not have a valid spatial reference.",
                        517,
                        "ERROR",
                    )

    def _predict_rasters(
        self, output_folder_path, rasters, match_field_names=None, confidence=False
    ):
        if not os.path.exists(os.path.dirname(output_folder_path)):
            raise Exception("Output directory doesn't exist")

        if os.path.exists(output_folder_path):
            raise Exception("Output Folder already exists")

        try:
            import arcpy
        except:
            raise Exception("This function requires arcpy.")

        try:
            import numpy as np
        except:
            raise Exception("This function requires numpy.")

        try:
            import pandas as pd
        except:
            raise Exception("This function requires pandas.")

        if not HAS_FAST_PROGRESS:
            raise Exception("This function requires fastprogress.")

        fields_needed = (
            self._data._categorical_variables + self._data._continuous_variables
        )

        cached_sr = arcpy.env.outputCoordinateSystem
        arcpy.env.outputCoordinateSystem = self._raster_sr(rasters[0])

        xmin = rasters[0].extent["xmin"]
        xmax = rasters[0].extent["xmax"]
        ymin = rasters[0].extent["ymin"]
        ymax = rasters[0].extent["ymax"]
        min_cell_size_x = rasters[0].mean_cell_width
        min_cell_size_y = rasters[0].mean_cell_height

        default_sr = self._raster_sr(rasters[0])

        for raster in rasters:
            point_upper_left = arcpy.PointGeometry(
                arcpy.Point(raster.extent["xmin"], raster.extent["ymax"]),
                self._raster_sr(raster),
            ).projectAs(default_sr)

            point_lower_right = arcpy.PointGeometry(
                arcpy.Point(raster.extent["xmax"], raster.extent["ymin"]),
                self._raster_sr(raster),
            ).projectAs(default_sr)

            cell_extent = arcpy.Extent(
                raster.extent["xmin"],
                raster.extent["ymin"],
                raster.extent["xmin"] + raster.mean_cell_width,
                raster.extent["ymin"] + raster.mean_cell_height,
                spatial_reference=self._raster_sr(raster),
            ).projectAs(default_sr)

            cx, cy = (
                abs(cell_extent.XMax - cell_extent.XMin),
                abs(cell_extent.YMax - cell_extent.YMin),
            )

            if xmin > point_upper_left.firstPoint.X:
                xmin = point_upper_left.firstPoint.X
            if ymax < point_upper_left.firstPoint.Y:
                ymax = point_upper_left.firstPoint.Y
            if xmax < point_lower_right.firstPoint.X:
                xmax = point_lower_right.firstPoint.X
            if ymin > point_lower_right.firstPoint.Y:
                ymin = point_lower_right.firstPoint.Y

            if min_cell_size_x < cx:
                min_cell_size_x = cx

            if min_cell_size_y < cy:
                min_cell_size_y = cy

        max_raster_columns = int(abs(math.ceil((xmax - xmin) / min_cell_size_x)))
        max_raster_rows = int(abs(math.ceil((ymax - ymin) / min_cell_size_y)))
        point_upper = arcpy.PointGeometry(arcpy.Point(xmin, ymax), default_sr)
        point_lower = arcpy.PointGeometry(arcpy.Point(xmax, ymin), default_sr)

        cell_extent = arcpy.Extent(
            xmin,
            ymin,
            xmin + min_cell_size_x,
            ymin + min_cell_size_y,
            spatial_reference=default_sr,
        )

        raster_data = {}
        for raster in rasters:
            field_name = raster.name

            point_upper_translated = point_upper.projectAs(self._raster_sr(raster))
            cell_extent_translated = cell_extent.projectAs(self._raster_sr(raster))

            if field_name not in fields_needed:
                if match_field_names and match_field_names.get(raster.name):
                    field_name = match_field_names.get(raster.name)

            ccxx, ccyy = (
                abs(cell_extent_translated.XMax - cell_extent_translated.XMin),
                abs(cell_extent_translated.YMax - cell_extent_translated.YMin),
            )

            raster_read = raster.read(
                origin_coordinate=(
                    point_upper_translated.firstPoint.X,
                    point_upper_translated.firstPoint.Y,
                ),
                ncols=max_raster_columns,
                nrows=max_raster_rows,
                cell_size=(ccxx, ccyy),
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

        for field in fields_needed:
            if (field not in list(raster_data.keys())) and (
                match_field_names and match_field_names.get(field, None) is None
            ):
                msg = arcpy_localization_helper(
                    "Data on which prediction is needed does not have the fields the model was trained on",
                    260152,
                    "ERROR",
                )
                exit()

        processed_data = []

        length_values = len(raster_data[list(raster_data.keys())[0]])
        for i in range(length_values):
            processed_row = []
            for raster_name in sorted(raster_data.keys()):
                processed_row.append(raster_data[raster_name][i])
            processed_data.append(processed_row)

        processed_df = pd.DataFrame(
            data=np.array(processed_data), columns=sorted(raster_data)
        )

        processed_numpy = self._data._process_data(processed_df, fit=False)

        predictions = self._predict(processed_numpy)
        if confidence and self._model._ml_task in [
            "multiclass_classification",
            "binary_classification",
        ]:
            prediction_confidence = np.amax(
                self._predict_proba(processed_numpy), axis=1
            )

        if isinstance(predictions[0], str):
            processed_df["predictions"] = predictions
            le = preprocessing.LabelEncoder()
            le.fit(processed_df["predictions"])
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            processed_df["predictions"] = le.transform(processed_df["predictions"])

            predictions = np.array(
                processed_df["predictions"].values.reshape(
                    [max_raster_rows, max_raster_columns]
                ),
                dtype=np.uint8,
            )
        else:
            predictions = np.array(
                predictions.reshape([max_raster_rows, max_raster_columns]),
                dtype="float64",
            )

        processed_raster = arcpy.NumPyArrayToRaster(
            predictions,
            arcpy.Point(xmin, ymin),
            x_cell_size=min_cell_size_x,
            y_cell_size=min_cell_size_y,
        )
        if isinstance(predictions[0], str):
            arcpy.management.BuildRasterAttributeTable(processed_raster)
            class_map = le_name_mapping
            "class_map=" + str(class_map)
            arcpy.management.CalculateField(
                processed_raster,
                "Class",
                expression="class_map.get(!Value!)",
                expression_type="PYTHON3",
                code_block="class_map=" + str(class_map),
            )
        processed_raster.save(output_folder_path)
        arcpy.env.outputCoordinateSystem = cached_sr

        return True
