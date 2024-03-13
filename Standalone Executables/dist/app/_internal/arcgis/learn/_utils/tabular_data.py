import random
import tempfile
import warnings
import sys
import math
import os
from pathlib import Path
import traceback

import arcgis
from arcgis.features import FeatureLayer

try:
    from fastai.tabular import TabularList
    from fastai.tabular import TabularDataBunch
    from fastai.tabular.transform import FillMissing, Categorify, Normalize
    from fastai.tabular import cont_cat_split, add_datepart
    from fastai.data_block import ItemLists, CategoryList, FloatList
    from .._utils.TSData import TimeSeriesList, To3dTensor
    from fastai.data_block import DatasetType
    import torch
    import pandas as pd
    from functools import reduce
    from .utils import arcpy_localization_helper

    HAS_FASTAI = True

except Exception as e:
    import_trace = traceback.format_exc()
    HAS_FASTAI = False

HAS_NUMPY = True
try:
    import numpy as np
except:
    HAS_NUMPY = False

HAS_SK_LEARN = True
try:
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
except:
    HAS_SK_LEARN = False

warnings.formatwarning = lambda msg, *args, **kwargs: f"{msg}\n"


class DummyTransform(object):
    def __int__(self):
        pass

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class TabularDataObject(object):
    _categorical_variables = []
    _continuous_variables = []
    _text_variables = []
    _image_variables = []
    dependent_variables = []

    @classmethod
    def prepare_data_for_layer_learner(
        cls,
        input_features,
        dependent_variable,
        feature_variables=None,
        raster_variables=None,
        date_field=None,
        cell_sizes=[3, 4, 5, 6, 7],
        distance_feature_layers=None,
        procs=None,
        val_split_pct=0.1,
        seed=42,
        stratify=False,
        batch_size=64,
        index_field=None,
        column_transforms_mapping=None,
        **kwargs,
    ):
        if not HAS_FASTAI:
            return

        feature_variables = feature_variables if feature_variables else []
        raster_variables = raster_variables if raster_variables else []

        random_split = True
        if kwargs.get("random_split") == False:
            random_split = False

        tabular_data = cls()
        (
            tabular_data._dataframe,
            tabular_data._field_mapping,
        ) = TabularDataObject._prepare_dataframe_from_features(
            input_features,
            dependent_variable,
            feature_variables,
            raster_variables,
            date_field,
            cell_sizes,
            distance_feature_layers,
            index_field,
            **kwargs,
        )

        if input_features is None:
            tabular_data._is_raster_only = True
        else:
            tabular_data._is_raster_only = False

        tabular_data._dataframe = tabular_data._dataframe.reindex(
            sorted(tabular_data._dataframe.columns), axis=1
        )

        tabular_data._categorical_variables = tabular_data._field_mapping[
            "categorical_variables"
        ]
        tabular_data._continuous_variables = tabular_data._field_mapping[
            "continuous_variables"
        ]
        tabular_data._text_variables = tabular_data._field_mapping["text_variables"]
        tabular_data._image_variables = tabular_data._field_mapping["image_variables"]
        tabular_data._embedding_variables = tabular_data._field_mapping[
            "embed_variables"
        ]
        tabular_data._dependent_variable = tabular_data._field_mapping[
            "dependent_variable"
        ]
        tabular_data._feature_field_variables = tabular_data._field_mapping[
            "feature_field_variables"
        ]
        tabular_data._raster_field_variables = tabular_data._field_mapping[
            "raster_field_variables"
        ]
        if tabular_data._dependent_variable:
            if isinstance(tabular_data._dependent_variable, list):
                for var in tabular_data._dependent_variable:
                    if (
                        var
                        in tabular_data._categorical_variables
                        + tabular_data._continuous_variables
                    ):
                        raise Exception(
                            "Variable to predict cannot be an explanatory variable"
                        )
            else:
                if (
                    tabular_data._dependent_variable
                    in tabular_data._categorical_variables
                    + tabular_data._continuous_variables
                ):
                    raise Exception(
                        "Variable to predict cannot be an explanatory variable"
                    )

            if (
                tabular_data._dataframe[tabular_data._dependent_variable]
                .isnull()
                .values.any()
            ):
                msg = arcpy_localization_helper(
                    "Rows having null values in dependent variable are removed and model will be trained with remaining data",
                    260145,
                    "WARNING",
                )
                tabular_data._dataframe = tabular_data._dataframe.dropna(
                    subset=tabular_data._dependent_variable
                )
        tabular_data._index_data = tabular_data._field_mapping["index_data"]
        tabular_data._index_field = index_field

        tabular_data._procs = procs
        tabular_data._column_transforms_mapping = column_transforms_mapping
        tabular_data._val_split_pct = val_split_pct
        tabular_data._bs = batch_size
        tabular_data._seed = seed
        tabular_data._cell_sizes = cell_sizes
        tabular_data._random_split = random_split

        tabular_data._is_empty = False

        validation_indexes = []
        if tabular_data._dependent_variable:
            random.seed(seed)
            if tabular_data._is_classification():
                dependent_variable_column = tabular_data._dataframe[
                    tabular_data._dependent_variable
                ]
                try:
                    total_val = len(dependent_variable_column.values)
                    unique_rows = dependent_variable_column.value_counts()
                    imabalanced_class_list = {}
                    for row, count in unique_rows.items():
                        if count < total_val * 0.01:
                            imabalanced_class_list[row] = count
                except Exception as e:
                    warnings.warn(f"Unable to check for class imbalance [reason : {e}]")

                if stratify:
                    if len(imabalanced_class_list) > 0:
                        try:
                            warnings.warn(
                                f"We see a class imbalance in the dataset. "
                                f'The class(es) {",".join([str(key) for key in imabalanced_class_list.keys()])} does '
                                f"not have enough data points in your dataset."
                            )
                        except:
                            warnings.warn("We see a class imbalance in the dataset")
                    try:
                        from sklearn.model_selection import train_test_split

                        if (
                            len(set(dependent_variable_column.values))
                            > len(dependent_variable_column.values) * val_split_pct
                        ):
                            classes = len(set(dependent_variable_column.values))
                            xlen = len(dependent_variable_column.values)
                            sample_shortage = math.ceil(
                                (classes - xlen * val_split_pct) / val_split_pct
                            )
                            req_instances_per_class = (xlen + sample_shortage) / classes
                            classes_below_req_intances = list(
                                dependent_variable_column.value_counts()[
                                    dependent_variable_column.value_counts()
                                    < req_instances_per_class
                                ].index
                            )
                            warnings.warn(
                                f"For valid statification all classes should have at least"
                                f" {str(req_instances_per_class)} data points, class(es)"
                                f' {",".join(str(classes_below_req_intances))} in your data does not meet the'
                                f" condition. Unable to perform stratified splitting, falling back to random split"
                            )
                            validation_indexes = tabular_data._dataframe.sample(
                                n=round(val_split_pct * len(tabular_data._dataframe)),
                                replace=False,
                                random_state=seed,
                            ).index.to_list()
                        else:
                            train, test = train_test_split(
                                tabular_data._dataframe,
                                test_size=val_split_pct,
                                random_state=seed,
                                stratify=dependent_variable_column,
                            )
                            validation_indexes = test.index.tolist()
                    except Exception as e:
                        warnings.warn(
                            f"Unable to perform stratified splitting [reason : {e}], falling back to random split"
                        )
                        validation_indexes = tabular_data._dataframe.sample(
                            n=round(val_split_pct * len(tabular_data._dataframe)),
                            replace=False,
                            random_state=seed,
                        ).index.to_list()
                else:
                    if len(imabalanced_class_list) > 0:
                        try:
                            warnings.warn(
                                f"We see a class imbalance in the dataset. The class(es) "
                                f'{",".join([str(key) for key in imabalanced_class_list.keys()])} '
                                f"does not have enough data points in your dataset. Although, "
                                f"class imbalance cannot be overcome easily, adding the parameter stratify = True "
                                f"will to a certain extent help get over this problem."
                            )
                        except:
                            warnings.warn("We see a class imbalance in the dataset")
                    validation_indexes = tabular_data._dataframe.sample(
                        n=round(val_split_pct * len(tabular_data._dataframe)),
                        replace=False,
                        random_state=seed,
                    ).index.to_list()
            else:
                if val_split_pct > 1:
                    val_split_pct = val_split_pct / len(tabular_data._dataframe)
                if tabular_data._random_split:
                    validation_indexes = random.sample(
                        range(len(tabular_data._dataframe)),
                        round(val_split_pct * len(tabular_data._dataframe)),
                    )
                else:
                    number_of_rec = math.ceil(
                        val_split_pct * len(tabular_data._dataframe)
                    )
                    validation_indexes = list(
                        range(
                            len(tabular_data._dataframe) - number_of_rec,
                            len(tabular_data._dataframe),
                        )
                    )

            tabular_data._validation_indexes = validation_indexes
        if tabular_data._dependent_variable:
            if tabular_data._is_classification():
                tabular_data._training_indexes = list(
                    set([i for i in tabular_data._dataframe.index])
                    - set(validation_indexes)
                )
            else:
                tabular_data._training_indexes = list(
                    set([i for i in range(len(tabular_data._dataframe))])
                    - set(validation_indexes)
                )

        if not tabular_data._dependent_variable:
            tabular_data._training_indexes = list(
                set([i for i in range(len(tabular_data._dataframe))])
                - set(validation_indexes)
            )
            tabular_data._validation_indexes = list(
                set([i for i in range(len(tabular_data._dataframe))])
            )
        if tabular_data._dependent_variable:
            tabular_data._is_classification = tabular_data._is_classification()
        else:
            tabular_data._is_classification = True

        tabular_data.path = Path(os.getcwd())
        return tabular_data

    @staticmethod
    def _min_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            if len(values) == 1:
                return_values.append(values[0])
                continue

            min_value = values[0]
            for value in values:
                if value < min_value:
                    min_value = value
            return_values.append(min_value)

        return return_values

    @staticmethod
    def _max_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(max(values))

        return return_values

    @staticmethod
    def _mean_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(sum(values) / len(values))

        return return_values

    @staticmethod
    def _majority_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(max(values, key=values.count))

        return return_values

    @staticmethod
    def _minority_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(min(values, key=values.count))

        return return_values

    @staticmethod
    def _sum_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(sum(values))

        return return_values

    @staticmethod
    def _std_dev_of(values_list):
        import statistics

        return_values = []

        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(statistics.stdev(values))

        return return_values

    @staticmethod
    def _variety(values_list):
        return_values = []
        for values in values_list:
            return_values.append(len(list(set(values))))

        return return_values

    @staticmethod
    def _get_calc(raster_type, calc_type):
        calc_type = calc_type.lower()

        cont_mapping = {
            "min": TabularDataObject._min_of,
            "max": TabularDataObject._max_of,
            "mean": TabularDataObject._mean_of,
            "majority": TabularDataObject._majority_of,
            "minority": TabularDataObject._minority_of,
            "std_dev": TabularDataObject._std_dev_of,
            "sum": TabularDataObject._sum_of,
            "variety": TabularDataObject._variety,
        }

        cat_mapping = {
            "majority": TabularDataObject._majority_of,
            "minority": TabularDataObject._minority_of,
            "variety": TabularDataObject._variety,
        }

        if raster_type:
            return cat_mapping.get(calc_type, TabularDataObject._majority_of)
        else:
            return cont_mapping.get(calc_type, TabularDataObject._mean_of)

    def _prepare_validation_databunch(self, dataframe):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            kwargs_variables = {"num_workers": 0} if sys.platform == "win32" else {}
            # kwargs_variables['tfm_y'] = True
            fm = FillMissing(self._categorical_variables, self._continuous_variables)
            fm.add_col = False
            fm(dataframe)
            databunch_half = (
                TabularList.from_df(
                    dataframe,
                    path=tempfile.NamedTemporaryFile().name,
                    cat_names=self._categorical_variables,
                    cont_names=self._continuous_variables,
                    procs=[Categorify, Normalize],
                )
                .split_by_idx([i for i in range(int(len(dataframe) / 2))])
                .label_empty()
                .databunch(**kwargs_variables)
            )

            databunch_second_half = (
                TabularList.from_df(
                    dataframe,
                    path=tempfile.NamedTemporaryFile().name,
                    cat_names=self._categorical_variables,
                    cont_names=self._continuous_variables,
                    procs=[Categorify, Normalize],
                )
                .split_by_idx(
                    [i for i in range(int(len(dataframe) / 2), len(dataframe))]
                )
                .label_empty()
                .databunch(**kwargs_variables)
            )

        return databunch_half, databunch_second_half

    @property
    def _is_unsupervised(self):
        if not self._dependent_variable:
            return True

        return False

    def _is_classification(self):
        if self._is_empty:
            return True

        if not HAS_NUMPY:
            raise Exception("This module requires numpy.")

        df = self._dataframe
        labels = df[self._dependent_variable]

        if labels.isna().sum().sum() != 0:
            msg = arcpy_localization_helper(
                "You have some missing values in dependent variable column.",
                260144,
                "ERROR",
            )
        classification_check_list = []
        for columns in labels.columns:
            unique_labels = labels[columns].unique()

            column_label = np.array(labels[columns])

            # from numbers import Integral

            if (
                isinstance(column_label[0], (float, np.float32))
                or len(unique_labels) > 20
            ):
                classification_check_list.append(False)
            else:
                classification_check_list.append(True)
        return all(classification_check_list)

    def _is_categorical(self, labels):
        unique_labels = labels.unique()

        labels = np.array(labels)

        from numbers import Integral

        if isinstance(labels[0], (float, np.float32)) or len(unique_labels) > 20:
            return False

        # if isinstance(int(labels[0]), (str, Integral)): # removing type casting.
        if isinstance(labels[0], (str, Integral)):
            return True

    @property
    def _databunch(self):
        if self._is_empty:
            return None

        if self._procs is not None and not isinstance(self._procs, list):
            self._procs = []

        return TabularDataObject._prepare_databunch(
            self._dataframe,
            self._field_mapping,
            self._procs,
            self._validation_indexes,
            self._bs,
            self._is_classification,
        )

    @property
    def _ml_data(self):
        if self._is_empty:
            return None, None, None, None

        if not HAS_NUMPY:
            raise Exception("This module requires numpy.")

        if not HAS_SK_LEARN:
            raise Exception("This module requires scikit-learn.")

        dataframe = self._dataframe
        # Dtype conversion because native pandas format will break the plotting libraries
        for i in dataframe.columns:
            if isinstance(dataframe.loc[:, i].dtype, pd.Float64Dtype):
                dataframe.loc[:, i] = dataframe.loc[:, i].astype(np.float64)

        labels = None
        # restore the behaviour as earler
        if isinstance(self._dependent_variable, list):
            self._dependent_variable = self._dependent_variable[0]

        if self._dependent_variable:
            labels = np.array(dataframe[self._dependent_variable])
            dataframe = dataframe.drop(self._dependent_variable, axis=1)

        if not self._procs:
            numerical_transformer = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler()
            )

            categorical_transformer = make_pipeline(SimpleImputer(strategy="constant"))

            self._procs = make_column_transformer(
                (numerical_transformer, self._continuous_variables),
                (numerical_transformer, self._embedding_variables),
                (categorical_transformer, self._categorical_variables),
            )

        _procs = self._procs

        self._encoder_mapping = None
        if self._categorical_variables:
            mapping = {}
            for variable in self._categorical_variables:
                labelEncoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                try:
                    dataframe[variable] = np.array(
                        labelEncoder.fit_transform(
                            dataframe[variable].values.astype(str).reshape(-1, 1)
                        ),
                        dtype="int64",
                    )
                except:
                    dataframe[variable] = np.array(
                        labelEncoder.fit_transform(
                            dataframe[variable]
                            .values.astype(str)
                            .to_numpy()
                            .reshape(-1, 1)
                        ),
                        dtype="int64",
                    )
                mapping[variable] = labelEncoder
            self._encoder_mapping = mapping

        try:
            processed_data = _procs.fit_transform(dataframe)
        except:
            msg = arcpy_localization_helper(
                "Unable to fit transforms. This could be because some of the columns in your dataset have multiple "
                "datatypes.",
                260143,
                "ERROR",
            )

        if self._is_classification:
            scaled_features_df = pd.DataFrame(
                processed_data,
                index=dataframe.index,
                columns=self._continuous_variables
                + self._embedding_variables
                + self._categorical_variables,
            )
            scaled_labels_df = pd.DataFrame(labels, index=dataframe.index)

            training_data = scaled_features_df.loc[self._training_indexes].to_numpy()
            training_labels = None
            if self._dependent_variable:
                training_labels = (
                    scaled_labels_df.loc[self._training_indexes].to_numpy().squeeze()
                )

            validation_data = scaled_features_df.loc[
                self._validation_indexes
            ].to_numpy()
            validation_labels = None
            if self._dependent_variable:
                validation_labels = (
                    scaled_labels_df.loc[self._validation_indexes].to_numpy().squeeze()
                )

            del scaled_features_df
            del scaled_labels_df
        else:
            training_data = processed_data.take(self._training_indexes, axis=0)
            training_labels = None
            if self._dependent_variable:
                training_labels = labels.take(self._training_indexes)

            validation_data = processed_data.take(self._validation_indexes, axis=0)
            validation_labels = None
            if self._dependent_variable:
                validation_labels = labels.take(self._validation_indexes)

        return training_data, training_labels, validation_data, validation_labels

    def _time_series_bunch(
        self, seq_len, location_var, normalize=True, bunch=True, multistep=False
    ):
        step = 1
        if multistep and len(list(self._dataframe.columns.values)) != 1:
            step = seq_len // 2

        self._index_seq = None
        if self._index_data is not None:
            bunched = []
            # Changing it because it misses the edge case
            # for i in range(len(self._index_data) - seq_len - 1):
            for i in range(len(self._index_data) - seq_len - (step - 1)):
                bunched.append(list(self._index_data[i : i + seq_len]))

            self._index_seq = np.array(bunched)

        if location_var:
            location_var_data = self._dataframe[location_var]
            self._dataframe = self._dataframe.drop(location_var, axis=1)
            if location_var in self._continuous_variables:
                self._continuous_variables.remove(location_var)
            elif location_var in self._categorical_variables:
                self._categorical_variables.remove(location_var)
            else:
                location_var_data = None
        else:
            location_var_data = None

        if self._is_raster_only:
            return self._raster_timeseries_bunch(normalize, bunch)

        if len(list(self._dataframe.columns.values)) == 1:
            return self._univariate_bunch(seq_len, normalize, bunch, location_var_data)
        else:
            return self._multivariate_bunch(
                seq_len, normalize, bunch, location_var_data, multistep=multistep
            )

    def _raster_timeseries_bunch(self, normalize=True, bunched=True):
        kwargs_variables = {"num_workers": 0} if sys.platform == "win32" else {}

        kwargs_variables["bs"] = self._bs

        if (
            hasattr(arcgis, "env")
            and getattr(arcgis.env, "_processorType", "") == "CPU"
        ):
            kwargs_variables["device"] = torch.device("cpu")

        processed_dataframe = self._col_transform(normalize=normalize)
        big_bunch = []

        proc_df = processed_dataframe.copy()
        proc_df = proc_df.drop(self._dependent_variable, axis=1)
        # Why this was inserted it can simply be converted to array

        for i in range(len(processed_dataframe)):
            big_bunch.append([proc_df.iloc[i].values])

        big_bunch = np.array(big_bunch)

        random.seed(self._seed)
        validation_indexes = random.sample(
            range(big_bunch.shape[0]), round(self._val_split_pct * big_bunch.shape[0])
        )

        self._validation_indexes_ts = validation_indexes

        self._training_indexes_ts = list(
            set([i for i in range(big_bunch.shape[0])]) - set(validation_indexes)
        )

        X_train = big_bunch.take(self._training_indexes_ts, axis=0)
        X_valid = big_bunch.take(self._validation_indexes_ts, axis=0)

        y_train = np.array(
            processed_dataframe[self._dependent_variable].take(
                self._training_indexes_ts
            )
        )
        y_valid = np.array(
            processed_dataframe[self._dependent_variable].take(
                self._validation_indexes_ts
            )
        )

        if bunched is False:
            return X_train, X_valid, y_train, y_valid

        data = (
            ItemLists(".", TimeSeriesList(X_train), TimeSeriesList(X_valid))
            .label_from_lists(y_train, y_valid, label_cls=FloatList)
            .databunch(**kwargs_variables)
        )

        return data

    def _multivariate_bunch(
        self, seq_len, normalize=True, bunched=True, location_var=None, multistep=False
    ):
        step = 1
        if multistep:
            step = seq_len // 2
        kwargs_variables = {"num_workers": 0} if sys.platform == "win32" else {}

        kwargs_variables["bs"] = self._bs

        if (
            hasattr(arcgis, "env")
            and getattr(arcgis.env, "_processorType", "") == "CPU"
        ):
            kwargs_variables["device"] = torch.device("cpu")

        processed_dataframe = self._col_transform(normalize)
        big_bunch = []
        target_bunch = []
        if location_var is not None:
            big_loc_processed_dataframe = pd.DataFrame()
            big_loc_processed_dataframe["temp_location"] = location_var
            unq_locations = location_var.unique()
        else:
            unq_locations = [None]
        # preserve the ordering of the column
        order_columns = (
            self._dependent_variable
            + self._continuous_variables
            + self._categorical_variables
        )
        processed_dataframe = processed_dataframe.loc[:, order_columns]

        base_index = 0
        all_index = []
        for k in range(len(unq_locations)):
            if unq_locations[0] is not None:
                loc_processed_dataframe = processed_dataframe[
                    big_loc_processed_dataframe["temp_location"] == unq_locations[k]
                ]
                loc_processed_dataframe.reset_index(inplace=True, drop=True)
            else:
                loc_processed_dataframe = processed_dataframe

            for i in range(len(loc_processed_dataframe) - seq_len - step + 1):
                bunch = []
                tb = []
                for col in list(loc_processed_dataframe.columns.values):
                    bunch.append(list(loc_processed_dataframe[col][i : i + seq_len]))
                    if col in self._dependent_variable:
                        small_tb = []
                        for st in range(step):
                            small_tb.append(
                                loc_processed_dataframe[col][i + st + seq_len]
                            )
                        tb.append(small_tb)
                big_bunch.append(bunch)
                target_bunch.append(
                    np.stack(tb, axis=1).ravel()
                )  # relying on the fastai loss calculation where they
                # flatten the output then calculate the loss
            total_slices = len(loc_processed_dataframe) - seq_len - step + 1
            base_index = self._sample_slice(
                len(loc_processed_dataframe),
                all_index,
                base_index,
                total_slices,
                multistep=multistep,
                step=step,
            )
        # generate the validatio index in slices
        big_bunch = np.array(big_bunch)
        target_bunch = np.array(target_bunch)
        random.seed(self._seed)
        validation_indexes = reduce(
            lambda x, y: x + y, map(lambda i: list(i[0] + np.array(i[1])), all_index)
        )
        self._validation_indexes_ts = validation_indexes

        self._training_indexes_ts = list(
            set([i for i in range(big_bunch.shape[0])]) - set(validation_indexes)
        )
        X_train = big_bunch.take(self._training_indexes_ts, axis=0)
        X_valid = big_bunch.take(self._validation_indexes_ts, axis=0)
        y_train = target_bunch.take(self._training_indexes_ts, axis=0)
        y_valid = target_bunch.take(self._validation_indexes_ts, axis=0)

        if bunched is False:
            return X_train, X_valid, y_train, y_valid

        data = (
            ItemLists(".", TimeSeriesList(X_train), TimeSeriesList(X_valid))
            .label_from_lists(y_train, y_valid, label_cls=FloatList)
            .databunch(**kwargs_variables)
        )

        return data

    def _univariate_bunch(self, seq_len, normalize=True, bunch=True, location_var=None):
        # handle the case, as the whole bunch is written as per string convert it to string
        _dependent_variable = self._dependent_variable[0]
        kwargs_variables = {"num_workers": 0} if sys.platform == "win32" else {}

        kwargs_variables["bs"] = self._bs

        if (
            hasattr(arcgis, "env")
            and getattr(arcgis.env, "_processorType", "") == "CPU"
        ):
            kwargs_variables["device"] = torch.device("cpu")

        df_columns = {}
        for i in range(seq_len):
            df_columns[f"att{i + 1}"] = []

        df_columns["target"] = []

        if self._is_classification:
            self._encoder_mapping = None
            mapping = {}
            labelEncoder = LabelEncoder()
            self._dataframe[_dependent_variable] = np.array(
                labelEncoder.fit_transform(self._dataframe[_dependent_variable]),
                dtype="int64",
            )
            mapping[_dependent_variable] = labelEncoder
            self._encoder_mapping = mapping

        if normalize:
            if not self._column_transforms_mapping.get(_dependent_variable):
                self._column_transforms_mapping[_dependent_variable] = [MinMaxScaler()]

            processed_dataframe = self._dataframe.copy()
            transformed_data = processed_dataframe[_dependent_variable]
            for transform in self._column_transforms_mapping[_dependent_variable]:
                try:
                    transformed_data = transform.fit_transform(
                        np.array(
                            transformed_data,
                            dtype=processed_dataframe[_dependent_variable].dtype,
                        ).reshape(-1, 1)
                    )
                except:
                    transformed_data = transform.fit_transform(
                        np.array(
                            transformed_data,
                            dtype=type(processed_dataframe[_dependent_variable][0]),
                        ).reshape(-1, 1)
                    )
                transformed_data = transformed_data.squeeze(1)

            try:
                processed_dataframe[_dependent_variable] = np.array(
                    transformed_data,
                    dtype=self._dataframe[_dependent_variable].dtype,
                )
            except:
                processed_dataframe[_dependent_variable] = np.array(
                    transformed_data,
                    dtype=type(processed_dataframe[_dependent_variable][0]),
                )
        else:
            processed_dataframe = self._dataframe.copy()

        import pandas as pd

        if location_var is not None:
            big_loc_processed_dataframe = pd.DataFrame()
            big_loc_processed_dataframe["temp_location"] = location_var
            unq_locations = location_var.unique()
        else:
            unq_locations = [None]

        base_index = 0
        all_index = []
        for k in range(len(unq_locations)):
            if unq_locations[0] is not None:
                loc_processed_dataframe = processed_dataframe[
                    big_loc_processed_dataframe["temp_location"] == unq_locations[k]
                ]
                loc_processed_dataframe.reset_index(inplace=True, drop=True)
            else:
                loc_processed_dataframe = processed_dataframe
            for i in range(len(loc_processed_dataframe[_dependent_variable]) - seq_len):
                for j in range(seq_len):
                    if (
                        len(loc_processed_dataframe[_dependent_variable])
                        > i + seq_len - 1
                    ):
                        df_columns[f"att{j + 1}"].append(
                            loc_processed_dataframe[_dependent_variable][i + j]
                        )
                    else:
                        continue

                df_columns["target"].append(
                    loc_processed_dataframe[_dependent_variable][i + seq_len]
                )
            total_slices = len(loc_processed_dataframe) - seq_len
            base_index = self._sample_slice(
                len(loc_processed_dataframe),
                all_index,
                base_index,
                total_slices,
                multistep=False,
            )
        df = pd.DataFrame(df_columns)

        columns = list(df.columns.values)
        columns.remove("target")

        random.seed(self._seed)
        validation_indexes = reduce(
            lambda x, y: x + y, map(lambda i: list(i[0] + np.array(i[1])), all_index)
        )
        self._validation_indexes_ts = validation_indexes

        self._training_indexes_ts = list(
            set([i for i in range(len(df))]) - set(validation_indexes)
        )

        y_train = np.array(df["target"].take(self._training_indexes_ts))
        y_valid = np.array(df["target"].take(self._validation_indexes_ts))

        X_train = To3dTensor(
            df.iloc[:, :-1].take(self._training_indexes_ts).values.astype(np.float32)
        )
        X_valid = To3dTensor(
            df.iloc[:, :-1].take(self._validation_indexes_ts).values.astype(np.float32)
        )

        if bunch is False:
            return X_train, X_valid, y_train, y_valid

        if self._is_classification:
            label_cls = CategoryList
        else:
            label_cls = FloatList

        data = (
            ItemLists(".", TimeSeriesList(X_train), TimeSeriesList(X_valid))
            .label_from_lists(y_train, y_valid, label_cls=label_cls)
            .databunch(**kwargs_variables)
        )

        return data

    def _sample_slice(
        self, total_length, all_index, base_index, total_slices, multistep=False, step=1
    ):
        if self._val_split_pct < 1:
            validation_no_rec = round(self._val_split_pct * total_length)
        else:
            validation_no_rec = self._val_split_pct
        if multistep:
            # First separate last n steps and then sample rest of the records from the bunchs
            number_of_extra_records = validation_no_rec - step
            extra_records = list(range(total_slices - step, total_slices))
            if number_of_extra_records > 0:
                extra_records += random.sample(
                    range(total_slices - step), number_of_extra_records
                )
            all_index.append((base_index, extra_records))
        else:
            if self._random_split:
                # validation_no_rec = round(self._val_split_pct * len(loc_processed_dataframe))
                all_index.append(
                    (base_index, random.sample(range(total_slices), validation_no_rec))
                )
            else:
                # number_of_rec = math.ceil(self._val_split_pct * len(loc_processed_dataframe))
                all_index.append(
                    (
                        base_index,
                        list(range(total_slices - validation_no_rec, total_slices)),
                    )
                )
        base_index += total_slices
        return base_index

    def _col_transform(self, normalize):
        self._encoder_mapping = None
        mapping = {}
        df = self._dataframe.copy()  # .drop(self._dependent_variable, axis=1)

        for col in list(df.columns.values):
            if self._is_categorical(df[col]):
                labelEncoder = LabelEncoder()
                df[col] = np.array(labelEncoder.fit_transform(df[col]), dtype="int64")
                mapping[col] = labelEncoder
                if col not in self._categorical_variables:
                    warnings.warn(
                        f"Field {col} is not marked as categorical. But, "
                        f"we inferred it as categorical variable. Treating it as categorical variable"
                        f" for processing. "
                    )
                    self._categorical_variables.append(col)
                    if col in self._continuous_variables:
                        self._continuous_variables.remove(col)

        self._encoder_mapping = mapping
        if normalize:
            for col in list(df.columns):
                if len(self._column_transforms_mapping.get(col, [])) == 0:
                    if col in self._encoder_mapping:
                        self._column_transforms_mapping[col] = [
                            self._encoder_mapping[col]
                        ]
                    else:
                        self._column_transforms_mapping[col] = [MinMaxScaler()]
                else:
                    # adding this to preserve the default label encoder.
                    if col in self._encoder_mapping:
                        self._column_transforms_mapping[col] = [
                            self._encoder_mapping[col]
                        ]

            processed_dataframe = df.copy()
            for col in list(df.columns):
                transformed_data = df[col]
                for transform in self._column_transforms_mapping.get(col, []):
                    if isinstance(transform, LabelEncoder):
                        transformed_data = np.array(transformed_data).reshape(-1, 1)
                    else:
                        try:
                            transformed_data = transform.fit_transform(
                                np.array(transformed_data, dtype=df[col].dtype).reshape(
                                    -1, 1
                                )
                            )
                        except:
                            transformed_data = transform.fit_transform(
                                np.array(
                                    transformed_data,
                                    dtype=type(df[col][0]),
                                ).reshape(-1, 1)
                            )
                        transformed_data = transformed_data.squeeze(1)
                try:
                    processed_dataframe[col] = np.array(
                        transformed_data, dtype=df[col].dtype
                    )
                except:
                    processed_dataframe[col] = np.array(
                        transformed_data, dtype=type(df[col][0])
                    )
        else:
            processed_dataframe = df.copy()

        return processed_dataframe

    def _process_data(self, dataframe, fit=True):
        if not HAS_NUMPY:
            raise Exception("This module requires numpy.")

        if not HAS_SK_LEARN:
            raise Exception("This module requires scikit-learn.")

        if not self._procs:
            numerical_transformer = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler()
            )

            categorical_transformer = make_pipeline(SimpleImputer(strategy="constant"))

            self._procs = make_column_transformer(
                (numerical_transformer, self._continuous_variables),
                (categorical_transformer, self._categorical_variables),
            )

        _procs = self._procs

        # if hasattr(self,'_encoder_mapping'):
        if self._encoder_mapping:
            for variable, encoder in self._encoder_mapping.items():
                try:
                    if fit:
                        dataframe[variable] = np.array(
                            encoder.fit_transform(
                                dataframe[variable].values.astype(str).reshape(-1, 1)
                            ),
                            dtype="int64",
                        )
                    else:
                        dataframe[variable] = np.array(
                            encoder.transform(
                                dataframe[variable].values.astype(str).reshape(-1, 1)
                            ),
                            dtype="int64",
                        )
                except:
                    if fit:
                        dataframe[variable] = np.array(
                            encoder.fit_transform(
                                dataframe[variable]
                                .values.astype(str)
                                .to_numpy()
                                .reshape(-1, 1)
                            ),
                            dtype="int64",
                        )
                    else:
                        dataframe[variable] = np.array(
                            encoder.transform(
                                dataframe[variable]
                                .values.astype(str)
                                .to_numpy()
                                .reshape(-1, 1)
                            ),
                            dtype="int64",
                        )

        if fit:
            processed_data = _procs.fit_transform(dataframe)
        else:
            processed_data = _procs.transform(dataframe)
        if self._procs:
            list_of_transformed_cols = []
            for cnt, transform in enumerate(self._procs.transformers):
                for col in self._procs.transformers[cnt][-1]:
                    list_of_transformed_cols.append(col)
            processed_orig_data = dataframe.copy()
            processed_orig_data[list_of_transformed_cols] = processed_data
            processed_data = processed_orig_data

        return processed_data

    def show_batch(self, rows=5, graph=False, seq_len=None):
        """
        Shows a chunk of data prepared without applying transforms.
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional integer. Number of rows of dataframe
                                or graph to plot. This parameter is not used
                                when plotting complete data.
        ---------------------   -------------------------------------------
        graph                   Optional boolean. Used for visualizing
                                time series data. The index_field passed
                                in prepare_tabulardata is used on the x-axis.
                                Use this option to plot complete data.
        ---------------------   -------------------------------------------
        seq_len                 Optional integer. Used for visualizing data
                                in the form of graph of seq_len duration.
        =====================   ===========================================
        """

        if seq_len is not None or graph is True:
            self._show_graph(seq_len=seq_len, rows=rows)
            return

        if not rows or rows <= 0:
            rows = self._bs
        elif rows > len(self._training_indexes):
            rows = len(self._training_indexes)

        random_batch = random.sample(self._training_indexes, rows)

        # using loc instead of iloc to get data when dataframe doesn't have continuous indexes
        if self._is_classification:
            return self._dataframe.loc[random_batch].sort_index()
        else:
            return self._dataframe.iloc[random_batch].sort_index()

    def _safe_div(self, arr):
        len_arr = len(arr) - 1
        if len_arr % 10 == 0:
            return np.linspace(0, len_arr, 10).astype(int)
        elif len_arr < 10:
            return np.linspace(0, len_arr, len_arr + 1).astype(int)
        else:
            return np.linspace(0, len_arr, 6).astype(int)

    def _show_graph(self, seq_len=None, rows=5):
        """
        Shows a batch of prepared data in the form of graphs
        """

        if self._is_unsupervised:
            raise Exception("Show Graphs is used for Time Series Network")

        import matplotlib.pyplot as plt

        # Check whether the index is timestamp
        sample_ticks = False
        index_data_copy = self._index_data
        if not pd.core.dtypes.common.is_datetime_or_timedelta_dtype(index_data_copy):
            # Try to convert the datatype to timestamp
            warnings.warn("Index field is not timestamp. Converting it to timestamp.")
            try:
                index_data_copy = pd.to_datetime(index_data_copy)
            except:
                sample_ticks = True

        if seq_len is not None:
            try:
                X_train, X_valid, y_train, y_valid = self._time_series_bunch(
                    seq_len, False, False
                )
            except:
                ts_bunch = self._time_series_bunch(seq_len, False, False)
                X_train = ts_bunch.train_ds.x.items
                y_train = ts_bunch.train_ds.y.items

            n_items = rows**2
            if n_items > len(X_train):
                n_items = len(X_train)

            sample = random.sample(range(len(X_train)), n_items)
            X_train_sample = np.array(X_train).take(sample, axis=0)
            y_train_sample = np.array(y_train).take(sample, axis=0)

            batched_index = []
            if index_data_copy is not None:
                indexes = 0
                while indexes < len(index_data_copy):
                    batched_index.append(index_data_copy[indexes : indexes + seq_len])
                    indexes = indexes + seq_len
            else:
                j = 0
                while j < n_items:
                    batched_index.append([i for i in range(seq_len)])
                    j = j + 1

            rows = int(math.sqrt(n_items))

            fig, axs = plt.subplots(rows, rows, figsize=(10, 10))

            for i in range(rows):
                for j in range(rows):
                    for predictor in X_train_sample[i + j]:
                        axs[i, j].plot(batched_index[i + j], predictor)

                    if isinstance(y_train_sample[i + j], (list, np.ndarray)):
                        val = ",".join([str(i) for i in y_train_sample[i + j]])
                    else:
                        val = y_train_sample[i + j]

                    axs[i, j].set_title(val)
                    if sample_ticks:
                        axs[i, j].xaxis.set_major_locator(
                            plt.FixedLocator(self._safe_div(range(seq_len)))
                        )
                    axs[i, j].tick_params(axis="x", labelrotation=60)

            plt.tight_layout()
            plt.show()
        else:
            # plotting the points
            # y = self._dataframe[self._dependent_variable]
            if index_data_copy is not None:
                x = index_data_copy
            else:
                x = [i for i in range(len(self._dataframe[self._dependent_variable]))]

            fig, axs = plt.subplots(
                len(list(self._dataframe.columns)),
                1,
                figsize=(25, 5 * len(list(self._dataframe.columns))),
            )
            counter = 0
            for col in list(self._dataframe.columns):
                if isinstance(axs, np.ndarray):
                    axs[counter].plot(x, self._dataframe[col], label=col)
                    if sample_ticks:
                        axs[counter].xaxis.set_major_locator(
                            plt.FixedLocator(
                                self._safe_div(range(len(self._dataframe[col])))
                            )
                        )
                    axs[counter].set_title(col)
                    axs[counter].tick_params(axis="x", labelrotation=60)
                else:
                    axs.plot(x, self._dataframe[col], label=col)
                    if sample_ticks:
                        axs.xaxis.set_major_locator(
                            plt.FixedLocator(
                                self._safe_div(range(len(self._dataframe[col])))
                            )
                        )
                    axs.set_title(col)
                    axs.tick_params(axis="x", labelrotation=60)
                counter = counter + 1

            plt.show()

    @staticmethod
    def _prepare_dataframe_from_features(
        input_features,
        dependent_variable,
        feature_variables=None,
        raster_variables=None,
        date_field=None,
        cell_sizes=[3, 4, 5, 6, 7],
        distance_feature_layers=None,
        index_field=None,
        **kwargs,
    ):
        feature_variables = feature_variables if feature_variables else []
        raster_variables = raster_variables if raster_variables else []
        distance_feature_layers = (
            distance_feature_layers if distance_feature_layers else []
        )
        feature_field_variables = []
        raster_field_variables = []
        continuous_variables = []
        categorical_variables = []
        text_variables = []
        image_variables = []
        for field in feature_variables:
            if isinstance(field, tuple):
                if field[1]:
                    if str(field[1]).lower() == "text":
                        text_variables.append(field[0])
                        feature_field_variables.append(field[0])
                    elif str(field[1]).lower() == "image":
                        image_variables.append(field[0])
                        feature_field_variables.append(field[0])
                    else:
                        categorical_variables.append(field[0])
                        feature_field_variables.append(field[0])
                else:
                    continuous_variables.append(field[0])
                    feature_field_variables.append(field[0])
            else:
                continuous_variables.append(field)
                feature_field_variables.append(field)

        rasters = []
        bands = []
        for i, raster in enumerate(raster_variables):
            if isinstance(raster, tuple):
                len_tuple = len(raster)
                rasters.append(raster[0])
                if len_tuple == 2:
                    if isinstance(raster[1], tuple):
                        bands.append(raster[1])
                        band_count = len(raster[1])
                        if band_count > raster[0].band_count:
                            raise (
                                "Incorrect band ids passed. The input raster has only "
                                + str(band_count)
                                + " bands"
                            )
                        for index, band in enumerate(raster[1]):
                            if band == 0:
                                continuous_variables.append(raster[0].name)
                                raster_field_variables.append(raster[0].name)
                            else:
                                continuous_variables.append(raster[0].name + f"_{band}")
                                raster_field_variables.append(
                                    raster[0].name + f"_{band}"
                                )
                    elif isinstance(raster[1], bool):
                        band_count = raster[0].band_count
                        if raster[1]:
                            for index in range(band_count):
                                if index == 0:
                                    categorical_variables.append(raster[0].name)
                                    raster_field_variables.append(raster[0].name)
                                else:
                                    categorical_variables.append(
                                        raster[0].name + f"_{index}"
                                    )
                                    raster_field_variables.append(
                                        raster[0].name + f"_{index}"
                                    )
                        else:
                            for index in range(band_count):
                                if index == 0:
                                    continuous_variables.append(raster[0].name)
                                    raster_field_variables.append(raster[0].name)
                                else:
                                    continuous_variables.append(
                                        raster[0].name + f"_{index}"
                                    )
                                    raster_field_variables.append(
                                        raster[0].name + f"_{index}"
                                    )
                    else:
                        raise Exception(
                            "The format of the raster variable passed is incorrect."
                        )
                elif len_tuple == 3:
                    if not isinstance(raster[1], bool):
                        raise Exception(
                            "The format of the raster variable passed is incorrect."
                        )
                    if not isinstance(raster[2], tuple):
                        raise Exception(
                            "The format of the raster variable passed is incorrect."
                        )
                    bands[i] = raster[2]
                    if raster[1]:
                        for index, band in enumerate(raster[2]):
                            if band == 0:
                                categorical_variables.append(raster[0].name)
                                raster_field_variables.append(raster[0].name)
                            else:
                                categorical_variables.append(
                                    raster[0].name + f"_{band}"
                                )
                                raster_field_variables.append(
                                    raster[0].name + f"_{band}"
                                )
                    else:
                        for index, band in enumerate(raster[2]):
                            if band == 0:
                                continuous_variables.append(raster[0].name)
                                raster_field_variables.append(raster[0].name)
                            else:
                                continuous_variables.append(raster[0].name + f"_{band}")
                                raster_field_variables.append(
                                    raster[0].name + f"_{band}"
                                )
            else:
                rasters.append(raster)
                band_count = raster.band_count
                for index in range(band_count):
                    if index == 0:
                        continuous_variables.append(raster.name)
                        raster_field_variables.append(raster.name)
                    else:
                        continuous_variables.append(raster.name + f"_{index}")
                        raster_field_variables.append(raster.name + f"_{index}")

        dataframe, index_data = TabularDataObject._process_layer(
            input_features,
            date_field,
            cell_sizes,
            distance_feature_layers,
            raster_variables,
            index_field,
            **kwargs,
        )
        measurer = np.vectorize(len)
        col_length = dict(
            zip(dataframe, measurer(dataframe.values.astype(str)).max(axis=0))
        )
        unique_values = {}
        for i in dataframe.columns:
            if i != "SHAPE":
                unique_values[i] = len(dataframe[i].unique())
        total_rows = dataframe.count().max()
        for col in categorical_variables:
            if unique_values[col] / total_rows > 0.5 and col_length[col] > 200:
                categorical_variables.remove(col)
                text_variables.append(col)
            elif (
                unique_values[col] / total_rows > 0.5
                and col_length[col] > 5
                and len(dataframe[col].iloc[0].split("\\")[0]) < 3
            ):
                categorical_variables.remove(col)
                image_variables.append(col)
            else:
                pass
        new_embd_cols = []
        if len(text_variables + image_variables) > 0:
            dataframe, new_embd_cols = _extract_embeddings(
                text_variables, image_variables, dataframe
            )
            # continuous_variables = continuous_variables + new_embd_cols
            # feature_field_variables = feature_field_variables + new_embd_cols

        dataframe_columns = dataframe.columns
        if distance_feature_layers:
            count = 1
            while f"DIST_{count}" in dataframe_columns:
                continuous_variables.append(f"DIST_{count}")
                count = count + 1

        if cell_sizes and not rasters:
            for res in cell_sizes:
                h3_field = f"zone{res}_id"
                if h3_field in dataframe_columns:
                    categorical_variables.append(h3_field)

        fields_to_keep = continuous_variables + categorical_variables + new_embd_cols
        if isinstance(dependent_variable, str):
            dependent_variable = [dependent_variable]

        # Changes introduced to capture dependent var as a list.
        if dependent_variable:
            fields_to_keep = fields_to_keep + dependent_variable

        try:
            for col in fields_to_keep:
                if not col in dataframe.columns:
                    msg = arcpy_localization_helper(
                        "Field does not exist within table", 728, "ERROR", str(col)
                    )
        except:
            pass

        for column in dataframe_columns:
            if column not in fields_to_keep:
                dataframe = dataframe.drop(column, axis=1)
            elif dependent_variable and column in dependent_variable:
                continue
            elif column in categorical_variables and dataframe[column].dtype == float:
                warnings.warn(f"Changing column {column} to continuous")
                categorical_variables.remove(column)
                continuous_variables.append(column)
            elif (
                column in categorical_variables
                and dataframe[column].unique().shape[0] > 20
            ):
                pass
                # warnings.warn(
                #    f"Column {column} has more than 20 unique value. Sure this is categorical?"
                # )

        if date_field:
            date_fields = [
                ("Year", True),
                ("Month", True),
                ("Week", True),
                ("Day", True),
                ("Dayofweek", True),
                ("Dayofyear", False),
                ("Is_month_end", True),
                ("Is_month_start", True),
                ("Is_quarter_end", True),
                ("Is_quarter_start", True),
                ("Is_year_end", True),
                ("Is_year_start", True),
                ("Hour", True),
                ("Minute", True),
                ("Second", True),
                ("Elapsed", False),
            ]
            for field in date_fields:
                if field[0] in dataframe_columns:
                    if field[1]:
                        categorical_variables.append(field[0])
                    else:
                        continuous_variables.append(field[1])

        return (
            dataframe,
            {
                "dependent_variable": dependent_variable,
                "categorical_variables": categorical_variables
                if categorical_variables
                else [],
                "continuous_variables": continuous_variables
                if continuous_variables
                else [],
                "text_variables": text_variables if text_variables else [],
                "image_variables": image_variables if image_variables else [],
                "embed_variables": new_embd_cols if new_embd_cols else [],
                "index_data": index_data,
                "feature_field_variables": feature_field_variables
                if feature_field_variables
                else [],
                "raster_field_variables": raster_field_variables
                if raster_field_variables
                else [],
            },
        )

    @staticmethod
    def _sdf_gptool_workflow(
        input_features,
        distance_feature,
        raster_list,
        index_field=None,
        is_table_obj=False,
    ):
        index_data = None
        data_source = None
        import arcpy
        import spatial_reference_helper
        import pandas as pd

        # if ((distance_feature) and (data_source)):F
        if not is_table_obj:
            if isinstance(input_features, tuple):
                input_features = input_features[0]
                data_source = str(input_features)
            else:
                try:
                    data_source = str(input_features)
                except:
                    data_source = input_features.dataSource
            count = 1
            for distance_layer in distance_feature:
                # field_1 = 'NEAR_FID_'+str(count)
                field_2 = "DIST_" + str(count)
                fields = [["NEAR_DIST", field_2]]
                arcpy.Near_analysis(data_source, distance_layer, field_names=fields)
                count = count + 1
            try:
                data_source_desc = arcpy.Describe(data_source)
                transformation = spatial_reference_helper.get_datum_transformation(
                    data_source_desc.spatialReference,
                    arcpy.SpatialReference(4326),
                    data_source_desc.extent,
                )
                sdf = pd.DataFrame.spatial.from_featureclass(
                    data_source, sr="4326", datum_transformation=transformation
                )
            except:
                sdf = sdf_from_table(input_features)
        else:
            sdf = sdf_from_table(input_features)

        rasters_data = {}
        if data_source:
            for cnt, raster in enumerate(raster_list):
                categorical = False
                if isinstance(raster, tuple):
                    if len(raster) == 2:
                        categorical = raster[1]
                    raster = raster[0]
                try:
                    sr = raster._engine_obj._raster.spatialReference
                except:
                    try:
                        import arcpy

                        sr = arcpy.SpatialReference(
                            raster.extent["spatialReference"]["wkid"]
                        )
                    except:
                        try:
                            import arcpy

                            sr = arcpy.SpatialReference(
                                raster.extent["spatialReference"]["wkt"]
                            )
                        except:
                            msg = arcpy_localization_helper(
                                "One or more input rasters do not have a valid spatial reference.",
                                517,
                                "ERROR",
                            )
                for i in range(raster.band_count):
                    if i == 0:
                        rasters_data[raster.name] = []
                    else:
                        rasters_data[raster.name + f"_{i}"] = []

                describe_obj = arcpy.Describe(data_source)
                if describe_obj.shapeType == "Polygon":
                    statistic = "MAJORITY" if categorical else "MEAN"
                    cached_oo = arcpy.env.overwriteOutput
                    arcpy.env.overwriteOutput = True
                    zonetable = arcpy.CreateTable_management("memory", "zonetable")

                    try:
                        arcpy.sa.ZonalStatisticsAsTable(
                            data_source,
                            describe_obj.oidFieldName,
                            raster.path + raster.name,
                            zonetable,
                            "DATA",
                            statistic,
                        )
                    except:
                        if statistic == "MAJORITY":
                            err_code = 110212
                            param = raster.name
                        else:
                            err_code = 10162
                            param = None

                        msg = arcpy_localization_helper(
                            "Pixel type of raster is float, which cannot be used as a categorical variable.",
                            err_code,
                            "ERROR",
                            param,
                        )
                        exit()

                    table_df = pd.DataFrame.spatial.from_table("memory\\zonetable")
                    arcpy.env.overwriteOutput = cached_oo

                    oidfield = table_df.columns[1]
                    new_index = pd.Index(np.arange(1, len(sdf) + 2, 1), name=oidfield)
                    rasdf = (
                        table_df.set_index(oidfield)
                        .reindex(new_index)
                        .reset_index()[statistic]
                    )
                    sdf[raster.name] = rasdf

                elif describe_obj.shapeType == "Point":
                    fields = ["SHAPE@X", "SHAPE@Y"]
                    with arcpy.da.SearchCursor(
                        data_source, fields, spatial_reference=sr
                    ) as cursor:
                        for row in cursor:
                            loc = "{locx} {locy}".format(locx=row[0], locy=row[1])
                            # print(u'{0}, {1}'.format(row[0], row[1]))
                            # if arcpy.Describe(data_source).shapeType == "Point":
                            try:
                                new_coordinate = _adjust_origin_coordinate(
                                    (row[0], row[1]),
                                    raster,
                                    (raster.mean_cell_width, raster.mean_cell_height),
                                )
                                raster_value = raster.read(
                                    origin_coordinate=(new_coordinate), ncols=1, nrows=1
                                )
                                value = raster_value[0][0]
                            except:
                                value = [np.NaN]
                            for i in range(len(value)):
                                if i == 0:
                                    rasters_data[raster.name].append(value[i])
                                else:
                                    rasters_data[raster.name + f"_{i}"].append(value[i])
                    for key, value in rasters_data.items():
                        sdf[key] = value
            if index_field in list(sdf.columns.values):
                index_data = sdf[index_field].values
        return sdf, index_data

    @staticmethod
    def _process_layer(
        input_features,
        date_field,
        cell_sizes,
        distance_layers,
        rasters,
        index_field,
        **kwargs,
    ):
        index_data = None
        attachment_list = kwargs.get("image_attach_list", None)
        if input_features is not None:
            if isinstance(input_features, FeatureLayer):
                import pandas as pd

                input_layer = input_features
                out_sr = None
                if cell_sizes and not rasters:
                    out_sr = 4326
                # sdf = input_features.query(out_sr=out_sr).sdf
                sdf = pd.DataFrame.spatial.from_layer(input_features)
                if attachment_list:
                    sdf["Images"] = attachment_list

            elif (
                hasattr(input_features, "dataSource")
                or str(input_features).endswith(".shp")
                or isinstance(input_features, tuple)
            ):
                sdf, index_data = TabularDataObject._sdf_gptool_workflow(
                    input_features,
                    distance_layers,
                    rasters,
                    index_field,
                    is_table_obj=False,
                )
                if attachment_list:
                    sdf["Images"] = attachment_list
                if cell_sizes and not rasters:
                    sdf = add_h3(sdf, cell_sizes)
                return sdf, index_data
            elif hasattr(input_features, "value"):
                sdf, index_data = TabularDataObject._sdf_gptool_workflow(
                    input_features,
                    distance_layers,
                    rasters,
                    index_field,
                    is_table_obj=True,
                )
                return sdf, index_data
            else:
                sdf = input_features.copy()
                if attachment_list:
                    sdf["Images"] = attachment_list
                input_layer = None
                try:
                    input_layer = sdf.spatial.to_feature_collection()
                except:
                    warnings.warn(
                        "Dataframe is not spatial, Rasters and distance layers will not work"
                    )

            if input_layer is not None and distance_layers:
                # Use proximity tool
                print("Calculating Distances.")
                count = 1
                for distance_layer in distance_layers:
                    output = arcgis.features.use_proximity.find_nearest(
                        input_layer, distance_layer, max_count=1
                    )
                    connecting_df = output["connecting_lines_layer"].query().sdf
                    near_dist = []

                    for i in range(len(connecting_df)):
                        near_dist.append(connecting_df.iloc[i]["Total_Miles"])

                    sdf[f"DIST_{count}"] = near_dist
                    count = count + 1

            # Process Raster Data to get information.
            rasters_data = {}

            if input_layer is not None:
                original_points = []
                for i in range(len(sdf)):
                    original_points.append(sdf.iloc[i]["SHAPE"])

                input_layer_spatial_reference = sdf.spatial.sr
                if cell_sizes and not rasters:
                    sdf = add_h3(sdf, cell_sizes)

                for raster in rasters:
                    raster_type = 0

                    raster_calc = TabularDataObject._mean_of

                    if isinstance(raster, tuple):
                        if isinstance(raster[1], bool):
                            if raster[1] is True:
                                raster_type = 1
                                raster_calc = TabularDataObject._majority_of
                            if len(raster) > 2:
                                raster_calc = TabularDataObject._get_calc(
                                    raster_type, raster[2]
                                )
                        else:
                            raster_calc = TabularDataObject._get_calc(
                                raster_type, raster[1]
                            )

                        raster = raster[0]

                    for i in range(raster.band_count):
                        if i == 0:
                            rasters_data[raster.name] = []
                        else:
                            rasters_data[raster.name + f"_{i}"] = []

                    shape_objects_transformed = arcgis.geometry.project(
                        original_points,
                        input_layer_spatial_reference,
                        raster.extent["spatialReference"],
                    )
                    for shape in shape_objects_transformed:
                        shape["spatialReference"] = raster.extent["spatialReference"]
                        if isinstance(shape, arcgis.geometry._types.Point):
                            try:
                                raster_value = raster.read(
                                    origin_coordinate=(shape["x"], shape["y"]),
                                    ncols=1,
                                    nrows=1,
                                )
                                value = raster_value[0][0]
                            except:
                                value = [0.0]
                        elif isinstance(shape, arcgis.geometry._types.Polygon):
                            xmin, ymin, xmax, ymax = shape.extent
                            start_x, start_y = (
                                xmin + (raster.mean_cell_width / 2),
                                ymin + (raster.mean_cell_height / 2),
                            )
                            values = []
                            while start_y < ymax:
                                while start_x < xmax:
                                    if shape.contains(
                                        arcgis.geometry._types.Point(
                                            {
                                                "x": start_x,
                                                "y": start_y,
                                                "sr": raster.extent["spatialReference"],
                                            }
                                        )
                                    ):
                                        raster_read = raster.read(
                                            origin_coordinate=(
                                                start_x - raster.mean_cell_width,
                                                start_y,
                                            ),
                                            ncols=1,
                                            nrows=1,
                                        )[0][0]
                                        if len(values) == 0:
                                            for band in raster_read:
                                                values.append([band])
                                        else:
                                            index = 0
                                            for band_value in raster_read:
                                                values[index].append(band_value)
                                                index = index + 1

                                    start_x = start_x + raster.mean_cell_width
                                start_y = start_y + raster.mean_cell_height
                                start_x = xmin + (raster.mean_cell_width / 2)

                            if len(values) == 0:
                                raster_read = raster.read(
                                    origin_coordinate=(
                                        shape.true_centroid["x"]
                                        - raster.mean_cell_width,
                                        shape.true_centroid["y"],
                                    ),
                                    ncols=1,
                                    nrows=1,
                                )[0][0]
                                for band_value in raster_read:
                                    values.append([band_value])

                            value = raster_calc(values)
                        else:
                            raise Exception(
                                "Input features can be point or polygon only."
                            )

                        for i in range(len(value)):
                            if i == 0:
                                rasters_data[raster.name].append(value[i])
                            else:
                                rasters_data[raster.name + f"_{i}"].append(value[i])

            # Append Raster data to sdf
            for key, value in rasters_data.items():
                sdf[key] = value
        else:
            try:
                import arcpy
            except:
                raise Exception("This function requires arcpy.")

            try:
                import pandas as pd
            except:
                raise Exception("This function requires pandas.")

            raster = rasters[0]
            if isinstance(raster, tuple):
                raster = raster[0]

            try:
                arcpy.env.outputCoordinateSystem = raster.extent["spatialReference"][
                    "wkt"
                ]
            except:
                arcpy.env.outputCoordinateSystem = raster.extent["spatialReference"][
                    "wkid"
                ]

            xmin = raster.extent["xmin"]
            xmax = raster.extent["xmax"]
            ymin = raster.extent["ymin"]
            ymax = raster.extent["ymax"]
            min_cell_size_x = raster.mean_cell_width
            min_cell_size_y = raster.mean_cell_height

            default_sr = raster.extent["spatialReference"]

            for raster in rasters:
                if isinstance(raster, tuple):
                    raster = raster[0]

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

            point_upper = arcgis.geometry.Point(
                {"x": xmin, "y": ymax, "sr": default_sr}
            )
            cell_size = arcgis.geometry.Point(
                {"x": min_cell_size_x, "y": min_cell_size_y, "sr": default_sr}
            )

            raster_data = {}
            for raster in rasters:
                if isinstance(raster, tuple):
                    raster = raster[0]
                field_name = raster.name
                point_upper_translated = arcgis.geometry.project(
                    [point_upper], default_sr, raster.extent["spatialReference"]
                )[0]
                cell_size_translated = arcgis.geometry.project(
                    [cell_size], default_sr, raster.extent["spatialReference"]
                )[0]
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

            sdf = pd.DataFrame.from_dict(raster_data)

        if date_field:
            try:
                add_datepart(sdf, date_field)
            except:
                pass

        if index_field in list(sdf.columns.values):
            index_data = sdf[index_field].values

        return sdf, index_data

    @staticmethod
    def _prepare_databunch(
        dataframe,
        fields_mapping,
        procs=None,
        validation_indexes=[],
        batch_size=64,
        is_classification=False,
    ):
        if procs is None:
            procs = [Categorify, Normalize]
            fm = FillMissing(
                fields_mapping["categorical_variables"],
                fields_mapping["continuous_variables"],
            )
            fm.add_col = False
            fm(dataframe)

        temp_file = tempfile.NamedTemporaryFile().name

        kwargs_variables = {"num_workers": 0} if sys.platform == "win32" else {}

        kwargs_variables["cat_names"] = fields_mapping["categorical_variables"]
        kwargs_variables["cont_names"] = fields_mapping["continuous_variables"]
        kwargs_variables["bs"] = batch_size

        if (
            hasattr(arcgis, "env")
            and getattr(arcgis.env, "_processorType", "") == "CPU"
        ):
            kwargs_variables["device"] = torch.device("cpu")

        sorted_dataframe = dataframe.copy()
        sorted_dataframe["sorted_index_col"] = range(0, len(dataframe))
        if is_classification:
            validation_indexes_sorted = sorted_dataframe.loc[validation_indexes][
                "sorted_index_col"
            ].to_list()
        else:
            validation_indexes_sorted = sorted_dataframe.iloc[validation_indexes][
                "sorted_index_col"
            ].to_list()
        del sorted_dataframe
        # CHanges to handle the pandas datatype issue
        dataframe[fields_mapping["categorical_variables"]] = dataframe[
            fields_mapping["categorical_variables"]
        ].astype("category")
        data_bunch = TabularDataBunch.from_df(
            temp_file,
            dataframe,
            fields_mapping["dependent_variable"],
            procs=procs,
            valid_idx=validation_indexes_sorted,
            **kwargs_variables,
        )

        return data_bunch

    @classmethod
    def _empty(
        cls,
        categorical_variables,
        continuous_variables,
        dependent_variable,
        encoder_mapping,
        procs=None,
        text_variables=None,
        image_variables=None,
        embedding_variables=None,
    ):
        class_object = cls()
        class_object._dependent_variable = dependent_variable
        class_object._continuous_variables = continuous_variables
        class_object._categorical_variables = categorical_variables
        class_object._text_variables = text_variables
        class_object._image_variables = image_variables
        class_object._embedding_variables = embedding_variables
        class_object._encoder_mapping = encoder_mapping
        class_object._is_empty = True
        class_object._procs = procs
        class_object.path = Path(os.path.abspath("."))

        return class_object


def explain_prediction(
    model, processed_df, index=0, random_index=False, predictor=None, global_pred=False
):
    tree_models = [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "AdaBoostClassifier",
        "AdaBoostRegressor",
        "BaggingClassifier",
        "BaggingRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "IsolationForest",
        "RandomTreesEmbedding",
        "StackingClassifier",
        "StackingRegressor",
        "VotingClassifier",
        "VotingRegressor",
        "HistGradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "ExtraTreeClassifier",
        "ExtraTreeRegressor",
        "LGBMRegressor",
        "LGBMClassifier",
        "XGBRegressor",
        "XGBClassifier",
        "CatBoostRegressor",
        "CatBoostClassifier",
    ]
    sklearn_regressors = [
        "LinearRegression",
        "Ridge",
        "RidgeCV",
        "SGDRegressor",
        "ARDRegression",
        "BayesianRidge",
        "PoissonRegressor",
        "TweedieRegressor",
        "GammaRegressor",
        "HuberRegressor",
        "RANSACRegressor",
        "TheilSenRegressor",
        "LinearSVR",
        "NuSVR",
        "SVR",
        "KernelRidge",
    ]
    sklearn_classifiers = [
        "LogisticRegression",
        "LogisticRegressionCV",
        "PassiveAggressiveClassifier",
        "Perceptron",
        "RidgeClassifier",
        "RidgeClassifierCV",
        "SGDClassifier",
        "LinearSVC",
        "NuSVC",
        "OneClassSVM",
        "SVC",
    ]
    if global_pred:
        if hasattr(model, "learn"):  # FCN
            global_interpretation(model, plot_type="bar", method="FCN")
        elif type(model._model).__name__ in sklearn_regressors:
            global_interpretation(model, plot_type="bar", method="KernelRegressor")
        elif type(model._model).__name__ in sklearn_classifiers:
            global_interpretation(model, plot_type="bar", method="KernelClassifier")
        elif type(model._model).__name__ in tree_models:
            global_interpretation(model, plot_type="bar", method="Tree")
        else:
            warnings.warn(
                "Feature importance plot cannot be generated for this model as it is not yet supported."
            )
            return
    else:
        if hasattr(model, "_model"):  # ML Models
            if type(model._model).__name__ in tree_models:
                show_local_interpretation(
                    model, processed_df, index, random_index, method="Tree"
                )
            elif type(model._model).__name__ in sklearn_regressors:
                show_local_interpretation(
                    model, processed_df, index, random_index, method="KernelRegressor"
                )
            elif type(model._model).__name__ in sklearn_classifiers:
                show_local_interpretation(
                    model, processed_df, index, random_index, method="KernelClassifier"
                )
            elif predictor == "Classifier":
                show_local_interpretation(
                    model, processed_df, index, random_index, method="KernelClassifier"
                )
            elif predictor == "Regressor":
                show_local_interpretation(
                    model, processed_df, index, random_index, method="KernelRegressor"
                )
            else:
                warnings.warn(
                    "Unrecognised Model: Explanation is not supported for this model yet!"
                )
        elif hasattr(model, "learn"):  # FCN
            show_local_interpretation(
                model, processed_df, index, random_index, method="FCN"
            )


def add_h3(sdf, cell_sizes):
    if "SHAPE" not in sdf.columns:
        return sdf
    if sdf["SHAPE"].iloc[0]["spatialReference"]["wkid"] == 4326:
        if (
            "polygon" in sdf.spatial.geometry_type
            or "point" in sdf.spatial.geometry_type
        ):
            try:
                sdf = point_to_h3(sdf, cell_sizes)
            except:
                warnings.warn("Cell sizes will not work.")
    return sdf


def point_to_h3(sdf, cell_sizes):
    import h3

    for res in cell_sizes:
        h3_id = []

        if "polygon" in sdf.spatial.geometry_type:
            for poly in sdf["SHAPE"]:
                centroid = poly.centroid
                h3_id.append(h3.geo_to_h3(centroid[1], centroid[0], res))

        elif "point" in sdf.spatial.geometry_type:
            h3_id = sdf["SHAPE"].apply(lambda x: h3.geo_to_h3(x["y"], x["x"], res))

        sdf[f"zone{res}_id"] = h3_id
    return sdf


def show_local_interpretation(
    model, processed_df, index=0, random_index=False, method="Tree"
):
    try:
        import shap
    except:
        raise Exception(traceback.format_exc())
    feature_variables = (
        model._data._categorical_variables + model._data._continuous_variables
    )
    if len(feature_variables) < 2:
        print(
            "Shap Explanation for prediction can be obtained only if number of explnatory variables > 1 "
        )
        return
    if method == "Tree":
        explainer = shap.TreeExplainer(model._model, algorithm="Tree")
    elif method == "KernelRegressor":
        if hasattr(model._data, "_training_indexes"):
            explainer = shap.KernelExplainer(
                model._model.predict, shap.sample(model._data._ml_data[0], 500)
            )
        else:
            warnings.warn(
                "To visualize the explanation of non tree models from sklearn, the model must be instantiated"
                " with the training data."
            )
            return
    elif method == "KernelClassifier":
        if hasattr(model._data, "_training_indexes"):
            explainer = shap.KernelExplainer(
                model._model.predict_proba,
                shap.sample(model._data._ml_data[0], 500),
                link="logit",
            )
        else:
            warnings.warn(
                "To visualize the explanation of non tree models from sklearn, the model must be instantiated"
                " with the training data."
            )
            return
    elif method == "FCN":
        if hasattr(model._data, "_training_indexes"):
            df = pd.DataFrame()
            for cnt, item in enumerate(
                np.random.choice(model._data._databunch.train_ds.x.items, 500)
            ):
                row = np.array(model._data._databunch.train_ds.x[item].data[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[cnt] = row
            explainer = shap.DeepExplainer(
                model.learn.model.layers, torch.tensor(df.transpose().values).cuda()
            )
        else:
            warnings.warn(
                "To visualize the explanation of fully connected network, the model must be instantiated"
                " with the training data."
            )
            return
    try:
        # feature_variables = model._data._feature_variables # If model is initialised with data
        feature_variables = (
            model._data._categorical_variables + model._data._continuous_variables
        )
    except:
        feature_variables = (
            processed_df.columns
        )  # In case when model is initialised only for prediction
    if random_index == False:
        # index_int=model._data._training_indexes.index(index)
        if method == "FCN":
            # train_tensor=torch.stack((torch.tensor(processed_df.iloc[[index]].to_numpy()).float(),
            #                          torch.tensor(processed_df.iloc[[index]].to_numpy()).float()),dim=0)
            processed_df = processed_df.iloc[[index]][feature_variables]
            train_tensor = torch.stack(
                (
                    torch.tensor(processed_df.values).squeeze().float(),
                    torch.tensor(processed_df.values).squeeze().float(),
                ),
                dim=0,
            )
        else:
            processed_df = processed_df.iloc[[index]]
            processed_numpy = model._data._process_data(
                processed_df.reindex(sorted(processed_df.columns), axis=1), fit=False
            )
    else:
        txt = (
            "The SHAP explanation is generated for {}th row of the dataframe. "
            "This row was randomly chosen since the parameter explain_index was passed as None"
        )
        if method == "FCN":
            processed_df = processed_df.sample(n=1)[feature_variables]
            print(txt.format(processed_df.index.values[0]))
            train_tensor = torch.stack(
                (
                    torch.tensor(processed_df.values).squeeze().float(),
                    torch.tensor(processed_df.values).squeeze().float(),
                ),
                dim=0,
            )
        else:
            processed_df = processed_df.sample(n=1)
            print(txt.format(processed_df.index.values[0]))
            processed_numpy = model._data._process_data(
                processed_df.reindex(sorted(processed_df.columns), axis=1), fit=False
            )
    # df_index = model._data._dataframe.iloc[index,:][feature_variables]
    if method == "Tree":
        shap_values = explainer.shap_values(processed_numpy)
        if isinstance(shap_values, list):
            shap.force_plot(
                explainer.expected_value[0],
                shap_values[0],
                processed_df,
                matplotlib=True,
            )
        else:
            shap.force_plot(
                explainer.expected_value, shap_values, processed_df, matplotlib=True
            )
    elif method == "KernelRegressor":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            shap_values = explainer.shap_values(processed_numpy, nsamples=100)
        if isinstance(shap_values, list):
            shap.force_plot(
                explainer.expected_value[0],
                shap_values[0],
                processed_df,
                matplotlib=True,
            )
        else:
            shap.force_plot(
                explainer.expected_value, shap_values, processed_df, matplotlib=True
            )
    elif method == "KernelClassifier":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            shap_values = explainer.shap_values(processed_numpy, nsamples=100)
        if isinstance(shap_values, list):
            shap.force_plot(
                explainer.expected_value[0],
                shap_values[0],
                processed_df,
                matplotlib=True,
                link="logit",
            )
        else:
            shap.force_plot(
                explainer.expected_value,
                shap_values,
                processed_df,
                matplotlib=True,
                link="logit",
            )
    elif method == "FCN":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            shap_values = explainer.shap_values(train_tensor)
        shap.force_plot(
            explainer.expected_value, shap_values[0], processed_df, matplotlib=True
        )


def _adjust_origin_coordinate(coordinate, raster, cell_size):
    import math

    x = coordinate[0]
    y = coordinate[1]
    xmin = raster.extent["xmin"]
    ymax = raster.extent["ymax"]
    dx = cell_size[0]
    dy = cell_size[1]
    x = math.floor((x - xmin) / dx)
    y = math.floor((ymax - y) / dy)
    xmin_new = xmin + x * dx
    ymax_new = ymax - y * dy
    return xmin_new, ymax_new


def sdf_from_table(input_features):
    try:
        import arcpy
    except:
        raise Exception("This method needs arcpy to be installed. Unable to continue")

    sdf = pd.DataFrame()
    data_type = arcpy.Describe(input_features).dataType
    if data_type in ["TableView", "TextFile"]:
        sdf = pd.DataFrame.spatial.from_table(str(input_features))
    if len(sdf) == 0:
        msg = arcpy_localization_helper(
            "Could not process the data. Your csv or table might contain columns with all null values. ",
            260200,
            "ERROR",
        )
    return sdf


def global_interpretation(model, plot_type="bar", method="KernelRegressor"):
    try:
        import shap
    except:
        raise Exception(traceback.format_exc())

    # explainer = shap.TreeExplainer(model._model)
    if hasattr(model._data, "_training_indexes"):
        feature_variables = (
            model._data._categorical_variables + model._data._continuous_variables
        )
        if len(feature_variables) < 2:
            print(
                "This method can be used only on datasets with more than 1 explanatory variables"
            )
            return
        df = pd.DataFrame(
            shap.sample(model._data._ml_data[0], 500), columns=feature_variables
        )
        if method == "KernelRegressor":
            explainer = shap.KernelExplainer(
                model._model.predict, shap.sample(model._data._ml_data[0], 100)
            )
        elif method == "Tree":
            explainer = shap.TreeExplainer(model._model)
        elif method == "KernelClassifier":
            explainer = shap.KernelExplainer(
                model._model.predict_proba,
                shap.sample(model._data._ml_data[0], 100),
                link="logit",
            )
        elif method == "FCN":
            df = pd.DataFrame()
            for cnt, item in enumerate(
                np.random.choice(model._data._databunch.train_ds.x.items, 500)
            ):
                row = np.array(model._data._databunch.train_ds.x[item].data[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[cnt] = row
            explainer = shap.DeepExplainer(
                model.learn.model.layers, torch.tensor(df.transpose().values).cuda()
            )
            feature_variables = (
                model._data._categorical_variables + model._data._continuous_variables
            )
            processed_df = df.transpose()
            processed_df.columns = feature_variables
            processed_df = processed_df.sample(n=1)[feature_variables]
            train_tensor = torch.stack(
                (
                    torch.tensor(processed_df.values).squeeze().float(),
                    torch.tensor(processed_df.values).squeeze().float(),
                ),
                dim=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                shap_values = explainer.shap_values(train_tensor)
            shap.summary_plot(shap_values, processed_df, plot_type="bar")
            return
    else:
        warnings.warn(
            "To visualize the explanation of non tree models from sklearn, the model must be instantiated"
            " with the training data."
        )
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        approximate = True
        if hasattr(model, "_model_type"):
            if model._model_type.startswith(
                "lightgbm."
            ) or model._model_type.startswith("catboost."):
                approximate = False
        shap_values = explainer.shap_values(df, approximate=approximate)
    if plot_type == "bar":
        return shap.summary_plot(shap_values, df, plot_type="bar")
    else:
        return shap.force_plot(
            explainer.expected_value[0], shap_values, df, matplotlib=True
        )
    return


def _extract_embeddings(text_variables, image_variables, dataframe):
    new_cols = []
    import tempfile
    from arcgis.learn import Embeddings

    for cnt1, var in enumerate(text_variables + image_variables):
        if var in text_variables:
            embeddings = Embeddings(dataset_type="text")
        else:
            embeddings = Embeddings(dataset_type="image")
        emb_array = embeddings.get(
            tempfile.TemporaryDirectory(),
            return_embeddings=True,
            dataframe=dataframe,
            text_column=var,
        )
        emb_list = emb_array.tolist()
        new_col_names = [
            "emb_" + str(cnt1) + "_" + str(cnt) for cnt in range(len(emb_list[0]))
        ]
        for col in new_col_names:
            new_cols.append(col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for cnt, val in enumerate(new_col_names):
                dataframe[val] = [value[cnt] for value in emb_list]
    return dataframe, new_cols
