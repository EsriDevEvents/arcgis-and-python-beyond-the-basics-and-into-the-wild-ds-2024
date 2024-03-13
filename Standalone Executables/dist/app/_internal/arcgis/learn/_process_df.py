try:
    import pandas as pd
    import numpy as np
    import re
    import sklearn
    import datetime
    from sklearn_pandas import DataFrameMapper
    from sklearn.preprocessing import StandardScaler
    from pandas.api.types import is_string_dtype, is_numeric_dtype
    import logging
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split

    has_deps = True
except:
    has_deps = False


def _raise_dep_error():
    raise Exception(
        "This method requires pandas, numpy, scikit-learn. Install it using pip install pandas numpy scikit-learn"
    )


def add_datepart(df, col_name, drop=True, errors="raise"):
    """
    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    df                  Required DataFrame.
    col_name            Required String.
    drop                Optional Boolean. If True drop the original col_name.
    errors              Optional
    ===============     ====================================================================

    :return: None
    """
    if not has_deps:
        _raise_dep_error()
    col = df[col_name]
    col_dtype = col.dtype
    if isinstance(col_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        col_dtype = np.datetime64
    if not np.issubdtype(col_dtype, np.datetime64):
        df[col_name] = col = pd.to_datetime(col, errors=errors)
    attr = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]
    if df[col_name][0].time():
        attr = attr + ["Hour", "Minute", "Second"]
    for n in attr:
        df[col_name + n] = getattr(col.dt, n.lower())
    if drop:
        df.drop(col_name, axis=1, inplace=True)


def _scale(df, mapper=None):
    """
    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    df                  DataFrame to be scaled.
    mapper              Parameters used for scaling.
    ===============     ====================================================================

    :return: mapper if passed as None
    """
    if mapper is None:
        map_f = [([n], StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper


def process_df(
    df,
    target=None,
    do_scale=False,
    add_date_feats=False,
    mapper=None,
    test_sz=0.2,
):
    """
    This function preprocess the dataframe in following order :
    a. drops SHAPE column,
    b. creates target feature,
    c. fill missing values,
    d. scales the numerical features,
    e. one hot encode categorical features,
    f. train-test split,
    and returns the preprocessed dataframe, target variable, and mapper.
    ==============   ====================================================================
    **Parameter**        **Description**
    --------------   --------------------------------------------------------------------
    df               DataFrame to be preprocessed.
    --------------   --------------------------------------------------------------------
    y                Optional string. Response(or target) variable.
    --------------   --------------------------------------------------------------------
    do_scale         Optional Boolean. If True scales the numerical features else do not.
    --------------   --------------------------------------------------------------------
    mapper           Optional.  Works if do_scale is True, it contains the
                     parameters(mean and standard deviation) obtained from training set
                     and can be used for  test set.
    --------------   --------------------------------------------------------------------

    :return: training set features, test set features, train set targets, test set targets, np.ndarray, DataFrameMapper
    """
    if not has_deps:
        _raise_dep_error()

    if target is None:
        print("none")
        raise Exception("y(target variable) not found!!!!!")
    df = df.copy()
    for i in df:
        if str(df[i].dtype) == "geometry":
            shape = df.pop(i)
    if target is not None:
        if not is_numeric_dtype(df[target]):
            type_y = "cat"
            y = df.pop(target).astype("category").cat.codes
        else:
            type_y = "num"
            y = df.pop(target).values
    else:
        type_y = None
        y = None
    if df.isnull().sum().sum():
        df.fillna(df.median(), inplace=True)  # for numerical columns!!
        df.fillna(
            df.mode().iloc[0, :], inplace=True
        )  # for remaining categorical columns
    if add_date_feats:
        for i in df:
            if np.issubdtype(df[i].dtype, np.datetime64):
                add_datepart(df, i)
    if do_scale:
        mapper = _scale(df, mapper)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for i in cat_cols:
        if df[i].nunique() > 50:
            df.drop(i, inplace=True, axis=1)
            logging.warning(
                "feature {i} contains more than 50 categories, dropping!!".format(i=i)
            )
    df = pd.get_dummies(df)
    if type_y == "num":  # regression type
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, mapper
    elif type_y == "cat":  # when y is not None and category type.
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(df, y):
            X_train = df.loc[train_index]
            y_train = y[train_index]
            X_test = df.loc[test_index]
            y_test = y[test_index]
        return X_train, X_test, y_train, y_test, mapper
