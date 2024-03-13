import numpy as np


def multi_cat_reweighing(df, _sensitive_feature, _target):
    df["weight"] = 1

    total_count = df["weight"].sum()

    total_no_of_favours = np.count_nonzero(df[_target] == 1)
    total_no_of_unfavours = np.count_nonzero(df[_target] == 0)

    categories = df[_sensitive_feature].unique()
    weights = []

    for val in categories:
        total_no_of_cat = np.count_nonzero(df[_sensitive_feature] == val)
        total_no_of_cat_with_favour = df[
            (df[_sensitive_feature] == val) & (df[_target] == 1)
        ][_target].count()
        total_no_of_cat_with_unfavour = df[
            (df[_sensitive_feature] == val) & (df[_target] == 0)
        ][_target].count()

        w_of_cat_with_favour = (
            total_no_of_favours
            * total_no_of_cat
            / (total_count * total_no_of_cat_with_favour)
        )
        w_of_cat_with_unfavour = (
            total_no_of_unfavours
            * total_no_of_cat
            / (total_count * total_no_of_cat_with_unfavour)
        )

        weights.append((val, w_of_cat_with_favour, w_of_cat_with_unfavour))

    return weights


def assign_weights(_df, _target, _sensitive_feature):
    weights = multi_cat_reweighing(_df, _sensitive_feature, _target)
    for c, w_fav, w_unfav in weights:
        cond_p_fav = np.where((_df[_sensitive_feature] == c) & (_df[_target] == 1))[0]
        _df.iloc[cond_p_fav, -1] = w_fav

        cond_p_unfav = np.where((_df[_sensitive_feature] == c) & (_df[_target] == 0))[0]
        _df.iloc[cond_p_unfav, -1] = w_unfav

    return _df["weight"]
