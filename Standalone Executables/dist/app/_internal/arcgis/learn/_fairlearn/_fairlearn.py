from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    equalized_odds_difference,
    equalized_odds_ratio,
    demographic_parity_difference,
    demographic_parity_ratio,
    false_positive_rate,
    false_negative_rate,
    count,
    accuracy_score_ratio,
)
from fairlearn.reductions import (
    BoundedGroupLoss,
    ZeroOneLoss,
    DemographicParity,
    ErrorRate,
    GridSearch,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import warnings


def score(
    _is_classification,
    y_true,
    y_pred,
):
    if _is_classification:
        return accuracy_score(y_true, y_pred)
    else:
        return mean_absolute_error(y_true, y_pred)


def calculate_metrics(
    is_classification,
    data,
    y_true,
    y_pred,
    group_test,
    sensitive_feature,
    fairness_metrics,
    visualize,
):
    if not is_classification:
        fairness_ratio_threshold = 0.7
        fairness_diff_threshold = 0.01
        if fairness_metrics is None:
            fairness_metrics = "RMSE"

        return get_regression_metrics(
            data,
            y_true,
            y_pred,
            group_test,
            fairness_metrics,
            visualize,
            fairness_ratio_threshold,
            fairness_diff_threshold,
        )
    else:
        return show_classification_score(
            data,
            y_true,
            y_pred,
            group_test,
            sensitive_feature,
            fairness_metrics,
            visualize,
        )


def get_regression_metrics(
    _data,
    y_test,
    y_pred,
    df_sensitive_feature,
    metric_name,
    visualize=True,
    fairness_ratio_threshold=0.7,
    fairness_diff_threshold=0.01,
):
    for col in df_sensitive_feature.columns:
        mdf = get_mdf(
            _data,
            df_sensitive_feature,
            col,
            y_test,
            y_pred,
        )

        metric_name = metric_name.upper()

        privileged_value = mdf.index[mdf[metric_name][1:].argmax() + 1]
        underprivileged_value = mdf.index[mdf[metric_name][1:].argmin() + 1]

        metric_min = mdf[metric_name].loc[privileged_value]
        metric_max = mdf[metric_name].loc[underprivileged_value]

        ratio = np.round(metric_min / metric_max, 4)
        diff = np.round(metric_max - metric_min, 4)

        is_ratio_fair = False
        is_diff_fair = False

        fairness_metric_ratio = ratio
        if ratio > fairness_ratio_threshold:
            is_ratio_fair = True

        fairness_metric_diff = diff
        if diff < fairness_diff_threshold:
            is_diff_fair = True

        try:
            if visualize:
                fig, ax = plt.subplots()

                xlabels = mdf.index.to_list()
                yy = mdf.loc[:, metric_name]
                ax.bar(xlabels, yy, align="center", tick_label=xlabels)

                ax.set_facecolor("blanchedalmond")
                ax.set_alpha(0.7)
                plt.xticks(rotation=45)

                ax.set_xlabel(col)
                ax.set_ylabel(metric_name)

                plt.show()

        except Exception as ex:
            print(f"Exception occured : {ex}")

        fairness_metrics = {}

        fairness_metrics[col] = {
            "fairness_metric_name": metric_name,
            "fairness_metric_diff": fairness_metric_diff,
            "fairness_metric_ratio": fairness_metric_ratio,
            "is_ratio_fair": is_ratio_fair,
            "is_diff_fair": is_diff_fair,
        }

        return pd.DataFrame(fairness_metrics).T


def get_mdf(_data, sensitive_features, col, y_test, y_pred):
    regression_metrics = {
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": lambda t, p, sample_weight=None: np.sqrt(
            mean_squared_error(t, p, sample_weight=sample_weight)
        ),
        "R2": r2_score,
        "MAPE": mean_absolute_percentage_error,
    }

    overall = {}

    for k, v in regression_metrics.items():
        overall[k] = v(y_test, y_pred)

    values = sensitive_features[col].unique()
    _encoder = _data._encoder_mapping[col]
    ix = np.asarray(values.astype(int).tolist()).reshape(-1, 1)
    labels = np.unique(_encoder.inverse_transform(ix)).tolist()

    all_metrics = [overall]

    for value in values:
        metrics = {}
        for k, v in regression_metrics.items():
            metrics[k] = v(
                y_test[sensitive_features[col] == value],
                y_pred[sensitive_features[col] == value],
            )
        all_metrics += [metrics]

    mdf = pd.DataFrame(all_metrics, index=["Overall"] + labels)
    return mdf


def show_classification_score(
    data, y_true, y_pred, group_test, sensitive_feature, fairness_metrics, visualize
):
    metrics = {
        "accuracy": accuracy_score,
        "false positive rate": false_positive_rate,
        "false negative rate": false_negative_rate,
        "selection rate": selection_rate,
        "count": count,
    }

    fairness_dict = {
        "equalized_odds_difference": equalized_odds_difference,
        "demographic_parity_difference": demographic_parity_difference,
        "equalized_odds_ratio": equalized_odds_ratio,
        "demographic_parity_ratio": demographic_parity_ratio,
    }

    if fairness_metrics is None:
        fairness_metrics = fairness_dict.keys()
    else:
        fairness_metrics = [fairness_metrics]

    mf = MetricFrame(
        metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=group_test
    )
    res = mf.by_group

    _encoder = data._encoder_mapping[sensitive_feature]

    ix = np.asarray(res.index.astype(int).tolist())
    res.index = _encoder.inverse_transform(ix.reshape(-1, 1))

    if visualize:
        plot_accuracy_metrics(res)

    results = (res,)

    if visualize:
        fig = plt.figure(figsize=(12, 9))

    res_summary = {}

    for num, fm in enumerate(fairness_metrics):
        is_diff = False
        try:
            _metrics = fairness_dict[fm]

            if "diff" in fm:
                is_diff = True
                thre = 0.25
            else:
                thre = 0.8

            val = _metrics(y_true, y_pred, sensitive_features=group_test)
        except ZeroDivisionError as zerror:
            raise Exception(
                "One or more metrics count is zero. Please check the classification data."
            )
        val = round(val, 2)
        sum_text = get_text(fm, val, thre, is_diff)

        if visualize:
            try:
                ax = fig.add_subplot(2, 2, num + 1)
                ax.set_title(fm)

                plot_diff(fig, ax, val, thre, fm, sum_text, is_diff)
                fig.tight_layout(pad=4)
            except Exception as ex:
                warnings.warn(f"Visualization cannot be completed. {ex}")

        text_summary = " ".join(sum_text)
        res_summary[fm] = (val, text_summary)

    results = results + (res_summary,)

    return results


def show_regression_score(
    _data,
    y_true,
    y_pred,
    group_test,
    sensitive_feature,
    fairness_metrics,
    visualize=False,
):
    if fairness_metrics is None:
        fairness_metrics = "mean_absolute_error"

    mae_frame = MetricFrame(
        metrics=mean_absolute_error,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=group_test,
    )

    if visualize:
        _encoder = _data._encoder_mapping[sensitive_feature]

        res_idx = mae_frame.by_group.index
        ix = np.asarray(res_idx.astype(int).tolist()).reshape(-1, 1)
        labels = np.unique(_encoder.inverse_transform(ix)).tolist()

        plot_regression_metrics(
            mae_frame.by_group, labels, fairness_metrics, sensitive_feature
        )

    return mae_frame.by_group


def plot_regression_metrics(res, xlabels, metric_name, _sensitive_feature):
    fig, ax = plt.subplots()

    ax.bar(res.index, res.values, align="center", tick_label=xlabels)

    ax.set_facecolor("blanchedalmond")
    ax.set_alpha(0.7)
    plt.xticks(rotation=45)

    ax.set_xlabel(_sensitive_feature)
    ax.set_ylabel(metric_name)

    plt.show()


def plot_accuracy_metrics(res):
    _sum = res["count"].sum()

    res["count"] = res["count"].apply(lambda x: x / _sum)
    ax = res.plot(kind="bar", figsize=(6.4, 3.5))
    ax.axhspan(0, 1, facecolor="blanchedalmond", alpha=0.15)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.show()


def get_text(fm, val, thre, is_diff):
    texts = [f"The value of {fm} is {val}"]
    if val > thre:
        texts.append(f"which is more than minimum threshold {thre}.")
    else:
        texts.append(f"which is less than minimum threshold {thre}.")

    if is_diff:
        texts.append("The ideal value of this metric is 0.")
        texts.append(f"Fairness for this metric is between 0 and {thre}.")
    else:
        texts.append("The ideal value of this metric is 1.")
        texts.append(f"Fairness for this metric is between {thre} and 1.")
    return texts


def plot_diff(fig, ax, val, thre, label, summary, is_diff):
    text_summary = "\n".join(summary)

    ax.axhspan(-1, 1, facecolor="blanchedalmond", alpha=0.15)

    red_patch = mpatches.Patch(color="darksalmon", label="Biased")
    green_patch = mpatches.Patch(color="paleturquoise", label="Fair")

    if is_diff:
        ax.set_ylim(-1, 1)

        thre = 0.25
        if val > thre:
            ax.bar(
                [1, 2],
                [0, val],
                width=0.35,
                tick_label=["", label],
                color=("darksalmon"),
            )
        else:
            ax.bar(
                [1, 2],
                [0, val],
                width=0.35,
                tick_label=["", label],
                color=("paleturquoise"),
            )

        plt.axhline(y=0, color="grey", linestyle="-")
        plt.axhline(y=thre, color="grey", linestyle="--")
        plt.axhline(y=-thre, color="grey", linestyle="--")

        plt.text(2.72, 0, "Fair", fontsize=10)
        plt.text(2.72, thre, "Bias", fontsize=10)
        plt.text(2.72, -thre, "Bias", fontsize=10)

        ax.annotate(
            "",
            xy=(1.15, 0.30),
            xycoords="axes fraction",
            xytext=(1.15, 0.70),
            arrowprops=dict(arrowstyle="<->", color="black"),
        )

        text_widget = ax.text(
            1, -1.6, text_summary, fontsize=12, color="black", ha="left", va="bottom"
        )
        text_widget.set_fontstyle("italic")

        plt.legend(handles=[red_patch, green_patch], loc="upper left")

    else:
        ax.set_ylim(0, 1)

        thre = 0.8

        if val < thre:
            ax.bar(
                [1, 2],
                [0, val],
                width=0.35,
                tick_label=["", label],
                color=("darksalmon"),
            )
        else:
            ax.bar(
                [1, 2],
                [0, val],
                width=0.35,
                tick_label=["", label],
                color=("paleturquoise"),
            )

        plt.axhline(y=0, color="grey", linestyle="-")
        plt.axhline(y=thre, color="grey", linestyle="--")

        plt.text(2.72, 0, "Bias", fontsize=10)
        plt.text(2.72, thre, "Fair", fontsize=10)

        ax.annotate(
            "",
            xy=(1.15, 0.30),
            xycoords="axes fraction",
            xytext=(1.15, 0.70),
            arrowprops=dict(arrowstyle="<-", color="black"),
        )

        text_widget = ax.text(
            1, -0.4, text_summary, fontsize=12, color="black", ha="left", va="bottom"
        )
        text_widget.set_fontstyle("italic")

        plt.legend(handles=[red_patch, green_patch], loc="lower left")

    ax.set_xlim(1, 2.7)
    ax.set_xticks([])
