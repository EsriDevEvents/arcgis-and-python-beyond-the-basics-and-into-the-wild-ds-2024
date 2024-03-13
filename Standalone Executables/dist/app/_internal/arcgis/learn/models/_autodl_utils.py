try:
    import arcgis as ag
    from fastai.basic_train import LearnerCallback
    import traceback
    import torch
    import matplotlib.pyplot as plt
    import pandas as pd
    import gc
    from IPython.display import clear_output
    from datetime import datetime as dt
    import io, os, base64
    import importlib
    import optuna
    import time
    import numpy as np

    from fastai.data_block import get_files
    from pathlib import Path
    from zipfile import ZipFile
    import tempfile

    HAS_FASTAI = True

    HAS_FASTAI = True
except Exception as e:
    print(e)
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False


class ToolIsCancelled(Exception):
    pass


self_obj = None
best_name_time = None
(
    all_val_losses,
    all_train_losses,
    dice,
    BestPerformingModel,
    best_backbone,
    best_model,
    timing,
) = ([], [], [], None, None, None, [])


class train_callback(LearnerCallback):
    def __init__(self, learn, stop_var):
        self.counter = 0
        self.stop_var = stop_var
        super().__init__(learn)

    def on_batch_end(self, **kwargs):
        # print(self.counter)
        self.counter += 1
        is_present = importlib.util.find_spec("arcpy")
        if is_present is not None:
            import arcpy

            if arcpy.env.isCancelled:
                raise ToolIsCancelled("Function aborted by User.")
        if self.counter > self.stop_var:
            self.counter = 0
            return {"stop_epoch": True}


def generate_output_report(
    df,
    output_folder,
    exhaustive_mode_studies,
    mode,
    save_to_folder,
    save_evaluated_models,
):
    content = ""
    header = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                
            .styled-table {
                border-collapse: collapse;
                font-size: 0.9em;
                font-family:Courier New;
            }

            .styled-table td, .styled-table th {
                border: 1px solid #ddd;
                padding: 8px;
            }

            .styled-table tr:nth-child(even){background-color: #f2f2f2;}

            .styled-table tr:hover {background-color: #e0ecf5;}

            .styled-table thead {
                padding-top: 6px;
                padding-bottom: 6px;
                text-align: left;
                background-color: #0099cc;
                color: white;
            }

            body {
                font-family: Arial;
                font-size: 1.0em;
                background-color: rgba(236, 243, 249, 0.15);
            }

            h1 {
                color: #004666;
                border-bottom: 1px solid rgba(0,70,102,0.3)
            }
            h2 {
                color: #004666;
                padding-bottom: 5px;
                margin-bottom: 0px;
            }

            ul {
                margin-top: 0px;
            }

            p {
                margin-top: 5px;
            }

            h3 {
                color: #004666;
                padding-bottom: 5px;
                margin-bottom: 0px;
            }
            a {
                font-weight: bold;
                color: #004666;
            }

            a:hover {
                cursor: pointer;
                color: #0099CC;
            }



                </style>
            </head>
            <body>
                <div id="main">
                <h1>AutoDL Leaderboard</h1>

        """

    footer = r"""
            <script>
                function toggleShow(elementId) {
                    var x = document.getElementById(elementId);
                    if (x.style.display === "none") {
                        x.style.display = "block";
                    } else {
                        x.style.display = "none";
                    }
                }
            </script>
        </body>
        </html>
    """
    # print(df)
    data_x = list(
        pd.Series(df.index).astype(str) + "_" + df["Model"] + "_" + df["backbone"]
    )[::-1]
    if "accuracy" in df.keys():
        data_y = list(df["accuracy"])[::-1]
        accuracy_fn = "accuracy"
    else:
        data_y = list(df["average_precision_score"])[::-1]
        accuracy_fn = "average_precision_score"

    fig = plt.figure(figsize=(20, 10))

    # creating the bar plot
    bars = plt.barh(data_x, data_y, color="#5ca3d0")
    # addlabels(data_x, data_y)
    # print(bars)

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            round(bar.get_width(), 2),
            size=14,
        )

    plt.ylabel("Model Names")
    plt.xlabel("accuracy")

    plt.title("AutDL Performance")
    # plt.show()

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format="png", dpi=400)
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    plt.close(fig)

    table = '<table class="styled-table">'

    table_head = f"<thead>\n<tr style='text-align: right;'>\n"
    if accuracy_fn == "average_precision_score":
        df_ = df[
            [
                "Model",
                "train_loss",
                "valid_loss",
                "average_precision_score",
                "lr",
                "training time",
                "optuna_study",
                "backbone",
            ]
        ]
    else:
        df_ = df[
            [
                "Model",
                "train_loss",
                "valid_loss",
                "accuracy",
                "dice",
                "lr",
                "training time",
                "optuna_study",
                "backbone",
            ]
        ]
    table_header_content = list(df_.keys())
    table_body_content = df_.values.tolist()
    # print(table_body_content)
    for thc in table_header_content:
        table_head += "<th style='text-align: left;'>" + thc + "</th>\n"

    table_head += "</tr>\n</thead>\n"

    table_content = "<tbody>"

    for i, tbc in enumerate(table_body_content, start=1):
        table_content += '<tr style="text-align: right;">\n'
        for j, data in enumerate(tbc, start=1):
            if j == 1:
                id_n = str(round(tbc[3], 6)).replace(".", "_")
                table_content += (
                    """<td style="text-align: left;"><u>
                            <a onclick="toggleShow(\'"""
                    + str(data)
                    + "_"
                    + tbc[-1]
                    + id_n
                    + """\');toggleShow('main')" >
                                """
                    + str(i)
                    + """_"""
                    + str(data)
                    + """
                            </u></a>
                        </td>\n"""
                )
            else:
                if j < len(tbc) - 2:
                    data = round(data, 6)
                table_content += (
                    """<td style="text-align: left;">
                                """
                    + str(data)
                    + """
                        </td>\n"""
                )

        table_content += "</tr>\n"
    table_content += "</tbody>\n"
    table = table + table_head + table_content + "</table> "

    content += table

    div_accuracy_chart = (
        """<h3>AutoDL model's accuracy graph</h3>
    <p><img style="width:1100px" alt="AutoDL Performance" src="data:image/png;base64,"""
        + pic_hash.decode("utf-8")
        + """" /></p>

    """
    )

    content += div_accuracy_chart

    # add model report

    if mode == "advanced":
        model_report = ""
        model_report += "<h2>Network wise study details</h2>"

        for i, ex in enumerate(exhaustive_mode_studies):
            model_report += "<hr/><h3>Network Name: " + ex.study_name + "</h3>"

            model_report += "<h4>Best parameter combination</h4>"
            best_params = ex.best_params
            for key, val in best_params.items():
                model_report += (
                    "<li><strong>" + str(key) + "</strong>: " + str(val) + "</li>"
                )

            try:
                img_details = optuna.importance.get_param_importances(ex)
                data_x = list(img_details.keys())
                data_y = list(img_details.values())

                fig = plt.figure(figsize=(12, 7))

                # creating the bar plot
                plt.bar(data_x, data_y, color="#5ca3d0")

                plt.xlabel("Parameter")
                plt.ylabel("importance percentage")
                plt.xticks(rotation=20)

                plt.title("Parameter Importance")
                # plt.show()

                pic_IObytes = io.BytesIO()
                fig.savefig(pic_IObytes, format="png", dpi=400)
                pic_IObytes.seek(0)
                pic_hash = base64.b64encode(pic_IObytes.read())
                plt.close(fig)
                model_report += "<h3>Parameter Importance</h3>"
                model_report += (
                    """
                    <p><img style="width:750px" alt="AutoDL Performance" src="data:image/png;base64,"""
                    + pic_hash.decode("utf-8")
                    + """" /></p>
                    """
                )
            except:
                pass

        if not save_to_folder:
            model_report += "<hr/><h2>Best Performing Model Report</h2>"
            folder_name = (
                "AutoDL_"
                + df["Model"][0]
                + "_"
                + df["backbone"][0]
                + "_"
                + df["timing"][0]
            )
            html_report = os.path.join(
                output_folder, "models", folder_name, "model_metrics.html"
            )

            model_metrics = open(html_report, "r")
            model_metrics_content = model_metrics.readlines()
            model_metrics.close()
            model_report += "\n".join(model_metrics_content)
        if save_to_folder:
            model_report += "<hr/><h2>Best Performing Model Report</h2>"
            html_report = os.path.join(output_folder, "model_metrics.html")
            model_metrics = open(html_report, "r")
            model_metrics_content = model_metrics.readlines()
            model_metrics.close()
            model_report += "\n".join(model_metrics_content)

        ## Ends
        # if not save_evaluated_models:
        #     model_report += "<br/><br/><p>* To display full model metrics enable Save Evaluated Models parameter.</p>"
        content += model_report

        content += "</div>"

        ## Hidden divs
        exhaustive_dfs = exhaustive_mode_studies

        for i, ex in enumerate(exhaustive_dfs):
            hidden_divs = ""
            name = ex.study_name
            best_params = ex.best_params
            ex_df = ex.trials_dataframe().sort_values("value", ascending=False)
            del ex_df["number"]
            del ex_df["datetime_start"]
            del ex_df["datetime_complete"]
            del ex_df["state"]
            for ind, dl in ex_df.iterrows():
                ids = name + "_" + dl["params_backbones"]
                hidden_divs += (
                    '<div id="'
                    + str(ids)
                    + str(round(dl["value"], 6)).replace(".", "_")
                    + '" style="display: none">'
                )
                hidden_divs += "<h1>Summary of " + str(name) + "</h1>"
                hidden_divs += (
                    """<p><a onclick="toggleShow(\'"""
                    + str(ids)
                    + str(round(dl["value"], 6)).replace(".", "_")
                    + """\');toggleShow('main')" >&lt;&lt; Go back</a></p>"""
                )
                hidden_divs += "<h2>Model name: <u>" + str(name) + "</u> </h2>"
                hidden_divs += "<h2>Model parameters</h2>"
                hidden_divs += "<ul>"
                for key, val in dl.items():
                    hidden_divs += (
                        "<li><strong>" + str(key) + "</strong>: " + str(val) + "</li>"
                    )

                hidden_divs += "</ul>"
                if save_evaluated_models:
                    hidden_divs += "<h2>Model Performance Report</h2>"
                    folder_name = (
                        "AutoDL_"
                        + name
                        + "_"
                        + dl["params_backbones"]
                        + "_"
                        + ex._timing[ind]
                    )
                    html_model_report = os.path.join(
                        output_folder, "models", folder_name, "model_metrics.html"
                    )
                    model_metrics = open(html_model_report, "r")
                    model_metrics_content = model_metrics.readlines()
                    model_metrics.close()
                    hidden_divs += "\n".join(model_metrics_content)

                if not save_evaluated_models:
                    hidden_divs += (
                        "<p>*To see model metrics enable Save Evaluated Models</p>"
                    )

                hidden_divs += "</div>"
            content += hidden_divs
        ## Hidden divs ends
    else:
        model_report = ""
        if not save_to_folder:
            model_report += "<hr/><h2>Best Performing Model Report</h2>"
            folder_name = (
                "AutoDL_"
                + df["Model"][0]
                + "_"
                + df["backbone"][0]
                + "_"
                + df["timing"][0]
            )
            html_report = os.path.join(
                output_folder, "models", folder_name, "model_metrics.html"
            )

            model_metrics = open(html_report, "r")
            model_metrics_content = model_metrics.readlines()
            model_metrics.close()
            model_report += "\n".join(model_metrics_content)
        if save_to_folder:
            model_report += "<hr/><h2>Best Performing Model Report</h2>"
            html_report = os.path.join(output_folder, "model_metrics.html")
            model_metrics = open(html_report, "r")
            model_metrics_content = model_metrics.readlines()
            model_metrics.close()
            model_report += "\n".join(model_metrics_content)
        content += model_report
        content += "</div>"
    display_html_path = os.path.join(output_folder, "README.html")

    html_template = header + content + footer

    f = open(display_html_path, "w")
    # html_template = header + content + hidden_divs + footer

    # writing the code into the file
    f.write(html_template)

    # close the file
    f.close()

    return html_template


def _get_model(model, **params):
    global self_obj
    self = self_obj
    return getattr(ag.learn, model)(self._data, **params)


def _objective(trial):
    global self_obj, temp_log_msg
    all_networks = self_obj._model_stats()
    params = {}
    for key, val in all_networks[trial.study.study_name]["params"].items():
        if key == "type_list":
            for k, v in val.items():
                params[k] = trial.suggest_categorical(k, v)
        if key == "type_float":
            for k, v in val.items():
                params[k] = trial.suggest_float(k, v[0], v[1])
        if key == "type_int":
            for k, v in val.items():
                params[k] = trial.suggest_int(k, v[0], v[1])

    if self_obj.verbose:
        log_msg = "{date}: selected params: {params}".format(
            date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
            params=params,
        )
        temp_log_msg.append(log_msg)

    # Get pretrained model
    model = _get_model(trial.study.study_name, **params)

    global all_val_losses, all_train_losses, dice, BestPerformingModel, best_backbone, best_model
    callbacks = [
        self_obj._train_callback(
            model.learn,
            self_obj._tiles_required // self_obj._data.batch_size,
        )
    ]
    model.fit(
        15,
        lr=trial.suggest_float("lr", 1e-4, 1e-2),
        early_stopping=True,
        callbacks=callbacks,
        checkpoint=False,
    )
    all_val_losses.append(model.learn.recorder.get_state()["val_losses"][-1])
    all_train_losses.append(float(model.learn.recorder.get_state()["losses"][-1]))
    if self_obj._dataset_type == "classification":
        dice.append(np.array(model.learn.recorder.get_state()["metrics"])[-1][1])
        accuracy = model.accuracy()
    else:
        avg_precision = model.average_precision_score()
        accuracy = sum(avg_precision.values()) / len(avg_precision.values())

    if accuracy >= self_obj._max_accuracy:
        self_obj._is_best = True
        self_obj._max_accuracy = accuracy
        self_obj.best_model = trial.study.study_name
        self_obj._best_backbone = params["backbones"]
        self_obj.BestPerformingModel = model
    best_name_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    ## Save the model
    if self_obj.verbose:
        log_msg = "{date}: Saving the model".format(
            date=dt.now().strftime("%d-%m-%Y %H:%M:%S")
        )
        print(log_msg)
        temp_log_msg.append(log_msg)

    # print(self_obj._save_evaluated_models)
    # name_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    global timing

    timing.append(best_name_time)
    if self_obj._save_evaluated_models:
        if self_obj._save_to_folder:
            model.save(
                self_obj._output_path
                + os.sep
                + "models"
                + os.sep
                + "AutoDL_"
                + trial.study.study_name
                + "_"
                + params["backbones"]
                + "_"
                + best_name_time
            )
            if self_obj.verbose:
                log_msg = "{date}: model saved at {path}".format(
                    date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                    path=os.path.join(
                        self_obj._output_path,
                        "models",
                        "AutoDL_"
                        + trial.study.study_name
                        + "_"
                        + params["backbones"]
                        + "_"
                        + best_name_time,
                    ),
                )
                print(log_msg)
                temp_log_msg.append(log_msg)
        else:
            model.save(
                "AutoDL_"
                + trial.study.study_name
                + "_"
                + params["backbones"]
                + "_"
                + best_name_time
            )
            if self_obj.verbose:
                log_msg = "{date}: model saved at {path}".format(
                    date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                    path=os.path.join(
                        self_obj._data.path,
                        "models",
                        "AutoDL_"
                        + trial.study.study_name
                        + "_"
                        + params["backbones"]
                        + "_"
                        + best_name_time,
                    ),
                )
                print(log_msg)
                temp_log_msg.append(log_msg)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return accuracy


def _train_exhaust_mode(self, time, name, val_losses, train_losses, d):
    global temp_log_msg, self_obj, all_val_losses, all_train_losses, dice, timing, best_name_time

    self_obj = self
    timing = []
    all_val_losses, all_train_losses, dice = val_losses, train_losses, d
    temp_log_msg = []

    sampler = optuna.samplers.TPESampler()
    if self_obj.verbose:
        log_msg = "{date}: A new study created in memory with name: {networks}".format(
            date=dt.now().strftime("%d-%m-%Y %H:%M:%S"),
            networks=name,
        )
        self_obj._logger_dict.append(log_msg)
        clear_output(wait=True)
        all_logs = "\n".join(self_obj._logger_dict)
        print(all_logs)
    study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=3, interval_steps=2
        ),
        direction="maximize",
        study_name=name,
    )
    study.optimize(
        func=_objective,
        n_trials=None,
        show_progress_bar=True,
        timeout=time,
        gc_after_trial=True,
    )
    if self_obj.verbose:
        self_obj._logger_dict.append("\n".join(temp_log_msg))
        clear_output(wait=True)
        all_logs = "\n".join(self_obj._logger_dict)
        print(all_logs)
    return study, self_obj, all_val_losses, all_train_losses, dice, timing


def _get_emd_path(emd_path):
    emd_path = Path(emd_path)
    if emd_path.suffix == ".dlpk":
        temp_path = _temp_dlpk(emd_path)
        emd_path = Path(temp_path)
        # return cls.from_model(temp_path)

    if emd_path.suffix != ".emd":
        list_files = get_files(emd_path, extensions=[".emd"])
        assert len(list_files) == 1
        # return cls.from_model(list_files[0])
        emd_path = list_files[0]
    return emd_path


def _temp_dlpk(dlpk_path):
    with ZipFile(dlpk_path, "r") as zip_obj:
        temp_dir = tempfile.TemporaryDirectory().name
        zip_obj.extractall(temp_dir)
    return temp_dir
