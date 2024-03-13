try:
    import spacy
    import numpy as np
    import pandas as pd
    from spacy.util import minibatch, compounding
    from fastprogress.fastprogress import master_bar, progress_bar
    from .._utils.common import _get_emd_path
    from .._utils.text_data import copy_metrics
    from ..models._codetemplate import entity_recognizer_placeholder

    HAS_SPACY = True
except:
    HAS_SPACY = False

import json, logging
from pathlib import Path
import random, os
import datetime
from copy import deepcopy
from collections.abc import Iterable
import tempfile
from .._utils._ner_utils import even_mults, _timelapsed
from ..models._arcgis_model import ArcGISModel, _create_zip


class _SpacyEntityRecognizer(ArcGISModel):
    """
    Creates an entity recognition model to extract text entities from unstructured text documents.
    Based on Spacy's `EntityRecognizer <https://spacy.io/api/entityrecognizer>`_

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Requires data object returned from
                            :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    lang                    Optional string. Language-specific code,
                            named according to the language’s `ISO code <https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes>`_
                            The default value is 'en' for English.
    =====================   ===========================================

    :return: ``_SpacyEntityRecognizer`` Object
    """

    def __init__(self, data=None, lang="en", *args, **kwargs):
        super().__init__(data)
        self._code = entity_recognizer_placeholder
        self._emd_template = {}
        self.model_dir = None
        self.saved_model_dir = None
        self.model = spacy.blank(lang)
        self.ner = self.model.create_pipe("ner")
        self.model.add_pipe(self.ner, last=True)
        self._address_tag = "Address"  # Defines the default addres field
        self.entities = (
            None  # Stores all the entity names from the training data into a list
        )
        self._has_address = (
            False  # Flag to identify if the training data has any address
        )
        self._trained = False  # Flag to check if model has been trained
        self.lang = lang
        self.optimizer = self.model.begin_training()
        if data:
            self._is_empty = False
            self._address_tag = data._address_tag
            self._has_address = data._has_address
            self.path = data.working_dir
            self.data = data
            self.train_ds = data.train_ds
            self.val_ds = data.val_ds
            for ent in data.entities:
                if ent not in self.ner.labels:
                    self.model.entity.add_label(ent)
        else:
            self._is_empty = True
            self.train_ds = None
            self.val_ds = None
            self.path = "."
            self._is_empty = True
        self.learn = self
        self.recorder = Recorder()
        self.model_characteristics_folder = "ModelCharacteristics"

        pretrained_path = kwargs.get("pretrained_path", None)
        if pretrained_path is not None:
            pretrained_path = str(_get_emd_path(pretrained_path))
            self.load(pretrained_path)

    def lr_find(self, allow_plot=True):
        """
        Runs the Learning Rate Finder, and displays the graph of it's output.
        Helps in choosing the optimum learning rate for training the model.
        """
        self._check_requisites()

        start_lr = 1e-6
        end_lr = 10
        num_it = 10
        smoothening = 4

        with tempfile.TemporaryDirectory(prefix="arcgisTemp_") as _tempfolder:
            checkpoint_path = os.path.join(_tempfolder, "tmp")
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.mkdir(os.path.dirname(checkpoint_path))
            self.model.to_disk(checkpoint_path)  # caches the current model state
            if self._trained:
                temp_optimizer = (
                    self.optimizer
                )  # preserving the current state of the model for later load
            trained = (
                self._trained
            )  # preserving the current state of the model for later load
            recorder = deepcopy(
                self.recorder
            )  # preserving the current state of the model for later load
            self.recorder.losses, self.recorder.val_loss, self.recorder.lrs = (
                [],
                [],
                [],
            )  # resetting the recorder
            lrs = even_mults(start_lr, end_lr, 14)
            epochs = int(
                np.ceil(num_it / (len(self.data.train_ds) / self.data.batch_size))
            )
            self.fit(
                lr=list(lrs), epochs=epochs * len(lrs), from_lr_find=True, num_it=num_it
            )
            from IPython.display import clear_output

            clear_output()

            N = smoothening  # smoothening factor
            self.recorder.losses = np.convolve(
                self.recorder.losses, np.ones((N,)) / N, mode="valid"
            ).tolist()
            self.recorder.lrs = np.convolve(
                self.recorder.lrs, np.ones((N,)) / N, mode="valid"
            ).tolist()
            lr, index = self._find_lr(
                losses_skipped=0, trailing_losses_skipped=0, section_factor=2
            )

            if allow_plot:
                self._show_lr_plot(index, losses_skipped=0, trailing_losses_skipped=1)
            self._trained = trained
            self.recorder = recorder
            import spacy

            self.model = spacy.load(checkpoint_path)
        return lr

    def unfreeze(self):
        """
        Not implemented for this model.
        """
        logging.error(
            "unfreeze() is not implemented for EntityRecognizer model with spaCy backbone."
        )

    def freeze(self):
        """
        Not implemented for this model.
        """
        logging.error(
            "freeze() is not implemented for EntityRecognizer model with spaCy backbone."
        )

    def fit(
        self,
        epochs=20,
        lr=None,
        one_cycle=True,
        early_stopping=False,
        checkpoint=True,
        **kwargs,
    ):
        """
        Trains an EntityRecognition model for 'n' number of epochs..

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        epoch                   Optional integer. Number of times the model will train
                                on the complete dataset.
        ---------------------   -------------------------------------------
        lr                      Optional float. Learning rate
                                to be used for training the model.
        ---------------------   -------------------------------------------
        one_cycle               Not implemented for this model.
        ---------------------   -------------------------------------------
        early_stopping          Not implemented for this model.
        ---------------------   -------------------------------------------
        checkpoint              Not implemented for this model.
        =====================   ===========================================
        """
        self._check_requisites()
        if (
            lr is None
        ):  # searching for the optimal learning rate when no learning rate is provided
            print("Finding optimum learning rate")
            lr = self.lr_find(allow_plot=False)

        if kwargs.get("from_lr_find", False) is False and isinstance(lr, slice):
            lr = lr.stop
            error_message = (
                "Passing slice of floats as `lr` value is not supported for models with `spacy` backbone."
                f" Picking up the highest value - `{lr}` of the slice as the learning rate."
            )
            logging.warning(error_message)

        if self.train_ds is None:
            return logging.warning("Cannot fit the model on empty data.")

        TRAIN_DATA = self.train_ds.data
        VAL_DATA = self.val_ds.data
        nlp = self.model

        if (
            "ner" not in nlp.pipe_names
        ):  # create the built-in pipeline components and add them to the pipeline
            # spacy.require_gpu()
            self.ner = nlp.create_pipe(
                "ner"
            )  # nlp.create_pipe works for built-ins that are registered with spaCy
            nlp.add_pipe(self.ner, last=True)

        for _, annotations in TRAIN_DATA:  # adding labels
            for ent in annotations.get("entities"):
                if ent[2] not in self.ner.labels:
                    self.ner.add_label(ent[2])

        other_pipes = [
            pipe for pipe in nlp.pipe_names if pipe != "ner"
        ]  # get names of other pipes to disable them during training
        with nlp.disable_pipes(*other_pipes):  # only train NER
            if not nlp.vocab.vectors.name:
                nlp.vocab.vectors.name = "spacy_pretrained_vectors"

            batch_size = self.data.batch_size
            n_iter = len(TRAIN_DATA) // batch_size
            if (
                "from_lr_find" in kwargs
            ):  # 'from_lr_find' kwarg specifies that the fit is call from lr_find.
                epochs_per_lr = epochs / len(lr)
                n_iter = min(kwargs.get("num_it"), n_iter)
                lr_find = True
            else:
                self.optimizer.alpha = lr
                lr_find = False
            mb = master_bar(range(epochs))
            mb.write(
                [
                    "epoch",
                    "losses",
                    "val_loss",
                    "precision_score",
                    "recall_score",
                    "f1_score",
                    "time",
                ],
                table=True,
            )
            losses_list = []

            for itn in mb:
                t_start = datetime.datetime.now()
                if lr_find:
                    self.optimizer.alpha = lr.pop(0)
                    losses_list = []
                    update_recorder = True
                random.shuffle(TRAIN_DATA)
                batches = minibatch(TRAIN_DATA, size=batch_size)
                losses = {}
                epoch_loss = []
                for batch_index in progress_bar(range(n_iter), parent=mb):
                    batch_index += 1
                    batch = next(batches)
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts, annotations, sgd=self.optimizer, drop=0.35, losses=losses
                    )
                    processed_len = len(batch) * batch_index
                    train_loss = (
                        losses["ner"] / processed_len
                    )  # normalized with processed_len
                    if lr_find:
                        losses_list.append(train_loss)
                    else:  # recording training loss per iteration.
                        epoch_loss.append(train_loss)

                if not lr_find:
                    self.recorder.losses.append(
                        sum(epoch_loss) / n_iter
                    )  # averaging loss per epoch

                if VAL_DATA:
                    if lr_find:
                        VAL_DATA = VAL_DATA[
                            :batch_size
                        ]  # running on a subset of val data incase of lr_find
                    val_batches = minibatch(VAL_DATA, size=batch_size)
                    val_losses = {}
                    val_loss_list = []
                    epoch_loss = []
                    for batch_index, val_batch in enumerate(val_batches):
                        batch_index += 1
                        processed_len_val = batch_size * (batch_index)
                        val_text, val_annotations = zip(*val_batch)
                        nlp.update(
                            val_text, val_annotations, sgd=None, losses=val_losses
                        )
                        val_loss = (
                            val_losses["ner"] / processed_len_val
                        )  # normalized with processed_len_val
                        if lr_find:
                            val_loss_list.append(val_loss)
                        else:  # recording validation loss per iteration.
                            epoch_loss.append(val_loss)
                    if not lr_find:
                        self.recorder.val_loss.append(
                            sum(epoch_loss) / batch_index
                        )  # averaging loss per epoch

                if lr_find:
                    self.recorder.losses.append(np.mean(losses_list))
                    self.recorder.lrs.append(self.optimizer.alpha)
                    self.recorder.val_loss.append(np.min(val_loss_list))
                    update_recorder = False
                    # break the epoch if loss overshoots or all the lrs are tested
                    if (
                        np.mean(losses_list) > 2 * np.min(self.recorder.losses)
                        or len(lr) == 0
                    ):
                        return
                    score = nlp.evaluate(self.val_ds[:batch_size])
                else:
                    score = nlp.evaluate(self.val_ds)
                precision_score, recall_score, f1_score, metrics_per_label = (
                    score.ents_p,
                    score.ents_r,
                    score.ents_f,
                    score.ents_per_type,
                )
                self.recorder.metrics["precision_score"].append(precision_score)
                self.recorder.metrics["recall_score"].append(recall_score)
                self.recorder.metrics["f1_score"].append(f1_score)
                self.recorder.metrics["metrics_per_label"].append(metrics_per_label)
                line = [
                    itn,
                    round(train_loss, 2),
                    round(val_loss, 2),
                    round(precision_score / 100, 2),
                    round(recall_score / 100, 2),
                    round(f1_score / 100, 2),
                    _timelapsed(t_start),
                ]
                line = [str(val) for val in line]
                mb.write(line, table=True)

        if not lr_find:
            self._trained = True
            self.model = nlp
            self.entities = list(self.model.entity.labels)
            self.lr = lr

    def _create_emd(self, path, compute_metrics=True):
        path = Path(path)
        self._emd_template = {}
        self._emd_template["ModelConfiguration"] = "_ner"
        self._emd_template["InferenceFunction"] = "EntityRecognizer.py"
        self._emd_template["ModelFile"] = str(Path(path).name)
        self._emd_template["ModelName"] = type(self).__name__
        self._emd_template["Labels"] = self.model.entity.labels
        self._emd_template["Lang"] = self.lang
        self._emd_template["saved_path"] = str(Path(path))
        if hasattr(self, "lr"):
            self._emd_template["LearningRate"] = str(
                self.lr
            )  # checking if model has lr
        if compute_metrics and len(
            self.recorder.metrics["precision_score"]
        ):  # checking if recorder has metrics
            self._emd_template["metrics"] = json.dumps(
                {
                    "precision_score": [self.recorder.metrics["precision_score"][-1]],
                    "recall_score": [self.recorder.metrics["recall_score"][-1]],
                    "f1_score": [self.recorder.metrics["f1_score"][-1]],
                    "metrics_per_label": [
                        self.recorder.metrics["metrics_per_label"][-1]
                    ],
                }
            )
        if self._has_address:
            self._emd_template["address_tag"] = self._address_tag

        json.dump(
            self._emd_template,
            open(path / Path(path.stem).with_suffix(".emd"), "w"),
            indent=4,
        )
        pathstr = path / Path(path.stem).with_suffix(".emd")
        return path.resolve()

    def save(self, name_or_path, **kwargs):
        """
        Saves the model weights, creates an Esri Model Definition.
        Train the model for the specified number of epochs and using the
        specified learning rates.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Name of the model to save. It
                                stores it at the pre-defined location. If path
                                is passed then it stores at the specified path
                                with model name as directory name. and creates
                                all the intermediate directories.
        =====================   ===========================================
        """
        return self._save(name_or_path, **kwargs)

    def _save_model_characteristics(self, model_characteristics_dir):
        if not os.path.exists(model_characteristics_dir):
            os.makedirs(model_characteristics_dir)

        fig = self.plot_losses(show=False)
        if fig:
            fig.savefig(os.path.join(model_characteristics_dir, "loss_graph.png"))
        from IPython.utils import io

        with io.capture_output() as captured:
            self.show_results().to_html(
                os.path.join(model_characteristics_dir, "results.html")
            )
            metrics = self.metrics_per_label()
            if metrics is not None:  # expecting metrics = None for older saved models.
                metrics.to_html(
                    os.path.join(model_characteristics_dir, "metrics.html")
                )  # writing metrics to html

    def _save(
        self,
        name_or_path,
        zip_files=True,
        save_html=True,
        publish=False,
        gis=None,
        compute_metrics=True,
        **kwargs,
    ):
        temp = self.path
        if not self._trained:
            return logging.error("Model needs to be fitted, before saving.")

        if kwargs.get("save_optimizer", False):
            logging.warning(
                "Setting `save_optimizer` = True will not have any effect on models with `spaCy` backbone"
            )

        if "\\" in name_or_path or "/" in name_or_path:
            path = Path(name_or_path)
            parent_path = path.parent
            name = path.parts[-1]
            self.model_dir = parent_path / name
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        else:
            self.model_dir = Path(self.path) / "models" / name_or_path
            name = name_or_path
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

        self.model.to_disk(self.model_dir)
        emd_path = self._create_emd(self.model_dir, compute_metrics=compute_metrics)
        with open(self.model_dir / self._emd_template["InferenceFunction"], "w") as f:
            f.write(self._code)

        if save_html:
            if self._is_empty:
                copy_metrics(
                    self.saved_model_dir,
                    self.model_dir,
                    self.model_characteristics_folder,
                )
            else:
                self._save_model_characteristics(
                    self.model_dir.absolute() / self.model_characteristics_folder
                )
                self._create_html(Path(self.model_dir.absolute() / self.model_dir.stem))

        if zip_files:
            _create_zip(name, str(self.model_dir))

        if publish:
            self._publish_dlpk(
                (emd_path / emd_path.stem).with_suffix(".dlpk"),
                gis=gis,
                overwrite=kwargs.get("overwrite", False),
            )
        print(f"Model has been saved to {str(emd_path)}")
        return emd_path

    def load(self, name_or_path):
        """
        Loads a saved EntityRecognition model from disk.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Path of the emd file.
        =====================   ===========================================
        """
        if "\\" in str(name_or_path) or "/" in str(name_or_path):
            name_or_path = str(_get_emd_path(str(name_or_path)))
            name_or_path = name_or_path
            model_path = Path(name_or_path).parent
        else:
            model_path = Path(self.path) / "models" / name_or_path
            name_or_path = (
                Path(self.path) / "models" / name_or_path / f"{name_or_path}.emd"
            )

        with open(name_or_path, "r", encoding="utf-8") as f:
            emd = f.read()
        emd = json.loads(emd)
        address_tag = emd.get("address_tag")
        if address_tag:
            self._has_address = True
            self._address_tag = address_tag
        self.model = spacy.load(model_path)
        self.ner = self.model.get_pipe("ner")
        self._trained = True
        self.entities = list(self.model.entity.labels)
        self.model_dir = Path(name_or_path).parent.resolve()
        self.recorder = Recorder()
        if emd.get("metrics"):
            self.recorder.metrics = json.loads(emd.get("metrics"))
        self.saved_model_dir = deepcopy(self.model_dir)

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates an :class:`~arcgis.learn.text.EntityRecognizer` from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Esri Model Definition
                                file.
        ---------------------   -------------------------------------------
        data                    Required DatabunchNER object or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data` function or None for
                                inferencing.

        =====================   ===========================================

        :return: :class:`~arcgis.learn.text.EntityRecognizer` Object
        """
        emd_path = Path(emd_path)
        ner = cls(data=data)
        ner.load(emd_path)
        ner._trained = True
        ner.entities = list(ner.model.entity.labels)
        return ner

    def _post_process_non_address_df(self, unprocessed_df):
        """
        This function post processes the output dataframe from extract_entities function and returns a processed dataframe.
        """
        processed_df = pd.DataFrame(columns=unprocessed_df.columns)
        for col in unprocessed_df.columns:  # converting all list columns to string
            if (
                pd.Series(filter(lambda x: x != "", unprocessed_df[col]))
                .apply(isinstance, args=([str]))
                .sum()
                == 0
            ):  # split if this condition
                processed_df[col] = unprocessed_df[col].apply(
                    ",".join
                )  # join the list to string and copy to the processed df
            else:
                processed_df[col] = unprocessed_df[col]  # copy to the processed df
        return processed_df

    def _post_process_address_df(self, unprocessed_df, drop):
        """
        This function post processes the output dataframe from extract_entities function and returns a processed dataframe with cleaned up missed detections.
        """
        address_tag = self._address_tag
        processed_df = pd.DataFrame(
            columns=unprocessed_df.columns
        )  # creating an empty processed dataframe
        for i, adds in unprocessed_df[
            address_tag
        ].items():  # duplicating rows with multiple addresses to be one row per address
            if len(adds) > 0:  # adding data for address documents
                for j, add in enumerate(adds):
                    curr_index = len(processed_df)
                    processed_df.loc[curr_index] = unprocessed_df.loc[i]
                    processed_df.loc[curr_index, address_tag] = add
            else:  # adding data for non-address documents
                curr_index = len(processed_df)
                processed_df.loc[curr_index] = unprocessed_df.loc[i]
                processed_df.loc[curr_index, address_tag] = ""
        drop_ids = []

        for i, add in processed_df[address_tag].items():
            if len(add.split(" ")) < 2:
                drop_ids.append(i)
        del unprocessed_df

        if drop:  # flag for dropping/not-dropping documents without address.
            processed_df.drop(drop_ids, inplace=True)
        cols = processed_df.columns
        processed_df.reset_index(drop=True, inplace=True)

        for col in processed_df.columns:  # converting all list columns to string
            if (
                col != address_tag
                and pd.Series(filter(lambda x: x != "", processed_df[col]))
                .apply(isinstance, args=([str]))
                .sum()
                == 0
            ):  ## split if this condition
                processed_df[col] = processed_df[col].apply(
                    ",".join
                )  # join the list to strind and copy to the processed df
            else:
                processed_df[col] = processed_df[col]  # copy to the processed df
        return processed_df

    def _extract_entities_text(self, text):
        """
        This function extracts entities from a string"

        Arguments:
        text:str

        Returns:
        spacy's doc object

        Example of how to visualize the results:
        [(ent.label_, ent.text) for ent in doc_object.ents]
        """
        return self.model(text)

    def extract_entities(self, text_list, drop=True, show_progress=True, **kwargs):
        """
        Extracts the entities from [documents in the mentioned path or text_list].

        Field defined as 'address_tag' in :meth:`~arcgis.learn.prepare_data`  function's class mapping
        attribute will be treated as a location. In cases where trained model extracts
        multiple locations from a single document, that document will be replicated
        for each location in the resulting dataframe.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_list               Required string(path) or list(documents).
                                List of documents for entity extraction OR
                                path to the documents.
        ---------------------   -------------------------------------------
        drop                    Optional bool.
                                If documents without address needs to be
                                dropped from the results.
        ---------------------   -------------------------------------------
        batch_size              Optional integer. Number of items to process
                                at once. (Reduce it if getting CUDA Out of Memory
                                Errors). Default is set to 4.
                                Not applicable for models with `spaCy` backbone.
        =====================   ===========================================

        :return: Pandas DataFrame
        """

        if self._trained:
            if isinstance(text_list, list):
                item_list = pd.Series(text_list)

            elif isinstance(text_list, str):
                item_names = os.listdir(text_list)
                item_list = pd.Series()
                text = []
                skipped_docs = []
                for item_name in item_names:
                    try:
                        with open(
                            f"{text_list}/{item_name}",
                            "r",
                            encoding="utf-16",
                            errors="ignore",
                        ) as f:
                            item_list[item_name] = f.read()
                    except:
                        try:
                            with open(
                                f"{text_list}/{item_name}",
                                "r",
                                encoding="utf-8",
                                errors="ignore",
                            ) as f:
                                item_list[item_name] = f.read()
                        except:
                            skipped_docs.append(item_name)
                if len(skipped_docs):
                    print(
                        "Unable to read the following documents ",
                        ", ".join(skipped_docs),
                    )

            # if self._address_tag not in self.entities and self._has_address==True:
            #     return logging.warning('Model\'s address tag does not match with any field in your data, one of the below steps could resolve your issue:\n\
            #         1. Set address tag to the address field in your data [your_model._address_tag=\'your_address_field\']\n\
            #         2. If your data does not have any address field set _has_address=False [your_model._has_address=False]')
            data_list = []
            for i, item in progress_bar(list(item_list.items()), display=show_progress):
                doc = self._extract_entities_text(
                    item
                )  # predicting entities using entity_extractor model
                text = doc.text
                tmp_ents = {}
                for ent in doc.ents:  # Preparing a dataframe from results
                    if tmp_ents.get(ent.label_) == None:
                        tmp_ents[ent.label_] = [] + [ent.text]
                    else:
                        tmp_ents[ent.label_].extend([ent.text])

                tmp_ents["TEXT"] = text
                if isinstance(i, Iterable):  # For test documents
                    tmp_ents["Filename"] = i
                else:  # for show_results()
                    tmp_ents["Filename"] = "Example_" + str(i)

                data_list.append(tmp_ents)

            df = pd.DataFrame(data_list, columns=["TEXT", "Filename"] + self.entities)
            df.fillna("", inplace=True)
            if self._has_address:
                df = self._post_process_address_df(
                    df, drop
                )  # Post processing the dataframe
            else:
                df = self._post_process_non_address_df(
                    df
                )  # Post processing the dataframe
            return df.reset_index(drop="True")
        else:
            return logging.error("Model needs to be fitted, before extraction.")

    def show_results(self, ds_type="valid"):
        """
        Runs entity extraction on a random batch from the mentioned ds_type.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        ds_type                 Optional string, defaults to valid.
        =====================   ===========================================

        :return: Pandas DataFrame
        """

        if not self._trained:
            return logging.warning("This model has not been trained")
        """
        Make predictions on a batch of documents from specified ds_type.
        ds_type:['valid'|'train] 
        """
        # if self._address_tag not in self.entities and self._has_address == True:
        #     return logging.warning('Model\'s address tag does not match with any field in your data, one of the below steps could resolve your issue:\n\
        #         1. Set address tag to the address field in your data [your_model._address_tag=\'your_address_field\']\n\
        #         2. If your data does not have any address field set _has_address=False [your_model._has_address=False]')
        if self._is_empty:
            return logging.warning("This model does not have data.")

        if ds_type.lower() == "valid":
            xs = self.val_ds._random_batch(self.val_ds.x)
            return self.extract_entities(xs, show_progress=False)
        elif ds_type.lower() == "train":
            xs = self.train_ds._random_batch(self.train_ds.x)
            return self.extract_entities(xs, show_progress=False)
        else:
            print("Please provide a valid ds_type:['valid'|'train']")

    def precision_score(self):
        if self._trained:
            precision_pct = self.recorder.metrics["precision_score"][-1]
            precision = round(precision_pct / 100, 2)
            return precision
        else:
            return logging.warning("This model has not been trained")

    def recall_score(self):
        if self._trained:
            recall_pct = self.recorder.metrics["recall_score"][-1]
            recall = round(recall_pct / 100, 2)
            return recall
        else:
            return logging.warning("This model has not been trained")

    def f1_score(self):
        if self._trained:
            f1_pct = self.recorder.metrics["f1_score"][-1]
            f1 = round(f1_pct / 100, 2)
            return f1
        else:
            return logging.warning("This model has not been trained")

    def metrics_per_label(self):
        if self._trained:  # for saving old(before metrics were implemented) models.
            if not len(self.recorder.metrics["metrics_per_label"]):
                return None

            metrics_df = pd.DataFrame(
                self.recorder.metrics["metrics_per_label"][-1]
            ).transpose()
            metrics_df.columns = ["Precision_score", "Recall_score", "F1_score"]
            metrics_df = metrics_df.apply(lambda x: round(x / 100, 2))
            return metrics_df
        else:
            return logging.warning("This model has not been trained")

    def plot_losses(self, show=True):
        """
        Plot training and validation losses.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        show                    Optional bool. Defaults to True
                                If set to False, figure will not be plotted
                                but will be returned, when set to True function
                                will plot the figure and return nothing.
        =====================   ===========================================

        :return: `matplotlib.figure.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_
        """
        self._check_requisites()

        if not len(self.recorder.losses):  # return none if the recorder is empty
            raise Exception("Model needs to be fitted, before saving.")
        import numpy as np
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        N = max(1, len(self.recorder.losses) // 10)  # smooth with a factor of N
        ax.plot(
            np.convolve(self.recorder.losses, np.ones((N,)) / N, mode="valid"),
            label="Train",
        )
        ax.plot(
            np.convolve(self.recorder.val_loss, np.ones((N,)) / N, mode="valid"),
            label="Validation",
        )
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend()

        if not show:
            plt.close()
            return fig
        else:
            plt.show()

    def _check_requisites(self):
        if getattr(self, "_is_empty", False):
            raise Exception("Can't call this function without data.")


class Recorder:
    def __init__(self):
        self.lrs = []
        self.losses = []
        self.val_loss = []
        self.metrics = {
            "precision_score": [],
            "recall_score": [],
            "f1_score": [],
            "metrics_per_label": [],
        }
