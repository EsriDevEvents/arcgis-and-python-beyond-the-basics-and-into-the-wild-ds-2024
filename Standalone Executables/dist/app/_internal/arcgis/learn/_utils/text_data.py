import re
import os
import sys
import copy
import types
import shutil
import random
import logging
import warnings
import traceback
import pandas as pd
from functools import partial
from .._utils.common import always_warn

HAS_FASTAI = True
try:
    import torch
    import arcgis
    from fastai.text import (
        TextList,
        TextClasDataBunch,
        pad_collate,
        TextDataBunch,
        SortishSampler,
        SortSampler,
        ItemList,
        ItemBase,
        Text,
    )
    from fastai.data_block import CategoryList, MultiCategoryList
    from ._seq2seq_utils import SequenceToSequenceTextList, teacher_forcing_tfm
    from .text_transforms import (
        TransformerNERDataset,
        TransformerNERDataBunch,
        process_text,
    )
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False

HAS_BEAUTIFULSOUP = True
try:
    from bs4 import BeautifulSoup
except:
    HAS_BEAUTIFULSOUP = False
else:
    warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

HAS_NUMPY = True
try:
    import numpy as np

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except:
    HAS_NUMPY = False


max_len = 100


def _raise_fastai_exception(exception):
    error_message = (
        f"{exception}\n\nThis module requires fastai, PyTorch and transformers as its dependencies.\n"
        "Install them using - 'conda install -c esri -c fastai -c pytorch arcgis=1.8.2 "
        "scikit-image=0.15.0 pillow=6.2.2 libtiff=4.0.10 fastai=1.0.60 pytorch=1.4.0 "
        "torchvision=0.5.0 scikit-learn=0.23.1 --no-pin'"
        "\n'conda install gdal=2.3.3'"
        "\n'pip install transformers==3.3.0'"
    )
    raise Exception(error_message)


# Overriding show_text_xys function of TextList to display the dataframe in desired fashion
def show_text_xys(self, xs, ys, max_len: int = max_len):
    "Show the `xs` (inputs) and `ys` (targets). `max_len` is the maximum number of tokens displayed."
    if not HAS_NUMPY:
        raise Exception("This module requires numpy.")

    from IPython.display import display

    names = ["idx", "text"] if self._is_lm else ["text", "target"]
    items = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        txt_x = " ".join(x.text.split(" ")[:max_len]) if max_len is not None else x.text
        items.append([i, txt_x] if self._is_lm else [txt_x, y])
    items = np.array(items)
    df = pd.DataFrame({n: items[:, i] for i, n in enumerate(names)}, columns=names)
    dataframe_style = (
        df.style.set_table_styles([dict(selector="th", props=[("text-align", "left")])])
        .set_properties(**{"text-align": "left"})
        .hide(axis="index")
    )
    display(dataframe_style)


# Overriding show_text_xyzs function of TextList to display the dataframe in desired fashion
def show_text_xyzs(self, xs, ys, zs, max_len: int = max_len):
    "Show `xs` (inputs), `ys` (targets) and `zs` (predictions). `max_len` is the maximum number of tokens displayed."
    if not HAS_NUMPY:
        raise Exception("This module requires numpy.")

    from IPython.display import display

    items, names = [], ["text", "target", "prediction"]
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        txt_x = " ".join(x.text.split(" ")[:max_len]) if max_len is not None else x.text
        items.append([txt_x, y, z])
    items = np.array(items)
    df = pd.DataFrame({n: items[:, i] for i, n in enumerate(names)}, columns=names)
    dataframe_style = (
        df.style.set_table_styles([dict(selector="th", props=[("text-align", "left")])])
        .set_properties(**{"text-align": "left"})
        .hide(axis="index")
    )
    display(dataframe_style)


def read_file(path):
    filename, file_extension = os.path.splitext(path)
    if file_extension == ".csv":
        return pd.read_csv(path, dtype="str")
    else:
        return pd.read_csv(path, sep="\t", dtype="str")


def save_data_in_model_metrics_html(text, path, model_characteristics_folder):
    file_path = os.path.join(path, "model_metrics.html")
    if os.path.exists(file_path):
        with open(file_path, mode="a+", encoding="utf-8") as fp:
            fp.write(text)

    folder_path = os.path.join(path, model_characteristics_folder)
    if os.path.exists(folder_path):
        with open(
            os.path.join(folder_path, "sample_results.html"), mode="w", encoding="utf-8"
        ) as fp:
            fp.write(text)


def copy_metrics(source, target, folder_name):
    source_file_path = os.path.join(source, "model_metrics.html")
    target_file_path = os.path.join(target, "model_metrics.html")
    if os.path.exists(source_file_path):
        if os.path.exists(target_file_path):
            os.remove(target_file_path)
        _ = shutil.copyfile(source_file_path, target_file_path)

    source_folder_path = os.path.join(source, folder_name)
    target_folder_path = os.path.join(target, folder_name)
    if os.path.exists(source_folder_path):
        if os.path.exists(target_folder_path):
            shutil.rmtree(target_folder_path)
        _ = shutil.copytree(source_folder_path, target_folder_path)


def extract_entities(tokens, labels):
    prev_label, token_list, entities = labels[0], [tokens[0]], []
    prev_label = prev_label.split("-")[-1]
    for token_index, (token, label) in enumerate(list(zip(tokens[1:], labels[1:]))):
        label = label.split("-")[-1]
        if label == "O":
            if prev_label != "O":
                entities.append((token_list, prev_label))
            token_list, prev_label = list(), label

        if prev_label == label:
            token_list.append(token)
        else:
            if prev_label != "O":
                entities.append((token_list, prev_label))
            token_list = list()
            prev_label = label
            token_list.append(token)
    if token_list:
        entities.append((token_list, prev_label))
    return entities


class TextDataObject:
    def __init__(self, task):
        self.emd = {}
        self.emd_path = None

        self._bs = None
        self._is_empty = True
        self._train_df = None
        self._valid_df = None
        self._text_cols = None
        self._label_cols = list()
        self._databunch = None
        self._training_indexes = list()
        self._task = task
        self._backbone = None

        self.databunch_kwargs = {"num_workers": 0} if sys.platform == "win32" else {}
        self.databunch_kwargs["pin_memory"] = True
        if (
            getattr(arcgis.env, "_processorType", "") == "GPU"
            and torch.cuda.is_available()
        ):
            self.databunch_kwargs["device"] = torch.device("cuda")
        elif getattr(arcgis.env, "_processorType", "") == "CPU":
            self.databunch_kwargs["device"] = torch.device("cpu")
        else:
            self.databunch_kwargs["device"] = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

    @classmethod
    def prepare_data_for_entity_recognition(
        cls,
        tokens_collection,
        tags_collection,
        address_tag,
        unique_tags,
        seed=42,
        batch_size=8,
        val_split_pct=0.1,
        label2id=None,
    ):
        if not HAS_FASTAI:
            _raise_fastai_exception(import_exception)

        text_data = cls("ner")
        random.seed(seed)
        # temp = list(zip(tokens_collection, tags_collection))
        # random.shuffle(temp)
        # tokens_collection, tags_collection = zip(*temp)

        random.Random(seed).shuffle(tokens_collection)
        random.Random(seed).shuffle(tags_collection)

        # used when creating databunch for ner
        split_index = int((1 - val_split_pct) * len(tokens_collection))

        text_data._bs = batch_size
        text_data._unique_tags = unique_tags
        text_data._address_tag = address_tag
        text_data._label2id = (
            label2id if label2id else {x: index for index, x in enumerate(unique_tags)}
        )

        text_data._train_tags = tags_collection[:split_index]
        text_data._train_tokens = tokens_collection[:split_index]

        text_data._valid_tags = tags_collection[split_index:]
        text_data._valid_tokens = tokens_collection[split_index:]
        if text_data._address_tag not in text_data._unique_tags:
            logging.warning(
                "No Address tag found in your data.\n\
            1. If your data has an address field, pass your address field name as address tag in class mapping \n\
            e.g. - data=prepare_data(dataset_type=ds_type,path=training_data_folder,\n\t\t\t\
                class_mapping={address_tag:address_field_name})\n\
            2. Else no action is required, if your data does not have any address information."
            )

        return text_data

    @classmethod
    def prepare_data_for_classification(
        cls,
        data,
        text_cols,
        label_cols,
        train_file="train.csv",
        valid_file=None,
        val_split_pct=0.1,
        seed=42,
        batch_size=8,
        process_labels=False,
        remove_html_tags=False,
        remove_urls=False,
        **kwargs,
    ):
        if not HAS_FASTAI:
            _raise_fastai_exception(import_exception)

        text_data = cls("classification")
        if not os.path.exists(data):
            raise Exception(f"Provided data directory - {data}, does not exists")

        training_file_path = os.path.join(data, train_file)

        if not os.path.exists(training_file_path):
            raise Exception(
                f"Provided data directory does not contain {train_file} file"
            )

        train_df = read_file(training_file_path)
        train_df = cls._preprocess_df(
            train_df,
            text_cols,
            label_cols,
            process_labels,
            remove_html_tags,
            remove_urls,
        )

        random.seed(seed)

        if valid_file is not None and os.path.exists(os.path.join(data, valid_file)):
            validation_file_exists = True
        else:
            validation_file_exists = False

        if validation_file_exists:
            valid_df = read_file(os.path.join(data, valid_file))
            valid_df = cls._preprocess_df(
                valid_df,
                text_cols,
                label_cols,
                process_labels,
                remove_html_tags,
                remove_urls,
            )
        else:
            if len(label_cols) == 1 and kwargs.get("stratify") != False:
                label_col = label_cols[0]
                from sklearn.model_selection import train_test_split

                x, y = train_df[text_cols], train_df[label_col]
                unique_labels = y.value_counts()[y.value_counts() == 1].index.tolist()

                for (
                    label
                ) in unique_labels:  # duplicating datapoints with unique classes.
                    idx = y[y == label].index.tolist()[0]
                    train_df = train_df.append(train_df.iloc[idx])
                train_df.reset_index(drop=True, inplace=True)
                x, y = train_df[text_cols], train_df[label_col]
                X_train, X_test, y_train, y_test = train_test_split(
                    x, y, test_size=val_split_pct, stratify=y
                )
                train_df = pd.concat([X_train, y_train], axis=1)
                valid_df = pd.concat([X_test, y_test], axis=1)
                temp_df = pd.DataFrame()
            else:
                validation_indexes = random.sample(
                    range(train_df.shape[0]), round(val_split_pct * train_df.shape[0])
                )
                training_indexes = list(
                    set([i for i in range(train_df.shape[0])]) - set(validation_indexes)
                )

                temp_df = copy.deepcopy(train_df)
                train_df = temp_df.loc[training_indexes]
                valid_df = temp_df.loc[validation_indexes]
            # Removing rows with empty strings in the text_cols from the training and validation data
            train_df[text_cols].replace("", np.nan, inplace=True)
            train_df.dropna(inplace=True)
            valid_df[text_cols].replace("", np.nan, inplace=True)
            valid_df.dropna(inplace=True)
            # Resetting dataframe indexes for training anf validation dataframe
            train_df.reset_index(drop=True, inplace=True)
            valid_df.reset_index(drop=True, inplace=True)
            del temp_df
        if len(label_cols) > 1:
            text_data.classes = label_cols
        else:
            text_data.classes = valid_df[label_cols[0]].unique().tolist()
        text_data._bs = batch_size
        text_data._text_cols = text_cols
        text_data._label_cols = label_cols
        text_data._train_df = train_df[[text_cols] + label_cols]
        text_data._valid_df = valid_df[[text_cols] + label_cols]
        text_data._training_indexes = range(0, train_df.shape[0])

        return text_data

    @classmethod
    def prepare_data_for_seq2seq(
        cls,
        data,
        text_cols,
        label_cols,
        train_file="train.csv",
        val_split_pct=0.1,
        seed=42,
        batch_size=8,
        process_labels=True,
        remove_html_tags=False,
        remove_urls=False,
    ):
        if not HAS_FASTAI:
            _raise_fastai_exception(import_exception)

        text_data = cls("sequence_translation")
        if not os.path.exists(data):
            raise Exception(f"Provided data directory - {data}, does not exists")

        training_file_path = os.path.join(data, train_file)

        if not os.path.exists(training_file_path):
            raise Exception(
                f"Provided data directory does not contain {train_file} file"
            )

        train_df = read_file(training_file_path)
        train_df = cls._preprocess_df(
            train_df,
            text_cols,
            label_cols,
            process_labels,
            remove_html_tags,
            remove_urls,
        )

        random.seed(seed)

        validation_indexes = random.sample(
            range(train_df.shape[0]), round(val_split_pct * train_df.shape[0])
        )
        training_indexes = list(
            set([i for i in range(train_df.shape[0])]) - set(validation_indexes)
        )

        temp_df = copy.deepcopy(train_df)
        train_df = temp_df.loc[training_indexes]
        valid_df = temp_df.loc[validation_indexes]
        # Removing rows with empty strings in the text_cols from the training and validation data
        train_df[text_cols].replace("", np.nan, inplace=True)
        train_df.dropna(inplace=True)
        valid_df[text_cols].replace("", np.nan, inplace=True)
        valid_df.dropna(inplace=True)
        # Resetting dataframe indexes for training anf validation dataframe
        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        del temp_df
        text_data._bs = batch_size
        text_data._text_cols = text_cols
        text_data._label_cols = label_cols
        text_data._train_df = train_df[[text_cols] + label_cols]
        text_data._valid_df = valid_df[[text_cols] + label_cols]
        text_data.val_split_pct = val_split_pct
        text_data._training_indexes = range(0, train_df.shape[0])
        text_data._model_type = None
        return text_data

    def get_databunch(self):
        if self._is_empty:
            return None
        else:
            return self._databunch

    def _prepare_databunch(
        self, tokenizer, vocab=None, pad_first=True, pad_idx=0, **kwargs
    ):
        """
        Wrapper to create fastai TextDataBunch Object
        """
        if not HAS_FASTAI:
            _raise_fastai_exception(import_exception)

        logger = kwargs.get("logger")
        classes = kwargs.get("classes", None)
        backbone = kwargs.get("backbone")
        if classes:
            label_cols = self._label_cols
            unique_labels = (
                list(self._train_df[label_cols[0]].unique())
                if len(label_cols) == 1
                else label_cols
            )
            if set(classes) != set(unique_labels):
                error_message = (
                    f"Looks like the - `{backbone}` is fine-tuned on the following labels - "
                    f"{classes} and your data consists of the following labels - `{unique_labels}`."
                    f"\nPlease use a base model of - `{backbone}` and fine-tune it on your data or use a "
                    f"model which is fine-tuned on a data having same labels as - `{unique_labels}`"
                )
                raise Exception(error_message)

        if logger:
            logger.info(f"Preparing databunch for task type - {self._task}")
        if self._task == "classification":
            self._databunch = TextClasDataBunch.from_df(
                ".",
                train_df=self._train_df,
                valid_df=self._valid_df,
                tokenizer=tokenizer,
                vocab=vocab,
                include_bos=False,
                include_eos=False,
                text_cols=self._text_cols,
                label_cols=self._label_cols,
                bs=self._bs,
                collate_fn=partial(pad_collate, pad_first=pad_first, pad_idx=pad_idx),
                classes=classes,
                **self.databunch_kwargs,
            )

            TextList.show_xys = types.MethodType(show_text_xys, TextList)
            TextList.show_xyzs = types.MethodType(show_text_xyzs, TextList)
        elif self._task == "ner":
            model_type, seq_length = kwargs["model_type"], kwargs["seq_len"]
            dl_kwargs = {"pin_memory": self.databunch_kwargs.get("pin_memory")}
            device, num_workers = (
                self.databunch_kwargs["device"],
                self.databunch_kwargs.get("num_workers"),
            )
            if num_workers:
                dl_kwargs["num_workers"] = num_workers

            train_ds = TransformerNERDataset(
                self._train_tokens,
                self._train_tags,
                tokenizer,
                self._label2id,
                model_type=model_type,
                seq_length=seq_length,
            )
            valid_ds = TransformerNERDataset(
                self._valid_tokens,
                self._valid_tags,
                tokenizer,
                self._label2id,
                model_type=model_type,
                seq_length=seq_length,
            )

            self._databunch = TransformerNERDataBunch.create(
                train_ds,
                valid_ds,
                tokenizer,
                bs=self._bs,
                device=device,
                seq_len=seq_length,
                **dl_kwargs,
            )
        else:
            raise Exception(
                f"Wrong task - {self._task} selected. Allowed values are 'ner', 'classification'"
            )

        self._is_empty = False
        self._backbone = kwargs.get("backbone")

    def _prepare_seq2seq_databunch(
        self, transformer_processor, pad_first, pad_idx, **kwargs
    ):
        """
        Wrapper to create fastai TextDataBunch Object
        """
        self._model_type = kwargs.get("model_type")

        if not HAS_FASTAI:
            return
        dl_tfms = None
        if self._model_type in ["t5", "bart", "marian"]:
            dl_tfms = teacher_forcing_tfm
        data = (
            SequenceToSequenceTextList.from_df(
                self._train_df, cols=self._text_cols, processor=transformer_processor
            )
            .split_by_rand_pct(valid_pct=self.val_split_pct)
            .label_from_df(
                cols=self._label_cols,
                label_cls=TextList,
                processor=transformer_processor,
            )
            .databunch(
                bs=self._bs,
                pad_first=pad_first,
                pad_idx=pad_idx,
                dl_tfms=dl_tfms,
                **self.databunch_kwargs,
            )
        )

        self._is_empty = False
        self._databunch = data

        self._backbone = kwargs.get("backbone")

    @staticmethod
    def _preprocess_df(
        dataframe,
        text_cols,
        label_cols,
        process_labels,
        remove_html_tags=False,
        remove_urls=False,
    ):
        """
        Do some pre-processing on the dataframe columns
        """
        dataframe[text_cols].fillna("", inplace=True)
        dataframe[text_cols] = dataframe.apply(
            lambda row: TextDataObject.process_text(
                row[text_cols], remove_html_tags, remove_urls
            ),
            axis=1,
        )
        if process_labels:
            for label in label_cols:
                dataframe[label] = dataframe.apply(
                    lambda row: TextDataObject.process_text(
                        row[label], remove_html_tags, remove_urls
                    ),
                    axis=1,
                )

        return dataframe

    @staticmethod
    def process_text(
        text,
        remove_html_tags=False,
        remove_urls=False,
        remove_space_before_after_spl_char=False,
    ):
        """
        Perform some basic cleanup like removing HTML tags, removing urls,
        removing spaces before and after special characters
        converting multiple white spaces to single white space
        """
        if remove_urls:
            text = re.sub(
                r"\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*", " ", text
            )

        # text = re.sub(r'<.*?>', '', text)
        if remove_html_tags:
            if not HAS_BEAUTIFULSOUP:
                raise Exception("This module requires BeautifulSoup.")
            text = BeautifulSoup(text, "html.parser").get_text(
                separator=" ", strip=True
            )

        if remove_space_before_after_spl_char:
            text = re.sub(" ([@.#$/:-]) ?", r"\1", text)

        if any([remove_html_tags, remove_urls, remove_space_before_after_spl_char]):
            text = re.sub(" +", " ", text)
        return text.strip()

    def show_batch(self, rows=5, max_len=max_len):
        """
        Shows a batch of dataframe prepared without applying transforms.
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional integer. Number of rows in the
                                dataframe to be shown on the function call.
                                The default value is `5`.
        ---------------------   -------------------------------------------
        max_len                 Optional integer. Maximum number of tokens to be
                                shown for the text field (source column) of the
                                dataframe. The default value is `100`.
        =====================   ===========================================
        """
        if self._task == "classification":
            return self._classification_show_batch(rows=rows, max_len=max_len)
        elif self._task == "ner":
            return self._ner_show_batch(rows=rows)
        elif self._task == "sequence_translation":
            return self._classification_show_batch(rows=rows, max_len=max_len)
        else:
            raise Exception(
                f"Wrong task - {self._task} selected. Allowed values are 'ner', 'classification'"
            )

    def _ner_show_batch(self, rows=5):
        results = []
        indexes = list(range(len(self._train_tokens)))
        random.shuffle(indexes)
        rows = min(len(indexes), rows)
        batch_tokens = [self._train_tokens[i] for i in indexes[:rows]]
        batch_labels = [self._train_tags[i] for i in indexes[:rows]]
        for index, (tokens, labels) in enumerate(zip(batch_tokens, batch_labels)):
            entity_dict = dict()
            entities = extract_entities(tokens, labels)
            text = process_text(" ".join(tokens))
            entity_dict["Text"] = text
            _ = [
                entity_dict.setdefault(x[1], []).append(process_text(" ".join(x[0])))
                for x in entities
                if x[0]
            ]
            results.append(entity_dict)
        df = pd.DataFrame(
            results,
        )
        df.fillna("", inplace=True)
        return df

    def _classification_show_batch(self, rows=5, max_len=max_len):
        processed_data = []
        rows = min(len(self._training_indexes), rows)
        random_batch = random.sample(self._training_indexes, rows)
        # rows = min(len(self._train_df), rows)
        # random_batch = np.random.randint(0,self._train_df.index.max(),rows)
        dataframe = self._train_df.loc[random_batch]
        for idx, item in dataframe.iterrows():
            if len(self._label_cols) > 1:
                target = ";".join(
                    [
                        column
                        for column in self._label_cols
                        if int(getattr(item, column))
                    ]
                )
            else:
                target = getattr(item, self._label_cols[0])
            source = getattr(item, self._text_cols)
            if max_len is not None:
                source = " ".join(source.split(" "))[:max_len]
            processed_data.append([source, target])

            dataframe = pd.DataFrame(processed_data, columns=["source", "target"])

        return (
            dataframe.style.set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="index")
        )

    def create_empty_object_for_ner(
        self, entities, address_tag, label2id, batch_size=4
    ):
        self._bs = batch_size
        self._unique_tags = entities
        self._address_tag = address_tag
        self._label2id = label2id
        self._train_tags, self._train_tokens = [], []
        self._valid_tags, self._valid_tokens = [], []

    def create_empty_object_for_classification(
        self, text_cols, label_cols, classes, is_multilabel=False
    ):
        self._text_cols = text_cols
        self._label_cols = label_cols

        text_list = TextList([], ignore_empty=True).split_by_idx([])
        if is_multilabel:
            self._databunch = text_list.label_const(
                0, label_cls=MultiCategoryList, classes=classes
            ).databunch()
        else:
            self._databunch = text_list.label_const(
                0, label_cls=CategoryList, classes=classes
            ).databunch()

        self._is_empty = False

    def create_empty_seq2seq_data(self, text_cols, label_cols):
        self._text_cols = text_cols
        self._label_cols = label_cols

        self._databunch = (
            SequenceToSequenceTextList([]).split_none().label_const().databunch()
        )
        self._is_empty = False

        # self._train_df = self._create_empty_df(batch_size, [text_cols] + label_cols)
        # self._valid_df = self._create_empty_df(batch_size, [text_cols] + label_cols)

    # @staticmethod
    # def _create_empty_df(batch_size, columns):
    #     l1 = ["" for x in columns]
    #     l2 = [l1 for x in range(batch_size)]
    #     return pd.DataFrame(l2, columns=columns)
