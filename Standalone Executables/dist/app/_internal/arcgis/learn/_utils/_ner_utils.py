import os
import json
import random
import logging
import tempfile
import datetime
import traceback
from pathlib import Path
from .text_data import TextDataObject

try:
    import spacy
    from spacy.gold import offsets_from_biluo_tags as _offsets_from_biluo_tags
    from spacy.gold import iob_to_biluo as _iob_to_biluo
    import pandas as pd
    import numpy as np

    HAS_SPACY = True
except Exception as e:
    spacy_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_SPACY = False


__all__ = ["_from_iob_tags", "_from_json", "_NERData", "even_mults", "_timelapsed"]


def _raise_spacy_import_error():
    error_message = (
        f"{spacy_exception}\n\n\n"
        "This module requires spacy version 2.1.8 or above and fastprogress."
        "Install it using 'pip install spacy==2.1.8 fastprogress pandas'"
    )
    raise Exception(error_message)


def even_mults(
    start: float, stop: float, n: int
):  # Taken from FastAI(https://github.com/fastai/fastai/blob/master/fastai/core.py#L150)
    "Build log-stepped array from `start` to `stop` in `n` steps."
    mult = stop / start
    step = mult ** (1 / (n - 1))
    return np.array([start * (step**i) for i in range(n)])


def _timelapsed(t_start):
    """returns timedelta in hh:mm:ss format"""
    b = datetime.datetime.now() - t_start
    h, r = divmod(b.seconds, 3600)
    m, s = divmod(r, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def _from_iob_tags(tokens_collection, tags_collection):
    """
    Converts training data from ``IOB`` format to spacy offsets.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    tokens_collection       Required [list]. List of token lists
                            Example: [[This,is,a,test],[This,is,a,test1]]
    ---------------------   -------------------------------------------
    tags_collection         Required [list]. List of tag lists
                            Example: [[B-tagname,O,O,O],[B-tagname,I-tagname,O,O]]
    =====================   ===========================================
    """

    train_data = []
    nlp = spacy.blank("en")
    for tags, tokens in zip(tags_collection, tokens_collection):
        try:
            tags = _iob_to_biluo(tags)

            doc = spacy.tokens.doc.Doc(
                nlp.vocab, words=tokens, spaces=[True] * (len(tokens) - 1) + [False]
            )
            # run the standard pipeline against it
            for name, proc in nlp.pipeline:
                doc = proc(doc)

            text = " ".join(tokens)
            tags = _offsets_from_biluo_tags(doc, tags)
            train_data.append((text, {"entities": tags}))
        except:
            pass

    return train_data


def _from_json(path, text_key="text", offset_key="labels", encoding="UTF-8"):
    """
    Converts training data from JSON format to spacy offsets.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    text_key                Optional:str='text. Json key under which text is available
    ---------------------   -------------------------------------------
    offset_key              Optional:str='labels. Json key under which offsets are available
    =====================   ===========================================
    json-schema:
    ----------
    {"id": 1, "text": "EU rejects ...", "labels": [[0,2,"ORG"], [11,17, "MISC"], [34,41,"ORG"]]}
    {"id": 2, "text": "Peter Blackburn", "labels": [[0, 15, "PERSON"]]}
    {"id": 3, "text": "President Obama", "labels": [[10, 15, "PERSON"]]}
    ----------
    returns: A list that can be consumed by ner_databunch.
    """

    train_data = []
    with open(path, "r", encoding=encoding) as f:
        data_list = f.readlines()
    for i, item in enumerate(data_list):
        try:
            execute_second_exeption = False
            train_data.append(
                (json.loads(item)[text_key], {"entities": json.loads(item)[offset_key]})
            )
        except KeyError as key:
            _key = key
            execute_second_exeption = True
        if execute_second_exeption:
            try:
                train_data.append(
                    (json.loads(item)["data"], {"entities": json.loads(item)["label"]})
                )
            except KeyError as key:
                raise Exception(
                    f"{_key} key not present in record {i} of input json file."
                )

    return train_data


def _get_tags_and_tokens_collection(path, ignore_tag_order=False, encoding="UTF-8"):
    unique_tags = set()
    tags_collection, tokens_collection = [], []
    tags_df = pd.read_csv(path / "tags.csv", encoding=encoding, dtype="str")
    tokens_df = pd.read_csv(path / "tokens.csv", encoding=encoding, dtype="str")

    for i, tags in tags_df.iterrows():
        if ignore_tag_order:
            tags = [x.split("-")[-1] for x in tags.dropna()]
        else:
            tags = [x for x in tags.dropna()]
        unique_tags.update(tags)
        tags_collection.append(tags)

    for i, tokens in tokens_df.iterrows():
        tokens_collection.append(list(tokens.dropna()))

    return tags_collection, tokens_collection, unique_tags


class _NERData:
    working_dir = None
    """
    #     Prepares a data object
    #
    #     =====================   ===========================================
    #     **Parameter**            **Description**
    #     ---------------------   -------------------------------------------
    #     dataset_type            Required string. ['ner_json', 'IOB', 'BILUO']
    #     ---------------------   -------------------------------------------
    #     address_tag             Optional dict. Address field/tag name
    #                             in the training data.
    #     ---------------------   -------------------------------------------
    #     val_split_pct           Optional Float. Percentage of training data to keep
    #                             as validation. The default value is 0.1.
    #     =====================   ===========================================
    #     returns: A list [text,{entities},text,{entities}] that can be ingested by ``EntityRecognizer``.
    """

    def __init__(
        self,
        dataset_type,
        path,
        batch_size,
        class_mapping=None,
        seed=42,
        val_split_pct=0.1,
        encoding="UTF-8",
    ):
        self.dataset_type = dataset_type
        self.path = path
        self.data = None
        self.backbone = None
        self.batch_size = batch_size
        self.class_mapping = class_mapping
        self.seed = seed
        self.val_split_pct = val_split_pct
        self.encoding = encoding
        self.prepare_data_for_spacy()

    def show_batch(self):
        return self.data.show_batch()

    def get_data_object(self):
        data = self.data
        if self.working_dir is None:
            path = getattr(data, "path", self.path)
            if os.path.isfile(path):
                path = Path(os.path.dirname(os.path.abspath(path)))
            data.working_dir = path
            from .._data import _prepare_working_dir

            _prepare_working_dir(path)
        else:
            data.working_dir = self.working_dir

        return data

    def prepare_data_for_transformer(self, ignore_tag_order=True, label2id=None):
        unique_tags = set()
        path = Path(self.path)
        tags_collection, tokens_collection = [], []
        if self.class_mapping:
            address_tag = self.class_mapping.get("address_tag")
        else:
            address_tag = "Address"

        if self.dataset_type == "ner_json":
            # converts json schema to a list of list of tokens and tags
            # json - schema:
            # ----------
            # {"id": 1, "text": "Officers were dispatched ...", "labels": [[30, 38, "Crime"], [45, 92, "Address"], ...]}
            # ----------
            # converted form:
            # [('Officers were dispatched to a', 'O'),
            #  ('robbery', 'Crime'),
            #  ('of the', 'O'),
            #  ('Associated Bank in the 1500 block of W Broadway', 'Address')]
            data_list = _from_json(path=path, encoding=self.encoding)
            for i, row in enumerate(data_list):
                prev_start = 0
                tmp_tags_list, tmp_tokens_list = [], []
                text, labels = row[0], row[1]["entities"]
                for item in sorted(labels, key=lambda x: x[0]):
                    c_text = text[prev_start : item[0]].strip()
                    tmp_tags_list.append("O")
                    unique_tags.add("O")
                    tmp_tokens_list.append(c_text)
                    c_text = text[item[0] : item[1]].strip()
                    tmp_tags_list.append(item[2])
                    tmp_tokens_list.append(c_text)
                    unique_tags.add(item[2])
                    prev_start = item[1]

                tags_collection.append(tmp_tags_list)
                tokens_collection.append(tmp_tokens_list)

        elif self.dataset_type in ["BIO", "IOB", "LBIOU", "BILUO"]:
            (
                tags_collection,
                tokens_collection,
                unique_tags,
            ) = _get_tags_and_tokens_collection(
                path, ignore_tag_order=ignore_tag_order, encoding=self.encoding
            )

            if ignore_tag_order:
                unique_tags = set({x.split("-")[-1] for x in unique_tags})
        else:
            error_message = (
                f"Wrong argument - {self.dataset_type} supplied for `dataset_type` parameter. "
                "Valid values are - 'ner_json', 'BIO', 'IOB', 'LBIOU' and 'BILUO'"
            )
            raise Exception(error_message)

        # unique_tags are formed after reading the data, label2id is a mapping of tags to numbers
        # provided by the model. If they differ then we have to initialize the model head according
        # to the tags provided in the data
        if label2id and unique_tags != label2id.keys():
            # label2id = None
            error_message = (
                f"Looks like the backbone is fine-tuned on the following entities - {list(label2id.keys())}"
                f" and your data consists of the following entities - `{unique_tags}`."
                f"\nPlease use a base model of the backbone and fine-tune it on your data or use a "
                f"model which is fine-tuned on a data having same labels as - `{unique_tags}`"
            )
            raise Exception(error_message)
        self.data = TextDataObject.prepare_data_for_entity_recognition(
            tokens_collection=tokens_collection,
            tags_collection=tags_collection,
            address_tag=address_tag,
            unique_tags=unique_tags,
            seed=self.seed,
            batch_size=self.batch_size,
            val_split_pct=self.val_split_pct,
            label2id=label2id,
        )
        self.backbone = "transformers"

    def prepare_data_for_spacy(self):
        if not HAS_SPACY:
            _raise_spacy_import_error()

        random.seed(self.seed)
        v_list = spacy.__version__.split(".")
        version = sum([int(j) * 10 ** (2 * i) for i, j in enumerate(v_list[::-1])])
        if version < 20108:  # checking spacy version
            error_message = (
                "Entity recognition model needs spacy version 2.1.8 or higher."
                f"Your current spacy version is {spacy.__version__}, please update using 'pip install'"
            )
            return logging.error(error_message)

        path = Path(self.path)
        train_data = []

        if self.class_mapping:
            address_tag = self.class_mapping.get("address_tag")
        else:
            address_tag = "Address"

        if self.dataset_type == "ner_json":
            train_data = _from_json(path=path, encoding=self.encoding)
            path = path.parent
        elif self.dataset_type == "BIO" or self.dataset_type == "IOB":
            tags_collection, tokens_collection, _ = _get_tags_and_tokens_collection(
                path, encoding=self.encoding
            )
            train_data = _from_iob_tags(
                tags_collection=tags_collection, tokens_collection=tokens_collection
            )
        elif self.dataset_type == "LBIOU" or self.dataset_type == "BILUO":
            nlp = spacy.blank("en")
            tags_collection, tokens_collection, _ = _get_tags_and_tokens_collection(
                path, encoding=self.encoding
            )

            for tags, tokens in zip(tags_collection, tokens_collection):
                try:
                    tags = _iob_to_biluo(tags)

                    doc = spacy.tokens.doc.Doc(
                        nlp.vocab,
                        words=tokens,
                        spaces=[True] * (len(tokens) - 1) + [False],
                    )
                    # run the standard pipeline against it
                    for name, proc in nlp.pipeline:
                        doc = proc(doc)
                    text = " ".join(tokens)
                    tags = _offsets_from_biluo_tags(doc, tags)
                    train_data.append((text, {"entities": tags}))
                except Exception as exception:
                    raise Exception(f"Exception while preparing data : {exception} ")
                    pass
        else:
            error_message = (
                f"Wrong argument - {self.dataset_type} supplied for `dataset_type` parameter. "
                "Valid values are - 'ner_json', 'BIO', 'IOB', 'LBIOU' and 'BILUO'"
            )
            raise Exception(error_message)

        self.data = spaCyNERDatabunch(
            train_data,
            val_split_pct=self.val_split_pct,
            batch_size=self.batch_size,
            address_tag=address_tag,
            test_ds=None,
        )
        self.data.path = path
        self.backbone = "spacy"


class _spaCyNERItemlist:
    """
    Creates a dataset to store data within ``ner_databunch`` object.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    batch_size              Batch size.
    ---------------------   -------------------------------------------
    data                    Required: list of tuple containing text and its entities.
    =====================   ===========================================

    :return: dataset.
    """

    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.entities = list(
            {
                i[2]
                for i in pd.concat(
                    [pd.Series(i["entities"]) for i in [o[1] for o in data]]
                )
            }
        )  ##Extracting all the unique entity names from input json
        self.data = data
        self.x = [o[0] for o in data]
        self.y = [o[1] for o in data]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def _random_batch(self, data):
        res = []
        for j in range(self.batch_size):
            res.append(random.choice(data))
        return res

    def _entities_to_dataframe(self, item):
        """
        This function is used to create pandas dataframe from training input data json.
        """
        text = item[0]
        df = pd.DataFrame(item[1].get("entities"))

        out_dict = {}
        if len(df):
            for x in df[2].unique():
                out_dict[x] = df[df[2] == x][[0, 1]].values.tolist()

        out = {}
        out["text"] = text
        for key in out_dict.keys():
            for tpl in out_dict.get(key):
                if out.get(key) == None:
                    out[key] = []
                out[key].append(text[tpl[0] : tpl[1]])
        return pd.Series(out)

    def show_batch(self):
        """
        This function shows a batch from the _spaCyNERItemlist.
        """
        data = self._random_batch(self.data)
        lst = []
        for item in data:
            lst.append(self._entities_to_dataframe(item))
        batch_df = pd.concat(lst, axis=1, sort=True).T

        text = batch_df["text"]
        batch_df.drop("text", axis=1, inplace=True)
        batch_df.insert(loc=0, column="text", value=text)
        return batch_df.fillna("")


class spaCyNERDatabunch:

    """
    Creates a databunch object.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    ds                      Required data list returned from _ner_prepare_data().
    ---------------------   -------------------------------------------
    val_split_pct           Optional float. Percentage of training data to keep
                            as validation.
                            The default Value is 0.1.
    ---------------------   -------------------------------------------
    batch_size              Optional integer. Batch size
                            The default value is 5.
    =====================   ===========================================

    :return: dataset
    """

    def __init__(self, ds, val_split_pct, batch_size, test_ds=None, address_tag=None):
        random.shuffle(ds)
        # creating an _spaCyNERItemlist with training dataset
        self.train_ds = _spaCyNERItemlist(
            batch_size, data=ds[: int(len(ds) * (1 - val_split_pct))]
        )
        # creating an _spaCyNERItemlist with validation dataset
        self.val_ds = _spaCyNERItemlist(
            batch_size, data=ds[int(len(ds) * (1 - val_split_pct)) :]
        )
        self.entities = list(
            set(self.train_ds.entities).union(set(self.val_ds.entities))
        )
        self._address_tag = address_tag
        self._has_address = True
        self.batch_size = batch_size
        if self.batch_size > len(self.train_ds):
            error_message = (
                f"Number of training data items ({len(self.train_ds)}) "
                f"is less than the batch size ({self.batch_size}). "
                "Please get more training data or lower the batch size"
            )
            logging.error(error_message)
        if self._address_tag not in self.entities:
            self._has_address = False
            logging.warning(
                "No Address tag found in your data.\n\
                1. If your data has an address field, pass your address field name as address tag in class mapping \n\
                e.g. - data=prepare_data(dataset_type=ds_type,path=training_data_folder,\n\t\t\t\
                    class_mapping={address_tag:address_field_name})\n\
                2. Else no action is required, if your data does not have any address information."
            )

    def show_batch(self):
        return self.train_ds.show_batch()
