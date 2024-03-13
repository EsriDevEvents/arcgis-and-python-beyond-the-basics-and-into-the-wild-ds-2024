import os
import re
import json
import arcgis
import logging
import warnings
import traceback
from datetime import datetime

HAS_NUMPY = True
HAS_FASTAI = True
HAS_TRANSFORMER = True
HAS_BEAUTIFULSOUP = True


try:
    import torch
    import pandas as pd
    from PIL import Image as PIL_Image
    from pathlib import Path
    from arcgis.auth.tools import LazyLoader

    cv2 = LazyLoader("cv2")
    h5py = LazyLoader("h5py")
    px = LazyLoader("plotly.express")
    go = LazyLoader("plotly.graph_objs")
    from torchvision import models
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from fastai.vision.data import imagenet_stats
    from fastprogress.fastprogress import progress_bar
    from .._data import _make_folder
    from ..models._arcgis_model import (
        _device_check,
        _resnet_family,
        _vgg_family,
        _densenet_family,
    )
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_FASTAI = False
else:
    warnings.filterwarnings("ignore", category=UserWarning, module="fastai")

try:
    import numpy as np
except Exception as e:
    HAS_NUMPY = False
else:
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

try:
    from bs4 import BeautifulSoup
except Exception as e:
    HAS_BEAUTIFULSOUP = False
else:
    warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

max_token_length = 512
allowed_text_extensions = ["csv", "txt", "json"]
allowed_image_extensions = ["png", "jpg", "jpeg", "tiff", "tif", "bmp"]


class TextModule:
    @staticmethod
    def preprocess_text(text, remove_urls=False, remove_html_tags=False):
        """
        Perform some basic cleanup like removing HTML tags, removing urls,
        and converting multiple white spaces to single white space
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

        if any([remove_html_tags, remove_urls]):
            text = re.sub(" +", " ", text)
        return text.strip()

    @classmethod
    def preprocess_text_list(cls, text_list, remove_urls=False, remove_html_tags=False):
        text_list = [
            cls.preprocess_text(x, remove_urls, remove_html_tags) for x in text_list
        ]
        return text_list

    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Max Pooling - Take the max value over time for every dimension
    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        token_embeddings[
            input_mask_expanded == 0
        ] = -1e9  # Set padding tokens to large negative value
        max_over_time = torch.max(token_embeddings, 1)[0]
        return max_over_time

    # Take the first token ([CLS]) from each sentence
    @staticmethod
    def cls_token(model_output, mask=None):
        return model_output[:, 0]


class Embeddings:
    """
    Creates an :class:`~arcgis.learn.Embeddings` Object. This object is capable of giving
    embeddings for text as well as images. The image embeddings are
    currently supported for RGB images only

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    dataset_type            Required string. The type of data for which
                            we would like to get the embedding vectors.
                            Valid values are `text` & `image`. Default
                            is set to `image`.

                            .. note::
                                The image embeddings are currently supported for `RGB` images only.
    ---------------------   -------------------------------------------
    backbone                Optional string. Specify the backbone/model-name
                            to be used to get the embedding vectors.
                            Default backbone for `image` dataset-type is
                            `resnet34` and for `text` dataset-type is
                            `sentence-transformers/distilbert-base-nli-stsb-mean-tokens`

                            To learn more about the available models for
                            for getting `text` embeddings, kindly visit:-
                            https://huggingface.co/sentence-transformers
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    working_dir             Option str. Path to a directory on local filesystem.
                            If directory is not present, it will be created.
                            This directory is used as the location to save the
                            model.
    =====================   ===========================================

    :return: :class:`~arcgis.learn.Embeddings` Object
    """

    def __init__(self, dataset_type="image", backbone=None, **kwargs):
        if not HAS_FASTAI:
            from .._data import _raise_fastai_import_error

            _raise_fastai_import_error(import_exception=import_exception)
        self._dataset_type = dataset_type

        if self._dataset_type == "image":
            self._allowed_extensions = allowed_image_extensions
        elif self._dataset_type == "text":
            self._allowed_extensions = allowed_text_extensions

        if "working_dir" in kwargs:
            self.working_dir = kwargs.get("working_dir")
        else:
            self.working_dir = Path.cwd()

        # _make_folder(os.path.join(os.path.abspath(self.working_dir), "embeddings"))

        self._file_path = None
        self.backbone = None
        self._tokenizer = None
        self._device = self._get_device()
        self._error_message = (
            f"Wrong backbone - {backbone} choosen for dataset-type - {self._dataset_type}. Kindly "
            f"call the `Embeddings.supported_backbones('{self._dataset_type}')` to see the "
            f"supported backbones for - {self._dataset_type} dataset-type."
        )

        from arcgis._impl.common._utils import _DisableLogger

        with _DisableLogger():
            self.model = self._load_model(dataset_type, backbone)
        self.model.to(self._device)

    @classmethod
    def supported_backbones(cls, dataset_type="image"):
        """
        Get available backbones/model-name for the given `dataset-type`

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        dataset_type            Required string. The type of data for which
                                we would like to get the embedding vectors.
                                Valid values are `text` & `image`. Default
                                is set to `image`
        =====================   ===========================================

        :return: a list containing the available models for the given `dataset-type`
        """
        if dataset_type == "image":
            return cls._get_image_compatible_backbones()
        elif dataset_type == "text":
            return cls._get_text_compatible_backbones()
        else:
            error_message = f"Wrong dataset-type - {dataset_type} provided. Valid values are 'image' or 'text'."
            raise Exception(error_message)

    @staticmethod
    def _get_image_compatible_backbones():
        return [
            *_resnet_family,
            *_densenet_family,
            *_vgg_family,
            models.mobilenet_v2.__name__,
        ]

    @staticmethod
    def _get_text_compatible_backbones():
        return [
            "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
            "sentence-transformers/bert-base-nli-max-tokens",
            "sentence-transformers/bert-base-nli-cls-token",
        ] + [
            "See all `TextEmbedding` models at https://huggingface.co/sentence-transformers"
        ]

    def _get_device(self):
        move_to_cpu = _device_check()
        if move_to_cpu:
            arcgis.env._processorType = "CPU"

        if (
            getattr(arcgis.env, "_processorType", "") == "GPU"
            and torch.cuda.is_available()
        ):
            device = torch.device("cuda")
        elif getattr(arcgis.env, "_processorType", "") == "CPU":
            device = torch.device("cpu")
        else:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        return device

    def _load_model(self, dataset_type="image", backbone=None):
        if dataset_type == "image":
            model = self._load_image_model(backbone)
        elif dataset_type == "text":
            model = self._load_text_model(backbone)
        else:
            error_message = f"Wrong dataset-type - {dataset_type} provided. Valid values are 'image' or 'text'."
            raise Exception(error_message)
        return model

    def _load_image_model(self, backbone=None):
        if backbone is None:
            backbone = "resnet34"

        if hasattr(models, backbone):
            model_type = getattr(models, backbone)
        else:
            raise Exception(self._error_message)

        self.backbone = backbone
        model = model_type(pretrained=True, progress=True)
        if "vgg" in self.backbone:
            model.classifier = model.classifier[:-1]
        else:
            layers = list(model.children())[:-1]
            model = torch.nn.Sequential(*layers)

        return model.eval()

    def _load_text_model(self, backbone=None):
        HAS_TRANSFORMER = True
        try:
            from transformers import AutoTokenizer, AutoModel
        except Exception as e:
            HAS_TRANSFORMER = False
        if not HAS_TRANSFORMER:
            raise Exception("This module requires transformers library.")
        if backbone is None:
            backbone = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
        self.backbone = backbone
        try:
            model = AutoModel.from_pretrained(backbone)
            self._tokenizer = AutoTokenizer.from_pretrained(
                backbone, config=model.config
            )
        except Exception as e:
            raise Exception(self._error_message)

        return model.eval()

    def get(
        self,
        text_or_list,
        batch_size=32,
        show_progress=True,
        return_embeddings=False,
        **kwargs,
    ):
        """
        Method to get the embedding vectors for the image/text items.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        text_or_list            Required string or List. String containing
                                directory path or list of directory paths where
                                image/text files are present for which the user wants
                                to get the embedding vectors.
        ---------------------   -------------------------------------------
        batch_size              Optional integer. The number of items to process
                                in one batch. Default is set to 32.
        ---------------------   -------------------------------------------
        show_progress           Optional boolean. If set to True, will display a
                                progress bar depicting the items processed so far.
                                Default is set to `True`.
        ---------------------   -------------------------------------------
        return_embeddings       Optional boolean. If set to True, a dataframe
                                containing the embeddings will be returned. If set
                                to False, they will be saved in a h5 file.
                                Default is set to `False`.
        =====================   ===========================================

        **kwargs**

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        normalize               Optional boolean. If set to `true`, will normalize
                                the image with `imagenet-stats` (mean and
                                std-deviation for each color channel in RGB image).
                                This argument is valid only for `dataset-type` image.
                                Default is set to True.
        ---------------------   -------------------------------------------
        file_extensions         Optional String or List. The file extension(s) for
                                which the user wish to get embedding vectors for.
                                Allowed values for `dataset-type` image are -
                                ['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp']
                                Allowed values for `dataset-type` text are -
                                ['csv', 'txt', 'json']

                                .. note::
                                        For json files, if we have nested json structures, then text will be extracted only from the 1st level.
        ---------------------   -------------------------------------------
        chip_size               Optional integer. Resize the image to
                                `chip_size X chip_size` pixels.
                                This argument is valid only for `dataset-type` image.
                                Default is set to 224
        ---------------------   -------------------------------------------
        encoding                Optional string. The encoding to read the text/csv/
                                json file. Applicable only for `dataset-type` text.
                                Default is `UTF-8`
        ---------------------   -------------------------------------------
        text_column             Optional string. The column that will be used to get
                                the text content from `csv` or `json` file types.
                                This argument is valid only for `dataset-type` text.
                                Default is set to `text`
        ---------------------   -------------------------------------------
        remove_urls             Optional boolean. If true, remove urls from text.
                                This argument is valid only for `dataset-type` text.
                                Default value is False.
        ---------------------   -------------------------------------------
        remove_html_tags        Optional boolean. If true, remove html tags from text.
                                This argument is valid only for `dataset-type` text.
                                Default value is False.
        ---------------------   -------------------------------------------
        pooling_strategy        Optional string. The transformer model gives embeddings
                                for each word/token present in the text. The type of
                                pooling to be done on those word/token vectors in order
                                to form the text embeddings.
                                Allowed values are - ['mean', 'max', 'first']
                                This argument is valid only for `dataset-type` text.
                                Default value is `mean`.
        =====================   ===========================================

        :return: The path of the H5 file where items & corresponding embeddings are saved.
        """
        if not HAS_NUMPY:
            raise Exception("This module requires numpy.")

        file_name = f"embeddings_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"
        self._file_path = os.path.join(self.working_dir, "embeddings", file_name)
        if os.path.exists(self._file_path):
            raise Exception(
                f"File to save the embeddings already present at - {self._file_path}. Kindly rename the file "
                f"or move the file to another location to proceed."
            )
        text_img_df = kwargs.get("dataframe", False)
        if isinstance(text_img_df, pd.DataFrame):
            col = kwargs.get("text_column", "text")
            item_list = text_img_df[col].values.tolist()
        else:
            item_list = self._get_items(text_or_list, **kwargs)
        if self._dataset_type == "image":
            ret = self._get_image(
                item_list, batch_size, show_progress, return_embeddings, **kwargs
            )
        else:
            ret = self._get_text(
                item_list, batch_size, show_progress, return_embeddings, **kwargs
            )

        return ret

    @staticmethod
    def _do_h5file_sanity(file_path):
        hf = h5py.File(file_path, "r")
        dataset_keys = list(hf.keys())
        if (
            len(dataset_keys) != 2
            or "embeddings" not in dataset_keys
            or "items" not in dataset_keys
        ):
            error_message = (
                f"Wrong H5 file passed. The H5 files consists of - {dataset_keys} dataset keys. "
                f"We expected the following dataset keys - ['items', 'embeddings']"
            )
            hf.close()
            raise Exception(error_message)

        embeddings_dataset, items_dataset = hf.get("embeddings"), hf.get("items")
        if embeddings_dataset.shape[0] != items_dataset.shape[0]:
            error_message = (
                f"Length of `embeddings`({embeddings_dataset.shape[0]}) and `items`"
                f"({items_dataset.shape[0]}) do not match. There's some problem with the "
                f"file present at location - {file_path}"
            )
            raise Exception(error_message)

        hf.close()

    def load(self, file_path, load_to_memory=True):
        """
        Load the extracted embeddings from the H5 file

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        file_path               Required string. The path to the H5 file which
                                gets auto generated after the call to the `get`
                                method of the :class:`~arcgis.learn.Embeddings` class
        ---------------------   -------------------------------------------
        load_to_memory          Optional Bool. whether or not to load the entire
                                content of the H5 file to memory. Loading very large
                                H5 files into the memory takes up lot of RAM space.
                                Use this parameter with caution for large H5 files.
                                Default is set to True.
        =====================   ===========================================

        :return: When `load_to_memory` param is `True` - A 2 item tuple containing
                  the numpy arrays of extracted embeddings and items
                  When `load_to_memory` param is `False` - A 3 item tuple containing
                  the H5 file handler & 2 H5 dataset object of extracted embeddings
                  and items
        """

        if os.path.exists(file_path) is False or os.path.isfile(file_path) is False:
            raise Exception(f"The path `{file_path}` does not exists, or not a file")

        self._do_h5file_sanity(file_path)
        hf = h5py.File(file_path, "r")
        embeddings_dataset, items_dataset = hf.get("embeddings"), hf.get("items")
        if load_to_memory:
            embeddings, items = np.array(embeddings_dataset), np.array(items_dataset)
            hf.close()
            return embeddings, items
        else:
            return hf, embeddings_dataset, items_dataset

    @staticmethod
    def _check_directory_validity(dir_paths):
        for directory in dir_paths:
            if os.path.exists(directory) is False:
                raise Exception(f"The path `{directory}` does not exists.")
            if os.path.isfile(directory):
                raise Exception(
                    f"The path `{directory}` seems like a file path. Please provide the directory path."
                )
            if os.path.isdir(directory) is False:
                raise Exception(
                    f"The path `{directory}` is not a valid directory path."
                )

    def _check_file_extension_validity(self, file_extensions):
        for extension in file_extensions:
            if extension not in self._allowed_extensions:
                raise Exception(
                    f"Extension -`{extension}` is not a valid extension for dataset-type - "
                    f"`{self._dataset_type}`. Allowed extension values are - {self._allowed_extensions}"
                )

    def _get_text_items(self, file_paths, text_column, encoding):
        text_list = []
        for file_path, ext in file_paths:
            if ext == "txt":
                with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                    text_list.append(f.read())
            elif ext == "csv":
                df = pd.read_csv(file_path, dtype="str")
                if text_column not in df.columns:
                    raise Exception(
                        f"CSV file - {file_path} doesn't contain the column - `{text_column}`"
                    )
                df.dropna(axis=0, subset=[text_column], inplace=True)
                text_list.extend(df[text_column].values.tolist())
            elif ext == "json":
                with open(file_path, "r", encoding=encoding) as f:
                    data_list = f.readlines()

                txt_list = []
                for item in data_list:
                    data_dict = json.loads(item)
                    if text_column not in data_dict:
                        raise Exception(
                            f"Row - \n{data_dict}\n\n in JSON file - {file_path} doesn't contain the key - "
                            f"`{text_column}`. Kindly fix the JSON file or pass the correct value of the "
                            f"`text_column` parameter in the `get` method call."
                        )
                    text = data_dict[text_column]
                    if text:
                        txt_list.append(text)

                text_list.extend(txt_list)
            else:
                raise Exception(
                    f"Extension -`{ext}` is not a valid extension for dataset-type - `{self._dataset_type}`"
                    f". Allowed extension values are - {self._allowed_extensions}"
                )

        return text_list

    def _get_items(self, dir_path, **kwargs):
        item_list = []
        file_extensions = kwargs.get("file_extensions", self._allowed_extensions)
        if isinstance(file_extensions, (str, bytes)):
            file_extensions = [file_extensions]
        file_extensions = [x.replace(".", "") for x in file_extensions]

        self._check_file_extension_validity(file_extensions)
        text_column = kwargs.get("text_column", "text")
        encoding = kwargs.get("encoding", "utf-8")
        if isinstance(dir_path, (str, bytes)):
            dir_path = [dir_path]
        self._check_directory_validity(dir_path)

        for rootdir in dir_path:
            for dirpath, dirs, files in os.walk(rootdir):
                for filename in files:
                    fname = os.path.join(dirpath, filename)
                    ext = os.path.splitext(filename)[-1].lower().replace(".", "")
                    if ext in file_extensions:
                        logging.info(
                            f"File name - {fname} will be considered for getting the embeddings"
                        )
                        item_list.append((fname, ext))

        if len(item_list) == 0:
            raise Exception(
                f"Not a single item found to extract the embeddings in folder(s) - {dir_path}. Kindly check"
                f" if the directory contains image/text files or check if you have passed the right set of"
                f" `file_extensions` to the method."
            )

        if self._dataset_type == "text":
            item_list = self._get_text_items(item_list, text_column, encoding)
        else:
            item_list = [x[0] for x in item_list]

        return item_list

    @staticmethod
    def _normalize(x, mean, std):
        z = (x - mean[..., None, None]) / std[..., None, None]
        return z

    @staticmethod
    def _insert_to_h5_file(file_handler, items, embeddings):
        dt = h5py.special_dtype(vlen=bytes)
        if len(file_handler.keys()) == 0:
            file_handler.create_dataset(
                "items", data=items, maxshape=(None,), dtype=dt, chunks=True
            )
            file_handler.create_dataset(
                "embeddings",
                data=embeddings,
                maxshape=(None, embeddings.shape[1]),
                chunks=True,
            )
        else:
            file_handler["items"].resize(
                (file_handler["items"].shape[0] + items.shape[0]), axis=0
            )
            file_handler["items"][-items.shape[0] :] = items
            file_handler["embeddings"].resize(
                (file_handler["embeddings"].shape[0] + embeddings.shape[0]), axis=0
            )
            file_handler["embeddings"][-embeddings.shape[0] :] = embeddings

    def _get_image(
        self,
        item_list,
        batch_size=32,
        show_progress=True,
        return_embeddings=False,
        **kwargs,
    ):
        if not return_embeddings:
            with h5py.File(self._file_path, "a") as hf:
                batch_embeddings = self._extract_img_embeddings(
                    item_list, batch_size=32, show_progress=True, **kwargs
                )
                self._insert_to_h5_file(hf, item_list, batch_embeddings)
            return self._file_path
        else:
            batch_embeddings = self._extract_img_embeddings(
                item_list, batch_size=32, show_progress=True, **kwargs
            )
            return batch_embeddings

    def _extract_img_embeddings(
        self, item_list, batch_size=32, show_progress=True, **kwargs
    ):
        normalize = kwargs.get("normalize", True)
        resize_to = kwargs.get("chip_size", 224)
        all_batch_embedding = np.empty([0, 512])
        mean, std = None, None
        if normalize:
            mean, std = imagenet_stats
            mean = torch.tensor(mean).float().to(self._device)
            std = torch.tensor(std).float().to(self._device)

        for i in progress_bar(
            range(0, len(item_list), batch_size), display=show_progress
        ):
            try:
                img_list = item_list[i : i + batch_size]
                img_batch = np.array(
                    [
                        np.array(
                            PIL_Image.open(img_path)
                            .convert("RGB")
                            .resize((resize_to, resize_to))
                        ).astype(float)
                        / 255.0
                        for img_path in img_list
                    ]
                )

                img_batch = (
                    torch.tensor(img_batch.transpose(0, 3, 1, 2))
                    .float()
                    .to(self._device)
                )
                if normalize:
                    img_batch = self._normalize(img_batch, mean, std)

                with torch.no_grad():
                    out = self.model(img_batch)
                    if "densenet" in self.backbone or "mobilenet" in self.backbone:
                        out = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))(out)
                    if (
                        "densenet" in self.backbone
                        or "mobilenet" in self.backbone
                        or "resnet" in self.backbone
                    ):
                        out = out.view(out.size(0), -1)

                img_list = np.array([x.encode() for x in img_list])
                batch_embeddings = (
                    torch.nn.functional.normalize(out.data).cpu().detach().numpy()
                )
                all_batch_embedding = np.append(
                    all_batch_embedding, batch_embeddings, axis=0
                )

                # self._insert_to_h5_file(hf, img_list, batch_embeddings)
            except Exception as e:
                raise Exception(e)
        return all_batch_embedding

    def _get_text(
        self,
        item_list,
        batch_size=32,
        show_progress=True,
        return_embeddings=False,
        **kwargs,
    ):
        if not return_embeddings:
            with h5py.File(self._file_path, "a") as hf:
                batch_embeddings = self._extract_text_embeddings(
                    item_list, batch_size=32, show_progress=True, **kwargs
                )
                self._insert_to_h5_file(hf, img_list, batch_embeddings)
            return self._file_path
        else:
            batch_embeddings = self._extract_text_embeddings(
                item_list, batch_size=32, show_progress=True, **kwargs
            )
            return batch_embeddings

    def _extract_text_embeddings(
        self, item_list, batch_size=32, show_progress=True, **kwargs
    ):
        remove_urls = kwargs.get("remove_urls", False)
        remove_html_tags = kwargs.get("remove_html_tags", False)
        pooling_strategy = kwargs.get("pooling_strategy", "mean")

        all_batch_embedding = np.empty([0, 768])
        if any([remove_urls, remove_html_tags]):
            item_list = TextModule.preprocess_text_list(
                item_list, remove_urls, remove_html_tags
            )
        # batch_embedding_list = []
        try:
            for i in progress_bar(
                range(0, len(item_list), batch_size), display=show_progress
            ):
                text_batch = item_list[i : i + batch_size]
                encoded_input = self._tokenizer(
                    text_batch,
                    padding=True,
                    truncation=True,
                    max_length=max_token_length,
                    return_tensors="pt",
                ).to(self._device)
                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                # First element of model_output contains all token embeddings
                # token_embeddings = torch.nn.functional.normalize(model_output[0])
                token_embeddings, attention_mask = (
                    model_output[0],
                    encoded_input["attention_mask"],
                )
                if pooling_strategy == "mean":
                    batch_embedding = TextModule.mean_pooling(
                        token_embeddings, attention_mask
                    )
                elif pooling_strategy == "max":
                    batch_embedding = TextModule.max_pooling(
                        token_embeddings, attention_mask
                    )
                elif pooling_strategy == "first":
                    batch_embedding = TextModule.cls_token(token_embeddings)
                else:
                    error_message = (
                        f"Wrong pooling-strategy - {pooling_strategy} choosen. Allowed values are - "
                        f"'mean', 'max' and 'first'. kindly choose from these options."
                    )
                    raise Exception(error_message)

                text_batch = np.array([x.encode() for x in text_batch])
                batch_embeddings = (
                    torch.nn.functional.normalize(batch_embedding)
                    .cpu()
                    .detach()
                    .numpy()
                )
                all_batch_embedding = np.append(
                    all_batch_embedding, batch_embeddings, axis=0
                )

        except Exception as e:
            raise Exception(e)
        return all_batch_embedding

    @staticmethod
    def _do_clustering(embeddings, item_list=None, n_clusters=5, dimensions=3):
        n_components = min(embeddings.shape[0], 64)
        transformed_embeddings = PCA(n_components=n_components).fit_transform(
            embeddings
        )
        embeddings_for_visualization = PCA(n_components=dimensions).fit_transform(
            embeddings
        )

        columns = ["x", "y", "z"] if dimensions == 3 else ["x", "y"]
        result = pd.DataFrame(embeddings_for_visualization, columns=columns)
        if item_list:
            result["item"] = item_list

        # DBSCAN clustering
        # eps = kwargs.pop("eps", 0.5)
        # min_samples, metric = kwargs.pop("min_samples", 10), kwargs.pop("metric", "cosine")
        # clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, **kwargs).fit(transformed_embeddings)
        # result['labels'] = clustering.labels_
        # outliers = result.loc[result.labels == -1, :]
        # clustered = result.loc[result.labels != -1, :]

        # KMeans clustering
        clustering = KMeans(n_clusters=n_clusters)
        labels = clustering.fit_predict(transformed_embeddings)
        result["labels"] = labels
        random_sample = result.reset_index(drop=True)

        if len(random_sample) > 500:
            num_items_per_group = int(500 / n_clusters)
            # random_sample = random_sample.sample(n=500).reset_index(drop=True)
            random_sample = (
                random_sample.groupby("labels")
                .apply(lambda x: x.sample(n=num_items_per_group, replace=True))
                .reset_index(drop=True)
            )

        # print(len(random_sample))
        return random_sample

    def _visualize_with_items(self, cluster_dataframe, dimensions=3):
        from ipywidgets import Image, Layout, HBox, Textarea

        image_data = {}
        widget_dict = dict(
            x=cluster_dataframe["x"],
            y=cluster_dataframe["y"],
            mode="markers",
            marker=dict(color=cluster_dataframe["labels"]),
        )
        if dimensions == 3:
            widget_dict.update({"z": cluster_dataframe["z"], "type": "scatter3d"})
        else:
            widget_dict.update({"type": "scatter"})

        fig = go.FigureWidget(data=[widget_dict])

        scatter = fig.data[0]

        if self._dataset_type == "image":
            for img_path in cluster_dataframe["item"]:
                img = np.array(PIL_Image.open(img_path))
                inmem_jpg = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[
                    1
                ].tobytes()
                image_data[img_path] = inmem_jpg

            widget = Image(
                value=image_data[cluster_dataframe.item.iloc[0]],
                layout=Layout(height="200px", width="200px"),
            )
        else:
            widget = Textarea(
                value=cluster_dataframe.item.iloc[0],
                disabled=True,
                layout=Layout(height="350px", width="200px"),
            )

        def hover_fn(trace, points, state):
            ind = points.point_inds[0]
            item = cluster_dataframe["item"][ind]
            widget.value = image_data[item] if self._dataset_type == "image" else item

        scatter.on_hover(hover_fn)
        return HBox([fig, widget])

    def visualize(
        self, file_path, visualize_with_items=True, n_clusters=5, dimensions=3
    ):
        """
        Method to visualize the embedding vectors for the image/text items.
        This method uses the K-Means clustering algorithm to partition the
        embeddings vectors into n-clusters. This requires the loading the
        entire content of the H5 file to RAM. Loading very large H5 files
        into the memory takes up lot of RAM space. Use this method with
        caution for large H5 files.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        file_path               Required string. The path to the H5 file which
                                gets auto generated after the call to the `get`
                                method of the :class:`~arcgis.learn.Embeddings` class.
        ---------------------   -------------------------------------------
        visualize_with_items    Optional Bool. Whether or not to visualize the
                                embeddings with items. Default is set to True.
        ---------------------   -------------------------------------------
        n_clusters              Optional integer. The number of clusters to create
                                for the embedding vectors. This value will be passed
                                to the `KMeans` algorithm to generate the clusters.
                                Default is set to 5.
        ---------------------   -------------------------------------------
        dimensions              Optional integer. The number of dimensions to project
                                the embedding vectors for visualization purpose.
                                Allowed values are `2` & `3`
                                Default is set to 3.
        =====================   ===========================================
        """

        self._do_h5file_sanity(file_path)
        hf = h5py.File(file_path, "r")
        embeddings_dataset, items_dataset = hf.get("embeddings"), hf.get("items")
        embeddings, item_list = (
            np.array(embeddings_dataset),
            np.array(items_dataset).tolist(),
        )
        hf.close()

        # For using DBSCAN clustering take `eps`, `metric` and `min_samples` in **kwargs and pass to below method
        cluster_df = self._do_clustering(
            embeddings, item_list, n_clusters=n_clusters, dimensions=dimensions
        )

        if visualize_with_items is False:
            if dimensions == 3:
                fig = px.scatter_3d(
                    cluster_df,
                    x=cluster_df.x,
                    y=cluster_df.y,
                    z=cluster_df.z,
                    color=cluster_df.labels,
                    hover_data=["labels"],
                )
            else:
                fig = px.scatter(
                    cluster_df,
                    x=cluster_df.x,
                    y=cluster_df.y,
                    color="labels",
                    hover_data=["labels"],
                )
            return fig
        else:
            return self._visualize_with_items(cluster_df, dimensions)
