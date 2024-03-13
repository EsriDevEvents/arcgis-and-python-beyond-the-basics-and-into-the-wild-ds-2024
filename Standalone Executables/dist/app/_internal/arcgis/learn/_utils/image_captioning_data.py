from fastai.vision.data import ImageList
import pandas as pd
import json
from pathlib import Path
import random
import types
from textwrap import wrap
import sys
from typing import List, Callable, Tuple, Dict, cast
import warnings
import traceback
import xml.etree.ElementTree as ET

from traitlets.traitlets import parse_notifier_name
from .env import raise_fastai_import_error

try:
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset, DataLoader
    from fastai.text import Tokenizer, Vocab
    import fastai
    from fastai.vision import (
        get_transforms,
        open_image,
        image2np,
        Image,
        imagenet_stats,
        normalize,
        denormalize,
    )
    from fastai.vision.transform import crop
    from fastai.data_block import DataBunch
    from .pointcloud_data import get_device
    from .._utils.common import get_nbatches, get_top_padding
    import matplotlib.patheffects as PathEffects
    import torch

    HAS_FASTAI = True
except ImportError:
    import_exception = traceback.format_exc()
    HAS_FASTAI = False

StateType = Dict[str, torch.Tensor]
StepFunctionType = Callable[
    [torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]
]
StepFunctionTypeNoTimestep = Callable[
    [torch.Tensor, StateType], Tuple[torch.Tensor, StateType]
]


# Modified BeamSearch from AllenNLP library.
class BeamSearchAttention:
    """
    Implements the beam search algorithm for decoding the most likely sequences.
    [0]: https://arxiv.org/abs/1702.01806
    # Parameters
    end_index : `int`
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : `int`, optional (default = `50`)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : `int`, optional (default = `10`)
        The width of the beam used.
    per_node_beam_size : `int`, optional (default = `beam_size`)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to `beam_size`. Setting this parameter
        to a number smaller than `beam_size` may give better results, as it can introduce
        more diversity into the search. See [Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017][0].
    """

    def __init__(
        self,
        end_index: int,
        max_steps: int = 50,
        beam_size: int = 10,
        per_node_beam_size: int = None,
    ) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size

    @staticmethod
    def reconstruct_sequences(predictions, backpointers):
        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = (
                predictions[timestep].gather(1, cur_backpointers.long()).unsqueeze(2)
            )

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(
                1, cur_backpointers.long()
            )

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers.long()).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        return reconstructed_predictions

    @torch.no_grad()
    def search(
        self,
        start_predictions: torch.Tensor,
        start_state: StateType,
        step: StepFunctionType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.
        Notes
        -----
        If your step function returns `-inf` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have `-inf` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from `search`
        and potentially discard sequences with non-finite log probability.
        # Parameters
        start_predictions : `torch.Tensor`
            A tensor containing the initial predictions with shape `(batch_size,)`.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.
        start_state : `StateType`
            The initial state passed to the `step` function. Each value of the state dict
            should be a tensor of shape `(batch_size, *)`, where `*` means any other
            number of dimensions.
        step : `StepFunctionType`
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape `(group_size,)`, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The `group_size` will be `batch_size * beam_size`, except in the initial
            step, for which it will just be `batch_size`.
            The function is expected to return a tuple, where the first element
            is a tensor of shape `(group_size, target_vocab_size)` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            `(group_size, *)`, where `*` means any other number of dimensions.
        # Returns
        `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of `(predictions, log_probabilities)`, where `predictions`
            has shape `(batch_size, beam_size, max_steps)` and `log_probabilities`
            has shape `(batch_size, beam_size)`.
        """

        # If the step function we're given does not take the time step argument, wrap it
        # in one that does.
        from inspect import signature

        step_signature = signature(step)
        if len(step_signature.parameters) < 3:
            old_step = cast(StepFunctionTypeNoTimestep, step)

            def new_step(
                last_predictions: torch.Tensor,
                state: Dict[str, torch.Tensor],
                time_step: int,
            ):
                return old_step(last_predictions, state)

            step = new_step

        batch_size = start_predictions.size()[0]

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []
        attention_maps = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state = step(start_predictions, start_state, 0)

        num_classes = start_class_log_probabilities.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise Exception(
                f"Target vocab size ({num_classes:d}) too small "
                f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                f"Please decrease beam_size or per_node_beam_size."
            )

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        (
            start_top_log_probabilities,
            start_predicted_classes,
        ) = start_class_log_probabilities.topk(self.beam_size)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn(
                "Empty sequences predicted. You may want to increase the beam size or ensure "
                "your step function is working properly.",
                RuntimeWarning,
            )
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            if state_tensor is None:
                continue
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = (
                state_tensor.unsqueeze(1)
                .expand(batch_size, self.beam_size, *last_dims)
                .reshape(batch_size * self.beam_size, *last_dims)
            )

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            class_log_probabilities, state = step(last_predictions, state, timestep + 1)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities,
            )

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(
                self.per_node_beam_size
            )

            # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probabilities = (
                last_log_probabilities.unsqueeze(2)
                .expand(batch_size, self.beam_size, self.per_node_beam_size)
                .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )

            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = (
                top_log_probabilities + expanded_last_log_probabilities
            )

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices.long()
            )

            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices / self.per_node_beam_size

            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                if state_tensor is None:
                    continue
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.view(
                    batch_size, self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                state[key] = (
                    state_tensor.reshape(batch_size, self.beam_size, *last_dims)
                    .gather(1, expanded_backpointer.long())
                    .reshape(batch_size * self.beam_size, *last_dims)
                )

            attention_maps.append(state["attention_map"])

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn(
                "Infinite log probabilities encountered. Some final sequences may not make sense. "
                "This can happen when the beam size is larger than the number of valid (non-zero "
                "probability) transitions that the step function produces.",
                RuntimeWarning,
            )

        reconstructed_predictions = self.reconstruct_sequences(
            predictions, backpointers
        )

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probabilities, attention_maps


# Image Captioning Data:

# Sources: json (can be of different formats)
# 	- can specify key where image names are present.
# 	- where captions are present.
#   - 'images|images,sentences'
# 	 CSV (image path | captions)
# 	- can specify column names for images.
# 	- column name for captions.
#   - image|captions

# Pytorch Dataset:

# def open_image(im_path):
#     pass
#     # get


class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        root_path,
        images,
        captions,
        tokenizer,
        vocab,
        transforms,
        chip_sz,
        resize_to,
        norm_stats,
        split,
        flip_vert,
    ):
        # set path and annotations
        self.root_path = Path(root_path)

        # get images
        self.images = images
        self.captions = captions

        # create_vocab using fastai functions and tokenizer or use preloaded vocab.
        self.tokenizer = tokenizer
        self.vocab = vocab

        # if transforms is None Apply default transforms
        self.split = split
        self.resize_to = resize_to
        self.chip_size = chip_sz
        self.transforms = transforms
        self.crop_tfm = crop(size=chip_sz, row_pct=(0, 1), col_pct=(0, 1))
        if transforms is None:
            self.train_transforms, self.val_transforms = get_transforms(
                flip_vert=flip_vert
            )
        elif transforms is False:
            self.train_transforms, self.val_transforms = None, None
        else:
            self.train_transforms, self.val_transforms = transforms

        self.norm_stats = list(map(torch.tensor, norm_stats))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # read image using gdal or PIL
        im = open_image(self.root_path / "images" / self.images[index])
        # Apply transforms on the image.

        # get caption using the index.
        cap = self.captions[index]
        cap = cap[random.randint(0, len(cap) - 1)]
        # convert to word index
        cap = self.tokenizer._process_all_1([cap])[0]
        cap = (
            [self.vocab.stoi["xxbos"]]
            + self.vocab.numericalize(cap)
            + [self.vocab.stoi["xxeos"]]
        )
        # return images and captions as indexes

        if len(im.shape) > 2:
            c, h, w = im.shape
        else:
            h, w = im.shape

        if self.chip_size > h or self.chip_size > w:
            crop_flag = False
        else:
            crop_flag = True

        size = (
            (self.resize_to, self.resize_to)
            if self.resize_to is not None
            else self.resize_to
        )
        if self.resize_to is not None:
            self.chip_size = self.resize_to

        if self.transforms is not False:
            if self.split == "train":
                if crop_flag:
                    im = im.apply_tfms(self.crop_tfm).apply_tfms(
                        self.train_transforms, size=size
                    )
                else:
                    im = im.apply_tfms(self.train_transforms, size=size)
            else:
                im = im.resize(self.chip_size)

        return normalize(im.px, *self.norm_stats), cap


def normalize(x, mean, std):
    "Normalize `x` with `mean` and `std`."
    to_tensor = lambda z: torch.tensor(z).to(x.device)
    if type(mean[0]) is not torch.Tensor:
        mean = to_tensor(mean)
        std = to_tensor(std)
    return (x - mean[..., None, None]) / std[..., None, None]


def denormalize(x, mean, std, do_x=True):
    "Denormalize `x` with `mean` and `std`."
    to_tensor = lambda z: torch.tensor(z).to(x.device)
    if type(mean[0]) is not torch.Tensor:
        mean = to_tensor(mean)
        std = to_tensor(std)
    return (
        x.cpu().float() * std[..., None, None] + mean[..., None, None]
        if do_x
        else x.cpu()
    )


def read_json(file_name):
    with open(file_name) as f:
        json_file = json.load(f)
    return json_file


def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df


def parse_json(obj, keys, index, caption_list, club_items):
    if index == (len(keys) - 1):
        if club_items:
            if type(obj) is list:
                caption_list.append([o[keys[index]] for o in obj])
            else:
                caption_list.append([obj[keys[index]]])
        else:
            if type(obj) is list:
                for o in obj:
                    caption_list.append(o[keys[index]])
            else:
                caption_list.append(obj[keys[index]])
    else:
        if type(obj) is list:
            for o in obj:
                parse_json(o[keys[index]], keys, index + 1, caption_list, club_items)
        else:
            obj = obj[keys[index]]
            parse_json(obj, keys, index + 1, caption_list, club_items)


def parse_xml_from_image_file(imagefile):
    """
    Function that returns captions for an image for image captioning data.
    input: imagefile (Path)
    returns: labels (List[str])
    """
    xmlfile = (
        imagefile.parents[1]
        / "labels"
        / imagefile.name.replace("{ims}".format(ims=imagefile.suffix), ".xml")
    )
    labels = ET.parse(xmlfile).getroot().find("object").find("name").text
    return labels


def get_annotations(ann_object, key, read_type):
    if read_type == "json":
        # 'images,filename', 'images,sentences,raw'
        input_key, annotation_key = key.split("|")
        keys = annotation_key.split(",")
        input_keys = input_key.split(",")
        depth = len(keys)
        all_captions = []
        all_images = []
        parse_json(ann_object, keys, 0, all_captions, club_items=True)
        parse_json(ann_object, input_keys, 0, all_images, club_items=False)
        return all_images, all_captions
    if read_type == "csv":
        raise NotImplementedError
    if read_type == "arcgis":
        all_images = ImageList.from_folder(ann_object).items
        all_captions = [[parse_xml_from_image_file(f)] for f in all_images]
        return all_images, all_captions


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(
        [im.px if isinstance(im, fastai.vision.image.Image) else im for im in images], 0
    )

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = torch.tensor([len(cap) for cap in captions]).long()
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.tensor(cap[:end]).long()
    return images, (targets, lengths)


def prepare_captioning_dataset(
    path, chip_sz, batch_size, val_split_pct, transforms, resize_to, **kwargs
):
    """
    If 'annotations.json' is present in root path, then we should use that,
    also pass 'annotations_key' in wherecaptions are present, seperated by commas
    Elif 'annotations.csv' is present in root path, we should use that.
    pass img col and caption col as seperated by commas
    elif 'labels' folder is present in root path, we should use it. this format is arcgis format
    else format not supported
    """

    if not HAS_FASTAI:
        raise_fastai_import_error(
            import_exception=import_exception, message="", installation_steps=" "
        )

    # Read file and get all captions.
    imagecap_kwargs = kwargs.get("image_captioning_kwargs", {})
    if (path / "annotations.json").exists():
        # RSICD Dataset.
        imagecap_kwargs = {"annotations_key": "images,filename|images,sentences,raw"}
        annotations_json = path / "annotations.json"
        annotations_key = imagecap_kwargs.get("annotations_key")
        annotation_object = read_json(annotations_json)
        ann_type = "json"

    elif (path / "annotations.csv").exists():
        annotations_csv = path / "annotations.csv"
        annotations_key = imagecap_kwargs.get("annotations_key")
        annotation_object = read_csv(annotations_csv)
        ann_type = "csv"

    elif (path / "labels").exists():
        annotation_object = path / "images"
        annotations_key = path / "labels"
        ann_type = "arcgis"

    else:
        raise Exception("Unindentified format")

    lang = imagecap_kwargs.get("language", "en")

    paired_images, paired_captions = get_annotations(
        annotation_object, annotations_key, ann_type
    )

    # create tokenizer and vocab
    tokenizer = Tokenizer(n_cpus=1, lang=lang)
    tokenized_captions = tokenizer.process_all([i for c in paired_captions for i in c])
    vocab = Vocab.create(tokenized_captions, max_vocab=100000, min_freq=4)
    del tokenized_captions

    norm_stats = kwargs.get("norm_stats", imagenet_stats)
    flip_vert = kwargs.get("imagery_type", "satellite") != "oriented"

    # create train val idxs using split pct
    total_files = len(paired_images)
    total_val_files = int(val_split_pct * total_files)
    # random shuffling and maintaining order.
    z = list(zip(paired_images, paired_captions))
    random.shuffle(z)
    paired_images, paired_captions = zip(*z)
    val_images = paired_images[:total_val_files]
    val_captions = paired_captions[:total_val_files]
    train_images = paired_images[total_val_files:]
    train_captions = paired_captions[total_val_files:]

    # instantiate ImageCaptioning class
    train_dataset = ImageCaptioningDataset(
        path,
        train_images,
        train_captions,
        tokenizer,
        vocab,
        transforms,
        chip_sz,
        resize_to,
        norm_stats,
        split="train",
        flip_vert=flip_vert,
    )

    valid_dataset = ImageCaptioningDataset(
        path,
        val_images,
        val_captions,
        tokenizer,
        vocab,
        transforms,
        chip_sz,
        resize_to,
        norm_stats,
        split="val",
        flip_vert=False,
    )

    # create train val dataloaders with appropriate batch size.
    databunch_kwargs = {"num_workers": 0} if sys.platform == "win32" else {}

    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, **databunch_kwargs
    )

    valid_dl = DataLoader(
        valid_dataset, batch_size=batch_size, collate_fn=collate_fn, **databunch_kwargs
    )

    # create Databunch
    device = get_device()
    data = DataBunch(train_dl, valid_dl, device=device, collate_fn=collate_fn)
    # Attach show batch to databunch
    data.show_batch = types.MethodType(show_batch, data)
    # attach vocab
    data.vocab = vocab
    # attach chip size
    data.chip_size = chip_sz
    # attach resize to
    data.resize_to = resize_to
    # add language
    data.lang = lang
    # add path
    data.path = path
    # return databunch.
    data._dataset_type = "ImageCaptioning"
    return data


def show_image_and_text(ax, image, text, show_coords):
    if isinstance(image, fastai.vision.image.Image):
        image = image.px
    ax.imshow(image2np(image))
    # ax.title.set_text('\n'.join(wrap(text, 40)))
    ax.set_title("\n".join(wrap(text, 40)), y=-0.15, pad=0.1)
    ax.title.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])
    if not show_coords:
        ax.axis("off")


def show_batch(self, rows=2, **kwargs):
    # create a grid of square root of rows
    figsize = kwargs.get("figsize", (5 * rows, 5 * rows))
    show_coords = kwargs.get("show_coords", False)
    fig, ax = plt.subplots(rows, rows, figsize=figsize)

    img_idxs = [random.randint(0, len(self.train_ds) - 1) for k in range(rows**2)]
    # iterate through the rows and get transformed images from the dataset class
    for k, idx in enumerate(img_idxs):
        img, captions = self.train_ds[idx]
        caption = self.vocab.textify(captions[1:-1])
        # use matplotlib for display
        show_image_and_text(
            ax[k // rows][k % rows],
            denormalize(img, *self.norm_stats),
            caption,
            show_coords,
        )


def show_results(self, rows, **kwargs):
    figsize = kwargs.get("figsize", (20, rows * 5))
    return_fig = kwargs.get("return_fig", False)
    show_coords = kwargs.get("show_coords", False)
    beam_width = kwargs.get("beam_width", 3)
    max_len = kwargs.get("max_len", 15)
    fig, ax = plt.subplots(rows, 2, figsize=figsize, squeeze=False)
    top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
    plt.subplots_adjust(top=top)
    fig.suptitle("Ground Truth / Prediction", fontsize=16)
    nbatches = (rows // self._data.valid_dl.batch_size) + 1
    dls = get_nbatches(self._data.valid_dl, nbatches)
    images, captions = [], []
    captions_gts = []

    # concatenating multiple batches after doing predictions.
    for x_in, current_caption_gt in zip(*dls):
        captions_gts.extend(
            [cap.tolist()[: length.item()] for cap, length in zip(*current_caption_gt)]
        )
        current_images, current_captions, _ = self.learn.model.sample(
            x_in, beam_width=beam_width, max_len=max_len
        )
        images.append(current_images)
        captions.extend(current_captions)
    images = torch.cat(images).cpu()
    # iterate through each image and caption print.
    for k, img in enumerate(images):
        # no need to textify predictions because they are already in text format
        caption_pred = captions[k]
        caption_gt = captions_gts[k]
        # textify gt because its still tokenized.
        caption_gt = self._data.vocab.textify(caption_gt[1:-1])
        # use matplotlib for display
        show_image_and_text(
            ax[k][0], denormalize(img, *self._data.norm_stats), caption_gt, show_coords
        )
        show_image_and_text(
            ax[k][1],
            denormalize(img, *self._data.norm_stats),
            caption_pred,
            show_coords,
        )

        if k + 1 == rows:
            break

    # delete empty plots in case of small datasets.
    if k + 1 < rows:
        for i in range(k + 1, rows):
            fig.delaxes(ax[i][0])
            fig.delaxes(ax[i][1])

    if return_fig:
        return fig
