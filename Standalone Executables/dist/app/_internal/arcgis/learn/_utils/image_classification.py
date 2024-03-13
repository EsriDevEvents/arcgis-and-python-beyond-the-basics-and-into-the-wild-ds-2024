import math
from itertools import compress
from .env import HAS_TENSORFLOW

if HAS_TENSORFLOW:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input,
        Conv2D,
        Dropout,
        ReLU,
        BatchNormalization,
        UpSampling2D,
        Reshape,
        Layer,
        AveragePooling2D,
        MaxPool2D,
        GlobalAveragePooling2D,
        GlobalMaxPool2D,
        Concatenate,
        Flatten,
        Dense,
    )
    from tensorflow.keras import Model
    from .._utils.fastai_tf_fit import (
        _tf_to_pytorch,
        _pytorch_to_tf_batch,
        _pytorch_to_tf,
    )
    from .common_tf import NormalizationLayerRGB

from .common import get_nbatches, image_batch_stretcher

try:
    import torch
    import fastai
    import numpy as np
    import matplotlib.pyplot as plt
    from fastai.vision import imagenet_stats
    import fastai
    from .._data import _extract_bands_tfm, _tensor_scaler, _tensor_scaler_tfm
    from .._data import _get_batch_stats, sniff_rgb_bands
    from .._utils.env import is_arcgispronotebook

    HAS_FASTAI = True
except:
    HAS_FASTAI = False

## Common section starts


def IC_show_results(self, nrows=5, **kwargs):
    type_data_loader = kwargs.get(
        "data_loader", "validation"
    )  # options : traininig, validation, testing
    if type_data_loader == "training":
        data_loader = self._data.train_dl
    elif type_data_loader == "validation":
        data_loader = self._data.valid_dl
    elif type_data_loader == "testing":
        data_loader = self._data.test_dl
    else:
        e = Exception(
            f"could not find {type_data_loader} in data. Please ensure that the data loader type is traininig, validation or testing "
        )
        raise (e)

    thresh = kwargs.get("thresh", 0.5)
    nodata = kwargs.get("nodata", 0)
    index = kwargs.get("start_index", 0)
    imsize = kwargs.get("imsize", 5)
    statistics_type = kwargs.get(
        "statistics_type", "dataset"
    )  # Accepted Values `dataset`, `DRA`
    stretch_type = kwargs.get(
        "stretch_type", "minmax"
    )  # Accepted Values `minmax`, `percentclip`

    # Get Batch
    n_items = min(nrows, len(data_loader.x))
    nbatches = math.ceil(n_items / self._data.batch_size)
    x_batch, y_batch = get_nbatches(data_loader, nbatches)
    x_batch = torch.cat(x_batch)
    y_batch = torch.cat(y_batch)

    # Get Predictions
    predictions_class_store = []
    predictions = []
    for i in range(0, x_batch.shape[0], self._data.batch_size):
        if self._backend == "pytorch":
            batch_preds = self.learn.pred_batch(
                batch=(
                    x_batch[i : i + self._data.batch_size],
                    y_batch[i : i + self._data.batch_size],
                )
            )
            # batch_preds: torch.tensor(B,C), where B is the batch size and C is the number of classes
            if self._data.dataset_type == "MultiLabeled_Tiles":
                predictions.append(batch_preds)
            else:
                confidences, class_idxs = torch.max(batch_preds, dim=1)
                predictions_class_store.extend(class_idxs)

        elif self._backend == "tensorflow":
            class_idxs, _confidences = TF_IC_predict_batch(
                self, x_batch[i : i + self._data.batch_size]
            )
            predictions_class_store.extend(class_idxs)

    # predictions will only hold values with Multilabel_Tiles
    # convert predictions from List[torch.tensor] to a torch.tensor
    if predictions:
        predictions = torch.cat(predictions)

    if self._is_multispectral:
        rgb_bands = kwargs.get("rgb_bands", self._data._symbology_rgb_bands)

        e = Exception(
            "`rgb_bands` should be a valid band_order, list or tuple of length 3 or 1."
        )
        symbology_bands = []
        if not (len(rgb_bands) == 3 or len(rgb_bands) == 1):
            raise (e)
        for b in rgb_bands:
            if type(b) == str:
                b_index = self._bands.index(b)
            elif type(b) == int:
                self._bands[
                    b
                ]  # To check if the band index specified by the user really exists.
                b_index = b
            else:
                raise (e)
            b_index = self._data._extract_bands.index(b_index)
            symbology_bands.append(b_index)

        # Denormalize X
        x_batch = (
            self._data._scaled_std_values[self._data._extract_bands]
            .view(1, -1, 1, 1)
            .to(x_batch)
            * x_batch
        ) + self._data._scaled_mean_values[self._data._extract_bands].view(
            1, -1, 1, 1
        ).to(
            x_batch
        )

        # Extract RGB Bands
        symbology_x_batch = x_batch[:, symbology_bands]
        if stretch_type is not None:
            symbology_x_batch = image_batch_stretcher(
                symbology_x_batch, stretch_type, statistics_type
            )

    else:
        # normalization stats
        norm_mean = torch.tensor(imagenet_stats[0]).to(x_batch).view(1, -1, 1, 1)
        norm_std = torch.tensor(imagenet_stats[1]).to(x_batch).view(1, -1, 1, 1)
        symbology_x_batch = (x_batch * norm_std) + norm_mean

    # Channel first to channel last for plotting
    symbology_x_batch = symbology_x_batch.permute(0, 2, 3, 1)

    # Clamp float values to range 0 - 1
    if symbology_x_batch.mean() < 1:
        symbology_x_batch = symbology_x_batch.clamp(0, 1)

    # Squeeze channels if single channel (1, 224, 224) -> (224, 224)
    if symbology_x_batch.shape[-1] == 1:
        symbology_x_batch = symbology_x_batch.squeeze()

    # Get color Array
    # color_array = self._data._multispectral_color_array

    # Handle Sparse Data (not to be applied with Multilabel data)
    if self._data.dataset_type != "MultiLabeled_Tiles" and y_batch.ndim > 1:
        y_batch = y_batch.max(-1)[1]

    # Plotting Ground Truth and Prediction side by side
    ncols = 2
    title_font_size = 16
    _top = 1 - (math.sqrt(title_font_size) / math.sqrt(100 * n_items * imsize))
    top = kwargs.get("top", _top)
    fig, axs = plt.subplots(
        nrows=n_items, ncols=ncols, figsize=(ncols * imsize, n_items * imsize)
    )
    fig.suptitle("Ground truth/Predictions", fontsize=title_font_size, weight="bold")
    plt.subplots_adjust(top=top)
    idx = 0
    for r in range(n_items):
        if n_items == 1:
            ax_i = axs
        else:
            ax_i = axs[r]

        # Get ground truth and prediction class names
        if self._data.dataset_type == "MultiLabeled_Tiles":
            one_hot_labels = y_batch[idx].tolist()
            gt_label = ";".join(compress(self._data.classes, one_hot_labels))
            one_hot_pred = (predictions[idx] >= thresh).tolist()
            prediction = ";".join(compress(self._data.classes, one_hot_pred))
        # For single label (Pytorch and TF)
        else:
            gt_label = self._data.classes[y_batch[idx].item()]
            prediction = self._data.classes[predictions_class_store[idx]]

        # Plot ground truth
        ax_ground_truth = ax_i[0]
        ax_ground_truth.axis("off")
        ax_ground_truth.imshow(symbology_x_batch[idx].cpu().numpy())
        ax_ground_truth.set_title(gt_label)

        # Plot Predictions
        ax_prediction = ax_i[1]
        ax_prediction.axis("off")
        ax_prediction.imshow(symbology_x_batch[idx].cpu().numpy())
        ax_prediction.set_title(prediction)

        idx += 1
    if is_arcgispronotebook():
        plt.show()
    return fig, axs


## Common section ends


## Tensorflow specific utils start ##


def TF_IC_predict_batch(self, imagetensor_batch):
    predictions = (
        self.learn.model(_pytorch_to_tf_batch(imagetensor_batch)).detach().numpy()
    )
    predictions_conf = (predictions.max(axis=-1) * 100).tolist()
    predicted_classes = predictions.argmax(axis=-1).tolist()
    return predicted_classes, predictions_conf


def TF_IC_get_head_output(arcgis_model_fc):
    backbone_name = arcgis_model_fc._backbone_initalized.name
    if "resnet" in backbone_name and any(
        x in backbone_name for x in ["50", "101", "151"]
    ):
        pyramid_scheme = [2048, 512]
    else:
        pyramid_scheme = [512]

    data_format = (
        tf.keras.backend.image_data_format()
    )  #'channels_first' or 'channels_last'
    if data_format == "channels_last":
        channel_axis = -1
    else:
        channel_axis = 1

    ## FC Head

    # Zero Block
    input_layer = arcgis_model_fc._backbone_initalized.output

    # First Block
    in_spatial_size = input_layer.shape[1]
    # head_block1_avgpool1 = AveragePooling2D(pool_size=(in_spatial_size, in_spatial_size), name="head_block1_globalavgpool1")(input_layer)
    # head_block1_avgpool1 = Flatten()(head_block1_avgpool1)
    head_block1_avgpool1 = GlobalAveragePooling2D(name="head_block1_globalavgpool1")(
        input_layer
    )

    # Because tflite (tensorflow 2.0.0) does not support REDUCE_MAX operation MaxPool2D is used and the output is
    # flatten that instead of directly using GlobalMaxPool2D
    # head_block1_maxpool1 = GlobalMaxPool2D(name="head_block1_globalmaxpool1")(input_layer)
    head_block1_maxpool1 = MaxPool2D(
        pool_size=(in_spatial_size, in_spatial_size), name="head_block1_globalmaxpool1"
    )(input_layer)
    head_block1_maxpool1 = Flatten()(head_block1_maxpool1)

    head_block1_concat1 = Concatenate(axis=channel_axis, name="head_block1_concat1")(
        [head_block1_avgpool1, head_block1_maxpool1]
    )

    # Out layer for further use
    out_layer = head_block1_concat1

    if arcgis_model_fc._fpn:
        # Fpn Block
        for pyramid_size in pyramid_scheme:
            fpn_dense = Dense(
                pyramid_size,
                activation="relu",
                name=f"head_fpnblock_dense_{pyramid_size}",
            )(out_layer)
            fpn_bn = BatchNormalization(
                axis=channel_axis, name=f"head_fpnblock_bn_{pyramid_size}"
            )(fpn_dense)
            out_layer = fpn_bn

    # Final Block
    head_final_drop1 = Dropout(0.3, name="head_final_drop1")(out_layer)
    head_final_logits_dense1 = Dense(
        arcgis_model_fc._data.c, name="head_final_logits_dense1"
    )(head_final_drop1)

    return head_final_logits_dense1


def get_TFIC_post_processed_model(arcgis_model, input_normalization=True):
    model = arcgis_model.learn.model
    input_layer = model.input
    model_output = model.output

    if input_normalization:
        input_layer = Input(tuple(input_layer.shape[1:]))
        x = NormalizationLayerRGB()(input_layer)
        model_output = model(x)
    output_layer = tf.nn.softmax(model_output, axis=-1)
    new_model = Model(input_layer, output_layer)
    return new_model


## Tensorflow specific utils end ##


## Funtions required for custom fastai databunch
def add_ms_attributes(data):
    if len(data.train_ds) < 300:
        norm_pct = 1
    else:
        norm_pct = 0.3

    # Statistics
    dummy_stats = {
        "batch_stats_for_norm_pct_0": {
            "band_min_values": None,
            "band_max_values": None,
            "band_mean_values": None,
            "band_std_values": None,
            "scaled_min_values": None,
            "scaled_max_values": None,
            "scaled_mean_values": None,
            "scaled_std_values": None,
        }
    }
    normstats_json_path = os.path.abspath(data.path / "esri_normalization_stats.json")
    if not os.path.exists(normstats_json_path):
        normstats = dummy_stats
        with open(normstats_json_path, "w", encoding="utf-8") as f:
            json.dump(normstats, f, ensure_ascii=False, indent=4)
    else:
        with open(normstats_json_path) as f:
            normstats = json.load(f)

    norm_pct_search = f"batch_stats_for_norm_pct_{round(norm_pct*100)}"
    if norm_pct_search in normstats:
        batch_stats = normstats[norm_pct_search]
        for s in batch_stats:
            if batch_stats[s] is not None:
                batch_stats[s] = torch.tensor(batch_stats[s])
    else:
        batch_stats = _get_batch_stats(
            data.x, norm_pct, _band_std_values=True, reshape=True
        )
        normstats[norm_pct_search] = dict(batch_stats)
        for s in normstats[norm_pct_search]:
            if normstats[norm_pct_search][s] is not None:
                normstats[norm_pct_search][s] = normstats[norm_pct_search][s].tolist()
        with open(normstats_json_path, "w", encoding="utf-8") as f:
            json.dump(normstats, f, ensure_ascii=False, indent=4)

    # batch_stats -> [band_min_values, band_max_values, band_mean_values, band_std_values, scaled_min_values, scaled_max_values, scaled_mean_values, scaled_std_values]
    data._band_min_values = batch_stats["band_min_values"]
    data._band_max_values = batch_stats["band_max_values"]
    data._band_mean_values = batch_stats["band_mean_values"]
    data._band_std_values = batch_stats["band_std_values"]
    data._scaled_min_values = batch_stats["scaled_min_values"]
    data._scaled_max_values = batch_stats["scaled_max_values"]
    data._scaled_mean_values = batch_stats["scaled_mean_values"]
    data._scaled_std_values = batch_stats["scaled_std_values"]

    # Prevent Divide by zeros
    data._band_max_values[data._band_min_values == data._band_max_values] += 1
    # data._scaled_std_values[data._scaled_std_values == 0]+=1e-02

    data = data.normalize(
        stats=(data._band_mean_values, data._band_std_values), do_x=True, do_y=False
    )

    data._do_normalize = True
    data._is_multispectral = True
    data._imagery_type = "multispectral"
    data._bands = list(range(data.x[0].shape[0]))
    data._extract_bands = list(range(data.x[0].shape[0]))
    data._symbology_rgb_bands = [0, 1, 2]
    data._train_tail = True


def adapt_fastai_databunch(data):
    # Case: from_model  method.
    if hasattr(data, "emd"):
        data._dataset_type = data.emd["MetaDataMode"]

    _is_ms = False
    # Case: from_model method
    if hasattr(data, "_is_multispectral"):
        _is_ms = data._is_multispectral

    if len(data.x.items) != 0:
        _is_ms = _is_ms or data.x[0].shape[0] != 3

    # Case: Customdatabunch RGB
    if not hasattr(data, "_dataset_type"):
        if isinstance(data.y[0], fastai.core.MultiCategory):
            data._dataset_type = "MultiLabeled_Tiles"
            data.dataset_type = data._dataset_type
        else:
            data._dataset_type = "Labeled_Tiles"
            data.dataset_type = data._dataset_type

    # Case: Cutomdatabunch Multispectral.
    if _is_ms:
        if hasattr(data, "stats"):
            # If dataset is already normalized just store the stats
            norm_stats = data.stats
            data._band_mean_values = torch.tensor(norm_stats[0])
            data._band_std_values = torch.tensor(norm_stats[1])
            data._scaled_mean_values = torch.tensor(norm_stats[0])
            data._scaled_std_values = torch.tensor(norm_stats[1])
            data._band_min_values = torch.tensor([0] * len(norm_stats[0]))
            data._band_max_values = torch.tensor([1] * len(norm_stats[0]))
            data._scaled_min_values = torch.tensor([0] * len(norm_stats[0]))
            data._scaled_max_values = torch.tensor([1] * len(norm_stats[0]))
            data._do_normalize = True
            data._is_multispectral = True
            data._imagery_type = "multispectral"
            data._bands = list(range(data.x[0].shape[0]))
            data._extract_bands = list(range(data.x[0].shape[0]))
            data._symbology_rgb_bands = [0, 1, 2]
            data._train_tail = True
        else:
            add_ms_attributes(data)

    if not hasattr(data, "chip_size"):
        data.chip_size = data.train_ds[0][0].shape[1]

    return data
