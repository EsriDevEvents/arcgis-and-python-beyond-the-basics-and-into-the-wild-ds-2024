import logging
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
    )
    from tensorflow.keras import Model
    from .._utils.fastai_tf_fit import (
        _tf_to_pytorch,
        _pytorch_to_tf_batch,
        _pytorch_to_tf,
    )
    from .common_tf import NormalizationLayerRGB, UpSample2DToSize

try:
    import torch
    import fastai
    import numpy as np
    from fastai.vision import imagenet_stats

    HAS_FASTAI = True
except:
    HAS_FASTAI = False


## Common start ##


## Common end ##

## Tensorflow specific utils start ##


def flatten_conv(output, n_anchors_per_cell, data_format):
    if data_format == "channels_last":
        x = output
    else:
        x = tf.transpose(output, perm=[0, 2, 3, 1])
    x = Reshape((-1, x.shape[-1] // n_anchors_per_cell))(x)
    return x


def get_ssd_head_output(ssd):
    data_format = (
        tf.keras.backend.image_data_format()
    )  #'channels_first' or 'channels_last'
    if data_format == "channels_last":
        bn_axis = -1
    else:
        bn_axis = 1

    n_anchors_per_cell = ssd._anchors_per_cell

    ## SSD Head

    # Zero Layer
    head_block0_drop1 = Dropout(0.3, name="head_block0_drop1")(
        ssd._backbone_initalized.output
    )

    # First Layer
    head_block1_conv1 = Conv2D(
        256,
        3,
        data_format=data_format,
        padding="same",
        activation="relu",
        name="head_block1_conv1",
    )(head_block0_drop1)
    head_block1_bn1 = BatchNormalization(axis=bn_axis, name="head_block1_bn1")(
        head_block1_conv1
    )
    head_block1_drop1 = Dropout(0.3, name="head_block1_drop1")(head_block1_bn1)

    outputs = []

    # Create Output Pipeline for each grid_size
    for grid in ssd.grids:
        # Grid upsampling Block
        head_block2_conv1 = Conv2D(
            256,
            3,
            data_format=data_format,
            padding="same",
            activation="relu",
            name=f"head_block2_conv1_{grid}",
        )(head_block1_drop1)
        head_block2_bn1 = BatchNormalization(
            axis=bn_axis, name=f"head_block2_bn1_{grid}"
        )(head_block2_conv1)
        head_block2_resize1 = UpSample2DToSize(
            grid, name=f"head_block2_resize1_{grid}"
        )(head_block2_bn1)
        head_block2_drop1 = Dropout(0.3, name=f"head_block2_drop1_{grid}")(
            head_block2_resize1
        )

        # Final Activations layer
        conv1 = Conv2D(
            ssd._data.c * n_anchors_per_cell,
            3,
            data_format=data_format,
            padding="same",
            name=f"head_block3_conv1_{grid}",
        )
        head_block3_conv1 = conv1(head_block2_drop1)
        conv2 = Conv2D(
            4 * n_anchors_per_cell,
            3,
            data_format=data_format,
            padding="same",
            name=f"head_block3_conv2_{grid}",
        )
        head_block3_conv2 = conv2(head_block2_drop1)

        # conv1.bias = conv1.bias - 4.

        head_block3_output1 = flatten_conv(
            head_block3_conv1, n_anchors_per_cell, data_format
        )
        head_block3_output2 = flatten_conv(
            head_block3_conv2, n_anchors_per_cell, data_format
        )

        outputs.append((head_block3_output1, head_block3_output2))

    output1 = tf.concat([a[0] for a in outputs], 1, name="concat1")
    output2 = tf.concat([a[1] for a in outputs], 1, name="concat2")
    return output1, output2


def remove_padded_y(image_y_bboxes, image_y_classes):
    try:
        bboxes = tf.reshape(image_y_bboxes, (-1, 4))
    except Exception:
        bboxes = tf.zeros((0, 4))
    # bboxes_to_keep = ( ( bboxes[:,2] - bboxes[:,0] ) != 0 ) tf 2.0.0
    bboxes_to_keep = tf.not_equal(bboxes[:, 2] - bboxes[:, 0], 0)
    return image_y_bboxes[bboxes_to_keep], image_y_classes[bboxes_to_keep]


def activations_to_bboxes(activations, anchors, grid_sizes):
    activation_bboxes = tf.tanh(activations)
    activation_centers = (activation_bboxes[:, :2] / 2 * grid_sizes) + anchors[:, :2]
    activation_height_widths = (activation_bboxes[:, 2:] / 2 + 1) * anchors[:, 2:]
    return height_width_to_corner(activation_centers, activation_height_widths)


def height_width_to_corner(centers, height_width):
    x = tf.concat([centers - (height_width / 2), centers + (height_width / 2)], axis=1)
    return x


def one_hot_embedding(labels, num_classes):
    return tf.gather(tf.eye(num_classes), labels)


def tf_loss_function_single_image(
    ssd, image_y_bboxes, image_y_classes, image_p_bboxes, image_p_classes
):
    image_y_bboxes, image_y_classes = remove_padded_y(image_y_bboxes, image_y_classes)

    # denormalize_bboxes
    image_y_bboxes = (image_y_bboxes + 1.0) / 2

    activation_bbox_corners = activations_to_bboxes(
        image_p_bboxes, ssd._anchors.cpu().numpy(), ssd._grid_sizes.cpu().numpy()
    )

    overlaps = ssd._jaccard(_tf_to_pytorch(image_y_bboxes), ssd._anchor_cnr.data)
    try:
        gt_overlap, gt_idx = ssd._map_to_ground_truth(overlaps, False)
        gt_idx = gt_idx.numpy()
    except Exception as e:
        return tf.constant(0.0), tf.constant(0.0)
    gt_clas = tf.gather(image_y_classes, gt_idx)
    pos = gt_overlap > 0.4
    pos_idx = torch.nonzero(pos)[:, 0].numpy()
    # gt_clas = tf.where( (pos.numpy()==0), 0, gt_clas) tf 2.0.0
    gt_clas = tf.compat.v2.where((pos.numpy() == 0), 0, gt_clas)
    ground_truth_classes = one_hot_embedding(gt_clas, ssd._data.c)
    gt_bbox = tf.gather(image_y_bboxes, gt_idx)

    location_loss = tf.reduce_sum(
        tf.abs(
            tf.gather(activation_bbox_corners, pos_idx) - tf.gather(gt_bbox, pos_idx)
        )
    )
    classification_loss = ssd._loss_function_classification(
        ground_truth_classes[:, 1:], image_p_classes[:, 1:]
    )
    return location_loss, classification_loss


class postProcessingLayer(Layer):
    def __init__(self, grid_sizes, anchors):
        super(postProcessingLayer, self).__init__()
        self.grid_sizes = grid_sizes
        self.anchors = anchors

    def __call__(self, preds):
        class_value_store = []
        confidence_store = []
        bbox_store = []
        batch_size = preds[0].shape[0]
        if batch_size is None:
            batch_size = 1
        for i in range(batch_size):
            class_values, confidence_scores, bboxes = TFOD_post_process_predictions(
                (preds[0][i], preds[1][i]), self.anchors, self.grid_sizes
            )
            class_value_store.append(class_values)
            confidence_store.append(confidence_scores)
            bbox_store.append(bboxes)
        return (
            tf.stack(class_value_store),
            tf.stack(confidence_store),
            tf.stack(bbox_store),
        )


def get_TFOD_post_processed_model(arcgis_model, input_normalization=True):
    model = arcgis_model.learn.model
    grid_sizes = _pytorch_to_tf(arcgis_model._grid_sizes)
    anchors = _pytorch_to_tf(arcgis_model._anchors)
    input_layer = model.input
    model_output = model.output

    if input_normalization:
        input_layer = Input(tuple(input_layer.shape[1:]))
        x = NormalizationLayerRGB()(input_layer)
        model_output = model(x)
    output_layer = postProcessingLayer(grid_sizes, anchors)(model_output)
    new_model = Model(input_layer, output_layer)
    return new_model


def TFOD_post_process_predictions(activations, anchors, grid_sizes):
    activation_classes, activation_bboxes = activations
    class_values = tf.argmax(activation_classes[:, 1:], axis=1)
    confidence_scores = tf.reduce_max(tf.sigmoid(activation_classes), axis=1)
    bboxes = TFOD_activations_to_bboxes(activation_bboxes, anchors, grid_sizes)
    return class_values, confidence_scores, bboxes


def TFOD_activations_to_bboxes(activation_bboxes, anchors, grid_sizes):
    activation_bboxes = tf.tanh(activation_bboxes)
    activation_centers = (activation_bboxes[:, :2] / 2 * grid_sizes) + anchors[:, :2]
    activation_hw = (activation_bboxes[:, 2:] / 2 + 1) * anchors[:, 2:]
    bboxes = tf.concat(
        [
            activation_centers - activation_hw / 2,
            activation_centers + activation_hw / 2,
        ],
        axis=1,
    )
    return bboxes


## Tensorflow specific utils end ##
