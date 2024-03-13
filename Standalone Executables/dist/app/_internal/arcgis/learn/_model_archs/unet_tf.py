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
    ZeroPadding2D,
    Cropping2D,
    SeparableConv2D,
)
from tensorflow.keras import Model
from .._utils.common_tf import (
    get_padding_info,
    get_cropping_info,
    get_skip_connection_info,
    get_channel_axis,
    UpSample2DToSize,
)


def get_unet_tf_head_output(
    backbone_initalized,
    data_format,
    n_output_classes,
    skip_lyr_idxs,
    skip_lyr_channels,
    mobile_optimized=False,
):
    # Get channel axis
    ca = channel_axis = get_channel_axis(data_format=data_format)

    # Mobile Optimized convolutions
    bottle_bottleneck_ratio = 2
    bottle_out_ratio = 1
    if mobile_optimized:
        conv_obj = SeparableConv2D
        bottle_bottleneck_ratio = 0.7
        bottle_out_ratio = 0.5
    else:
        conv_obj = Conv2D

    ### Unet Middle Block - Bottleneck ###
    # conv block 1
    bottle_channels = backbone_initalized.output.shape[ca]
    bottle_block_conv1_bn0 = BatchNormalization(
        axis=channel_axis, name=f"bottle_block_conv1_bn0"
    )(backbone_initalized.output)
    bottle_block_conv1 = conv_obj(
        round(bottle_channels * bottle_bottleneck_ratio),
        3,
        data_format=data_format,
        padding="same",
        activation="relu",
        name=f"bottle_block_conv1",
    )(bottle_block_conv1_bn0)
    bottle_block_conv1_bn1 = BatchNormalization(
        axis=channel_axis, name=f"bottle_block_conv1_bn1"
    )(bottle_block_conv1)

    # conv block 3
    bottle_block_conv3 = conv_obj(
        round(bottle_channels * bottle_out_ratio),
        3,
        data_format=data_format,
        padding="same",
        activation="relu",
        name=f"bottle_block_conv3",
    )(
        bottle_block_conv1_bn1
    )  # (bottle_block_conv2_drop1)
    bottle_block_conv3_bn1 = BatchNormalization(
        axis=channel_axis, name=f"bottle_block_conv3_bn1"
    )(bottle_block_conv3)
    bottle_block_conv3_drop1 = Dropout(0.3, name="bottle_block_conv3_drop1")(
        bottle_block_conv3_bn1
    )

    block_output_layer = bottle_block_conv3_drop1

    ### Decoder Block ###
    for i, idx in enumerate(skip_lyr_idxs):
        # decoder block i

        skip_con_output = backbone_initalized.layers[idx].output
        block_in_channels = block_output_layer.shape[ca]
        conv1_channels = round(block_in_channels / 2)
        conv2_channels = conv3_channels = round(
            (block_in_channels + skip_lyr_channels[i]) / 2
        )
        if idx == skip_lyr_idxs[-1]:
            conv_obj = Conv2D
            conv1_channels = block_in_channels
            if not mobile_optimized:
                conv2_channels = round(conv2_channels / 2)
        else:
            if mobile_optimized:
                conv2_channels = round((conv1_channels + skip_lyr_channels[i]) / 2)
        skip_con_size = skip_con_output.shape[2]

        # make input channels equal to target
        decoder_block_conv1 = SeparableConv2D(
            conv1_channels,
            3,
            data_format=data_format,
            padding="same",
            activation="relu",
            name=f"decoder_block{i}_conv1",
        )(block_output_layer)
        decoder_block_bn1 = BatchNormalization(
            axis=channel_axis, name=f"decoder_block{i}_bn1"
        )(decoder_block_conv1)

        # Resize input
        decoder_block_upsample1 = UpSample2DToSize(
            skip_con_size, name=f"decoder_block{i}_resize1"
        )(decoder_block_bn1)

        # Concatenate data from skip connection
        decoder_block_concat1 = Concatenate(
            axis=channel_axis, name=f"decoder_block{i}_concat1"
        )([decoder_block_upsample1, skip_con_output])

        # first conv on concatenated data
        decoder_block_conv2 = conv_obj(
            conv2_channels,
            3,
            data_format=data_format,
            padding="same",
            activation="relu",
            name=f"decoder_block{i}_conv2",
        )(decoder_block_concat1)
        decoder_block_bn2 = BatchNormalization(
            axis=channel_axis, name=f"decoder_block{i}_bn2"
        )(decoder_block_conv2)

        if idx == skip_lyr_idxs[-1]:
            block_output_layer = decoder_block_bn2
        else:
            # second conv on concatenated data
            decoder_block_conv3 = conv_obj(
                conv3_channels,
                3,
                data_format=data_format,
                padding="same",
                activation="relu",
                name=f"decoder_block{i}_conv3",
            )(decoder_block_bn2)
            decoder_block_bn3 = BatchNormalization(
                axis=channel_axis, name=f"decoder_block{i}_bn3"
            )(decoder_block_conv3)
            block_output_layer = decoder_block_bn3
        #
    decoder_block_output = block_output_layer

    # Final Classification Conv
    class_block_drop1 = Dropout(0.3, name="class_block_drop1")(decoder_block_output)
    class_block_conv1 = Conv2D(
        n_output_classes,
        3,
        data_format=data_format,
        padding="same",
        activation=None,
        name=f"class_block_conv1",
    )(class_block_drop1)

    # Cropping Block ##
    top_crop, bottom_crop, left_crop, right_crop = get_cropping_info(
        decoder_block_output, backbone_initalized.input, data_format=data_format
    )
    crop_block_crop1 = Cropping2D(
        cropping=((top_crop, bottom_crop), (left_crop, right_crop)),
        data_format=data_format,
        name=f"crop_block_crop1",
    )(class_block_conv1)

    model_output_layer = crop_block_crop1

    return model_output_layer


def get_unet_tf_model(backbone_initalized, data, mobile_optimized=False):
    """
    creates a U-net model from the supplied backbone
    that can be used with the supplied data.
    """
    # Get Data Format
    data_format = (
        tf.keras.backend.image_data_format()
    )  #'channels_first' or 'channels_last'

    # get skip connection information
    skip_lyr_idxs, skip_lyr_channels = get_skip_connection_info(
        backbone_initalized, data_format=data_format
    )

    # get head output
    head_output_layer = get_unet_tf_head_output(
        backbone_initalized=backbone_initalized,
        data_format=data_format,
        n_output_classes=data.c,
        skip_lyr_idxs=skip_lyr_idxs,
        skip_lyr_channels=skip_lyr_channels,
        mobile_optimized=mobile_optimized,
    )

    # Return Model
    return Model(inputs=backbone_initalized.input, outputs=head_output_layer)


def analyze_pred_unet_tf(activations):
    return tf.argmax(activations, axis=get_channel_axis())
