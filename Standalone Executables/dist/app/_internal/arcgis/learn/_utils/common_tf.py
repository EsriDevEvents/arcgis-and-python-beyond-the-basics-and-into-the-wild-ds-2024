from .env import HAS_TENSORFLOW

if HAS_TENSORFLOW:
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    from tensorflow.keras import applications
    import inspect
    from .._utils.fastai_tf_fit import _pytorch_to_tf_batch


## Layers Start ##
# Imagenet Stats
mean = tf.convert_to_tensor([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.convert_to_tensor([0.229, 0.224, 0.225], dtype=tf.float32)


class NormalizationLayerRGB(Layer):
    def __call__(self, input):
        x = (input + 1.0) / 2.0
        x = (x - mean) / std
        return x


class UpSample2DToSize(Layer):
    def __init__(self, output_size, name=None, **kwargs):
        super(UpSample2DToSize, self).__init__()
        if isinstance(output_size, (int, float)):
            output_size = (output_size, output_size)
        self.size = output_size

    def call(self, input):
        if tf.keras.backend.image_data_format() == "channels_first":
            x = tf.transpose(input, perm=[0, 2, 3, 1])
            x = self.resize(x)
            return tf.transpose(x, perm=[0, 3, 1, 2])
        else:
            return self.resize(input)

    def resize(self, input):
        return tf.image.resize(input, self.size)


## Layers END ##

## Backbone Functions Start ##
backbone_store = {}
for prop in dir(applications):
    prop_obj = getattr(applications, prop)
    if inspect.isfunction(prop_obj):
        backbone_store[prop.lower()] = prop_obj


def handle_backbone_parameter(backbone):
    if backbone is None:
        backbone = "ResNet50"
    if type(backbone) == str:
        backbone = get_backbone(backbone)
    return backbone


def get_backbone(backbone_name: str):
    return backbone_store[backbone_name.lower()]


_mobile_opt_backbones = ["MobileNetV2", "MobileNet"]
mobile_opt_backbones = [get_backbone(backbone) for backbone in _mobile_opt_backbones]


def check_backbone_is_mobile_optimized(backbone):
    return backbone in mobile_opt_backbones


## Backbone Functions end ##


def get_input_shape(chip_size=None, dl=None):
    if dl is not None:
        x, y = next(iter(dataloader))
        chip_size = x.shape[-1]
    if tf.keras.backend.image_data_format() == "channels_last":
        in_shape = [chip_size, chip_size, 3]
    else:
        in_shape = [3, chip_size, chip_size]
    return in_shape


def get_padding_info(op_layer, target_layer, data_format=None):
    """
    Returns the required padding info for matching two layers
    to same spatial size using ZeroPadding.

    Here op_layer is smaller than the target_layer in spatial size.
    """
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    if data_format == "channels_last":
        _, target_rows, target_cols, _ = target_layer.shape
        _, op_rows, op_cols, _ = op_layer.shape
    else:  # channel first
        _, _, target_rows, target_cols = target_layer.shape
        _, _, op_rows, op_cols = op_layer.shape

    top_pad = (target_rows - op_rows) // 2
    left_pad = (target_cols - op_cols) // 2
    bottom_pad = (target_rows - op_rows) - top_pad
    right_pad = (target_cols - op_cols) - left_pad

    return top_pad, bottom_pad, left_pad, right_pad


def get_cropping_info(op_layer, target_layer, data_format=None):
    """
    Returns the required cropping info for matching two layers
    to same spatial size using ZeroPadding.

    Here op_layer is bigger than the target_layer in spatial size.
    """
    top_crop, bottom_crop, left_crop, right_crop = get_padding_info(
        target_layer, op_layer, data_format=data_format
    )
    return top_crop, bottom_crop, left_crop, right_crop


def get_skip_connection_info(backbone_initalized, thresh=1.5, data_format=None):
    """
    Returns information about the possible skip connections
    that can be used by any U-Net type model architecture on
    the supplied backbone.
    """
    ca = get_channel_axis(data_format=data_format)
    skip_lyr_idxs = []
    skip_lyr_channels = []
    layers = backbone_initalized.layers
    input_shape = previous_shape = layers[0].input.shape
    for i, layer in enumerate(layers):
        new_shape = layer.output.shape
        if new_shape[1] * thresh < previous_shape[1]:
            skip_lyr_idxs.insert(0, i - 1)
            skip_lyr_channels.insert(0, previous_shape[ca])
        previous_shape = new_shape
    return skip_lyr_idxs, skip_lyr_channels


def get_channel_axis(data_format=None):
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    # Get channel axis
    if data_format == "channels_last":
        channel_axis = -1
    else:
        channel_axis = 1
    return channel_axis


def predict_batch_tf(self, imagetensor_batch):
    return self.learn.model(
        _pytorch_to_tf_batch(imagetensor_batch), training=False
    ).detach()
