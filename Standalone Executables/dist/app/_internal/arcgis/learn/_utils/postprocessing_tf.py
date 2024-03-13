from .env import HAS_TENSORFLOW

if HAS_TENSORFLOW:
    from .common_tf import NormalizationLayerRGB
    from tensorflow.keras.layers import Layer, Input
    from tensorflow.keras.models import Model
    from .object_detection import get_TFOD_post_processed_model
    from .image_classification import get_TFIC_post_processed_model
    from .pixel_classification import analyze_pred_TFPC


def get_post_processed_model_tf(arcgis_model, input_normalization=True):
    if arcgis_model.__class__.__name__ == "SingleShotDetector":
        return get_TFOD_post_processed_model(
            arcgis_model, input_normalization=input_normalization
        )
    elif arcgis_model.__class__.__name__ == "FeatureClassifier":
        return get_TFIC_post_processed_model(
            arcgis_model, input_normalization=input_normalization
        )
    elif arcgis_model.__class__.__name__ == "UnetClassifier":
        postprocessing_function = analyze_pred_TFPC

    model = arcgis_model.learn.model
    input_layer = model.input
    model_output = model.output

    if input_normalization:
        input_layer = Input(tuple(input_layer.shape[1:]))
        x = NormalizationLayerRGB()(input_layer)
        model_output = model(x)
    output_layer = postprocessing_function(model_output)
    new_model = Model(input_layer, output_layer)
    return new_model
