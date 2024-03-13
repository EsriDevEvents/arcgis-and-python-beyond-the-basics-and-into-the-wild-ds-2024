""" Utility functions internally used by the store.
"""

from arcgis.features import FeatureSet
from ... import workforce


def _should_use_async_apply_edits(feature_layer):
    return (
        "advancedEditingCapabilities" in feature_layer.properties
        and "supportsAsyncApplyEdits"
        in feature_layer.properties["advancedEditingCapabilities"]
        and feature_layer.properties["advancedEditingCapabilities"][
            "supportsAsyncApplyEdits"
        ]
    )


def add_features(feature_layer, features, use_global_ids=False):
    """Adds features to the feature_layer.  The input features will be updated upon successful
    adding on the server, such that they contain the server-assigned object_ids and global_ids.
    :param feature_layer: An arcgis.features.FeatureLayer.
    :param features: list of arcgis.features.Features.
    :param use_global_ids: use global ids or not
    :returns: The added features.
    :raises ServerError: Indicates that the server rejected the new features.
    """
    if features:
        feature_set = FeatureSet(features)
        response = feature_layer.edit_features(
            adds=feature_set,
            use_global_ids=use_global_ids,
            future=_should_use_async_apply_edits(feature_layer),
        )
        if _should_use_async_apply_edits(feature_layer):
            add_results = response.result()[0]["addResults"]
        else:
            add_results = response["addResults"]
        errors = [result["error"] for result in add_results if not result["success"]]
        if errors:
            raise workforce.ServerError(errors)
        for feature, add_results in zip(features, add_results):
            feature.attributes[feature_layer.properties("objectIdField")] = add_results[
                "objectId"
            ]
            feature.attributes[feature_layer.properties("globalIdField")] = add_results[
                "globalId"
            ]
    return features


def update_features(feature_layer, features):
    """Updates features in a feature_layer.
    :param feature_layer: An arcgis.features.FeatureLayer.
    :param features: list of arcgis.features.Features.  Each feature must have an object id.
    :raises ServerError: Indicates that the server rejected the updates.
    """
    if features:
        response = feature_layer.edit_features(
            updates=FeatureSet(features),
            future=_should_use_async_apply_edits(feature_layer),
        )
        if _should_use_async_apply_edits(feature_layer):
            update_results = response.result()[0]["updateResults"]
        else:
            update_results = response["updateResults"]
        errors = [result["error"] for result in update_results if not result["success"]]
        if errors:
            raise workforce.ServerError(errors)
    return features


def remove_features(feature_layer, features):
    """Removes features from a feature_layer.
    :param feature_layer: An arcgis.features.FeatureLayer.
    :param features: list of arcgis.features.Features.  Each feature must have an object id.
    :raises ServerError: Indicates that the server rejected the removals.
    """
    if features:
        object_id_attr = feature_layer.properties["objectIdField"]
        object_ids = ",".join(
            [str(feature.attributes[object_id_attr]) for feature in features]
        )
        response = feature_layer.edit_features(
            deletes=object_ids, future=_should_use_async_apply_edits(feature_layer)
        )
        if _should_use_async_apply_edits(feature_layer):
            delete_results = response.result()[0]["deleteResults"]
        else:
            delete_results = response["deleteResults"]
        errors = [result["error"] for result in delete_results if not result["success"]]
        if errors:
            raise workforce.ServerError(errors)


def validate(validate_fn, **kwargs):
    """Runs a Model validation routine, and raises the first ValidationError if any are returned.
    :param validate_fn: A function that takes no positional arguments, and returns a list of
    ValidationErrors.  Any **kwargs given to this function will be passed to validate_fn.
    :raises ValidationError: Indicates that the validation_fn returned a ValidationError.
    """
    validation_failures = validate_fn(**kwargs)
    if validation_failures:
        raise validation_failures[0]
