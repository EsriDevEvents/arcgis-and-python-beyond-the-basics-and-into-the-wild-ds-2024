# Dask Mapping
"""
Mapping Holds the Plot function for creating a FeatureCollection JSON plus the render options
"""
import uuid
import json
import dask.dataframe as dd
from arcgis.widgets import MapView

###########################################################################
##  Helper Lambda
###########################################################################
_fn_method = lambda a, op, **kwargs: getattr(a, op)(**kwargs)


###########################################################################
def dask_plot(df, map_widget=None, renderer=None):
    """

    Plot draws the data on a web map. The user can describe in simple terms how to
    renderer spatial data using symbol.  To make the process simplier a pallette
    for which colors are drawn from can be used instead of explicit colors.


    ======================  =========================================================
    **Explicit Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    df                      required Dask DataFrame. This is the data to map.
    ----------------------  ---------------------------------------------------------
    map_widget              optional WebMap object. This is the map to display the
                            data on.
    ----------------------  ---------------------------------------------------------
    renderer                Optional dict-like.  The renderer definition for the dataset.
    ======================  =========================================================



    """
    renderer = None
    name = None
    map_exists = True
    if not hasattr(df, "spatial"):
        raise ValueError("DataFrame must be spatially enabled.")

    if df.spatial.renderer:
        renderer = json.loads(df.spatial.renderer.json)
    if name is None:
        name = uuid.uuid4().hex[:7]
    if map_widget is None:
        map_exists = False
        map_widget = MapView()
    assert isinstance(df, dd.DataFrame)

    feature_collections = df.map_partitions(
        lambda part: _fn_method(part.spatial, "to_feature_collection", **{"name": name})
    ).compute()
    if len(feature_collections) == 1:
        if map_exists:
            feature_collections[0].layer["layerDefinition"]["drawingInfo"][
                "renderer"
            ] = renderer
            map_widget.add_layer(feature_collections[0], options={"title": name})
        else:
            feature_collections[0].layer["layerDefinition"]["drawingInfo"][
                "renderer"
            ] = renderer
            map_widget.add_layer(feature_collections[0], options={"title": name})
    else:
        main_fc = feature_collections[0]
        main_fc.layer["layerDefinition"]["drawingInfo"]["renderer"] = renderer
        for fc in feature_collections[1:]:
            main_fc.properties["featureSet"]["features"].extend(
                fc.properties["featureSet"]["features"]
            )
            # fc.layer['layerDefinition']['drawingInfo']['renderer'] = renderer
        if map_exists:
            map_widget.add_layer(main_fc, options={"title": name})
        else:
            map_widget.add_layer(main_fc, options={"title": name})
    if map_exists == False:
        return map_widget
    return True
