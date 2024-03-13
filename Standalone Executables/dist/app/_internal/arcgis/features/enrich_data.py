"""
These functions help you explore the character of areas. Detailed demographic data and statistics are returned for your
chosen areas.

enrich_layer retrieves information about the people, places, and businesses in a specific area, or within a selected
travel time or distance from a location.
"""
from __future__ import annotations
from typing import Any, Optional, Union
from arcgis.auth.tools import LazyLoader

_util = LazyLoader("arcgis._impl.common._utils")
_arcgis = LazyLoader("arcgis")
network = LazyLoader("arcgis.network")
_features = LazyLoader("arcgis.features")


# --------------------------------------------------------------------------
def enrich_layer(
    input_layer: Union[
        _arcgis.gis.Item,
        _features.FeatureCollection,
        _features.FeatureLayer,
        _features.FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    data_collections: Optional[list[str]] = [],
    analysis_variables: Optional[list[str]] = [],
    country: Optional[str] = None,
    buffer_type: Optional[str] = None,
    distance: Optional[float] = None,
    units: Optional[str] = None,
    output_name: Optional[Union[_features.FeatureLayer, str]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[_arcgis.gis.GIS] = None,
    estimate: bool = False,
    return_boundaries: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/enrich_layer/enrich_layer.png

    The ``enrich_layer`` method enriches your data by getting facts about the people, places, and businesses that surround
    your data locations. For example: What kind of people live here? What do people like to do in this area? What are
    their habits and lifestyles? What kind of businesses are there in this area?

    The result will be a new layer of input features that includes all demographic and geographic information from given data collections.

    =====================================================================     ====================================================================
    **Parameter**                                                              **Description**
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    input_layer                                                               Required layer. The features to enrich with new data. See :ref:`Feature Input<FeatureInput>`.
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    data_collections                                                          Optional list of strings. This optional parameter defines the collections of data you want to use to enrich your features.
                                                                              Its value is a list of strings. If you don't provide this parameter, you must provide the analysis_variables parameter.

                                                                              For more information about data collections and the values for this parameter, visit the `Esri Demographics site <http://doc.arcgis.com/en/esri-demographics/data/data-browser.htm>`_.
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    analysis_variables                                                        Optional list of strings. The parameter defines the specific variables within a data collection you want to use to
                                                                              your features. Its value is a list of strings in the form of "dataCollection.VariableName". If you don't provide
                                                                              this parameter, you must provide the dataCollections parameter. You can provide both parameters.
                                                                              For example, if you want all variables in the KeyGlobalFacts data collection, specify it in the dataCollections
                                                                              parameter and use this parameter for specific variables in other collections.

                                                                              For more information about variables in data collections, visit the `Esri Demographics site <http://doc.arcgis.com/en/esri-demographics/data/data-browser.htm>`_. Each data collection
                                                                              has a PDF file describing variables and their names.
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    country                                                                   Optional string. This optional parameter further defines what is returned from data collection.
                                                                              For example, your input features may be countries in Western Europe, and you want to enrich them with the
                                                                              KeyWEFacts data collection. However, you only want data for France, not every country in your input layer.
                                                                              The value is the two-character country code.

                                                                              For more information about data collections and the values for this parameter, visit the `Esri Demographics site <http://doc.arcgis.com/en/esri-demographics/data/data-browser.htm>`_.
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    buffer_type (Required if input_layer contains point or line features)     Optional string. If your input features are points or lines, you must define an area around your features that you want to enrich. Features that are within (or equal to) the distances you enter will be enriched.

                                                                              Choice list: ['StraightLine', 'Driving Distance', 'Driving Time ', 'Rural Driving Distance', 'Rural Driving Time', 'Trucking Distance', 'Trucking Time', 'Walking Distance', 'Walking Time']
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    distance (Required if input_layer contains point or line features)        Optional float. A value that defines the search distance or time. The units of the distance value is supplied by the units parameter.
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    units                                                                     Optional string. The linear unit to be used with the distance value(s) specified in the distance parameter.

                                                                              Choice list: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles', 'Seconds', 'Minutes'. 'Hours']
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    output_name                                                               Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                                                              feature layer will cause the new layer to be appended to the Feature Service.
                                                                              If overwrite is True in context, new layer will overwrite existing layer.
                                                                              If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    context                                                                   Optional dict. Additional settings such as processing extent and output spatial reference.
                                                                              For enrich_layer, there are three settings.

                                                                              - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                                                              - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                                                              - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 11+


                                                                              .. code-block:: python

                                                                                # Example Usage

                                                                                                          "ymin": -9187921.892449,
                                                                                                          "xmax": 3174104.927313,
                                                                                                          "ymax": -9175500.875353,
                                                                                                          "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                                                                  "outSR": {"wkid": 3857},
                                                                                                  "overwrite": True}

    ---------------------------------------------------------------------     --------------------------------------------------------------------
    gis                                                                       Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    return_boundaries                                                         Optional boolean. Applies only for point and line input features. If True, a result layer of areas is returned.
                                                                              The returned areas are defined by the specified buffer_type. For example, if using a buffer_type of StraightLine with
                                                                              a distance of 5 miles, your result will contain areas with a 5 mile radius around the input features and requested
                                                                              analysis_variables variables. If False, the resulting layer will return the same features as the input layer with
                                                                              analysis_variables variables.

                                                                              The default value is False.
    ---------------------------------------------------------------------     --------------------------------------------------------------------
    future                                                                    Optional, If True, a future object will be returned and the process
                                                                              will not wait for the task to complete.
                                                                              The default is False, which means wait for results.
    =====================================================================     ====================================================================

    :returns :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection`.
    If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

        USAGE EXAMPLE: To enrich US block groups with population as analysis variable.
        blkgrp_enrich = enrich_layer(block_groups,
                                     analysis_variables=["AtRisk.MP27002A_B"],
                                     country='US',
                                     output_name='enrich layer')

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "input_layer": input_layer,
        "data_collections": data_collections,
        "analysis_variables": analysis_variables,
        "country": country,
        "buffer_type": buffer_type,
        "distance": distance,
        "units": units,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "return_boundaries": return_boundaries,
        "future": future,
    }

    params = _util.inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.enrich_layer, **kwargs
    )

    if isinstance(buffer_type, str):
        if buffer_type != "StraightLine":
            route_service = network.RouteLayer(
                gis.properties.helperServices.route.url, gis=gis
            )
            buffer_type = [
                i
                for i in route_service.retrieve_travel_modes()["supportedTravelModes"]
                if i["name"] == buffer_type
            ][0]
            if "buffer_type" in params:
                params["buffer_type"] = buffer_type
        else:
            pass

    return gis._tools.featureanalysis.enrich_layer(**params)
