"""
These tools help answer one of the most common questions posed in spatial analysis: What is near what?

```create_buffers()``` creates areas of a specified distance from features.
"""
from __future__ import annotations
import json as _json

import logging as _logging
from typing import Any, Optional, Union
import arcgis as _arcgis
from arcgis.features import FeatureSet, FeatureCollection
from arcgis.features.layer import FeatureLayer, FeatureLayerCollection
from arcgis.geoprocessing._support import _execute_gp_tool
from arcgis.gis import GIS, Item
from ._util import (
    _id_generator,
    _feature_input,
    _set_context,
    _create_output_service,
    GAJob,
    _prevent_bds_item,
)
from arcgis._impl.common._utils import inspect_function_inputs
from arcgis.geoprocessing import import_toolbox

_log = _logging.getLogger(__name__)

_use_async = True


def group_by_proximity(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    spatial_relationship: str,
    spatial_near_distance: Optional[float] = None,
    spatial_near_distance_unit: Optional[str] = None,
    temporal_relationship: Optional[str] = None,
    temporal_near_distance: Optional[float] = None,
    temporal_near_distance_unit: Optional[str] = None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The Group By Proximity tool groups features that are within spatial
    proximity or spatiotemporal proximity of each other.

    ===================================================================    =============================================================================
    **Parameter**                                                           **Description**
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    input_layer                                                            Required layer. The point, line, or polygon features to be grouped.
                                                                           See :ref:`Feature Input<gaxFeatureInput>`.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    spatial_relationship                                                   Required String. The type of relationship to group by.

                                                                           Choice List:

                                                                             * ``Intersects``
                                                                             * ``Touches``
                                                                             * ``NearGeodesic``
                                                                             * ``NearPlanar``
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    spatial_near_distance                                                  Optional Float. A float value used for the search distance to determine if
                                                                           features are near one another. This is only applied if ``NearGeodesic`` or
                                                                           ``NewPlanar`` is the selected ``spatial_relationship``.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    spatial_near_distance_unit                                             Optional String. The linear unit to be used with the distance value specified
                                                                           in `spatial_near_distance`. The default value is Meters.

                                                                           Choice List:

                                                                             * ``Meters``
                                                                             * ``Kilometers``
                                                                             * ``Feet``
                                                                             * ``Miles``
                                                                             * ``NauticalMiles``
                                                                             * ``Yards``
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    temporal_relationship                                                  Optional String. The type of temporal relationship to group by.

                                                                           Choice List:

                                                                             * ``Intersects``
                                                                             * ``Near``
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    temporal_near_distance                                                 Optional Float. A float value used for the temporal search distance to determine if features are near one another.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    temporal_near_distance_unit                                            Optional String. The temporal unit to be used with the distance value specified in `temporal_near_distance`.

                                                                           Choice List:

                                                                             * ``Milliseconds``
                                                                             * ``Seconds``
                                                                             * ``Minutes``
                                                                             * ``Hours``
                                                                             * ``Days``
                                                                             * ``Weeks``
                                                                             * ``Months``
                                                                             * ``Years``
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    output_name                                                            Optional string. The task will create a feature service of the results. You define the name of the service.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    gis                                                                    Optional :class:`~arcgis.gis.GIS`. The GIS object where the analysis will take place.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    context                                                                Optional string. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                                                                            * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                                            * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                                                            * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                                            * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    future                                                                 Optional Boolean. If ``True``, a GAJob is returned instead of results. The GAJob can be queried on the status of the execution.
    ===================================================================    =============================================================================

    :return:
        * :class:`~arcgis.gis.Item` when `future=False`
        * ``GAJob`` when `future=True`

    """
    input_features = _prevent_bds_item(input_layer)

    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = import_toolbox(url, gis=gis)

    if output_name is None:
        output_service_name = _id_generator(prefix="Group_By_Proximity_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Group By Proximity",
        output_datastore=output_datastore,
    )

    params = {
        "input_features": input_features,
        "spatial_relationship": spatial_relationship,
        "spatial_near_distance": spatial_near_distance,
        "spatial_near_distance_unit": spatial_near_distance_unit,
        "temporal_relationship": temporal_relationship,
        "temporal_near_distance": temporal_near_distance,
        "temporal_near_distance_unit": temporal_near_distance_unit,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": True,
    }

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    kwargs = {}
    for key, value in params.items():
        if key != "field":
            if value is not None:
                kwargs[key] = value
        elif key == "field" and value:
            kwargs[key] = value
    params = inspect_function_inputs(tbx.group_by_proximity, **kwargs)
    params["future"] = True

    try:
        gpjob = tbx.group_by_proximity(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except Exception as e:
        output_service.delete()
        raise
    return


def trace_proximity_events(
    input_points: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    spatial_search_distance: float,
    spatial_search_distance_unit: str,
    temporal_search_distance: float,
    temporal_search_distance_unit: str,
    entity_id_field: str,
    entities_of_interest_ids: Optional[list[str]] = None,
    entities_of_interest_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    distance_method: str = "Planar",
    include_tracks_layer: bool = False,
    max_trace_depth: Optional[int] = None,
    attribute_match_criteria: Optional[str] = None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The Trace Proximity Events task analyzes time-enabled point features representing moving entities.
    The task will follow entities of interest in space (location) and time to see which other entities
    the entities of interest have interacted with. The trace will continue from entity to entity to a
    configurable maximum degrees of separation from the original entity of interest.


    ===================================================================    =============================================================================
    **Parameter**                                                                                    **Description**
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    input_points                                                           Required Layer. A layer that will be used in analysis.
                                                                           See :ref:`Feature Input<gaxFeatureInput>`.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    spatial_search_distance                                                Required Float. The maximum distance between two points to be considered in
                                                                           proximity. Features closer together in space and that also meet
                                                                           `temporal_search_distance` criteria are considered in proximity of each other.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    spatial_search_distance_unit                                           Required String. The unit of of measure for `spatial_search_distance`.

                                                                           Choice List:

                                                                             * ``Meters``
                                                                             * ``Kilometers``
                                                                             * ``Feet``
                                                                             * ``Miles``
                                                                             * ``NauticalMiles``
                                                                             * ``Yards``
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    temporal_search_distance                                               Required Float. The maximum duration between two points that are considered
                                                                           in proximity. Features closer together in time and that also meet the
                                                                           ``spatial_search_distance`` criteria are considered in proximity of each other.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    temporal_search_distance_unit                                          Required String. The unit of `temporal_search_distance`.

                                                                           Choice List:

                                                                             * ``Milliseconds``
                                                                             * ``Seconds``
                                                                             * ``Minutes``
                                                                             * ``Hours``
                                                                             * ``Days``
                                                                             * ``Weeks``
                                                                             * ``Months``
                                                                             * ``Years``
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    entity_id_field                                                        Required String. The field used to identify distinct entities.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    entities_of_interest_ids                                               Optional List. JSON used to specify one or more entities that you are
                                                                           interested in tracing from. You can optionally include a time to start tracing
                                                                           from. If you do not specify a time, January 1, 1970, at 12:00 a.m. will be used.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    entities_of_interest_layer                                             Optional Layer. A feature class used to specify one or more entities that you
                                                                           are interested in tracing from.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    distance_method                                                        Required String. The distance type that will be used for the ``spatial_search_distance``.
                                                                           The default is ``Planar``.

                                                                           Choice List:

                                                                             * ``Planar``
                                                                             * ``Geodesic``
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    include_tracks_layer                                                   Optional Boolean. Determines whether or not an additional layer will be
                                                                           created containing the first trace event in tracks and all subsequent
                                                                           features. The default is ``False``.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    max_trace_depth                                                        Optional Integer. The maximum degrees of separation between an entity of
                                                                           interest and an entity further down the trace.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    attribute_match_criteria                                               Optional String. One or more attributes used to constrain the proximity
                                                                           events. Entities will only be considered near when the `spatial_search_distance`
                                                                           and `temporal_search_distance` criteria are met and the two entities have
                                                                           equal values of the attributes specified.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    output_name                                                            Optional string. The task will create a feature service of the results. You define the name of the service.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    gis                                                                    Optional :class:`~arcgis.gis.GIS`. The GIS object where the analysis will take place.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    context                                                                Optional string. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                                                                             * ``extent`` - a bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                                             * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                                                             * ``outSR`` - the features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                                             * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    future                                                                 optional Boolean. If ``True``, a GPJob is returned instead of results.
                                                                           The GPJob can be queried on the status of the execution.
    ===================================================================    =============================================================================

    :return:

        * :class:`~arcgis.gis.Item` when `future=False`
        * ``GAJob`` when `future=True`

    """
    input_points = _prevent_bds_item(input_points)

    if (
        isinstance(input_points, FeatureCollection)
        and "layers" in input_points.properties
        and len(input_points.properties.layers) > 0
    ):
        input_points = FeatureSet.from_dict(
            featureset_dict=input_points._lazy_properties.layers[0].featureSet
        )
    if (
        isinstance(entities_of_interest_layer, FeatureCollection)
        and "layers" in entities_of_interest_layer.properties
        and len(entities_of_interest_layer.layers) > 0
    ):
        entities_of_interest_layer = FeatureSet.from_dict(
            featureset_dict=entities_of_interest_layer._lazy_properties.layers[
                0
            ].featureSet
        )

    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = import_toolbox(url_or_item=url, gis=gis)

    if output_name is None:
        output_service_name = _id_generator(prefix="Trace Proximity Events_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Trace Proximity Events",
        output_datastore=output_datastore,
    )

    params = {
        "input_points": input_points,
        "entity_id_field": entity_id_field,
        "entities_of_interest_ids": entities_of_interest_ids,
        "entities_of_interest_layer": entities_of_interest_layer or "",
        "distance_method": distance_method or "Planar",
        "spatial_search_distance": spatial_search_distance,
        "spatial_search_distance_unit": spatial_search_distance_unit,
        "temporal_search_distance": temporal_search_distance,
        "temporal_search_distance_unit": temporal_search_distance_unit,
        "include_tracks_layer": include_tracks_layer,
        "max_trace_depth": max_trace_depth,
        "attribute_match_criteria": attribute_match_criteria,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": True,
    }

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    kwargs = {}
    for key, value in params.items():
        if key != "field":
            if value is not None:
                kwargs[key] = value
        elif key == "field" and value:
            kwargs[key] = value
    params = inspect_function_inputs(tbx.trace_proximity_events, **kwargs)
    params["future"] = True

    try:
        gpjob = tbx.trace_proximity_events(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise

    return


def create_buffers(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    distance: Optional[float] = None,
    distance_unit: Optional[str] = None,
    field: Optional[str] = None,
    method: str = "Planar",
    dissolve_option: str = "None",
    dissolve_fields: Optional[str] = None,
    summary_fields: Optional[list[dict[str, Any]]] = None,
    multipart: bool = False,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """

    .. image:: _static/images/create_buffers_geo/create_buffers_geo.png

    Buffers are typically used to create areas that can be further analyzed
    using other tools such as ``aggregate_points``. For example, ask the question,
    "What buildings are within one mile of the school?" The answer can be found
    by creating a one-mile buffer around the school and overlaying the buffer
    with the layer containing building footprints. The end result is a layer
    of those buildings within one mile of the school.

    ================================================    =========================================================
    **Parameter**                                       **Description**
    ------------------------------------------------    ---------------------------------------------------------
    input_layer                                         Required layer. The point, line, or polygon features to be buffered.
                                                        See :ref:`Feature Input<gaxFeatureInput>`.
    ------------------------------------------------    ---------------------------------------------------------
    distance (Required if field is not provided)        Optional float. A float value used to buffer the input features.
                                                        You must supply a value for either the distance or field parameter.
                                                        You can only enter a single distance value. The units of the
                                                        distance value are supplied by the ``distance_unit`` parameter.

                                                        The default value is 1 when `field` is None.
    ------------------------------------------------    ---------------------------------------------------------
    distance_unit (Required if distance is used)        Optional string. The linear unit to be used with the value specified in distance.

                                                        Choice list:

                                                          * ``Feet``
                                                          * ``Yards``
                                                          * ``Miles``
                                                          * ``Meters``
                                                          * ``Kilometers``
                                                          * ``NauticalMiles``

                                                        The default value is ``Miles`` when `field` is None.
    ------------------------------------------------    ---------------------------------------------------------
    field (Required if distance not provided)           Optional string. A field on the ``input_layer`` containing a buffer distance or a field expression.
                                                        A buffer expression must begin with an equal sign (=). To learn more about buffer expressions
                                                        see: `Buffer Expressions <https://developers.arcgis.com/rest/services-reference/bufferexpressions.htm>`_
    ------------------------------------------------    ---------------------------------------------------------
    method                                              Optional string. The method used to apply the buffer with. There are two methods to choose from:

                                                        Choice list:['Geodesic', 'Planar']

                                                        * ``Planar`` - This method applies a Euclidean buffers and is appropriate for local analysis on projected data. This is the default.
                                                        * ``Geodesic`` - This method is appropriate for large areas and any geographic coordinate system.
    ------------------------------------------------    ---------------------------------------------------------
    dissolve_option                                     Optional string. Determines how output polygon attributes are processed.

                                                        Choice list:

                                                          * ``All``
                                                          * ``List``
                                                          * ``None``

                                                        +--------------------------------------+---------------------------------------------------------------------------------------------------+
                                                        |Value                                 | Description                                                                                       |
                                                        +--------------------------------------+---------------------------------------------------------------------------------------------------+
                                                        | ``All`` - All features are dissolved | You can calculate summary statistics and determine if you want multipart or single part features. |
                                                        | into one feature.                    |                                                                                                   |
                                                        +--------------------------------------+---------------------------------------------------------------------------------------------------+
                                                        | ``List`` - Features with the same    | You can calculate summary statistics and determine if you want multipart or single part features. |
                                                        | value in the specified field         |                                                                                                   |
                                                        | will be dissolve together.           |                                                                                                   |
                                                        +--------------------------------------+---------------------------------------------------------------------------------------------------+
                                                        | ``None`` - No features are dissolved.| There are no additional dissolve options.                                                         |
                                                        +--------------------------------------+---------------------------------------------------------------------------------------------------+
    ------------------------------------------------    ---------------------------------------------------------
    dissolve_fields                                     Specifies the fields to dissolve on. Multiple fields may be provided.
    ------------------------------------------------    ---------------------------------------------------------
    summary_fields                                      Optional string. A list of field names and statistical summary types
                                                        that you want to calculate for resulting polygons. Summary statistics
                                                        are only available if dissolveOption = List or All. By default, all
                                                        statistics are returned.

                                                        Example: ``[{"statisticType": "statistic type", "onStatisticField": "field name"}]``

                                                          * ``onStatisticField`` is the name of the field in the input point layer.

                                                          * ``statisticType`` is one of the following for numeric fields:

                                                            * ``Count`` - Totals the number of values of all the points in each polygon.
                                                            * ``Sum`` - Adds the total value of all the points in each polygon.
                                                            * ``Mean`` - Calculates the average of all the points in each polygon.
                                                            * ``Min`` - Finds the smallest value of all the points in each polygon.
                                                            * ``Max`` - Finds the largest value of all the points in each polygon.
                                                            * ``Range`` - Finds the difference between the Min and Max values.
                                                            * ``Stddev`` - Finds the standard deviation of all the points in each polygon.
                                                            * ``Var`` - Finds the variance of all the points in each polygon.

                                                          * ``statisticType`` is one of the following for string fields:

                                                            * ``Count`` - Totals the number of strings for all the points in each polygon.
                                                            * ``Any`` - Returns a sample string of a point in each polygon.

    ------------------------------------------------    ---------------------------------------------------------
    multipart                                           Optional boolean. Determines if output features are multipart or single part.
                                                        This option is only available if a ``dissolve_option`` is applied.
    ------------------------------------------------    ---------------------------------------------------------
    output_name                                         Optional string. The task will create a feature service of the results. You define the name of the service.
    ------------------------------------------------    ---------------------------------------------------------
    gis                                                 Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------------------    ---------------------------------------------------------
    context                                             Optional dict. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                                                          * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                          * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                                          * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                          * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    ------------------------------------------------    ---------------------------------------------------------
    future                                              Optional boolean. If ``True``, the value is returned as a GPJob.

                                                        The default value is ``False``
    ================================================    =========================================================

    :return:
        :class:`~arcgis.features.FeatureLayerCollection`

    .. code-block:: python

            # Usage Example: To create buffer based on distance field.

            buffer = create_buffers(input_layer=lyr,
                                    field='dist',
                                    method='Geodesic',
                                    dissolve_option='All',
                                    dissolve_fields='Date')
    """
    if field and distance:
        raise ValueError(
            "Both a distance and a field cannot be specified.  Choose one and resubmit."
        )
    if field is None and distance is None:
        distance = 1
    if distance and distance_unit is None:
        distance_unit = "Miles"

    input_layer = _prevent_bds_item(input_layer)

    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url

    if (
        isinstance(input_layer, FeatureCollection)
        and "layers" in input_layer.properties
        and len(input_layer.properties.layers) > 0
    ):
        input_layer = FeatureSet.from_dict(
            featureset_dict=input_layer._lazy_properties.layers[0].featureSet
        )
    kwargs = {
        "input_layer": input_layer,
        "distance": distance,
        "distance_unit": distance_unit,
        "field": field,
        "method": method,
        "dissolve_option": dissolve_option,
        "dissolve_fields": dissolve_fields,
        "summary_fields": summary_fields,
        "multipart": multipart,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": True,
    }
    params = {}
    for key, value in kwargs.items():
        if key != "field":
            if value is not None:
                params[key] = value
        elif key == "field" and value:
            params[key] = value
    if distance is None:
        params["distance"] = None
    if distance_unit is None:
        params["distance_unit"] = None
    if output_name is None:
        output_service_name = _id_generator(prefix="Create Buffers_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Create Buffers",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    tbx = import_toolbox(url_or_item=url, gis=gis)
    params = inspect_function_inputs(tbx.create_buffers, **params)
    params["future"] = True

    try:
        gpjob = tbx.create_buffers(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise
