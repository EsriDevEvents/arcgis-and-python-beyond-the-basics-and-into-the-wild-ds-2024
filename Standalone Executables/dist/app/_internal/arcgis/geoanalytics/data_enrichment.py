"""

These tools are used for data enrichment using geoanalytics

"""
from __future__ import annotations
from datetime import datetime
import json as _json
import logging as _logging
from typing import Any, Optional, Union
import arcgis as _arcgis
from arcgis._impl.common._utils import inspect_function_inputs
from arcgis.features.feature import FeatureCollection
from arcgis.features.layer import FeatureLayer, FeatureLayerCollection
from arcgis.geoprocessing import import_toolbox as _import_toolbox
from arcgis.geoanalytics._util import (
    _id_generator,
    _feature_input,
    _set_context,
    _create_output_service,
    GAJob,
    _prevent_bds_item,
)
from arcgis.gis import GIS, Item

_log = _logging.getLogger(__name__)

_use_async = True


def calculate_motion_statistics(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    track_fields: str,
    motion_statistics: str = "All",
    track_history_window: int = 3,
    idle_tol_dist: Optional[float] = None,
    idle_tol_unit: Optional[str] = None,
    idle_time_tol: Optional[float] = None,
    idle_time_tol_unit: Optional[str] = None,
    time_boundary_split: Optional[float] = None,
    split_unit: Optional[str] = None,
    time_bound_ref: Optional[datetime] = None,
    dist_method: str = "Geodesic",
    distance_unit: str = "Meters",
    duration_unit: str = "Seconds",
    speed_unit: str = "MetersPerSecond",
    accel_unit: str = "MetersPerSecondSquared",
    elev_unit: str = "meters",
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    context: Optional[dict[str, Any]] = None,
    future: bool = False,
):
    """
    The Calculate Motion Statistics task calculates motion statistics and
    descriptors for time-enabled points that represent one or more moving
    entities. Points are grouped together into tracks representing each
    entity using a unique identifier. Motion statistics are calculated at
    each point using one or more points in the track history. Calculations
    include summaries of distance traveled, duration, elevation, speed,
    acceleration, bearing, and idle status. The result is a new point layer
    enriched with the requested statistics.

    For example, a city is monitoring snowplow operations and wants to
    better understand vehicle movement. The Calculate Motion Statistics
    tool can be used to determine idle locations and time spent idling,
    average and maximum speeds over time, total distance covered, and other
    statistics.

    .. note::
        Only available at ArcGIS Enterprise 10.9 and later.

    ======================  ===============================================================
    **Parameter**            **Description**
    ----------------------  ---------------------------------------------------------------
    input_layer             Required layer. The time-enabled point features that will be
                            grouped into tracks and analyzed. The input layer must be of
                            time type instant. See :ref:`Feature Input<gaxFeatureInput>`.
    ----------------------  ---------------------------------------------------------------
    track_fields            Required String. The fields used to identify distinct tracks.
                            There can be multiple trackFields in seperated by commas.
    ----------------------  ---------------------------------------------------------------
    motion_statistics       Optional String. The type of motion statistics to calculated.
                            The allowed values are:
                             * ``distance``
                             * ``speed``
                             * ``acceleration``
                             * ``duration``
                             * ``elevation``
                             * ``slope``
                             * ``idle``
                             * ``bearing``
                             * ``all``

                             ``all`` is the default.
    ----------------------  ---------------------------------------------------------------
    track_history_window    Optional Integer. The number of observations (including the
                            current observation) that will be used when calculating summary
                            statistics that are not instantaneous. This includes minimum,
                            maximum, average, and total statistics. The default track history
                            window is **3**, which means that at each point in a track
                            summary statistic will be calculated using the current
                            observation and the previous three observations. This parameter
                            does not affect instantaneous statistics or idle classification.
    ----------------------  ---------------------------------------------------------------
    idle_tol_dist           Optional Float. Used along with ``idle_time_tol`` to decide if an
                            entity is idling. An entity is idling when it hasn't moved more
                            than this distance in at least the amount of time specified by
                            ``idle_time_tol``. The units of the time values are supplied by
                            the ``idle_tol_unit`` parameter.

                            This value is only used for statistics in the Idle group.
    ----------------------  ---------------------------------------------------------------
    idle_tol_unit           Optional String. The unit of distance for idle distance tolerance.
    ----------------------  ---------------------------------------------------------------
    idle_time_tol           Optional Float. The lead amount of time used to determine if someone is idling.
    ----------------------  ---------------------------------------------------------------
    idle_time_tol_unit      Optional String. The time tolerance unit for idling.
    ----------------------  ---------------------------------------------------------------
    time_boundary_split     Optional Float. A time boundary allows your to analyze values within a defined time span.
    ----------------------  ---------------------------------------------------------------
    split_unit              Optional String. The unit of time represented in the ``time_boundary_split``.
    ----------------------  ---------------------------------------------------------------
    time_bound_ref          Optional Datetime. A date that specifies the reference time to
                            align the time boundary to, represented in milliseconds from epoch.
    ----------------------  ---------------------------------------------------------------
    dist_method             Optional String. The method used to calculate distances between
                            track observations.

                            There are two methods to choose from:
                             * ``Planar`` - measures distances using an
                               Euclidean plane and will not calculate statistics across the
                               date line.
                             * ``Geodesic`` - calculations vary depending upon:

                               * If the spatial reference can be panned, calculations
                                 will cross the date line when appropriate
                               * If the spatial reference cannot be panned, calculations will be limited to the
                                 coordinate system extent and may not wrap.
    ----------------------  ---------------------------------------------------------------
    distance_unit           Optional String. The units for all results in the Distance
                            motion statistics group.

                            Options:
                             * ``Meters``
                             * ``Kilometers``
                             * ``Feet``
                             * ``Miles``
                             * ``NauticalMiles``
                             * ``Yards``

                            ``Meters`` is the default.
    ----------------------  ---------------------------------------------------------------
    duration_unit           Optional String. The units for all results in the Duration motion statistics group.

                            Options:
                                * ``Milliseconds``
                                * ``Seconds``
                                * ``Minutes``
                                * ``Hours``
                                * ``Days``
                                * ``Weeks``
                                * ``Months``
                                * ``Years``

                            ``Seconds`` is the default.
    ----------------------  ---------------------------------------------------------------
    speed_unit              Optional String. The units for all results in the Speed motion statistics group.

                            Options:
                             * ``MetersPerSecond`` (default)
                             * ``KilometersPerHour``
                             * ``FeetPerSecond``
                             * ``MilesPerHour``
                             * ``NauticalMilesPerHour``

                            ``MetersPerSecond`` is the default.
    ----------------------  ---------------------------------------------------------------
    accel_unit              Optional String. The units for all results in the Acceleration motion statistics group.

                            Options:
                              * ``MetersPerSecondSquared``
                              * ``FeetPerSecondSquared``

                            ``MetersPerSecondSquared`` is the default.
    ----------------------  ---------------------------------------------------------------
    elev_unit               Optional String. The units for all results in the Elevation motion statistics group.

                            Options:
                                * ``Meters``
                                * ``Kilometers``
                                * ``Feet``
                                * ``Miles``
                                * ``NauticalMiles``
                                * ``Yards``

                            ``Meters`` is the default.
    ----------------------  ---------------------------------------------------------------
    output_name             optional string. The task will create a feature service of the
                            results. You define the name of the service.
    ----------------------  ---------------------------------------------------------------
    gis                     optional :class:`~arcgis.gis.GIS`. The GIS object where the analysis will take place.
    ----------------------  ---------------------------------------------------------------
    context                 Optional dict. The context parameter contains additional settings that affect task execution. For this task, there are five settings:

                                * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
                                * ``defaultAggregationStyles`` - If set to 'True', results will have square, hexagon, and triangle aggregation styles enabled on results map services.
    ----------------------  ---------------------------------------------------------------
    future                  optional boolean. If ``True``, a GPJob is returned instead of
                            results. The GPJob can be queried on the status of the execution.

                            The default value is ``False``.
    ======================  ===============================================================

    :return:
        :class:`~arcgis.features.FeatureLayer`

    """

    input_layer = _prevent_bds_item(input_layer)
    tool_name = "CalculateMotionStatistics"
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)

    params = {
        "input_layer": input_layer,
        "track_fields": track_fields,
        "track_history_window": track_history_window,
        "motion_statistics": motion_statistics,
        "idle_distance_tolerance": idle_tol_dist,
        "idle_distance_tolerance_unit": idle_tol_unit,
        "idle_time_tolerance": idle_time_tol,
        "idle_time_tolerance_unit": idle_time_tol_unit,
        "time_boundary_split": time_boundary_split,
        "time_boundary_split_unit": split_unit,
        "time_boundary_reference": time_bound_ref,
        "distance_method": dist_method,
        "distance_unit": distance_unit,
        "duration_unit": duration_unit,
        "speed_unit": speed_unit,
        "acceleration_unit": accel_unit,
        "elevation_unit": elev_unit,
        "output_name": output_name,
        "context": context,
        "future": future,
    }

    if output_name is None:
        output_service_name = _id_generator(prefix="Calculate_Motion_Stats_")
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
        "Calculate Motion Statistics",
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
        params["output_name"] = output_service_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_service_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    for key in list(params.keys()):
        if params[key] is None:
            del params[key]
    param_db = inspect_function_inputs(tbx.calculate_motion_statistics, **params)
    param_db["future"] = True
    try:
        gpjob = tbx.calculate_motion_statistics(**param_db)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise


def enrich_from_grid(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    grid_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    enrichment_attributes: Optional[str] = None,
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    context: Optional[dict[str, Any]] = None,
    future: bool = False,
):
    """
    .. image:: _static/images/enrich_from_grid/enrich_from_grid.png

    The Enrich From Multi-Variable Grid task joins attributes from a multivariable grid to a point layer.
    The multivariable grid must be created using the ``build_multivariable_grid`` task. Metadata from the
    multivariable grid is used to efficiently enrich the input point features, making it faster than the
    Join Features task. Attributes in the multivariable grid are joined to the input point features when
    the features intersect the grid.

    The attributes in the multivariable grid can be used as explanatory variables when modeling spatial
    relationships with your input point features, and this task allows you to join those attributes to
    the point features quickly.

    .. note::
        Only available at ArcGIS Enterprise 10.7 and later.

    ======================  ===============================================================
    **Parameter**            **Description**
    ----------------------  ---------------------------------------------------------------
    input_layer             Required layer. The point features that will be enriched
                            by the multi-variable grid. See :ref:`Feature Input<gaxFeatureInput>`.
    ----------------------  ---------------------------------------------------------------
    grid_layer              Required layer. The multivariable grid layer created using the
                            `Build Multi-Variable Grid <https://enterprise.arcgis.com/en/portal/latest/use/geoanalytics-build-multi-variable-grid.htm>`_
                            tool.
                            See :ref:`Feature Input<gaxFeatureInput>`.
    ----------------------  ---------------------------------------------------------------
    enrichment_attributes   optional string. A list of fields in the multi-variable grid
                            that will be joined to the input point features. If the
                            attributes are not provided, all fields in the multi-variable
                            grid will be joined to the input point features.
    ----------------------  ---------------------------------------------------------------
    output_name             optional string. The task will create a feature service of the
                            results. You define the name of the service.
    ----------------------  ---------------------------------------------------------------
    gis                     optional :class:`~arcgis.gis.GIS`. The GIS object where the analysis will take place.
    ----------------------  ---------------------------------------------------------------
    context                 Optional dict. The context parameter contains additional settings that affect task execution. For this task, there are five settings:

                             * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                             * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                             * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
                             * ``defaultAggregationStyles`` - If set to 'True', results will have square, hexagon, and triangle aggregation styles enabled on results map services.
    ----------------------  ---------------------------------------------------------------
    future                  optional boolean. If ``True``, a GPJob is returned instead of
                            results. The GPJob can be queried on the status of the execution.

                            The default value is ``False``.
    ======================  ===============================================================

    :return:
        :class:`~arcgis.features.FeatureLayer`

    .. code-block:: python

            # Usage Example: To enrich a layer of crime data with a multivariable grid containing demographic information.

            enrich_result = enrich_from_grid(input_layer=crime_lyr,
                                             grid_layer=mvg_layer,
                                             output_name="chicago_crimes_enriched")


    """
    input_layer = _prevent_bds_item(input_layer)
    tool_name = "EnrichFromMultiVariableGrid"
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)

    params = {
        "input_features": input_layer,
        "grid_layer": grid_layer,
        "enrich_attributes": enrichment_attributes,
        "output_name": output_name,
        "context": context,
        "future": future,
        "gis": gis,
    }
    for key in list(params.keys()):
        if params[key] is None:
            del params[key]
    if (
        isinstance(params["grid_layer"], _arcgis.gis.Item)
        and "layers" in params["grid_layer"]
        and len(params["grid_layer"].layers) > 0
    ):
        params["grid_layer"] = params["grid_layer"].layers[0]._lyr_dict
    elif isinstance(params["grid_layer"], _arcgis.gis.Layer):
        params["grid_layer"] = params["grid_layer"]._lyr_dict

    params = inspect_function_inputs(tbx.enrich_from_multi_variable_grid, **params)
    params["future"] = True

    if output_name is None:
        output_service_name = _id_generator(prefix="Enrich_Grid_")
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
        "Enrich Grid Layers",
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
        params["output_name"] = output_service_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_service_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    try:
        gpjob = tbx.enrich_from_multi_variable_grid(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise
