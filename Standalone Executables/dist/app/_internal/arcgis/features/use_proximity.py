"""
These functions help you answer one of the most common questions posed in spatial analysis: "What is near what?"

connect_origins_to_destinations measures the travel time or distance between pairs of points.
create_buffers create areas of equal distance from features.
create_drive_time_areas finds areas around locations that can be reached within a time period.
find_nearest identifies those places that are the closest to known locations.
plan_routes determines the best way to route a fleet of vehicles to visit many stops.
"""
from __future__ import annotations
from datetime import datetime
import logging
from re import U
from typing import Any, Optional, Union
import arcgis as _arcgis
from arcgis._impl.common._utils import _date_handler
from arcgis.features.feature import FeatureCollection
from arcgis.features.layer import FeatureLayer, FeatureLayerCollection
from arcgis.gis import GIS, Item
import arcgis.network as network
from .._impl.common._utils import inspect_function_inputs
from arcgis import network

_logger = logging.getLogger()


# --------------------------------------------------------------------------
def connect_origins_to_destinations(
    origins_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    destinations_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    measurement_type: Optional[str] = None,
    origins_layer_route_id_field: Optional[str] = None,
    destinations_layer_route_id_field: Optional[str] = None,
    time_of_day: Optional[datetime] = None,
    time_zone_for_time_of_day: str = "GeoLocal",
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    point_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    line_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    polygon_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    future: bool = False,
    route_shape: str = "FollowStreets",
    include_route_layers: bool = False,
):
    """
    .. image:: _static/images/connect_origins_to_destinations/connect_origins_to_destinations.png

    The Connect Origins to Destinations task measures the travel time or distance between pairs of points. Using this tool, you can

    * Calculate the total distance or time commuters travel on their home-to-work trips.
    * Measure how far customers are traveling to shop at your stores. Use this information to define your market reach, especially when targeting advertising campaigns or choosing new store locations.
    * Calculate the expected trip mileage for your fleet of vehicles. Afterward, run the Summarize Within tool to report mileage by state or other region.

    You provide starting and ending points, and the tool returns a layer containing route lines, including measurements, between the
    paired origins and destinations.

    ===================================     ===============================================================
    **Parameter**                            **Description**
    -----------------------------------     ---------------------------------------------------------------
    origins_layer                           Required layer. The starting point or points of the
                                            routes to be generated. See :ref:`Feature Input<FeatureInput>`.
    -----------------------------------     ---------------------------------------------------------------
    destinations_layer                      Required layer. The routes end at points in the
                                            destinations layer. See :ref:`Feature Input<FeatureInput>`.
    -----------------------------------     ---------------------------------------------------------------
    measurement_type                        Required string. The origins and destinations can be connected by measuring straight-line distance,
                                            or by measuring travel time or travel distance along a street network using various modes of transportation known as travel modes.

                                            Valid values are a string, StraightLine, which indicates Euclidean distance to be used as distance measure or a Python dictionary representing settings for a travel mode.

                                            When using a travel mode for the measurement_type, you need to specify a dictionary
                                            containing the settings for a travel mode supported by your organization. The code in the example section below generates
                                            a valid Python dictionary and then passes it as the value for the measurement_type parameter.

                                            Supported travel modes: ['Driving Distance', 'Driving Time', 'Rural Driving Distance', 'Rural Driving Time',
                                            'Trucking Distance', 'Trucking Time', 'Walking Distance', 'Walking Time']
    -----------------------------------     ---------------------------------------------------------------
    origins_layer_route_id_field            Optional string. Specify the field in the origins layer
                                            containing the IDs that pair origins with destinations.

                                            * The ID values must uniquely identify points in the origins layer

                                            * Each ID value must also correspond with exactly one route ID value in the destinations layer.
                                              Route IDs that match across the layers create origin-destination pairs, which the tool connects
                                              together.

                                            * Specifying origins_layer_route_id_field is optional when there is exactly one point feature in
                                              the origins or destinations layer. The tool will connect all origins to the one destination or the
                                              one origin to all destinations, depending on which layer contains one point.

    -----------------------------------     ---------------------------------------------------------------
    destinations_layer_route_id_field       Optional string. Specify the field in the destinations layer containing the IDs that pair origins
                                            with destinations.

                                            * The ID values must uniquely identify points in the destinations layer.

                                            * Each ID value must also correspond with exactly one route ID value in the origins layer. Route
                                              IDs that match across the layers create origin-destination pairs, which the tool connects together.

                                            * Specifying destinations_layer_route_id_field is optional when there is exactly one point
                                              feature in the origins or destinations layer. The tool will connect all origins to the one
                                              destination or the one origin to all destinations, depending on which layer contains one point.
    -----------------------------------     ---------------------------------------------------------------
    time_of_day                             Optional datetime.datetime. Specify whether travel times should consider traffic conditions. To use
                                            traffic in the analysis,
                                            set measurement_type to a travel mode object whose impedance_attribute_name property is set to
                                            travel_time and assign a value
                                            to time_of_day. (A travel mode with other impedance_attribute_name values don't support traffic.) The time_of_day value represents
                                            the time at which travel begins, or departs, from the origin points. The time is specified as datetime.datetime.

                                            The service supports two kinds of traffic: typical and live. Typical traffic references travel speeds that are made up of historical
                                            averages for each five-minute interval spanning a week. Live traffic retrieves speeds from a traffic feed that processes phone probe
                                            records, sensors, and other data sources to record actual travel speeds and predict speeds for the near future.

                                            The `data coverage <http://www.arcgis.com/home/webmap/viewer.html?webmap=b7a893e8e1e04311bd925ea25cb8d7c7>`_ page shows the countries
                                            Esri currently provides traffic data for.

                                            Typical Traffic:

                                            To ensure the task uses typical traffic in locations where it is available, choose a time and day of the week, and then convert the day
                                            of the week to one of the following dates from 1990:

                                            * Monday - 1/1/1990
                                            * Tuesday - 1/2/1990
                                            * Wednesday - 1/3/1990
                                            * Thursday - 1/4/1990
                                            * Friday - 1/5/1990
                                            * Saturday - 1/6/1990
                                            * Sunday - 1/7/1990
                                            Set the time and date as datetime.datetime.

                                            For example, to solve for 1:03 p.m. on Thursdays, set the time and date to 1:03 p.m., 4 January 1990; and convert to
                                            datetime eg. datetime.datetime(1990, 1, 4, 1, 3).

                                            Live Traffic:

                                            To use live traffic when and where it is available, choose a time and date and convert to datetime.

                                            Esri saves live traffic data for 4 hours and references predictive data extending 4 hours into the future. If the time and date you
                                            specify for this parameter is outside the 24-hour time window, or the travel time in the analysis continues past the predictive data window, the task falls back to typical traffic speeds.

                                            # Examples:
                                            from datetime import datetime

                                            * "time_of_day": datetime(1990, 1, 4, 1, 3) # 13:03, 4 January 1990. Typical traffic on
                                              Thursdays at 1:03 p.m.
                                            * "time_of_day": datetime(1990, 1, 7, 17, 0) # 17:00, 7 January 1990. Typical traffic on Sundays at
                                              5:00 p.m.
                                            * "time_of_day": datetime(2014, 10, 22, 8, 0) # 8:00, 22 October 2014. If the current time is
                                              between 8:00 p.m., 21 Oct. 2014 and 8:00 p.m., 22 Oct. 2014,
                                              live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are
                                              referenced.
                                            * "time_of_day": datetime(2015, 3, 18, 10, 20) # 10:20, 18 March 2015. If the current time is
                                              between 10:20 p.m., 17 Mar. 2015 and 10:20 p.m., 18 Mar. 2015, live traffic speeds are
                                              referenced in the analysis; otherwise, typical traffic speeds are referenced.
    -----------------------------------     ---------------------------------------------------------------
    time_zone_for_time_of_day               Optional string. Specify the time zone or zones of the timeOfDay parameter.
                                            Choice list: ['GeoLocal', 'UTC']

                                            GeoLocal-refers to the time zone in which the originsLayer points are located.

                                            UTC-refers to Coordinated Universal Time.
    -----------------------------------     ---------------------------------------------------------------
    include_route_layers                    Optional Boolean. When include_route_layers is set to True,
                                            each route from the result is also saved as a route layer item.
                                            A route layer includes all the information for a particular route
                                            such as the stops assigned to the route as well as the travel directions.
                                            Creating route layers is useful if you want to share individual
                                            routes with other members in your organization.
                                            The route layers use the output feature service name provided in the ``output_name``
                                            parameter as a prefix and the route name generated as part of the analysis is added to create a
                                            unique name for each route layer.

                                            Caution: Route layers cannot be created when the output is a feature collection.
                                            The task will raise an error if output_name is not specified
                                            (which indicates feature collection output) and include_route_layers is True.

                                            The maximum number of route layers that can be created is 1,000.
                                            If the result contains more than 1,000 routes and include_route_layers is True,
                                            the task will only create the output feature service.
    -----------------------------------     ---------------------------------------------------------------
    output_name                             Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                            feature layer will cause the new layer to be appended to the Feature Service.
                                            If overwrite is True in context, new layer will overwrite existing layer.
                                            If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -----------------------------------     ---------------------------------------------------------------
    context                                 Optional dict. Additional settings such as processing extent
                                            and output spatial reference.
                                            For connect_origins_to_destinations, there are three settings.

                                            - ``extent`` - a bounding box that defines the analysis area. Only those features in the
                                              input_layer that intersect the bounding box will be analyzed.
                                            - ``outSR`` - the output features will be projected into the output spatial reference referred to
                                              by the `wkid`.
                                            - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new
                                              feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                                .. code-block:: python

                                                    # Example Usage
                                                    context = {"extent": {"xmin": 3164569.408035,
                                                                        "ymin": -9187921.892449,
                                                                        "xmax": 3174104.927313,
                                                                        "ymax": -9175500.875353,
                                                                        "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                                "outSR": {"wkid": 3857},
                                                                "overwrite": True}
    -----------------------------------     ---------------------------------------------------------------
    gis                                     Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified,
                                            the active GIS is used.
    -----------------------------------     ---------------------------------------------------------------
    estimate                                Optional Boolean. Is True, the number of credits needed
                                            to run the operation will be returned as a float.
    -----------------------------------     ---------------------------------------------------------------
    point_barrier_layer                     Optional layer. Specify one or more point features that
                                            act as temporary restrictions (in other words, barriers) when
                                            traveling on the underlying streets.

                                            A point barrier can model a fallen tree, an accident, a downed
                                            electrical line, or anything that completely blocks traffic at
                                            a specific position along the street. Travel is permitted on the
                                            street but not through the barrier. See :ref:`Feature Input<FeatureInput>`.
    -----------------------------------     ---------------------------------------------------------------
    line_barrier_layer                      Optional layer. Specify one or more line features that prohibit
                                            travel anywhere the lines intersect the streets.

                                            A line barrier prohibits travel anywhere the barrier intersects the
                                            streets. For example, a parade or protest that blocks traffic across
                                            several street
                                            segments can be modeled with a line barrier.
                                            See :ref:`Feature Input<FeatureInput>`.
    -----------------------------------     ---------------------------------------------------------------
    polygon_barrier_layer                   Optional string. Specify one or more polygon features
                                            that completely restrict travel on the streets intersected
                                            by the polygons.

                                            One use of this type of barrier is to model floods covering
                                            areas of the street network and making road travel there impossible.
                                            See :ref:`Feature Input<FeatureInput>`.
    -----------------------------------     ---------------------------------------------------------------
    future                                  Optional boolean. If True, a future object will be returned and the process
                                            will not wait for the task to complete. The default is False, which means wait for results.
    -----------------------------------     ---------------------------------------------------------------
    route_shape                             Optional String. Specify the shape of the route that connects
                                            each origin to it's destination when using a travel mode.

                                            Values: FollowStreets or StraightLine

                                            Default: FollowStreets

                                            + FollowStreets - The shape is based on the underlying street network.
                                              This option is best when you want to generate the routes between
                                              origins and destinations. This is the default value when using a
                                              travel mode.
                                            + StraightLine - The shape is a straight line connecting
                                              the origin-destination pair. This option is best when you want to g
                                              enerate spider diagrams or desire lines (for example, to show which
                                              stores customers are visiting). This is the default value when not using
                                              a travel mode.

                                            The best route between an origin and it's matched destination is always calculated based on the travel mode, regardless of which route shape is chosen.
    ===================================     ===============================================================


    :return: A dictionary with the following keys:

        "routes_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

        "unassigned_origins_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

        "unassigned_destinations_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

    .. code-block:: python

        # USAGE EXAMPLE: To retrieve trvel modes and run connect_origins_to_destinations tool.

        This example creates route between esri regional offices to esri headquarter.

        routes =  connect_origins_to_destinations(origins_layer=esri_regional,
                                         destinations_layer=dest_layer,
                                         measurement_type='Rural Driving Distance',
                                         time_of_day=datetime(1990, 1, 4, 1, 3),
                                         output_name="routes_from_offices_to_hq")
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "origins_layer": origins_layer,
        "destinations_layer": destinations_layer,
        "measurement_type": measurement_type,
        "origins_layer_route_id_field": origins_layer_route_id_field,
        "destinations_layer_route_id_field": destinations_layer_route_id_field,
        "time_of_day": time_of_day,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "point_barrier_layer": point_barrier_layer,
        "line_barrier_layer": line_barrier_layer,
        "polygon_barrier_layer": polygon_barrier_layer,
        "future": future,
        "route_shape": route_shape,
        "include_route_layers": include_route_layers,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.connect_origins_to_destinations,
        **kwargs,
    )
    try:
        if (
            isinstance(measurement_type, str)
            and str(measurement_type).lower() != "straightline"
        ):
            route_service = network.RouteLayer(
                gis.properties.helperServices.route.url, gis=gis
            )
            travelmodes = route_service.retrieve_travel_modes().get(
                "supportedTravelModes", []
            )
            tm = [
                i
                for i in travelmodes
                if i["name"].lower() == str(measurement_type).lower()
            ]
            if tm:
                params["measurement_type"] = tm[0]
            else:
                params["measurement_type"] = measurement_type
        elif measurement_type is None:
            route_service = network.RouteLayer(
                gis.properties.helperServices.route.url, gis=gis
            )
            travelmodes = route_service.retrieve_travel_modes()
            tm = [
                stm
                for stm in travelmodes["supportedTravelModes"]
                if stm["id"] == travelmodes["defaultTravelMode"]
            ]
            if tm:
                params["measurement_type"] = tm[0]
            else:
                params["measurement_type"] = measurement_type
            # measurement_type = ""
    except Exception as e:
        msg = f"Using the given measurement_type without validation due to the following error: {str(e)}"
        _logger.warn(msg)
        params["measurement_type"] = measurement_type

    return gis._tools.featureanalysis.connect_origins_to_destinations(**params)


# --------------------------------------------------------------------------
def create_buffers(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    distances: Optional[list[str]] = [],
    field: Optional[str] = None,
    units: str = "Meters",
    dissolve_type: str = "None",
    ring_type: str = "Disks",
    side_type: str = "Full",
    end_type: str = "Round",
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/create_buffers/create_buffers.png

    .. |Disks| image:: _static/images/create_buffers/buffers_disks.png
    .. |Dissolve| image:: _static/images/create_buffers/buffers_dissolve.png
    .. |Flat| image:: _static/images/create_buffers/buffers_flat.png
    .. |Full| image:: _static/images/create_buffers/buffers_full.png
    .. |Left| image:: _static/images/create_buffers/buffers_left.png
    .. |None| image:: _static/images/create_buffers/buffers_none.png
    .. |Outside| image:: _static/images/create_buffers/buffers_outside.png
    .. |Right| image:: _static/images/create_buffers/buffers_right.png
    .. |Rings| image:: _static/images/create_buffers/buffers_rings.png
    .. |Round| image:: _static/images/create_buffers/buffers_round.png
    .. |Unspecified| image:: _static/images/create_buffers/buffers_unspecified.png

    The ``create_buffers`` task creates polygons that cover a given distance from a point,
    line, or polygon feature. Buffers are typically used to create areas that can be
    further analyzed using a tool such as ``overlay_layers``. For example, if the question
    is "What buildings are within one mile of the school?", the answer can be found by
    creating a one-mile buffer around the school and overlaying the buffer with the layer
    containing building footprints. The end result is a layer of those buildings within
    one mile of the school.

    =========================    =======================================================================================================================
    **Parameter**                 **Description**
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    input_layer                  Required point, line or polygon feature layer. The input features to be buffered. See :ref:`Feature Input<FeatureInput>`.
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    distances                    Optional list of floats to buffer the input features. The distance(s) that will be buffered. You must supply values
                                 for either the ``distances`` or ``field`` parameter. You can enter a single distance value or multiple values.
                                 The units of the distance values is suppied by the units parameter.
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    field                        Optional string. A field on the ``input_layer`` containing a buffer distance. Buffers will be created using field values.
                                 Unlike the ``distances`` parameter, multiple distances are not supported on field input.
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    units                        Optional string. The linear unit to be used with the distance value(s) specified in distances or contained in the field value.

                                 Choice list: ['Meters', 'Kilometers', 'Feet', 'Miles', 'NauticalMiles', 'Yards']

                                 The default is 'Meters'.
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    dissolve_type                Optional string. Determines how overlapping buffers are processed.

                                 Choice list: ['None', 'Dissolve']

                                 +------------+---------------------------------------------------------------------------------+
                                 | |None|     | ``None``-Overlapping areas are kept. This is the default.                       |
                                 +------------+---------------------------------------------------------------------------------+
                                 | |Dissolve| | ``Dissolve``-Overlapping areas are combined.                                    |
                                 +------------+---------------------------------------------------------------------------------+

    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    ring_type                    Optional string. Determines how multiple-distance buffers are processed.

                                 Choice list: ['Disks', 'Rings']

                                 +-----------+--------------------------------------------------------------------------------------------------+
                                 | |Disks|   | ``Disks``-buffers are concentric and will overlap. For example, if your distances are 10 and 14, |
                                 |           | the result will be two buffers, one from 0 to 10 and one from 0 to 14. This is the default.      |
                                 +-----------+--------------------------------------------------------------------------------------------------+
                                 | |Rings|   | ``Rings`` buffers will not overlap. For example, if your distances are 10 and 14, the result will|
                                 |           | be two buffers, one from 0 to 10 and one from 10 to 14.                                          |
                                 +-----------+--------------------------------------------------------------------------------------------------+

    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    side_type                    Optional string. When buffering line features, you can choose which side of the line to buffer.

                                 Typically, you choose both sides (Full, which is the default). Left and right are determined as
                                 if you were walking from the first x,y coordinate of the line (the start coordinate) to the last
                                 x,y coordinate of the line (the end coordinate). Choosing left or right usually means you know
                                 that your line features were created and stored in a particular direction (for example, upstream
                                 or downstream in a river network).

                                 When buffering polygon features, you can choose whether the buffer includes or excludes the polygon
                                 being buffered.

                                 Choice list: ['Full', 'Left', 'Right', 'Outside']

                                 +---------------+----------------------------------------------------------------------------------------------------+
                                 | |Full|        | ``Full``-both sides of the line will be buffered. This is the default for line featuress.          |
                                 |               |                                                                                                    |
                                 +---------------+----------------------------------------------------------------------------------------------------+
                                 | |Left|        | ``Left``-only the right side of the line will be buffered.                                         |
                                 +---------------+----------------------------------------------------------------------------------------------------+
                                 | |Right|       | ``Right``-only the right side of the line will be buffered.                                        |
                                 +---------------+----------------------------------------------------------------------------------------------------+
                                 | |Outside|     | ``Outside`` when buffering a polygon, the polygon being buffered is excluded in the result buffer. |
                                 +---------------+----------------------------------------------------------------------------------------------------+
                                 | |Unspecified| | If ``side_type`` not supplied, the polygon being buffered is included in the result buffer.        |
                                 |               | This is the  default for polygon features.                                                         |
                                 +---------------+----------------------------------------------------------------------------------------------------+

    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    end_type                     Optional string. The shape of the buffer at the end of line input features. This parameter is not
                                 valid for polygon input features. At the ends of lines the buffer can be rounded (Round) or be
                                 straight across (Flat).

                                 Choice list: ['Round', 'Flat']

                                 +---------+-------------------------------------------------------------------------------+
                                 | |Round| | ``Round``-buffers will be rounded at the ends of lines. This is the default.  |
                                 +---------+-------------------------------------------------------------------------------+
                                 | |Flat|  | ``Flat``-buffers will be flat at the ends of lines.                           |
                                 +---------+-------------------------------------------------------------------------------+

    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    output_name                  Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                 feature layer will cause the new layer to be appended to the Feature Service.
                                 If overwrite is True in context, new layer will overwrite existing layer.
                                 If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    context                      Optional dict. Additional settings such as processing extent
                                 and output spatial reference.
                                 For create_buffers, there are three settings.

                                 - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                 - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                 - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                     .. code-block:: python

                                         # Example Usage
                                         context = {"extent": {"xmin": 3164569.408035,
                                                             "ymin": -9187921.892449,
                                                             "xmax": 3174104.927313,
                                                             "ymax": -9175500.875353,
                                                             "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                     "outSR": {"wkid": 3857},
                                                     "overwrite": True}
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    gis                          Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    estimate                     Optional boolean. If True, the estimated number of credits required to run the operation will be returned.
    -------------------------    -----------------------------------------------------------------------------------------------------------------------
    future                       Optional boolean. If True, a future object will be returned and the process
                                 will not wait for the task to complete. The default is False, which means wait for results.
    =========================    =======================================================================================================================

    :return: result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection`.


    .. code-block:: python

        USAGE EXAMPLE: To create 5 mile buffer around US parks, within the specified extent.

        polygon_lyr_buffer = create_buffers(input_layer=parks_lyr,
                                 distances=[5],
                                 units='Miles',
                                 ring_type='Rings',
                                 end_type='Flat',
                                 output_name='create_buffers',
                                 context={"extent":{"xmin":-12555831.656684224,"ymin":5698027.566358956,"xmax":-11835489.102124758,"ymax":6104672.556836072,"spatialReference":{"wkid":102100,"latestWkid":3857}}})
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "input_layer": input_layer,
        "distances": distances,
        "field": field,
        "units": units,
        "dissolve_type": dissolve_type,
        "ring_type": ring_type,
        "side_type": side_type,
        "end_type": end_type,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.create_buffers, **kwargs
    )
    return gis._tools.featureanalysis.create_buffers(**params)


# --------------------------------------------------------------------------
def create_drive_time_areas(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    break_values: list[int] = [5, 10, 15],
    break_units: str = "Minutes",
    travel_mode: Optional[str] = None,
    overlap_policy: str = "Overlap",
    time_of_day: Optional[datetime] = None,
    time_zone_for_time_of_day: str = "GeoLocal",
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    point_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    line_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    polygon_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    future: bool = False,
    travel_direction: str = "AwayFromFacility",
    show_holes: bool = False,
    include_reachable_streets: bool = False,
):
    """
    .. image:: _static/images/create_drive_time_areas/create_drive_time_areas.png

    .. |Overlap| image:: _static/images/create_drive_time_areas/drive_time_overlap.png
    .. |Dissolve| image:: _static/images/create_drive_time_areas/drive_time_dissolve.png
    .. |Split| image:: _static/images/create_drive_time_areas/drive_time_split.png

    The ``create_drive_time_areas`` method creates areas that can be reached within a
    given drive time or drive distance. It can help you answer questions such as:

    * How far can I drive from here in five minutes?
    * What areas are covered within a three-mile drive distance of my stores?
    * What areas are within four minutes of our fire stations?

    See `Create Drive-Time Areas <https://developers.arcgis.com/rest/analysis/api-reference/create-drivetime.htm>`_
    for details on the `Spatial Analysis Service <https://developers.arcgis.com/rest/analysis/api-reference/getting-started.htm>`_
    that runs this task.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    input_layer                  Required point feature layer. The points around which travel areas
                                 based on a mode of transportation will be drawn.
                                 See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    travel_mode                  Optional string or dict. Specify the mode of transportation for the analysis.

                                 Choice list: ['Driving Distance', 'Driving Time', 'Rural Driving Distance', 'Rural Driving Time', 'Trucking Distance', 'Trucking Time', 'Walking Distance', 'Walking Time']

                                 The default is 'Driving Time'.
    -------------------------    ---------------------------------------------------------
    break_values                 Optional list of floats. The size of the polygons to create.
                                 The units for break_values is specified with the break_units parameter.

                                 By setting many unique values in the list, polygons of different sizes are generated around each input location.

                                 The default is [5, 10, 15].
    -------------------------    ---------------------------------------------------------
    break_units                  Optional string. The units of the break_values parameter.

                                 To create areas showing how far you can go along roads or walkways within a given time, specify a time unit.
                                 Alternatively, specify a distance unit to generate areas bounded by a maximum travel distance.

                                 When the travel_mode is time based, a time unit should be specified for the break_units. When the
                                 travel_mode is distance based, a distance unit should be specified for the break_units.

                                 Choice list: ['Seconds', 'Minutes', Hours', 'Feet', 'Meters', 'Kilometers', 'Feet', 'Miles', 'Yards']

                                 The default is 'Minutes'.
    -------------------------    ---------------------------------------------------------
    overlap_policy               Optional string. Determines how overlapping areas are processed.

                                 Choice list: ['Overlap', 'Dissolve', 'Split']

                                 +---------------+-----------------------------------------------------------------------------------------------------+
                                 | |Overlap|     | ``Overlap``-Overlapping areas are kept. This is the default.                                        |
                                 +---------------+-----------------------------------------------------------------------------------------------------+
                                 | |Dissolve|    | ``Dissolve``-Overlapping areas are combined by break value. Because the areas are dissolved,        |
                                 |               | use this option when you need to know the areas that can be reached within a                        |
                                 |               | given time or distance, but you don't need to know which input points are nearest.                  |
                                 +---------------+-----------------------------------------------------------------------------------------------------+
                                 | |Split|       | ``Split``-Overlapping areas are split in the middle. Use this option when you need to know          |
                                 |               | the one nearest input location to the covered area.                                                 |
                                 +---------------+-----------------------------------------------------------------------------------------------------+

                                 The default is 'Overlap'

    -------------------------    ---------------------------------------------------------
    time_of_day                  Optional datetime.datetime. Specify whether travel times should consider traffic conditions. To use traffic in the analysis,
                                 set measurement_type to a travel mode object whose impedance_attribute_name property is set to travel_time and assign a value
                                 to time_of_day. (A travel mode with other impedance_attribute_name values don't support traffic.) The time_of_day value represents
                                 the time at which travel begins, or departs, from the origin points. The time is specified as datetime.datetime.

                                 The service supports two kinds of traffic: typical and live. Typical traffic references travel speeds that are made up of historical
                                 averages for each five-minute interval spanning a week. Live traffic retrieves speeds from a traffic feed that processes phone probe
                                 records, sensors, and other data sources to record actual travel speeds and predict speeds for the near future.

                                 The `data coverage <http://www.arcgis.com/home/webmap/viewer.html?webmap=b7a893e8e1e04311bd925ea25cb8d7c7>`_ page shows the countries
                                 Esri currently provides traffic data for.

                                 Typical Traffic:

                                 To ensure the task uses typical traffic in locations where it is available, choose a time and day of the week, and then convert the day
                                 of the week to one of the following dates from 1990:

                                 * Monday - 1/1/1990
                                 * Tuesday - 1/2/1990
                                 * Wednesday - 1/3/1990
                                 * Thursday - 1/4/1990
                                 * Friday - 1/5/1990
                                 * Saturday - 1/6/1990
                                 * Sunday - 1/7/1990
                                 Set the time and date as datetime.datetime.

                                 For example, to solve for 1:03 p.m. on Thursdays, set the time and date to 1:03 p.m., 4 January 1990; and convert to
                                 datetime eg. datetime.datetime(1990, 1, 4, 1, 3).

                                 Live Traffic:

                                 To use live traffic when and where it is available, choose a time and date and convert to datetime.

                                 Esri saves live traffic data for 4 hours and references predictive data extending 4 hours into the future. If the time and date you
                                 specify for this parameter is outside the 24-hour time window, or the travel time in the analysis continues past the predictive data window, the task falls back to typical traffic speeds.

                                 Examples:
                                 from datetime import datetime

                                 * "time_of_day": datetime(1990, 1, 4, 1, 3) # 13:03, 4 January 1990. Typical traffic on Thursdays at 1:03 p.m.
                                 * "time_of_day": datetime(1990, 1, 7, 17, 0) # 17:00, 7 January 1990. Typical traffic on Sundays at 5:00 p.m.
                                 * "time_of_day": datetime(2014, 10, 22, 8, 0) # 8:00, 22 October 2014. If the current time is between 8:00 p.m., 21 Oct. 2014 and 8:00 p.m., 22 Oct. 2014,
                                   live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are referenced.
                                 * "time_of_day": datetime(2015, 3, 18, 10, 20) # 10:20, 18 March 2015. If the current time is between 10:20 p.m., 17 Mar. 2015 and 10:20 p.m., 18 Mar. 2015,
                                   live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are referenced.
    -------------------------    ---------------------------------------------------------
    time_zone_for_time_of_day    Optional string. Specify the time zone or zones of the time_of_day parameter.

                                 Choice list: ['GeoLocal', 'UTC']

                                 GeoLocal-refers to the time zone in which the originsLayer points are located.

                                 UTC-refers to Coordinated Universal Time.

                                 The default is 'GeoLocal'.
    -------------------------    ---------------------------------------------------------
    output_name                  Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                 feature layer will cause the new layer to be appended to the Feature Service.
                                 If overwrite is True in context, new layer will overwrite existing layer.
                                 If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------    ---------------------------------------------------------
    context                      Optional dict. Additional settings such as processing extent
                                 and output spatial reference.
                                 For create_drive_time_areas, there are three settings.

                                 - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                 - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                 - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                 .. code-block:: python

                                    # Example Usage
                                         context = {"extent": {"xmin": 3164569.408035,
                                                             "ymin": -9187921.892449,
                                                             "xmax": 3174104.927313,
                                                             "ymax": -9175500.875353,
                                                             "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                     "outSR": {"wkid": 3857},
                                                     "overwrite": True}
    -------------------------    ---------------------------------------------------------
    gis                          Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    -------------------------    ---------------------------------------------------------
    estimate                     Optional boolean. If True, the estimated number of credits required to run the operation will be returned.
    -------------------------    ---------------------------------------------------------
    point_barrier_layer          Optional layer. Specify one or more point features that act as temporary restrictions (in other words, barriers)
                                 when traveling on the underlying streets.

                                 A point barrier can model a fallen tree, an accident, a downed electrical line, or anything that completely blocks
                                 traffic at a specific position along the street. Travel is permitted on the street but not through the barrier. See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    line_barrier_layer           Optional layer. Specify one or more line features that prohibit travel anywhere the lines intersect the streets.

                                 A line barrier prohibits travel anywhere the barrier intersects the streets. For example, a parade or protest that blocks traffic across several street
                                 segments can be modeled with a line barrier. See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    polygon_barrier_layer        Optional string. Specify one or more polygon features that completely restrict travel on the streets intersected by the polygons.

                                 One use of this type of barrier is to model floods covering areas of the street network and making road travel there impossible. See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    future                       Optional boolean. If True, a future object will be returned and the process
                                 will not wait for the task to complete. The default is False, which means wait for results.
    -------------------------    ---------------------------------------------------------
    travel_direction             Optiona String. Specify whether the direction of travel used to generate the travel areas is toward or away from the input locations.

                                 Values: AwayFromFacility or TowardsFacility

                                 The travel direction can influence how the areas are generated. CreateDriveTimeAreas will obey one-way streets, avoid illegal turns, and follow other rules based on the direction of travel. You should select the direction of travel based on the type of input locations and the context of your analysis. For example, the drive-time area for a pizza delivery store should be created away from the facility, whereas the drive-time area for a hospital should be created toward the facility.
    -------------------------    ---------------------------------------------------------
    show_holes                   Optional boolean. When set to true, the output areas will include holes if some streets couldn't be reached without exceeding the cutoff or due to travel restrictions imposed by the travel mode.
    -------------------------    ---------------------------------------------------------
    include_reachable_streets    Optional string. Only applicable if :attr:`output_name` is specified.
                                 When `True` (and :attr:`output_name` is specified), a second layer named
                                 `Reachable Streets` is created in the output :class:`Feature Layer<arcgis.features.FeatureLayerCollection>`.

                                 This layer contains the streets that were used to define the drive time
                                 area polygons. Set this to true if you want a potentially more accurate
                                 result of which streets are actually covered within a specific travel
                                 distance than what the drive-time areas would contain.
    =========================    =========================================================

    :return: result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else Feature Collection.


    .. code-block:: python

        USAGE EXAMPLE: To create drive time areas around USA airports, within the specified extent.

        target_area4 = create_drive_time_areas(airport_lyr,
                                       break_values=[2, 4],
                                       break_units='Hours',
                                       travel_mode='Trucking Time',
                                       overlap_policy='Split',
                                       time_of_day=datetime(2019, 5, 13, 7, 52),
                                       output_name='create_drive_time_areas',
                                       context={"extent":{"xmin":-11134400.655784884,"ymin":3368261.7800108367,"xmax":-10682810.692676282,"ymax":3630899.409198575,"spatialReference":{"wkid":102100,"latestWkid":3857}}})
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "input_layer": input_layer,
        "break_values": break_values,
        "break_units": break_units,
        "travel_mode": travel_mode,
        "overlap_policy": overlap_policy,
        "time_of_day": time_of_day,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "point_barrier_layer": point_barrier_layer,
        "line_barrier_layer": line_barrier_layer,
        "polygon_barrier_layer": polygon_barrier_layer,
        "future": future,
        "travel_direction": travel_direction,
        "show_holes": show_holes,
        "include_reachable_streets": include_reachable_streets,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.create_drive_time_areas, **kwargs
    )

    try:
        if isinstance(travel_mode, str):
            travel_mode = network._utils.find_travel_mode(
                gis=gis, travel_mode=travel_mode
            )
            params["travel_mode"] = travel_mode
        elif isinstance(travel_mode, dict):
            params["travel_mode"] = travel_mode
        else:
            params["travel_mode"] = network._utils.find_travel_mode(gis=gis)
    except Exception as e:
        msg = f"Using the given travel_mode without validation due to the following error: {str(e)}"
        _logger.warn(msg)
        params["travel_mode"] = travel_mode
    if time_of_day:
        params["time_of_day"] = _date_handler(time_of_day)
    if include_reachable_streets:
        if not output_name or not bool(output_name.strip()):
            raise Exception(
                "output_name must be specified when include_reachable_streets is True."
            )

    return gis._tools.featureanalysis.create_drive_time_areas(**params)


# --------------------------------------------------------------------------
def find_nearest(
    analysis_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    near_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    measurement_type: str = "StraightLine",
    max_count: int = 100,
    search_cutoff: float = 2147483647,
    search_cutoff_units: Optional[str] = None,
    time_of_day: Optional[datetime] = None,
    time_zone_for_time_of_day: str = "GeoLocal",
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    include_route_layers: Optional[bool] = None,
    point_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    line_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    polygon_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    future: bool = False,
):
    """
    .. image:: _static/images/find_nearest/find_nearest.png

    The ``find_nearest`` method measures the straight-line distance, driving distance, or driving time from
    features in the analysis layer to features in the near layer, and copies the nearest features in the near
    layer to a new layer. Connecting lines showing the measured path are returned as well. ``find_nearest`` also
    reports the measurement and relative rank of each nearest feature. There are options to limit the number
    of nearest features to find or the search range in which to find them. The results from this method can help
    you answer the following kinds of questions:

    * What is the nearest park from here?
    * Which hospital can I reach in the shortest drive time? And how long would the trip take on a Tuesday at 5:30 p.m. during rush hour?
    * What are the road distances between major European cities?
    * Which of these patients reside within two miles of these chemical plants?

    Find Nearest returns a layer containing the nearest features and a line layer that links the start locations to their nearest locations.
    The connecting line layer contains information about the start and nearest locations and the distances between.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    analysis_layer               Required layer. The features from which the nearest locations are found. This layer can have point, line, or polygon features. See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    near_layer                   Required layer. The nearest features are chosen from this layer. This layer can have point, line, or polygon features. See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    measurement_type             Required string. Specify the mode of transportation for the analysis.

                                 Choice list: ['StraightLine', 'Driving Distance', 'Driving Time ', 'Rural Driving Distance', 'Rural Driving Time', 'Trucking Distance', 'Trucking Time', 'Walking Distance', 'Walking Time']

                                 The default is 'StraightLine'.
    -------------------------    ---------------------------------------------------------
    max_count                    Optional string. The maximum number of nearest locations to find for each feature in ``analysis_layer``. The default is the maximum cutoff allowed by the service, which is 100.

                                 Note that setting a maxCount for this parameter doesn't guarantee that many features will be found. The ``search_cutoff`` and other constraints may also reduce the number of features found.
    -------------------------    ---------------------------------------------------------
    search_cutoff                Optional float. The maximum range to search for nearest locations from each feature in the ``analysis_layer``.
                                 The units for this parameter is always minutes when ``measurement_type`` is set to a time based travel mode;
                                 otherwise the units are set in the ``search_cutoff_units`` parameter.

                                 The default is to search without bounds.
    -------------------------    ---------------------------------------------------------
    search_cutoff_units          The units of the ``search_cutoff`` parameter. This parameter is ignored when ``measurement_type`` is set to a time based travel
                                 mode because the units for ``search_cutoff`` are always minutes in those cases. If ``measurement_type`` is set to StraightLine or another distance-based travel mode, and a value for ``search_cutoff`` is specified, set the cutoff units using this parameter.

                                 Choice list: ['Kilometers', 'Meters', 'Miles', 'Feet', '']

                                 The default value is null, which causes the service to choose either miles or kilometers according to the units property of the user making the request.
    -------------------------    ---------------------------------------------------------
    time_of_day                  Optional datetime.datetime. Specify whether travel times should consider traffic conditions. To use traffic in the analysis, set ``measurement_type`` to a travel mode object whose impedance_attribute_name property is set to travel_time and assign a value to ``time_of_day``. (A travel mode with other impedance_attribute_name values don't support traffic.) The ``time_of_day`` value represents the time at which travel begins, or departs, from the origin points. The time is specified as datetime.datetime.

                                 The service supports two kinds of traffic: typical and live. Typical traffic references travel speeds that are made up of historical averages for each five-minute interval spanning a week. Live traffic retrieves speeds from a traffic feed that processes phone probe records, sensors, and other data sources to record actual travel speeds and predict speeds for the near future.

                                 The `data coverage <http://www.arcgis.com/home/webmap/viewer.html?webmap=b7a893e8e1e04311bd925ea25cb8d7c7>`_ page shows the countries Esri currently provides traffic data for.

                                 Typical Traffic:

                                 To ensure the task uses typical traffic in locations where it is available, choose a time and day of the week, and then convert the day of the week to one of the following dates from 1990:

                                 * Monday - 1/1/1990
                                 * Tuesday - 1/2/1990
                                 * Wednesday - 1/3/1990
                                 * Thursday - 1/4/1990
                                 * Friday - 1/5/1990
                                 * Saturday - 1/6/1990
                                 * Sunday - 1/7/1990
                                 Set the time and date as datetime.datetime.

                                 For example, to solve for 1:03 p.m. on Thursdays, set the time and date to 1:03 p.m., 4 January 1990; and convert to datetime eg. datetime.datetime(1990, 1, 4, 1, 3).

                                 Live Traffic:

                                 To use live traffic when and where it is available, choose a time and date and convert to datetime.

                                 Esri saves live traffic data for 4 hours and references predictive data extending 4 hours into the future. If the time and date you specify for this parameter is outside the 24-hour time window, or the travel time in the analysis continues past the predictive data window, the task falls back to typical traffic speeds.

                                 Examples:
                                 from datetime import datetime

                                 * "time_of_day": datetime(1990, 1, 4, 1, 3) # 13:03, 4 January 1990. Typical traffic on Thursdays at 1:03 p.m.
                                 * "time_of_day": datetime(1990, 1, 7, 17, 0) # 17:00, 7 January 1990. Typical traffic on Sundays at 5:00 p.m.
                                 * "time_of_day": datetime(2014, 10, 22, 8, 0) # 8:00, 22 October 2014. If the current time is between 8:00 p.m., 21 Oct. 2014 and 8:00 p.m., 22 Oct. 2014, live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are referenced.
                                 * "time_of_day": datetime(2015, 3, 18, 10, 20) # 10:20, 18 March 2015. If the current time is between 10:20 p.m., 17 Mar. 2015 and 10:20 p.m., 18 Mar. 2015, live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are referenced.

    -------------------------    ---------------------------------------------------------
    time_zone_for_time_of_day    Optional string. Specify the time zone or zones of the ``time_of_day`` parameter.

                                 Choice list: ['GeoLocal', 'UTC']

                                 ``GeoLocal``-refers to the time zone in which the origins_layer points are located.

                                 ``UTC``-refers to Coordinated Universal Time.
    -------------------------    ---------------------------------------------------------
    include_route_layers         Optional boolean. When ``include_route_layers`` is set to True, each route from the result is also saved as a route layer item.
                                 A route layer includes all the information for a particular route such as the stops assigned to the route as well
                                 as the travel directions. Creating route layers is useful if you want to share individual routes with other members in your organization.
                                 The route layers use the output feature service name provided in the ``output_name`` parameter as a prefix and the route name generated as part
                                 of the analysis is added to create a unique name for each route layer.

                                 **Caution:**

                                 Route layers cannot be created when the output is a feature collection. The task will raise an error if ``output_name`` is not
                                 specified (which indicates feature collection output) and ``include_route_layers`` is True.

                                 The maximum number of route layers that can be created is 1,000. If the result contains more than 1,000 routes
                                 and ``include_route_layers`` is True, the task will only create the output feature service.
    -------------------------    ---------------------------------------------------------
    point_barrier_layer          Optional layer. Specify one or more point features that act as temporary restrictions (in other words, barriers) when traveling on the underlying streets.

                                 A point barrier can model a fallen tree, an accident, a downed electrical line, or anything that completely blocks traffic at a specific
                                 position along the street. Travel is permitted on the street but not through the barrier.
    -------------------------    ---------------------------------------------------------
    line_barrier_layer           Optional layer. Specify one or more line features that prohibit travel anywhere the lines intersect the streets.

                                 A line barrier prohibits travel anywhere the barrier intersects the streets. For example, a parade or protest that blocks traffic across
                                 several street segments can be modeled with a line barrier.
    -------------------------    ---------------------------------------------------------
    polygon_barrier_layer        Optional layer. Specify one or more polygon features that completely restrict travel on the streets intersected by the polygons.

                                 One use of this type of barrier is to model floods covering areas of the street network and making road travel there impossible.
    -------------------------    ---------------------------------------------------------
    output_name                  Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                 feature layer will cause the new layer to be appended to the Feature Service.
                                 If overwrite is True in context, new layer will overwrite existing layer.
                                 If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------    ---------------------------------------------------------
    context                      Optional dict. Additional settings such as processing extent
                                 and output spatial reference.
                                 For find_nearest, there are three settings.

                                 - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                 - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                 - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                     .. code-block:: python

                                         # Example Usage
                                         context = {"extent": {"xmin": 3164569.408035,
                                                             "ymin": -9187921.892449,
                                                             "xmax": 3174104.927313,
                                                             "ymax": -9175500.875353,
                                                             "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                     "outSR": {"wkid": 3857},
                                                     "overwrite": True}
    -------------------------    ---------------------------------------------------------
    gis                          Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    -------------------------    ---------------------------------------------------------
    estimate                     Optional boolean. If True, the estimated number of credits required to run the operation will be returned.
    -------------------------    ---------------------------------------------------------
    future                       Optional boolean. If True, a future object will be returned and the process
                                 will not wait for the task to complete. The default is False, which means wait for results.
    =========================    =========================================================

    :return: A dictionary with the following keys:

       "nearest_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

       "connecting_lines_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

    .. code-block:: python

        #USAGE EXAMPLE: To find which regional office can be reached in the shortest drive time from esri headquarter.

        result1 = find_nearest(analysis_layer=esri_hq_lyr,
                               near_layer=regional_offices_lyr,
                               measurement_type="Driving Time",
                               output_name="find nearest office",
                               include_route_layers=True,
                               point_barrier_layer=road_closures_lyr))
    """
    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "analysis_layer": analysis_layer,
        "near_layer": near_layer,
        "measurement_type": measurement_type,
        "max_count": max_count,
        "search_cutoff": search_cutoff,
        "search_cutoff_units": search_cutoff_units,
        "time_of_day": time_of_day,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "include_route_layers": include_route_layers,
        "point_barrier_layer": point_barrier_layer,
        "line_barrier_layer": line_barrier_layer,
        "polygon_barrier_layer": polygon_barrier_layer,
        "future": future,
    }

    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.find_nearest, **kwargs
    )
    params["estimate"] = estimate
    try:
        if (
            isinstance(measurement_type, str)
            and str(measurement_type).lower() != "straightline"
        ):
            route_service = network.RouteLayer(
                gis.properties.helperServices.route.url, gis=gis
            )
            travelmodes = route_service.retrieve_travel_modes().get(
                "supportedTravelModes", []
            )
            tm = [
                i
                for i in travelmodes
                if i["name"].lower() == str(measurement_type).lower()
            ]
            if tm:
                params["measurement_type"] = tm[0]
            else:
                params["measurement_type"] = measurement_type
    except Exception as e:
        msg = f"Using the given measurement_type without validation due to the following error: {str(e)}"
        _logger.warn(msg)
        params["measurement_type"] = measurement_type
    params["time_of_day"] = _date_handler(time_of_day)
    return gis._tools.featureanalysis.find_nearest(**params)


# --------------------------------------------------------------------------
def plan_routes(
    stops_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    route_count: int,
    max_stops_per_route: int,
    route_start_time: datetime,
    start_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    start_layer_route_id_field: Optional[str] = None,
    return_to_start: bool = True,
    end_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    end_layer_route_id_field: Optional[str] = None,
    travel_mode: Optional[str] = None,
    stop_service_time: float = 0,
    max_route_time: float = 525600,
    include_route_layers: bool = False,
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    point_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    line_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    polygon_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    future: bool = False,
):
    """

    .. image:: _static/images/plan_routes/plan_routes.png

    .. |balanced| image:: _static/images/plan_routes/balanced.png
    .. |partially_balanced| image:: _static/images/plan_routes/partially_balanced.png
    .. |unbalanced| image:: _static/images/plan_routes/unbalanced.png

    The ``plan_routes`` method determines how to efficiently divide tasks among a mobile workforce.

    You provide the input, which includes a set of stops and the number of vehicles available to
    visit the stops, and the tool assigns the stops to vehicles and returns routes showing how each
    vehicle can reach their assigned stops in the least amount of time.

    With ``plan_routes``, mobile workforces reach more jobsites in less time, which increases
    productivity and improves customer service. Organizations often use ``plan_routes`` to:

    * Inspect homes, restaurants, and construction sites
    * Provide repair, installation, and technical services
    * Deliver items and small packages
    * Make sales calls
    * Provide van transportation from spectators' homes to events

    The output from ``plan_routes`` includes a layer of routes showing the shortest paths to visit
    the stops; a layer of the stops assigned to routes, as well as any stops that couldn't be reached
    due to the given parameter settings; and a layer of directions containing the travel itinerary for each route.

    ============================    ==================================================================================================
    **Parameter**                    **Description**
    ----------------------------    --------------------------------------------------------------------------------------------------
    stops_layer                     Required feature layer. The points that the vehicles, drivers, or routes, should visit.
                                    The fields on the input stops are included in the output stops, so if your input
                                    layer has a field such as Name, Address, or ProductDescription, that information
                                    will be available in the results. See :ref:`Feature Input<FeatureInput>`.
    ----------------------------    --------------------------------------------------------------------------------------------------
    route_count                     Required integer. The number of vehicles that are available to visit the stops.
                                    The method supports up to 100 vehicles.

                                    The default value is 0.

                                    The method may be able to find and return a solution that uses fewer vehicles than
                                    the number you specify for this parameter. The number of vehicles returned also
                                    depends on four other parameters: the total number of stops in ``stops_layer``, the
                                    number of stops per vehicle you allow (``max_stops_per_route``), the travel time between
                                    stops, the time spent at each stop (``stop_service_time``), and any limit you set on the
                                    total route time per vehicle (``max_route_time``).
    ----------------------------    --------------------------------------------------------------------------------------------------
    max_stops_per_route             Required integer. The maximum number of stops a route, or vehicle, is allowed to visit.
                                    The largest value you can specify is 200. The default value is zero.

                                    This is one of two parameters that balance the overall workload across routes.
                                    The other is ``max_route_time``

                                    By lowering the maximum number of stops that can be assigned to each vehicle, the vehicles
                                    are more likely to have an equal number of stops assigned to them. This helps
                                    balance workloads among drivers. The drawback, however, is that it may result in a
                                    solution that is less efficient.

                                    By increasing the stops per vehicle, the tool has more freedom to find more efficient solutions;
                                    however, the workload may be unevenly distributed among drivers and vehicles. Note that you can
                                    balance workloads by time instead of number of stops by specifying a value for the ``max_route_time`` parameter.

                                    The following examples demonstrate the effects of limiting the maximum stops per vehicle or the
                                    total time per vehicle. In all of these examples, two routes start at the same location
                                    and visit a total of six stops.

                                    +----------------------------+----------------------------------------------------------------------------------------------------------+
                                    | |balanced|                 | Balanced travel times and stops per route:                                                               |
                                    |                            |                                                                                                          |
                                    |                            | The stops are more or less uniformly spread apart, so setting ``max_stops_per_route``=3 to evenly        |
                                    |                            | distribute the workload results in routes that are roughly the same duration.                            |
                                    |                            |                                                                                                          |
                                    +----------------------------+----------------------------------------------------------------------------------------------------------+
                                    | |partially_balanced|       | Balanced stops per route but unbalanced travel times:                                                    |
                                    |                            |                                                                                                          |
                                    |                            | Five of the six stops are clustered near the starting location, but one stop is set apart                |
                                    |                            | and requires a much longer drive to be reached. Dividing the stops equally between the two               |
                                    |                            | routes ( ``max_stops_per_route`` =3) causes unbalanced travel times.                                     |
                                    +----------------------------+----------------------------------------------------------------------------------------------------------+
                                    | |unbalanced|               | Unbalanced stops per route but balanced travel times:                                                    |
                                    |                            |                                                                                                          |
                                    |                            | The stops are in the same location as the previous graphic. By increasing the value of                   |
                                    |                            | ``max_stops_per_route`` to 4, and limiting the total travel time per vehicle (``max_route_time``),       |
                                    |                            | the travel times are balanced even though one route visits more stops.                                   |
                                    +----------------------------+----------------------------------------------------------------------------------------------------------+

    ----------------------------    --------------------------------------------------------------------------------------------------
    route_start_time                Required datetime.datetime. Specify when the vehicles or people start their routes.
                                    The time is specified as datetime.
                                    The starting time value is the same for all routes; that is, all routes start at the same time.

                                    Time zones affect what value you assign to ``route_start_time``. The time zone for the start time is
                                    based on the time zone in which the starting point is geographically located. For instance,
                                    if you have one route starting location and it is located in Pacific Standard Time (PST),
                                    the time you specify for ``route_start_time`` is in PST.

                                    There are a couple of scenarios to beware of given that starting times are based on where
                                    the starting points are located. One situation to be careful of is when you are located in
                                    one time zone but your starting locations are in another times zone. For instance, assume
                                    you are in Pacific Standard Time (UTC-8:00) and the vehicles you are routing are stationed
                                    in Mountain Standard Time (UTC-7:00). If it is currently 9:30 a.m. PST (10:30 a.m. MST)
                                    and your vehicles need to begin their routes in 30 minutes, you would set the start time
                                    to 11:00 a.m. That is, the starting locations for the routes are in the Mountain time zone,
                                    and it is currently 10:30 a.m. there, therefore, a starting time of 30 minutes from now is 11:00 a.m.
                                    Make sure you set the parameter according to the proper time zone.

                                    The other situation that requires caution is where starting locations are spread across
                                    multiple time zones. The time you set for ``route_start_time`` is specific to the time zone in
                                    which the starting location is regardless of whether there are one or more starting locations
                                    in the problem you submit. For instance, if one route starts from a point in PST and another
                                    route starts from MST, and you enter 11:00 a.m. as the start time, the route in PST will start
                                    at 11:00 a.m. PST and the route in MST will start at 11:00 a.m. MST a one-hour difference. The
                                    starting times are the same in local time, but offset in actual time, or UTC.

                                    The service automatically determines the time zones of the input starting locations (``start_layer``) for you.

                                    Examples:

                                    * datetime(2014, 10, 22, 8, 0) # 8:00, 22 October 2014. Routes will depart their
                                      starting locations at 8:00 a.m., 22 October. Any routes with starting points in Mountain
                                      Standard Time start at 8:00 a.m., 22 October 2014 MST; any routes with starting points in
                                      Pacific Standard Time start at 8:00 a.m. 22 October 2014 PST, and so on.
                                    * datetime(2015, 3, 18, 10, 20) # 10:20, 18 March 2015.
    ----------------------------    --------------------------------------------------------------------------------------------------
    start_layer                     Required feature layer. Provide the locations where the people or vehicles start their routes.
                                    You can specify one or many starting locations.

                                    If specifying one, all routes will start from the one location. If specifying many starting
                                    locations, each route needs exactly one predefined starting location, and the following criteria must be met:

                                    The number of routes (``route_count``) must equal the number of points in ``start_layer``. (However,
                                    when only one point is included in ``start_layer``, it is assumed that all routes start from
                                    the same location, and the two numbers can be different.)
                                    The starting location for each route must be identified with the ``start_layer_route_id_field``
                                    parameter. This implies that the input points in ``start_layer`` have a unique identifier.
                                    Bear in mind that if you also have many ending locations, those locations need to be
                                    predetermined as well. The predetermined start and end locations of each route are
                                    paired together by matching route ID values.
                                    See the the section of this topic entitled Starting and ending locations of
                                    routes to learn more. See :ref:`Feature Input<FeatureInput>`.
    ----------------------------    --------------------------------------------------------------------------------------------------
    start_layer_route_id_field      Optional string. Choose a field that uniquely identifies points in start_layer.
                                    This parameter is required when ``start_layer`` has more than one point; it is ignored otherwise.

                                    The ``start_layer_route_id_field`` parameter helps identify where routes begin and
                                    indicates the names of the output routes.

                                    See the the section of this topic entitled Starting and ending locations
                                    of routes to learn more.
    ----------------------------    --------------------------------------------------------------------------------------------------
    return_to_start                 Optional boolean. A True value indicates each route must end its trip at the same place where
                                    it started. The starting location is defined by the ``start_layer`` and ``start_layer_route_id_field`` parameters.

                                    The default value is True.
    ----------------------------    --------------------------------------------------------------------------------------------------
    end_layer                       Optional layer. Provide the locations where the people or vehicles end their routes.

                                    If ``end_layer`` is not specified, ``return_to_start`` must be set to True.

                                    You can specify one or many ending locations.

                                    If specifying one, all routes will end at the one location. If specifying many ending
                                    locations, each route needs exactly one predefined ending location, and the following criteria must be met:

                                    + The number of routes (``route_count``) must equal the number of points in ``end_layer``. (However, when only one point is included in ``end_layer``, it is assumed that all routes
                                      end at the same location, and the two numbers can be different.)
                                    + The ending location for each route must be identified with the ``start_layer_route_id_field`` parameter. This implies that the input points in endLayer have a unique identifier.
                                      Bear in mind that if you also have many starting locations, those locations need to be predetermined as well. The predetermined start and end locations of each route are paired
                                      together by matching route ID values. See :ref:`Feature Input<FeatureInput>`.
    ----------------------------    --------------------------------------------------------------------------------------------------
    end_layer_route_id_field        Optional string. Choose a field that uniquely identifies points in ``end_layer``.
                                    This parameter is required when ``end_layer`` has more than one point; it is ignored
                                    if there is one point or if ``return_to_start`` is True.

                                    The ``end_layer_route_id_field`` parameter helps identify where routes end and indicates the names of the output routes.

                                    See the the section of this topic entitled Starting and ending locations of routes to learn more.
    ----------------------------    --------------------------------------------------------------------------------------------------
    travel_mode                     Optional string. Optional string. Specify the mode of transportation for the analysis.

                                    Choice list: ['Driving Distance', 'Driving Time', 'Rural Driving Distance', 'Rural Driving Time', 'Trucking Distance', 'Trucking Time', 'Walking Distance', 'Walking Time']
    ----------------------------    --------------------------------------------------------------------------------------------------
    stop_service_time               Optional float. Indicates how much time, in minutes, is spent at each stop.
                                    The units are minutes. All stops are assinged the same service duration from
                                    this parameter unique values for individual stops cannot be specified with this service.
    ----------------------------    --------------------------------------------------------------------------------------------------
    max_route_time                  Optional float. The amount of time you specify here limits the maximum duration of each route.
                                    The maximum route time is an accumulation of travel time and the total service time at visited
                                    stops (``stop_service_time``). This parameter is commonly used to prevent drivers from working
                                    too many hours or to balance workloads across routes or drivers.

                                    The units are 'minutes'. The default value, which is also the maximum value, is 525600 minutes, or one year.
    ----------------------------    --------------------------------------------------------------------------------------------------
    include_route_layers            Optional boolean. When ``include_route_layers`` is set to True, each route from the result is also
                                    saved as a route layer item. A route layer includes all the information for a particular route such as the stops assigned to
                                    the route as well as the travel directions. Creating route layers is useful if you want to share individual routes with other
                                    members in your organization. The route layers use the output feature service name provided in the ``output_name`` parameter as a
                                    prefix and the route name generated as part of the analysis is added to create a unique name for each route layer.
    ----------------------------    --------------------------------------------------------------------------------------------------
    output_name                     Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                    feature layer will cause the new layer to be appended to the Feature Service.
                                    If overwrite is True in context, new layer will overwrite existing layer.
                                    If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ----------------------------    --------------------------------------------------------------------------------------------------
    context                         Optional dict. Additional settings such as processing extent and output spatial reference.
                                    For plan_routes, there are three settings.

                                    - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                    - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                    - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                        .. code-block:: python

                                            # Example Usage
                                            context = {"extent": {"xmin": 3164569.408035,
                                                                "ymin": -9187921.892449,
                                                                "xmax": 3174104.927313,
                                                                "ymax": -9175500.875353,
                                                                "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                        "outSR": {"wkid": 3857},
                                                        "overwrite": True}
    ----------------------------    --------------------------------------------------------------------------------------------------
    gis                             Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ----------------------------    --------------------------------------------------------------------------------------------------
    estimate                        Optional boolean. If True, the number of credits to run the operation will be returned.
    ----------------------------    --------------------------------------------------------------------------------------------------
    point_barrier_layer             Optional feature layer. Specify one or more point features that act as temporary restrictions (in other words, barriers) when traveling on the underlying streets.

                                    A point barrier can model a fallen tree, an accident, a downed electrical line, or anything that completely blocks traffic at a specific position along the street. Travel is permitted on the street but not through the barrier. See :ref:`Feature Input<FeatureInput>`.
    ----------------------------    --------------------------------------------------------------------------------------------------
    line_barrier_layer              Optional feature layer. Specify one or more line features that prohibit travel anywhere the lines intersect the streets.

                                    A line barrier prohibits travel anywhere the barrier intersects the streets. For example, a parade or protest that blocks traffic across several street segments can be modeled with a line barrier. See :ref:`Feature Input<FeatureInput>`.
    ----------------------------    --------------------------------------------------------------------------------------------------
    polygon_barrier_layer           Optional feature layer. Specify one or more polygon features that completely restrict travel on the streets intersected by the polygons.

                                    One use of this type of barrier is to model floods covering areas of the street network and making road travel there impossible. See :ref:`Feature Input<FeatureInput>`.
    ----------------------------    --------------------------------------------------------------------------------------------------
    future                          Optional boolean. If True, a future object will be returned and the process
                                    will not wait for the task to complete. The default is False, which means wait for results.
    ============================    ==================================================================================================

    :return: :class:`~arcgis.features.FeatureLayer` if ``output_name`` is specified, else dict with the following keys:


        "routes_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

        "assigned_stops_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

        "unassigned_stops_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

    .. code-block:: python

        # USAGE EXAMPLE: To plan routes to provide cab from employee residence to office.
        route = plan_routes(stops_layer=employee_residence,
                            route_count=4,
                            max_stops_per_route=4,
                            route_start_time=datetime(2019, 6, 20, 6, 0),
                            start_layer=office_location,
                            start_layer_route_id_field='n_office',
                            return_to_start=False,
                            end_layer=office_location,
                            end_layer_route_id_field='n_office',
                            travel_mode='Driving Time',
                            stop_service_time=5,
                            include_route_layers=False,
                            output_name='plan route for employees')

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "stops_layer": stops_layer,
        "route_count": route_count,
        "max_stops_per_route": max_stops_per_route,
        "route_start_time": route_start_time,
        "start_layer": start_layer,
        "start_layer_route_id_field": start_layer_route_id_field,
        "return_to_start": return_to_start,
        "end_layer": end_layer,
        "end_layer_route_id_field": end_layer_route_id_field,
        "travel_mode": travel_mode,
        "stop_service_time": stop_service_time,
        "max_route_time": max_route_time,
        "include_route_layers": include_route_layers,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "point_barrier_layer": point_barrier_layer,
        "line_barrier_layer": line_barrier_layer,
        "polygon_barrier_layer": polygon_barrier_layer,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.plan_routes, **kwargs
    )
    params["estimate"] = estimate
    try:
        if isinstance(travel_mode, str):
            travel_mode = network._utils.find_travel_mode(
                gis=gis, travel_mode=travel_mode
            )
            params["travel_mode"] = travel_mode
        elif isinstance(travel_mode, dict):
            params["travel_mode"] = travel_mode
        else:
            params["travel_mode"] = network._utils.find_travel_mode(gis=gis)
    except Exception as e:
        msg = f"Using the given travel_mode without validation due to the following error: {str(e)}"
        _logger.warn(msg)
        params["travel_mode"] = travel_mode
    params["route_start_time"] = _date_handler(route_start_time)
    return gis._tools.featureanalysis.plan_routes(**params)
