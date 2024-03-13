import logging as _logging
from typing import Optional
import arcgis
from datetime import datetime
from arcgis.geoprocessing import import_toolbox
from arcgis.features import FeatureSet
from arcgis.gis import GIS
from arcgis.mapping import MapImageLayer
from arcgis.geoprocessing import DataFile, LinearUnit, RasterData
from arcgis.geoprocessing._support import _execute_gp_tool
from arcgis._impl.common._utils import _validate_url
from ._routing_utils import _create_toolbox

_log = _logging.getLogger(__name__)

_use_async = True

default_stops = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {
            "alias": "Route Name",
            "name": "RouteName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {"alias": "Sequence", "name": "Sequence", "type": "esriFieldTypeInteger"},
        {
            "alias": "Additional Time",
            "name": "AdditionalTime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "Additional Distance",
            "name": "AdditionalDistance",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "Time Window Start",
            "name": "TimeWindowStart",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "Time Window End",
            "name": "TimeWindowEnd",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "Curb Approach",
            "name": "CurbApproach",
            "type": "esriFieldTypeSmallInteger",
        },
    ],
    "geometryType": "esriGeometryPoint",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}

default_point_barriers = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {
            "alias": "Barrier Type",
            "name": "BarrierType",
            "type": "esriFieldTypeInteger",
        },
        {
            "alias": "Additional Time",
            "name": "Additional_Time",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "Additional Distance",
            "name": "Additional_Distance",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "CurbApproach",
            "name": "CurbApproach",
            "type": "esriFieldTypeSmallInteger",
        },
    ],
    "geometryType": "esriGeometryPoint",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}

default_line_barriers = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {
            "alias": "SHAPE_Length",
            "name": "SHAPE_Length",
            "type": "esriFieldTypeDouble",
        },
    ],
    "geometryType": "esriGeometryPolyline",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}

default_polygon_barriers = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {
            "alias": "Barrier Type",
            "name": "BarrierType",
            "type": "esriFieldTypeInteger",
        },
        {
            "alias": "Scaled Time Factor",
            "name": "ScaledTimeFactor",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "Scaled Distance Factor",
            "name": "ScaledDistanceFactor",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "SHAPE_Length",
            "name": "SHAPE_Length",
            "type": "esriFieldTypeDouble",
        },
        {"alias": "SHAPE_Area", "name": "SHAPE_Area", "type": "esriFieldTypeDouble"},
    ],
    "geometryType": "esriGeometryPolygon",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}

default_restrictions = """['Avoid Unpaved Roads', 'Avoid Private Roads', 'Driving an Automobile', 'Through Traffic Prohibited', 'Roads Under Construction Prohibited', 'Avoid Gates', 'Avoid Express Lanes', 'Avoid Carpool Roads']"""

default_attributes = {
    "fields": [
        {"alias": "ObjectID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {
            "alias": "AttributeName",
            "name": "AttributeName",
            "type": "esriFieldTypeString",
            "length": 255,
        },
        {
            "alias": "ParameterName",
            "name": "ParameterName",
            "type": "esriFieldTypeString",
            "length": 255,
        },
        {
            "alias": "ParameterValue",
            "name": "ParameterValue",
            "type": "esriFieldTypeString",
            "length": 25,
        },
    ],
    "features": [
        {
            "attributes": {
                "OBJECTID": 1,
                "AttributeName": "Any Hazmat Prohibited",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 2,
                "AttributeName": "Avoid Carpool Roads",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 3,
                "AttributeName": "Avoid Express Lanes",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 4,
                "AttributeName": "Avoid Ferries",
                "ParameterValue": "AVOID_MEDIUM",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 5,
                "AttributeName": "Avoid Gates",
                "ParameterValue": "AVOID_MEDIUM",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 6,
                "AttributeName": "Avoid Limited Access Roads",
                "ParameterValue": "AVOID_MEDIUM",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 7,
                "AttributeName": "Avoid Private Roads",
                "ParameterValue": "AVOID_MEDIUM",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 8,
                "AttributeName": "Avoid Roads Unsuitable for Pedestrians",
                "ParameterValue": "AVOID_HIGH",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 9,
                "AttributeName": "Avoid Stairways",
                "ParameterValue": "AVOID_HIGH",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 10,
                "AttributeName": "Avoid Toll Roads",
                "ParameterValue": "AVOID_MEDIUM",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 11,
                "AttributeName": "Avoid Toll Roads for Trucks",
                "ParameterValue": "AVOID_MEDIUM",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 12,
                "AttributeName": "Avoid Truck Restricted Roads",
                "ParameterValue": "AVOID_HIGH",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 13,
                "AttributeName": "Avoid Unpaved Roads",
                "ParameterValue": "AVOID_HIGH",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 14,
                "AttributeName": "Axle Count Restriction",
                "ParameterValue": "0",
                "ParameterName": "Number of Axles",
            }
        },
        {
            "attributes": {
                "OBJECTID": 15,
                "AttributeName": "Axle Count Restriction",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 16,
                "AttributeName": "Driving a Bus",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 17,
                "AttributeName": "Driving a Delivery Vehicle",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 18,
                "AttributeName": "Driving a Taxi",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 19,
                "AttributeName": "Driving a Truck",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 20,
                "AttributeName": "Driving an Automobile",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 21,
                "AttributeName": "Driving an Emergency Vehicle",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 22,
                "AttributeName": "Height Restriction",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 23,
                "AttributeName": "Height Restriction",
                "ParameterValue": "0",
                "ParameterName": "Vehicle Height (meters)",
            }
        },
        {
            "attributes": {
                "OBJECTID": 24,
                "AttributeName": "Kingpin to Rear Axle Length Restriction",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 25,
                "AttributeName": "Kingpin to Rear Axle Length Restriction",
                "ParameterValue": "0",
                "ParameterName": "Vehicle Kingpin to Rear Axle Length (meters)",
            }
        },
        {
            "attributes": {
                "OBJECTID": 26,
                "AttributeName": "Length Restriction",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 27,
                "AttributeName": "Length Restriction",
                "ParameterValue": "0",
                "ParameterName": "Vehicle Length (meters)",
            }
        },
        {
            "attributes": {
                "OBJECTID": 28,
                "AttributeName": "Preferred for Pedestrians",
                "ParameterValue": "PREFER_LOW",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 29,
                "AttributeName": "Riding a Motorcycle",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 30,
                "AttributeName": "Roads Under Construction Prohibited",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 31,
                "AttributeName": "Semi or Tractor with One or More Trailers Prohibited",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 32,
                "AttributeName": "Single Axle Vehicles Prohibited",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 33,
                "AttributeName": "Tandem Axle Vehicles Prohibited",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 34,
                "AttributeName": "Through Traffic Prohibited",
                "ParameterValue": "AVOID_HIGH",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 35,
                "AttributeName": "Truck with Trailers Restriction",
                "ParameterValue": "0",
                "ParameterName": "Number of Trailers on Truck",
            }
        },
        {
            "attributes": {
                "OBJECTID": 36,
                "AttributeName": "Truck with Trailers Restriction",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 37,
                "AttributeName": "Use Preferred Hazmat Routes",
                "ParameterValue": "PREFER_MEDIUM",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 38,
                "AttributeName": "Use Preferred Truck Routes",
                "ParameterValue": "PREFER_MEDIUM",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 39,
                "AttributeName": "WalkTime",
                "ParameterValue": "5",
                "ParameterName": "Walking Speed (km/h)",
            }
        },
        {
            "attributes": {
                "OBJECTID": 40,
                "AttributeName": "Walking",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 41,
                "AttributeName": "Weight Restriction",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 42,
                "AttributeName": "Weight Restriction",
                "ParameterValue": "0",
                "ParameterName": "Vehicle Weight (kilograms)",
            }
        },
        {
            "attributes": {
                "OBJECTID": 43,
                "AttributeName": "Weight per Axle Restriction",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 44,
                "AttributeName": "Weight per Axle Restriction",
                "ParameterValue": "0",
                "ParameterName": "Vehicle Weight per Axle (kilograms)",
            }
        },
        {
            "attributes": {
                "OBJECTID": 45,
                "AttributeName": "Width Restriction",
                "ParameterValue": "PROHIBITED",
                "ParameterName": "Restriction Usage",
            }
        },
        {
            "attributes": {
                "OBJECTID": 46,
                "AttributeName": "Width Restriction",
                "ParameterValue": "0",
                "ParameterName": "Vehicle Width (meters)",
            }
        },
    ],
    "displayFieldName": "",
    "exceededTransferLimit": False,
}

default_tolerance = {"distance": 10, "units": "esriMeters"}


def find_routes(
    stops: FeatureSet,
    measurement_units: str = "Minutes",
    analysis_region: Optional[str] = None,
    reorder_stops_to_find_optimal_routes: bool = False,
    preserve_terminal_stops: str = "Preserve First",
    return_to_start: bool = False,
    use_time_windows: bool = False,
    time_of_day: Optional[datetime] = None,
    time_zone_for_time_of_day: str = "Geographically Local",
    uturn_at_junctions: str = "Allowed Only at Intersections and Dead Ends",
    point_barriers: Optional[FeatureSet] = None,
    line_barriers: Optional[FeatureSet] = None,
    polygon_barriers: Optional[FeatureSet] = None,
    use_hierarchy: bool = True,
    restrictions: Optional[str] = None,
    attribute_parameter_values: Optional[FeatureSet] = None,
    route_shape: str = "True Shape",
    route_line_simplification_tolerance: Optional[LinearUnit] = None,
    populate_route_edges: bool = False,
    populate_directions: bool = True,
    directions_language: str = "en",
    directions_distance_units: str = "Miles",
    directions_style_name: str = "NA Desktop",
    travel_mode: str = "Custom",
    impedance: str = "Drive Time",
    overrides: Optional[dict] = None,
    time_impedance: str = "TravelTime",
    save_route_data: bool = False,
    distance_impedance: str = "Kilometers",
    output_format: str = "Feature Set",
    save_output_na_layer: bool = False,
    time_zone_for_time_windows: str = "Geographically Local",
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """

    ``find_routes`` determines the shortest paths to visit the input stops and
    returns the driving directions, information about the visited stops,
    and the route paths, including travel time and distance. The tool is
    capable of finding routes that visit several input stops in a sequence
    you predetermine or in the  sequence that minimizes overall travel. You
    can group the input stops into different routes using the RouteName
    field, and the tool will output one route for each group of stops,
    allowing you to generate routes for many vehicles in a single solve
    operation.

    ======================================  ==========================================================================================================================================
    **Parameter**                            **Description**
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    stops                                   Required :class:`~arcgis.features.FeatureSet` . Specify the locations you want the output route
                                            or routes to visit. You can add up to 10,000 stops and assign up to 150 stops to a single
                                            route. (Assign stops to routes using the ``RouteName`` attribute). When specifying the stops,
                                            you can set properties for each one, such as its name or service time, by using
                                            attributes.
                                            The stops can be specified with the following attributes:

                                            * ``Name``: The name of the stop. This name is used when generating driving directions. It is common to pass the name of a business, person, or street address at the stop. If a value is not specified, an automatically generated name such as Location 1 or Location 2 is assigned to each stop.

                                            * ``TimeWindowStart``: The earliest time the stop can be visited. Specify this attribute if you want to limit when
                                              a route can arrive at a stop; for instance, you may want to make deliveries to a restaurant between
                                              busy lunch and dinner hours (for example, sometime between 2:00 and 4:30 p.m.) to facilitate the work
                                              for you and the restaurant staff.

                                              The value is specified as an integer that represents the number of milliseconds since epoch (January 1, 1970).

                                              This value can be specified either in UTC or local time, depending on the value given for
                                              the ``timeWindowsAreUTC`` parameter.

                                              If you specify this attribute, you also need to specify the ``TimeWindowEnd`` attribute.
                                            * ``TimeWindowEnd``: The latest time the stop can be visited. Together, the ``TimeWindowStart`` and ``TimeWindowEnd``
                                              attributes make up the time window within which a route can visit the stop. As with ``TimeWindowStart``,
                                              the ``TimeWindowEnd`` value is specified as an integer that represents the number of milliseconds since
                                              epoch (January 1, 1970) and is interpreted as UTC or local time, depending on the value specified for
                                              the ``timeWindowsAreUTC`` parameter.

                                              The time window specified using the ``TimeWindowStart`` and ``TimeWindowEnd`` attributes is not considered a hard
                                              constraint by the service. That is, the service doesn't fail if the stop cannot be visited during the time
                                              window; instead, the service tries to find a route that visits the stop during its time window, but if time
                                              window violations are inevitable, the service tries to find a solution that minimizes the time-window violation
                                              time for all stops in the problem.

                                              If a route has to arrive early at the stop, a wait time is added to the total travel time of the route.
                                              Similarly, if the route arrives late at the stop, a violation time is added to the total travel time of
                                              the route. For example, If the time window on the stop is set as 10:00 AM to 11:00 AM and the earliest a
                                              route can reach the stop is 11:25 AM, a violation of 25 minutes is added to the total travel time.
                                            * ``RouteName``: The name of the route to which the stop belongs. Use this attribute to group stops into different
                                              routes and, therefore, solve multiple routes in a single request. For example, if you want to find two distinct
                                              routes - a route with 4 stops and another with 6 stops - set RouteName to Route1 for each of the four stops and Route2
                                              for each of the 6 stops. The service will produce two distinct routes and driving directions for each group of
                                              stops in a single request.

                                              If ``RouteName`` is not specified for any stops, all stops belong to the same route. If ``RouteName`` is not
                                              specified for some stops, those stops are treated as unassigned and are not included in any route.
                                            * ``Sequence``: The output routes will visit the stops in the order you specify with this
                                              attribute. Within a group of stops that have the same RouteName value, the sequence
                                              number should be greater than 0 but not greater than the total number of stops. Also, the
                                              sequence number should not be duplicated. If Reorder Stops To Find Optimal Routes is
                                              checked (True), all but possibly the first and last sequence values for each route name
                                              are ignored so the tool can find the sequence that minimizes overall travel for each
                                              route. (The settings for Preserve Ordering of Stops and Return to Start determine whether
                                              the first or last sequence values for each route are ignored.)
                                            * ``CurbApproach``: Specify the direction a vehicle may arrive at and depart from the stop. One of the
                                              integers listed in the Coded value column in the following table must be specified as a value of this
                                              attribute. The values in the Setting column are the descriptive names for the CurbApproach attribute
                                              values that you may have come across when using ArcGIS Network Analyst extension extension software.

                                              =========================  ===============================================================
                                              **Setting**                **Description**
                                              -------------------------  ---------------------------------------------------------------
                                              Either side of vehicle     |either|
                                                                         The vehicle can approach and depart the stop in either direction,
                                                                         so a U-turn is allowed at the stop. This is the default value. This setting
                                                                         can be chosen if it is possible and desirable for your vehicle to turn
                                                                         around at the stop. This decision may depend on the width of the road and
                                                                         the amount of traffic or whether the stop has a parking lot where vehicles
                                                                         can pull in and turn around.
                                              -------------------------  ---------------------------------------------------------------
                                              right side of vehicle      |right|
                                                                         When the vehicle approaches and departs the stop, the stop must be on the
                                                                         right side of the vehicle. A U-turn is prohibited. This is typically used
                                                                         for vehicles like busses that must arrive with the bus stop on the right-hand side.
                                              -------------------------  ---------------------------------------------------------------
                                              left side of vehicle       |left|
                                                                         When the vehicle approaches and departs the stop, the stop must be on the left
                                                                         side of the vehicle. A U-turn is prohibited. This is typically used for vehicles
                                                                         like busses that must arrive with the bus stop on the left-hand side.
                                              -------------------------  ---------------------------------------------------------------
                                              No U-Turn                  |turn|
                                                                         When the vehicle approaches the stop, the stop can be on either side of the vehicle;
                                                                         however, when it departs, the vehicle must continue in the same direction it arrived in. A U-turn is prohibited.
                                              =========================  ===============================================================

                                              The ``CurbApproach`` property is designed to work with both kinds of national driving standards: right-hand
                                              traffic (United States) and left-hand traffic (United Kingdom). First, consider a stop on the left side of a vehicle.
                                              It is always on the left side regardless of whether the vehicle travels on the left or right half of the road. What
                                              may change with national driving standards is your decision to approach from the right or left side. For example,
                                              if you want to arrive at a stop and not have a lane of traffic between the vehicle and the stop, you would choose
                                              Right side of vehicle in the United States but Left side of vehicle in the United Kingdom.
                                            * ``Attr_TravelTime``: Specify the amount of time for cars, in minutes, that will be spent at the stop when
                                              the route visits it. This attribute can be used to model the time required to provide some kind of service
                                              while you are at the stop. It can also be used to specify some additional time required to reach the actual
                                              location on the street from where the route starts or time required to reach the actual destination location
                                              from the location on the street where the route ends. The value for this attribute is included in the total
                                              travel time for the route and is also displayed in driving directions as service time. A zero or null value
                                              indicates that the stop requires no service time.

                                              For example, suppose you are finding the best route through three stops. Suppose it requires 2 minutes to
                                              walk to the street location from where the route starts, you need to spend 10 minutes at Stop 2, and it takes 5
                                              minutes to walk from the street location to the destination. The Attr_TravelTime attribute should be given
                                              values of 2, 10, and 5 for Stop 1, Stop 2, and Stop 3, respectively. If it takes 10 minutes to travel from Stop
                                              1 to Stop 2 and 10 minutes to travel from Stop 2 to Stop 3, the total travel time to reach Stop 3 is displayed
                                              as 37 minutes (2 + 10 + 10 + 10 + 5), even though there is only 20 minutes of traveling to reach Stop 3.
                                            * ``Attr_TruckTravelTime``: Specify the amount of time for trucks, in minutes, that will be added to the total
                                              travel time of the route at the stop. The attribute value can be used to model the time spent at the stop.

                                              The value for this attribute is included in the total travel time for the route and is also displayed in
                                              driving directions as service time. A zero or null value indicates that the incident requires no service time.
                                              The default value is 0.
                                            * ``Attr_WalkTime``: Specify the amount of time for pedestrians, in minutes, that will be added to the total
                                              travel time of the route at the stop. The attribute value can be used to model the time spent at the incident.
                                              The value for this attribute is included in the total travel time for the route and is also displayed in
                                              driving directions as service time. A zero or null value indicates that the incident requires no service time.
                                              The default value is 0.
                                            * ``Attr_Miles``: Specify the distance in miles that will be added when calculating total distance of the route.
                                              Generally the locations of the stops are not exactly on the streets but are set back somewhat from the road.
                                              The Attr_Miles attribute can be used to model the distance between the actual stop location and its location
                                              on the street if it is important to include that distance in the total travel distance.
                                            * ``Attr_Kilometers``: Specify the distance in kilometers that will be added when calculating total
                                              distance of the route. Generally the locations of the stops are not exactly on the streets but are set back
                                              somewhat from the road. The Attr_Kilometers attribute can be used to model the distance between the actual
                                              stop location and its location on the street if it is important to include that distance in the total travel distance.
                                            * ``LocationType``: The stop type.

                                              =================  ==============  ===============================================================
                                              **Setting**        Coded value     **Description**
                                              -----------------  --------------  ---------------------------------------------------------------
                                              Stop               0               A location that the route should visit. This is the default.
                                              -----------------  --------------  ---------------------------------------------------------------
                                              Waypoint           1               A location that the route should travel through without making
                                                                                 a stop. Waypoints can be used to force the route to take a specific
                                                                                 path (to go through the waypoint) without being considered an actual stop.
                                                                                 Waypoints do not appear in driving directions.
                                              -----------------  --------------  ---------------------------------------------------------------
                                              Break              2               A location where the route stops for the driver to take a break.
                                              =================  ==============  ===============================================================

                                            * ``Bearing``: Specify the direction the vehicle or person is moving in. Bearing is measured clockwise from
                                              true north and must be in degrees. Typically, values are between 0 and 360; however, negative values are
                                              interpreted by subtracting them from 360 degrees.

                                            * ``BearingTol``: Short for bearing tolerance, this field specifies the maximum acceptable difference between the
                                              heading of a vehicle and a tangent line from the point on a street where Network Analyst attempts to locate the
                                              vehicle. The bearing tolerance is used to determine whether the direction in which a vehicle is moving generally
                                              aligns with the underlying road. If they align within the given tolerance, the vehicle is located on that edge;
                                              if not, the next nearest eligible edge is evaluated.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    measurement_units                       Required string. Specify the units that should be used to measure and report the total travel time or travel
                                            distance for the output routes. The units you choose for this parameter determine whether the tool will measure
                                            distance or time to find the best routes. Choose a time unit to minimize travel time for your chosen travel
                                            mode (driving or walking time, for instance). To minimize travel distance for the given travel mode, choose a
                                            distance unit. Your choice also determines in which units the tool will report total time or distance in the results.

                                            Choice list:['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles', 'NauticalMiles', 'Seconds', 'Minutes', 'Hours', 'Days']

    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    analysis_region                         Optional string. Specify the region in which to perform the analysis. If a value is not specified for this parameter, the tool
                                            will automatically calculate the region name based on the location of the input points. Setting the name of the region is required only if the
                                            auto-detection of the region name is not accurate for your inputs. To specify a region, use one of the following values:  Europe,Japan,Korea,MiddleEast,
                                            and Africa.  NorthAmerica, SouthAmerica, SouthAsia,Thailand; The following region names are no longer supported and will be removed in future releases.
                                            If you specify one of the deprecated region names, the tool automatically assigns a supported region name for your region. Greece redirects to Europe.
                                            India redirects to South Asia. Oceania redirects to South Asia. South East Asia redirects to South Asia. Taiwan redirects to South Asia.

                                            Choice list:['Europe', 'Japan', 'Korea', 'MiddleEastAndAfrica', 'NorthAmerica', 'SouthAmerica', 'SouthAsia', 'Thailand']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    reorder_stops_to_find                   Optional boolean. Specify whether to visit the stops in the order you define or the order the tool determines will
                                            minimize overall travel.

                                            * Checked (True) - The tool determines the sequence that will minimize overall travel
                                              distance or time. It can reorder stops and account for time windows at stops. Additional parameters allow you to
                                              preserve the first or last stops while allowing the tool to reorder the intermediary stops.

                                            * Unchecked (False) - The stops are visited in the order you define. This is the default option. You can set the order
                                              of stops using a Sequence attribute in the input stops features or let the sequence be determined by the Object ID
                                              of the stops. Finding the optimal stop order and the best routes is commonly known as solving the traveling salesman
                                              problem (TSP).
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    preserve_terminal_stops                 Optional string. When Reorder Stops to Find Optimal Routes is checked (or True), you have options to preserve
                                            the starting or ending stops and the tool can reorder the rest. The first and last stops are determined by their
                                            Sequence attribute values or, if the Sequence values are null, by their Object ID values.

                                            Preserve First - The tool won't reorder the first stop. Choose this option if you are starting from a known location, such as your home,
                                            headquarters, or current location.Preserve Last-The tool won't reorder the last stop. The output routes may start
                                            from any stop feature but must end at the predetermined last stop. Preserve First and Last-The tool won't reorder
                                            the first and last stops.

                                            Preserve None - The tool may reorder any stop, including the first and last stops. The
                                            route may start or end at any of the stop features.Preserve Terminal Stops is ignored when Reorder Stops to Find
                                            Optimal Routes is unchecked (or False).

                                            Choice list: ['Preserve First', 'Preserve Last', 'Preserve First and Last', 'Preserve None']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    return_to_start                         Optional boolean. Choose whether routes should start and end at the same location. With this option you can
                                            avoid duplicating the first stop feature and sequencing the duplicate stop at the end.The starting location of
                                            the route is the stop feature with the lowest value in the Sequence attribute. If the Sequence values are null,
                                            it is the stop feature with the lowest Object ID value.

                                            Checked (True) - The route should start and end at the first
                                            stop feature. This is the default value. When Reorder Stops to Find Optimal Routes and Return to Start are both
                                            checked (or True), Preserve Terminal Stops must be set to Preserve First.

                                            Unchecked (False) - The route won't start and end at the first stop feature.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    use_time_windows                        Optional boolean. Check this option (or set it to True) if any input stops have time windows that specify when
                                            the route should reach the stop. You can add time windows to input stops by entering time values in the
                                            ``TimeWindowStart`` and ``TimeWindowEnd`` attributes.

                                            Checked (True) - The input stops have time windows and you want the tool to try to honor them.

                                            Unchecked (False) - The input stops don't have time windows, or if they do, you don't want the tool to try to
                                            honor them. This is the default value.The tool will take slightly longer to run when Use Time Windows is checked
                                            (or True), even when none of the input stops have time windows, so it is recommended to uncheck this option
                                            (set to False) if possible.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_of_day                             Optional datetime. Specifies the time and date at which the routes should
                                            begin. If you are modeling the driving travel mode and specify the current date and time as the value
                                            for this parameter, the tool will use live traffic conditions to
                                            find the best routes and the total travel time will be based
                                            on traffic conditions. Specifying a time of day results in more accurate
                                            routes and estimations of travel times because the
                                            travel times account for the traffic conditions that are applicable
                                            for that date and time.The Time Zone for Time of Day parameter specifies whether this time and date refer to
                                            UTC or the time zone in which the stop is located.The tool ignores this parameter when Measurement Units isn't
                                            set to a time-based unit.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_zone_for_time_of_day               Optional string. Specifies the time zone of the Time of Day parameter.Geographically Local-The Time of Day
                                            parameter refers to the time zone in which the first stop of a route is located. If you are generating many
                                            routes that start in multiple times zones, the start times are staggered in Coordinated Universal Time (UTC).
                                            For example, a Time of Day value of 10:00 a.m., 2 January, would mean a start time of 10:00 a.m. Eastern Standard
                                            Time (UTC-3:00) for routes beginning in the Eastern Time Zone and 10:00 a.m. Central Standard Time (UTC-4:00) for
                                            routes beginning in the Central Time Zone. The start times are offset by one hour in UTC. The arrive and depart
                                            times and dates recorded in the output Stops feature class will refer to the local time zone of the first stop
                                            for each route.UTC-The Time of Day parameter refers to Coordinated Universal Time (UTC). Choose this option if
                                            you want to generate a route for a specific time, such as now, but aren't certain in which time zone the first
                                            stop will be located. If you are generating many routes spanning multiple times zones, the start times in UTC are
                                            simultaneous. For example, a Time of Day value of 10:00 a.m., 2 January, would mean a start time of 5:00 a.m.
                                            Eastern Standard Time (UTC-5:00) for routes beginning in the Eastern Time Zone and 4:00 a.m. Central Standard
                                            Time (UTC-6:00) for routes beginning in the Central Time Zone. Both routes would start at 10:00 a.m. UTC. The
                                            arrive and depart times and dates recorded in the output Stops feature class will refer to UTC.

                                            Choice list: ['Geographically Local', 'UTC']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    uturn_at_junctions                      Optional string. Use this parameter to restrict or permit the service area to make U-turns at junctions.
                                            In order to understand the parameter values, consider for a moment the following terminology: a junction is
                                            a point where a street segment ends and potentially connects to one or more other segments; a pseudo-junction is
                                            a point where exactly two streets connect to one another; an intersection is a point where three or more streets
                                            connect; and a  dead-end is where one street segment ends without connecting to another. Given this information,
                                            the parameter can have the following values:

                                            Choice list:['ALLOW_UTURNS', 'NO_UTURNS', 'ALLOW_DEAD_ENDS_ONLY', 'ALLOW_DEAD_ENDS_AND_INTERSECTIONS_ONLY']

                                            ========================================  ================================================
                                            **Parameter**                             **Description**
                                            ----------------------------------------  ------------------------------------------------
                                            ALLOW_UTURNS                              |ALLOW_UTURNS|
                                                                                      U-turns are permitted everywhere. Allowing U-turns implies
                                                                                      that the vehicle can turn around at a junction or intersection
                                                                                      and double back on the same street.
                                            ----------------------------------------  ------------------------------------------------
                                            ALLOW_DEAD_ENDS_AND _INTERSECTIONS_ONLY   |ALLOW_DEAD_ENDS_AND_INTERSECTIONS_ONLY|
                                                                                      U-turns are prohibited at
                                                                                      junctions where exactly two
                                                                                      adjacent streets meet.
                                            ----------------------------------------  ------------------------------------------------
                                            ALLOW_DEAD_ENDS_ONLY                      |ALLOW_DEAD_ENDS_ONLY|
                                                                                      U-turns are prohibited at all junctions and interesections and are permitted only at dead ends.
                                            ----------------------------------------  ------------------------------------------------
                                            NO_UTURNS                                 U-turns are prohibited at all junctions, intersections, and dead-ends.
                                                                                      Note that even when this parameter value is chosen, a route can still
                                                                                      make U-turns at stops. If you wish to prohibit U-turns at a stop, you can set
                                                                                      its CurbApproach property to the appropriate value (3).

                                                                                      The default value for this parameter is 'ALLOW_UTURNS'.
                                            ========================================  ================================================
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    point_barriers                          Optional :class:`~arcgis.features.FeatureSet`  . Specify one or more points to act as temporary
                                            restrictions or represent additional time or distance that may be
                                            required to travel on the underlying streets. For example, a point
                                            barrier can be used to represent a fallen tree along a street or
                                            time delay spent at a railroad crossing.

                                            The tool imposes a limit of 250 points that can be added
                                            as barriers.
                                            When specifying the point barriers, you can set properties for each one,
                                            such as its name or barrier type, by using attributes. The point barriers
                                            can be specified with the following attributes:

                                            * ``Name``: The name of the barrier.
                                            * ``BarrierType``: Specifies whether the point barrier restricts travel completely or adds time or distance when it is crossed. The value for this attribute is specified as one of the following integers (use the numeric code, not the name in parentheses):
                                              * 0 (Restriction) - Prohibits travel through the barrier. The barrier is referred to as a restriction point barrier since it acts as a restriction.
                                              * 2 (Added Cost) - Traveling through the barrier increases the travel time or distance by the amount specified in the Additional_Time or Additional_Distance field. This barrier type is referred to as an added-cost point barrier.

                                            * ``Additional_Time``: Indicates how much travel time is added when the
                                              barrier is traversed. This field is applicable only for added-cost
                                              barriers and only if the measurement units are time based. This field
                                              value must be greater than or equal to zero, and its units are the same as those specified in the
                                              Measurement Units parameter.
                                            * ``Additional_Distance``: Indicates how much distance is added when the barrier is
                                              traversed. This field is applicable only for added-cost barriers
                                              and only if the measurement units are distance based. The field value
                                              must be greater than or equal to zero, and its units are the same as those specified in the
                                              Measurement Units parameter.
                                            * ``Additional_Cost``: Indicates how much cost is added when the barrier is traversed.
                                              This field is applicable only for added-cost barriers and only if the travel mode used
                                              for the analysis uses an impedance attribute that is neither time-based or distance-based.
                                            * ``FullEdge``: Specify how the restriction point barriers are applied to the edge elements
                                              during the analysis. The field value is specified as one of the following integers (use the
                                              numeric code, not the name in parentheses):

                                              * 0 (False): Permits travel on the edge up to the barrier, but not through it. This is the default value.
                                              * 1 (True): Restricts travel anywhere on the associated edge.
                                            * ``CurbApproach``: Specifies the direction of traffic that is affected by the barrier.
                                              The field value is specified as one of the following integers (use the numeric code,
                                              not the name in parentheses):

                                              * 0 (Either side of vehicle): The barrier affects travel over the edge in both directions.
                                              * 1 (Right side of vehicle): Vehicles are only affected if the barrier is on their right
                                                side during the approach. Vehicles that traverse the same edge but approach the barrier
                                                on their left side are not affected by the barrier.
                                              * 2 (Left side of vehicle): Vehicles are only affected if the barrier is on their left side
                                                during the approach. Vehicles that traverse the same edge but approach the barrier on their
                                                right side are not affected by the barrier.
                                              Since junctions are points and don't have a side, barriers on junctions affect all vehicles
                                              regardless of the curb approach.

                                              The ``CurbApproach`` property was designed to work with both kinds of national driving standards:
                                              right-hand traffic (United States) and left-hand traffic (United Kingdom). First, consider a
                                              facility on the left side of a vehicle. It is always on the left side regardless of whether
                                              the vehicle travels on the left or right half of the road. What may change with national driving
                                              standards is your decision to approach a facility from one of two directions, that is, so it
                                              ends up on the right or left side of the vehicle. For example, if you want to arrive at a facility
                                              and not have a lane of traffic between the vehicle and the facility, you would choose Right side of
                                              vehicle (1) in the United States but Left side of vehicle (2) in the United Kingdom.
                                            * ``Bearing``: The direction in which a point is moving. The units are degrees and are measured clockwise
                                              from true north. This field is used in conjunction with the BearingTol field.

                                              Bearing data is usually sent automatically from a mobile device equipped with a GPS receiver.
                                              Try to include bearing data if you are loading an input location that is moving, such as a pedestrian or a vehicle.

                                              Using this field tends to prevent adding locations to the wrong edges, which can occur when a vehicle
                                              is near an intersection or an overpass for example. Bearing also helps the tool determine on which side
                                              of the street the point is.
                                            * ``BearingTol``: The bearing tolerance value creates a range of acceptable bearing values when locating moving
                                              points on an edge using the Bearing field. If the value from the Bearing field is within the range of acceptable
                                              values that are generated from the bearing tolerance on an edge, the point can be added as a network location
                                              there; otherwise, the closest point on the next-nearest edge is evaluated.

                                              The units are in degrees, and the default value is 30. Values must be greater than 0 and less than 180.
                                              A value of 30 means that when ArcGIS Network Analyst extension attempts to add a network location on an edge,
                                              a range of acceptable bearing values is generated 15 degrees to either side of the edge (left and right) and
                                              in both digitized directions of the edge.
                                            * ``NavLatency``: This field is only used in the solve process if Bearing and BearingTol also have values;
                                              however, entering a ``NavLatency`` value is optional, even when values are present in Bearing and BearingTol.
                                              ``NavLatency`` indicates how much time is expected to elapse from the moment GPS information is sent from a
                                              moving vehicle to a server and the moment the processed route is received by the vehicle's navigation device.

                                              The time units of ``NavLatency`` are the same as the units specified by the timeUnits property of the analysis object.

    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    line_barriers                           Optional :class:`~arcgis.features.FeatureSet`  .  Specify one or more lines that prohibit travel anywhere
                                            the lines intersect the streets. For example, a parade or protest
                                            that blocks traffic across several street segments can be modeled
                                            with a line barrier. A line barrier can also quickly fence off
                                            several roads from being traversed, thereby channeling possible
                                            routes away from undesirable parts of the street
                                            network.

                                            The tool imposes a limit on the number of streets you can
                                            restrict using the Line Barriers parameter. While there is no limit on
                                            the number of lines you can specify as line barriers, the combined
                                            number of streets intersected by all the lines cannot exceed
                                            500.
                                            When specifying the line barriers, you can set a name property for each one by using the following attribute:
                                            * ``Name``: The name of the barrier.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    polygon_barriers                        Optional :class:`~arcgis.features.FeatureSet`  . Specify polygons that either completely restrict travel or
                                            proportionately scale the time or distance required to travel on
                                            the streets intersected by the polygons.

                                            The service imposes a limit on the number of streets you
                                            can restrict using the Polygon Barriers parameter. While there is
                                            no limit on the number of polygons you can specify as the polygon
                                            barriers, the combined number of streets intersected by all the
                                            polygons should not exceed 2,000.
                                            When specifying the polygon barriers, you can set properties for each one,
                                            such as its name or barrier type, by using attributes.
                                            The polygon barriers can be specified with the following attributes:

                                            * ``Name``: The name of the barrier.
                                            * ``BarrierType``: Specifies whether the barrier restricts travel completely or scales the time or distance for traveling through it. The field value is specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Restriction) - Prohibits traveling through any part of the barrier.
                                                The barrier is referred to as a restriction polygon barrier since it
                                                prohibits traveling on streets intersected by the barrier. One use
                                                of this type of barrier is to model floods covering areas of the
                                                street that make traveling on those streets impossible.

                                              * 1 (Scaled Cost)-Scales the time or distance required to travel the
                                                underlying streets by a factor specified using the ScaledTimeFactor
                                                or ScaledDistanceFactor fields. If the streets are partially
                                                covered by the barrier, the travel time or distance is apportioned
                                                and then scaled. For example, a factor 0.25 would mean that travel
                                                on underlying streets is expected to be four times faster than
                                                normal. A factor of 3.0 would mean it is expected to take three
                                                times longer than normal to travel on underlying streets. This
                                                barrier type is referred to as a scaled-cost polygon barrier. It
                                                might be used to model storms that reduce travel speeds in specific
                                                regions.

                                            * ``ScaledTimeFactor``: This is the factor by which the travel time of the streets
                                              intersected by the barrier is multiplied. This field is applicable
                                              only for scaled-cost barriers and only if the measurement units are time
                                              based. The field value must be greater than zero.
                                            * ``ScaledDistanceFactor``: This is the factor by which the distance of the streets
                                              intersected by the barrier is multiplied. This attribute is
                                              applicable only for scaled-cost barriers and only if the measurement
                                              units are distance based. The attribute value must be greater than
                                              zero.
                                            * ``ScaledCostFactor``: This is the factor by which the cost of the streets intersected by the barrier is multiplied.
                                              The field value must be greater than zero.

                                              This field is applicable only for scaled-cost barriers and only if the travel mode used for the analysis uses
                                              an impedance attribute that is neither time based nor distance based.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    use_hierarchy                           Optional boolean. Specify whether hierarchy should be used when finding the best routes.

                                            Checked (True) - Use hierarchy when finding routes. When
                                            hierarchy is used, the tool prefers higher-order streets, such as
                                            freeways, to lower-order streets, such as local roads, and can be used
                                            to simulate the driver preference of traveling on freeways instead
                                            of local roads even if that means a longer trip. This is especially
                                            true when finding routes to faraway locations, because drivers on long-distance trips tend to prefer traveling on freeways where stops, intersections, and turns can be avoided. Using hierarchy is computationally faster,
                                            especially for long-distance routes, as the tool has to select the
                                            best route from a relatively smaller subset of streets.

                                            Unchecked (False) - Do not use hierarchy when finding routes. If
                                            hierarchy is not used, the tool considers all the streets and doesn't
                                            prefer higher-order streets when finding the route. This is often
                                            used when finding short-distance routes within a city.

                                            The tool automatically reverts to using hierarchy if the
                                            straight-line distance between orders, depots, or orders and depots is
                                            greater than 50 miles, even if you have set this parameter to not use hierarchy.
                                            The value you provide for this parameter is ignored unless Travel Mode is set to Custom, which is the default value.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    restrictions                            Optional string. Specify which restrictions should be honored by the tool when finding the best routes.  The value you provide for this parameter
                                            is ignored unless Travel Mode is set to Custom, which is the default value. A restriction represents a driving
                                            preference or requirement. In most cases, restrictions cause roads
                                            to be prohibited. For instance, using an Avoid Toll Roads restriction will result in a route that will include
                                            roads only when it is absolutely required to travel on toll roads in order to visit an incident or a facility.
                                            Height Restriction makes it possible to route around any clearances that are lower than the height of your vehicle.
                                            If you are carrying corrosive materials on your vehicle, using the Any Hazmat Prohibited restriction prevents hauling
                                            the materials along roads where it is marked as illegal to do so.

                                            Below is a list of available restrictions and a short description.
                                            Some restrictions require an additional value to be
                                            specified for their desired use. This value needs to be associated
                                            with the restriction name and a specific parameter intended to work
                                            with the restriction. You can identify such restrictions if their
                                            names appear under the ``AttributeName`` column in the Attribute
                                            Parameter Values parameter. The ``ParameterValue`` field should be
                                            specified in the Attribute Parameter Values parameter for the
                                            restriction to be correctly used when finding traversable roads.

                                            Some restrictions are supported only in certain countries; their availability is stated by region in the list below.
                                            Of the restrictions that have limited availability within a region, you can check whether the restriction is available
                                            in a particular country by looking at the table in the Country List section of the Data coverage for network analysis
                                            services web page. If a country has a value of  Yes in the Logistics Attribute column, the restriction with select
                                            availability in the region is supported in that country. If you specify restriction names that are not available in
                                            the country where your incidents are located, the service ignores the invalid restrictions. The service also ignores
                                            restrictions whose Restriction Usage parameter value is between 0 and 1 (see the Attribute Parameter Value parameter).
                                            It prohibits all restrictions whose Restriction Usage parameter value is greater than 0.

                                            Choice list:['Any Hazmat Prohibited', 'Avoid Carpool Roads', 'Avoid Express Lanes', 'Avoid Ferries', 'Avoid Gates',
                                            'Avoid Limited Access Roads', 'Avoid Private Roads', 'Avoid Roads Unsuitable for Pedestrians', 'Avoid Stairways',
                                            'Avoid Toll Roads', 'Avoid Toll Roads for Trucks', 'Avoid Truck Restricted Roads', 'Avoid Unpaved Roads',
                                            'Axle Count Restriction', 'Driving a Bus', 'Driving a Delivery Vehicle', 'Driving a Taxi', 'Driving a Truck',
                                            'Driving an Automobile', 'Driving an Emergency Vehicle', 'Height Restriction',
                                            'Kingpin to Rear Axle Length Restriction', 'Length Restriction', 'Preferred for Pedestrians',
                                            'Riding a Motorcycle', 'Roads Under Construction Prohibited', 'Semi or Tractor with One or More Trailers Prohibited',
                                            'Single Axle Vehicles Prohibited', 'Tandem Axle Vehicles Prohibited', 'Through Traffic Prohibited',
                                            'Truck with Trailers Restriction', 'Use Preferred Hazmat Routes', 'Use Preferred Truck Routes', 'Walking',
                                            'Weight Restriction', 'Weight per Axle Restriction', 'Width Restriction']

                                            The service supports the restriction names listed in the following table:

                                            ========================================  ================================================
                                            **Parameter**                             **Description**
                                            ----------------------------------------  ------------------------------------------------
                                            Any Hazmat Prohibited                     The results will not include roads
                                                                                      where transporting any kind of hazardous material is
                                                                                      prohibited.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Carpool Roads                       The results will avoid roads that are
                                                                                      designated exclusively for carpool (high-occupancy)
                                                                                      vehicles.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Express Lanes                       The results will avoid roads designated
                                                                                      as express lanes.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Ferries                             The results will avoid ferries.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Gates                               The results will avoid roads where there are
                                                                                      gates such as keyed access or guard-controlled
                                                                                      entryways.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Limited Access Roads                The results will avoid roads
                                                                                      that are limited access highways.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Private Roads                       The results will avoid roads that are
                                                                                      not publicly owned and maintained.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Roads Unsuitable for Pedestrians    The result will avoid roads that are unsuitable for pedestrians.
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Stairways                           The result will avoid all stairways on a pedestrian suitable route.
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Toll Roads                          The results will avoid toll
                                                                                      roads.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Toll Roads for Trucks               The result will avoid all toll roads for trucks
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Truck Restricted Roads              The result will avoid roads where trucks are not allowed except when making deliveries.
                                            ----------------------------------------  ------------------------------------------------
                                            Avoid Unpaved Roads                       The results will avoid roads that are
                                                                                      not paved (for example, dirt, gravel, and so on).
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Axle Count Restriction                    The results will not include roads
                                                                                      where trucks with the specified number of axles are prohibited. The
                                                                                      number of axles can be specified using the Number of Axles
                                                                                      restriction parameter.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Driving a Bus                             The results will not include roads where
                                                                                      buses are prohibited. Using this restriction will also ensure that
                                                                                      the results will honor one-way streets.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Driving a Taxi                            The results will not include roads where
                                                                                      taxis are prohibited. Using this restriction will also ensure that
                                                                                      the results will honor one-way streets.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Driving a Truck                           The results will not include roads where
                                                                                      trucks are prohibited. Using this restriction will also ensure that
                                                                                      the results will honor one-way streets.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Driving an Automobile                     The results will not include roads
                                                                                      where automobiles are prohibited. Using this restriction will also
                                                                                      ensure that the results will honor one-way streets.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Driving an Emergency Vehicle              The results will not include
                                                                                      roads where emergency vehicles are prohibited. Using this
                                                                                      restriction will also ensure that the results will honor one-way
                                                                                      streets.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Height Restriction                        The results will not include roads
                                                                                      where the vehicle height exceeds the maximum allowed height for the
                                                                                      road. The vehicle height can be specified using the Vehicle Height
                                                                                      (meters) restriction parameter.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Kingpin to Rear Axle                      The results will
                                            Length Restriction                        not include roads where the vehicle length exceeds the maximum
                                                                                      allowed kingpin to rear axle for all trucks on the road. The length
                                                                                      between the vehicle kingpin and the rear axle can be specified
                                                                                      using the Vehicle Kingpin to Rear Axle Length (meters) restriction
                                                                                      parameter.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Length Restriction                        The results will not include roads
                                                                                      where the vehicle length exceeds the maximum allowed length for the
                                                                                      road. The vehicle length can be specified using the Vehicle Length
                                                                                      (meters) restriction parameter.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Preferred for Pedestrians                 The result prefers paths designated for pedestrians.
                                            ----------------------------------------  ------------------------------------------------
                                            Riding a Motorcycle                       The results will not include roads
                                                                                      where motorcycles are prohibited. Using this restriction will also
                                                                                      ensure that the results will honor one-way streets.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Roads Under Construction Prohibited       The results will not
                                                                                      include roads that are under construction.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Semi or Tractor with One                  The results will not include roads where semis or tractors with
                                            or More Trailers Prohibited               one or more trailers are prohibited.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Single Axle Vehicles Prohibited           The results will not
                                                                                      include roads where vehicles with single axles are
                                                                                      prohibited.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Tandem Axle Vehicles Prohibited           The results will not
                                                                                      include roads where vehicles with tandem axles are
                                                                                      prohibited.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Through Traffic Prohibited                The results will not include
                                                                                      roads where through traffic (non local) is prohibited.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Truck with Trailers Restriction           The results will not
                                                                                      include roads where trucks with the specified number of trailers on
                                                                                      the truck are prohibited. The number of trailers on the truck can
                                                                                      be specified using the Number of Trailers on Truck restriction
                                                                                      parameter.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Use Preferred Hazmat Routes               The results will prefer roads
                                                                                      that are designated for transporting any kind of hazardous
                                                                                      materials.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Use Preferred Truck Routes                The results will prefer roads
                                                                                      that are designated as truck routes, such as the roads that are
                                                                                      part of the national network as specified by the National Surface
                                                                                      Transportation Assistance Act in the United States, or roads that
                                                                                      are designated as truck routes by the state or province, or roads
                                                                                      that are preferred by the trucks when driving in an
                                                                                      area.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Walking                                   The results will not include roads where
                                                                                      pedestrians are prohibited.
                                                                                      Availability: All countries
                                            ----------------------------------------  ------------------------------------------------
                                            Weight Restriction                        The results will not include roads
                                                                                      where the vehicle weight exceeds the maximum allowed weight for the
                                                                                      road. The vehicle weight can be specified using the Vehicle Weight
                                                                                      (kilograms) restriction parameter.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Weight per Axle Restriction               The results will not include
                                                                                      roads where the vehicle weight per axle exceeds the maximum allowed
                                                                                      weight per axle for the road. The vehicle weight per axle can be
                                                                                      specified using the Vehicle Weight per Axle (kilograms) restriction
                                                                                      parameter.
                                                                                      Availability: Select countries in North America and Europe
                                            ----------------------------------------  ------------------------------------------------
                                            Width Restriction                         The results will not include roads where
                                                                                      the vehicle width exceeds the maximum allowed width for the road.
                                                                                      The vehicle width can be specified using the Vehicle Width (meters)
                                                                                      restriction parameter.
                                                                                      Availability: Select countries in North America and Europe
                                            ========================================  ================================================

    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    attribute_parameter_values              Optional :class:`~arcgis.features.FeatureSet`  .  Specify additional values required by some restrictions, such as the weight of a vehicle
                                            for Weight Restriction. You can also use the attribute parameter to specify whether any restriction prohibits,
                                            avoids, or prefers travel on roads that use the restriction. If the restriction is
                                            meant to avoid or prefer roads, you can further specify the degree
                                            to which they are avoided or preferred using this
                                            parameter. For example, you can choose to never use toll roads, avoid them as much as possible, or even highly
                                            prefer them.
                                            The value you provide for this parameter is ignored unless Travel Mode is set to Custom, which is the default value.
                                            If you specify the Attribute Parameter Values parameter from a feature class, the field names on the feature class must match the fields as described below:

                                            * ``AttributeName``: Lists the name of the restriction.
                                            * ``ParameterName``: Lists the name of the parameter associated with the restriction. A restriction can have one or more ParameterName field
                                              values based on its intended use.
                                            * ``ParameterValue``: The value for ``ParameterName`` used by the tool when evaluating the restriction.

                                              Attribute Parameter Values is dependent on the Restrictions parameter. The ``ParameterValue`` field is applicable only
                                              if the restriction name is specified as the value for the
                                              Restrictions parameter.

                                              In Attribute Parameter Values, each restriction (listed as `AttributeName`) has a ``ParameterName`` field
                                              value, Restriction Usage, that specifies whether the restriction
                                              prohibits, avoids, or prefers travel on the roads associated with
                                              the restriction and the degree to which the roads are avoided or
                                              preferred. The Restriction Usage ``ParameterName`` can be assigned any of
                                              the following string values or their equivalent numeric values
                                              listed within the parentheses:

                                              * ``PROHIBITED`` (-1) - Travel on the roads using the restriction is completely prohibited.

                                              * ``AVOID_HIGH`` (5) - It
                                                is highly unlikely for the tool to include in the route the roads
                                                that are associated with the restriction.

                                              * ``AVOID_MEDIUM`` (2) - It
                                                is unlikely for the tool to include in the route the roads that are
                                                associated with the restriction.

                                              * ``AVOID_LOW`` (1.3) - It
                                                is somewhat unlikely for the tool to include in the route the roads
                                                that are associated with the restriction.

                                              * ``PREFER_LOW`` (0.8) - It
                                                is somewhat likely for the tool to include in the route the roads
                                                that are associated with the restriction.

                                              * ``PREFER_MEDIUM`` (0.5) - It is likely for the tool to include in the route the roads that
                                                are associated with the restriction.

                                              * ``PREFER_HIGH`` (0.2) - It is highly likely for the tool to include in the route the roads
                                                that are associated with the restriction.

                                              In most cases, you can use the default value, PROHIBITED,
                                              for the Restriction Usage if the restriction is dependent on a
                                              vehicle-characteristic such as vehicle height. However, in some
                                              cases, the value for Restriction Usage depends on your routing
                                              preferences. For example, the Avoid Toll Roads restriction has the
                                              default value of AVOID_MEDIUM for the Restriction Usage parameter.
                                              This means that when the restriction is used, the tool will try to
                                              route around toll roads when it can. AVOID_MEDIUM also indicates
                                              how important it is to avoid toll roads when finding the best
                                              route; it has a medium priority. Choosing AVOID_LOW would put lower
                                              importance on avoiding tolls; choosing AVOID_HIGH instead would
                                              give it a higher importance and thus make it more acceptable for
                                              the service to generate longer routes to avoid tolls. Choosing
                                              PROHIBITED would entirely disallow travel on toll roads, making it
                                              impossible for a route to travel on any portion of a toll road.
                                              Keep in mind that avoiding or prohibiting toll roads, and thus
                                              avoiding toll payments, is the objective for some; in contrast,
                                              others prefer to drive on toll roads because avoiding traffic is
                                              more valuable to them than the money spent on tolls. In the latter
                                              case, you would choose PREFER_LOW, PREFER_MEDIUM, or PREFER_HIGH as
                                              the value for Restriction Usage. The higher the preference, the
                                              farther the tool will go out of its way to travel on the roads
                                              associated with the restriction.

                                            ========================================  =========================  =======================
                                            **AttributeName**                             **ParameterName**      **ParameterValue**
                                            ----------------------------------------  -------------------------  -----------------------
                                            Any Hazmat Prohibited                     Restriction Usage          PROHIBITED

                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Carpool Roads                        Restriction Usage         PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Express Lanes                        Restriction Usage         PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Ferries                              Restriction Usage         AVOID_MEDIUM
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Gates                               Restriction Usage          AVOID_MEDIUM
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Limited Access Roads                Restriction Usage          AVOID_MEDIUM
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Private Roads                       Restriction Usage          AVOID_MEDIUM
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Roads Unsuitable for Pedestrians    Restriction Usage          AVOID_HIGH
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Stairways                           Restriction Usage          AVOID_HIGH
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Toll Roads                          Restriction Usage          AVOID_MEDIUM
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Toll Roads for Trucks               Restriction Usage          AVOID_MEDIUM
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Truck Restricted Roads              Restriction Usage          AVOID_HIGH
                                            ----------------------------------------  -------------------------  -----------------------
                                            Avoid Unpaved Roads                       Restriction Usage          AVOID_HIGH
                                            ----------------------------------------  -------------------------  -----------------------
                                            Axle Count Restriction                    Number of Axles            0

                                                                                      Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Driving a Bus                             Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Driving a Taxi                            Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Driving a Truck                           Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Driving an Automobile                     Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Driving an Emergency Vehicle              Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Height Restriction                        Restriction Usage          PROHIBITED

                                                                                      Vehicle Height (meters)    0
                                            ----------------------------------------  -------------------------  -----------------------
                                            Kingpin to Rear Axle                      Restriction Usage          PROHIBITED
                                            Length Restriction
                                                                                      Vehicle Kingpin to Rear    0
                                                                                      Axle Length (meters)
                                            ----------------------------------------  -------------------------  -----------------------
                                            Length Restriction                        Restriction Usage          PROHIBITED

                                                                                      Vehicle Length (meters)    0
                                            ----------------------------------------  -------------------------  -----------------------
                                            Preferred for Pedestrians                 Restriction Usage          PREFER_LOW
                                            ----------------------------------------  -------------------------  -----------------------
                                            Riding a Motorcycle                       Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Roads Under Construction Prohibited       Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Semi or Tractor with One                  Restriction Usage          PROHIBITED
                                            or more trailers prohibited
                                            ----------------------------------------  -------------------------  -----------------------
                                            Single Axle Vehicles Prohibited           Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Tandem Axle Vehicles Prohibited           Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            Through Traffic Prohibited                Restriction Usage          AVOID_HIGH
                                            ----------------------------------------  -------------------------  -----------------------
                                            Truck with Trailers Restriction           Restriction Usage          PROHIBITED

                                                                                      Number of Trailers         0
                                                                                      on Truck
                                            ----------------------------------------  -------------------------  -----------------------
                                            Use Preferred Hazmat Routes               Restriction Usage          PREFER_MEDIUM
                                            ----------------------------------------  -------------------------  -----------------------
                                            Use Preferred Truck Routes                Restriction Usage          PREFER_HIGH
                                            ----------------------------------------  -------------------------  -----------------------
                                            Walking                                   Restriction Usage          PROHIBITED
                                            ----------------------------------------  -------------------------  -----------------------
                                            WalkTime                                  Walking Speed (km/h)       5
                                            ----------------------------------------  -------------------------  -----------------------
                                            Weight Restriction                        Restriction Usage          PROHIBITED

                                                                                      Vehicle Weight             0
                                                                                      (kilograms)
                                            ----------------------------------------  -------------------------  -----------------------
                                            Weight per Axle Restriction               Restriction Usage          PROHIBITED

                                                                                      Vehicle Weight per         0
                                                                                      Axle (kilograms)
                                            ----------------------------------------  -------------------------  -----------------------
                                            Width Restriction                         Restriction Usage          PROHIBITED

                                                                                      Vehicle Width              0
                                                                                      (meters)
                                            ========================================  =========================  =======================

    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    route_shape                             Optional string. Specify the type of route features that are output by the
                                            tool. The parameter can be specified using one of the following
                                            values:

                                            * True Shape - Return the exact shape of the resulting route
                                              that is based on the underlying streets.

                                            * Straight Line - Return a straight line between the
                                              incident and the facility.

                                            * None - Do not return any shapes for the routes. This value
                                              can be useful in cases where you are only interested in determining
                                              the total travel time or travel distance between the closest
                                              facility and the incident.

                                            When the Route Shape parameter is set to True Shape, the
                                            generalization of the route shape can be further controlled using
                                            the appropriate values for the Route Line Simplification Tolerance
                                            parameters.

                                            No matter which value you choose for the Route Shape
                                            parameter, the best route is always determined by minimizing the
                                            travel time or the travel distance, never using the straight-line
                                            distance between incidents and
                                            facilities. This means that only the route shapes are different,
                                            not the underlying streets that are searched when finding the
                                            route.

                                            Choice list: ['True Shape', 'Straight Line', 'None']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    route_line_simplif ication_tolerance    Optional LinearUnit. Specify by how much you want to simplify the geometry of the output lines for
                                            routes and directions. The value you provide for this parameter is ignored unless Travel Mode is set to
                                            Custom, which is the default value. The tool also ignores this parameter if the ``populate_route_lines`` parameter
                                            is unchecked (False).
                                            Simplification maintains critical
                                            points on a route, such as turns at intersections, to define the
                                            essential shape of the route and removes other points. The
                                            simplification distance you specify is the maximum allowable offset
                                            that the simplified line can deviate from the original line.
                                            Simplifying a line reduces the number of vertices that are part of
                                            the route geometry. This improves the tool execution
                                            time.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    populate_route_edges                    Optional boolean. Specify whether the tool should generate edges for each route. Route edges represent
                                            the individual street features or other similar features that are traversed by a route. The output Route
                                            Edges layer is commonly used to see which streets or paths are traveled on the most or least by the resultant
                                            routes.

                                            Checked (True) - Generate route edges. The output Route Edges layer is populated with line features.

                                            Unchecked (False) - Don't generate route edges. The output Route Edges layer is returned, but it is empty.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    populate_directions                     Optional boolean. Specify whether the tool should generate driving directions for
                                            each route.
                                            Checked (True):
                                            Indicates that the directions will be generated
                                            and configured based on the values for the Directions Language,
                                            Directions Style Name, and Directions Distance Units
                                            parameters.
                                            Unchecked (False):
                                            Directions are not generated, and the tool
                                            returns an empty Directions layer.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    directions_language                     Optional string. Specify the language that should be used when generating
                                            driving directions.
                                            This parameter is used only when the populate_directions parameter is checked, or set to True.
                                            The parameter value can be specified using one of the following two- or five-character language codes:
                                            ar-Arabic cs-Czech  de-German el-Greek  en-English  es-Spanish et-Estonian  fr-French  he-Hebrew  it-Italian
                                            ja-Japanese  ko-Korean  lt-Lithuanian lv-Latvian  nl-Dutch  pl-Polish
                                            pt-BR-Brazilian Portuguese pt-PT-European Portuguese
                                            ru-Russian  sv-Swedish  th-Thai tr-Turkish
                                            zh-CN-Simplified Chinese

                                            If an unsupported language code is specified, the tool
                                            returns the directions using the default language,
                                            English.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    directions_distance_units               Optional string. Specify the units for displaying travel distance in the
                                            driving directions. This parameter is used only when the Populate
                                            Directions parameter is checked, or set to True.

                                            Choice list: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles', 'NauticalMiles']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    directions_style_name                   Optional string. Specify the name of the formatting style for the
                                            directions. This parameter is used only when the Populate
                                            Directions parameter is checked, or set to True. The parameter can be specified
                                            using the following values:

                                            Choice list:['NA Desktop', 'NA Navigation']

                                            ``NA Desktop``: Generates turn-by-turn directions suitable for printing.

                                            ``NA Navigation``: Generates turn-by-turn directions designed for an in-vehicle navigation device.

    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    travel_mode                             Optional string. Specify the mode of transportation to model in the analysis. Travel modes are managed in ArcGIS Online and can be configured by the administrator of your
                                            organization to better reflect your organization's workflows. You need to specify the name of a travel mode supported by your organization.

                                            To get a list of supported travel mode names, run the ``GetTravelModes`` tool from the Utilities toolbox available under the same GIS Server connection
                                            you used to access the tool. The ``GetTravelModes`` tool adds a table, Supported Travel Modes, to the application. Any value in the Travel Mode Name field from the
                                            Supported Travel Modes table can be specified as input. You can also specify the value from Travel Mode Settings field as input. This speeds up the tool execution as the
                                            tool does not have to lookup the settings based on the travel mode name.

                                            The default value, Custom, allows you to configure your own travel mode using the custom travel mode parameters (UTurn at Junctions, Use Hierarchy, Restrictions, Attribute Parameter Values,  and Impedance).
                                            The default values of the custom travel mode parameters model travelling by car. You may want to choose Custom and set the custom travel mode parameters listed above to model a pedestrian with a fast walking speed
                                            or a truck with a given height, weight, and cargo of certain hazardous materials. You may choose to do this to try out different settings to get desired analysis results.
                                            Once you have identified the analysis settings, you should work with your organization's administrator and save these settings as part of new or existing travel mode so that
                                            everyone in your organization can rerun the analysis with the same settings.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    impedance                               Optional string. Specify the impedance, which is a value that represents the effort or cost of traveling
                                            along road segments or on other parts of the transportation network.
                                            Travel time is an impedance; a car taking one minute to travel a mile along an empty road is an example of impedance.
                                            Travel times can vary by travel mode-a pedestrian may take more than 20 minutes to walk the same mile-so it is important to
                                            choose the right impedance for the travel mode you are modeling. Choose from the following impedance values: Drive Time-Models
                                            travel times for a car. These travel times are static for each road and don't fluctuate with traffic. Truck Time-Models travel
                                            times for a truck.  These travel times are static for each road and don't fluctuate with traffic. Walk Time-Models travel
                                            times for a pedestrian. The value you provide for this parameter is ignored unless Travel Mode is set to Custom, which is
                                            the default value.

                                            Choice list:['Drive Time', 'Truck Time', 'Walk Time']

                                            The default value is 'Drive Time'.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_zone_usage_for _time_windows       Optional string. Specifies the time zone for the input date-time fields supported by the tool. This
                                            parameter specifies the time zone for the following fields: ``TimeWindowStart1``, ``TimeWindowEnd1``, ``TimeWindowStart2``,
                                            ``TimeWindowEnd2``, ``InboundArriveTime``, and ``OutboundDepartTime`` on orders. ``TimeWindowStart1``, ``TimeWindowEnd1``,
                                            ``TimeWindowStart2``, and ``TimeWindowEnd2`` on depots. ``EarliestStartTime`` and ``LatestStartTime`` on routes.
                                            ``TimeWindowStart`` and ``TimeWindowEnd`` on breaks.

                                            Choice list:['UTC', 'GEO_LOCAL']

                                            GEO_LOCAL: The date-time values associated with the orders
                                            or depots are in the time zone in which the orders and depots are located. For routes, the date-time
                                            values are based on the time zone in which the starting depot for the route is located. If a route does not have a starting depot,
                                            all orders and depots across all the routes must be in a single time zone. For breaks, the date-time values are based on the time
                                            zone of the routes. For example, if your depot is located in an area that follows eastern standard time and has the first time window
                                            values (specified as TimeWindowStart1 and TimeWindowEnd1) of 8 AM and 5 PM, the time window values will be treated as 8:00 a.m. and 5:00 p.m.
                                            eastern standard time.

                                            UTC: The date-time values associated with the orders or depots are in the in coordinated universal time (UTC) and are not based on the time zone
                                            in which the orders or depots are located. For example, if your depot is located in an area that follows eastern standard time and has the first
                                            time window values (specified as TimeWindowStart1 and TimeWindowEnd1) of 8 AM and 5 PM, the time window values will be treated as 12:00 p.m.
                                            and 9:00 p.m. eastern standard time assuming the eastern standard time is obeying the daylight saving time. Specifying the date-time values
                                            in UTC is useful if you do not know the time zone in which the orders or depots are located or when you have orders and depots in multiple time
                                            zones, and you want all the date-time values to start simultaneously. The UTC option is applicable only when your network dataset
                                            defines a time zone attribute. Otherwise, all the date-time values are always treated as GEO_LOCAL.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    save_output_na_layer                    Optional boolean. Specify if the tool should save the analysis settings as a network analysis layer file. You cannot
                                            directly work with this file even when you open the file in an ArcGIS Desktop application like ArcMap. It is meant
                                            to be sent to Esri Technical Support to diagnose the quality of results returned from the tool.

                                            Checked (True) - Save the network analysis layer file. The file is downloaded in a temporary directory on your machine.
                                            In ArcGIS Pro, the location of the downloaded file can be determined  by viewing the value for the Output Network Analysis
                                            Layer parameter in the entry corresponding to the tool execution in the Geoprocessing history of your Project. In ArcMap,
                                            the location of the file can be determined by accessing the Copy Location option in the shortcut menu on the Output Network
                                            Analysis Layer parameter in the entry corresponding to the tool execution in the Geoprocessing Results window.

                                            Unchecked (False)-Do not save the network analysis layer file. This is the default.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    overrides                               Optional string.  Specify additional settings that can influence the behavior of the solver when finding solutions for the network analysis problems.
                                            The value for this parameter needs to be specified in JavaScript Object Notation (JSON). For example, a valid value is of the following form {"overrideSetting1" : "value1", "overrideSetting2" : "value2"}. The override setting name is always enclosed in double quotation marks. The values can be a number, Boolean, or a string. The default value for this parameter is no
                                            value, which indicates not to override any solver
                                            settings. Overrides are advanced settings that should be
                                            used only after careful analysis of the results obtained before and
                                            after applying the settings. A list of supported override settings
                                            for each solver and their acceptable values can be obtained by
                                            contacting Esri Technical Support.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    save_route_data                         Optional boolean. Choose whether the output includes a zip file that contains a file geodatabase holding the inputs
                                            and outputs of the analysis in a format that can be used to share route layers with ArcGIS Online or Portal for
                                            ArcGIS.
                                            True: Save the route data as a zip file. The file is downloaded in a temporary directory on your machine. In ArcGIS Pro, the location of the downloaded file can be determined by viewing the value for the Output Route Data parameter in the entry corresponding to the tool execution in the Geoprocessing history of your Project. In ArcMap, the location of the file can be determined by accessing the Copy Location option in the shortcut menu on the Output Route Data parameter in the entry corresponding to the tool execution in the Geoprocessing Results window.
                                            False: Do not save the route data. This is the default.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_impedance                          Optional string. Specify the time-based impedance, which is a value that represents the travel time along road segments or on other parts of the transportation network.If the impedance for the travel mode, as specified using the impedance parameter, is time-based, the value for time_impedance and impedance parameters should be identical. Otherwise the service will return an error.

                                            Choice list:['Minutes', 'TravelTime', 'TimeAt1KPH', 'WalkTime', 'TruckMinutes', 'TruckTravelTime']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    distance_impedance                      Optional string. Specify the distance-based impedance, which is a value that represents the travel distance along road segments or on other parts of the transportation network.If the impedance for the travel mode, as specified using the impedance parameter, is distance-based, the value for distance_impedance and impedance parameters should be identical. Otherwise the service will return an error.

                                            Choice list:['Miles', 'Kilometers']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    output_format                           Optional. Specify the format in which the output features are created.

                                            Choose from the following formats:

                                            * Feature Set - The output features are returned as feature classes and tables. This is the default.
                                            * JSON File - The output features are returned as a compressed file containing the JSON representation of the outputs. When this option is specified, the output is a single file (with a .zip extension) that contains one or more JSON files (with a .json extension) for each of the outputs created by the service.
                                            * GeoJSON File - The output features are returned as a compressed file containing the GeoJSON representation of the outputs. When this option is specified, the output is a single file (with a .zip extension) that contains one or more GeoJSON files (with a .geojson extension) for each of the outputs created by the service.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    gis                                     Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    future                                  Optional boolean. If True, a future object will be returned and the process
                                            will not wait for the task to complete. The default is False, which means wait for results.
    ======================================  ==========================================================================================================================================

    :return: the following as a named tuple:

    * solve_succeeded - Solve Succeeded as a bool
    * output_routes - Output Routes as a FeatureSet
    * output_route_edges - Output Route Edges as a FeatureSet
    * output_directions - Output Directions as a FeatureSet
    * output_stops - Output Stops as a FeatureSet
    * output_network_analysis_layer - Output Network Analysis Layer as a DataFile
    * output_route_data - Output Route Data as a DataFile
    * output_result_file - Output Result File as a DataFile

      Click `FindRoutes <https://developers.arcgis.com/rest/network/api-reference/route-asynchronous-service.htm>`_ for additional help.
    """

    if gis is None:
        gis = arcgis.env.active_gis
    url = gis.properties.helperServices.asyncRoute.url
    url = _validate_url(url, gis)
    tbx = _create_toolbox(url, gis=gis)
    defaults = dict(
        zip(tbx.find_routes.__annotations__.keys(), tbx.find_routes.__defaults__)
    )
    if stops is None:
        stops = default_stops

    if point_barriers is None:
        point_barriers = defaults["point_barriers"]

    if line_barriers is None:
        line_barriers = defaults["line_barriers"]

    if polygon_barriers is None:
        polygon_barriers = defaults["polygon_barriers"]

    if restrictions is None:
        restrictions = defaults["restrictions"]

    if attribute_parameter_values is None:
        attribute_parameter_values = defaults["attribute_parameter_values"]

    if route_line_simplification_tolerance is None:
        route_line_simplification_tolerance = defaults[
            "route_line_simplification_tolerance"
        ]
    from arcgis._impl.common._utils import inspect_function_inputs

    params = {
        "stops": stops,
        "measurement_units": measurement_units,
        "analysis_region": analysis_region,
        "reorder_stops_to_find_optimal_routes": reorder_stops_to_find_optimal_routes,
        "preserve_terminal_stops": preserve_terminal_stops,
        "return_to_start": return_to_start,
        "use_time_windows": use_time_windows,
        "time_of_day": time_of_day,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "uturn_at_junctions": uturn_at_junctions,
        "point_barriers": point_barriers,
        "line_barriers": line_barriers,
        "polygon_barriers": polygon_barriers,
        "use_hierarchy": use_hierarchy,
        "restrictions": restrictions,
        "attribute_parameter_values": attribute_parameter_values,
        "route_shape": route_shape,
        "route_line_simplification_tolerance": route_line_simplification_tolerance,
        "populate_route_edges": populate_route_edges,
        "populate_directions": populate_directions,
        "directions_language": directions_language,
        "directions_distance_units": directions_distance_units,
        "directions_style_name": directions_style_name,
        "travel_mode": travel_mode,
        "impedance": impedance,
        "time_zone_for_time_windows": time_zone_for_time_windows,
        "save_output_network_analysis_layer": save_output_na_layer,
        "overrides": overrides,
        "save_route_data": save_route_data,
        "time_impedance": time_impedance,
        "distance_impedance": distance_impedance,
        "output_format": output_format,
        "gis": gis,
        "future": True,
    }
    params = inspect_function_inputs(tbx.find_routes, **params)
    params["future"] = True
    job = tbx.find_routes(**params)
    if future:
        return job
    res = job.result()
    return res


find_routes.__annotations__ = {
    "stops": FeatureSet,
    "measurement_units": str,
    "analysis_region": str,
    "reorder_stops_to_find_optimal_routes": bool,
    "preserve_terminal_stops": str,
    "return_to_start": bool,
    "use_time_windows": bool,
    "time_of_day": datetime,
    "time_zone_for_time_of_day": str,
    "uturn_at_junctions": str,
    "point_barriers": FeatureSet,
    "line_barriers": FeatureSet,
    "polygon_barriers": FeatureSet,
    "use_hierarchy": bool,
    "restrictions": str,
    "attribute_parameter_values": FeatureSet,
    "route_shape": str,
    "route_line_simplification_tolerance": LinearUnit,
    "populate_route_edges": bool,
    "populate_directions": bool,
    "directions_language": str,
    "directions_distance_units": str,
    "directions_style_name": str,
    "travel_mode": str,
    "impedance": str,
    "return": tuple,
}
