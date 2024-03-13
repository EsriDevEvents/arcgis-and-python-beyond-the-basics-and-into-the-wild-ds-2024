from __future__ import annotations
import logging as _logging
from typing import Any, Optional, Union

import arcgis
from datetime import datetime
from arcgis.features import FeatureSet
from arcgis.gis import GIS
from arcgis.mapping import MapImageLayer
from arcgis.geoprocessing import DataFile, LinearUnit, RasterData
from arcgis.geoprocessing import import_toolbox
from arcgis.geoprocessing._support import _execute_gp_tool
from arcgis._impl.common._utils import _validate_url
from ._routing_utils import _create_toolbox

_log = _logging.getLogger(__name__)

# _url = "https://logistics.arcgis.com/arcgis/rest/services/World/ClosestFacility/GPServer"
# closest_facility = import_toolbox(gis.properties.helperServices.asyncClosestFacility.url[:-len('/FindClosestFacilities')], gis, True)
_use_async = True


default_incidents = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {"alias": "ID", "name": "ID", "type": "esriFieldTypeString", "length": 128},
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

default_facilities = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {"alias": "ID", "name": "ID", "type": "esriFieldTypeString", "length": 128},
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


def find_closest_facilities(
    incidents: Union[FeatureSet, dict[str, Any], list[FeatureSet]],
    facilities: Union[FeatureSet, dict[str, Any], list[FeatureSet]],
    measurement_units: str = "Minutes",
    analysis_region: Optional[str] = None,
    number_of_facilities_to_find: int = 1,
    cutoff: Optional[float] = None,
    travel_direction: str = "Incident to Facility",
    use_hierarchy: bool = True,
    time_of_day: Optional[datetime] = None,
    time_of_day_usage: str = "Start Time",
    uturn_at_junctions: str = "Allowed Only at Intersections and Dead Ends",
    point_barriers: Optional[
        Union[FeatureSet, dict[str, Any], list[FeatureSet]]
    ] = None,
    line_barriers: Optional[Union[FeatureSet, dict[str, Any], list[FeatureSet]]] = None,
    polygon_barriers: Optional[
        Union[FeatureSet, dict[str, Any], list[FeatureSet]]
    ] = None,
    restrictions: Optional[str] = None,
    attribute_parameter_values: Union[
        FeatureSet, dict[str, Any], list[FeatureSet]
    ] = None,
    route_shape: str = "True Shape",
    route_line_simplification_tolerance: Optional[LinearUnit] = None,
    populate_directions: bool = False,
    directions_language: str = "en",
    directions_distance_units: str = "Miles",
    directions_style_name: str = "NA Desktop",
    time_zone_for_time_of_day: str = "Geographically Local",
    travel_mode: str = "Custom",
    impedance: str = "Drive Time",
    save_output_network_analysis_layer: bool = False,
    overrides=None,
    save_route_data: bool = False,
    time_impedance: str = "TravelTime",
    distance_impedance: str = "Kilometers",
    output_format: str = "Feature Set",
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    Finds one or more facilities that are closest from an incident based on travel time or travel distance and outputs the best routes,
    driving directions between the incidents and the chosen facilities, and a copy of the chosen facilities. You can use the tool, for example,
    to find the closest hospital to an accident, the closest police cars to a crime scene, or the closest store to a customer's address., When
    finding closest facilities, you can specify how many to find and whether the direction of travel is toward or away from them. You can also
    specify the time of day to account for travel times based on live or predictive traffic conditions for that time and date. For instance,
    you can use the tool to search for hospitals within a 15-minute drive time of the site of an accident at a given time of day. Any hospitals
    that take longer than 15 minutes to reach based on the traffic conditions will not be included in the results.

    ======================================  ==========================================================================================================================================
    **Parameter**                            **Description**
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    incidents                               Required :class:`~arcgis.features.FeatureSet` . Specify one or more incidents (up to 1,000). These are the locations from which the
                                            tool searches for the nearby locations.
                                            When specifying the incidents, you can set properties for each one, such as its name or service time,
                                            by using attributes. The incidents can be specified with the following attributes:

                                            * ``Name`` - The name of the incident. The name is used in the driving directions. If the name is
                                              not specified, a unique name prefixed with Location is automatically generated in the output routes and directions.
                                            * ``ID`` - A unique identifier for the incident. The identifier is included in the output routes (as
                                              the IncidentID field) and can help join additional information from the output routes, such as the
                                              total travel time or total distance, to attributes from your incidents or vice versa. If the ID isn't
                                              specified, the service autogenerates a unique identifier for each incident.
                                            * ``AdditionalTime``-The amount
                                              of time spent at the incident, which is added to the total time of the route. The units for this attribute
                                              value are specified by the Measurement Units parameter. The attribute value is included in the analysis
                                              only when the measurement units are time based. The default value is 0. If you are finding the closest
                                              fire stations from fire incidents to estimate response times, the ``AdditionalTime`` attribute can store the
                                              amount of time it takes firefighters to hook up their equipment at the location of the incident before
                                              they can begin fighting the fire.
                                            * ``AdditionalDistance``- The extra distance traveled at the incident, which
                                              is added to the total distance of the route. The units for this attribute value are specified by the
                                              Measurement Units parameter. The attribute value is included in the analysis only when the measurement
                                              units are distance based. The default value is 0. Generally, the location of an incident, such as a home,
                                              isn't exactly on the streets; it is set back somewhat from the road. This attribute value can be used to
                                              model the distance between the actual incident location and its location on the street, if it is important
                                              to include that distance in the total travel distance.
                                            * ``CurbApproach`` - Specifies the direction a vehicle may arrive at and depart from the incident.
                                              The field value is specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              =========================  ===============================================================
                                              **Setting**                **Description**
                                              -------------------------  ---------------------------------------------------------------
                                              Either side of vehicle     |either|
                                                                          The vehicle can approach and depart the order in either
                                                                          direction, so a U-turn is allowed at the order. This setting
                                                                          can be chosen if it is possible and desirable for your vehicle
                                                                          to turn around at the order. This decision may depend on the
                                                                          width of the road and the amount of traffic or whether the
                                                                          order has a parking lot where vehicles can pull in and turn
                                                                          around.
                                              -------------------------  ---------------------------------------------------------------
                                              right side of vehicle      |right|
                                                                          When the vehicle approaches and departs the order, the order must
                                                                          be on the right side of the vehicle. A U-turn is prohibited. This is
                                                                          typically used for vehicles like buses that must arrive with the bus
                                                                          stop on the right side.
                                              -------------------------  ---------------------------------------------------------------
                                              left side of vehicle       |left|
                                                                          When the vehicle approaches and departs the order, the curb must
                                                                          be on the left side of the vehicle. A U-turn is prohibited.
                                                                          This is typically used for vehicles like buses that must arrive
                                                                          with the bus stop on the left-hand side.
                                              -------------------------  ---------------------------------------------------------------
                                              No U-Turn                  |turn|
                                                                          When the vehicle approaches the order, the curb can be on either side
                                                                          of the vehicle; however, the vehicle must depart without turning around.
                                              =========================  ===============================================================

                                              The ``CurbApproach`` property is designed to work with both kinds of national driving standards: right-hand traffic (United States) and
                                              left-hand traffic (United Kingdom). First, consider an incident on the left side of a vehicle. It is always on the left side regardless
                                              of whether the vehicle travels on the left or right half of the road. What may change with national driving standards is your decision
                                              to approach an incident from one of two directions, that is, so it ends up on the right or left side of the vehicle. For example, if you
                                              want to arrive at an incident and not have a lane of traffic between the vehicle and the incident, you would choose Right side of vehicle (1)
                                              in the United States but Left side of vehicle (2) in the United Kingdom.
                                            * ``Attr_TravelTime``: Specifies the amount of time for cars, in minutes, that will be added to the total travel time of the route between the
                                              incident and the closest facility. The attribute value can be used to model the time spent at the incident. For example, if you are finding
                                              the three closest fire stations from a fire incident, the attribute can store the amount of time spent at the fire incident. This could be
                                              the time it takes for firefighters to hook up their equipment and begin fighting the fire.
                                              The value for this attribute is included in the total travel time for the route and is also displayed in driving directions as service time.
                                              A zero or null value indicates that the incident requires no service time. The default value is 0.
                                            * ``Attr_TruckTravelTime``: Specifies the amount of time for trucks, in minutes, that will be added to the total travel time of the route between
                                              the incident and the closest facility. The attribute value can be used to model the time spent at the incident.
                                              The value for this attribute is included in the total travel time for the route and is also displayed in driving directions as service time. A zero or null value indicates
                                              that the incident requires no service time. The default value is 0.
                                            * ``Attr_WalkTime``: Specifies the amount of time for pedestrians, in minutes, that will be added to the total travel time of the route
                                              between the incident and the closest facility. The attribute value can be used to model the time spent at the incident.
                                              The value for this attribute is included in the total travel time for the route and is also displayed in walking directions as service time.
                                              A zero or null value indicates that the incident requires no service time. The default value is 0.
                                            * ``Attr_Miles``: Specifies the distance in miles that will be added to the total distance of the route between
                                              the incident and the closest facility. Generally the locations of the incidents are not exactly on the streets
                                              but are set back somewhat from the road. This attribute value can be used to model the distance between the
                                              actual incident location and its location on the street if it is important to include that distance in the
                                              total travel distance. The default value is 0.
                                            * ``Attr_Kilometers``: Specifies the distance in kilometers that will added to the total distance of the route
                                              between the incident and the closest facility. Generally the locations of the incidents are not exactly on the
                                              streets but are set back somewhat from the road. This attribute value can be used to model the distance between
                                              the actual incident location and its location on the street if it is important to include that distance in the
                                              total travel distance. The default value is 0.
                                            * ``Cutoff_TravelTime``: Specify the travel time for cars, in minutes, at which to stop searching for facilities
                                              for a given incident. Any incident beyond the cutoff value will not be searched.
                                              If ``Cutoff_TravelTime`` is not set for an incident, the service will use the value specified as the defa``ultCutoff parameter.
                                              The value for the ``Cutoff_TravelTime`` attribute allows the ability to overwrite the ``defaultCutoff`` value on a per incident basis.
                                              The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Cutoff_TruckTravelTime``: Specify the travel time for trucks, in minutes, at which to stop searching for facilities for a
                                              given incident. Any incident beyond the cutoff value will not be searched.
                                              If ``Cutoff_TruckTravelTime`` is not set for an incident, the service will use the value specified as the ``defaultCutoff`` parameter.
                                              The value for the ``Cutoff_TruckTravelTime`` attribute allows the ability to overwrite the ``defaultCutoff`` value on a per incident basis.
                                              The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Cutoff_WalkTime``: Specify the travel time for pedestrians, in minutes, at which to stop searching for facilities for a given incident. Any incident beyond the cutoff value will not be searched.
                                              If Cutoff_WalkTime is not set for an incident, the service will use the value specified as the ``defaultCutoff`` parameter. The value for the Cutoff_WalkTime attribute allows the ability to overwrite the defaultCutoff value on a per incident basis.
                                              The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Cutoff_Miles``: Specify the travel distance in miles at which to stop searching for facilities for a given incident. Any incident beyond the cutoff value will not be searched.
                                              If ``Cutoff_Miles`` is not set for an incident, the service will use the value specified as the ``defaultCutoff`` parameter. The value for the Cutoff_Miles attribute allows the
                                              ability to overwrite the ``defaultCutoff`` value on a per incident basis.
                                              The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Cutoff_Kilometers``: Specify the travel distance in kilometers at which to stop searching for facilities for a given incident.
                                              Any incident beyond the cutoff value will not be searched.
                                              If ``Cutoff_Kilometers`` is not set for an incident, the service will use the value specified as the ``defaultCutoff`` parameter.
                                              The value for the ``Cutoff_Kilometers`` attribute allows the ability to overwrite the ``defaultCutoff`` value on a per incident basis.
                                              The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``TargetFacilityCount``: Specify the number of facilities that need to be found for the incident.
                                              If ``TargetFacilityCount`` is not set for an incident, the service will use the value specified as the
                                              ``defaultTargetFacilityCount`` parameter. The value for the ``TargetFacilityCount`` attribute allows the ability to
                                              overwrite the ``defaultTargetFacilityCount`` value on a per incident basis. The default value for this attribute
                                              is null which causes the service to use the value set for the ``defaultTargetFacilityCount`` parameter. If the
                                              ``TargetFacilityCount`` attribute is set to a value other than null, the ``defaultTargetFacilityCount`` value is
                                              overwritten.
                                            * ``Bearing``: Specify the direction the vehicle or person is moving in. ``Bearing`` is measured clockwise from true
                                              north and must be in degrees. Typically, values are between 0 and 360; however, negative values are interpreted
                                              by subtracting them from 360 degrees.
                                            * ``BearingTol``: Short for bearing tolerance, this field specifies the maximum acceptable difference between the heading
                                              of a vehicle and a tangent line from the point on a street where Network Analyst attempts to locate the vehicle.
                                              The bearing tolerance is used to determine whether the direction in which a vehicle is moving generally aligns with the underlying road.
                                              If they align within the given tolerance, the vehicle is located on that edge; if not, the next nearest eligible edge is evaluated.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    facilities                              Required :class:`~arcgis.features.FeatureSet` . Specify one or more facilities (up to 1,000). These are the locations that are searched for when
                                            finding the closest location. mWhen specifying the facilities, you can set properties for each one, such as its
                                            name or service time, by using attributes. The facilities can be specified with the following attributes:

                                            * ``Name`` - The name of the facility. The name is used in the driving
                                              directions. If the name is not specified, a unique name prefixed
                                              with Location is automatically generated in the output routes and
                                              directions.
                                            * ``ID`` - A unique identifier for the facility. The identifier is included in the output routes (as the
                                              FacilityID field) and the output closest facilities as FacilityID fields. The FacilityID field can be used
                                              to join additional information from the output routes, such as the total travel time or total distance,
                                              to attributes from your facilities. If the ID isn't specified, the service autogenerates a unique identifier
                                              for each facility.
                                            * ``AdditionalTime``-The amount of time spent at the facility, which is added to the total time
                                              of the route. The units for this attribute value are specified by the Measurement Units parameter. The
                                              attribute value is included in the analysis only when the measurement units are time based. The default value is 0.
                                              If you are finding the closest fire stations to fire incidents, ``AdditionalTime`` can store the
                                              time it tends to takes a crew to don the appropriate protective equipment
                                              and exit the fire station.
                                            * ``AdditionalDistance`` - The extra distance traveled at the facility, which is added to the total distance of
                                              the route. The units for this attribute value are specified by the Measurement Units parameter. The attribute
                                              value is included in the analysis only when the measurement units are distance based. The default value is 0.
                                              Generally the location of a facility, such as a fire station, isn't exactly on a street; it is set back somewhat
                                              from the road. ``AdditionalDistance`` can model the distance between the actual facility location and its location on
                                              the street, if it is important to include that distance in the total travel distance.
                                            * ``CurbApproach``:  Specifies the direction a vehicle may arrive at and depart
                                              from the facility. The field value is specified as one of the
                                              following integers (use the numeric code, not the name in parentheses):

                                              =========================  ===============================================================
                                              **Setting**                **Description**
                                              -------------------------  ---------------------------------------------------------------
                                              Either side of vehicle     |either|
                                                                         The vehicle can approach and depart the order in either
                                                                         direction, so a U-turn is allowed at the order. This setting
                                                                         can be chosen if it is possible and desirable for your vehicle
                                                                         to turn around at the order. This decision may depend on the
                                                                         width of the road and the amount of traffic or whether the
                                                                         order has a parking lot where vehicles can pull in and turn
                                                                         around.
                                              -------------------------  ---------------------------------------------------------------
                                              right side of vehicle      |right|
                                                                         When the vehicle approaches and departs the order, the order must
                                                                         be on the right side of the vehicle. A U-turn is prohibited. This is
                                                                         typically used for vehicles like buses that must arrive with the bus
                                                                         stop on the right side.
                                              -------------------------  ---------------------------------------------------------------
                                              left side of vehicle       |left|
                                                                         When the vehicle approaches and departs the order, the curb must
                                                                         be on the left side of the vehicle. A U-turn is prohibited.
                                                                         This is typically used for vehicles like buses that must arrive
                                                                         with the bus stop on the left-hand side.
                                              -------------------------  ---------------------------------------------------------------
                                              No U-Turn                  |turn|
                                                                         When the vehicle approaches the order, the curb can be on either side
                                                                         of the vehicle; however, the vehicle must depart without turning around.
                                              =========================  ===============================================================

                                              The ``CurbApproach`` property was designed to work with both kinds of national driving standards: right-hand
                                              traffic (United States) and left-hand traffic (United Kingdom). First, consider a facility on the left side
                                              of a vehicle. It is always on the left side regardless of whether the vehicle travels on the left or right
                                              half of the road. What may change with national driving standards is your decision to approach a facility
                                              from one of two directions, that is, so it ends up on the right or left side of the vehicle. For example,
                                              if you want to arrive at a facility and not have a lane of traffic between the vehicle and the facility,
                                              you would choose Right side of vehicle (1) in the United States but Left side of vehicle (2) in the United Kingdom.
                                            * ``Attr_TravelTime``: Specifies the amount of time for cars, in minutes, that will be added to the total travel
                                              time of the route between the incident and the closest facility. The attribute value can be used to specify the
                                              turnout time for the facility. For example, when finding the three closest fire stations from a fire incident,
                                              this attribute can be used to store the time it takes a crew to don the appropriate protective equipment and exit
                                              the fire station.
                                              The value for this attribute is included in the total travel time for the route and is also displayed in driving
                                              directions as service time. A zero or null value indicates that the facility requires no service time. The default
                                              value is 0.
                                            * ``Attr_TruckTravelTime``: Specifies the amount of time for trucks, in minutes, that will be added to the total
                                              travel time of the route between the incident and the closest facility.
                                              The value for this attribute is included in the total travel time for the route and is also displayed in driving
                                              directions as service time. A zero or null value indicates that the facility requires no service time. The default value is 0.
                                            * ``Attr_WalkTime``: Specifies the amount of time for pedestrians, in minutes, that will be added to the total travel time
                                              of the route between the incident and the closest facility.
                                              The value for this attribute is included in the total travel time for the route and is also displayed in walking directions
                                              as service time. A zero or null value indicates that the facility requires no service time. The default value is 0.
                                            * ``Attr_Miles``: Specifies the distance in miles that will added to the total distance of the route between the incident and
                                              the closest facility. Generally the locations of the facilities are not exactly on the streets but are set back somewhat from the
                                              road. This attribute value can be used to model the distance between the actual facility location and its location on the street if
                                              it is important to include that distance in the total travel distance. The default value is 0.
                                            * ``Attr_Kilometers``: Specifies the distance in kilometers that will added to the total distance of the route between the incident
                                              and the closest facility. Generally the locations of the facilities are not exactly on the streets but are set back somewhat from the road.
                                              This attribute value can be used to model the distance between the actual facility location and its location on the street if it is important
                                              to include that distance in the total travel distance. The default value is 0.
                                            * ``Cutoff_TravelTime``: Specify the travel time for cars, in minutes, at which to stop searching for facilities for a given incident.
                                              Any incident beyond the cutoff value will not be searched.
                                              If ``Cutoff_TravelTime`` is not set for a facility, the service will use the value specified as the ``defaultCutoff`` parameter.
                                              The value for the ``Cutoff_TravelTime`` attribute allows the ability to overwrite the ``defaultCutoff`` value on a per facility basis.
                                              The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Cutoff_TruckTravelTime``: Specify the travel time for trucks, in minutes, at which to stop searching for facilities for a given incident. Any incident beyond the cutoff value will not be searched.
                                              If ``Cutoff_TruckTravelTime`` is not set for a facility, the service will use the value specified as the ``defaultCutoff`` parameter. The value for the Cutoff_TruckTravelTime attribute allows the ability
                                              to overwrite the ``defaultCutoff`` value on a per facility basis. The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Cutoff_WalkTime``: Specify the travel time for trucks, in minutes, at which to stop searching for facilities for a given incident. Any incident beyond the cutoff value will not be searched.
                                              ``Cutoff_WalkTime`` is not set for a facility, the service will use the value specified as the ``defaultCutoff`` parameter. The value for the Cutoff_WalkTime attribute allows the ability to overwrite the defaultCutoff value on a per facility basis. The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Cutoff_Miles``: Specify the travel distance in miles at which to stop searching for facilities for a given incident. Any incident beyond the cutoff value will not be searched.
                                              If ``Cutoff_Miles`` is not set for a facility, the service will use the value specified as the ``defaultCutoff`` parameter. The value for the Cutoff_Miles attribute allows the ability to overwrite the defaultCutoff value on a per facility basis. The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Cutoff_Kilometers``: Specify the travel distance in kilometers at which to stop searching for facilities for a given incident. Any incident beyond the cutoff value will not be searched.
                                              If Cutoff_Kilometers is not set for a facility, the service will use the value specified as the ``defaultCutoff`` parameter. The value for the Cutoff_Kilometers attribute allows the ability to overwrite the defaultCutoff value on a per facility basis. The default value for this attribute is null which indicates not to use any cutoff.
                                            * ``Bearing``: Specify the direction the vehicle or person is moving in. ``Bearing`` is measured clockwise from true north and must be in degrees.
                                              Typically, values are between 0 and 360; however, negative values are interpreted by subtracting them from 360 degrees.
                                            * ``BearingTol``: Short for bearing tolerance, this field specifies the maximum acceptable difference between the heading of a vehicle and a tangent
                                              line from the point on a street where Network Analyst attempts to locate the vehicle. The bearing tolerance is
                                              used to determine whether the direction in which a vehicle is moving
                                              generally aligns with the underlying road. If they align within the given tolerance, the vehicle is located on
                                              that edge; if not, the next nearest eligible edge is evaluated.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    measurement_units                       Required parameter. Specify the units that should be used to measure and report the total travel time or travel
                                            distance for the output routes. The tool finds the closest facility by measuring the travel time or the travel distance
                                            along streets.
                                            The units you choose for this parameter determine whether the tool will measure driving distance or driving time to find
                                            what is closest. Choose a time unit to measure driving time. To measure driving distance, choose a distance unit. Your
                                            choice also determines in which units the tool will report total driving time or distance in the results.

                                            Choice list:['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles', 'Nautical Miles', 'Seconds', 'Minutes', 'Hours', 'Days']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    analysis_region                         Optional string. Specify the region in which to perform the analysis. If a value is not specified for this parameter, the tool
                                            will automatically calculate the region name based on the location
                                            of the input points. Setting the name of the region is recommended to speed up the
                                            tool execution.

                                            Choice list:['NorthAmerica', 'SouthAmerica', 'Europe', 'MiddleEastAndAfrica', 'India', 'SouthAsia', 'SouthEastAsia', 'Thailand', 'Taiwan', 'Japan', 'Oceania', 'Greece', 'Korea']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    number_of_facilities_to_find            Optional integer. Specify the number of
                                            closest facilities to find per incident. This is useful in
                                            situations, such as a fire, where multiple fire engines may be
                                            required from different fire stations. You can specify, for
                                            example, to find the three nearest fire stations to a fire.
                                            The tool can find up to 100 facilities from each incident.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    cutoff                                  Optional float. Specify the travel time or travel distance value at which
                                            to stop searching for facilities for a given incident. For
                                            instance, while finding the closest hospitals from the site of an
                                            accident, a cutoff value of 15 minutes would mean that the tool
                                            would search for the closest hospital within 15 minutes from the
                                            incident. If the closest hospital is 17 minutes away, no routes
                                            will be returned in the output routes. A cutoff value is especially
                                            useful when searching for multiple facilities.

                                            The units for this parameter is specified by the Measurement Units parameter.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    travel_direction                        Optional string. Specify whether you want to search for the closest
                                            facility as measured from the incident to the facility or from the facility to the incident.

                                            * Facility to Incident - Direction of travel is from
                                              facilities to incidents.

                                            * Incident to Facility - Direction of travel is from
                                              incidents to facilities.

                                            Using one of the parameter values can find different
                                            facilities as the travel time along some streets may vary based on
                                            the travel direction and one-way restrictions. For instance, a
                                            facility may be a 10-minute drive from the incident while traveling
                                            from the incident to the facility, but while traveling from the
                                            facility to the incident, it may be a 15-minute journey because of
                                            different travel time in that direction. If you are setting a value for Time of Day, traffic may also cause the Facility to Incident and Incident to Facility options to return different results.

                                            Fire departments commonly use the Facility to Incident
                                            value for the parameter since they are concerned with the time it
                                            takes to travel from the fire station (facility) to the location of
                                            the emergency (incident). A retail store (facility) is more
                                            concerned with the time it takes the shoppers (incidents) to reach
                                            the store; therefore, stores commonly choose Incident to Facility.

                                            Choice list: ['Incident to Facility', 'Facility to Incident']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    use_hierarchy                           Optional boolean. Specify whether hierarchy should be used when finding the best
                                            route between the facility and the incident.

                                            * Checked (True) - Use hierarchy when finding routes. When
                                              hierarchy is used, the tool prefers higher-order streets (such as
                                              freeways) to lower-order streets (such as local roads), and can be used
                                              to simulate the driver preference of traveling on freeways instead
                                              of local roads even if that means a longer trip. This is especially
                                              true when finding routes to faraway facilities, because drivers on long-distance
                                              trips tend to prefer traveling on freeways where stops, intersections, and turns can be avoided.
                                              Using hierarchy is computationally faster,
                                              especially for long-distance routes, as the tool has to select the
                                              best route from a relatively smaller subset of streets.

                                            * Unchecked (False) - Do not use hierarchy when finding routes. If
                                              hierarchy is not used, the tool considers all the streets and doesn't
                                              prefer higher-order streets when finding the route. This is often
                                              used when finding short-distance routes within a city.

                                            The tool automatically reverts to using hierarchy if the
                                            straight-line distance between facilities and incidents is
                                            greater than 50 miles, even if you have set this parameter to not use hierarchy.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_of_day                             Optional datetime. Specifies the time and date at which the routes should
                                            begin or end. The value is used as the start time or end time for
                                            the route depending on the value for the Time of Day Usage
                                            parameter. If you specify the current date and time as the value
                                            for this parameter, the tool will use live traffic conditions to
                                            find the closest facilities and the total travel time will be based
                                            on traffic conditions.

                                            Specifying a time of day results in more accurate
                                            estimation of travel time between the incident and facility because the
                                            travel times account for the traffic conditions that are applicable
                                            for that date and time.
                                            The Time Zone for Time of Day parameter specifies whether this time and date
                                            refer to UTC or the time zone in which the facility or incident is located.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_of_day_usage                       Optional string. Indicates whether the Time of Day parameter value
                                            represents the arrival or departure time for the routes.

                                            * Start Time - When this option is chosen, the tool finds
                                              the best route considering the Time of Day parameter value as the
                                              departure time from the facility or incident.

                                            * End Time - When this option is chosen, the tool considers
                                              the Time of Day parameter value as the arrival time at the facility
                                              or incident. This option is useful if you want to know what time to
                                              depart from a location so you arrive at the destination at the
                                              time specified in Time of Day.

                                            Choice list: ['Start Time', 'End Time']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    uturn_at_junctions                      Optional string. The U-Turn policy at junctions. Allowing U-turns implies the solver can turn around at a junction and double back on the same street.

                                            Given that junctions represent street intersections and dead ends, different vehicles  may be able to turn around
                                            at some junctions but not at others-it depends on whether the junction represents an intersection or dead end.
                                            To accommodate, the U-turn policy parameter is implicitly specified by how many edges connect to the junction,
                                            which is known as junction valency. The acceptable values for this parameter are listed below; each is followed
                                            by a description of its meaning in terms of junction valency.
                                            Allowed-U-turns are permitted at junctions with any number of connected edges. This is the default value.
                                            Not Allowed-U-turns are prohibited at all junctions, regardless of junction valency. Note, however, that
                                            U-turns are still permitted at network locations even when this setting is chosen; however, you can set the
                                            individual network locations' CurbApproach property to prohibit U-turns there as well.  Allowed only at Dead
                                            Ends-U-turns are prohibited at all junctions, except those that have only one adjacent edge (a dead end).
                                            Allowed only at Intersections and Dead Ends-U-turns are prohibited at junctions where exactly two adjacent edges meet
                                            but are permitted at intersections (junctions with three or more adjacent edges) and dead ends (junctions with exactly
                                            one adjacent edge).  Oftentimes, networks have extraneous junctions in the middle of road segments. This option
                                            prevents vehicles from making U-turns at these locations.

                                            Choice list: ['Allowed', 'Not Allowed', 'Allowed Only at Dead Ends', 'Allowed Only at Intersections and Dead Ends']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    point_barriers                          Optional :class:`~arcgis.features.FeatureSet` . Specify one or more points to act as temporary
                                            restrictions or represent additional time or distance that may be
                                            required to travel on the underlying streets. For example, a point
                                            barrier can be used to represent a fallen tree along a street or
                                            time delay spent at a railroad crossing.

                                            The tool imposes a limit of 250 points that can be added
                                            as barriers.
                                            When specifying the point barriers, you can set properties for each one, such as its name or barrier type,
                                            by using attributes. The point barriers can be specified with the following attributes:

                                            * ``Name``: The name of the barrier.
                                            * ``BarrierType``: Specifies whether the point barrier restricts travel
                                              completely or adds time or distance when it is crossed. The value
                                              for this attribute is specified as one of the following
                                              integers (use the numeric code, not the name in parentheses):

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
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    line_barriers                           Optional :class:`~arcgis.features.FeatureSet` .  Specify one or more lines that prohibit travel anywhere
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
    polygon_barriers                        Optional :class:`~arcgis.features.FeatureSet` . Specify polygons that either completely restrict travel or
                                            proportionately scale the time or distance required to travel on
                                            the streets intersected by the polygons.

                                            The service imposes a limit on the number of streets you
                                            can restrict using the Polygon Barriers parameter. While there is
                                            no limit on the number of polygons you can specify as the polygon
                                            barriers, the combined number of streets intersected by all the
                                            polygons should not exceed 2,000.
                                            When specifying the polygon barriers, you can set properties for each one, such as its name or barrier type,
                                            by using attributes. The polygon barriers can be specified with the following attributes:

                                            * ``Name``: The name of the barrier.
                                            * ``BarrierType``: Specifies whether the barrier restricts travel completely
                                              or scales the time or distance for traveling through it. The field
                                              value is specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Restriction) - Prohibits traveling through any part of the barrier.
                                                The barrier is referred to as a restriction polygon barrier since it
                                                prohibits traveling on streets intersected by the barrier. One use
                                                of this type of barrier is to model floods covering areas of the
                                                street that make traveling on those streets impossible.

                                              * 1 (Scaled Cost) - Scales the time or distance required to travel the
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
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    restrictions                            Optional string. Specify which restrictions should be honored by the tool when finding the best routes between
                                            incidents and facilities. A restriction represents a driving preference or requirement. In most cases, restrictions
                                            cause roads to be prohibited, but they can also cause them to be avoided or preferred. For instance, using an Avoid
                                            Toll Roads restriction will result in a route that will include toll roads only when it is absolutely required to
                                            travel on toll roads in order to visit an incident or a facility. Height Restriction makes it possible to route
                                            around any clearances that are lower than the height of your vehicle. If you are carrying corrosive materials on
                                            your vehicle, using the Any Hazmat Prohibited restriction prevents hauling the materials along roads where it is
                                            marked as illegal to do so. Below is a list of available restrictions and a short description.
                                            Some restrictions require an additional value to be
                                            specified for their desired use. This value needs to be associated
                                            with the restriction name and a specific parameter intended to work
                                            with the restriction. You can identify such restrictions if their
                                            names appear under the AttributeName column in the Attribute
                                            Parameter Values parameter. The ParameterValue field should be
                                            specified in the Attribute Parameter Values parameter for the
                                            restriction to be correctly used when finding traversable roads.
                                            Some restrictions are supported only in certain countries; their availability is stated by region in the list below.
                                            Of the restrictions that have limited availability within a region, you can check whether the restriction is available
                                            in a particular country by looking at the table in the Country List section of the Data coverage for network analysis
                                            services web page. If a country has a value of  Yes in the Logistics Attribute column, the restriction with select
                                            availability in the region is supported in that country. If you specify restriction names that are not available in the
                                            country where your incidents are located, the service ignores the invalid restrictions. The service also ignores
                                            restrictions whose Restriction Usage parameter value is between 0 and 1 (see the Attribute Parameter Value parameter).
                                            It prohibits all restrictions whose Restriction Usage parameter value is greater than 0.
                                            The tool supports the following restrictions:

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

                                            Choice list:['Any Hazmat Prohibited', 'Avoid Carpool Roads', 'Avoid Express Lanes', 'Avoid Ferries', 'Avoid Gates',
                                            'Avoid Limited Access Roads', 'Avoid Private Roads', 'Avoid Roads Unsuitable for Pedestrians', 'Avoid Stairways',
                                            'Avoid Toll Roads', 'Avoid Toll Roads for Trucks', 'Avoid Truck Restricted Roads', 'Avoid Unpaved Roads', 'Axle Count Restriction',
                                            'Driving a Bus', 'Driving a Delivery Vehicle', 'Driving a Taxi', 'Driving a Truck', 'Driving an Automobile', 'Driving an Emergency Vehicle',
                                            'Height Restriction', 'Kingpin to Rear Axle Length Restriction', 'Length Restriction', 'Preferred for Pedestrians', 'Riding a Motorcycle',
                                            'Roads Under Construction Prohibited', 'Semi or Tractor with One or More Trailers Prohibited', 'Single Axle Vehicles Prohibited',
                                            'Tandem Axle Vehicles Prohibited', 'Through Traffic Prohibited', 'Truck with Trailers Restriction', 'Use Preferred Hazmat Routes',
                                            'Use Preferred Truck Routes', 'Walking', 'Weight Restriction', 'Weight per Axle Restriction', 'Width Restriction']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    attribute_parameter_values              Optional :class:`~arcgis.features.FeatureSet` . Specify additional values required by some restrictions, such as the weight of a vehicle for
                                            Weight Restriction. You can also use the attribute parameter to specify whether any restriction prohibits, avoids,
                                            or prefers travel on roads that use the restriction. If the restriction is
                                            meant to avoid or prefer roads, you can further specify the degree
                                            to which they are avoided or preferred using this
                                            parameter. For example, you can choose to never use toll roads, avoid them as much as possible, or even highly prefer them.
                                            The values you provide for this parameter are ignored unless Travel Mode is set to Custom.
                                            If you specify the Attribute Parameter Values parameter from a
                                            feature class, the field names on the feature class must match the fields as described below:

                                            * ``AttributeName``: Lists the name of the restriction.
                                            * ``ParameterName``: Lists the name of the parameter associated with the
                                              restriction. A restriction can have one or more ParameterName field
                                              values based on its intended use.
                                            * ``ParameterValue``: The value for ParameterName used by the tool
                                              when evaluating the restriction.

                                              Attribute Parameter Values is dependent on the
                                              Restrictions parameter. The ParameterValue field is applicable only
                                              if the restriction name is specified as the value for the
                                              Restrictions parameter.

                                              In Attribute Parameter Values, each
                                              restriction (listed as AttributeName) has a ParameterName field
                                              value, Restriction Usage, that specifies whether the restriction
                                              prohibits, avoids, or prefers travel on the roads associated with
                                              the restriction and the degree to which the roads are avoided or
                                              preferred. The Restriction Usage ParameterName can be assigned any of
                                              the following string values or their equivalent numeric values
                                              listed within the parentheses:

                                              * PROHIBITED (-1)- Travel on the roads using the restriction is completely prohibited.
                                              * AVOID_HIGH (5) - It is highly unlikely for the tool to include in the route the roads that are associated with the restriction.
                                              * AVOID_MEDIUM (2) - It is unlikely for the tool to include in the route the roads that are associated with the restriction.
                                              * AVOID_LOW (1.3) - It is somewhat unlikely for the tool to include in the route the roads that are associated with the restriction.
                                              * PREFER_LOW (0.8)  - It is somewhat likely for the tool to include in the route the roads that are associated with the restriction.
                                              * PREFER_MEDIUM (0.5) - It is likely for the tool to include in the route the roads that are associated with the restriction.
                                              * PREFER_HIGH (0.2) - It is highly likely for the tool to include in the route the roads that are associated with the restriction.

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

                                            Choice list:['True Shape', 'Straight Line', 'None']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    route_line_simplification_tolerance     Optional LinearUnit. Specify by how much you want to simplify the geometry of the output lines for
                                            routes and directions. The tool ignores this parameter if the Route Shape parameter isn't set to True Shape.
                                            Simplification maintains critical
                                            points on a route, such as turns at intersections, to define the
                                            essential shape of the route and removes other points. The
                                            simplification distance you specify is the maximum allowable offset
                                            that the simplified line can deviate from the original line.
                                            Simplifying a line reduces the number of vertices that are part of
                                            the route geometry. This improves the tool execution
                                            time.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    populate_directions                     Optional boolean. Specify whether the tool should generate driving directions for
                                            each route.
                                            Checked (True): Indicates that the directions will be generated
                                            and configured based on the values for the Directions Language,
                                            Directions Style Name, and Directions Distance Units
                                            parameters.
                                            Unchecked (False): Directions are not generated, and the tool
                                            returns an empty Directions layer.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    directions_language                     Optional string. Specify the language that should be used when generating
                                            driving directions.

                                            This parameter is used only when the Populate
                                            Directions parameter is checked, or set to True.

                                            The parameter value can be
                                            specified using one of the following two- or five-character language codes:

                                            Choice list: ['ar-Arabic', 'cs-Czech', 'de-German', 'el-Greek', 'en-English', 'es-Spanish', 'et-Estonian', 'fr-French', 'he-Hebrew', 'it-Italian',
                                            'ja-Japanese', 'ko-Korean', 'lt-Lithuanian', 'lv-Latvian', 'nl-Dutch', 'pl-Polish', 'pt-BR-Brazilian', 'Portuguese', 'pt-PT-European', 'Portuguese', 'ru-Russian',
                                            'sv-Swedish', 'th-Thai', 'tr-Turkish', 'zh-CN-Simplified', 'Chinese']

                                            If an unsupported language code is specified, the tool
                                            returns the directions using the default language,
                                            English.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    directions_distance_units               Optional string. Specify the units for displaying travel distance in the
                                            driving directions. This parameter is used only when the Populate
                                            Directions parameter is checked, or set to True.
                                            Miles Kilometers Meters Feet Yards NauticalMiles

                                            Choice list: ['Miles', 'Kilometers', 'Meters', 'Feet', 'Yards', 'NauticalMiles']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    directions_style_name                   Optional string. Specify the name of the formatting style for the
                                            directions. This parameter is used only when the Populate
                                            Directions parameter is checked, or set to True. The parameter can be specified
                                            using the following values:

                                            ``NA Desktop``: Generates turn-by-turn directions suitable for printing.

                                            ``NA Navigation``: Generates turn-by-turn directions designed for an in-vehicle navigation device.

                                            Choice list: ['NA Desktop', 'NA Navigation']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_zone_for_time_of_day               Optional string. Specifies the time zone of the Time of Day parameter.
                                            Geographically Local: The Time of Day parameter refers to the time zone in which the facilities or incidents are
                                            located.  If Time of Day Usage is set to Start Time and Travel Direction is Facility to Incident, this is the time
                                            zone of the facilities.  If Time of Day Usage is set to Start Time and Travel Direction is Incident to Facility,
                                            this is the time zone of the incidents.  If Time of Day Usage is set to End Time and Travel Direction is Facility
                                            to Incident, this is the time zone of the incidents.  If Time of Day Usage is set to End Time and Travel Direction
                                            is Incident to Facility, this is the time zone of the facilities.  UTC: The Time of Day parameter refers to
                                            Coordinated Universal Time (UTC). Choose this option if you want to find what's nearest for a specific time, such as
                                            now, but aren't certain in which time zone the facilities or incidents will be located.
                                            Irrespective of the Time Zone for Time of Day setting, if your facilities
                                            and incidents are in multiple time zones, the following rules are
                                            enforced by the tool:

                                            All incidents must be in the same time zone when: Specifying a start time and traveling from incident to facility. Specifying an end time and traveling from facility to incident.

                                            All facilities must be in the same time zone when: Specifying a start time and traveling from facility to incident. Specifying an end time and traveling from incident to facility.

                                            Choice list:['Geographically Local', 'UTC']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    travel_mode                             Optional string. Specify the mode of transportation to model in the analysis. Travel modes are managed in ArcGIS
                                            Online and can be configured by the administrator of your organization to better reflect your organization's
                                            workflows. You need to specify the name of a travel mode supported by your organization.

                                            To get a list of supported travel mode names, run the GetTravelModes tool from the Utilities toolbox available under the same GIS Server connection
                                            you used to access the tool. The GetTravelModes tool adds a table, Supported Travel Modes, to the application. Any value in the Travel Mode Name field from the
                                            Supported Travel Modes table can be specified as input. You can also specify the value from Travel Mode Settings field as input. This speeds up the tool execution as the
                                            tool does not have to lookup the settings based on the travel mode name.

                                            The default value, Custom, allows you to configure your own travel mode using the custom travel mode parameters (UTurn at Junctions, Use Hierarchy, Restrictions, Attribute Parameter Values, and Impedance).
                                            The default values of the custom travel mode parameters model travelling by car. You may want to choose Custom and set the custom travel mode parameters listed above to model a pedestrian with a fast walking speed
                                            or a truck with a given height, weight, and cargo of certain hazardous materials. You may choose to do this to try out different settings to get desired analysis results.
                                            Once you have identified the analysis settings, you should work with your organization's administrator and save these settings as part of new or existing travel mode so that
                                            everyone in your organization can rerun the analysis with the same settings.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    impedance                               Optional string. Specify the impedance, which is a value that represents the effort or cost of traveling along
                                            road segments or on other parts of the transportation network.
                                            Travel distance is an impedance; the length of a road in kilometers can be thought of as impedance.
                                            Travel distance in this sense is the same for all modes-a kilometer for a pedestrian is also a kilometer for a car.
                                            (What may change is the pathways on which the different modes are allowed to travel, which affects distance between points,
                                            and this is modeled by travel mode settings.) Travel time can also be an impedance; a car may take one minute to travel a mile
                                            along an empty road. Travel times can vary by travel mode-a pedestrian may take more than 20  minutes to walk the same mile,
                                            so it is important to choose the right impedance for the travel mode you are modeling.  Choose from the following impedance values:
                                            Drive Time-Models travel times for a car. These travel times are dynamic and fluctuate according to traffic flows in areas where
                                            traffic data is available. This is the default value. Truck Time-Models travel times for a truck.  These travel times are static for
                                            each road and don't fluctuate with traffic. Walk Time-Models travel times for a pedestrian. Travel Distance-Stores  length measurements
                                            along roads and paths. To model walk distance, choose this option and ensure Walking is  set in the Restriction parameter. Similarly,
                                            to model drive or truck distance, choose Travel Distance here and set the appropriate restrictions so your vehicle travels only on
                                            roads where it is permitted to do so. The value you provide for this parameter is ignored unless Travel Mode is set to Custom, which
                                            is the default value. If you choose Drive Time, Truck Time, or Walk Time, the Measurement Units parameter must be set to a time-based
                                            value; if you choose Travel Distance for Impedance, Measurement Units must be distance-based.

                                            Choice list:['Drive Time', 'Truck Time', 'Walk Time', 'Travel Distance']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    gis                                     Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    future                                  Optional boolean. If True, a future object will be returned and the process
                                            will not wait for the task to complete. The default is False, which means wait for results.
    ======================================  ==========================================================================================================================================

    :return: the following as a named tuple:

    * output_routes - Output Routes as a FeatureSet
    * output_directions - Output Directions as a FeatureSet
    * solve_succeeded - Solve Succeeded as a bool
    * output_closest_facilities - Output Closest Facilities as a FeatureSet


    Click `FindClosestFacilities <https://developers.arcgis.com/rest/network/api-reference/closest-facility-asynchronous-service.htm>`_ for additional help.

    """
    if gis is None:
        gis = arcgis.env.active_gis
    url = gis.properties.helperServices.asyncClosestFacility.url[
        : -len("/FindClosestFacilities")
    ]
    url = _validate_url(url, gis)
    tbx = _create_toolbox(url, gis=gis)
    defaults = dict(
        zip(
            tbx.find_closest_facilities.__annotations__.keys(),
            tbx.find_closest_facilities.__defaults__,
        )
    )
    if restrictions is None:
        restrictions = defaults["restrictions"]
    if incidents is None:
        incidents = defaults["incidents"]

    if facilities is None:
        facilities = defaults["facilities"]

    if point_barriers is None:
        point_barriers = defaults["point_barriers"]

    if line_barriers is None:
        line_barriers = defaults["line_barriers"]

    if polygon_barriers is None:
        polygon_barriers = defaults["polygon_barriers"]

    if attribute_parameter_values is None:
        attribute_parameter_values = defaults["attribute_parameter_values"]

    if route_line_simplification_tolerance is None:
        route_line_simplification_tolerance = defaults[
            "route_line_simplification_tolerance"
        ]

    from arcgis._impl.common._utils import inspect_function_inputs

    params = {
        "incidents": incidents,
        "facilities": facilities,
        "measurement_units": measurement_units,
        "analysis_region": analysis_region,
        "number_of_facilities_to_find": number_of_facilities_to_find,
        "cutoff": cutoff,
        "travel_direction": travel_direction,
        "use_hierarchy": use_hierarchy,
        "time_of_day": time_of_day,
        "time_of_day_usage": time_of_day_usage,
        "uturn_at_junctions": uturn_at_junctions,
        "point_barriers": point_barriers,
        "line_barriers": line_barriers,
        "polygon_barriers": polygon_barriers,
        "restrictions": restrictions,
        "attribute_parameter_values": attribute_parameter_values,
        "route_shape": route_shape,
        "route_line_simplification_tolerance": route_line_simplification_tolerance,
        "populate_directions": populate_directions,
        "directions_language": directions_language,
        "directions_distance_units": directions_distance_units,
        "directions_style_name": directions_style_name,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "travel_mode": travel_mode,
        "impedance": impedance,
        "save_output_network_analysis_layer": save_output_network_analysis_layer,
        "overrides": overrides,
        "save_route_data": save_route_data,
        "time_impedance": time_impedance,
        "distance_impedance": distance_impedance,
        "output_format": output_format,
        "gis": gis,
        "future": True,
    }
    params = inspect_function_inputs(tbx.find_closest_facilities, **params)
    job = tbx.find_closest_facilities(**params)
    if future:
        return job
    return job.result()


find_closest_facilities.__annotations__ = {
    "incidents": FeatureSet,
    "facilities": FeatureSet,
    "measurement_units": str,
    "analysis_region": str,
    "number_of_facilities_to_find": int,
    "cutoff": float,
    "travel_direction": str,
    "use_hierarchy": bool,
    "time_of_day": datetime,
    "time_of_day_usage": str,
    "uturn_at_junctions": str,
    "point_barriers": FeatureSet,
    "line_barriers": FeatureSet,
    "polygon_barriers": FeatureSet,
    "restrictions": str,
    "attribute_parameter_values": FeatureSet,
    "route_shape": str,
    "route_line_simplification_tolerance": LinearUnit,
    "populate_directions": bool,
    "directions_language": str,
    "directions_distance_units": str,
    "directions_style_name": str,
    "time_zone_for_time_of_day": str,
    "travel_mode": str,
    "impedance": str,
    "return": tuple,
}
