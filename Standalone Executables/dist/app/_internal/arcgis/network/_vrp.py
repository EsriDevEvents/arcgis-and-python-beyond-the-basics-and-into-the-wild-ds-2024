import logging as _logging
import json
from typing import Optional
import arcgis
from datetime import datetime
from arcgis.features import FeatureSet
from arcgis.gis import GIS
from arcgis.geoprocessing import LinearUnit
from arcgis._impl.common._utils import _validate_url
from arcgis.network import _utils
from ._routing_utils import _create_toolbox

_log = _logging.getLogger(__name__)

_url = "https://logistics.arcgis.com/arcgis/rest/services/World/VehicleRoutingProblem/GPServer"
_use_async = True


default_orders = {
    "fields": [
        {"alias": "ObjectID", "name": "ObjectID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {"alias": "ServiceTime", "name": "ServiceTime", "type": "esriFieldTypeDouble"},
        {
            "alias": "TimeWindowStart1",
            "name": "TimeWindowStart1",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "TimeWindowEnd1",
            "name": "TimeWindowEnd1",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "TimeWindowStart2",
            "name": "TimeWindowStart2",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "TimeWindowEnd2",
            "name": "TimeWindowEnd2",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "MaxViolationTime1",
            "name": "MaxViolationTime1",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "MaxViolationTime2",
            "name": "MaxViolationTime2",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "InboundArriveTime",
            "name": "InboundArriveTime",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "OutboundDepartTime",
            "name": "OutboundDepartTime",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "DeliveryQuantities",
            "name": "DeliveryQuantities",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {
            "alias": "PickupQuantities",
            "name": "PickupQuantities",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {"alias": "Revenue", "name": "Revenue", "type": "esriFieldTypeDouble"},
        {
            "alias": "SpecialtyNames",
            "name": "SpecialtyNames",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {
            "alias": "AssignmentRule",
            "name": "AssignmentRule",
            "type": "esriFieldTypeInteger",
        },
        {
            "alias": "RouteName",
            "name": "RouteName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {"alias": "Sequence", "name": "Sequence", "type": "esriFieldTypeInteger"},
        {
            "alias": "CurbApproach",
            "name": "CurbApproach",
            "type": "esriFieldTypeInteger",
        },
    ],
    "geometryType": "esriGeometryPoint",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}


default_depots = {
    "fields": [
        {"alias": "ObjectID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {
            "alias": "TimeWindowStart1",
            "name": "TimeWindowStart1",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "TimeWindowEnd1",
            "name": "TimeWindowEnd1",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "TimeWindowStart2",
            "name": "TimeWindowStart2",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "TimeWindowEnd2",
            "name": "TimeWindowEnd2",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "CurbApproach",
            "name": "CurbApproach",
            "type": "esriFieldTypeInteger",
        },
        {"alias": "Bearing", "name": "Bearing", "type": "esriFieldTypeDouble"},
        {"alias": "BearingTol", "name": "BearingTol", "type": "esriFieldTypeDouble"},
        {"alias": "NavLatency", "name": "NavLatency", "type": "esriFieldTypeDouble"},
    ],
    "geometryType": "esriGeometryPoint",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}
default_routes = {
    "fields": [
        {"alias": "ObjectID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {
            "alias": "StartDepotName",
            "name": "StartDepotName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {
            "alias": "EndDepotName",
            "name": "EndDepotName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {
            "alias": "StartDepotServiceTime",
            "name": "StartDepotServiceTime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "EndDepotServiceTime",
            "name": "EndDepotServiceTime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "EarliestStartTime",
            "name": "EarliestStartTime",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "LatestStartTime",
            "name": "LatestStartTime",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "ArriveDepartDelay",
            "name": "ArriveDepartDelay",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "Capacities",
            "name": "Capacities",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {"alias": "FixedCost", "name": "FixedCost", "type": "esriFieldTypeDouble"},
        {
            "alias": "CostPerUnitTime",
            "name": "CostPerUnitTime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "CostPerUnitDistance",
            "name": "CostPerUnitDistance",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "OvertimeStartTime",
            "name": "OvertimeStartTime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "CostPerUnitOvertime",
            "name": "CostPerUnitOvertime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "MaxOrderCount",
            "name": "MaxOrderCount",
            "type": "esriFieldTypeInteger",
        },
        {
            "alias": "MaxTotalTime",
            "name": "MaxTotalTime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "MaxTotalTravelTime",
            "name": "MaxTotalTravelTime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "MaxTotalDistance",
            "name": "MaxTotalDistance",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "SpecialtyNames",
            "name": "SpecialtyNames",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {
            "alias": "AssignmentRule",
            "name": "AssignmentRule",
            "type": "esriFieldTypeInteger",
        },
    ],
    "features": [],
    "displayFieldName": "",
    "exceededTransferLimit": False,
}

default_breaks = {
    "fields": [
        {"alias": "ObjectID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {
            "alias": "RouteName",
            "name": "RouteName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {"alias": "Precedence", "name": "Precedence", "type": "esriFieldTypeInteger"},
        {"alias": "ServiceTime", "name": "ServiceTime", "type": "esriFieldTypeDouble"},
        {
            "alias": "TimeWindowStart",
            "name": "TimeWindowStart",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "TimeWindowEnd",
            "name": "TimeWindowEnd",
            "type": "esriFieldTypeDate",
            "length": 8,
        },
        {
            "alias": "MaxViolationTime",
            "name": "MaxViolationTime",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "MaxTravelTimeBetweenBreaks",
            "name": "MaxTravelTimeBetweenBreaks",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "MaxCumulWorkTime",
            "name": "MaxCumulWorkTime",
            "type": "esriFieldTypeDouble",
        },
        {"alias": "IsPaid", "name": "IsPaid", "type": "esriFieldTypeInteger"},
        {"alias": "Sequence", "name": "Sequence", "type": "esriFieldTypeInteger"},
    ],
    "features": [],
    "displayFieldName": "",
    "exceededTransferLimit": False,
}
default_route_zones = {
    "fields": [
        {"alias": "ObjectID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {
            "alias": "RouteName",
            "name": "RouteName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {"alias": "IsHardZone", "name": "IsHardZone", "type": "esriFieldTypeInteger"},
        {
            "alias": "Shape_Length",
            "name": "Shape_Length",
            "type": "esriFieldTypeDouble",
        },
        {"alias": "Shape_Area", "name": "Shape_Area", "type": "esriFieldTypeDouble"},
    ],
    "geometryType": "esriGeometryPolygon",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}

default_route_renewals = {
    "fields": [
        {"alias": "ObjectID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {
            "alias": "RouteName",
            "name": "RouteName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {
            "alias": "DepotName",
            "name": "DepotName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {"alias": "ServiceTime", "name": "ServiceTime", "type": "esriFieldTypeDouble"},
        {
            "alias": "Sequences",
            "name": "Sequences",
            "type": "esriFieldTypeString",
            "length": 128,
        },
    ],
    "features": [],
    "displayFieldName": "",
    "exceededTransferLimit": False,
}

default_order_pairs = {
    "fields": [
        {"alias": "ObjectID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {
            "alias": "FirstOrderName",
            "name": "FirstOrderName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {
            "alias": "SecondOrderName",
            "name": "SecondOrderName",
            "type": "esriFieldTypeString",
            "length": 128,
        },
        {
            "alias": "MaxTransitTime",
            "name": "MaxTransitTime",
            "type": "esriFieldTypeDouble",
        },
    ],
    "features": [],
    "displayFieldName": "",
    "exceededTransferLimit": False,
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
        {"alias": "ObjectID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {"alias": "BarrierType", "name": "BarrierType", "type": "esriFieldTypeInteger"},
        {"alias": "Scaled_Time", "name": "Scaled_Time", "type": "esriFieldTypeDouble"},
        {
            "alias": "Scaled_Distance",
            "name": "Scaled_Distance",
            "type": "esriFieldTypeDouble",
        },
        {
            "alias": "Shape_Length",
            "name": "Shape_Length",
            "type": "esriFieldTypeDouble",
        },
        {"alias": "Shape_Area", "name": "Shape_Area", "type": "esriFieldTypeDouble"},
    ],
    "geometryType": "esriGeometryPolygon",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}

default_restrictions = """['Avoid Unpaved Roads', 'Avoid Private Roads', 'Driving an Automobile', 'Through Traffic Prohibited', 'Roads Under Construction Prohibited', 'Avoid Gates', 'Avoid Express Lanes', 'Avoid Carpool Roads']"""

default_param_values = {
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


def edit_vehicle_routing_problem(
    orders: FeatureSet,
    depots: FeatureSet,
    routes: FeatureSet,
    breaks: Optional[FeatureSet] = None,
    time_units: Optional[str] = None,
    distance_units: Optional[str] = None,
    analysis_region: Optional[str] = None,
    default_date: Optional[datetime] = None,
    uturn_policy: Optional[str] = None,
    time_window_factor: Optional[str] = None,
    spatially_cluster_routes: bool = True,
    route_zones: Optional[FeatureSet] = None,
    route_renewals: Optional[FeatureSet] = None,
    order_pairs: Optional[FeatureSet] = None,
    excess_transit_factor: Optional[str] = None,
    point_barriers: Optional[FeatureSet] = None,
    line_barriers: Optional[FeatureSet] = None,
    polygon_barriers: Optional[FeatureSet] = None,
    use_hierarchy_in_analysis: bool = True,
    restrictions: Optional[str] = None,
    attribute_parameter_values: Optional[dict] = None,
    populate_route_lines: bool = True,
    route_line_simplification_tolerance: Optional[float] = None,
    populate_directions: bool = False,
    directions_language: Optional[str] = None,
    directions_style_name: Optional[str] = None,
    travel_mode: str = "Custom",
    impedance: Optional[str] = None,
    time_zone_usage_for_time_fields: Optional[str] = None,
    save_output_layer: bool = False,
    overrides: Optional[dict] = None,
    save_route_data: Optional[bool] = None,
    time_impedance: Optional[str] = None,
    distance_impedance: Optional[str] = None,
    populate_stop_shapes: bool = False,
    output_format: Optional[str] = None,
    gis: Optional[GIS] = None,
    ignore_invalid_order_locations: bool = False,
):
    """
    This ArcGIS Online service solves a vehicle routing problem (VRP) to find the best routes for a
    fleet of vehicles. It is similar to `solve_vehicle_routing_problem`, but `edit_vehicle_routing_problem`
    is designed to make a few, small edits to the results of a VRP and re-solve or solve a small VRP
    analysis of only two routes.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    orders                                   Required FeatureSet. Specify one or more orders (up to 2,000).
                                             These are the locations that the routes of the vehicle routing
                                             problem (VRP) analysis should visit. An order can represent a
                                             delivery (for example, furniture delivery), a pickup (such as an
                                             airport shuttle bus picking up a passenger), or some type of service
                                             or inspection (a tree trimming job or building inspection, for
                                             instance).
    ------------------------------------     --------------------------------------------------------------------
    depots                                   Required FeatureSet. These represent the location where the routes
                                             will start and end at for the routes.
    ------------------------------------     --------------------------------------------------------------------
    routes                                   Required FeatureSet. Specify one or more routes (up to 100). A route
                                             specifies vehicle and driver characteristics; after solving, it also
                                             represents the path between depots and orders.

                                             A route can have start and end depot service times, a fixed or
                                             flexible starting time, time-based operating costs, distance-based
                                             operating costs, multiple capacities, various constraints on a
                                             driver's workday, and so on. When specifying the routes, you can set
                                             properties for each one by using attributes.
    ------------------------------------     --------------------------------------------------------------------
    breaks                                   Optional :class:`~arcgis.features.FeatureSet` . These are the rest periods, or breaks, for the
                                             routes in a given vehicle routing problem. A break is associated with
                                             exactly one route, and it can be taken after completing an order,
                                             while en-route to an order, or prior to servicing an order. It has a
                                             start time and a duration, for which the driver may or may not be
                                             paid.
                                             There are three options for establishing when a break begins: using
                                             a time window, a maximum travel time, or a maximum work time.
    ------------------------------------     --------------------------------------------------------------------
    time_units                               Optional String. The time units for all time-based field values in
                                             the analysis.  Allowed values: `Seconds`, `Minutes`, `Hours`, or `Days`
    ------------------------------------     --------------------------------------------------------------------
    distance_units                           Optional String. The distance units for all distance-based field
                                             values in the analysis. Allowed values: `Meters`,`Kilometers`,`Feet`,`Yards`,`Miles`,`NauticalMiles`
    ------------------------------------     --------------------------------------------------------------------
    analysis_region                          Optional String. Specify the region in which to perform the analysis.
                                             If a value is not specified for this parameter, the tool will
                                             automatically calculate the region name based on the location of the
                                             input points.
    ------------------------------------     --------------------------------------------------------------------
    default_date                             Optional Datetime. The default date for time field values that
                                             specify a time of day without including a date.
    ------------------------------------     --------------------------------------------------------------------
    uturn_policy                             Optional String. Use this parameter to restrict or permit the service
                                             area to make U-turns at junctions.

                                             The parameter can have the following values:

                                             - `ALLOW_UTURNS-U` - turns are permitted everywhere. Allowing
                                               U-turns implies that the vehicle can turn around at any junction and
                                               double back on the same street. This is the default value.

                                             - `NO_UTURNS-U` - turns are prohibited at all junctions: pseudo-junctions, intersections, and dead-ends.
                                               Note, however, that U-turns may be permitted even when this option is chosen.

                                             - `ALLOW_DEAD_ENDS_ONLY-U` - turns are prohibited at all junctions, except those that have only one connected street feature (a dead end).

                                             - `ALLOW_DEAD_ENDS_AND_INTERSECTIONS_ONLY-U` - turns are prohibited at pseudo-junctions where exactly two adjacent streets meet, but U-turns are permitted
                                               at intersections and dead ends. This prevents turning around in the middle of the road where one length of road happened to be digitized as two street features.
                                               The value you provide for this parameter is ignored unless Travel Mode is set to Custom, which is the default value.

    ------------------------------------     --------------------------------------------------------------------
    time_window_factor                       Optional String. Rates the importance of honoring time windows.
                                             Allowed values: `High`, `Medium` or `Low`.
    ------------------------------------     --------------------------------------------------------------------
    spatially_cluster_routes                 Optional Boolean. If true, the dynamic seed points are automatically
                                             created for all routes and the orders assigned to an individual
                                             route are spatially clustered. This will reduce route intersections.
                                             When using `route_zones` set this parameter to `False`.
    ------------------------------------     --------------------------------------------------------------------
    route_zones                              Optional :class:`~arcgis.features.FeatureSet` . Delineates work territories for given routes. A
                                             route zone is a polygon feature and is used to constrain routes to
                                             servicing only those orders that fall within or near the specified
                                             area.
    ------------------------------------     --------------------------------------------------------------------
    route_renewals                           Optional :class:`~arcgis.features.FeatureSet` . Specifies the intermediate depots that routes can visit to
                                             reload or unload the cargo they are delivering or picking up.
    ------------------------------------     --------------------------------------------------------------------
    order_pairs                              Optional :class:`~arcgis.features.FeatureSet` . This parameter pairs pickup and delivery orders
                                             so they are serviced by the same route.
    ------------------------------------     --------------------------------------------------------------------
    excess_transit_factor                    Optional String.  Rates the importance of reducing excess transit time of
                                             order pairs. Excess transit time is the amount of time exceeding the
                                             time required to travel directly between the paired orders. Excess
                                             time can be caused by driver breaks or travel to intermediate orders
                                             and depots. Allowed values: `High`, `Medium` or `Low`.
    ------------------------------------     --------------------------------------------------------------------
    point_barriers                           Optional :class:`~arcgis.features.FeatureSet` . Specify points that either completely restrict travel or
                                             proportionately scale the time or distance required to travel on the streets intersected
                                             by the polygons.
    ------------------------------------     --------------------------------------------------------------------
    line_barriers                            Optional :class:`~arcgis.features.FeatureSet` . Specify polylines that either completely restrict travel or
                                             proportionately scale the time or distance required to travel on the streets intersected
                                             by the polygons.
    ------------------------------------     --------------------------------------------------------------------
    polygon_barriers                         Optional :class:`~arcgis.features.FeatureSet` . Specify polygons that either completely restrict travel or
                                             proportionately scale the time or distance required to travel on the streets intersected
                                             by the polygons.
    ------------------------------------     --------------------------------------------------------------------
    use_hierarchy_in_analysis                Optional Boolean. Specify whether hierarchy should be used when finding the best
                                             routes. True means use hierarchy when finding routes.
    ------------------------------------     --------------------------------------------------------------------
    restrictions                             Optional String. Specify which restrictions should be honored by the tool when finding
                                             the best routes. A restriction represents a driving preference or requirement.
    ------------------------------------     --------------------------------------------------------------------
    attribute_parameter_values               Optional Dict.  Specify additional values required by some restrictions, such as the weight
                                             of a vehicle for Weight Restriction. You can also use the attribute parameter to specify
                                             whether any restriction prohibits, avoids, or prefers travel on roads that use the
                                             restriction.
    ------------------------------------     --------------------------------------------------------------------
    populate_route_lines                     Optional Boolean. Specify if the output route line should be generated.
    ------------------------------------     --------------------------------------------------------------------
    route_line_simplification_tolerance      Optional Float. Specify by how much you want to simplify the geometry of the output lines
                                             for routes and directions.The value you provide for this parameter is ignored unless Travel
                                             Mode is set to Custom, which is the default value.The tool also ignores this parameter if the
                                             populate_route_lines parameter is false. Simplification maintains critical points on a route,
                                             such as turns at intersections, to define the essential shape of the route and removes other
                                             points.
    ------------------------------------     --------------------------------------------------------------------
    populate_directions                      Optional Boolean.  If True, the directions will be returned.
    ------------------------------------     --------------------------------------------------------------------
    directions_language                      Optional String. Determines the output language of the directions.
    ------------------------------------     --------------------------------------------------------------------
    directions_style_name                    Optional String. Determines the style of the directions.
    ------------------------------------     --------------------------------------------------------------------
    travel_mode                              Optional String. Specify the mode of transportation to model in the analysis.
    ------------------------------------     --------------------------------------------------------------------
    impedance                                Optional String. Specify the impedance, which is a value that represents the effort or cost
                                             of traveling along road segments or on other parts of the transportation network.
    ------------------------------------     --------------------------------------------------------------------
    time_zone_usage_for_time_fields          Optional String. Specifies the time zone for the input date-time fields supported by the tool. The default is GEO_LOCAL.
    ------------------------------------     --------------------------------------------------------------------
    save_output_layer                        Optional Boolean. Specify if the tool should save the analysis settings as a network analysis
                                             layer file. True means save the data and False means do not.
    ------------------------------------     --------------------------------------------------------------------
    overrides                                Optional Dict. Specify additional settings that can influence the behavior of the solver when
                                             finding solutions for the network analysis problems.
    ------------------------------------     --------------------------------------------------------------------
    save_route_data                          Optional Boolean. Choose whether the output includes a zip file that contains a file geodatabase
                                             holding the inputs and outputs of the analysis in a format that can be used to share route layers
                                             with ArcGIS Online or Portal for ArcGIS. True - Save the route data as a zip file. The
                                             file is downloaded in a temporary directory on your machine. False - Do not save the route data.
                                             This is the default.

    ------------------------------------     --------------------------------------------------------------------
    time_impedance                           Optional String. Specify the time-based impedance, which is a value that represents the travel time
                                             along road segments or on other parts of the transportation network. If the impedance for the travel
                                             mode, as specified using the impedance parameter, is time-based, the value for time_impedance and
                                             impedance parameters should be identical. Otherwise the service will return an error.
    ------------------------------------     --------------------------------------------------------------------
    distance_impedance                       Optional String. If the impedance for the travel mode, as specified using the impedance parameter, is
                                             distance-based, the value for distance_impedance and impedance parameters should be identical. Otherwise
                                             the service will return an error. Specify the distance-based impedance, which is a value that represents
                                             the travel distance along road segments or on other parts of the transportation network.
    ------------------------------------     --------------------------------------------------------------------
    populate_stop_shape                      Optional Boolean. Specify if the tool should create the shapes for the output assigned and unassigned stops.
    ------------------------------------     --------------------------------------------------------------------
    output_format                            Optional String. Specify the format in which the output features are created. Choose from the following formats:

                                             - Feature Set: The output features are returned as feature classes and tables. This is the default.
                                             - JSON File: The output features are returned as a compressed file containing the JSON representation of the outputs.
                                               When this option is specified, the output is a single file (with a .zip extension) that contains one or more JSON
                                               files (with a .json extension) for each of the outputs created by the service.
                                             - GeoJSON File: The output features are returned as a compressed file containing the GeoJSON representation of the
                                               outputs. When this option is specified, the output is a single file (with a .zip extension) that contains one or more
                                               GeoJSON files (with a .geojson extension) for each of the outputs created by the service.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` . If provided this connection is used to perform the operation.
    ------------------------------------     --------------------------------------------------------------------
    ignore_invalid_order_locations           Specifies whether invalid orders will be ignored when solving the vehicle routing problem.

                                             ``True`` - The solve operation will ignore any invalid orders and return a solution, given it didn't encounter any other errors. If you need to generate routes and deliver them to drivers immediately, you may be able to ignore invalid orders, solve, and distribute the routes to your drivers. Next, resolve any invalid orders from the last solve and include them in the VRP analysis for the next workday or work shift.

                                             ``False`` - The solve operation will fail when any invalid orders are encountered. An invalid order is an order that the VRP solver can't reach. An order may be unreachable for a variety of reasons, including if it's located on a prohibited network element, it isn't located on the network at all, or it's located on a disconnected portion of the network.
    ====================================     ====================================================================

    :return: Named Tuple

    """
    if travel_mode is None:
        travel_mode = "Custom"
    if gis is None:
        gis = arcgis.env.active_gis
    url = gis.properties.helperServices.syncVRP.url[
        : -len("/EditVehicleRoutingProblem")
    ]
    url = _validate_url(url, gis)
    tbx = _create_toolbox(url, gis=gis)
    defaults = dict(
        zip(
            tbx.edit_vehicle_routing_problem.__annotations__.keys(),
            tbx.edit_vehicle_routing_problem.__defaults__,
        )
    )
    if isinstance(travel_mode, str):
        travel_mode = _utils.find_travel_mode(gis=gis, travel_mode=travel_mode)
    elif isinstance(travel_mode, dict):
        defaults["travel_mode"] = travel_mode
    else:
        travel_mode = _utils.find_travel_mode(
            gis=gis, travel_mode=_utils.default_travel_mode(gis=gis)
        )
    if breaks is None:
        breaks = defaults["breaks"]
    if time_units is None:
        time_units = defaults["time_units"]
    if distance_units is None:
        distance_units = defaults["distance_units"]
    if analysis_region is None:
        analysis_region = defaults["analysis_region"]
    if default_date is None:
        default_date = defaults["default_date"]
    if uturn_policy is None:
        uturn_policy = defaults["uturn_policy"]
    if time_window_factor is None:
        time_window_factor = defaults["time_window_factor"]
    if spatially_cluster_routes is None:
        spatially_cluster_routes = defaults["spatially_cluster_routes"]
    if route_zones is None:
        route_zones = defaults["route_zones"]
    if route_renewals is None:
        route_renewals = defaults["route_renewals"]
    if order_pairs is None:
        order_pairs = defaults["order_pairs"]
    if excess_transit_factor is None:
        excess_transit_factor = defaults["excess_transit_factor"]
    if point_barriers is None:
        point_barriers = defaults["point_barriers"]
    if line_barriers is None:
        line_barriers = defaults["line_barriers"]
    if polygon_barriers is None:
        polygon_barriers = defaults["polygon_barriers"]
    if use_hierarchy_in_analysis is None:
        use_hierarchy_in_analysis = defaults["use_hierarchy_in_analysis"]
    if restrictions is None:
        restrictions = defaults["restrictions"]
    if attribute_parameter_values is None:
        attribute_parameter_values = defaults["attribute_parameter_values"]
    if populate_route_lines is None:
        populate_route_lines = defaults["populate_route_lines"]
    if route_line_simplification_tolerance is None:
        route_line_simplification_tolerance = defaults[
            "route_line_simplification_tolerance"
        ]
    if populate_directions is None:
        populate_directions = defaults["populate_directions"]
    if directions_language is None:
        directions_language = defaults["directions_language"]
    if directions_style_name is None:
        directions_style_name = defaults["directions_style_name"]
    if travel_mode is None:
        travel_mode = defaults["travel_mode"]
    if impedance is None:
        impedance = defaults["impedance"]
    if time_zone_usage_for_time_fields is None:
        time_zone_usage_for_time_fields = defaults["time_zone_usage_for_time_fields"]
    if save_output_layer is None:
        save_output_layer = False
    if overrides is None:
        overrides = defaults["overrides"]
    if save_route_data is None:
        save_route_data = defaults["save_route_data"]
    if time_impedance is None:
        time_impedance = defaults.get("time_impedance", None)
    if distance_impedance is None:
        distance_impedance = defaults.get("distance_impedance", None)
    if populate_stop_shapes is None:
        populate_stop_shapes = defaults.get("populate_stop_shapes", None)
    if output_format is None:
        output_format = defaults.get("output_format", None)
    from arcgis._impl.common._utils import inspect_function_inputs

    params = {
        "orders": orders,
        "depots": depots,
        "routes": routes,
        "breaks": breaks,
        "time_units": time_units,
        "distance_units": distance_units,
        "analysis_region": analysis_region,
        "default_date": default_date,
        "uturn_policy": uturn_policy,
        "time_window_factor": time_window_factor,
        "spatially_cluster_routes": spatially_cluster_routes,
        "route_zones": route_zones,
        "route_renewals": route_renewals,
        "order_pairs": order_pairs,
        "excess_transit_factor": excess_transit_factor,
        "point_barriers": point_barriers,
        "line_barriers": line_barriers,
        "polygon_barriers": polygon_barriers,
        "use_hierarchy_in_analysis": use_hierarchy_in_analysis,
        "restrictions": restrictions,
        "attribute_parameter_values": attribute_parameter_values,
        "populate_route_lines": populate_route_lines,
        "route_line_simplification_tolerance": route_line_simplification_tolerance,
        "populate_directions": populate_directions,
        "directions_language": directions_language,
        "directions_style_name": directions_style_name,
        "travel_mode": travel_mode,
        "impedance": impedance,
        "time_zone_usage_for_time_fields": time_zone_usage_for_time_fields,
        "save_output_layer": save_output_layer,
        "overrides": overrides,
        "save_route_data": save_route_data,
        "time_impedance": time_impedance,
        "distance_impedance": distance_impedance,
        "populate_stop_shapes": populate_stop_shapes,
        "output_format": output_format,
        "ignore_invalid_order_locations": ignore_invalid_order_locations,
        "gis": gis,
        "future": True,
    }
    params = inspect_function_inputs(tbx.edit_vehicle_routing_problem, **params)
    result = tbx.edit_vehicle_routing_problem(**params)
    return result


def solve_vehicle_routing_problem(
    orders: FeatureSet,
    depots: FeatureSet,
    routes: FeatureSet,
    breaks: Optional[FeatureSet] = None,
    time_units: str = "Minutes",
    distance_units: str = "Miles",
    analysis_region: Optional[str] = None,
    default_date: Optional[datetime] = None,
    uturn_policy: str = "ALLOW_DEAD_ENDS_AND_INTERSECTIONS_ONLY",
    time_window_factor: str = "Medium",
    spatially_cluster_routes: bool = True,
    route_zones: Optional[FeatureSet] = None,
    route_renewals: Optional[FeatureSet] = None,
    order_pairs: Optional[FeatureSet] = None,
    excess_transit_factor: str = "Medium",
    point_barriers: Optional[FeatureSet] = None,
    line_barriers: Optional[FeatureSet] = None,
    polygon_barriers: Optional[FeatureSet] = None,
    use_hierarchy_in_analysis: bool = True,
    restrictions: Optional[str] = None,
    attribute_parameter_values: Optional[FeatureSet] = None,
    populate_route_lines: bool = True,
    route_line_simplification_tolerance: Optional[LinearUnit] = None,
    populate_directions: bool = False,
    directions_language: str = "en",
    directions_style_name: str = "NA Desktop",
    travel_mode: str = "Custom",
    impedance: str = "Drive Time",
    gis: Optional[GIS] = None,
    time_zone_usage_for_time_fields: str = "GEO_LOCAL",
    save_output_layer: bool = False,
    overrides: Optional[dict] = None,
    save_route_data: bool = False,
    time_impedance: Optional[str] = None,
    distance_impedance: Optional[str] = None,
    populate_stop_shapes: bool = False,
    output_format: Optional[str] = None,
    future: bool = False,
    ignore_invalid_order_locations: bool = False,
):
    """
    .. |either| image:: _static/images/solve_vehicle_routing_problem/routing_either_side.png
    .. |left| image:: _static/images/solve_vehicle_routing_problem/routing_left_side.png
    .. |turn| image:: _static/images/solve_vehicle_routing_problem/routing_no_u_turn.png
    .. |right| image:: _static/images/solve_vehicle_routing_problem/routing_right_side.png
    .. |ALLOW_UTURNS| image:: _static/images/solve_vehicle_routing_problem/routing_ALLOW_UTURNS.png
    .. |NO_UTURNS| image:: _static/images/solve_vehicle_routing_problem/routing_NO_UTURNS.png
    .. |ALLOW_DEAD_ENDS_ONLY| image:: _static/images/solve_vehicle_routing_problem/routing_ALLOW_DEAD_ENDS_ONLY.png
    .. |ALLOW_DEAD_ENDS_AND_INTERSECTIONS_ONLY| image:: _static/images/solve_vehicle_routing_problem/routing_ALLOW_DEAD_ENDS_AND_INTERSECTIONS_ONLY.png


    ``solve_vehicle_routing_problem`` tool solves a vehicle routing problem (VRP) to find the best routes for a fleet of vehicles.
    A dispatcher managing a fleet of vehicles is often required to make decisions about vehicle routing. One such decision
    involves how to best assign a group of customers to a fleet of vehicles and to sequence and schedule their visits. The
    objectives in solving such vehicle routing problems (VRP) are to provide a high level of customer service by honoring
    any time windows while keeping the overall operating and investment costs for each route as low as possible. The
    constraints are to complete the routes with available resources and within the time limits imposed by driver work
    shifts, driving speeds, and customer commitments. This method can be used to determine solutions for such complex
    fleet management tasks.

    Consider an example of delivering goods to grocery stores from a central warehouse location.
    A fleet of three trucks is available at the warehouse. The warehouse operates only within a certain time window (from
    8:00 a.m. to 5:00 p.m.) during which all trucks must return back to the warehouse. Each truck has a capacity of 15,000
    pounds, which limits the amount of goods it can carry. Each store has a demand for a specific amount of goods (
    in pounds) that needs to be delivered, and each store has time windows that confine when deliveries should be made.
    Furthermore, the driver can work only eight hours per day, requires a break for lunch, and is paid for the time spent
    on driving and servicing the stores. The goal is to come up with an itinerary for each driver (or route) such that the
    deliveries can be made while honoring all the service requirements and minimizing the total time spent on a particular
    route by the driver.

    ======================================    ==========================================================================================================================================
    **Parameter**                              **Description**
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    orders                                    Required :class:`~arcgis.features.FeatureSet`. Specify one or more orders (up to 2,000). These are the locations
                                              that the routes of the vehicle routing problem (VRP) analysis
                                              should visit. An order can represent a delivery (for example, furniture delivery),
                                              a pickup (such as an airport shuttle bus picking up a passenger), or some type of service
                                              or inspection (a tree trimming job or building inspection, for instance).

                                              When specifying the orders, you can set properties for each one, such as its name or service time, by using
                                              attributes. The orders can be specified with the following attributes:

                                              * ``ObjectID``: The system-managed ID field.
                                              * ``Name``:  The name of the order. The name must be unique. If the name is left null, a name is automatically generated at solve time.
                                              * ``Description``: The descriptive information about the order. This can contain any textual information for the order and has no restrictions for uniqueness. You may want to store a client's ID number in the Name field and the client's actual name or address in the ``Description`` field.
                                              * ``ServiceTime``:  This property specifies how much time will be spent at the network location when the route visits it; that is, it stores the impedance value for the network location. A zero or null value indicates the network location requires no service time. The unit for this field value is specified by the ``time_units`` parameter.
                                              * ``TimeWindowStart1``: The beginning time of the first time window for the network location. This field can contain a null value; a null value indicates no beginning time.

                                                A time window only states when a vehicle can arrive at an order; it doesn't state when the service time must be completed. To account for service time and leave before the time window is over, subtract ``ServiceTime`` from the ``TimeWindowEnd1`` field.

                                                The time window fields (``TimeWindowStart1``, ``TimeWindowEnd1``, ``TimeWindowStart2``, and ``TimeWindowEnd2``) can contain a time-only value or a date and time value. If a time field such as ``TimeWindowStart1`` has a time-only value (for example, 8:00 AM), the date is assumed to be the date specified by the Default Date parameter. Using date and time values (for example, 7/11/2010 8:00 AM) allows you to set time windows that span multiple days.

                                                When solving a problem that spans multiple time zones, each order's time-window values refer to the time zone in which the order is located.

                                              * ``TimeWindowEnd1``: The ending time of the first window for the network location. This field can contain a null value; a null value indicates no ending time.
                                              * ``TimeWindowStart2``: The beginning time of the second time window for the network location. This field can contain a null value; a null value indicates that there is no second time window.

                                                If the first time window is null, as specified by the ``TimeWindowStart1`` and ``TimeWindowEnd1`` fields, the second time window must also be null.

                                                If both time windows are non null, they can't overlap. Also, the second time window must occur after the first.

                                              * ``TimeWindowEnd2``: The ending time of the second time window for the network location. This field can contain a null value.

                                                When ``TimeWindowStart2`` and ``TimeWindowEnd2`` are both null, there is no second time window.

                                                When ``TimeWindowStart2`` is not null but ``TimeWindowEnd2`` is null, there is a second time window that has a starting time but no ending time. This is valid.

                                              * ``MaxViolationTime1``: A time window is considered violated if the arrival time occurs after the time window has ended. This field specifies the maximum allowable violation time for the first time window of the order. It can contain a zero value but can't contain negative values. A zero value indicates that a time window violation at the first time window of the order is unacceptable; that is, the first time window is hard. On the other hand, a null value indicates that there is no limit on the allowable violation time. A nonzero value specifies the maximum amount of lateness; for example, a route can arrive at an order up to 30 minutes beyond the end of its first time window.

                                                The unit for this field value is specified by the Time Field Units parameter

                                                Time window violations can be tracked and weighted by the
                                                solver. Because of this, you can direct the VRP solver to take one
                                                of three approaches:

                                                * Minimize the overall violation time, regardless of the
                                                  increase in travel cost for the fleet.

                                                * Find a solution that balances overall violation time and
                                                  travel cost.

                                                * Ignore the overall violation time; instead, minimize
                                                  the travel cost for the fleet.

                                                By assigning an importance level for the Time Window
                                                Violation Importance parameter, you are essentially choosing one of
                                                these three approaches. In any case, however, the solver will
                                                return an error if the value set for ``MaxViolationTime1`` is
                                                surpassed.

                                              * ``MaxViolationTime2``: The maximum allowable violation time for the second time window of the order. This field is analogous to the ``MaxViolationTime1`` field.
                                              * ``InboundArriveTime``: Defines when the item to be delivered to the order will be ready at the starting depot. The order can be assigned to a route only if the inbound arrive time the route's latest start time value; this way, the route cannot leave the depot before the item is ready to be loaded onto it.

                                                This field can help model scenarios involving inbound-wave transshipments. For example, a
                                                job at an order requires special materials that are not currently
                                                available at the depot. The materials are being shipped from another location and will arrive
                                                at the depot at 11:00 a.m. To ensure a route that leaves before the shipment arrives isn't
                                                assigned to the order, the order's inbound arrive time is set to 11:00 a.m. The special materials
                                                arrive at 11:00 a.m., they are loaded onto the vehicle, and the vehicle departs from the depot
                                                to visit its assigned orders.

                                                .. note::
                                                    * The route's start time, which includes service times, must occur after the inbound arrive time.
                                                      If a route begins before an order's inbound arrive time, the order cannot be assigned to the route.
                                                      The assignment is invalid even if the route has a start-depot service time that lasts until after the inbound arrive time.
                                                    * This time field can contain a time-only value or a date and time value. If a time-only value is
                                                      set (for example, 11:00 AM), the date is assumed to be the date specified by the Default Date parameter.
                                                      The default date is ignored, however, when any time field in the Depots, Routes, Orders, or Breaks includes a
                                                      date with the time. In that case, specify all such fields with a date and time (for example, 7/11/2015 11:00 AM).
                                                    * The VRP solver honors InboundArriveTime regardless of the DeliveryQuantities value.
                                                    * If an outbound depart time is also specified, its time value must occur after the inbound arrive time.

                                              * ``OutboundDepartTime``: Defines when the item to be picked up at the order must arrive at the ending depot.
                                                The order can be assigned to a route only if the route can visit the order and reach its end depot before
                                                the specified outbound depart time.

                                                This field can help model scenarios involving outbound-wave transshipments.
                                                For instance, a shipping company sends out delivery trucks to pick up packages from orders and bring them into a
                                                depot where they are forwarded on to other facilities, en route to their final destination.
                                                At 3:00 p.m. every day, a semitrailer stops at the depot to pick up the high-priority packages and take them
                                                directly to a central processing station. To avoid delaying the high-priority packages until the next day's 3:00 p.m.
                                                trip, the shipping company tries to have delivery trucks pick up the high-priority packages from orders and bring them
                                                to the depot before the 3:00 p.m. deadline. This is done by setting  the outbound depart time to 3:00 p.m.

                                                .. note::
                                                    * The route's end time, including service times, must occur before the outbound depart time. If a route reaches
                                                      a depot but doesn't complete its end-depot service time prior to the order's outbound depart time, the order cannot
                                                      be assigned to the route.
                                                    * This time field can contain a time-only value or a date and time value. If a time-only value is
                                                      set (for example, 11:00 AM), the date is assumed to be the date specified by the Default Date parameter.
                                                      The default date is ignored, however, when any time field in Depots, Routes, Orders, or Breaks includes
                                                      a date with the time. In that case, specify all such fields with a date and time (for example, 7/11/2015 11:00 AM).
                                                    * The VRP solver honors ``OutboundDepartTime`` regardless of the ``PickupQuantities`` value.
                                                    * If an inbound arrive time is also specified, its time value must occur before the  outbound depart time.

                                              * ``DeliveryQuantities``: The size of the delivery. You can specify size in any dimension you want, such as weight,
                                                volume, or quantity. You can even specify multiple dimensions, for example, weight and volume.

                                                Enter delivery quantities without indicating units.
                                                For example, if a 300-pound object needs to be delivered to an
                                                order, enter 300. You will need to remember that the value is in
                                                pounds.

                                                If you are tracking multiple dimensions, separate
                                                the numeric values with a space. For instance, if you are recording
                                                the weight and volume of a delivery that weighs 2,000 pounds and
                                                has a volume of 100 cubic feet, enter 2000 100. Again, you need to
                                                remember the units-in this case, pounds and cubic feet. You also
                                                need to remember the sequence in which the values and their corresponding
                                                units are entered.

                                                Make sure that Capacities for Routes and
                                                ``DeliveryQuantities`` and ``PickupQuantities`` for Orders are specified in
                                                the same manner; that is, the values need to be in the same units,
                                                and if you are using multiple dimensions, the dimensions need to be
                                                listed in the same sequence for all parameters. So if you specify
                                                weight in pounds, followed by volume in cubic feet for
                                                ``DeliveryQuantities``, the capacity of your routes and the pickup
                                                quantities of your orders need to be specified the same way: weight in
                                                pounds, then volume in cubic feet. If you mix units or change the
                                                sequence, you will get unwanted results without receiving any
                                                warning messages.

                                                An empty string or null value is equivalent to all
                                                dimensions being zero. If the string has an insufficient number of
                                                values in relation to the capacity count, or dimensions being
                                                tracked, the remaining values are treated as zeros. Delivery
                                                quantities can't be negative.

                                              * ``PickupQuantities``: The size of the pickup. You can specify size in any
                                                dimension you want, such as weight, volume, or quantity. You can
                                                even specify multiple dimensions, for example, weight and volume.
                                                You cannot, however, use negative values. This field is analogous
                                                to the ``DeliveryQuantities`` field of Orders.

                                                In the case of an exchange visit, an order can have
                                                both delivery and pickup quantities.

                                              * ``Revenue``: The income generated if the order is included in a
                                                solution. This field can contain a null value-a null value
                                                indicates zero revenue-but it can't have a negative
                                                value.

                                                Revenue is included in optimizing the objective
                                                function value but is not part of the solution's operating cost;
                                                that is, the TotalCost field in the route class never includes
                                                revenue in its output. However, revenue weights the relative
                                                importance of servicing orders.

                                              * ``SpecialtyNames``: A space-separated string containing the names of the
                                                specialties required by the order. A null value indicates that the
                                                order doesn't require specialties.

                                                The spelling of any specialties listed in the Orders
                                                and Routes classes must match exactly so that the VRP solver can
                                                link them together.

                                                To illustrate what specialties are and how they
                                                work, assume a lawn care and tree trimming company has a portion of
                                                its orders that requires a bucket truck to trim tall trees. The
                                                company would enter ``BucketTruck`` in the ``SpecialtyNames`` field for
                                                these orders to indicate their special need. ``SpecialtyNames`` would
                                                be left as null for the other orders. Similarly, the company would
                                                also enter ``BucketTruck`` in the ``SpecialtyNames`` field of routes that
                                                are driven by trucks with hydraulic booms. It would leave the field
                                                null for the other routes. At solve time, the VRP solver assigns
                                                orders without special needs to any route, but it only assigns
                                                orders that need bucket trucks to routes that have
                                                them.

                                              * ``AssignmentRule``: This field specifies the rule for assigning the order to a
                                                route. It is constrained by a domain of values, which are listed
                                                below (use the numeric code, not the name in parentheses).

                                                * 0 (Exclude) - The order is to be excluded from the
                                                  subsequent solve operation.
                                                * 1 (Preserve route and relative sequence) - The solver must
                                                  always assign the order to the preassigned route and at the
                                                  preassigned relative sequence during the solve operation. If this
                                                  assignment rule can't be followed, it results in an order
                                                  violation. With this setting, only the relative sequence is
                                                  maintained, not the absolute sequence. To illustrate what this
                                                  means, imagine there are two orders: A and B. They have sequence
                                                  values of 2 and 3, respectively. If you set their AssignmentRule
                                                  field values to Preserve route and relative sequence, A and B's
                                                  actual sequence values may change after solving because other
                                                  orders, breaks, and depot visits could still be sequenced before,
                                                  between, or after A and B. However, B cannot be sequenced before A.
                                                * 2 (Preserve route) - The solver must always assign the
                                                  order to the preassigned route during the solve operation. A valid
                                                  sequence must also be set even though the sequence may or may not
                                                  be preserved. If the order can't be assigned to the specified
                                                  route, it results in an order violation.
                                                * 3 (Override) - The solver tries to preserve the route
                                                  and sequence preassignment for the order during the solve
                                                  operation. However, a new route or sequence for the order may
                                                  be assigned if it helps minimize the overall value of the objective
                                                  function. This is the default value.
                                                * 4 (Anchor first) - The solver ignores the route and sequence preassignment (if any)
                                                  for the order during the solve operation. It assigns a route to the order and makes
                                                  it the first order on that route to minimize the overall value of the objective function.
                                                * 5 (Anchor last) - The solver ignores the route and sequence preassignment (if any) for the
                                                  order during the solve operation. It assigns a route to the order and makes it the last
                                                  order on that route to minimize the overall value of the objective function.
                                                  This field can't contain a null value.

                                              * ``CurbApproach``:  Specifies the direction a vehicle may arrive at and depart from the order.
                                                The field value is specified as one of the following integers shown in the parentheses (use the
                                                numeric code, not the name in parentheses):

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


                                                The ``CurbApproach`` property is designed to work with both kinds of national driving standards:
                                                right-hand traffic (United States) and left-hand traffic (United Kingdom). First, consider an
                                                order on the left side of a vehicle. It is always on the left side regardless of whether the
                                                vehicle travels on the left or right half of the road. What may change with national driving
                                                standards is your decision to approach an order from one of two directions, that is, so it ends
                                                up on the right or left side of the vehicle. For example, if you want to arrive at an order and
                                                not have a lane of traffic between the vehicle and the order, you would choose 1 (Right side of
                                                vehicle) in the United States but 2 (Left side of vehicle) in the United Kingdom.

                                              * ``RouteName``: The name of the route to which the order is assigned. As an input field, this field is used to preassign
                                                an order to a specific route. (A maximum of 200 orders can be preassigned to one route name.) It can contain a null value,
                                                indicating that the order is not preassigned to any route, and the solver determines the best possible route assignment for the order.
                                                If this is set to null, the sequence field must also be set to null.

                                                After a solve operation, if the order is routed, the ``RouteName`` field contains the name of the route to which the order is
                                                assigned.

                                              * ``Sequence``: This indicates the sequence of the order on its assigned route.

                                                As an input field, this field is used to specify the
                                                relative sequence for an order on the route. This field can contain
                                                a null value specifying that the order can be placed anywhere along
                                                the route. A null value can only occur together with a null
                                                ``RouteName`` value.

                                                The input sequence values are positive and unique
                                                for each route (shared across renewal depot visits, orders, and
                                                breaks) but do not need to start from 1 or be
                                                contiguous.

                                                After a solve operation, the Sequence field contains
                                                the sequence value of the order on its assigned route. Output
                                                sequence values for a route are shared across depot visits, orders,
                                                and breaks; start from 1 (at the starting depot); and are
                                                consecutive. So the smallest possible output sequence value for a
                                                routed order is 2, since a route always begins at a depot.

                                              * ``Bearing``: The direction in which a point is moving. The units are degrees and are measured
                                                clockwise from true north. This field is used in conjunction with the BearingTol field.
                                                Bearing data is usually sent automatically from a mobile device equipped with a GPS receiver.
                                                Try to include bearing data if you are loading an input location that is moving, such as a pedestrian or a vehicle.
                                                Using this field tends to prevent adding locations to the wrong edges, which can occur when a vehicle is near an
                                                intersection or an overpass for example. Bearing also helps the tool determine on which side of the street the point is.

                                              * ``BearingTol``: The bearing tolerance value creates a range of acceptable bearing values when
                                                locating moving points on an edge using the Bearing field. If the value from the Bearing field is
                                                within the range of acceptable values that are generated from the bearing tolerance on an edge, the
                                                point can be added as a network location there; otherwise, the closest point on the next-nearest edge is evaluated.

                                                The units are in degrees, and the default value is 30. Values must be greater than 0 and less than 180.
                                                A value of 30 means that when ArcGIS Network Analyst extension attempts to add a network location on an
                                                edge, a range of acceptable bearing values is generated 15 degrees to either side of the edge (left and right)
                                                and in both digitized directions of the edge.

                                              * ``NavLatency``: This field is only used in the solve process if Bearing and BearingTol also have values; however,
                                                entering a ``NavLatency`` value is optional, even when values are present in Bearing and ``BearingTol``. ``NavLatency``
                                                indicates how much time is expected to elapse from the moment GPS information is sent from a moving vehicle
                                                to a server and the moment the processed route is received by the vehicle's navigation device.

                                                The time units of ``NavLatency`` are the same as the units specified by the ``timeUnits`` property of the analysis object.

    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    depots                                    Required :class:`~arcgis.features.FeatureSet` . Specify one or more depots for the given vehicle routing problem. A depot is a location that
                                              a vehicle departs from at the beginning of its workday and returns to at the end of the workday. Vehicles are
                                              loaded (for deliveries) or unloaded (for pickups) at depots at the start of the route. In some cases, a depot can also act as a
                                              renewal location whereby the vehicle can unload or reload and continue performing deliveries and pickups. A depot has open and
                                              close times, as specified by a hard time window. Vehicles can't arrive at a depot outside of this time window.
                                              When specifying the orders, you can set properties for each one, such as its name or service time, by using attributes.

                                              The orders can be specified with the following attributes:

                                              * ``ObjectID``: The system-managed ID field.

                                              * ``Name``: The name of the depot. The ``StartDepotName`` and ``EndDepotName`` fields of the Routes record set reference the names you specify
                                                here. It is also referenced by the Route Renewals record set, when used. Depot names are case insensitive and have to be nonempty and unique.

                                              * ``Description`` : The descriptive information about the depot location. This can contain any textual information and has
                                                no restrictions for uniqueness.For example, if you want to note which region a depot is in or the depot's address and telephone number, you can enter
                                                the information here rather than in the Name field.

                                              * ``TimeWindowStart1``: The beginning time of the first time window for the network location. This field can contain a null value;
                                                a null value indicates no beginning time.

                                                Time window fields can contain a time-only value or a date and time value. If a time field has a time-only value (for example,
                                                8:00 AM), the date is assumed to be the date specified by the Default Date parameter of the analysis layer. Using date and time
                                                values (for example, 7/11/2010 8:00 AM) allows you to set time windows that span multiple days.

                                                When solving a problem that spans multiple time zones, each depot's time-window values refer to the time zone in which the depot is located.

                                              * ``TimeWindowEnd1``: The ending time of the first window for the network
                                                location. This field can contain a null value; a null value
                                                indicates no ending time.

                                              * ``TimeWindowStart2``: The beginning time of the second time window for the network location.
                                                This field can contain a null value; a null value indicates that there is no second time window.

                                                If the first time window is null, as specified by the
                                                ``TimeWindowStart1`` and ``TimeWindowEnd1`` fields, the second time window
                                                must also be null.

                                                If both time windows are nonnull, they can't overlap.
                                                Also, the second time window must occur after the first.

                                              * ``TimeWindowEnd2``: The ending time of the second time window for the network
                                                location. This field can contain a null value.

                                                When ``TimeWindowStart2`` and ``TimeWindowEnd2`` are both null,
                                                there is no second time window.

                                                When ``TimeWindowStart2`` is not null but ``TimeWindowEnd2`` is
                                                null, there is a second time window that has a starting time but no
                                                ending time. This is valid.

                                              * ``CurbApproach``:  Specifies the direction a vehicle may arrive at and depart
                                                from the depot. The field value is specified as one of the
                                                following integers shown in the parentheses (use the numeric code, not the name in parentheses):

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
                                                                            with the bus stop on the left side.
                                                -------------------------  ---------------------------------------------------------------
                                                No U-Turn                  |turn|
                                                                            When the vehicle approaches the order, the curb can be on either side
                                                                            of the vehicle; however, the vehicle must depart without turning around.
                                                =========================  ===============================================================

                                                The ``CurbApproach`` property is designed to work with both kinds of national driving standards: right-hand traffic (United States)
                                                and left-hand traffic (United Kingdom). First, consider a depot on the left side of a vehicle. It is always on the left side
                                                regardless of whether the vehicle travels on the left or right half of the road. What may change with national driving standards
                                                is your decision to approach a depot from one of two directions, that is, so it ends up on the right or left side of the vehicle.
                                                For example, if you want to arrive at a depot and not have a lane of traffic between the vehicle and the depot, you would choose
                                                1 (Right side of vehicle) in the United States but 2 (Left side of vehicle) in the United Kingdom.

                                              * ``Bearing``: The direction in which a point is moving. The units are degrees and measured in a clockwise fashion from true north.
                                                This field is used in conjunction with the ``BearingTol`` field.

                                                Bearing data is usually sent automatically from a mobile
                                                device equipped with a GPS receiver. Try to include bearing
                                                data if you are loading an order that is moving, such as a
                                                pedestrian or a vehicle.

                                                Using this field tends to prevent adding locations to the
                                                wrong edges, which can occur when a vehicle is near an intersection
                                                or an overpass, for example. Bearing also helps the tool
                                                determine on which side of the street the point is.
                                                For more information, see the Bearing and Bearing Tolerance Help topic (http://links.esri.com/bearing-and-bearing-tolerance).

                                              * ``BearingTol``: The bearing tolerance value creates a range of acceptable
                                                bearing values when locating moving points on an edge using the
                                                Bearing field. If the value from the Bearing field is within the
                                                range of acceptable values that are generated from the bearing
                                                tolerance on an edge, the point can be added as a network location
                                                there; otherwise, the closest point on the next-nearest edge is
                                                evaluated.

                                                The units are in degrees and the default value is 30.
                                                Values must be greater than zero and less than 180.

                                                A value of 30 means that when Network Analyst attempts to
                                                add a network location on an edge, a range of acceptable bearing
                                                values is generated 15 degrees to either side of the edge (left and
                                                right) and in both digitized directions of the edge.
                                                For more information, see the Bearing and Bearing Tolerance topic in the ArcGIS help system (http://links.esri.com/bearing-and-bearing-tolerance).

                                              * ``NavLatency``: This field is only used in the solve process if Bearing
                                                and BearingTol also have values; however, entering a ``NavLatency``
                                                value is optional, even when values are present in Bearing and
                                                BearingTol. ``NavLatency`` indicates how much time is expected to
                                                elapse from the moment GPS information is sent from a moving
                                                vehicle to a server and the moment the processed route is received
                                                by the vehicle's navigation device. The time units of ``NavLatency``
                                                are the same as the units of the cost attribute specified by the
                                                parameter Time Attribute.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    routes                                    Required :class:`~arcgis.features.FeatureSet` .  Specify one or more routes (up to 100). A route specifies vehicle and driver
                                              characteristics; after solving, it also represents the path between
                                              depots and orders.

                                              A route can have start and end depot service times, a  fixed or flexible starting time, time-based operating costs,
                                              distance-based operating costs, multiple capacities, various constraints on a driver's workday, and so on.
                                              When specifying the routes, you can set properties for each one by using attributes.
                                              The routes can be specified with the following attributes:

                                              * ``Name``: The name of the route. The name must be unique.

                                                The tool  generates a unique name at solve time if the field value is null; therefore, entering a value is optional in
                                                most cases. However, you must enter a name if your analysis includes breaks, route renewals, route zones, or orders that are
                                                preassigned to a route because the route name is used as a foreign
                                                key in these cases. Note that route names are case insensitive.
                                              * ``StartDepotName``: The name of the starting depot for the route. This field
                                                is a foreign key to the Name field in Depots.

                                                If the ``StartDepotName`` value is null, the route will begin
                                                from the first order assigned. Omitting the start depot is useful
                                                when the vehicle's starting location is unknown or irrelevant to
                                                your problem. However, when ``StartDepotName`` is null, EndDepotName
                                                cannot also be null.

                                                If the route is making deliveries and ``StartDepotName`` is
                                                null, it is assumed the cargo is loaded on the vehicle at a virtual
                                                depot before the route begins. For a route that has no renewal
                                                visits, its delivery orders (those with nonzero DeliveryQuantities
                                                values in the Orders class) are loaded at the start depot or
                                                virtual depot. For a route that has renewal visits, only the
                                                delivery orders before the first renewal visit are loaded at the
                                                start depot or virtual depot.
                                              * ``EndDepotName``: The name of the ending depot for the route. This field is a foreign key to the Name field in the Depots class.
                                              * ``StartDepotServiceTime``: The service time at the starting depot. This can be used to model the time spent for loading the vehicle.
                                                This field can contain a null value; a null value indicates zero service time.

                                                The unit for this field value is specified by the Time
                                                Field Units parameter.

                                                The service times at the start and end depots are fixed
                                                values (given by the ``StartDepotServiceTime`` and ``EndDepotServiceTime``
                                                field values) and do not take into account the actual load for a
                                                route. For example, the time taken to load a vehicle at the
                                                starting depot may depend on the size of the orders. As such, the
                                                depot service times could be given values corresponding to a full
                                                truckload or an average truckload, or you could make your own time
                                                estimate.
                                              * ``EndDepotServiceTime``: The service time at the ending depot. This can be used to model the time spent for unloading the vehicle.
                                                This field can contain a null value; a null value indicates zero service time.

                                                The unit for this field value is specified by the Time
                                                Field Units parameter.

                                                The service times at the start and end depots are fixed
                                                values (given by the ``StartDepotServiceTime`` and ``EndDepotServiceTime``
                                                field values) and do not take into account the actual load for a
                                                route. For example, the time taken to load a vehicle at the
                                                starting depot may depend on the size of the orders. As such, the
                                                depot service times could be given values corresponding to a full
                                                truckload or an average truckload, or you could make your own time
                                                estimate.
                                              * ``EarliestStartTime``: The earliest allowable starting time for the route. This
                                                is used by the solver in conjunction with the time window of the
                                                starting depot for determining feasible route start
                                                times.

                                                This field can't contain null values and has a default
                                                time-only value of 8:00 AM; the default value is interpreted as
                                                8:00 a.m. on the date given by the Default Date
                                                parameter.

                                                When solving a problem that spans multiple time zones, the
                                                time zone for ``EarliestStartTime`` is the same as the time zone in which the starting depot is located.
                                              * ``LatestStartTime``: The latest allowable starting time for the route. This
                                                field can't contain null values and has a default time-only value
                                                of 10:00 AM; the default value is interpreted as 10:00 a.m. on the
                                                date given by the Default Date property of the analysis
                                                layer.

                                                When solving a problem that spans multiple time zones, the
                                                time zone for ``LatestStartTime`` is the same as the time zone in which the starting depot is located.
                                              * ``ArriveDepartDelay``: This field stores the amount of travel time needed to
                                                accelerate the vehicle to normal travel speeds, decelerate it to a
                                                stop, and move it off and on the network (for example, in and out
                                                of parking). By including an ``ArriveDepartDelay`` value, the VRP
                                                solver is deterred from sending many routes to service physically
                                                coincident orders.

                                                The cost for this property is incurred between visits to
                                                noncoincident orders, depots, and route renewals. For example, when
                                                a route starts from a depot and visits the first order, the total
                                                arrive/depart delay is added to the travel time. The same is true
                                                when traveling from the first order to the second order. If the
                                                second and third orders are coincident, the ``ArriveDepartDelay`` value
                                                is not added between them since the vehicle doesn't need to move.
                                                If the route travels to a route renewal, the value is added to the
                                                travel time again.

                                                Although a vehicle needs to slow down and stop for a break
                                                and accelerate afterwards, the VRP solver cannot add the
                                                ``ArriveDepartDelay`` value for breaks. This means that if a route
                                                leaves an order, stops for a break, and continues to the next
                                                order, the arrive/depart delay is added only once, not
                                                twice.

                                                To illustrate, assume there are five coincident orders in
                                                a high-rise building, and they are serviced by three different
                                                routes. This means three arrive/depart delays would be incurred;
                                                that is, three drivers would need to separately find parking places
                                                and enter the same building. However, if the orders could be
                                                serviced by just one route instead, only one driver would need to
                                                park and enter the building-only one arrive/depart delay would be
                                                incurred. Since the VRP solver tries to minimize cost, it will try
                                                to limit the arrive/depart delays and thus choose the single-route
                                                option. (Note that multiple routes may need to be sent when other
                                                constraints-such as specialties, time windows, or
                                                capacities-require it.)
                                                The unit for this field value is specified by the time_units parameter.
                                              * ``Capacities``: The maximum capacity of the vehicle. You can specify
                                                capacity in any dimension you want, such as weight, volume, or
                                                quantity. You can even specify multiple dimensions, for example,
                                                weight and volume.

                                                Enter capacities without indicating units. For example,
                                                assume your vehicle can carry a maximum of 40,000 pounds; you would
                                                enter 40000. You need to remember for future reference that the
                                                value is in pounds.

                                                If you are tracking multiple dimensions, separate the
                                                numeric values with a space. For instance, if you are recording
                                                both weight and volume and your vehicle can carry a maximum weight
                                                of 40,000 pounds and a maximum volume of 2,000 cubic feet,
                                                Capacities should be entered as 40000 2000. Again, you need to
                                                remember the units. You also need to remember the sequence in which the
                                                values and their corresponding units are entered (pounds
                                                followed by cubic feet in this case).

                                                Remembering the units and the unit sequence is important
                                                for a couple of reasons: one, so you can reinterpret the
                                                information later; two, so you can properly enter values for the
                                                ``DeliveryQuantities`` and ``PickupQuantities`` fields for the orders. To
                                                elaborate on the second point, note that the VRP solver
                                                simultaneously refers to Capacities, ``DeliveryQuantities``, and
                                                ``PickupQuantities`` to make sure that a route doesn't become
                                                overloaded. Since units can't be entered in the field, the VRP tool can't make unit conversions, so you need to enter the
                                                values for the three fields using the same units and the same unit
                                                sequence to ensure the values are correctly interpreted. If you mix
                                                units or change the sequence in any of the three fields, you will
                                                get unwanted results without receiving any warning messages. Thus,
                                                it is a good idea to set up a unit and unit-sequence standard
                                                beforehand and continually refer to it whenever entering values for
                                                these three fields.

                                                An empty string or null value is equivalent to all values
                                                being zero. Capacity values can't be negative.

                                                If the Capacities string has an insufficient number of
                                                values in relation to the ``DeliveryQuantities`` or ``PickupQuantities``
                                                fields for orders, the remaining values are treated as
                                                zero.

                                              * ``FixedCost``: A fixed monetary cost that is incurred only if the route
                                                is used in a solution (that is, it has orders assigned to it). This
                                                field can contain null values; a null value indicates zero fixed
                                                cost. This cost is part of the total route operating
                                                cost.
                                              * ``CostPerUnitTime``: The monetary cost incurred-per unit of work time-for the
                                                total route duration, including travel times as well as service
                                                times and wait times at orders, depots, and breaks. This field
                                                can't contain a null value and has a default value of
                                                1.0.
                                                The unit for this field value is specified by the time_units parameter.
                                              * ``CostPerUnitDistance``: The monetary cost incurred-per unit of distance
                                                traveled-for the route length (total travel distance). This field
                                                can contain null values; a null value indicates zero
                                                cost.
                                                The unit for this field value is specified by the distance_units parameter.
                                              * ``OvertimeStartTime``: The duration of regular work time before overtime
                                                computation begins. This field can contain null values; a null
                                                value indicates that overtime does not apply.
                                                The unit for this field value is specified by the ``time_units`` parameter.
                                                For example, if the driver is to be paid overtime pay when
                                                the total route duration extends beyond eight hours,
                                                ``OvertimeStartTime`` is specified as 480 (8 hours * 60 minutes/hour),
                                                given the ``time_units`` parameter is set to Minutes.
                                              * ``CostPerUnitOvertime``: The monetary cost incurred per time unit of overtime work.
                                                This field can contain null values; a null value indicates that the
                                                ``CostPerUnitOvertime`` value is the same as the ``CostPerUnitTime``
                                                value.
                                              * ``MaxOrderCount``: The maximum allowable number of orders on the route. This
                                                field can't contain null values and has a default value of
                                                30. This value cannot exceed 200.
                                              * ``MaxTotalTime``: The maximum allowable route duration. The route duration
                                                includes travel times as well as service and wait times at orders,
                                                depots, and breaks. This field can contain null values; a null
                                                value indicates that there is no constraint on the route
                                                duration.
                                                The unit for this field value is specified by the time_units parameter.
                                              * ``MaxTotalTravelTime``: The maximum allowable travel time for the route. The
                                                travel time includes only the time spent driving on the network and
                                                does not include service or wait times.

                                                This field can contain null values; a null value indicates
                                                there is no constraint on the maximum allowable travel time. This
                                                field value can't be larger than the ``MaxTotalTime`` field
                                                value.
                                                The unit for this field value is specified by the time_units parameter.
                                              * ``MaxTotalDistance``: The maximum allowable travel distance for the
                                                route.
                                                The unit for this field value is specified by the distance_units parameter.
                                                This field can contain null values; a null value indicates
                                                that there is no constraint on the maximum allowable travel
                                                distance.
                                              * ``SpecialtyNames``: A space-separated string containing the names of the
                                                specialties supported by the route. A null value indicates that the
                                                route does not support any specialties.

                                                This field is a foreign key to the ``SpecialtyNames`` field in
                                                the orders class.

                                                To illustrate what specialties are and how they work,
                                                assume a lawn care and tree trimming company has a portion of its
                                                orders that requires a bucket truck to trim tall trees. The company
                                                would enter ``BucketTruck`` in the ``SpecialtyNames`` field for these
                                                orders to indicate their special need. ``SpecialtyNames`` would be left
                                                as null for the other orders. Similarly, the company would also
                                                enter ``BucketTruck`` in the ``SpecialtyNames`` field of routes that are
                                                driven by trucks with hydraulic booms. It would leave the field
                                                null for the other routes. At solve time, the VRP solver assigns
                                                orders without special needs to any route, but it only assigns
                                                orders that need bucket trucks to routes that have them.
                                              * ``AssignmentRule``: This specifies whether the route can be used or not when
                                                solving the problem. This field is constrained by a domain of
                                                values, which are listed below (use the numeric code, not the name in parentheses).

                                                * 1 (Include)  - The route is included in the solve operation.
                                                  This is the default value.

                                                * 2 (Exclude) - The route is excluded from the solve
                                                  operation.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    breaks                                    Optional :class:`~arcgis.features.FeatureSet` . These are the rest periods, or breaks, for the routes in a given
                                              vehicle routing problem. A break is associated with exactly one
                                              route, and it can be taken after completing an order, while en
                                              route to an order, or prior to servicing an order. It has a start
                                              time and a duration, for which the driver may or may not be paid.
                                              There are three options for establishing when a break begins: using
                                              a time window, a maximum travel time, or a maximum work
                                              time.
                                              When specifying the breaks, you can set properties for each one, such as its name or service time, by using attributes.
                                              The breaks parameter can be specified with the following attributes:

                                              * ``RouteName``: The name of the route that the break applies to. Although a break is assigned to exactly one route, many breaks can be assigned to the same route.

                                                This field is a foreign key to the Name field in the
                                                routes parameter, so it can't have a null value.

                                              * ``Precedence``: Precedence values sequence the breaks of a given route.
                                                Breaks with a precedence value of 1 occur before those with a value
                                                of 2, and so on.

                                                All breaks must have a precedence value, regardless of
                                                whether they are time-window, maximum-travel-time, or
                                                maximum-work-time breaks.

                                              * ``ServiceTime``: The duration of the break. This field can contain null
                                                values; a null value indicates no service time.
                                                The unit for this field value is specified by the time_units parameter.

                                              * ``TimeWindowStart``: The starting time of the break's time window.

                                                If this field is null and ``TimeWindowEnd`` has a valid
                                                time-of-day value, the break is allowed to start any time before the
                                                ``TimeWindowEnd`` value.

                                                If this field has a value, the ``MaxTravelTimeBetweenBreaks`` and
                                                ``MaxCumulWorkTime`` field values must be null; moreover, all other breaks in the
                                                analysis layer must have null values for ``MaxTravelTimeBetweenBreaks``
                                                and ``MaxCumulWorkTime``.

                                                An error will occur at solve time if a route has multiple
                                                breaks with overlapping time windows.

                                                The time window fields in breaks can contain a time-only
                                                value or a date and time value. If a time field, such as
                                                ``TimeWindowStart``, has a time-only value (for example, 12:00 PM), the
                                                date is assumed to be the date specified by the ``default_date``
                                                parameter. Using date and time values (for example, 7/11/2012 12:00
                                                PM) allows you to specify time windows that span two or more days.
                                                This is especially beneficial when a break should be taken sometime
                                                before and after midnight.
                                                When solving a problem that spans multiple time zones, each break's time-window
                                                values refer to the time zone in which the associated route, as specified by the ``RouteName`` field,
                                                is located.

                                              * ``TimeWindowEnd``: The ending time of the break's time window.

                                                If this field is null and ``TimeWindowStart`` has a valid
                                                time-of-day value, the break is allowed to start any time after the
                                                ``TimeWindowStart`` value.

                                                If this field has a value, ``MaxTravelTimeBetweenBreaks`` and
                                                ``MaxCumulWorkTime`` must be null; moreover, all other breaks in the
                                                analysis layer must have null values for ``MaxTravelTimeBetweenBreaks``
                                                and ``MaxCumulWorkTime``.

                                              * ``MaxViolationTime``: This field specifies the maximum allowable violation time
                                                for a time-window break. A time window is considered violated if
                                                the arrival time falls outside the time range.

                                                A zero value indicates the time window cannot be violated;
                                                that is, the time window is hard. A nonzero value specifies the
                                                maximum amount of lateness; for example, the break can begin up to
                                                30 minutes beyond the end of its time window, but the lateness is
                                                penalized as per the Time Window Violation Importance
                                                parameter.

                                                This property can be null; a null value with
                                                ``TimeWindowStart`` and ``TimeWindowEnd`` values indicates that there is no
                                                limit on the allowable violation time. If
                                                ``MaxTravelTimeBetweenBreaks`` or ``MaxCumulWorkTime`` has a value,
                                                ``MaxViolationTime`` must be null.
                                                The unit for this field value is specified by the time_units parameter.

                                              * ``MaxTravelTimeBetweenBreaks``: The maximum amount of travel time that can be accumulated
                                                before the break is taken. The travel time is accumulated either
                                                from the end of the previous break or, if a break has not yet been
                                                taken, from the start of the route.

                                                If this is the route's final break,
                                                ``MaxTravelTimeBetweenBreaks`` also indicates the maximum travel time
                                                that can be accumulated from the final break to the end
                                                depot.

                                                This field is designed to limit how long a person can
                                                drive until a break is required. For instance, if the Time Field
                                                Units parameter (time_units for Python) of the analysis is set to
                                                Minutes, and ``MaxTravelTimeBetweenBreaks`` has a value of 120, the
                                                driver will get a break after two hours of driving. To assign a
                                                second break after two more hours of driving, the second break's
                                                ``MaxTravelTimeBetweenBreaks`` property should be 120.

                                                If this field has a value, ``TimeWindowStart``, ``TimeWindowEnd``,
                                                ``MaxViolationTime``, and ``MaxCumulWorkTime`` must be null for an analysis
                                                to solve successfully.
                                                The unit for this field value is specified by the time_units parameter.

                                              * ``MaxCumulWorkTime``: The maximum amount of work time that can be accumulated
                                                before the break is taken. Work time is always accumulated from the
                                                beginning of the route.

                                                Work time is the sum of travel time and service times at
                                                orders, depots, and breaks. Note, however, that this excludes wait
                                                time, which is the time a route (or driver) spends waiting at an
                                                order or depot for a time window to begin.

                                                This field is designed to limit how long a person can work
                                                until a break is required. For instance, if the ``time_units``
                                                parameter is set to Minutes,
                                                ``MaxCumulWorkTime`` has a value of 120, and ``ServiceTime`` has a value of
                                                15, the driver will get a 15-minute break after two hours of
                                                work.

                                                Continuing with the last example, assume a second break is
                                                needed after three more hours of work. To specify this break, you
                                                would enter 315 (five hours and 15 minutes) as the second break's
                                                ``MaxCumulWorkTime`` value. This number includes the ``MaxCumulWorkTime``
                                                and ``ServiceTime`` values of the preceding break, along with the three
                                                additional hours of work time before granting the second break. To
                                                avoid taking maximum-work-time breaks prematurely, remember that
                                                they accumulate work time from the beginning of the route and that
                                                work time includes the service time at previously visited depots,
                                                orders, and breaks.

                                                If this field has a value, ``TimeWindowStart``, ``TimeWindowEnd``,
                                                ``MaxViolationTime``, and ``MaxTravelTimeBetweenBreaks`` must be null for
                                                an analysis to solve successfully.
                                                The unit for this field value is specified by the time_units parameter.

                                              * ``IsPaid``: A Boolean value indicating whether the break is paid or
                                                unpaid. A True value indicates that the time spent at the break is
                                                included in the route cost computation and overtime determination.
                                                A False value indicates otherwise. The default value is
                                                True.

                                              * ``Sequence``: As an input field, this indicates the sequence of the
                                                break on its route. This field can contain null values. The input
                                                sequence values are positive and unique for each route (shared
                                                across renewal depot visits, orders, and breaks) but need not start
                                                from 1 or be contiguous.

                                                The solver modifies the sequence field. After solving,
                                                this field contains the sequence value of the break on its route.
                                                Output sequence values for a route are shared across depot visits,
                                                orders, and breaks; start from 1 (at the starting depot); and are
                                                consecutive.

                                              * ``ArriveTimeUTC``: The date and time value indicating the arrival time in UTC time.

                                              * ``DepartTimeUTC``: The date and time value indicating the departure time in UTC time.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    time_units                                Optional string. The time units for all time-based field values in the
                                              analysis. Many features and records in a VRP analysis have fields
                                              for storing time values, such as ``ServiceTime`` for orders and
                                              ``CostPerUnitTime`` for routes. To minimize data entry requirements,
                                              these field values don't include units. Instead, all distance-based
                                              field values must be entered in the same units, and this parameter
                                              is used to specify the units of those values.

                                              Note that output time-based fields use the same units specified by this parameter.

                                              Choice list:['Seconds', 'Minutes', 'Hours', 'Days']
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    distance_units                            Optional string. The distance units for all distance-based field values in
                                              the analysis. Many features and records in a VRP analysis have
                                              fields for storing distance values, such as ``MaxTotalDistance`` and
                                              ``CostPerUnitDistance`` for Routes. To minimize data entry
                                              requirements, these field values don't include units. Instead, all
                                              distance-based field values must be entered in the same units, and
                                              this parameter is used to specify the units of those
                                              values.

                                              Note that output distance-based fields use the same units
                                              specified by this parameter.

                                              Choice list:['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles', 'NauticalMiles']
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    analysis_region                           Optional string. Specify the region in which to perform the analysis. If a value is not specified for this parameter, the tool
                                              will automatically calculate the region name based on the location
                                              of the input points. Setting the name of the region is recommended to speed up the
                                              tool execution. To specify a region, use one of
                                              the following values:  Europe Greece  India  Japan Korea  MiddleEastAndAfrica  NorthAmerica  Oceania  SouthAmerica  SouthEastAsia Taiwan Thailand

                                              Choice list:['NorthAmerica', 'SouthAmerica', 'Europe', 'MiddleEastAndAfrica', 'India', 'SouthAsia', 'SouthEastAsia', 'Thailand', 'Taiwan', 'Japan', 'Oceania', 'Greece', 'Korea']
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    default_date                              Optional datetime. The default date for time field values that specify a time
                                              of day without including a date. You can find these time fields in various input
                                              parameters, such as the ServiceTime attributes in the orders and breaks parameters.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    uturn_policy                              Optional string. Use this parameter to restrict or permit the service area to make U-turns at junctions.
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
                                                                                        U-turns are prohibited at all junctions and interesections
                                                                                        and are permitted only at dead ends.
                                              ----------------------------------------  ------------------------------------------------
                                              NO_UTURNS                                 U-turns are prohibited at all junctions, intersections, and dead-ends.
                                                                                        Note that even when this parameter value is chosen, a route can still
                                                                                        make U-turns at stops. If you wish to prohibit U-turns at a stop, you can set
                                                                                        its CurbApproach property to the appropriate value (3).

                                                                                        The default value for this parameter is 'ALLOW_UTURNS'.
                                              ========================================  ================================================
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    time_window_factor                        Optional string. Rates the importance of honoring time windows. There are three options described below.

                                              Choice list:['High', 'Medium', 'Low']

                                              * ``High`` - Places more importance on arriving at stops on time
                                                than on minimizing drive times. Organizations that make
                                                time-critical deliveries or that are very concerned with customer
                                                service would choose High.

                                              * ``Medium`` - This is the default value. Balances the importance
                                                of minimizing drive times and arriving within time
                                                windows.

                                              * ``Low`` - Places more importance on minimizing drive times and
                                                less on arriving at stops on time. You may want to use this setting
                                                if you have a growing backlog of service requests. For the purpose
                                                of servicing more orders in a day and reducing the backlog, you can
                                                choose Low even though customers might be inconvenienced with your
                                                late arrivals.

                                              The default value is 'Medium'.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    spatially_cluster_routes                  Optional boolean.

                                              CLUSTER (True) - Dynamic seed points are automatically created for
                                              all routes and the orders assigned to an individual
                                              route are spatially clustered. Clustering orders tends to keep
                                              routes in smaller areas and reduce how often different route lines
                                              intersect one another; yet, clustering also tends to increase
                                              overall travel times.

                                              NO_CLUSTER (False) - Dynamic seed points aren't
                                              created. Choose this option if route zones are
                                              specified.

                                              The default value is 'True'.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    route_zones                               Optional :class:`~arcgis.features.FeatureSet`. Delineates work territories for given routes. A route zone
                                              is a polygon feature and is used to constrain routes to servicing
                                              only those orders that fall within or near the specified area. Here
                                              are some examples of when route zones may be useful:

                                              * Some of your employees don't have the required permits to
                                                perform work in certain states or communities. You can create a
                                                hard route zone so they only visit orders in areas where they meet
                                                the requirements.

                                              * One of your vehicles breaks down frequently so you want to
                                                minimize response time by having it only visit orders that are
                                                close to your maintenance garage. You can create a soft or hard
                                                route zone to keep the vehicle nearby.

                                              When specifying the route zones, you need to set properties for each one, such as its associated route,
                                              by using attributes. The route zones can be specified with the following attributes:

                                              * ``RouteName``: The name of the route to which this zone applies. A route
                                                zone can have a maximum of one associated route. This field can't
                                                contain null values, and it is a foreign key to the Name field in
                                                the feature in the routes parameter.

                                              * ``IsHardZone``: A Boolean value indicating a hard or soft route zone. A
                                                True value indicates that the route zone is hard; that is, an order
                                                that falls outside the route zone polygon can't be assigned to the
                                                route. The default value is 1 (True). A False value (0) indicates
                                                that such orders can still be assigned, but the cost of servicing
                                                the order is weighted by a function that is based on the Euclidean
                                                distance from the route zone. Basically, this means that as the
                                                straight-line distance from the soft zone to the order increases,
                                                the likelihood of the order being assigned to the route
                                                decreases.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    route_renewals                            Optional :class:`~arcgis.features.FeatureSet`. Specifies the intermediate depots that routes can visit to
                                              reload or unload the cargo they are delivering or picking up.
                                              Specifically, a route renewal links a route to a depot. The
                                              relationship indicates the route can renew (reload or unload while
                                              en route) at the associated depot.

                                              Route renewals can be used to model scenarios in which a
                                              vehicle picks up a full load of deliveries at the starting depot,
                                              services the orders, returns to the depot to renew its load of
                                              deliveries, and continues servicing more orders. For example, in
                                              propane gas delivery, the vehicle may make several deliveries until
                                              its tank is nearly or completely depleted, visit a refueling point,
                                              and make more deliveries.

                                              Here are a few rules and options to consider when also
                                              working with route seed points:

                                              * The reload/unload point, or renewal location, can be
                                                different from the start or end depot.

                                              * Each route can have one or many predetermined renewal
                                                locations.

                                              * A renewal location may be used more than once by a single
                                                route.

                                              * In some cases where there may be several potential renewal
                                                locations for a route, the closest available renewal location is
                                                chosen by the solver.

                                              When specifying the route renewals, you need to set properties for each one, such
                                              as the name of the depot where the route renewal can occur, by using attributes.
                                              The route renewals can be specified with the following attributes:

                                              * ``ObjectID``: The system-managed ID field.

                                              * ``DepotName``: The name of the depot where this renewal takes place. This
                                                field can't contain a null value and is a foreign key to the Name
                                                field in the depots parameter.

                                              * ``RouteName``: The name of the route to which this renewal applies. This
                                                field can't contain a null value and is a foreign key to the Name
                                                field in the routes parameter.

                                              * ``ServiceTime``: The service time for the renewal. This field can contain a
                                                null value; a null value indicates zero service time.
                                                The unit for this field value is specified by the time_units parameter.
                                                The time taken to load a vehicle at a renewal depot may
                                                depend on the size of the vehicle and how full or empty the vehicle
                                                is. However, the service time for a route renewal is a fixed value
                                                and does not take into account the actual load. As such, the
                                                renewal service time should be given a value corresponding to a
                                                full truckload, an average truckload, or another time estimate of
                                                your choice.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    order_pairs                               Optional :class:`~arcgis.features.FeatureSet` . This parameter pairs pickup and delivery orders so they are serviced by the same route.

                                              Sometimes it is required that the pickup and delivery for
                                              orders be paired. For example, a courier company might need to have
                                              a route pick up a high-priority package from one order and deliver
                                              it to another without returning to a depot, or sorting station, to
                                              minimize delivery time. These related orders can be assigned to the
                                              same route with the appropriate sequence by using order pairs.
                                              Moreover, restrictions on how long the package can stay in the
                                              vehicle can also be assigned; for example, the package might be a
                                              blood sample that has to be transported from the doctor's office to
                                              the lab within two hours.
                                              When specifying the order pairs, you need to set properties for each one, such as the names of the two orders,
                                              by using attributes. The order pairs can be specified with the following attributes:

                                              * ``ObjectID``: The system-managed ID field.

                                              * ``FirstOrderName``: The name of the first order of the pair. This field is a
                                                foreign key to the Name field in the orders parameter.

                                              * ``SecondOrderName``: The name of the second order of the pair. This field is a
                                                foreign key to the name field in the orders parameter.

                                                The first order in the pair must be a pickup order; that
                                                is, the value for its ``DeliveryQuantities`` field is null. The second
                                                order in the pair must be a delivery order; that is, the value for
                                                its ``PickupQuantities`` field is null. The quantity picked up
                                                at the first order must agree with the quantity delivered
                                                at the second order. As a special case, both orders may have zero
                                                quantities for scenarios where capacities are not used.

                                                The order quantities are not loaded or unloaded at
                                                depots.

                                              * ``MaxTransitTime``: The maximum transit time for the pair. The transit time is
                                                the duration from the departure time of the first order to the
                                                arrival time at the second order. This constraint limits the
                                                time-on-vehicle, or ride time, between the two orders. When a
                                                vehicle is carrying people or perishable goods, the ride time is
                                                typically shorter than that of a vehicle carrying packages or
                                                nonperishable goods. This field can contain null values; a null
                                                value indicates that there is no constraint on the ride
                                                time.
                                                The unit for this field value is specified by the time_units parameter.
                                                Excess transit time (measured with respect to the direct
                                                travel time between order pairs) can be tracked and weighted by the
                                                solver. Because of this, you can direct the VRP solver to take one
                                                of three approaches:

                                                * Minimize the overall excess transit time, regardless of
                                                  the increase in travel cost for the fleet.

                                                * Find a solution that balances overall violation time and
                                                  travel cost.

                                                * Ignore the overall excess transit time and, instead,
                                                  minimize the travel cost for the fleet.

                                              By assigning an importance level for the ``excess_transit_factor`` parameter, you
                                              are in effect choosing one of these
                                              three approaches. Regardless of the importance level, the solver
                                              will always return an error if the ``MaxTransitTime`` value is
                                              surpassed.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    excess_transit_factor                     Optional string. Rates the importance of reducing excess transit time of
                                              order pairs. Excess transit time is the amount of time exceeding
                                              the time required to travel directly between the paired orders.
                                              Excess time can be caused by driver breaks or travel to
                                              intermediate orders and depots. Listed below are the three values
                                              you can choose from.

                                              Choice list:['High', 'Medium', 'Low']

                                              * ``High`` - The solver tries to find a solution with the least
                                                excess transit time between paired orders at the expense of
                                                increasing the overall travel costs. It makes sense to use this
                                                setting if you are transporting people between paired orders and
                                                you want to shorten their ride time. This is characteristic of taxi
                                                services.

                                              * ``Medium`` - This is the default setting. The solver looks for
                                                a balance between reducing excess transit time and reducing the
                                                overall solution cost.

                                              * ``Low`` - The solver tries to find a solution that minimizes
                                                overall solution cost, regardless of excess transit time. This
                                                setting is commonly used with courier services. Since couriers
                                                transport packages as opposed to people, they don't need to worry
                                                about ride time. Using Low allows the couriers to service paired
                                                orders in the proper sequence and minimize the overall solution
                                                cost.

                                              The default value is 'Medium'.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    point_barriers                            Optional :class:`~arcgis.features.FeatureSet` . Specify one or more points to act as temporary
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

                                                * 0 (Restriction) - Prohibits travel through the barrier. The barrier
                                                  is referred to as a restriction point barrier since it acts as a
                                                  restriction.

                                                * 2 (Added Cost) - Traveling through the barrier increases the travel
                                                  time or distance by the amount specified in the
                                                  Additional_Time or Additional_Distance field. This barrier type is
                                                  referred to as an added-cost point barrier.

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
                                                from true north. This field is used in conjunction with the ``BearingTol`` field.

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

                                              * ``NavLatency``: This field is only used in the solve process if Bearing and ``BearingTol`` also have values;
                                                however, entering a ``NavLatency`` value is optional, even when values are present in Bearing and ``BearingTol``.
                                                ``NavLatency`` indicates how much time is expected to elapse from the moment GPS information is sent from a
                                                moving vehicle to a server and the moment the processed route is received by the vehicle's navigation device.

                                                The time units of ``NavLatency`` are the same as the units specified by the ``timeUnits`` property of the analysis object.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    line_barriers                             Optional :class:`~arcgis.features.FeatureSet`.  Specify one or more lines that prohibit travel anywhere
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
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    polygon_barriers                          Optional :class:`~arcgis.features.FeatureSet` . Specify polygons that either completely restrict travel or
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

                                              * ``BarrierType``: Specifies whether the barrier restricts travel completely
                                                or scales the time or distance for traveling through it. The field
                                                value is specified as one of the following integers (use the numeric code, not the name in parentheses):

                                                * 0 (Restriction) - Prohibits traveling through any part of the barrier.
                                                  The barrier is referred to as a restriction polygon barrier since it
                                                  prohibits traveling on streets intersected by the barrier. One use
                                                  of this type of barrier is to model floods covering areas of the
                                                  street that make traveling on those streets impossible.

                                                * 1 (Scaled Cost) - Scales the time or distance required to travel the
                                                  underlying streets by a factor specified using the ``ScaledTimeFactor``
                                                  or ``ScaledDistanceFactor`` fields. If the streets are partially
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
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    use_hierarchy_in_analysis                 Optional boolean. Specify whether hierarchy should be used when finding the best routes.

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
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    restrictions                              Optional string. Specify which restrictions should be honored by the tool when finding the best routes.  The value you provide for this parameter
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

    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    attribute_parameter_values                Optional :class:`~arcgis.features.FeatureSet` .  Specify additional values required by some restrictions, such as the weight of a vehicle
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

                                              * ``PROHIBITED`` (-1) - Travel on the roads using the restriction is completely
                                                prohibited.

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

    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    populate_route_lines                      Optional boolean.

                                              Checked (True) - The output routes will have the
                                              exact shape of the underlying streets.

                                              Unchecked (False) - No shape is generated for the
                                              output routes, yet the routes will still contain tabular information about the solution.
                                              You won't be able to generate driving directions if
                                              route lines aren't created.

                                              When the Route Shape parameter is set to True Shape, the
                                              generalization of the route shape can be further controlled using
                                              the appropriate values for the Route Line Simplification Tolerance
                                              parameters.

                                              No matter which value you choose for the Route Shape
                                              parameter, the best routes are always determined by minimizing the
                                              travel along the streets, never using the straight-line
                                              distance. This means that only the route shapes are different,
                                              not the underlying streets that are searched when finding the
                                              route.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    route_line_simplifi cation_tolerance      Optional LinearUnit. Specify by how much you want to simplify the geometry of the output lines for
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
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    populate_directions                       Optional boolean. Specify whether the tool should generate driving directions for
                                              each route.
                                              Checked (True):
                                              Indicates that the directions will be generated
                                              and configured based on the values for the Directions Language,
                                              Directions Style Name, and Directions Distance Units
                                              parameters.
                                              Unchecked (False):
                                              Directions are not generated, and the tool
                                              returns an empty Directions layer.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    directions_language                       Optional string. Specify the language that should be used when generating
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
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    directions_style_name                     Optional string. Specify the name of the formatting style for the
                                              directions. This parameter is used only when the Populate
                                              Directions parameter is checked, or set to True. The parameter can be specified
                                              using the following values:

                                              Choice list:['NA Desktop', 'NA Navigation']

                                              * `NA Desktop`: Generates turn-by-turn directions suitable for printing.

                                              * `NA Navigation`: Generates turn-by-turn directions designed for an in-vehicle navigation device.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    travel_mode                               Optional string. Specify the mode of transportation to model in the analysis. Travel modes are managed in ArcGIS Online and can be configured by the administrator of your
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
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    impedance                                 Optional string. Specify the impedance, which is a value that represents the effort or cost of traveling
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
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    gis                                       Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    time_zone_usage_ for_time_fields          Optional string. Specifies the time zone for the input date-time fields supported by the tool. This
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
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    save_output_layer                         Optional boolean. Specify if the tool should save the analysis settings as a network analysis layer file.
                                              You cannot directly work with this file even when you open the file in an ArcGIS Desktop application like ArcMap.
                                              It is meant to be sent to Esri Technical Support to diagnose the quality of results returned from the tool.
                                              True: Save the network analysis layer file. The file is downloaded in a temporary directory on your machine. In ArcGIS Pro, the location of the downloaded file can be determined by viewing the value for the Output Network Analysis Layer parameter in the entry corresponding to the tool execution in the Geoprocessing history of your Project. In ArcMap, the location of the file can be determined by accessing the Copy Location option in the shortcut menu on the Output Network Analysis Layer parameter in the entry corresponding to the tool execution in the Geoprocessing Results window.
                                              False: Do not save the network analysis layer file. This is the default.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    overrides                                 Optional string. Specify additional settings that can influence the behavior of the solver when finding solutions
                                              for the network analysis problems. The value for this parameter needs to be specified in dict. For example, a valid value is of the following form {"overrideSetting1" : "value1", "overrideSetting2" :
                                              "value2"}. The override setting name is always enclosed in double quotes. The values can be a number, Boolean,
                                              or string. The default value for this parameter is no value, which indicates not to override any solver settings. Overrides
                                              are advanced settings that should be used only after careful analysis of the results obtained before and after applying
                                              the settings. A list of supported override settings for each solver and their acceptable values can be obtained by contacting
                                              Esri Technical Support.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    save_route_data                           Optional boolean. Choose whether the output includes a zip file that contains a file geodatabase holding the inputs
                                              and outputs of the analysis in a format that can be used to share route layers with ArcGIS Online or Portal for
                                              ArcGIS.
                                              True: Save the route data as a zip file. The file is downloaded in a temporary directory on your machine. In ArcGIS Pro,
                                              the location of the downloaded file can be determined by viewing the value for the Output Route Data parameter in the entry
                                              corresponding to the tool execution in the Geoprocessing history of your Project. In ArcMap, the location of the file can be
                                              determined by accessing the Copy Location option in the shortcut menu on the Output Route Data parameter in the entry corresponding
                                              to the tool execution in the Geoprocessing Results window.
                                              False: Do not save the route data. This is the default.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    time_impedance                            Optional string. Specify the time-based impedance.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    distance_impedence                        Optional string. Specify the distance-based impedance.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    populate_stop_shapes                      Optional boolean. Specify if the tool should create the shapes for the output assigned and unassigned stops.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    output_format                             Optional. Specify the format in which the output features are created.

                                              Choose from the following formats:

                                              * Feature Set - The output features are returned as feature classes and tables. This is the default.
                                              * JSON File - The output features are returned as a compressed file containing the JSON representation of the outputs. When this option is specified, the output is a single file (with a .zip extension) that contains one or more JSON files (with a .json extension) for each of the outputs created by the service.
                                              * GeoJSON File - The output features are returned as a compressed file containing the GeoJSON representation of the outputs. When this option is specified, the output is a single file (with a .zip extension) that contains one or more GeoJSON files (with a .geojson extension) for each of the outputs created by the service.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    future                                    Optional boolean. If True, a future object will be returned and the process
                                              will not wait for the task to complete. The default is False, which means wait for results.
    --------------------------------------    ------------------------------------------------------------------------------------------------------------------------------------------
    ignore_invalid_order_locations            Specifies whether invalid orders will be ignored when solving the vehicle routing problem.

                                              `True` - The solve operation will ignore any invalid orders and return a solution, given it didn't encounter any other errors. If you need to generate routes and deliver them to drivers immediately, you may be able to ignore invalid orders, solve, and distribute the routes to your drivers. Next, resolve any invalid orders from the last solve and include them in the VRP analysis for the next workday or work shift.
                                              `False` - The solve operation will fail when any invalid orders are encountered. An invalid order is an order that the VRP solver can't reach. An order may be unreachable for a variety of reasons, including if it's located on a prohibited network element, it isn't located on the network at all, or it's located on a disconnected portion of the network.
    ======================================    ==========================================================================================================================================

    :return: the following as a named tuple:

    * out_unassigned_stops - Output Unassigned Stops as a FeatureSet
    * out_stops - Output Stops as a FeatureSet
    * out_routes - Output Routes as a FeatureSet
    * out_directions - Output Directions as a FeatureSet
    * solve_succeeded - Solve Succeeded as a bool
    * out_network_analysis_layer - Output Network Analysis Layer as a file
    * out_route_data - Output Route Data as a file

    """
    if gis is None:
        gis = arcgis.env.active_gis
    url = gis.properties.helperServices.asyncVRP.url[
        : -len("/SolveVehicleRoutingProblem")
    ]
    url = _validate_url(url, gis)
    tbx = _create_toolbox(url, gis=gis)
    defaults = dict(
        zip(
            tbx.solve_vehicle_routing_problem.__annotations__.keys(),
            tbx.solve_vehicle_routing_problem.__defaults__,
        )
    )
    if time_impedance is None:
        time_impedance = defaults.get("time_impedance", None)
    if distance_impedance is None:
        distance_impedance = defaults.get("distance_impedance", None)
    if output_format is None:
        output_format = defaults.get("output_format", None)
    if orders is None:
        orders = defaults["orders"]
    if depots is None:
        depots = defaults["depots"]
    if routes is None:
        routes = defaults["routes"]
    if breaks is None:
        breaks = defaults["breaks"]
    if route_zones is None:
        route_zones = defaults["route_zones"]
    if route_renewals is None:
        route_renewals = defaults["route_renewals"]
    if order_pairs is None:
        order_pairs = defaults["order_pairs"]

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

    if isinstance(overrides, dict):
        overrides = json.dumps(overrides)
    from arcgis._impl.common._utils import inspect_function_inputs

    params = {
        "orders": orders,
        "depots": depots,
        "routes": routes,
        "breaks": breaks,
        "time_units": time_units,
        "distance_units": distance_units,
        "analysis_region": analysis_region,
        "default_date": default_date,
        "uturn_policy": uturn_policy,
        "time_window_factor": time_window_factor,
        "spatially_cluster_routes": spatially_cluster_routes,
        "route_zones": route_zones,
        "route_renewals": route_renewals,
        "order_pairs": order_pairs,
        "excess_transit_factor": excess_transit_factor,
        "point_barriers": point_barriers,
        "line_barriers": line_barriers,
        "polygon_barriers": polygon_barriers,
        "use_hierarchy_in_analysis": use_hierarchy_in_analysis,
        "restrictions": restrictions,
        "attribute_parameter_values": attribute_parameter_values,
        "populate_route_lines": populate_route_lines,
        "route_line_simplification_tolerance": route_line_simplification_tolerance,
        "populate_directions": populate_directions,
        "directions_language": directions_language,
        "directions_style_name": directions_style_name,
        "travel_mode": travel_mode,
        "impedance": impedance,
        "time_zone_usage_for_time_fields": time_zone_usage_for_time_fields,
        "save_output_layer": save_output_layer,
        "overrides": overrides,
        "save_route_data": save_route_data,
        "time_impedance": time_impedance,
        "distance_impedance": distance_impedance,
        "populate_stop_shapes": populate_stop_shapes,
        "output_format": output_format,
        "gis": gis,
        "future": True,
        "ignore_invalid_order_locations": ignore_invalid_order_locations,
    }
    params = inspect_function_inputs(tbx.solve_vehicle_routing_problem, **params)
    params["future"] = True
    job = tbx.solve_vehicle_routing_problem(**params)

    if future:
        return job
    return job.result()


solve_vehicle_routing_problem.__annotations__ = {
    "orders": FeatureSet,
    "depots": FeatureSet,
    "routes": FeatureSet,
    "breaks": FeatureSet,
    "time_units": str,
    "distance_units": str,
    "analysis_region": str,
    "default_date": datetime,
    "uturn_policy": str,
    "time_window_factor": str,
    "spatially_cluster_routes": bool,
    "route_zones": FeatureSet,
    "route_renewals": FeatureSet,
    "order_pairs": FeatureSet,
    "excess_transit_factor": str,
    "point_barriers": FeatureSet,
    "line_barriers": FeatureSet,
    "polygon_barriers": FeatureSet,
    "use_hierarchy_in_analysis": bool,
    "restrictions": str,
    "attribute_parameter_values": FeatureSet,
    "populate_route_lines": bool,
    "route_line_simplification_tolerance": arcgis.geoprocessing.LinearUnit,
    "populate_directions": bool,
    "directions_language": str,
    "directions_style_name": str,
    "travel_mode": str,
    "impedance": str,
    "time_zone_usage_for_time_fields": str,
    "save_output_layer": bool,
    "overrides": str,
    "save_route_data": bool,
    "return": tuple,
}
