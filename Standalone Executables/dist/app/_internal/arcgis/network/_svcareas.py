import logging as _logging
from typing import Optional
import arcgis
from datetime import datetime
from arcgis.features import FeatureSet
from arcgis.gis import GIS
from arcgis.mapping import MapImageLayer
from arcgis.geoprocessing import DataFile, LinearUnit, RasterData
from arcgis.geoprocessing._support import _execute_gp_tool
from arcgis._impl.common._utils import _validate_url
from ._routing_utils import _create_toolbox

_log = _logging.getLogger(__name__)

_use_async = True

default_facilities = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
    ],
    "geometryType": "esriGeometryPoint",
    "displayFieldName": "",
    "exceededTransferLimit": False,
    "spatialReference": {"latestWkid": 4326, "wkid": 4326},
    "features": [],
}

default_trim = {"distance": 100, "units": "esriMeters"}

default_tolerance = {"distance": 10, "units": "esriMeters"}

default_point_barriers = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {"alias": "BarrierType", "name": "BarrierType", "type": "esriFieldTypeInteger"},
        {
            "alias": "AdditionalCost",
            "name": "AdditionalCost",
            "type": "esriFieldTypeDouble",
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
        {"alias": "BarrierType", "name": "BarrierType", "type": "esriFieldTypeInteger"},
        {
            "alias": "ScaledCostFactor",
            "name": "ScaledCostFactor",
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


def generate_service_areas(
    facilities: FeatureSet,
    break_values: str = "5 10 15",
    break_units: str = "Minutes",
    analysis_region: Optional[str] = None,
    travel_direction: str = "Away From Facility",
    time_of_day: Optional[datetime] = None,
    use_hierarchy: bool = False,
    uturn_at_junctions: str = "Allowed Only at Intersections and Dead Ends",
    polygons_for_multiple_facilities: str = "Overlapping",
    polygon_overlap_type: str = "Rings",
    detailed_polygons: bool = False,
    polygon_trim_distance: Optional[LinearUnit] = None,
    polygon_simplification_tolerance: Optional[LinearUnit] = None,
    point_barriers: Optional[FeatureSet] = None,
    line_barriers: Optional[FeatureSet] = None,
    polygon_barriers: Optional[FeatureSet] = None,
    restrictions: Optional[str] = None,
    attribute_parameter_values: Optional[FeatureSet] = None,
    time_zone_for_time_of_day: str = "Geographically Local",
    travel_mode: str = "Custom",
    impedance: str = "Drive Time",
    save_output_network_analysis_layer: bool = False,
    overrides: Optional[dict] = None,
    time_impedance: Optional[str] = None,
    distance_impedance: Optional[str] = None,
    polygon_detail: Optional[str] = None,
    output_type: Optional[str] = None,
    output_format: Optional[str] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    .. image:: _static/images/generate_service_areas/generate_service_areas.png

    This ``generate_service_areas`` tool determines network service areas around facilities. A network service area is a region that
    encompasses all streets that can be accessed within a given distance or travel time from one or more facilities. For instance, the
    10-minute service area for a facility includes all the streets that can be reached within 10 minutes from that facility. Service areas
    are commonly used to visualize and measure accessibility. For example, a three-minute drive-time polygon around a grocery store can
    determine which residents are able to reach the store within three minutes and are thus more likely to shop there.

    =================================================     ========================================================================
    **Parameter**                                          **Description**
    -------------------------------------------------     ------------------------------------------------------------------------
    facilities                                            Required :class:`~arcgis.features.FeatureSet` . The facilities around which service areas are
                                                          generated. You can load up to 1,000 facilities. The facilities feature set has an
                                                          associated attribute table. The fields in the attribute table are listed and
                                                          described below:

                                                          * ``ObjectID``: The system-managed ID field.
                                                          * ``Name``:  The name of the facility. If the name is not specified, a name is automatically
                                                            generated at solve time.

                                                          All fields from the input facilities are included in the output
                                                          polygons when the Polygons for Multiple Facilities parameter is set to Overlapping or Not
                                                          Overlapping. The ObjectID field on the input facilities is transferred to the ``FacilityOID`` field
                                                          on the output polygons.
    -------------------------------------------------     ------------------------------------------------------------------------
    break_values                                          Required string. Specifies the size and number of service area
                                                          polygons to generate for each facility. The units are determined by the Break Units value.

                                                          Multiple polygon breaks can be set to create concentric service areas per facility. For
                                                          instance, to find 2-, 3-, and 5-mile service areas for each facility, type 2 3 5, separating
                                                          the values with a space, and set Break Units to Miles. There is no limit to the number of
                                                          break values you specify.

                                                          The size of the maximum break value can't exceed the equivalent of 300 minutes or 300 miles
                                                          (482.80 kilometers). When generating detailed polygons, the maximum service-area size is
                                                          limited to 15 minutes and 15 miles (24.14 kilometers).
    -------------------------------------------------     ------------------------------------------------------------------------
    break_units                                           Required string. The unit for the Break Values parameter. The units
                                                          you choose for this parameter determine whether the tool will create service areas by
                                                          measuring driving distance or driving time. Choose a time unit to measure driving time. To
                                                          measure driving distance, choose a distance unit. Your choice also determines in which units
                                                          the tool will report total driving time or distance in the results. The choices are: Meters
                                                          Kilometers Feet Yards Miles Nautical Miles Seconds Minutes Hours Days

                                                          Choice list: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles', 'Nautical Miles', 'Seconds',
                                                          'Minutes', 'Hours', 'Days']
    -------------------------------------------------     ------------------------------------------------------------------------
    analysis_region                                       Optional string. Specify the region in which to perform the
                                                          analysis. If a value is not specified for this parameter, the tool will automatically
                                                          calculate the region name based on the location of the input points. Setting the name of the
                                                          region is recommended to speed up the tool execution.

                                                          Choice list: ['NorthAmerica', 'SouthAmerica', 'Europe', 'MiddleEastAndAfrica', 'India',
                                                          'SouthAsia', 'SouthEastAsia', 'Thailand', 'Taiwan', 'Japan', 'Oceania', 'Greece', 'Korea']
    -------------------------------------------------     ------------------------------------------------------------------------
    travel_direction                                      Optional string. Specifies whether the direction of travel used to
                                                          generate the service area polygons is toward or away from the facilities.

                                                          * ``Away From Facility`` - The service area is generated in the direction away from the facilities.

                                                          * ``Towards Facility`` - The service area is created in the direction towards the facilities.

                                                          The direction of travel may change the shape of the polygons because impedances on opposite
                                                          sides of streets may differ or one-way restrictions may exist, such as one-way streets. The
                                                          direction you should choose depends on the nature of your service area analysis. The service
                                                          area for a pizza delivery store, for example, should be created away from the facility,
                                                          whereas the service area of a hospital should be created toward the facility.

                                                          Choice list: ['Away From Facility', 'Towards Facility']
    -------------------------------------------------     ------------------------------------------------------------------------
    time_of_day                                           Optional datetime. The time to depart from or arrive at the
                                                          facilities. The interpretation of this value depends on whether travel is toward or away from
                                                          the facilities.
                                                          It represents the departure time if Travel Direction is set to Away from Facility.
                                                          It represents the arrival time if Travel Direction is set to Toward Facility.

                                                          You can use the Time Zone for Time of Day parameter to specify whether this time and date
                                                          refers to UTC or the time zone in which the facility is located.

                                                          Repeatedly solving the same analysis, but using different Time of Day values, allows you to see
                                                          how a facility's reach changes over time. For instance, the five-minute service area around a
                                                          fire station may start out large in the early morning, diminish during the morning rush hour,
                                                          grow in the late morning, and so on, throughout the day.
    -------------------------------------------------     ------------------------------------------------------------------------
    use_hierarchy                                         Optional boolean. Specify whether hierarchy should be used when
                                                          finding the best route between the facility and the incident.

                                                          Checked (True) - Use the hierarchy attribute for the analysis. Using a hierarchy results in the
                                                          solver preferring higher-order edges to lower-order edges. Hierarchical solves are faster, and
                                                          they can be used to simulate the preference of a driver who chooses to travel on freeways over
                                                          local roads when possible-even if that means a longer trip.

                                                          Unchecked (False) - Do not use the hierarchy attribute for the analysis. Not using a hierarchy
                                                          yields an accurate service area measured along all edges of the network dataset regardless of
                                                          hierarchy level.

                                                          Regardless of whether the Use Hierarchy parameter is checked (True), hierarchy is always used when
                                                          the largest break value exceeds 240 minutes or 240 miles (386.24 kilometers).
    -------------------------------------------------     ------------------------------------------------------------------------
    uturn_at_junctions                                    Optional string. Use this parameter to restrict or permit the service
                                                          area to make U-turns at junctions. In order to understand the parameter values, consider for a
                                                          moment the following terminology: a junction is a point where a street segment ends and potentially
                                                          connects to one or more other segments; a pseudo-junction is a point where exactly two streets
                                                          connect to one another; an intersection is a point where three or more streets connect; and a
                                                          dead-end is where one street segment ends without connecting to another. Given this information, the
                                                          parameter can have the following values:

                                                          ========================================  ================================================
                                                          **Parameter**                             **Description**
                                                          ----------------------------------------  ------------------------------------------------
                                                          ALLOW_UTURNS                              |ALLOW_UTURNS|
                                                                                                    U-turns are permitted everywhere. Allowing U-turns implies
                                                                                                    that the vehicle can turn around at a junction or intersection
                                                                                                    and double back on the same street.
                                                          ----------------------------------------  ------------------------------------------------
                                                          ALLOW_DEAD_ENDS_AND                       |ALLOW_DEAD_ENDS_AND_INTERSECTIONS_ONLY|
                                                          _INTERSECTIONS_ONLY                       U-turns are prohibited at junctions where exactly two
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

                                                          Choice list: ['Allowed', 'Not Allowed', 'Allowed Only at Dead Ends',
                                                          'Allowed Only at Intersections and Dead Ends']
    -------------------------------------------------     ------------------------------------------------------------------------
    polygons_for_multiple_facilities                      Optional string.  Choose how service area polygons are generated when multiple facilities are present in the analysis.

                                                          * ``Overlapping`` - Creates individual polygons for each facility. The polygons can overlap each other.
                                                            This is the default value.

                                                          * ``Not Overlapping`` - Creates individual polygons such that a polygon from one facility cannot
                                                            overlap polygons from other facilities; furthermore, any portion of the network can only be
                                                            covered by the service area of the nearest facility.
                                                          * ``Merge by Break Value`` - Creates and joins the polygons of different facilities that have the same
                                                            break value.

                                                          When using Overlapping or Not Overlapping, all fields from the input facilities are included in the
                                                          output polygons, with the exception that values from the input ObjectID field are transferred to the
                                                          FacilityOID field of the output polygons. The FacilityOID field is null when merging by break value,
                                                          and the input fields are not included in the output.

                                                          Choice list: ['Overlapping', 'Not Overlapping', 'Merge by Break Value']
    -------------------------------------------------     ------------------------------------------------------------------------
    polygon_overlap_type                                  Optional string. Specifies the option to create concentric service area
                                                          polygons as disks or rings. This option is applicable only when multiple break
                                                          values are specified for the facilities.

                                                          * ``Rings`` - The polygons representing larger breaks exclude the polygons of smaller breaks.
                                                            This creates polygons going between consecutive breaks. Use this option if you want to
                                                            find the area from one break to another. For instance, if you create 5- and 10-minute
                                                            service areas, then the 10-minute service area polygon will exclude the area under the
                                                            5-minute service area polygon. This is the default value.

                                                          * ``Disks`` - Creates polygons going from the facility to the break. For instance, if you
                                                            create 5- and 10-minute service areas, then the 10-minute service area polygon will
                                                            include the area under the 5-minute service area polygon.

                                                          Choice list: ['Rings', 'Disks']
    -------------------------------------------------     ------------------------------------------------------------------------
    detailed_polygons                                     Optional boolean. Specifies the option to create detailed or generalized
                                                          polygons.

                                                          Unchecked (False) - Creates generalized polygons, which are
                                                          generated quickly and are fairly accurate. This is the
                                                          default.

                                                          Checked (True) - Creates detailed polygons, which
                                                          accurately model the service area lines and may contain islands of
                                                          unreached areas. This option is much slower than generating
                                                          generalized polygons. This option isn't supported when using
                                                          hierarchy.
                                                          If your facilities are in an urban area with a grid-like street network,
                                                          the difference between generalized and detailed service areas would be
                                                          minimal. However, if your facilities are in a region containing mountain
                                                          and rural roads, the detailed service areas may present significantly
                                                          more accurate results than generalized service areas.

                                                          The tool supports generating detailed polygons only if the largest value specified in the Break Values
                                                          parameter is less than or equal to 15 minutes or 15 miles (24.14 kilometers).
    -------------------------------------------------     ------------------------------------------------------------------------
    polygon_trim_distance                                 Optional LinearUnit. Specifies the distance within which the service area
                                                          polygons are trimmed. This is useful when finding service areas in
                                                          places that have a sparse street network and you don't want the
                                                          service area to cover large areas where there are no street
                                                          features.

                                                          The default value is 100 meters. No value or a value of 0 for this parameter
                                                          specifies that the service area polygons should not be trimmed. This
                                                          parameter value is ignored when using hierarchy.
    -------------------------------------------------     ------------------------------------------------------------------------
    polygon_simplification_tolerance                      Optional LinearUnit. Specify by how much you want to
                                                          simplify the polygon geometry.

                                                          Simplification maintains critical vertices of a polygon to define its essential
                                                          shape and removes other vertices. The simplification distance you specify is the
                                                          maximum offset the simplified polygon boundaries can deviate from the original
                                                          polygon boundaries. Simplifying a polygon reduces the number of vertices and tends
                                                          to reduce drawing times.
    -------------------------------------------------     ------------------------------------------------------------------------
    point_barriers                                        Optional :class:`~arcgis.features.FeatureSet`  . Specify one or more points to act as temporary
                                                          restrictions or represent additional time or distance that may be required to travel on the
                                                          underlying streets. For example, a point barrier can be used to represent a fallen tree along a
                                                          street or time delay spent at a railroad crossing.

                                                          The tool imposes a limit of 250 points that can be added as barriers.
                                                          When specifying the point barriers, you can set properties for each one, such as its name or
                                                          barrier type, by using attributes. The point barriers can be specified with the following
                                                          attributes:

                                                          * ``Name``: The name of the barrier.
                                                          * ``BarrierType``: Specifies whether the point barrier restricts travel
                                                            completely or adds time or distance when it is crossed. The value
                                                            for this attribute is specified as one of the following
                                                            integers (use the numeric code, not the name in parentheses):

                                                            * 0 (Restriction) - Prohibits travel through the barrier. The barrier
                                                              is referred to as a restriction point barrier since it acts as a
                                                              restriction.

                                                            * 2 (Added Cost) - Traveling through the barrier increases the travel
                                                              time or distance by the amount specified in the Additional_Time or
                                                              Additional_Distance field. This barrier type is  referred to as an
                                                              added-cost point barrier.

                                                          * ``Additional_Time``: Indicates how much travel time is added when the
                                                            barrier is traversed. This field is applicable only for added-cost
                                                            barriers and only if the Break Units value is time based. This field
                                                            value must be greater than or equal to zero, and its units are the same
                                                            as those specified in the Break Units parameter.

                                                          * ``Additional_Distance``: Indicates how much distance is added when the
                                                            barrier is traversed. This field is applicable only for added-cost barriers
                                                            and only if the Break Units value is distance based. The field value
                                                            must be greater than or equal to zero, and its units are the same as those
                                                            specified in the Break Units parameter.
    -------------------------------------------------     ------------------------------------------------------------------------
    line_barriers                                         Optional :class:`~arcgis.features.FeatureSet`  . Specify one or more lines that prohibit travel
                                                          anywhere the lines intersect the streets. For example, a parade or protest that blocks traffic
                                                          across several street segments can be modeled with a line barrier. A line barrier can also
                                                          quickly fence off several roads from being traversed, thereby channeling possible routes away
                                                          from undesirable parts of the street network.

                                                          The tool imposes a limit on the number of streets you can
                                                          restrict using the Line Barriers parameter. While there is no limit on
                                                          the number of lines you can specify as line barriers, the combined
                                                          number of streets intersected by all the lines cannot exceed 500.

                                                          When specifying the line barriers, you can set a name property for each one by using the following
                                                          attribute:

                                                          * ``Name``: The name of the barrier.
    -------------------------------------------------     ------------------------------------------------------------------------
    polygon_barriers                                      Optional :class:`~arcgis.features.FeatureSet`  . Specify polygons that either completely restrict travel or
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
                                                          * ``BarrierType``: Specifies whether the barrier restricts travel completely or scales the time or distance for traveling through it. The field value is specified as one of the following integers (use the numeric code, not the name in parentheses):

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
    -------------------------------------------------     ------------------------------------------------------------------------
    restrictions                                          Optional string. Specify which restrictions should be honored by the tool when finding the best routes
                                                          between facilities and demand points. A restriction represents a driving
                                                          preference or requirement. In most cases, restrictions cause roads
                                                          to be prohibited. For instance, using an Avoid Toll Roads restriction will result in a route that will
                                                          include toll roads only when it is absolutely required to travel on toll roads in order to visit an
                                                          incident or a facility. Height Restriction makes it possible to route around any clearances that are
                                                          lower than the height of your vehicle. If you are carrying corrosive materials on your vehicle, using
                                                          the Any Hazmat Prohibited restriction prevents hauling the materials along roads where it is marked as
                                                          illegal to do so.
                                                          Below is a list of available restrictions and a short description.
                                                          Some restrictions require an additional value to be
                                                          specified for their desired use. This value needs to be associated
                                                          with the restriction name and a specific parameter intended to work
                                                          with the restriction. You can identify such restrictions if their
                                                          names appear under the AttributeName column in the Attribute
                                                          Parameter Values parameter. The ParameterValue field should be
                                                          specified in the Attribute Parameter Values parameter for the
                                                          restriction to be correctly used when finding traversable roads.
                                                          Some restrictions are supported only in certain countries; their availability is stated by region in
                                                          the list below. Of the restrictions that have limited availability within a region, you can check whether
                                                          the restriction is available in a particular country by looking at the table in the Country List section
                                                          of the Data coverage for network analysis services web page. If a country has a value of  Yes in the
                                                          Logistics Attribute column, the restriction with select availability in the region is supported in that country.
                                                          If you specify restriction names that are not available in the country where your incidents are located,
                                                          the service ignores the invalid restrictions. The service also ignores restrictions whose Restriction Usage
                                                          parameter value is between 0 and 1 (see the Attribute Parameter Value parameter). It prohibits all restrictions
                                                          whose Restriction Usage parameter value is greater than 0.
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

                                                          Choice list: ['Any Hazmat Prohibited', 'Avoid Carpool Roads', 'Avoid Express Lanes', 'Avoid Ferries',
                                                          'Avoid Gates', 'Avoid Limited Access Roads', 'Avoid Private Roads', 'Avoid Roads Unsuitable for Pedestrians',
                                                          'Avoid Stairways', 'Avoid Toll Roads', 'Avoid Toll Roads for Trucks', 'Avoid Truck Restricted Roads',
                                                          'Avoid Unpaved Roads', 'Axle Count Restriction', 'Driving a Bus', 'Driving a Delivery Vehicle', 'Driving a Taxi',
                                                          'Driving a Truck', 'Driving an Automobile', 'Driving an Emergency Vehicle', 'Height Restriction',
                                                          'Kingpin to Rear Axle Length Restriction', 'Length Restriction', 'Preferred for Pedestrians', 'Riding a Motorcycle',
                                                          'Roads Under Construction Prohibited', 'Semi or Tractor with One or More Trailers Prohibited',
                                                          'Single Axle Vehicles Prohibited', 'Tandem Axle Vehicles Prohibited', 'Through Traffic Prohibited',
                                                          'Truck with Trailers Restriction', 'Use Preferred Hazmat Routes', 'Use Preferred Truck Routes', 'Walking',
                                                          'Weight Restriction', 'Weight per Axle Restriction', 'Width Restriction']
    -------------------------------------------------     ------------------------------------------------------------------------
    attribute_parameter_values                            Optional :class:`~arcgis.features.FeatureSet`  . Specify additional values required by some
                                                          restrictions, such as the weight of a vehicle for Weight Restriction. You can also use the attribute
                                                          parameter to specify whether any restriction prohibits, avoids, or prefers travel on roads that use the
                                                          restriction. If the restriction is meant to avoid or prefer roads, you can further specify the degree
                                                          to which they are avoided or preferred using this parameter. For example, you can choose to never use
                                                          toll roads, avoid them as much as possible, or even highly prefer them.

                                                          The values you provide for this parameter are ignored unless Travel Mode is set to Custom. If you specify
                                                          the Attribute Parameter Values parameter from a  feature class, the field names on the feature class must
                                                          match the fields as described below:

                                                          * ``AttributeName``: Lists the name of the restriction.

                                                          * ``ParameterName``: Lists the name of the parameter associated with the
                                                            restriction. A restriction can have one or more ParameterName field
                                                            values based on its intended use.

                                                          * ``ParameterValue``: The value for ParameterName used by the tool
                                                            when evaluating the restriction.

                                                            Attribute Parameter Values is dependent on the Restrictions parameter. The ParameterValue field is
                                                            applicable only if the restriction name is specified as the value for the Restrictions parameter.

                                                            In Attribute Parameter Values, each restriction (listed as AttributeName) has a ParameterName field
                                                            value, Restriction Usage, that specifies whether the restriction prohibits, avoids, or prefers travel
                                                            on the roads associated with the restriction and the degree to which the roads are avoided or
                                                            preferred. The Restriction Usage ParameterName can be assigned any of the following string values or
                                                            their equivalent numeric values listed within the parentheses:

                                                            *  PROHIBITED (-1) - Travel on the roads using the restriction is completely prohibited.
                                                            *  AVOID_HIGH (5) - It is highly unlikely for the tool to include in the route the roads that are associated with the restriction.
                                                            *  AVOID_MEDIUM (2) - It is unlikely for the tool to include in the route the roads that are associated with the restriction.
                                                            *  AVOID_LOW (1.3) - It is somewhat unlikely for the tool to include in the route the roads that are associated with the restriction.
                                                            *  PREFER_LOW (0.8) - It is somewhat likely for the tool to include in the route the roads that are associated with the restriction.
                                                            *  PREFER_MEDIUM (0.5) - It is likely for the tool to include in the route the roads that are associated with the restriction.
                                                            *  PREFER_HIGH (0.2) - It is highly likely for the tool to include in the route the roads that are associated with the restriction.

                                                            In most cases, you can use the default value, PROHIBITED, for the Restriction Usage if the restriction
                                                            is dependent on a vehicle-characteristic such as vehicle height. However, in some cases, the value for
                                                            Restriction Usage depends on your routing preferences. For example, the Avoid Toll Roads restriction
                                                            has the default value of AVOID_MEDIUM for the Restriction Usage parameter. This means that when the
                                                            restriction is used, the tool will try to route around toll roads when it can. AVOID_MEDIUM also indicates
                                                            how important it is to avoid toll roads when finding the best route; it has a medium priority. Choosing
                                                            AVOID_LOW would put lower importance on avoiding tolls; choosing AVOID_HIGH instead would give it a higher
                                                            importance and thus make it more acceptable for the service to generate longer routes to avoid tolls.
                                                            Choosing PROHIBITED would entirely disallow travel on toll roads, making it impossible for a route to
                                                            travel on any portion of a toll road. Keep in mind that avoiding or prohibiting toll roads, and thus
                                                            avoiding toll payments, is the objective for some; in contrast, others prefer to drive on toll roads
                                                            because avoiding traffic is more valuable to them than the money spent on tolls. In the latter case, you
                                                            would choose PREFER_LOW, PREFER_MEDIUM, or PREFER_HIGH as the value for Restriction Usage. The higher the
                                                            preference, the farther the tool will go out of its way to travel on the roads associated with the
                                                            restriction.
    -------------------------------------------------     ------------------------------------------------------------------------
    time_zone_for_time_of_day                             Optional string. Specifies the time zone or zones of the Time of Day
                                                          parameter.

                                                          * ``Geographically Local``: The Time of Day parameter refers to the time zone or zones in which the facilities
                                                            are located. Therefore, the start or end times of the service areas are staggered by time zone. Setting
                                                            Time of Day to 9:00 a.m., choosing geographically local for Time Zone for Time of Day, and solving causes
                                                            service areas to be generated for 9:00 a.m. Eastern Time for any facilities in the Eastern Time Zone,
                                                            9:00 a.m. Central Time for facilities in the Central Time Zone, 9:00 a.m. Mountain Time for facilities in
                                                            the Mountain Time Zone, and so on, for facilities in different time zones. If stores in a chain that span
                                                            the U.S. open at 9:00 a.m. local time, this parameter value could be chosen to find market territories at
                                                            opening time for all stores in one solve. First, the stores in the Eastern Time Zone open and a polygon is
                                                            generated, then an hour later stores open in Central Time, and so on. Nine o'clock is always in local time
                                                            but staggered in real time.

                                                          * ``UTC``: The Time of Day parameter refers to Coordinated Universal Time (UTC).
                                                            Therefore, all facilities are reached or departed from simultaneously, regardless of the time zone each is
                                                            in. Setting Time of Day to 2:00 p.m., choosing UTC, then solving causes service areas to be generated for
                                                            9:00 a.m. Eastern Standard Time for any facilities in the Eastern Time Zone, 8:00 a.m. Central Standard
                                                            Time for facilities in the Central Time Zone, 7:00 a.m. Mountain Standard Time for facilities in the
                                                            Mountain Time Zone, and so on, for facilities in different time zones. The scenario above assumes standard
                                                            time. During daylight saving time, the Eastern, Central, and Mountain Times would each be one hour ahead
                                                            (that is, 10:00, 9:00, and 8:00 a.m., respectively). One of the cases in which the UTC option is useful is
                                                            to visualize emergency-response coverage for a jurisdiction that is split into two time zones. The emergency
                                                            vehicles are loaded as facilities. Time of Day is set to now in UTC. (You need to determine what  the current
                                                            time and date are  in terms of UTC to correctly use this option.) Other properties are set and the analysis
                                                            is solved. Even though a time-zone boundary divides the vehicles, the results show areas that can be reached
                                                            given current traffic conditions. This same process  can be used for other times as well, not just for now.

                                                          Irrespective of the Time Zone for Time of Day setting, all facilities must be in the same time zone
                                                          when Time of Day has a nonnull value and Polygons for Multiple Facilities is set to create merged or
                                                          nonoverlapping polygons.

                                                          Choice list: ['Geographically Local', 'UTC']
    -------------------------------------------------     ------------------------------------------------------------------------
    travel_mode                                           Optional string.  Specify the mode of transportation to model in the analysis. Travel
                                                          modes are managed in ArcGIS Online and can be configured by the administrator of your organization to better
                                                          reflect your organization's workflows. You need to specify the name of a travel mode supported by your
                                                          organization.

                                                          To get a list of supported travel mode names, run the GetTravelModes tool from the Utilities toolbox
                                                          available under the same GIS Server connection you used to access the tool. The GetTravelModes tool
                                                          adds a table, Supported Travel Modes, to the application. Any value in the Travel Mode Name field from the
                                                          Supported Travel Modes table can be specified as input. You can also specify the value from Travel Mode
                                                          Settings field as input. This speeds up the tool execution as the tool does not have to lookup the settings
                                                          based on the travel mode name.

                                                          The default value, Custom, allows you to configure your own travel mode using the custom travel mode
                                                          parameters (UTurn at Junctions, Use Hierarchy, Restrictions, Attribute Parameter Values,  and Impedance).
                                                          The default values of the custom travel mode parameters model travelling by car. You may want to choose
                                                          Custom and set the custom travel mode parameters listed above to model a pedestrian with a fast walking speed
                                                          or a truck with a given height, weight, and cargo of certain hazardous materials. You may choose to do this
                                                          to try out different settings to get desired analysis results. Once you have identified the analysis settings,
                                                          you should work with your organization's administrator and save these settings as part of new or existing
                                                          travel mode so that everyone in your organization can rerun the analysis with the same settings.
    -------------------------------------------------     ------------------------------------------------------------------------
    impedance                                             Optional string. Specify the impedance, which is a value that represents the effort or cost
                                                          of traveling along road segments or on other parts of the transportation network.
                                                          Travel distance is an impedance; the length of a road in kilometers can be thought of as impedance. Travel
                                                          distance in this sense is the same for all modes-a kilometer for a pedestrian is also a kilometer for a car.
                                                          (What may change is the pathways on which the different modes are allowed to travel, which affects distance
                                                          between points, and this is modeled by travel mode settings.) Travel time can also be an impedance; a car
                                                          may take one minute to travel a mile along an empty road. Travel times can vary by travel mode-a pedestrian
                                                          may take more than 20  minutes to walk the same mile, so it is important to choose the right impedance for
                                                          the travel mode you are modeling.  Choose from the following impedance values: Drive Time-Models travel times
                                                          for a car. These travel times are dynamic and fluctuate according to traffic flows in areas where traffic data
                                                          is available. This is the default value. Truck Time-Models travel times for a truck.  These travel times are
                                                          static for each road and don't fluctuate with traffic. Walk Time-Models travel times for a pedestrian. Travel
                                                          Distance-Stores  length measurements along roads and paths. To model walk distance, choose this option and
                                                          ensure Walking is  set in the Restriction parameter. Similarly, to model drive or truck distance, choose Travel
                                                          Distance here and set the appropriate restrictions so your vehicle travels only on roads where it is permitted
                                                          to do so. The value you provide for this parameter is ignored unless Travel Mode is set to Custom, which is the
                                                          default value. If you choose Drive Time, Truck Time, or Walk Time, the Measurement Units parameter must be set
                                                          to a time-based value; if you choose Travel Distance for Impedance, Measurement Units must be distance-based.

                                                          Choice list:['Drive Time', 'Truck Time', 'Walk Time', 'Travel Distance']
    -------------------------------------------------     ------------------------------------------------------------------------
    save_output_network_analysis_layer                    Optional boolean. Specify if the tool should save the analysis settings as a network analysis layer file.
                                                          You cannot directly work with this file even when you open the file in an ArcGIS Desktop application like ArcMap.
                                                          It is meant to be sent to Esri Technical Support to diagnose the quality of results returned from the tool.
                                                          True: Save the network analysis layer file. The file is downloaded in a temporary directory on your machine.
                                                          In ArcGIS Pro, the location of the downloaded file can be determined by viewing the value for the Output Network Analysis
                                                          Layer parameter in the entry corresponding to the tool execution in the Geoprocessing history of your Project. In ArcMap,
                                                          the location of the file can be determined by accessing the Copy Location option in the shortcut menu on the Output Network
                                                          Analysis Layer parameter in the entry corresponding to the tool execution in the Geoprocessing Results window.
                                                          False: Do not save the network analysis layer file. This is the default.
    -------------------------------------------------     ------------------------------------------------------------------------
    overrides                                             Optional string. Specify additional settings that can influence the behavior of the solver when finding solutions
                                                          for the network analysis problems. The value for this parameter needs to be specified in dict. For example, a valid value is
                                                          of the following form {"overrideSetting1" : "value1", "overrideSetting2" :
                                                          "value2"}. The override setting name is always enclosed in double quotes. The values can be a number, Boolean,
                                                          or string. The default value for this parameter is no value, which indicates not to override any solver settings. Overrides
                                                          are advanced settings that should be used only after careful analysis of the results obtained before and after applying
                                                          the settings. A list of supported override settings for each solver and their acceptable values can be obtained by contacting
                                                          Esri Technical Support.
    -------------------------------------------------     ------------------------------------------------------------------------
    time_impedance                                        Optional string. Specify the time-based impedance.
    -------------------------------------------------     ------------------------------------------------------------------------
    distance_impedence                                    Optional string. Specify the distance-based impedance.
    -------------------------------------------------     ------------------------------------------------------------------------
    polygon_detail                                        Optional string. Specify the detail of the polygon you want to create.

                                                          Choice list: ["Generalized", "Standard", "High"]
    -------------------------------------------------     ------------------------------------------------------------------------
    output_format                                         Optional. Specify the format in which the output features are created.

                                                          Choose from the following formats:

                                                          * Feature Set - The output features are returned as feature classes and tables. This is the default.
                                                          * JSON File - The output features are returned as a compressed file containing the
                                                            JSON representation of the outputs. When this option is specified, the output is a single
                                                            file (with a .zip extension) that contains one or more JSON files (with a .json extension)
                                                            for each of the outputs created by the service.
                                                          * GeoJSON File - The output features are returned as a compressed file containing the GeoJSON
                                                            representation of the outputs. When this option is specified, the output is a single file
                                                            (with a .zip extension) that contains one or more GeoJSON files (with a .geojson extension)
                                                            for each of the outputs created by the service.
    -------------------------------------------------     ------------------------------------------------------------------------
    gis                                                   Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    -------------------------------------------------     ------------------------------------------------------------------------
    future                                                Optional boolean. If True, a future object will be returned and the process
                                                          will not wait for the task to complete. The default is False, which means wait for results.
    =================================================     ========================================================================

    :return: the following as a named tuple:

    * service_areas - Service Areas as a FeatureSet
    * solve_succeeded - Solve Succeeded as a boolean

    Click `GenerateServiceAreas <https://developers.arcgis.com/rest/network/api-reference/service-area-asynchronous-service.htm>`_ for additional help.

    .. code-block:: python

            # Usage Example: To determine network service areas around facilities at time breaks of 5, 10, 15 min of drive time.

            agg_result = generate_service_areas(facilities=facilities,
                                                break_values=[5, 10, 15],
                                                break_units="Minutes",
                                                time_of_day=current_time)
    """

    from arcgis.geoprocessing import import_toolbox

    if gis is None:
        gis = arcgis.env.active_gis

    url = gis.properties.helperServices.asyncServiceArea.url[
        : -len("/GenerateServiceAreas")
    ]
    url = _validate_url(url, gis)
    tbx = _create_toolbox(url, gis)
    defaults = dict(
        zip(
            tbx.generate_service_areas.__annotations__.keys(),
            tbx.generate_service_areas.__defaults__,
        )
    )
    if facilities is None:
        facilities = defaults["facilities"]
    if polygon_trim_distance is None:
        polygon_trim_distance = defaults["polygon_trim_distance"]
    if polygon_simplification_tolerance is None:
        polygon_simplification_tolerance = defaults["polygon_simplification_tolerance"]
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
    if overrides is None:
        overrides = defaults["overrides"]
    if time_impedance is None and "time_impedance" in defaults:
        time_impedance = defaults["time_impedance"]
    if distance_impedance is None and "distance_impedance" in defaults:
        distance_impedance = defaults["distance_impedance"]
    if polygon_detail is None:
        polygon_detail = defaults.get("polygon_detail", None)
    if output_type is None:
        output_type = defaults.get("output_type", None)
    if output_format is None:
        output_format = defaults.get("output_format", None)

    if isinstance(break_values, list):
        break_values = " ".join(map(str, break_values))
    from arcgis._impl.common._utils import inspect_function_inputs

    params = {
        "facilities": facilities,
        "break_values": break_values,
        "break_units": break_units,
        "analysis_region": analysis_region,
        "travel_direction": travel_direction,
        "time_of_day": time_of_day,
        "use_hierarchy": use_hierarchy,
        "uturn_at_junctions": uturn_at_junctions,
        "polygons_for_multiple_facilities": polygons_for_multiple_facilities,
        "polygon_overlap_type": polygon_overlap_type,
        "detailed_polygons": detailed_polygons,
        "polygon_trim_distance": polygon_trim_distance,
        "polygon_simplification_tolerance": polygon_simplification_tolerance,
        "point_barriers": point_barriers,
        "line_barriers": line_barriers,
        "polygon_barriers": polygon_barriers,
        "restrictions": restrictions,
        "attribute_parameter_values": attribute_parameter_values,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "travel_mode": travel_mode,
        "impedance": impedance,
        "save_output_network_analysis_layer": save_output_network_analysis_layer,
        "overrides": overrides,
        "time_impedance": time_impedance,
        "distance_impedance": distance_impedance,
        "polygon_detail": polygon_detail,
        "output_type": output_type,
        "output_format": output_format,
        "gis": gis,
        "future": True,
    }
    params = inspect_function_inputs(tbx.generate_service_areas, **params)
    params["future"] = True
    job = tbx.generate_service_areas(**params)
    if future:
        return job
    return job.result()


generate_service_areas.__annotations__ = {
    "facilities": FeatureSet,
    "break_values": str,
    "break_units": str,
    "analysis_region": str,
    "travel_direction": str,
    "time_of_day": datetime,
    "use_hierarchy": bool,
    "uturn_at_junctions": str,
    "polygons_for_multiple_facilities": str,
    "polygon_overlap_type": str,
    "detailed_polygons": bool,
    "polygon_trim_distance": LinearUnit,
    "polygon_simplification_tolerance": LinearUnit,
    "point_barriers": FeatureSet,
    "line_barriers": FeatureSet,
    "polygon_barriers": FeatureSet,
    "restrictions": str,
    "attribute_parameter_values": FeatureSet,
    "time_zone_for_time_of_day": str,
    "travel_mode": str,
    "impedance": str,
    "return": tuple,
}
