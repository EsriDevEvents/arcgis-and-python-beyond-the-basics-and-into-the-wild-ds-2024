import logging as _logging
from typing import Optional
import arcgis
from datetime import datetime
from arcgis.features import FeatureSet
from arcgis.gis import GIS
from arcgis.mapping import MapImageLayer
from arcgis.geoprocessing import DataFile, LinearUnit, RasterData
from arcgis.geoprocessing import import_toolbox
from arcgis._impl.common._utils import _validate_url
from ._routing_utils import _create_toolbox

_log = _logging.getLogger(__name__)

_use_async = True

default_origins = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
        {
            "alias": "Target Destination Count",
            "name": "TargetDestinationCount",
            "type": "esriFieldTypeInteger",
        },
        {"alias": "Cutoff", "name": "Cutoff", "type": "esriFieldTypeDouble"},
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

default_destinations = {
    "fields": [
        {"alias": "OBJECTID", "name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"alias": "Name", "name": "Name", "type": "esriFieldTypeString", "length": 128},
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
    "features": [],
    "displayFieldName": "",
    "exceededTransferLimit": False,
}


def generate_origin_destination_cost_matrix(
    origins: FeatureSet,
    destinations: FeatureSet,
    travel_mode: str = "Custom",
    time_units: str = "Minutes",
    distance_units: str = "Kilometers",
    analysis_region: Optional[str] = None,
    number_of_destinations_to_find: Optional[int] = None,
    cutoff: Optional[float] = None,
    time_of_day: Optional[datetime] = None,
    time_zone_for_time_of_day: str = "Geographically Local",
    point_barriers: Optional[FeatureSet] = None,
    line_barriers: Optional[FeatureSet] = None,
    polygon_barriers: Optional[FeatureSet] = None,
    uturn_at_junctions: str = "Allowed Only at Intersections and Dead Ends",
    use_hierarchy: bool = True,
    restrictions: Optional[str] = None,
    attribute_parameter_values: Optional[FeatureSet] = None,
    impedance: str = "Drive Time",
    origin_destination_line_shape: str = "None",
    save_output_network_analysis_layer: bool = False,
    overrides: Optional[dict] = None,
    time_impedance: Optional[str] = None,
    distance_impedance: Optional[str] = None,
    output_format: Optional[str] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """

    The ``generate_origin_destination_cost_matrix`` tool creates an origin-destination (OD) cost matrix from multiple origins to multiple destinations.
    An OD cost matrix is a table that contains the travel time and travel distance from each origin to each destination. Additionally, it ranks the
    destinations that each origin connects to in ascending order based on the minimum time or distance required to travel from that origin to each
    destination. The best path on the street network is discovered for each origin-destination pair, and the travel times and travel distances are
    stored as attributes of the output lines. Even though the lines are straight for performance reasons, they always store the travel time and travel
    distance along the street network, not straight-line distance.

    ======================================  ==========================================================================================================================================
    **Parameter**                            **Description**
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    origins                                 Required :class:`~arcgis.features.FeatureSet` . Specify locations that function as starting points in generating the paths to destinations.
                                            You can add up to 200 origins.
                                            When specifying the origins, you can set properties for each one, such as its name or the number of destinations
                                            to find from the origin, by using attributes. The origins can be specified with the following attributes:

                                            * ``Name`` - The name of the origin. The name can be an unique identifier for the origin. The name is included in the output lines (as the OriginName field) and in the output origins (as the Name field) and can be used to join additional information from the tool outputs to the attributes of your origins. If the name is not specified, a unique name prefixed with Location is automatically generated in the output origins. An auto-generated origin name is not included in the output lines.
                                            * ``TargetDestinationCount``-The maximum number of
                                              destinations that must be found for the origin. If a value is not specified, the value from the Number of Destinations
                                              to Find parameter is used. Cutoff-Specify the travel time or travel distance value at which to stop searching for
                                              destinations from the origin. Any destination beyond the cutoff value will not be considered.  The value needs to be
                                              in the units specified by the Time Units parameter if the impedance attribute in your travel mode is time based or in
                                              the units specified by the Distance Units parameter if the impedance attribute in your travel mode is distance based.
                                              If a value is not specified, the value from the Cutoff parameter is used.
                                            * ``CurbApproach`` - Specifies the direction a vehicle may depart from the origin. The field value is specified as one of the
                                              following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Either side of vehicle)-The vehicle can depart the origin in either direction, so a U-turn is allowed at the origin. This setting can be chosen if it is possible and practical for your vehicle to turn around at the origin. This decision may depend on the width of the road and the amount of traffic or whether the origin has a parking lot where vehicles can enter and turn around.
                                              * 1 ( Right side of vehicle)-When the vehicle departs the origin, the origin must be on the right side of the vehicle. A U-turn is prohibited. This is typically used for vehicles such as buses that must depart from the bus stop on the right-hand side.
                                              * 2 (Left side of vehicle)-When the vehicle departs the origin, the curb must be on the left side of the vehicle. A U-turn is prohibited. This is typically used for vehicles such as buses that must depart from the bus stop on the left-hand side.
                                              * 3 (No U-Turn)-For this tool, the No U-turn (3) value functions the same as Either side of vehicle. The CurbApproach property is designed to work with both kinds of national driving standards: right-hand traffic (United States) and left-hand traffic (United Kingdom). First, consider an origin on the left side of a vehicle. It is always on the left side regardless of whether the vehicle travels on the left or right half of the road. What may change with national driving standards is your decision to depart the origin from one of two directions, that is, so it ends up on the right or left side of the vehicle. For example, if you want to depart from an origin and not have a lane of traffic between the vehicle and the origin, you would choose Right side of vehicle (1) in the United States but Left side of vehicle (2) in the United Kingdom.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    destinations                            Required :class:`~arcgis.features.FeatureSet` . Specify locations that function as ending points in generating the paths from origins.
                                            You can add up to 200 destinations. When specifying the destinations, you can set properties for each one, such as its name,
                                            by using attributes. The destinations can be specified with the following attributes:

                                            * ``Name`` - The name of the destination. The name can be an unique identifier for the destination. The name is included in
                                              the output lines (as the DestinationName field) and in the output destinations (as the Name field) and can be used to join
                                              additional information from the tool outputs to the attributes of your destinations.
                                              If the name is not specified, a unique name prefixed with Location is automatically generated in the output destinations.
                                              An auto-generated destination name is not included in the output lines.
                                            * ``CurbApproach`` - Specifies the direction a vehicle may arrive at the destination. The field value is specified as one of the following
                                              integers (use the numeric code, not the name in parentheses):

                                              * 0 (Either side of vehicle)- The vehicle can arrive the destination in either direction, so a U-turn is allowed at the destination. This setting can be chosen if it is possible and practical for your vehicle to turn around at the destination. This decision may depend on the width of the road and the amount of traffic or whether the destination has a parking lot where vehicles can enter and turn around.
                                              * 1 ( Right side of vehicle)- When the vehicle arrives at the destination, the destination must be on the right side
                                                of the vehicle. A U-turn is prohibited. This is typically used for vehicles such as buses that must arrive at the bus stop on the
                                                right-hand side.
                                              * 2 (Left side of vehicle)-When the vehicle arrives at the destination, the curb must be on the left side of the vehicle.
                                                A U-turn is prohibited. This is typically used for vehicles such as buses that must arrive at the bus stop on the left-hand side.
                                              * 3 (No U-Turn)-For this tool, the No U-turn (3) value functions the same as Either side of vehicle. The CurbApproach property is
                                                designed to work with both kinds of national driving standards: right-hand traffic (United States) and left-hand traffic (United Kingdom).
                                                First, consider a destination on the left side of a vehicle. It is always on the left side regardless of whether the vehicle travels
                                                on the left or right half of the road. What may change with national driving standards is your decision to arrive at the destination
                                                from one of two directions, that is, so it ends up on the right or left side of the vehicle. For example, if you want to arrive at the
                                                destination and not have a lane of traffic between the vehicle and the destination, you would choose Right side of vehicle (1) in the
                                                United States but Left side of vehicle (2) in the United Kingdom.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    travel_mode                             Optional string. Specify the mode of transportation to model in the analysis. Travel modes are managed in ArcGIS Online and can be configured by the administrator of your
                                            organization to better reflect your organization's workflows. You need to specify the name of a travel mode supported by your organization.

                                            To get a list of supported travel mode names, run the GetTravelModes tool from the Utilities toolbox available under the same GIS Server connection
                                            you used to access the tool. The GetTravelModes tool adds a table, Supported Travel Modes, to the application. Any value in the Travel Mode Name field from the
                                            Supported Travel Modes table can be specified as input. You can also specify the value from Travel Mode Settings field as input. This speeds up the tool execution as the
                                            tool does not have to lookup the settings based on the travel mode name.

                                            The default value, Custom, allows you to configure your own travel mode using the custom travel mode parameters (UTurn at Junctions, Use Hierarchy, Restrictions, Attribute Parameter Values,  and Impedance).
                                            The default values of the custom travel mode parameters model travelling by car. You may want to choose Custom and set the custom travel mode parameters listed above to model a pedestrian with a fast walking speed
                                            or a truck with a given height, weight, and cargo of certain hazardous materials. You may choose to do this to try out different settings to get desired analysis results.
                                            Once you have identified the analysis settings, you should work with your organization's administrator and save these settings as part of new or existing travel mode so that
                                            everyone in your organization can rerun the analysis with the same settings.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_units                              Optional string. Specify the units that should be used to measure and report the total travel time between each origin-destination pair.

                                            Choice list:['Seconds', 'Minutes', 'Hours', 'Days']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    distance_units                          Optional string. Specify the units that should be used to measure and report the total travel distance between each origin-destination pair.

                                            Choice list:['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles', 'NauticalMiles']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    analysis_region                         Optional string. Specify the region in which to perform the analysis. If a value is not specified for this parameter, the tool
                                            will automatically calculate the region name based on the location
                                            of the input points. Setting the name of the region is recommended to speed up the
                                            tool execution.

                                            Choice list:['NorthAmerica', 'SouthAmerica', 'Europe', 'MiddleEastAndAfrica', 'India', 'SouthAsia', 'SouthEastAsia', 'Thailand', 'Taiwan', 'Japan', 'Oceania', 'Greece', 'Korea']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    number_of_destinations_to_find          Optional integer. Specify the maximum number of destinations to find per origin. If a value for this parameter is
                                            not specified, the output matrix includes travel costs from each origin to every destination. Individual origins
                                            can have their own values (specified as the TargetDestinationCount field) that override the Number of Destinations
                                            to Find parameter value.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    cutoff                                  Optional float. Specify the travel time or travel distance value at which to stop searching for destinations
                                            from a given origin. Any destination beyond the cutoff value will not be considered. Individual origins can have
                                            their own values (specified as the Cutoff field) that override the Cutoff parameter value.

                                            The value needs to be in the units specified by the Time Units parameter if the impedance attribute of your travel mode is time based or
                                            in the units specified by the Distance Units parameter if the impedance attribute of your travel mode is distance based.
                                            If a value is not specified, the tool will not enforce any travel time or travel distance limit when searching for destinations.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_of_day                             Optional datetime. Specifies the time and date at which the routes should
                                            begin. If you are modeling the driving  travel mode and specify the current date and time as the value
                                            for this parameter, the tool will use live traffic conditions to
                                            find the best routes and the total travel time will be based
                                            on traffic conditions.

                                            Specifying a time of day results in more accurate
                                            routes and estimations of travel times because the
                                            travel times account for the traffic conditions that are applicable
                                            for that date and time.
                                            The Time Zone for Time of Day parameter specifies whether this time and date refer to UTC or the time zone in which the stop is located. The
                                            tool ignores this parameter when Measurement Units isn't set to a time-based unit.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_zone_for_time_of_day               Optional string. Specifies the time zone of the Time of Day parameter.

                                            * ``Geographically Local``: The Time of Day parameter refers to the time zone in which the first stop of a route is located.  If you are generating many
                                              routes that start in multiple times zones,  the start times are staggered in Coordinated Universal Time (UTC). For example, a Time of Day value of
                                              10:00 a.m., 2 January, would mean a start time of  10:00 a.m. Eastern Standard Time (3:00 p.m. UTC) for routes beginning in the Eastern Time Zone
                                              and 10:00 a.m. Central Standard Time (4:00 p.m. UTC) for routes beginning in the Central Time Zone. The start times are offset by one hour in UTC.
                                              The arrive and depart times and dates recorded in the output Stops feature class will refer to the local time zone of the first stop for each route.
                                            * ``UTC``: The Time of Day parameter refers to Coordinated Universal Time (UTC). Choose this option if you want to generate a route for a specific time,
                                              such as now, but aren't certain in which time zone the first stop will be located. If you are generating many routes spanning multiple times zones,
                                              the start times in UTC are simultaneous. For example, a Time of Day value of 10:00 a.m., 2 January, would mean a start time of  5:00 a.m. Eastern Standard
                                              Time (UTC-5:00) for routes beginning in the Eastern Time Zone and 4:00 a.m. Central Standard Time (UTC-6:00) for routes beginning in the Central Time
                                              Zone. Both routes would start at 10:00 a.m. UTC. The arrive and depart times and dates recorded in the output Stops feature class will refer to UTC.

                                            Choice list:['Geographically Local', 'UTC']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    point_barriers                          Optional :class:`~arcgis.features.FeatureSet`  . Specify one or more points to act as temporary restrictions or represent additional time or
                                            distance that may be required to travel on the underlying streets. For example, a point
                                            barrier can be used to represent a fallen tree along a street or time delay spent at a railroad crossing.

                                            The tool imposes a limit of 250 points that can be added as barriers.
                                            When specifying the point barriers, you can set properties for each one, such as its name or barrier type,
                                            by using attributes. The point barriers can be specified with the following attributes:

                                            * ``Name``: The name of the barrier.
                                            * ``BarrierType``: Specifies whether the point barrier restricts travel completely or adds time or distance when it is crossed. The value for this attribute is specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Restriction)-Prohibits travel through the barrier. The barrier
                                                is referred to as a restriction point barrier since it acts as a
                                                restriction.
                                              * 2 (Added Cost)-Traveling through the barrier increases the travel
                                                time or distance by the amount specified in the ``Additional_Time`` or ``Additional_Distance`` field. This barrier type is referred to as an added-cost point barrier.

                                            * ``Additional_Time``: Indicates how much travel time is added when the barrier is traversed. This field is applicable only for added-cost barriers and only if the measurement units are time based. This field value must be greater than or equal to zero, and its units are the same as those specified in the Measurement Units parameter.
                                            * ``Additional_Distance``: Indicates how much distance is added when the barrier is traversed. This field is applicable only for added-cost barriers and only if the measurement units are distance based. The field value must be greater than or equal to zero, and its units are the same as those specified in the Measurement Units parameter.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    line_barriers                           Optional :class:`~arcgis.features.FeatureSet`  . Specify one or more lines that prohibit travel anywhere
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
                                            When specifying the polygon barriers, you can set properties for each one, such as its name or barrier type,
                                            by using attributes. The polygon barriers can be specified with the following attributes:

                                            * ``Name``: The name of the barrier.
                                            * ``BarrierType``: Specifies whether the barrier restricts travel completely or scales the time or distance for traveling through it. The field value is specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Restriction)-Prohibits traveling through any part of the barrier. The barrier is referred to as a restriction polygon barrier since it prohibits traveling on streets intersected by the barrier. One use of this type of barrier is to model floods covering areas of the street that make traveling on those streets impossible.
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
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    uturn_at_junctions                      Optional string. The U-Turn policy at junctions. Allowing U-turns implies the solver can turn around at a
                                            junction and double back on the same street.

                                            Given that junctions represent street intersections and dead ends, different vehicles  may be able to turn around at
                                            some junctions but not at others-it depends on whether the junction represents an intersection or dead end.
                                            To accommodate, the U-turn policy parameter is implicitly specified by how many edges, or streets, connect to the
                                            junction, which is known as junction valency. The acceptable values for this parameter are listed below; each is
                                            followed by a description of its meaning in terms of junction valency.

                                            Choice list:['Allowed', 'Not Allowed', 'Allowed Only at Dead Ends', 'Allowed Only at Intersections and Dead Ends']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    use_hierarchy                           Optional boolean. Specify whether hierarchy should be used when finding the shortest paths between stops.
                                            Checked (True): Use hierarchy when finding routes. When hierarchy is used, the tool prefers higher-order  streets (such as
                                            freeways) to lower-order streets (such as local roads), and can be used
                                            to simulate the driver preference of traveling on freeways instead
                                            of local roads even if that means a longer trip. This is especially
                                            true when finding routes to faraway locations, because drivers on long-distance trips tend to prefer traveling on
                                            freeways where stops, intersections, and turns can be avoided. Using hierarchy is computationally faster,
                                            especially for long-distance routes, since the tool can determine the
                                            best route from a relatively smaller subset of streets.
                                            Unchecked (False):
                                            Do not use hierarchy when finding routes. If
                                            hierarchy is not used, the tool considers all the streets and doesn't
                                            prefer higher-order streets when finding the route. This is often
                                            used when finding short routes within a city.

                                            The tool automatically reverts to using hierarchy if the
                                            straight-line distance between facilities and demand points is
                                            greater than 50 miles (80.46
                                            kilometers), even if you have set this parameter to not use hierarchy.
                                            This parameter is ignored unless Travel Mode is set to Custom. When modeling a custom walking mode,
                                            it is recommended to turn off hierarchy since the hierarchy is designed for motorized vehicles.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    restrictions                            Optional string. Specify which restrictions should be honored by the tool when finding the best routes.
                                            A restriction represents a driving preference or requirement. In most cases, restrictions cause roads to be
                                            prohibited, but they can also cause them to be avoided or preferred. For instance, using an Avoid Toll Roads
                                            restriction will result in a route that will include toll roads only when it is absolutely required to travel
                                            on toll roads in order to visit a stop. Height Restriction makes it possible to route around any clearances
                                            that are lower than the height of your vehicle. If you are carrying corrosive materials on your vehicle,
                                            using the Any Hazmat Prohibited restriction prevents hauling the materials along roads where it is marked
                                            as illegal to do so. The values you provide for this parameter are ignored unless Travel Mode is set to
                                            Custom. Below is a list of available restrictions and a short description.
                                            Some restrictions require an additional value to be specified for their desired use. This value needs to be associated
                                            with the restriction name and a specific parameter intended to work with the restriction. You can identify such restrictions if their
                                            names appear under the AttributeName column in the Attribute
                                            Parameter Values parameter. The ParameterValue field should be
                                            specified in the Attribute Parameter Values parameter for the
                                            restriction to be correctly used when finding traversable roads.
                                            Some restrictions are supported only in certain countries; their availability is stated by region in the list below.
                                            Of the restrictions that have limited availability within a region, you can check whether the restriction is available
                                            in a particular country by looking at the table in the Country List section of the Data coverage for network analysis
                                            services web page. If a country has a value of  Yes in the Logistics Attribute column, the restriction with select
                                            availability in the region is supported in that country. If you specify restriction names that are not available
                                            in the country where your incidents are located, the service ignores the invalid restrictions. The service also
                                            ignores restrictions whose Restriction Usage parameter value is between 0 and 1 (see the Attribute Parameter Value
                                            parameter). It prohibits all restrictions whose Restriction Usage parameter value is greater than 0.

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
                                            'Avoid Toll Roads', 'Avoid Toll Roads for Trucks', 'Avoid Truck Restricted Roads', 'Avoid Unpaved Roads',
                                            'Axle Count Restriction', 'Driving a Bus', 'Driving a Delivery Vehicle', 'Driving a Taxi', 'Driving a Truck',
                                            'Driving an Automobile', 'Driving an Emergency Vehicle', 'Height Restriction', 'Kingpin to Rear Axle Length Restriction',
                                            'Length Restriction', 'Preferred for Pedestrians', 'Riding a Motorcycle', 'Roads Under Construction Prohibited',
                                            'Semi or Tractor with One or More Trailers Prohibited', 'Single Axle Vehicles Prohibited', 'Tandem Axle Vehicles Prohibited',
                                            'Through Traffic Prohibited', 'Truck with Trailers Restriction', 'Use Preferred Hazmat Routes', 'Use Preferred Truck Routes',
                                            'Walking', 'Weight Restriction', 'Weight per Axle Restriction', 'Width Restriction']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    attribute_parameter_values              Optional :class:`~arcgis.features.FeatureSet`  . Specify additional values required by some restrictions, such as the weight of a vehicle
                                            for Weight Restriction. You can also use the attribute parameter to specify whether any restriction prohibits,
                                            avoids, or prefers travel on roads that use the restriction. If the restriction is
                                            meant to avoid or prefer roads, you can further specify the degree
                                            to which they are avoided or preferred using this
                                            parameter. For example, you can choose to never use toll roads, avoid them as much as possible, or even highly
                                            prefer them.
                                            The value you provide for this parameter is ignored unless Travel Mode is set to Custom, which is the default value.
                                            If you specify the Attribute Parameter Values parameter from a feature class, the field names on the feature class must match the fields as described below:

                                            * ``AttributeName``: Lists the name of the restriction.
                                            * ``ParameterName``: Lists the name of the parameter associated with the restriction. A restriction can have one or more ParameterName field values based on its intended use.
                                            * ``ParameterValue``: The value for ``ParameterName`` used by the tool when evaluating the restriction.
                                              Attribute Parameter Values is dependent on the Restrictions parameter. The ParameterValue field is applicable only
                                              if the restriction name is specified as the value for the
                                              Restrictions parameter.
                                              In Attribute Parameter Values, each restriction (listed as AttributeName) has a ParameterName field
                                              value, Restriction Usage, that specifies whether the restriction
                                              prohibits, avoids, or prefers travel on the roads associated with
                                              the restriction and the degree to which the roads are avoided or
                                              preferred. The Restriction Usage ParameterName can be assigned any of
                                              the following string values or their equivalent numeric values
                                              listed within the parentheses:

                                              * ``PROHIBITED`` (-1) - Travel on the roads using the restriction is completely prohibited.
                                              * ``AVOID_HIGH`` (5) - It is highly unlikely for the tool to include in the route the roads that are associated with the restriction.
                                              * ``AVOID_MEDIUM`` (2) - It is unlikely for the tool to include in the route the roads that are associated with the restriction.
                                              * ``AVOID_LOW`` (1.3) - It is somewhat unlikely for the tool to include in the route the roads that are associated with the restriction.
                                              * ``PREFER_LOW`` (0.8) - It is somewhat likely for the tool to include in the route the roads that are associated with the restriction.
                                              * ``PREFER_MEDIUM`` (0.5) - It is likely for the tool to include in the route the roads that are associated with the restriction.
                                              * ``PREFER_HIGH`` (0.2) - It is highly likely for the tool to include in the route the roads.

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
                                              **AttributeName**                         **ParameterName**          **ParameterValue**
                                              ----------------------------------------  -------------------------  -----------------------
                                              Any Hazmat Prohibited                     Restriction Usage           PROHIBITED

                                              ----------------------------------------  -------------------------  -----------------------
                                              Avoid Carpool Roads                       Restriction Usage          PROHIBITED
                                              ----------------------------------------  -------------------------  -----------------------
                                              Avoid Express Lanes                       Restriction Usage          PROHIBITED
                                              ----------------------------------------  -------------------------  -----------------------
                                              Avoid Ferries                             Restriction Usage          AVOID_MEDIUM
                                              ----------------------------------------  -------------------------  -----------------------
                                              Avoid Gates                               Restriction Usage          AVOID_MEDIUM
                                              ----------------------------------------  -------------------------  -----------------------
                                              Avoid Limited Access Roads                Restriction Usage          AVOID_MEDIUM
                                              ----------------------------------------  -------------------------  -----------------------
                                              Avoid Private Roads                       Restriction Usage          AVOID_MEDIUM
                                              ----------------------------------------  -------------------------  -----------------------
                                              Avoid Roads Unsuitable for Pedestrians    Restriction Usage          AVOID_HIGH
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
                                              ---------------------------------------  -------------------------  -----------------------
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

                                                                                        Vehicle Weight per          0
                                                                                        Axle (kilograms)
                                              ----------------------------------------  -------------------------  -----------------------
                                              Width Restriction                         Restriction Usage          PROHIBITED

                                                                                        Vehicle Width              0
                                                                                        (meters)
                                              ========================================  =========================  =======================
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    impedance                               Optional string.  Specify the impedance, which is a value that represents the effort or cost of traveling along
                                            road segments or on other parts of the transportation network.
                                            Travel distance is an impedance; the length of a road in kilometers can be thought of as impedance. Travel distance
                                            in this sense is the same for all modes-a kilometer for a pedestrian is also a kilometer for a car. (What may change
                                            is the pathways on which the different modes are allowed to travel, which affects distance between points, and this
                                            is modeled by travel mode settings.) Travel time can also be an impedance; a car may take one minute to travel a
                                            mile along an empty road. Travel times can vary by travel mode-a pedestrian may take more than 20  minutes to walk
                                            the same mile, so it is important to choose the right impedance for the travel mode you are modeling.
                                            Choose from the following impedance values: Drive Time-Models travel times for a car. These travel times are
                                            dynamic and fluctuate according to traffic flows in areas where traffic data is available. Truck Time-Models
                                            travel times for a truck.  These travel times are static for each road and don't fluctuate with traffic. This is
                                            the default value. Walk Time-Models travel times for a pedestrian. Travel Distance-Stores  length measurements
                                            along roads and paths. To model walk distance, choose this option and ensure Walking is  set in the Restriction
                                            parameter. Similarly, to model drive or truck distance, choose Travel Distance here and set the appropriate
                                            restrictions so your vehicle travels only on roads where it is permitted to do so. The value you provide for this
                                            parameter is ignored unless Travel Mode is set to Custom. If you choose Drive Time, Truck Time, or Walk Time, the
                                            Measurement Units parameter must be set to a time-based value; if you choose Travel Distance for Impedance,
                                            Measurement Units must be distance-based.

                                            Choice list:['Drive Time', 'Truck Time', 'Walk Time', 'Travel Distance']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    origin_destination_line_shape           Optional string. The resulting lines of an OD cost matrix can be represented with either straight-line geometry
                                            or no geometry at all. In both cases, the route is always computed along the street network by minimizing the
                                            travel time or the travel distance, never using the straight-line
                                            distance between origins and destinations.

                                            ``Straight Line``: Straight lines connect origins and destinations.
                                            ``None``: Do not return any shapes for the lines that connect origins and destinations. This is useful when you have a large number of origins and destinations and are interested only in the OD cost matrix table (and not the output line shapes).

                                            Choice list:['None', 'Straight Line']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    save_output_layer                       Optional boolean. Specify if the tool should save the analysis settings as a network analysis layer file.
                                            You cannot directly work with this file even when you open the file in an ArcGIS Desktop application like ArcMap.
                                            It is meant to be sent to Esri Technical Support to diagnose the quality of results returned from the tool.
                                            True: Save the network analysis layer file. The file is downloaded in a temporary directory on your machine. In ArcGIS Pro, the location of the downloaded file can be determined by viewing the value for the Output Network Analysis Layer parameter in the entry corresponding to the tool execution in the Geoprocessing history of your Project. In ArcMap, the location of the file can be determined by accessing the Copy Location option in the shortcut menu on the Output Network Analysis Layer parameter in the entry corresponding to the tool execution in the Geoprocessing Results window.
                                            False: Do not save the network analysis layer file. This is the default.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    overrides                               Optional string. Specify additional settings that can influence the behavior of the solver when finding solutions
                                            for the network analysis problems. The value for this parameter needs to be specified in JavaScript Object Notation
                                            (JSON). For example, a valid value is of the following form {"overrideSetting1" : "value1", "overrideSetting2" :
                                            "value2"}. The override setting name is always enclosed in double quotes. The values can be a number, Boolean,
                                            or string. The default value for this parameter is no value, which indicates not to override any solver settings. Overrides
                                            are advanced settings that should be used only after careful analysis of the results obtained before and after applying
                                            the settings. A list of supported override settings for each solver and their acceptable values can be obtained by contacting
                                            Esri Technical Support.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    save_route_data                         Optional boolean. Choose whether the output includes a zip file that contains a file geodatabase holding the inputs
                                            and outputs of the analysis in a format that can be used to share route layers with ArcGIS Online or Portal for
                                            ArcGIS.
                                            True: Save the route data as a zip file. The file is downloaded in a temporary directory on your machine. In ArcGIS Pro, the location of the downloaded file can be determined by viewing the value for the Output Route Data parameter in the entry corresponding to the tool execution in the Geoprocessing history of your Project. In ArcMap, the location of the file can be determined by accessing the Copy Location option in the shortcut menu on the Output Route Data parameter in the entry corresponding to the tool execution in the Geoprocessing Results window.
                                            False: Do not save the route data. This is the default.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_impedance                          Optional string. Specify the time-based impedance.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    distance_impedence                      Optional string. Specify the distance-based impedance.
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

    :returns: the following as a named tuple:

    * solve_succeeded - Solve Succeeded as a bool
    * output_origin_destination_lines - Output Origin Destination Lines as a FeatureSet
    * output_origins - Output Origins as a FeatureSet
    * output_destinations - Output Destinations as a FeatureSet

    Click `GenerateOriginDestinationCostMatrix <https://developers.arcgis.com/rest/network/api-reference/origin-destination-cost-matrix-service.htm>`_ for additional help.
    """

    if gis is None:
        gis = arcgis.env.active_gis
    url = gis.properties.helperServices.asyncODCostMatrix.url
    url = _validate_url(url, gis)
    tbx = _create_toolbox(url, gis=gis)
    defaults = dict(
        zip(
            tbx.generate_origin_destination_cost_matrix.__annotations__.keys(),
            tbx.generate_origin_destination_cost_matrix.__defaults__,
        )
    )
    if origins is None:
        origins = defaults["origins"]
    if destinations is None:
        destinations = defaults["destinations"]
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
    if output_format is None and "output_format" in defaults:
        output_format = defaults["output_format"]
    from arcgis._impl.common._utils import inspect_function_inputs

    params = {
        "origins": origins,
        "destinations": destinations,
        "travel_mode": travel_mode,
        "distance_units": distance_units,
        "analysis_region": analysis_region,
        "number_of_destinations_to_find": number_of_destinations_to_find,
        "cutoff": cutoff,
        "time_of_day": time_of_day,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "point_barriers": point_barriers,
        "line_barriers": line_barriers,
        "polygon_barriers": polygon_barriers,
        "uturn_at_junctions": uturn_at_junctions,
        "use_hierarchy": use_hierarchy,
        "restrictions": restrictions,
        "attribute_parameter_values": attribute_parameter_values,
        "impedance": impedance,
        "origin_destination_line_shape": origin_destination_line_shape,
        "save_output_network_analysis_layer": save_output_network_analysis_layer,
        "overrides": overrides,
        "time_impedance": time_impedance,
        "distance_impedance": distance_impedance,
        "output_format": output_format,
        "gis": gis,
        "future": True,
    }
    params = inspect_function_inputs(
        tbx.generate_origin_destination_cost_matrix, **params
    )
    params["future"] = True
    job = tbx.generate_origin_destination_cost_matrix(**params)
    if future:
        return job
    return job.result()


generate_origin_destination_cost_matrix.__annotations__ = {
    "origins": FeatureSet,
    "destinations": FeatureSet,
    "travel_mode": str,
    "time_units": str,
    "distance_units": str,
    "analysis_region": str,
    "number_of_destinations_to_find": int,
    "cutoff": float,
    "time_of_day": datetime,
    "time_zone_for_time_of_day": str,
    "point_barriers": FeatureSet,
    "line_barriers": FeatureSet,
    "polygon_barriers": FeatureSet,
    "uturn_at_junctions": str,
    "use_hierarchy": bool,
    "restrictions": str,
    "attribute_parameter_values": FeatureSet,
    "impedance": str,
    "origin_destination_line_shape": str,
    "return": tuple,
}
