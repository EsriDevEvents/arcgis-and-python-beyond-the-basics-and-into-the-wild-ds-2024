# -*- coding: utf-8 -*-
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


def solve_location_allocation(
    facilities: FeatureSet,
    demand_points: FeatureSet,
    measurement_units: Optional[str] = None,
    analysis_region: Optional[str] = None,
    problem_type: Optional[str] = None,
    number_of_facilities_to_find: Optional[int] = None,
    default_measurement_cutoff: Optional[float] = None,
    default_capacity: Optional[float] = None,
    target_market_share: Optional[float] = None,
    measurement_transformation_model: Optional[str] = None,
    measurement_transformation_factor: Optional[float] = None,
    travel_direction: Optional[str] = None,
    time_of_day: Optional[datetime] = None,
    time_zone_for_time_of_day: Optional[str] = None,
    uturn_at_junctions: Optional[str] = None,
    point_barriers: Optional[FeatureSet] = None,
    line_barriers: Optional[FeatureSet] = None,
    polygon_barriers: Optional[FeatureSet] = None,
    use_hierarchy: bool = True,
    restrictions: Optional[str] = None,
    attribute_parameter_values: Optional[FeatureSet] = None,
    allocation_line_shape: Optional[str] = None,
    travel_mode: str = "Custom",
    impedance: Optional[str] = None,
    save_output_network_analysis_layer: bool = False,
    overrides: Optional[dict] = None,
    time_impedance: Optional[str] = None,
    distance_impedance: Optional[str] = None,
    output_format: Optional[str] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``solve_location_allocation`` tool chooses the best location or locations from a set of input locations. Input to this tool includes facilities,
    which provide goods or services, and demand points, which consume the goods and services. The objective is to find the facilities that supply the
    demand points most efficiently. The tool solves this problem by analyzing various ways the demand points can be assigned to the different facilities.
    The solution is the scenario that allocates the most demand to facilities and minimizes overall travel. The output includes the solution facilities,
    demand points associated with their assigned facilities, and lines connecting demand points to their facilities. The location-allocation tool can be
    configured to solve specific problem types. Examples include the following: A retail store wants to see which potential store locations would need to
    be developed to capture 10 percent of the retail market in the area. A fire department wants to determine where it should locate fire stations to reach
    90 percent of the community within a four-minute response time. A police department wants to preposition personnel given past criminal activity at night.
    After a storm, a disaster response agency wants to find the best locations to set up triage facilities, with limited patient capacities, to tend to the
    affected population.

    ======================================  ==========================================================================================================================================
    **Parameter**                             **Description**
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    facilities                              Required :class:`~arcgis.features.FeatureSet` . Specify one or more ``facilities`` (up to 1,000). The tool chooses the best locations
                                            from the set of ``facilities`` you specify here. In a competitive analysis, in which  you try to find the best
                                            locations in the face of competition, the ``facilities`` of the competitors are specified here as well.
                                            When defining the ``facilities``, you can set properties for each one, such as the facility name or type, by using attributes.
                                            ``Facilities`` can be specified with the following fields: Name-The name of the facility. The name is included in the name of
                                            output allocation lines if the facility is part of the solution.

                                            * ``FacilityType`` - Specifies whether the facility is a candidate, required, or competitor facility. The field value is
                                              specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Candidate) - A facility that may be part of the solution.
                                              * 1 (Required) - A facility that must be part of the solution.
                                              * 2 (Competitor) - A rival facility that potentially removes demand from your ``facilities``. Competitor ``facilities`` are specific to the Maximize Market Share and Target Market Share problem types; they are ignored in other problem types. Weight-The relative weighting of the facility, which is used to rate the attractiveness, desirability, or bias of one facility compared to another. For example, a value of 2.0 could capture the preference of customers who prefer, at a ratio of 2 to 1, shopping in one facility over another facility. Factors that potentially affect facility weight include square footage, neighborhood, and age of the building. Weight values other than one are only honored by the maximize market share and target market share problem types; they are ignored in other problem types.

                                            * ``Capacity`` - The Capacity field is specific to the Maximize Capacitated Coverage problem type; the other problem types
                                              ignore this field.  Capacity specifies how much weighted demand the facility is capable of supplying. Excess demand won't
                                              be allocated to a facility even if that demand is within the facility's default measurement cutoff. Any value assigned to
                                              the Capacity field overrides the Default Capacity parameter (Default_Capacity in Python) for the given facility.

                                            * ``CurbApproach`` - Specifies the direction a vehicle may arrive at or depart from the facility. The field value is
                                              specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Either side of vehicle)-The facility can be visited from either the right or left side of the vehicle.
                                              * 1 (Right side of vehicle)-Arrive at or depart the facility so that it is on the right side of the vehicle. This is typically used for vehicles such as buses that must arrive with the bus stop on the right-hand side so that passengers can disembark at the curb.
                                              * 2 (Left side of vehicle)-Arrive at or depart the facility so that it is on the left side of the vehicle. When the vehicle approaches and departs the facility, the curb must be on the left side of the vehicle. This is typically used for vehicles such as buses that must arrive with the bus stop on the left-hand side so that passengers can disembark at the curb.

                                              The ``CurbApproach`` property is designed to work with both kinds of national driving standards: right-hand traffic (United States) and
                                              left-hand traffic (United Kingdom). First, consider a facility on the left side of a vehicle. It is always on the left side regardless
                                              of whether the vehicle travels on the left or right half of the road. What may change with national driving standards is your decision
                                              to approach a facility  from one of two directions, that is, so it ends up on the right or left side of the vehicle. For example,
                                              if you want to arrive at a facility and not have a lane of traffic between the vehicle and the incident, you would choose Right side
                                              of vehicle (1) in the United States but Left side of vehicle (2) in the United Kingdom.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    demand_points                           Required :class:`~arcgis.features.FeatureSet` . Specify one or more demand points (up to 10,000). The tool chooses the best facilities based in
                                            large part on how they serve the demand points specified here.   When defining the demand points, you can set
                                            properties for each one, such as the demand-point name or weight, by using attributes. Demand points can be specified
                                            with the following fields: Name-The name of the demand point. The name is included in the name of an output allocation
                                            line or lines if the demand point is part of the solution.  GroupName-The name of the group the demand point is part of.
                                            This property is ignored for the maximize capacitated coverage, target market share, and maximize market share problem types.
                                            If demand points share a group name, the solver allocates all members of the group to the same facility. (If constraints, such
                                            as a cutoff distance, prevent any of the demand points in the group from reaching the same facility, none of the demand points
                                            are allocated.) Weight-The relative weighting of the demand point. A value of 2.0 means the demand point is twice as important
                                            as one with a weight of 1.0. If demand points represent households, weight could indicate the number of people in each household.
                                            Cutoff_Time-The demand point can't be allocated to a facility that is beyond the travel time indicated here. This field value
                                            overrides the value of the Default Measurement Cutoff parameter.  The units for this attribute value are specified by the Measurement
                                            Units parameter. The attribute value is referenced during the analysis only when the measurement units are time based. The default
                                            value is null, which means there isn't an override cutoff. Cutoff_Distance-The demand point can't be allocated to a facility that is
                                            beyond the travel distance indicated here. This field value overrides the value of the Default Measurement Cutoff parameter.
                                            The units for this attribute value are specified by the Measurement Units parameter. The attribute value is referenced during the
                                            analysis only when the measurement units are distance based. The default value is null, which means there isn't an override cutoff.

                                            * ``CurbApproach`` - Specifies the direction a vehicle may arrive at or depart from the demand point.
                                              The field value is specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Either side of vehicle)-The demand point can be visited from either the right or left side of the vehicle.
                                              * 1 (Right side of vehicle)-Arrive at or depart the demand point so that it is on the right side of the vehicle. This is typically used for
                                                vehicles such as buses that must arrive with the bus stop on the right-hand side so that passengers can disembark at the curb.
                                              * 2 (Left side of vehicle)-Arrive at or depart the demand point so that it is on the left side of the vehicle. When the vehicle approaches
                                                and departs the demand point, the curb must be on the left side of the vehicle. This is typically used for vehicles such as buses that must
                                                arrive with the bus stop on the left-hand side so that passengers can disembark at the curb.

                                              The ``CurbApproach`` property is designed to work with both kinds of national driving standards: right-hand traffic (United States) and
                                              left-hand traffic (United Kingdom). First, consider a demand point on the left side of a vehicle. It is always on the left side regardless
                                              of whether the vehicle travels on the left or right half of the road. What may change with national driving standards is your decision to
                                              approach a demand point  from one of two directions, that is, so it ends up on the right or left side of the vehicle. For example, if you
                                              want to arrive at a demand point and not have a lane of traffic between the vehicle and the demand point, you would choose Right side of vehicle
                                              (1) in the United States but Left side of vehicle (2) in the United Kingdom.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    measurement_units                       Required string. Specify the units that should be used to measure the travel times or travel distances between
                                            demand points and facilities. The tool chooses the best facilities based on which ones can reach, or be reached by,
                                            the most amount of weighted demand with the least amount travel.
                                            The  output allocation lines report travel distance or travel time in different units, including the units you specify for this parameter.

                                            Choice list: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles', 'NauticalMiles', 'Seconds', 'Minutes', 'Hours', 'Days']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    analysis_region                         Optional string. Specify the region in which to perform the analysis. If a value is not specified for this parameter,
                                            the tool will automatically calculate the region name based on the location
                                            of the input points. Setting the name of the region is recommended to speed up the
                                            tool execution.

                                            Choice list: ['NorthAmerica', 'SouthAmerica', 'Europe', 'MiddleEastAndAfrica', 'India', 'SouthAsia', 'SouthEastAsia', 'Thailand', 'Taiwan', 'Japan', 'Oceania', 'Greece', 'Korea']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    problem_type                            Optional string. Specifies the objective of the location-allocation analysis. The default objective is to minimize impedance.

                                            * ``Minimize Impedance``: This is also known as the P-Median problem type. Facilities are located such that the sum of all
                                              weighted travel time or distance between demand points and solution facilities is minimized.
                                              (Weighted travel is the amount of demand allocated to a facility multiplied by the travel distance or time to the facility.) This problem type is traditionally used to
                                              locate warehouses, because it can reduce the overall transportation costs of delivering goods to outlets. Since Minimize Impedance reduces the overall distance the public
                                              needs to travel to reach the chosen facilities, the minimize impedance problem without an impedance cutoff is ordinarily regarded as more equitable than other problem
                                              types for locating some public-sector facilities such as libraries, regional airports, museums, department of motor vehicles offices, and health clinics.
                                              The following list describes how the minimize impedance problem type handles demand:
                                              * A demand point that cannot reach any facilities, due to setting a cutoff distance or time, is not allocated.
                                              * A demand point that can only reach one facility has all its demand weight allocated to that facility.
                                              * A demand point that can reach two or more facilities has all its demand weight allocated to the nearest facility only.

                                            * ``Maximize Coverage``: Facilities are located such that as much demand as possible is allocated to solution facilities
                                              within the impedance cutoff. Maximize Coverage is frequently used to locate fire stations, police stations, and ERS centers, because emergency services are often required to
                                              arrive at all demand points within a specified response time. Note that it is important for all organizations, and critical for emergency services, to have accurate and precise data so that
                                              analysis results correctly model real-world results. Pizza delivery businesses, as opposed to eat-in pizzerias, try to locate stores where they can cover the most people within a certain drive time.
                                              People who order pizzas for delivery don't typically worry about how far away the pizzeria is; they are mainly concerned with the pizza arriving within an advertised time window. Therefore, a
                                              pizza-delivery business would subtract pizza-preparation time from their advertised delivery time and solve a maximize coverage problem to choose the candidate facility that would capture the most
                                              potential customers in the coverage area. (Potential customers of eat-in pizzerias are more affected by distance, since they need to travel to the restaurant; thus, the attendance maximizing or
                                              market share problem types would better suit eat-in restaurants.)
                                              The following list describes how the Maximize Coverage problem handles demand:
                                              * A demand point that cannot reach any facilities due to cutoff distance or time is not allocated.
                                              * A demand point that can only reach one facility has all its demand weight allocated to that facility.
                                              * A demand point that can reach two or more facilities has all its demand weight allocated to the nearest facility only.

                                            * ``Maximize Capacitated Coverage``: Facilities are located such that all or the greatest amount of demand can be served without exceeding the capacity of any facility. Maximize
                                              Capacitated Coverage behaves like  either the Minimize Impedance or Maximize Coverage problem type but with the added constraint of capacity. You can specify a capacity for an individual facility
                                              by assigning a numeric value to  its corresponding Capacity field on the input facilities. If the Capacity field value is null, the facility is assigned a capacity from the Default Capacity property.
                                              Use-cases for Maximize Capacitated Coverage include creating territories that encompass a given number of people or businesses, locating hospitals or other medical facilities with a limited number
                                              of beds or patients who can be treated, or locating warehouses whose inventory isn't assumed to be unlimited.
                                              The following list describes how the Maximize Capacitated Coverage problem handles demand:
                                              * Unlike Maximize Coverage, Maximize Capacitated Coverage doesn't require a value for the Default Measurement Cutoff; however, when an cutoff is specified, any demand point outside the cutoff time or distance of all facilities is not allocated.
                                              * An allocated demand point has all or none of its demand weight assigned to a facility; that is, demand isn't apportioned with this problem type.
                                              * If the total demand that can reach a facility is greater than the capacity of the facility, only the demand points that maximize total captured demand and minimize total weighted travel are allocated.

                                              .. note::
                                                You may notice an apparent inefficiency when a demand point is allocated to a facility that isn't the  nearest solution facility. This may occur when demand points have varying weights and when the
                                                demand point in question can reach more than one facility. This kind of result indicates the nearest solution facility didn't have adequate capacity for the weighted demand, or the most efficient
                                                solution for the entire problem required one or more local inefficiencies. In either case, the solution is correct.

                                            * ``Minimize Facilities``: Facilities are chosen such that as much weighted demand as
                                              possible are allocated to solution facilities within the travel time or distance cutoff; additionally, the number of facilities required to cover demand is minimized. Minimize Facilities is the
                                              same as Maximize Coverage but with the exception of the number of facilities to locate, which in this case is determined by the solver. When the cost of building facilities is not a limiting factor,
                                              the same kinds of organizations that use Maximize Coverage (emergency response, for instance) use Minimize Facilities so that all possible demand points will be covered.
                                              The following list describes how the Minimize Facilities problem handles demand:
                                              * A demand point that cannot reach any facilities due to a cutoff distance or time is not allocated.
                                              * A demand point that can only  reach one facility has all its demand weight allocated to that facility.
                                              * A demand point that can reach two or more facilities has all its demand weight allocated to the nearest facility only.

                                            * ``Maximize Attendance``: Facilities are chosen such that as much demand weight as possible is allocated to facilities while
                                              assuming the demand weight decreases in relation to the distance between the facility and the demand point. Specialty stores
                                              that have little or no competition benefit from this problem type, but it may also be beneficial to general retailers and restaurants that
                                              don't have the data on competitors necessary to perform market share problem types. Some businesses that might benefit from this problem
                                              type include coffee shops, fitness centers, dental and medical offices, and electronics stores. Public transit bus stops are often
                                              chosen with the help of Maximize Attendance. Maximize Attendance assumes that the farther people have to travel to reach your facility,
                                              the less likely they are to use it. This is reflected in how the amount of demand allocated to facilities diminishes with distance.
                                              The following list describes how the Maximize Attendance problem handles demand:
                                              * A demand point that cannot reach any facilities due to a cutoff distance or time is not allocated.
                                              * When a demand point can reach a facility, its demand weight is only partially allocated to the facility. The amount allocated decreases as a function of the maximum cutoff distance (or time) and the travel distance (or time) between the facility and the demand point.
                                              * The weight of a demand point that can reach more than one facility is proportionately allocated to the nearest facility only.

                                            * ``Maximize Market Share``: A specific number of facilities are chosen such that the allocated demand is maximized in the presence
                                              of competitors. The goal is to capture as much of the total market share as possible with a given number of facilities,
                                              which you specify. The total market share is the sum of all demand weight for valid demand points. The market share problem
                                              types require the most data because, along with knowing your own facilities' weight,
                                              you also need to know that of your competitors' facilities. The same types of facilities that use the Maximize Attendance problem type
                                              can also use market share problem types given that they have comprehensive
                                              information that includes competitor data. Large discount stores typically use  Maximize Market Share to locate a finite set of new stores.
                                              The market share problem types use a Huff model, which is also known as a gravity model or spatial interaction.
                                              The following list describes how the Maximize Market Share problem handles demand:

                                              * A demand point that cannot reach any facilities due to a cutoff distance or time is not allocated.
                                              * A demand point that can only reach one facility has all its demand weight allocated to that facility.
                                              * A demand point that can reach two or more facilities has all its demand weight allocated to them; furthermore, the weight is
                                                split among the facilities proportionally to the facilities' attractiveness (facility weight) and inversely proportional to
                                                the distance between the facility and demand point. Given equal facility weights,
                                                this means more demand weight is assigned to near facilities than far facilities.
                                              * The total market share, which can be used to calculate the captured market share, is the sum of the weight of all valid demand points.

                                            * ``Target Market Share``: Target Market Share chooses the minimum number of facilities necessary to capture a specific percentage
                                              of the total market share in the presence of competitors. The total market share is the
                                              sum of all demand weight for valid demand points. You set the percent of the market share you want to reach and let the solver choose
                                              the fewest number of facilities necessary to meet that threshold.
                                              The market share problem types require the most data because, along with knowing your own facilities' weight, you also need to know
                                              that of your competitors' facilities. The same types of facilities that use
                                              the Maximize Attendance problem type can also use market share problem types given that they have comprehensive information that includes
                                              competitor data. Large discount stores typically use the Target Market Share
                                              problem type when they want to know how much expansion would be required to reach a certain level of the market share or see what strategy would be needed just to maintain their current market share given the introduction
                                              of new competing facilities. The results often represent what stores would like to do if budgets weren't a concern. In other cases where budget is a concern, stores revert to the Maximize Market Share problem and
                                              simply capture as much of the market share as possible with a limited number of facilities.
                                              The following list describes how the target market share problem handles demand:

                                              * The total market share, which is used in calculating the captured market share, is the sum of the weight of all valid demand points.
                                              * A demand point that cannot reach any facilities due to a cutoff distance or time is not allocated.
                                              * A demand point that can only reach one facility has all its demand weight allocated to that facility.
                                              * A demand point that can reach two or more facilities has all its demand weight allocated to them; furthermore,
                                                the weight is split among the facilities proportionally to the facilities' attractiveness (facility weight) and inversely
                                                proportional to the distance between the facility and demand point. Given equal facility weights, this means more demand
                                                weight is assigned to near facilities than far facilities.

                                            Choice list:['Maximize Attendance', 'Maximize Capacitated Coverage', 'Maximize Coverage', 'Maximize Market Share', 'Minimize Facilities', 'Minimize Impedance', 'Target Market Share']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    number_of_facilities_to_find            Optional integer. Specify the number of facilities the solver should choose. The default value is 1.
                                            The facilities with a  FacilityType field value of 1 (Required) are always chosen first. Any excess facilities to
                                            choose are picked from candidate facilities, which have a FacilityType field value of 2. Any facilities that have a
                                            FacilityType value of 3 (Chosen) before solving are treated as candidate facilities at solve time. If the number of
                                            facilities to find is less than the number of required facilities, an error occurs. Number of Facilities to Find is
                                            disabled for the Minimize Facilities and Target Market Share problem types since the solver determines the minimum
                                            number of facilities needed to meet the objectives.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    default_measurement_cutoff              Optional float. Specifies the maximum travel time or distance allowed between a demand point and the facility it is
                                            allocated to. If a demand point is outside the cutoff of a facility, it cannot be allocated to that facility.
                                            The default value is none, which means the cutoff limit doesn't apply.
                                            The units for this parameter are the same as those specified by the Measurement Units parameter.
                                            The travel time or distance cutoff is measured by the shortest path along roads.  This property might be used
                                            to model the maximum distance that people are willing to travel to visit stores or the maximum time that is
                                            permitted for a fire department to reach anyone in the community. Note that demand points have Cutoff_Time and
                                            ``Cutoff_Distance`` fields, which, if set accordingly, overrides the Default Measurement Cutoff parameter. You might
                                            find that people in rural areas are willing to travel up to 10 miles to reach a facility while urbanites are only
                                            willing to travel up to two miles. Assuming Measurement Units is set to Miles, you can model this behavior by setting
                                            the default measurement cutoff to 10 and the ``Cutoff_Distance`` field value of the demand points in urban areas to 2.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    default_capacity                        Optional float. This property is specific to the Maximize Capacitated Coverage problem type. It is the default
                                            capacity assigned to all facilities in the analysis. You can override the default capacity for a facility by
                                            specifying a value in the facility's  Capacity field.
                                            The default value is 1.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    target_market_share                     Optional float. This parameter is specific to the Target Market Share problem type. It is the percentage of the
                                            total demand weight that you want the chosen and required facilities to capture. The solver chooses the minimum
                                            number of facilities needed to capture the target market share specified here.
                                            The default value is 10 percent.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    measurement_transformation_model        Optional string. This sets the equation for transforming the network cost between facilities and demand points.
                                            This property, coupled with the Impedance Parameter, specifies how severely the network impedance between
                                            facilities and demand points influences the solver's choice of facilities. In the following list of transformation
                                            options, d refers to demand points and f, facilities. "Impedance" refers to the shortest travel distance or time
                                            between two locations. So impedancedf is the shortest-path (time or distance) between demand point d and facility f,
                                            and costdf is the transformed travel time or distance between the facility and demand point. Lambda (λ) denotes the impedance parameter.
                                            The Measurement Units setting determines whether travel time or distance is analyzed.

                                            * ``Linear``: costdf = λ * impedancedf The transformed travel time or distance between the facility and the demand point is the same as the time or distance of the shortest path between the two locations. With this option, the impedance parameter (λ) is always set to one. This is the default.
                                            * ``Power``: costdf = impedancedfλ The transformed travel time or distance between the facility and the demand point is equal to the time or distance of the shortest path raised to the power specified by the impedance parameter (λ). Use the Power option with a positive impedance parameter to specify higher weight to nearby facilities.
                                            * ``Exponential``: costdf = e(λ * impedancedf) The transformed travel time or distance between the facility and the demand point is equal to the mathematical constant e raised to the power specified by the shortest-path network impedance multiplied with the impedance parameter (λ). Use the Exponential option with a positive impedance parameter to specify a high weight to nearby facilities.

                                            Choice list: ['Linear', 'Power', 'Exponential']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    measurement_transformation_factor       Optional float. Provides a parameter value to the equations specified in the Measurement Transformation
                                            Model parameter. The parameter value is ignored when the impedance transformation is of type linear.
                                            For power and exponential impedance transformations, the value should be nonzero.

                                            The default value is 1.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    travel_direction                        Optional string. Specify whether to measure travel times or distances from facilities to demand points or from demand
                                            points to facilities. The default value is to measure from facilities to demand points.

                                            * ``Facility to Demand``: Direction of travel is from facilities to demand points. This is the default.
                                            * ``Demand to Facility``: Direction of travel is from demand points to facilities.

                                            Travel times and distances may change based on direction of travel. If going from point A to point B,
                                            you may encounter less traffic or have a shorter path, due to one-way streets and turn restrictions,
                                            than if you were traveling in the opposite direction. For instance, going from point A to point B may
                                            only take 10 minutes, but going the other direction may take 15 minutes. These differing measurements may
                                            affect whether demand points can be assigned to certain facilities because of cutoffs or, in problem types
                                            where demand is apportioned, affect how much demand is captured. Fire departments commonly measure from
                                            facilities to demand points since they are concerned with the time it takes to travel from the fire station
                                            to the location of the emergency. A retail store is more concerned with the time it takes shoppers to reach
                                            the store; therefore, stores commonly measure from demand points to facilities. Travel Direction also
                                            determines the meaning of any start time that is provided. See the Time of Day parameter for more information.

                                            Choice list:['Demand to Facility', 'Facility to Demand']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_of_day                             Optional datetime. Specify the time at which travel begins. This property is ignored unless Measurement Units
                                            are time based. The default is no time or date. When Time of Day isn't specified, the solver uses generic
                                            speeds-typically those from posted speed limits.
                                            Traffic constantly changes in reality, and as it changes, travel times between facilities and demand points
                                            also fluctuate. Therefore, indicating different time and date values over several analyses may affect how
                                            demand is allocated to facilities and which facilities are chosen in the results.  The time of day always
                                            indicates a start time. However, travel may start from facilities or demand points; it depends on what you
                                            choose for the Travel Direction parameter. The Time Zone for Time of Day parameter specifies whether this
                                            time and date refer to UTC or the time zone in which the facility or demand point is located.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    time_zone_for_time_of_day               Optional string. Specifies the time zone of the Time of Day parameter. The default is geographically local.

                                            * ``Geographically Local``: The Time of Day parameter refers to the time zone in which the facilities or demand
                                              points are located. If Travel Direction is facilities to demand points, this is the time zone of the facilities.
                                              If Travel Direction is demand points to facilities, this is the time zone of the demand points.
                                            * ``UTC``: The Time of Day parameter refers to Coordinated Universal Time (UTC). Choose this option if you want to choose the best
                                              location for a specific time, such as now, but aren't certain in which time zone the facilities or demand
                                              points will be located. Irrespective of the Time Zone for Time of Day setting, the following rules are
                                              enforced by the tool if your facilities and demand points are in multiple time zones: All facilities must
                                              be in the same time zone when specifying a time of day and travel is from facility to demand. All demand
                                              points must be in the same time zone when specifying a time of day and travel is from demand to facility.

                                            Choice list:['Geographically Local', 'UTC']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    uturn_at_junctions                      Optional string. The U-Turn policy at junctions. Allowing U-turns implies the solver can turn around at a
                                            junction and double back on the same street. Given that junctions represent street intersections and dead ends,
                                            different vehicles  may be able to turn around at some junctions but not at others-it depends on whether the junction
                                            represents an intersection or dead end. To accommodate, the U-turn policy parameter is implicitly specified by how
                                            many edges, or streets, connect to the junction, which is known as junction valency. The acceptable values for this
                                            parameter are listed below; each is followed by a description of its meaning in terms of junction valency.

                                            * ``Allowed``: U-turns are permitted at junctions with any number of connected edges, or streets. This is the default value.
                                            * ``Not Allowed``: U-turns are prohibited at all junctions, regardless of junction valency.
                                            * ``Allowed only at Dead Ends``: U-turns are prohibited at all junctions, except those that have only one adjacent edge (a dead end).
                                            * ``Allowed only at Intersections and Dead Ends``: U-turns are prohibited at junctions where exactly two adjacent edges meet but are
                                              permitted at intersections (junctions with three or more adjacent edges) and dead ends (junctions with exactly one adjacent edge).
                                              Oftentimes, networks modeling streets have extraneous junctions in the middle of road segments. This option prevents vehicles from
                                              making U-turns at these locations. This parameter is ignored unless Travel Mode is set to Custom.

                                            Choice list: ['Allowed', 'Not Allowed', 'Allowed Only at Dead Ends', 'Allowed Only at Intersections and Dead Ends']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    point_barriers                          Optional :class:`~arcgis.features.FeatureSet` . Specify one or more points to act as temporary
                                            restrictions or represent additional time or distance that may be
                                            required to travel on the underlying streets. For example, a point
                                            barrier can be used to represent a fallen tree along a street or
                                            time delay spent at a railroad crossing. The tool imposes a limit of 250 points that can be added as barriers.
                                            When specifying the point barriers, you can set properties for each one, such as its name or barrier type,
                                            by using attributes. The point barriers can be specified with the following attributes:

                                            * ``Name``: The name of the barrier.

                                            * ``BarrierType``: Specifies whether the point barrier restricts travel completely or adds time or distance when it is crossed. The value for this attribute is specified as one of the following integers (use the numeric code, not the name in parentheses):

                                              * 0 (Restriction)-Prohibits travel through the barrier. The barrier
                                                is referred to as a restriction point barrier since it acts as a
                                                restriction.

                                              * 2 (Added Cost)-Traveling through the barrier increases the travel
                                                time or distance by the amount specified in the
                                                Additional_Time or Additional_Distance field. This barrier type is
                                                referred to as an added-cost point barrier.

                                            * ``Additional_Time``: Indicates how much travel time is added when the
                                              barrier is traversed. This field is applicable only for added-cost
                                              barriers and only if the measurement units are time based. This field
                                              value must be greater than or equal to zero, and its units are the same as those specified in the Measurement Units parameter.

                                            * ``Additional_Distance``: Indicates how much distance is added when the barrier is
                                              traversed. This field is applicable only for added-cost barriers
                                              and only if the measurement units are distance based. The field value
                                              must be greater than or equal to zero, and its units are the same as those specified in the
                                              Measurement Units parameter.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    line_barriers                           Optional :class:`~arcgis.features.FeatureSet` . Specify one or more lines that prohibit travel anywhere
                                            the lines intersect the streets. For example, a parade or protest
                                            that blocks traffic across several street segments can be modeled
                                            with a line barrier. A line barrier can also quickly fence off
                                            several roads from being traversed, thereby channeling possible
                                            routes away from undesirable parts of the street
                                            network. The tool imposes a limit on the number of streets you can
                                            restrict using the Line Barriers parameter. While there is no limit on
                                            the number of lines you can specify as line barriers, the combined
                                            number of streets intersected by all the lines cannot exceed
                                            500. When specifying the line barriers, you can set a name property for each one by using the following attribute:

                                            * Name: The name of the barrier.
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

                                              * 0 (Restriction) - Prohibits traveling through any part of the barrier. The barrier is referred to as a restriction polygon barrier since it prohibits traveling on streets intersected by the barrier. One use of this type of barrier is to model floods covering areas of the street that make traveling on those streets impossible.
                                              * 1 (Scaled Cost) - Scales the time or distance required to travel the underlying streets by a factor specified using the ScaledTimeFactor or ScaledDistanceFactor fields. If the streets are partially covered by the barrier, the travel time or distance is apportioned and then scaled. For example, a factor 0.25 would mean that travel on underlying streets is expected to be four times faster than normal. A factor of 3.0 would mean it is expected to take three times longer than normal to travel on underlying streets. This barrier type is referred to as a scaled-cost polygon barrier. It might be used to model storms that reduce travel speeds in specific regions.

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
    use_hierarchy                           Optional boolean. Specify whether hierarchy should be used when finding the shortest path between the
                                            facilities and demand points.

                                            Checked (True):
                                            Use hierarchy when measuring between facilities and demand points. When
                                            hierarchy is used, the tool prefers higher-order streets (such as
                                            freeways) to lower-order streets (such as local roads), and can be used
                                            to simulate the driver preference of traveling on freeways instead
                                            of local roads even if that means a longer trip. This is especially
                                            true when finding routes to faraway locations, because drivers on long-distance trips tend to prefer traveling on freeways where stops, intersections, and turns can be avoided. Using hierarchy is computationally faster,
                                            especially for long-distance routes, since the tool can determine the
                                            best route from a relatively smaller subset of streets.

                                            Unchecked (False):
                                            Do not use hierarchy when measuring between facilities and demand points. If
                                            hierarchy is not used, the tool considers all the streets and doesn't
                                            prefer higher-order streets when finding the route. This is often
                                            used when finding short-distance routes within a city.
                                            The tool automatically reverts to using hierarchy if the
                                            straight-line distance between facilities and demand points is
                                            greater than 50 miles, even if you have set this parameter to not use hierarchy.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    restrictions                            Optional string. Specify which restrictions should be honored by the tool when finding the best routes
                                            between facilities and demand points. A restriction represents a driving
                                            preference or requirement. In most cases, restrictions cause roads
                                            to be prohibited. For instance, using an Avoid Toll Roads restriction will result in a route that will
                                            include toll roads only when it is absolutely required to travel on toll roads in order to visit an incident
                                            or a facility. Height Restriction makes it possible to route around any clearances that are lower than the
                                            height of your vehicle. If you are carrying corrosive materials on your vehicle, using the Any Hazmat Prohibited
                                            restriction prevents hauling the materials along roads where it is marked as illegal to do so.
                                            Below is a list of available restrictions and a short description.
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
                                            in a particular country by looking at the table in the Country List section of the Data coverage for network analysis services web page. If a country has a value of  Yes in the Logistics Attribute column, the restriction with select availability in the region is supported in that country. If you specify restriction names that are not available in the country where your incidents are located, the service ignores the invalid restrictions. The service also ignores restrictions whose Restriction Usage parameter value is between 0 and 1 (see the Attribute Parameter Value parameter). It prohibits all restrictions whose Restriction Usage parameter value is greater than 0.
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

                                            Choice list:['Any Hazmat Prohibited', 'Avoid Carpool Roads', 'Avoid Express Lanes', 'Avoid Ferries', 'Avoid Gates',
                                            'Avoid Limited Access Roads', 'Avoid Private Roads', 'Avoid Roads Unsuitable for Pedestrians', 'Avoid Stairways', 'Avoid Toll Roads',
                                            'Avoid Toll Roads for Trucks', 'Avoid Truck Restricted Roads', 'Avoid Unpaved Roads', 'Axle Count Restriction', 'Driving a Bus',
                                            'Driving a Delivery Vehicle', 'Driving a Taxi', 'Driving a Truck', 'Driving an Automobile', 'Driving an Emergency Vehicle',
                                            'Height Restriction', 'Kingpin to Rear Axle Length Restriction', 'Length Restriction', 'Preferred for Pedestrians', 'Riding a Motorcycle',
                                            'Roads Under Construction Prohibited', 'Semi or Tractor with One or More Trailers Prohibited', 'Single Axle Vehicles Prohibited',
                                            'Tandem Axle Vehicles Prohibited', 'Through Traffic Prohibited', 'Truck with Trailers Restriction', 'Use Preferred Hazmat Routes', 'Use Preferred Truck Routes', 'Walking', 'Weight Restriction', 'Weight per Axle Restriction', 'Width Restriction']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    attribute_parameter_values              Optional :class:`~arcgis.features.FeatureSet` . Specify additional values required by some restrictions, such as the weight of a vehicle
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
                                            * ``ParameterValue``: The value for ParameterName used by the tool when evaluating the restriction.
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
                                              * ``AVOID_LOW`` (1.3) - It is somewhat unlikely for the tool to include in the route the roads that are associated with the restriction
                                              * ``PREFER_LOW`` (0.8) - It is somewhat likely for the tool to include in the route the roads that are associated with the restriction.
                                              * ``PREFER_MEDIUM`` (0.5) - It is likely for the tool to include in the route the roads that are associated with the restriction.
                                              * ``PREFER_HIGH`` (0.2) - It is highly likely for the tool to include in the route the roads that are associated with the restriction.

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
                                              Any Hazmat Prohibited                     Restriction Usage          PROHIBITED

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

                                                                                        Vehicle Weight per         0
                                                                                        Axle (kilograms)
                                              ----------------------------------------  -------------------------  -----------------------
                                              Width Restriction                         Restriction Usage          PROHIBITED

                                                                                        Vehicle Width              0
                                                                                        (meters)
                                              ========================================  =========================  =======================

    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    allocation_line_shape                   Optional string. The default is to output straight lines.

                                            Specify the type of line features that are output by the
                                            tool. The parameter accepts one of the following
                                            values:
                                            Straight Line: Return straight lines between solution facilities and the demand points allocated to them. This is the default.  Drawing straight lines on a map helps you visualize how demand is allocated. None: Return a table containing data about the shortest paths between solution facilities and the demand points allocated to them, but don't return lines.
                                            No matter which value you choose for the Allocation Line Shape
                                            parameter, the shortest route is always determined by minimizing the
                                            travel time or the travel distance, never using the straight-line
                                            distance between demand points and
                                            facilities. That is, this parameter only changes the output line shapes; it doesn't change the measurement method.

                                            Choice list:['None', 'Straight Line']
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
    impedance                               Optional string. Specify the impedance, which is a value that represents the effort or cost of traveling
                                            along road segments or on other parts of the transportation network.
                                            Travel time is an impedance; a car taking one minute to travel a mile along an empty road is an example of impedance. Travel times can vary by travel mode-a pedestrian may take more than 20 minutes to walk the same mile-so it is important to choose the right impedance for the travel mode you are modeling. Choose from the following impedance values: Drive Time-Models travel times for a car. These travel times are static for each road and don't fluctuate with traffic. Truck Time-Models travel times for a truck.  These travel times are static for each road and don't fluctuate with traffic. Walk Time-Models travel times for a pedestrian. The value you provide for this parameter is ignored unless Travel Mode is set to Custom, which is the default value.

                                            Choice list:['Drive Time', 'Truck Time', 'Walk Time']
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    gis                                     Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    --------------------------------------  ------------------------------------------------------------------------------------------------------------------------------------------
    future                                  Optional boolean. If True, a future object will be returned and the process
                                            will not wait for the task to complete. The default is False, which means wait for results.
    ======================================  ==========================================================================================================================================

    :return: the following as a named tuple:

    * solve_succeeded - Solve Succeeded as a bool
    * output_allocation_lines - Output Allocation Lines as a FeatureSet
    * output_facilities - Output Facilities as a FeatureSet

    * output_demand_points - Output Demand Points as a FeatureSet

    Click `solveLocationAllocation <https://developers.arcgis.com/rest/network/api-reference/location-allocation-service.htm>`_ for additional help.
    """

    if gis is None:
        gis = arcgis.env.active_gis
    url = gis.properties.helperServices.asyncLocationAllocation.url
    url = _validate_url(url, gis)
    tbx = _create_toolbox(url, gis=gis)
    defaults = dict(
        zip(
            tbx.solve_location_allocation.__annotations__.keys(),
            tbx.solve_location_allocation.__defaults__,
        )
    )
    if default_capacity is None:
        default_capacity = defaults["default_capacity"]
    if target_market_share is None:
        target_market_share = defaults["target_market_share"]
    if number_of_facilities_to_find is None:
        number_of_facilities_to_find = defaults["number_of_facilities_to_find"]
    if problem_type is None:
        problem_type = defaults["problem_type"]

    if measurement_units is None:
        measurement_units = defaults["measurement_units"]

    if measurement_transformation_model is None:
        measurement_transformation_model = defaults["measurement_transformation_model"]
    if measurement_transformation_factor is None:
        measurement_transformation_factor = defaults[
            "measurement_transformation_factor"
        ]
    if travel_direction is None:
        travel_direction = defaults["travel_direction"]
    if time_of_day is None:
        time_of_day = defaults["time_of_day"]
    if time_zone_for_time_of_day is None:
        time_zone_for_time_of_day = defaults["time_zone_for_time_of_day"]
    if use_hierarchy is None:
        use_hierarchy = defaults["use_hierarchy"]

    if uturn_at_junctions:
        uturn_at_junctions = defaults["uturn_at_junctions"]
    if impedance is None:
        impedance = defaults["impedance"]
    if travel_mode is None:
        travel_mode = defaults["travel_mode"]
    if allocation_line_shape is None:
        allocation_line_shape = defaults["allocation_line_shape"]
    if restrictions is None:
        restrictions = defaults["restrictions"]
    if facilities is None:
        facilities = defaults["facilities"]

    if demand_points is None:
        demand_points = defaults["demand_points"]

    if point_barriers is None:
        point_barriers = defaults["point_barriers"]

    if line_barriers is None:
        line_barriers = defaults["line_barriers"]

    if polygon_barriers is None:
        polygon_barriers = defaults["polygon_barriers"]

    if attribute_parameter_values is None:
        attribute_parameter_values = defaults["attribute_parameter_values"]
    from arcgis._impl.common._utils import inspect_function_inputs

    params = {
        "facilities": facilities,
        "demand_points": demand_points,
        "measurement_units": measurement_units,
        "analysis_region": analysis_region,
        "problem_type": problem_type,
        "number_of_facilities_to_find": number_of_facilities_to_find,
        "default_measurement_cutoff": default_measurement_cutoff,
        "default_capacity": default_capacity,
        "target_market_share": target_market_share,
        "measurement_transformation_model": measurement_transformation_model,
        "measurement_transformation_factor": measurement_transformation_factor,
        "travel_direction": travel_direction,
        "time_of_day": time_of_day,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "uturn_at_junctions": uturn_at_junctions,
        "point_barriers": point_barriers,
        "line_barriers": line_barriers,
        "polygon_barriers": polygon_barriers,
        "use_hierarchy": use_hierarchy,
        "restrictions": restrictions,
        "attribute_parameter_values": attribute_parameter_values,
        "allocation_line_shape": allocation_line_shape,
        "travel_mode": travel_mode,
        "impedance": impedance,
        "save_output_network_analysis_layer": save_output_network_analysis_layer,
        "overrides": overrides,
        "time_impedance": time_impedance,
        "distance_impedance": distance_impedance,
        "output_format": output_format,
        "gis": gis,
        "future": True,
    }
    params = inspect_function_inputs(tbx.solve_location_allocation, **params)
    params["future"] = True
    job = tbx.solve_location_allocation(**params)
    if future:
        return job
    return job.result()


solve_location_allocation.__annotations__ = {
    "facilities": FeatureSet,
    "demand_points": FeatureSet,
    "measurement_units": str,
    "analysis_region": str,
    "problem_type": str,
    "number_of_facilities_to_find": int,
    "default_measurement_cutoff": float,
    "default_capacity": float,
    "target_market_share": float,
    "measurement_transformation_model": str,
    "measurement_transformation_factor": float,
    "travel_direction": str,
    "time_of_day": datetime,
    "time_zone_for_time_of_day": str,
    "uturn_at_junctions": str,
    "point_barriers": FeatureSet,
    "line_barriers": FeatureSet,
    "polygon_barriers": FeatureSet,
    "use_hierarchy": bool,
    "restrictions": str,
    "attribute_parameter_values": FeatureSet,
    "allocation_line_shape": str,
    "travel_mode": str,
    "impedance": str,
    "return": tuple,
}
