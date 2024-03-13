from __future__ import annotations
from arcgis.auth.tools._lazy import LazyLoader

from typing import Any
from dataclasses import dataclass, field
from enum import Enum

arcgis = LazyLoader("arcgis")
__all__ = ["TraversabilityScopeEnum", "FilterScopeEnum"]


###########################################################################
def _parse_enum(value: Enum | Any | None) -> Any | None:
    """returns the Enum's value or the current value"""
    if isinstance(value, Enum):
        return value.value
    else:
        return value


###########################################################################
class TraversabilityScopeEnum(Enum):
    JUNCTIONS = "junctions"
    EDGES = "edges"
    JUNCTIONS_AND_EDGES = "junctionsAndEdges"


###########################################################################
class FilterScopeEnum(Enum):
    JUNCTIONS = "junctions"
    EDGES = "edges"
    JUNCTIONS_AND_EDGES = "junctionsAndEdges"


###########################################################################
@dataclass
class TraceConfiguration:
    """
    ========================================        ==========================================================
    **Parameter**                                    **Description**
    ----------------------------------------        ----------------------------------------------------------
    domain_network_name                             Required string. Specifies the name of the domain network
                                                    where the trace is starting. This is required for
                                                    subnetwork-based traces.
    ----------------------------------------        ----------------------------------------------------------
    tier_name                                       Required string. Specifies the name of the tier where the
                                                    trace is starting. This is required for subnetwork-based
                                                    traces.
    ----------------------------------------        ----------------------------------------------------------
    shortest_path_network_attribute_name            Required string for a shortest path trace; otherwise,
                                                    it's optional. It specifies the network attribute name
                                                    used for determining cost when calculating the shortest path.
    ----------------------------------------        ----------------------------------------------------------
    propagators                                     Required list of dictionaries. This is an array of objects.
                                                    The default is null.

                                                    Syntax:

                                                        [
                                                            {
                                                            "networkAttributeName" : "<string>",
                                                            "substitutionAttributeName": "<string>",
                                                            "propagatorfunctionType" : "bitwiseAnd" | "min", | "max"
                                                            "operator : "equal" | "notEqual"
                                                                        | "greaterThan"
                                                                        | "greaterThanEqual"
                                                                        | "lessThan"
                                                                        | "lessThanEqual"
                                                                        | "includesTheValues”
                                                                        | "doesNotIncludeTheValues"
                                                                        | "includesAny"
                                                                        | "doesNotIncludeAny",
                                                            "value" : string (numeric),
                                                            "propagatedAttributeName": "<string>"
                                                            }
                                                        ]
    ----------------------------------------        ----------------------------------------------------------
    target_tier_name                                Optional string. Specifies the name of the tier where an
                                                    upstream or downstream trace ends.
    ----------------------------------------        ----------------------------------------------------------
    subnetwork_name                                 Optional string. Specifies the name of the subnetwork that
                                                    be traced. The starting points of the trace are the controllers
                                                    of this subnetwork.
    ----------------------------------------        ----------------------------------------------------------
    filter_bitset_network_attribute_name            Optional string used during a loops trace to only return
                                                    loops with the same bit set all around the loop. This is
                                                    used during upstream and downstream traces to ensure that
                                                    trace results include any bit that is set in the starting
                                                    points for the network attribute.
    ----------------------------------------        ----------------------------------------------------------
    traversability_scope                            Optional TraversibilityScopeEnum or string specifying the
                                                    network element types to which the condition, category, or
                                                    function barriers apply. The default is junctionsAndEdges.
    ----------------------------------------        ----------------------------------------------------------
    filter_scope                                    Optional FilterScopeEnum or string specifying the network
                                                    element types to which the filter barriers or filter
                                                    function barriers apply. The default is junctionsAndEdges.
    ----------------------------------------        ----------------------------------------------------------
    condition_barriers                              Optional list of dictionaries.Each dictionary represents
                                                    network attribute or category conditions that serve as
                                                    barriers (the default is null).
                                                    If is_specific_value is true, the network attribute is
                                                    compared to a specific value; otherwise, the network
                                                    attribute is compared to another network attribute.

                                                    Syntax:

                                                        [
                                                            {
                                                                "name" : <string>,
                                                                "operator" : "equal" | "notEqual"
                                                                        | "greaterThan"
                                                                        | "greaterThanEqual"
                                                                        | "lessThan"
                                                                        | "lessThanEqual"
                                                                        | "includesTheValues"
                                                                        | "doesNotIncludeTheValues"
                                                                        | "includesAny"
                                                                        | "doesNotIncludeAny",
                                                                "value" : <string>,
                                                                "combineUsingOr" : <true | false>,
                                                                "isSpecificValue" : <true | false>
                                                            }
                                                        ]
    ----------------------------------------        ----------------------------------------------------------
    function_barriers                               Optional list of dictionaries. Each dictionary represents
                                                    a function barrier.

                                                    Syntax:

                                                        [
                                                            {
                                                                "functionType" : "add" | "subtract" |
                                                                            "average" | "count" |
                                                                            "min" | "max",
                                                                "networkAttributeName" : <string>,
                                                                "operator" : "equal" | "notEqual"
                                                                        | "greaterThan"
                                                                        | "greaterThanEqual"
                                                                        | "lessThan"
                                                                        | "lessThanEqual"
                                                                        | "includesTheValues"
                                                                        | "doesNotIncludeTheValues"
                                                                        | "includesAny"
                                                                        | "doesNotIncludeAny",
                                                                "value" : <string>,
                                                                "useLocalValues":true | false
                                                            }
                                                        ]
    ----------------------------------------        ----------------------------------------------------------
    filter_barriers                                 Optional list of dictionaries. Each dictionary represents
                                                    network attribute or category conditions that serve as
                                                    barriers (the default is null). If is_specific_value is
                                                    true, the network attribute is compared to a specific value;
                                                    otherwise, the network attribute is compared to
                                                    another network attribute.

                                                    Syntax:

                                                        [
                                                            {
                                                                "name" : <string>,
                                                                "operator" : "equal" | "notEqual"
                                                                        | "greaterThan"
                                                                        | "greaterThanEqual"
                                                                        | "lessThan"
                                                                        | "lessThanEqual"
                                                                        | "includesTheValues"
                                                                        | "doesNotIncludeTheValues"
                                                                        | "includesAny"
                                                                        | "doesNotIncludeAny",
                                                                "value" : string (numeric),
                                                                "combineUsingOr" : <true | false>,
                                                                "isSpecificValue" : <true | false>
                                                            }
                                                        ]
    ----------------------------------------        ----------------------------------------------------------
    filter_function_barriers                        Optional list of dictionaries. Each dictionary represents
                                                    a filter function barrier.

                                                    Syntax:

                                                        [
                                                            {
                                                                "functionType" : "add" | "subtract" |
                                                                            "average" | "count" |
                                                                            "min" | "max",
                                                                "networkAttributeName" : <string>,
                                                                "operator" : "equal" | "notEqual"
                                                                        | "greaterThan"
                                                                        | "greaterThanEqual"
                                                                        | "lessThan"
                                                                        | "lessThanEqual"
                                                                        | "includesTheValues"
                                                                        | "doesNotIncludeTheValues"
                                                                        | "includesAny"
                                                                        | "doesNotIncludeAny",
                                                                "value" : string (numeric)
                                                            }
                                                        ]
    ----------------------------------------        ----------------------------------------------------------
    functions                                       Optional list of dictionaries. Each dictionary represents
                                                    a function. Each function can have an optional list of
                                                    network attribute conditions.

                                                    Syntax:

                                                        [
                                                            {
                                                                "functionType" : "add" | "subtract"
                                                                            | "average" | "count"
                                                                            | "min" | "max",
                                                                "networkAttributeName" : <string>,
                                                                "conditions":
                                                                [
                                                                {
                                                                    "name" : <string>,
                                                                    "type" : "networkAttribute" | "category",
                                                                    "operator" : "equal" | "notEqual"
                                                                        | "greaterThan"
                                                                        | "greaterThanEqual |
                                                                        | "lessThan"
                                                                        | "lessThanEqual"
                                                                        | "includesTheValues"
                                                                        | "doesNotIncludeTheValues"
                                                                        | "includesAny"
                                                                        | "doesNotIncludeAny",
                                                                    "value" : <string>,
                                                                    "combineUsingOr" : <true | false>,
                                                                    "isSpecificValue" : <true | false>
                                                                }
                                                            ]
                                                            }

                                                        ]
    ----------------------------------------        ----------------------------------------------------------
    nearest_neighbor                                Optional dictionary that specifies the parameters needed
                                                    for calculating nearest neighbors.

                                                    Syntax:

                                                        {
                                                            "count" : int
                                                            "costNetworkAttributeName" : string,
                                                            "nearestCategories" : array of string,
                                                            "nearestAssets"
                                                            [
                                                                {
                                                                "networkSourceId" : long,
                                                                "assetGroupCode" : long,
                                                                "assetTypeCode" : long
                                                                }
                                                            ]
                                                        }
    ----------------------------------------        ----------------------------------------------------------
    output_filters                                  Optional list of dictionaries specifying the output filter.

                                                    Syntax:

                                                        [
                                                            {
                                                                "networkSourceId" : long,
                                                                "assetGroupCode" : long,
                                                                "assetTypeCode" : long
                                                            }
                                                        ]
    ----------------------------------------        ----------------------------------------------------------
    output_conditions                               Optional list of dictionaries specifying the type of features
                                                    returned based on a network attribute or category.

                                                    Syntax:

                                                        [
                                                            {
                                                                "name": "<string>",
                                                                "type": "networkAttribute" | "category",
                                                                "operator": "equal" | "notEqual"
                                                                            | "greaterThan"
                                                                            | "greaterThanEqual"
                                                                            | "lessThan"
                                                                            | "lessThanEqual"
                                                                            | "includesTheValues"
                                                                            | "doesNotIncludeTheValues"
                                                                            | "includesAny"
                                                                            | "doesNotIncludeAny",
                                                                    "value": <string>,
                                                                "combineUsingOr": <true | false>,
                                                                "isSpecificValue": <true | false>
                                                            }
                                                        ]
    ----------------------------------------        ----------------------------------------------------------
    include_containers                              Optional property specifying whether to include containers
                                                    in the trace result. The default is false.
    ----------------------------------------        ----------------------------------------------------------
    include_content                                 Optional property specifying whether to include content
                                                    in the trace result. The default is false.
    ----------------------------------------        ----------------------------------------------------------
    include_structures                              Optional property specifying whether to include structures
                                                    in the trace result. The default is false.
    ----------------------------------------        ----------------------------------------------------------
    include_barriers                                Optional property specifying whether to include barriers
                                                    in the trace result. The default is true.
    ----------------------------------------        ----------------------------------------------------------
    validate_consistency                            Optional property specifying whether to validate the
                                                    consistency of the trace results. The default is true.
    ----------------------------------------        ----------------------------------------------------------
    validate_locatability                           Optional property specifying whether to validate whether
                                                    traversed junction or edge objects have the necessary
                                                    containment, attachment, or connectivity association in
                                                    their association hierarchy. The default is false.
    ----------------------------------------        ----------------------------------------------------------
    include_isolated                                Optional property specifying whether to include isolated
                                                    features for an isolation trace. The default is false.
    ----------------------------------------        ----------------------------------------------------------
    ignore_barriers_at_starting_points              Optional property specifying whether dynamic barriers in
                                                    the trace configuration are ignored for starting points.
                                                    This can be useful when performing an upstream protective
                                                    device trace using the discovered protective devices
                                                    (barriers) as starting points to find subsequent upstream
                                                    protective devices. The default is false.
    ----------------------------------------        ----------------------------------------------------------
    include_up_to_first_spatial_container           Optional property specifying whether to limit the containers
                                                    returned in the trace result. This property depends on the
                                                    include_containers property and no-ops if include_containers
                                                    is false. If includeContainers is true and this property
                                                    is true, containment associations up to and including the
                                                    first spatial container are returned; otherwise, all c
                                                    ontainment associations are returned. The default is false.
    ----------------------------------------        ----------------------------------------------------------
    allow_indeterminate_flow                        Optional property specifying whether network features
                                                    with indeterminate flow stop traversability or are included
                                                    in the trace results. This property is only honored when
                                                    running an upstream, downstream, or isolation trace.
    ========================================        ==========================================================
    """

    domain_network_name: str
    tier_name: str
    shortest_path_network_attribute_name: str | None = None
    propagators: list[dict] | None = None
    target_tier_name: str | None = None
    subnetwork_name: str | None = None
    filter_bitset_network_attribute_name: str | None = None
    traversability_scope: TraversabilityScopeEnum | str | None = (
        TraversabilityScopeEnum.JUNCTIONS_AND_EDGES
    )
    filter_scope: FilterScopeEnum | str | None = FilterScopeEnum.JUNCTIONS_AND_EDGES
    condition_barriers: list[dict] | None = None
    function_barriers: list[dict] | None = None
    filter_barriers: list[dict] | None = None
    filter_function_barriers: list[dict] | None = None
    functions: list[dict] | None = None
    nearest_neighbor: dict | None = None
    output_filters: list[dict] | None = None
    output_conditions: list[dict] | None = None
    include_containers: bool = False
    include_content: bool = False
    include_structures: bool = False
    include_barriers: bool = True
    validate_consistency: bool = True
    validate_locatability: bool = False
    include_isolated: bool = False
    ignore_barriers_at_starting_points: bool = False
    include_up_to_first_spatial_container: bool = False
    allow_indeterminate_flow: bool | None = None
    _dict_data: dict | None = field(init=False)

    def __str__(self):
        return f"<TraceConfiguration: Domain Network Name= {self.domain_network_name}, Tier Name= {self.tier_name}>"

    def __repr__(self):
        return self.__str__()

    def __post_init__(self):
        self._dict_data = {
            "domainNetworkName": self.domain_network_name,
            "tierName": self.tier_name,
            "shortestPathNetworkAttributeName": self.shortest_path_network_attribute_name
            or "",
            "propagators": self.propagators or [],
            "targetTierName": self.target_tier_name or "",
            "subnetworkName": self.subnetwork_name or "",
            "filterBitsetNetworkAttributeName": self.filter_bitset_network_attribute_name
            or "",
            "traversabilityScope": _parse_enum(self.traversability_scope),
            "filterScope": _parse_enum(self.filter_scope),
            "conditionBarriers": self.condition_barriers or [],
            "functionBarriers": self.function_barriers or [],
            "filterBarriers": self.filter_barriers or [],
            "filterFunctionBarriers": self.filter_function_barriers or [],
            "functions": self.functions or [],
            "nearestNeighbor": self.nearest_neighbor or {},
            "outputFilters": self.output_filters or [],
            "outputConditions": self.output_conditions or [],
            "includeContainters": self.include_containers,
            "includeContent": self.include_content,
            "includeStructures": self.include_structures,
            "includeBarriers": self.include_barriers,
            "ValidateConsistency": self.validate_consistency,
            "validateLocatability": self.validate_locatability,
            "includeIsolated": self.include_isolated,
            "ignoreBarriersAtStartingPoints": self.ignore_barriers_at_starting_points,
            "includeUpToFirstSpatialContainer": self.include_up_to_first_spatial_container,
            "allowIndeterminateFlow": self.allow_indeterminate_flow,
        }

    def to_dict(self):
        return {
            "domainNetworkName": self.domain_network_name,
            "tierName": self.tier_name,
            "shortestPathNetworkAttributeName": self.shortest_path_network_attribute_name,
            "propagators": self.propagators,
            "targetTierName": self.target_tier_name,
            "subnetworkName": self.subnetwork_name,
            "filterBitsetNetworkAttributeName": self.filter_bitset_network_attribute_name,
            "traversabilityScope": _parse_enum(self.traversability_scope),
            "filterScope": _parse_enum(self.filter_scope),
            "conditionBarriers": self.condition_barriers,
            "functionBarriers": self.function_barriers,
            "filterBarriers": self.filter_barriers,
            "filterFunctionBarriers": self.filter_function_barriers,
            "functions": self.functions,
            "nearestNeighbor": self.nearest_neighbor,
            "outputFilters": self.output_filters,
            "outputConditions": self.output_conditions,
            "includeContainters": self.include_containers,
            "includeContent": self.include_content,
            "includeStructures": self.include_structures,
            "includeBarriers": self.include_barriers,
            "ValidateConsistency": self.validate_consistency,
            "validateLocatability": self.validate_locatability,
            "includeIsolated": self.include_isolated,
            "ignoreBarriersAtStartingPoints": self.ignore_barriers_at_starting_points,
            "includeUpToFirstSpatialContainer": self.include_up_to_first_spatial_container,
            "allowIndeterminateFlow": self.allow_indeterminate_flow,
        }

    @classmethod
    def from_config(cls, config) -> TraceConfiguration:
        """Loads the Trace Configurations from a Configuration"""
        return TraceConfiguration(
            domain_network_name=config["domainNetworkName"],
            tier_name=config["tierName"],
            shortest_path_network_attribute_name=config[
                "shortestPathNetworkAttributeName"
            ],
            propagators=config["propagators"],
            target_tier_name=config["targetTierName"],
            subnetwork_name=config["subnetworkName"],
            filter_bitset_network_attribute_name=config[
                "filterBitsetNetworkAttributeName"
            ],
            traversability_scope=config["traversabilityScope"],
            filter_scope=config["filterScope"],
            condition_barriers=config["conditionBarriers"],
            function_barriers=config["functionBarriers"],
            filter_barriers=config["filterBarriers"],
            filter_function_barriers=config["filterFunctionBarriers"],
            functions=config["functions"],
            nearest_neighbor=config["nearestNeighbor"],
            output_filters=config["outputFilters"],
            output_conditions=config["outputConditions"],
            include_containers=config["includeContainers"],
            include_content=config["includeContent"],
            include_structures=config["includeStructures"],
            include_barriers=config["includeBarriers"],
            validate_consistency=config["validateConsistency"],
            validate_locatability=config["validateLocatability"],
            include_isolated=config["includeIsolated"],
            ignore_barriers_at_starting_points=config["ignoreBarriersAtStartingPoints"],
            include_up_to_first_spatial_container=config[
                "includeUpToFirstSpatialContainer"
            ],
            allow_indeterminate_flow=config["allowIndeterminateFlow"],
        )
