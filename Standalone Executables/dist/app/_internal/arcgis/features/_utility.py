from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Union
from arcgis import env
from arcgis._impl.common._mixins import PropertyMap
from arcgis.features._trace_configuration import TraceConfiguration
from arcgis._impl.common._deprecate import deprecated


########################################################################
class UtilityNetworkManager(object):
    """
    The Utility Network Service exposes analytic capabilities (tracing)
    as well as validation of network topology and management of
    subnetworks (managing sources, updating subnetworks, exporting
    subnetworks, and so on). The Utility Network Service is conceptually
    similar to the Network Analysis Service for transportation networks.

    =====================   ===========================================
    **Inputs**              **Description**
    ---------------------   -------------------------------------------
    url                     Required String. The web endpoint to the utility service.
    ---------------------   -------------------------------------------
    version                 Optional :class:`~arcgis.features._version.Version`. The `Version` class where the branch version will take place.
    ---------------------   -------------------------------------------
    gis                     Optional :class:`~arcgis.gis.GIS` . The `GIS` connection object.
    =====================   ===========================================


    """

    _con = None
    _gis = None
    _url = None
    _property = None
    _version_guid = None
    _version_name = None
    _version = None

    # ----------------------------------------------------------------------
    def __init__(self, url, version=None, gis=None):
        """Constructor"""
        if gis is None:
            gis = env.active_gis
        self._gis = gis
        self._con = gis._portal.con
        self._url = url
        if version:
            self._version = version
            self._version_name = version.properties.versionName
            if self._version_name in ["SDE.DEFAULT", "DBO.DEFAULT"]:
                # fix due to locking issue
                self._version_guid = None
            else:
                self._version_guid = version._guid
        else:
            self._version = None
            self._version_guid = None
            self._version_name = None

    # ----------------------------------------------------------------------
    def _init(self):
        """initializer"""
        try:
            res = self._con.get(self._url, {"f": "json"})
            self._property = PropertyMap(res)
        except Exception as e:
            self._property = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return f"< Utility Network Server @ {self._url} >"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"< Utility Network Server @ {self._url} >"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties for the service"""
        if self._property is None:
            self._init()
        return self._property

    # ----------------------------------------------------------------------
    def trace(
        self,
        locations: list[dict],
        trace_type: str,
        moment: int | None = None,
        configuration: dict | TraceConfiguration | None = None,
        result_type: str | None = None,
        result_types: list[dict] | None = None,
        trace_config_global_id: str | None = None,
        out_sr: int | None = None,
        pbf: bool = False,
    ) -> dict:
        """
        A trace refers to a pre-configured algorithm that systematically
        travels a network to return results. Generalized traces allow you to
        trace across multiple types of domain networks. For example, running
        a Connected trace from your electric network through to your gas
        network. An assortment of options is provided with trace to support
        various analytic work flows. All traces use the network topology to
        read cached information about network features. This can improve
        performance of complex traces on large networks. Trace results are
        not guaranteed to accurately represent a utility network when dirty
        areas are present. The network topology must be validated to ensure
        it reflects the most recent edits or updates made to the network.

        .. note::
            The active portal account must be licensed with the ArcGIS Utility
            Network user type extention to use this operation.

        =======================    ==================================================
        **Parameter**              **Description**
        -----------------------    --------------------------------------------------
        locations                  Required list of dictionaries. The locations for
                                   starting points and barriers. An empty array must
                                   be used when performing a subnetwork trace if a
                                   subnetworkName is provided as part of the
                                   `configuration`—for example, `locations=[]`.


                                   The location is ignored by the trace if the following
                                   required properties are not defined:

                                   * `percentAlong` : required for edge features and objects.
                                   * `terminalID` : required for junction features and objects.


                                   .. code-block:: python

                                       [{
                                           "traceLocationType" : "startingPoint" | "barrier",
                                           "globalId" : <guid>,
                                           "terminalId" : <int>,   // optional
                                           "percentAlong" : <double>, // optional
                                           "isFilterBarrier" : true | false // optional Introduced at 10.8.1
                                       }]
        -----------------------    --------------------------------------------------
        trace_type                 Required string. Specifies the core algorithm that
                                   will be executed to analyze the network. Can be
                                   configured using the `configuration` parameter.

                                   Values:

                                    'connected' | 'subnetwork' | 'subnetworkController' | 'upstream' | 'downstream' | 'loops' | 'shortestPath' | 'isolation'
        -----------------------    --------------------------------------------------
        moment                     Optional Integer. Specifies the session moment. This
                                   should only be specified if you do not want to use
                                   the current moment.

                                   Example: moment = <Epoch time in milliseconds>
        -----------------------    --------------------------------------------------
        configuration              Optional dictionary or TraceConfiguration object.
                                   Specifies the collection of trace configuration
                                   properties. Depending on the `trace_type`, some
                                   properties are required.

                                   To see all configuration properties see:
                                   `Trace Configuration Properties
                                   <https://developers.arcgis.com/rest/services-reference/enterprise/trace-utility-network-server-.htm#GUID-F0C932FD-B403-4223-9B00-E44D156C7DF9/>`_
        -----------------------    --------------------------------------------------
        result_types               Optional parameter specifying hte types of results
                                   to return.

                                   .. code-block::
                                       [{
                                           "type" : "elements" | "aggregatedGeometry" | "connectivity",
                                           "includeGeometry" : true | false,
                                           "includePropagatedValues": true | false,
                                           "networkAttributeNames" :["attribute1Name","attribute2Name",...],
                                           "diagramTemplateName": <value>,
                                           "resultTypeFields":[{"networkSourceId":<int>,"fieldname":<value>},...]
                                       },...]
        -----------------------    --------------------------------------------------
        trace_config_global_id     Optional String. The global ID of a named trace configuration.
                                   When specified, this configuration is used instead of the
                                   traceConfiguration parameter. Additionally, named trace
                                   configurations are persisted with their own trace type so the
                                   trace type parameter is ignored.
        -----------------------    --------------------------------------------------
        out_sr                     Optional Integer. The output spatial reference.
        -----------------------    --------------------------------------------------
        pbf                        Optional Boolean. If True, the results are returned in
                                   the PBF format. The default is False.
        =======================    ==================================================

        :return:
            A dictionary with keys and value types of:

                | {
                |    "traceResults": {
                |        "elements": list,
                |        "diagramName": str,
                |        "globalFunctionResults": list,
                |        "kFeaturesForKNNFound": bool,
                |        "startingPointsIgnored" bool,
                |        "warnings": list
                |    }
                |    "success": bool
                | }

        """
        url = "%s/trace" % self._url
        if isinstance(configuration, TraceConfiguration):
            configuration = configuration.to_dict()
        params = {
            "f": "pbf" if pbf is True else "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "traceType": trace_type,
            "moment": moment,
            "traceLocations": locations,
            "traceConfiguration": configuration,
        }
        if trace_config_global_id:
            params["traceConfigurationGlobalId"] = trace_config_global_id
        if self._gis.version <= [7, 3]:
            params["resultType"] = result_type
        else:
            params["resultTypes"] = result_types
        if out_sr:
            params["outSR"] = out_sr
        if pbf is True:
            return self._con.post(url, params, force_bytes=True)
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def disable_topology(self) -> dict:
        """
        Disables the network topology for a utility network. When the
        topology is disabled, feature and association edits do not generate
        dirty areas. Analytics and diagram generation can't be performed if
        the topology is not present.

        When the topology is disabled, the following happens:

        - All current rows in the topology tables are deleted.
        - No dirty areas are generated from edits.
        - Remaining error features still exist and can be cleaned up without the overhead of dirty areas.

        To perform certain network configuration tasks, the network
        topology must be disabled.

        - This operation must be executed by the portal utility network owner.
        - The topology can be disabled in the default version or in a named version.

        :return: Dictionary indicating 'success' or 'error'

        """
        url = "%s/disableTopology" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def enable_topology(self, error_count: int = 10000) -> dict:
        """
        Enabling the network topology for a utility network is done on the
        **DEFAULT** version. Enabling is **not** supported in named versions.
        When the topology is enabled, all feature and association edits
        generate dirty areas, which are then consumed when the network
        topology is updated.

        When topology is enabled, the following happens:
        - Any existing errors are deleted.
        - The topology is updated for the full extent of the network.
        - Any newly discovered errors are added to the dirty areas sublayer.
        - The topology is marked as enabled.

        .. note::
            The active portal account must be licensed with the ArcGIS Utility
            Network user type extension to use this operation.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        error_count                              Optional Integer. Sets the threshold when the `enable_topology` will
                                                 stop if the maximum number of errors is met. The default value is
                                                 10,000.
        ====================================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        if self._version_name.lower().find("default") == -1:
            raise Exception("Current version is not the `DEFAULT` version.")

        params = {"f": "json", "maxErrorCount": error_count}
        url = "%s/enableTopology" % self._url
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def disable_subnetwork_controller(
        self,
        network_source_id: str,
        global_id: str,
        terminal_id: str,
        out_sr: int | None = None,
    ) -> dict:
        """
        A subnetwork controller (or simply, a source or a sink) is the
        origin (or destination) of resource flow for a subpart of the
        network. Examples of subnetwork controllers are circuit breakers in
        electric networks, or town border stations in gas networks.
        Subnetwork controllers correspond to devices that have the
        Subnetwork Controller network capability set. A source is removed
        with `disable_subnetwork_controller`.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        network_source_id                           Required String. The network source ID that the subnetwork controller
                                                    participates in.
        ------------------------------------        --------------------------------------------------------------------
        global_id                                   Required String. The global ID of the device being disabled as a
                                                    network controller.
        ------------------------------------        --------------------------------------------------------------------
        terminal_id                                 Required String. The terminal ID of the device being disabled as a
                                                    network controller.
        ------------------------------------        --------------------------------------------------------------------
        out_sr                                      Required int. The output spatial reference as a wkid.
        ====================================        ====================================================================

        """

        url = "%s/disableSubnetworkController" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "networkSourceId": network_source_id,
            "featureGlobalId": global_id,
            "terminalId": terminal_id,
        }
        if out_sr:
            params["outSR"] = out_sr
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def enable_subnetwork_controller(
        self,
        network_source_id: str,
        global_id: str,
        terminal_id: str,
        subnetwork_controller_name: str,
        tier_name: str,
        subnetwork_name: str | None = None,
        description: str | None = None,
        notes: str | None = None,
        out_sr: int | None = None,
    ) -> dict:
        """
        A subnetwork controller is the origin (or destination) of resource
        flow for a subpart of the network (e.g., a circuit breaker in
        electric networks, or a town border station in gas networks).
        Controllers correspond to Devices that have the Subnetwork
        Controller network capability set.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        network_source_id                           Required String. The network source ID that the subnetwork controller
                                                    participates in.
        ------------------------------------        --------------------------------------------------------------------
        global_id                                   Required String. The global ID of the device being enabled as a
                                                    network controller.
        ------------------------------------        --------------------------------------------------------------------
        terminal_id                                 Required String. The terminal ID of the device being enabled as a
                                                    network controller.
        ------------------------------------        --------------------------------------------------------------------
        subnetwork_controller_name                  Required String. The name of the subnetwork controller.
        ------------------------------------        --------------------------------------------------------------------
        tier_name                                   Required String. The name of the tier.
        ------------------------------------        --------------------------------------------------------------------
        subnetwork_name                             Optional String. Specifies the name of the subnetwork.
        ------------------------------------        --------------------------------------------------------------------
        description                                 Optional String. Represents the description of the subnetwork controller.
        ------------------------------------        --------------------------------------------------------------------
        notes                                       Optional String. The notes associated with the subnetwork controller.
        ------------------------------------        --------------------------------------------------------------------
        out_sr                                      Optional Integer. The output spatial reference as a wkid (integer).
        ====================================        ====================================================================

        """

        url = "%s/enableSubnetworkController" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "networkSourceId": network_source_id,
            "featureGlobalId": global_id,
            "terminalID": terminal_id,
            "subnetworkControllerName": subnetwork_controller_name,
            "subnetworkName": subnetwork_name,
            "tierName": tier_name,
            "description": description,
            "notes": notes,
        }
        if not out_sr is None:
            params["outSR"] = out_sr
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def export_subnetwork(
        self,
        domain_name: str,
        tier_name: str,
        subnetwork_name: str,
        trace_configuration: dict | TraceConfiguration | None = None,
        export_acknowledgement: bool = False,
        result_type: str | None = None,
        result_types: list[dict] | None = None,
        moment: int | None = None,
        run_async: bool = False,
        out_sr: int | None = None,
        pbf: bool = False,
    ) -> dict:
        """
        The `export_subnetwork` operation is used to export information
        about a subnetwork into a JSON file. That information can then be
        consumed by outside systems such as outage management and asset
        tracking. The exportSubnetwork operation allows you to delete
        corresponding rows in the Subnetwork Sources table as long as the
        IsDeleted attribute is set to True. This indicates a source feeding
        the subnetwork has been removed.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        domain_name                                 Required String. The name of the domain network of which the subnetwork
                                                    is a part.
        ------------------------------------        --------------------------------------------------------------------
        tier_name                                   Required String. The name of the tier of which the subnetwork is a part.
        ------------------------------------        --------------------------------------------------------------------
        subnetwork_name                             Required String. The name of the subnetwork.
        ------------------------------------        --------------------------------------------------------------------
        trace_configuration                         Optional Dictionary or TraceConfiguration object. Specifies the collection of trace
                                                    configuration parameters.
                                                    See: `Trace <https://developers.arcgis.com/rest/services-reference/enterprise/trace-utility-network-server-.htm#GUID-F0C932FD-B403-4223-9B00-E44D156C7DF9/>`_
        ------------------------------------        --------------------------------------------------------------------
        export_acknowledgement                      Optional Boolean. Specify whether the export is acknowledged.
        ------------------------------------        --------------------------------------------------------------------
        result_types                                Optional list of dictionaries. Specifies the type of results to return.

                                                    .. code-block:: python

                                                        [
                                                            {
                                                                "type" : "features" | "geometries" | "network" | "connectivity" | "controllers" | "associations" | "aggregatedGeometry" |
                                                                "diagram" | "elements" |  "associations",
                                                                "includeGeometry" : true | false,
                                                                "includePropagatedValues": true | false,
                                                                "includeDomainDescriptions": true | false,
                                                                "networkAttributeNames" :["attribute1Name","attribute2Name",...],
                                                                "diagramTemplateName": <value>,
                                                                "resultTypeFields":[{"networkSourceId":<int>,"fieldname":<value>},...]
                                                            },...
                                                        ]
        ------------------------------------        --------------------------------------------------------------------
        moment                                      Optional Integer. Specify the session moment if you do not want to use
                                                    the current moment.
        ------------------------------------        --------------------------------------------------------------------
        out_sr                                      Optional Integer. Optional parameter specifying the output spatial reference.
        ------------------------------------        --------------------------------------------------------------------
        pbf                                         Optional Boolean. If true, the response will be in PBF format.
        ====================================        ====================================================================

        :return:
            A dictionary with keys and value types of:

                | {
                |    "moment": int,
                |    "url": str,
                |    "subnetworkHasBeenDeleted": bool,
                |    "success": bool
                | }

        """

        url = "%s/exportSubnetwork" % self._url
        if isinstance(trace_configuration, TraceConfiguration):
            trace_configuration = trace_configuration.to_dict()
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
            "domainNetworkName": domain_name,
            "tierName": tier_name,
            "subnetworkName": subnetwork_name,
            "exportAcknowledgement": export_acknowledgement,
            "traceConfiguration": trace_configuration,
            "async": run_async,
        }
        if self._gis.version <= [7, 3]:
            params["resultType"] = result_type
        else:
            params["resultTypes"] = result_types
        if out_sr:
            params["outSR"] = out_sr
        if pbf:
            params["f"] = "pbf"
            return self._con.post(url, params, force_bytes=True)
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_network_moments(
        self,
        moments_to_return: list[str] | None = None,
        moment: int | None = None,
    ) -> dict:
        """
        The `query_network_moments` operation returns the moments related
        to the network topology and operations against the topology. This
        includes when the topology was initially enabled, when it was last
        validated, when the topology was last disabled (and later enabled),
        and when the definition of the utility network was last modified.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        moments_to_return                           Optional List of Strings. Represents the collection of validate moments to
                                                    return. Default is all.

                                                    Values:

                                                            [ "initialEnableTopology" | "fullValidateTopology" | "partialValidateTopology" | "enableTopology" | "disableTopology" | "definitionModification" | "updateIsConnected" | "indexUpdate" | "all"]
                                                    Example:

                                                        moments_to_return=["enableTopology","initialEnableTopology"]
        ------------------------------------        --------------------------------------------------------------------
        moment                                      Optional Integer. Specify the session moment if you do not want to use
                                                    the current moment.
        ====================================        ====================================================================

        :return:
            A dictionary with keys and value types of:

                | {"networkMoments": list,
                | "validateNetworkTopology": bool,
                | "success": bool}
        """
        url = "%s/queryNetworkMoments" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "momentsToReturn": moments_to_return
            if moments_to_return is not None
            else ["all"],
            "moment": moment,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    @deprecated(deprecated_in="2.1.0", removed_in=None, current_version="2.2.0")
    def query_overrides(
        self,
        attribute_ids: Optional[list[str]] = None,
        all_attributes: bool = False,
        all_connectivity: bool = False,
    ):
        """
        Network attributes support the ability to have their values
        overridden without having to edit features and validate the network
        topology (build the index). The utility network also supports the
        ability to place ephemeral connectivity (e.g., jumpers in an
        electrical network) between two devices or junctions without having
        to edit features or connectivity associations and validate the
        network topology (build the index). This operation allows the
        client to query all the overrides associated with the network
        attributes (by network attribute id). In addition, all connectivity
        overrides are returned.
        """
        url = "%s/queryOverrides" % self._url
        params = {
            "f": "json",
            "attributeIDs": attribute_ids,
            "allAttributes": all_attributes,
            "allConnectivity": all_connectivity,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def synthesize_association_geometries(
        self,
        attachment_associations: bool = False,
        connectivity_associations: bool = False,
        containment_associations: bool = False,
        count: int = 200,
        extent: dict = None,
        out_sr: int | dict[str, Any] | None = None,
        moment: int | None = None,
    ) -> dict:
        """
        The `synthesize_association_geometries` operation is used to export
        geometries representing associations that are synthesized as line
        segments corresponding to the geometries of the devices at the
        endpoints. All features associated with an association must be in
        the specified extent in order for the geometry to be synthesized.
        If only zero or one of the devices/junctions intersects the extent,
        then no geometry will be synthesized.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        attachment_associations                     Optional Boolean. Whether to return attachment associations.
        ------------------------------------        --------------------------------------------------------------------
        connectivity_associations                   Optional Boolean. Represents whether to return connectivity associations.
        ------------------------------------        --------------------------------------------------------------------
        containment_associations                    Optional Boolean. Whether to return containment associations.
        ------------------------------------        --------------------------------------------------------------------
        count                                       Required Int. Represents the maximum number of geometries that
                                                    can be synthesized and returned in the result.
        ------------------------------------        --------------------------------------------------------------------
        extent                                      Required Dictionary. Represents the envelope of the area to
                                                    synthesize association geometries.

                                                    .. code-block:: python

                                                        {
                                                            "xmin": <minimum x-coordinate>,
                                                            "ymin": <minimum y-coordinate>,
                                                            "xmax": <maximum x-coordinate>,
                                                            "ymax": <maximum y-coordinate>,
                                                            "spatialReference": {
                                                            "wkid": <spatial reference well-known identifier>,
                                                            "latestWkid": <the current wkid value associated with the wkid>
                                                            }
                                                        }
        ------------------------------------        --------------------------------------------------------------------
        out_sr                                      Optional Dictionary. Represents the output spatial reference.
        ------------------------------------        --------------------------------------------------------------------
        moment                                      Optional Integer. Specify the session moment if you do not want to use
                                                    the current moment.
        ====================================        ====================================================================

        :return:
            A dictionary with keys and value types of:

                | {"maxGeometryCountExceeded": bool,
                | "associations": list,
                | "success": bool}
        """
        url = "%s/synthesizeAssociationGeometries" % self._url
        params = {
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
            "attachmentAssociations": attachment_associations,
            "connectivityAssociations": connectivity_associations,
            "containmentAssociations": containment_associations,
            "maxGeometryCount": count,
            "extent": extent,
            "outSR": out_sr,
            "f": "json",
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def update_is_connected(self) -> dict:
        """

        Utility network features have an attribute called IsConnected that
        lets you know if a feature is connected to a source or sink, and
        therefore it could potentially be part of an existing subnetwork.
        The `update_is_connected` operation updates this attribute on
        features in the specified utility network. This operation can only
        be executed on the default version by the portal utility network
        owner.
        """
        url = "%s/updateIsConnected" % self._url
        params = {"f": "json"}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def update_subnetwork(
        self,
        domain_name: str,
        tier_name: str,
        subnetwork_name: str | None = None,
        all_subnetwork_tier: bool = False,
        continue_on_failure: bool = False,
        trace_configuration: dict | None = None,
    ) -> dict:
        """
        A subnetwork is updated by calling the `update_subnetwork` operation.
        With this operation, one or all of the subnetworks in a single tier
        can be updated. When a subnetwork is updated, four things can occur;
        the Subnetwork Name attribute is updated for all features in the
        subnetwork, the record representing the subnetwork inside the
        SubnetLine class is refreshed, the Subnetworks table is updated and
        finally diagrams are generated or updated for the subnetwork.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        domain_name                                 Required String. The name fo the domain network that the subnetwork
                                                    is a part of.
        ------------------------------------        --------------------------------------------------------------------
        tier_name                                   Required String. The name of the tier that the subnetwork is a part of.
        ------------------------------------        --------------------------------------------------------------------
        subnetwork_name                             Optional String. Represents the name of the subnetwork to update. If
                                                    this parameter is not specified, the `all_subnetwork_tier` parameter
                                                    should be set to `True`. Otherwise an error will occur.
        ------------------------------------        --------------------------------------------------------------------
        all_subnetwork_tier                         Optional Bool. Set to `True` when all the subnetworks in a tier
                                                    need to be updated.
        ------------------------------------        --------------------------------------------------------------------
        continue_on_failure                         Optional Bool. Continue updating subnetworks when `all_subnetwork_tier`
                                                    is `True` and a failure occurs when processing a subnetwork.
        ------------------------------------        --------------------------------------------------------------------
        trace_configuration                         Optional Dictionary. Represents the collection of trace configuration
                                                    parameters. See `trace` method to get parameters.
        ====================================        ====================================================================

        :return: Dictionary of the JSON response.

        """
        url = "%s/updateSubnetwork" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "domainNetworkName": domain_name,
            "tierName": tier_name,
            "subnetworkName": subnetwork_name,
            "allSubnetworksInTier": all_subnetwork_tier,
            "continueOnFailure": continue_on_failure,
            "traceConfiguration": trace_configuration,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def validate_topology(
        self,
        envelope: dict[str, Any],
        run_async: bool = False,
        return_edits: bool = False,
        validate_set: list[dict] | None = None,
        out_sr: int | None = None,
    ) -> dict:
        """
        Validating the network topology for a utility network maintains
        consistency between feature editing space and network topology space.
        Validating a network topology may include all or a subset of the
        dirty areas present in the network. Validation of network topology
        is supported synchronously and asynchronously.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        envelope                                    Required Dictionary. The envelope of the area to validate.

                                                    .. code-block:: python

                                                        {
                                                            "xmin": <minimum x-coordinate>,
                                                            "ymin": <minimum y-coordinate>,
                                                            "xmax": <maximum x-coordinate>,
                                                            "ymax": <maximum y-coordinate>,
                                                            "spatialReference": {
                                                            "wkid": <spatial reference well-known identifier>,
                                                            "latestWkid": <the current wkid value associated with the wkid>
                                                            }
                                                        }
        ------------------------------------        --------------------------------------------------------------------
        run_async                                   Optional Boolean. If Turem the request is processed as an asynchronous
                                                    job. The URL is returned to check the status of a job.
        ------------------------------------        --------------------------------------------------------------------
        return_edits                                Optional Boolean. Returned results are organized in a layer-by-layer fashion.
                                                    If `return_edits` is set to True, each layer may have edited features
                                                    returned in an editedFeatures object.
                                                    The editedFeatures object returns full features including the original
                                                    features prior to delete; the original and current features for updates;
                                                    and the current rows for inserts, which may contain implicit changes
                                                    (for example, as a result of a calculation rule).

                                                    The response includes no editedFeatures and 'exceededTransferLimit = true'
                                                    if the count of edited features to return is more than the maxRecordCount.
                                                    If clients are using this parameter to maintain a cache, they should
                                                    invalidate the cache when exceededTransferLimit = true is returned.
                                                    If the server encounters an error when generating the list
                                                    of edits is the response, exceededTransferLimit = true is also returned.

                                                    Edited features are returned in the spatial reference
                                                    of the feature service as defined by the service's spatialReferenceobject
                                                    or by the spatialReference of the layer's extent object.
        ------------------------------------        --------------------------------------------------------------------
        validate_set                                Optional List of Dictionary. Introduced at Enterprise 10.9.1, it specifies
                                                    the set of features and objects to validate.

                                                    .. code-block:: python

                                                        [
                                                            {
                                                                "sourceId": <int>,
                                                                "globalIds": [<guid>]
                                                            }
                                                        ]
        ------------------------------------        --------------------------------------------------------------------
        out_sr                                      Optional integer. The output spatial reference.
        ====================================        ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        url = "%s/validateNetworkTopology" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "validateArea": envelope,
            "async": run_async,
            "returnEdits": return_edits,
        }
        if self._gis.version >= [9, 2]:
            params["validateSet"] = validate_set
        if out_sr:
            params["outSR"] = out_sr
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    @deprecated(deprecated_in="2.1.0", removed_in=None, current_version="2.2.0")
    def apply_overrides(
        self,
        adds: Optional[Union[list, dict[str, Any]]] = None,
        deletes: Optional[Union[list, dict[str, Any]]] = None,
    ):
        """
        Network attributes support the ability to have their values
        overridden without having to edit features and validate the network
        topology (build the index). The utility network also supports the
        ability to place ephemeral connectivity (for example, jumpers in an
        electrical network) between two devices or junctions without having
        to edit features or connectivity associations and validate the
        network topology (build the index). When specified by the client, a
        trace operation may optionally incorporate the network attribute
        and connectivity override values when the trace is run on.


        """
        url = "%s/applyOverrides" % self._url
        params = {"f": "json", "adds": adds, "deletes": deletes}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def associations(self) -> dict:
        """
        The associations resource provides access to operations that
        allow you to query and extract useful information from the
        associations table of a utility network.

        Available starting at Enterprise 10.9.1

        :return: A dictionary with two keys

            {"associations":list, "success": bool}
        """

        if self._gis.version >= [9, 2]:
            url = "%s/associations" % self._url
            params = {"f": "json"}
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_associations(
        self,
        elements: list[dict] | None = None,
        moment: int | None = None,
        types: list[str] | None = None,
        return_deletes: bool = False,
    ) -> dict:
        """
        The query operation allows you to query the associations table
        and return association information for network features in a utility network.

        Available starting at Enterprise 10.9.1

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        elements                                    Required List of Dictionary. The feature or object elements for which
                                                    the association is querried.

                                                    .. code-block:: python

                                                        [{
                                                            "networkSourceId": <int>,
                                                            "globalId" : <guid>,
                                                            "terminalId": <int> //optional
                                                        }]
        ------------------------------------        --------------------------------------------------------------------
        moment                                      Optional Epoch time in milliseconds. Specify if you do not want to
                                                    use the current moment.
        ------------------------------------        --------------------------------------------------------------------
        types                                       Optional List of String(s). Specify the association types to be queried.

                                                    Values:

                                                        "connectivity" | "attachment" | "contianment" | "junctionEdgeFromConnectivity" | "junctionMidspanConnectivity" | "junctionEdgeToConnectivity"
        ------------------------------------        --------------------------------------------------------------------
        return_deletes                              Optional Boolean. Specify whether to return logically deleted associations.
        ====================================        ====================================================================

        """
        if self._gis.version >= [9, 2]:
            url = "%s/associations/query" % self._url
            params = {
                "f": "json",
                "gdbVersion": self._version_name,
                "elements": elements,
                "returnDeletes": return_deletes,
            }
            if elements:
                params["moment"] = moment
            if types:
                params["types"] = types

            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def traverse_associations(
        self,
        elements: list[dict],
        moment: int | None = None,
        type: str = "unspecified",
        direction: str = "descending",
        dirty_filter: str = "none",
        error_filter: str = "none",
        stop_at_first_spatial: bool = True,
        max_depth: int | None = None,
    ) -> dict:
        """
        The `traverse_associations` operation allows you to obtain and extract useful
        information from the associations table in a utility network.

        The `type` parameter is used to provide the following predefined traversal types:

        * dirtyAreaExpansion—Returns associations and objects that have been modified and are marked as dirty. Completes a downward traversal, followed by an ascending traversal, with an exit filter on the first spatial feature in each direction.

        * firstContainers—Completes an ascending traversal on containment associations, with an exit filter on the first spatial feature.

        * spatialParents—Completes an ascending traversal on all association types, with an exit filter on the first spatial feature.

        * topContainers—Completes an ascending traversal to return associations and objects with no exit filter.

        * errorsNotModified—Completes a downward traversal to return associations in error, with an exit filter on the first spatial feature.

        * modifiedObjects—Completes a downward traversal to return associations that are dirty, with an exit filter on the first spatial feature.

        To create a custom traversal the `direction`, `dirty_filter`, `error_filter`, `stop_at_first_spatial`, and `max_depth` parameters can be used. When a traversal type is specified using the type parameter other than the default `unspecified``, these parameters are ignored.

        Available starting at Enterprise 10.9.1

        .. note::
            Associations are not traversed from spatial features to nonspatial objects
            and back to spatial features when the exit filter is placed on the first
            spatial feature.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        elements                                    Required List of Dictionary. The feature or object elements for which
                                                    the association is queried.

                                                    .. code-block:: python

                                                        [{
                                                            "networkSourceId": <int>,
                                                            "globalId" : <guid>,
                                                            "terminalId": <int> //optional
                                                        }]
        ------------------------------------        --------------------------------------------------------------------
        moment                                      Optional Epoch time in milliseconds. Specify if you do not want to
                                                    use the current moment.
        ------------------------------------        --------------------------------------------------------------------
        type                                        Optional String. Specify the association types to be queried.

                                                    Values:

                                                        "unspecified" | "dirtyAreaExpansion" | "firstContainers" | "spatialParents" | "topContainers" | "errorsNotModified" | "modifiedObjects"
        ------------------------------------        --------------------------------------------------------------------
        direction                                   Optional String. Specify the direction of the association traversal.

                                                    Values:

                                                        "ascending" | "descending"
        ------------------------------------        --------------------------------------------------------------------
        dirty_filter                                Optional String. Specify whether to filter based on the dirty status
                                                    of the association.

                                                    .. note::
                                                        When `dirty_filter` and `error_filter` are specified together,
                                                        the filters are combined using the AND expression

                                                    Values:

                                                        "none" | "dirty" | "notDirty"
        ------------------------------------        --------------------------------------------------------------------
        error_filter                                Optional String. Specify whether to filter associations based on the
                                                    error code.

                                                    Values:

                                                       "none" | "inError" | "notInError"
        ------------------------------------        --------------------------------------------------------------------
        stop_at_first_spatial                       Optional Bool. Specify whether to stop the traversal of associations
                                                    from nonspatial objext to feature when a spatial feature is encountered.
                                                    The traversal will stop at the feature and will not traverse to the
                                                    next nonspatial object.
        ------------------------------------        --------------------------------------------------------------------
        max_depth                                   Optional Integer. Control how many hops through the association graph
                                                    are allowed in either the ascending or descending direction.
        ====================================        ====================================================================

        """
        if self._gis.version >= [9, 2]:
            url = "%s/associations/traverse" % self._url
            params = {
                "f": "json",
                "gdbVersion": self._version_name,
                "elements": elements,
                "moment": moment,
                "type": type,
                "direction": direction,
                "dirtyStatusFilter": dirty_filter,
                "errorStatusFilter": error_filter,
                "stopAtFirstSpatial": stop_at_first_spatial,
                "maxDepth": max_depth,
            }
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def locations(self) -> dict:
        """
        The `locations` resource provides access to an operation that allows
        you to query the locatability of a provided set of objects and
        optionally synthesize geometry to be returned.

        Introduced at Enterprise 10.9.1

        :return: "success" if able to reach locations, else "error"
        """

        if self._gis.version >= [9, 2]:
            url = "%s/locations" % self._url
            params = {"f": "json"}
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_locations(
        self,
        elements: list[dict],
        max_geom_count: int,
        moment: int | None = None,
        attachment_associations: bool = False,
        connectivity_associations: bool = False,
        containment_associations: bool = False,
        locations: bool = False,
        out_sr: int | dict | None = None,
    ) -> dict:
        """
        The query operation queries the locatability of the provided set of objects
        and optionally synthesizes geometry to be returned for each object in a
        geometry bag as a collection of points and polylines.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        elements                                    Required List of Dictionary. The set of objects for which to get
                                                    locatability and synthesize the geometries.

                                                    .. code-block:: python

                                                        [{
                                                            "sourceId": <int>,
                                                            "globalIds" : [<guid>],
                                                        }]
        ------------------------------------        --------------------------------------------------------------------
        max_geom_count                              Required Int. The maximum number of geometries that can be synthesized
                                                    and returned in the result.
        ------------------------------------        --------------------------------------------------------------------
        moment                                      Optional Epoch time in milliseconds. Specify if you do not want to
                                                    use the current moment.
        ------------------------------------        --------------------------------------------------------------------
        attachment_associations                     Optional Boolean. Whether to synthesize the geometry representing the
                                                    structural attachment associations.
        ------------------------------------        --------------------------------------------------------------------
        conectivity_associations                    Optional Boolean. Whether to synthesize the geometry representing the
                                                    connectivity associations.
        ------------------------------------        --------------------------------------------------------------------
        containment_associations                    Optional Boolean. Whether to synthesize the geometry representing the
                                                    containment associations.
        ------------------------------------        --------------------------------------------------------------------
        locations                                   Optional Bool. Specify whether to synthesize the geometry representing
                                                    the derived location of the object. This option only affects the
                                                    results when objects are features or nonspatial objects.
        ------------------------------------        --------------------------------------------------------------------
        out_sr                                      Optional Dictionary or Integer. The output spatial reference.
        ====================================        ====================================================================

        :return:
            A dictionary with keys and value types of

                | {"exceededTransferLimit": bool,
                | "objects": list,
                | "associations": list,
                | "success": bool}
        """

        if self._gis.version >= [9, 2]:
            url = "%s/locations/query" % self._url
            params = {
                "f": "json",
                "gdbVersion": self._version_name,
                "sessionId": self._version_guid,
                "objects": elements,
                "maxGeometryCount": max_geom_count,
                "moment": moment,
                "attachmentAssociations": attachment_associations,
                "connectivityAssociations": connectivity_associations,
                "containmentAssociations": containment_associations,
                "locations": locations,
                "outSR": out_sr,
            }
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def trace_configurations(self) -> TraceConfigurationsManager:
        """
        The `trace_configurations` resource provides access to all trace
        configuration operations for a utility network.
        It is returned as an array of named trace configurations with the creator,
        name, and global ID for each.

        :return: An instance of TraceConfigurationsManager Class
        """

        if self._gis.version >= [9, 2]:
            url = "%s/traceConfigurations" % self._url
            return TraceConfigurationsManager(
                url,
                version=self._version,
                gis=self._gis,
                service_url=self._url,
            )

    # ----------------------------------------------------------------------


class TraceConfigurationsManager(object):
    """
    The traceConfigurations resource provides access to all trace configuration
    operations for a network service. It is returned as an array of named trace
    configurations with the creator, name, and global ID for each.


    The TraceConfigurationsManager allows methods to be done on a trace configuration.
    """

    _con = None
    _gis = None
    _url = None
    _property = None
    _version_guid = None
    _version_name = None
    _service_url = None

    # ----------------------------------------------------------------------
    def __init__(self, url, version=None, gis=None, service_url=None):
        """Constructor"""
        if gis is None:
            gis = env.active_gis
        self._gis = gis
        self._con = gis._portal.con
        self._url = url
        self._service_url = service_url
        if version:
            self._version = version
            self._version_guid = version._guid
            self._version_name = version.properties.versionName
        else:
            self._version = None
            self._version_guid = None
            self._version_name = None

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return f"< Trace Configuration Manager @ {self._url} >"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"< Trace Configuration Manager @ {self._url} >"

    # ----------------------------------------------------------------------
    def list(self) -> dict:
        """
        List of all trace configurations in a service.

        :return:
            A dictionary with two keys: {"traceConfigurations": list, "success": bool}
        """
        return self._con.post(self._url, {"f": "json"})

    # ----------------------------------------------------------------------
    def query(
        self,
        global_ids: list[str] | None = None,
        creators: list[str] | None = None,
        tags: list[str] | None = None,
        names: list[str] | None = None,
        as_trace_configuration_class: bool = False,
    ) -> dict:
        """
        The query operation returns all properties from one or more
        named trace configurations in a utility network.

        ============================        ===========================================
        **Parameter**                        **Description**
        ----------------------------        -------------------------------------------
        global_ids                          Optional list of strings. Specify the global
                                            IDs of the named trace configs to be queried.
        ----------------------------        -------------------------------------------
        creators                            Optional list of strings. The creators of
                                            the named trace configurations to be queried.
        ----------------------------        -------------------------------------------
        tags                                Optional list of strings. The user tags of
                                            the named trace configurations to be queried.
        ----------------------------        -------------------------------------------
        names                               Optional list of strings. The names of the
                                            named trace configurations to be queried.
        ----------------------------        -------------------------------------------
        as_trace_configuration_class        Optional boolean. If True the list for
                                            "traceCongifurations" in return will be a list
                                            of TraceConfiguration class instances. If False,
                                            the list will be a list of dictionaries of trace
                                            configuration instances. The default is False.
        ============================        ===========================================

        :return:
            A dictionary with two keys: {"traceConfigurations": list, "success": bool}
        """
        if self._gis.version >= [9, 2]:
            url = "%s/query" % self._url
            params = {
                "f": "json",
                "globalIds": global_ids,
                "creators": creators,
                "tags": tags,
                "names": names,
            }
            res = self._con.post(url, params)
            if as_trace_configuration_class:
                trace_configs = []
                for trace in res["traceConfigurations"]:
                    trace = TraceConfiguration.from_config(trace)
                    trace_configs.append(trace)
                res["traceConfigurations"] = trace_configs
            return res

    # ----------------------------------------------------------------------
    def delete(self, global_ids: list[str]) -> dict:
        """
        The delete operation provides the ability to delete one or more named
        trace configurations in a utility network. A named trace configuration
        can only be deleted by an administrator or its creator.

        :return: A dictionary with key "success" indicating True or False.
        """
        if self._gis.version >= [9, 2]:
            url = "%s/delete" % self._url
            params = {
                "f": "json",
                "globalIds": global_ids,
            }
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def create(
        self,
        name: str,
        trace_type: str,
        trace_config: dict | TraceConfiguration,
        description: str | None = None,
        result_types: list[dict] | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """
        The create operation on the traceConfigurations resource provides the
        ability to create a single named trace configuration. Named trace
        configurations store the properties of a complex trace in a utility
        network and can be shared through a map service consumed by a web map
        or field app. Multiple parameters and properties are provided with
        the create operation that support the analytic workflows associated
        with the trace operation.

        If your trace configuration already exists, use the query method to find it.

        ======================      ===============================================
        **Parameter**                **Description**
        ----------------------      -----------------------------------------------
        name                        Required String. The altered name of the trace
                                    configuration.
        ----------------------      -----------------------------------------------
        trace_type                  Required String. Specify the core algorithm that
                                    will be used to analyze the network. Trace types
                                    can be configured using the `trace_config` parameter.

                                    Values:

                                        "connected" | "subnetwork" | "upstream" | "subnetworkController" | "downstream" | "loops" | "shortenPath" | "isolation"
        ----------------------      -----------------------------------------------
        trace_config                Required Dictionary or TraceConfiguration object.
                                    Specify the collection of
                                    altered trace configuration properties.

                                    See: `Properties <https://developers.arcgis.com/rest/services-reference/enterprise/trace-utility-network-server-.htm#GUID-F0C932FD-B403-4223-9B00-E44D156C7DF9/>`_
        ----------------------      -----------------------------------------------
        description                 Optional String. Specify the altered description
                                    of the trace configuration.
        ----------------------      -----------------------------------------------
        result_types                Optional List of Dictionary. Specify the altered
                                    types of results to return.

                                    .. code-block:: python

                                        [{
                                            "type" : "elements" | "aggregatedGeometry",
                                            "includeGeometry" : true | false,
                                            "includePropagatedValues": true | false,
                                            "networkAttributeNames" :["attribute1Name","attribute2Name",...],
                                            "diagramTemplateName": <value>,
                                            "resultTypeFields":[{"networkSourceId":<int>,"fieldname":<value>},...]
                                        },...]

        ----------------------      -----------------------------------------------
        tags                        Optional List of String(s). Specify the altered
                                    user-provided tags.
        ======================      ===============================================

        :return: A dictionary with key "success" indicating True or False.
        """
        if isinstance(trace_config, TraceConfiguration):
            trace_config = trace_config.to_dict()
        if self._gis.version >= [9, 2]:
            url = "%s/create" % self._url
            params = {
                "f": "json",
                "gdbVersion": self._version_name,
                "name": name,
                "description": description,
                "traceType": trace_type,
                "traceConfiguration": trace_config,
                "resultTypes": result_types,
                "tags": tags,
            }
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def alter(
        self,
        global_id: str,
        name: str | None = None,
        description: str | None = None,
        trace_type: str = "connected",
        trace_config: dict | TraceConfiguration | None = None,
        result_types: list[dict] | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """
        The alter operation provides the ability to alter a single named
        trace configuration. A named trace configuration can only be altered
        by an administrator or the creator of the configuration.
        For example, you can update an existing trace configuration to
        accommodate changes in the network or address incorrectly set parameters
        without the need to delete and re-create a trace configuration.
        This enables existing map services to continue use of the named trace
        configuration without requiring the map to be republished.

        ======================      ===============================================
        **Parameter**                **Description**
        ----------------------      -----------------------------------------------
        global_id                   Required String. Specifying the global ID of
                                    the named trace configuration to alter.
        ----------------------      -----------------------------------------------
        name                        Optional String. The altered name of the trace
                                    configuration.
        ----------------------      -----------------------------------------------
        description                 Optional String. Specify the altered description
                                    of the trace configuration.
        ----------------------      -----------------------------------------------
        trace_type                  Optional String. Specify the core algorithm that
                                    will be used to analyze the network. Trace types
                                    can be configured using the `trace_config` parameter.

                                    Values:

                                            "connected" | "subnetwork" | "upstream" | "subnetworkController" | "downstream" | "loops" | "shortenPath" | "isolation"
        ----------------------      -----------------------------------------------
        trace_config                Optional Dictionary or instance of TraceConfiguration
                                    class. Specify the collection of
                                    altered trace configuration properties.

                                    See: `Properties <https://developers.arcgis.com/rest/services-reference/enterprise/trace-utility-network-server-.htm#GUID-F0C932FD-B403-4223-9B00-E44D156C7DF9/>`_
        ----------------------      -----------------------------------------------
        result_types                Optional List of Dictionary. Specify the altered
                                    types of results to return.

                                    .. code-block:: python

                                        [{
                                            "type" : "elements" | "aggregatedGeometry",
                                            "includeGeometry" : true | false,
                                            "includePropagatedValues": true | false,
                                            "networkAttributeNames" :["attribute1Name","attribute2Name",...],
                                            "diagramTemplateName": <value>,
                                            "resultTypeFields":[{"networkSourceId":<int>,"fieldname":<value>},...]
                                        },...]

        ----------------------      -----------------------------------------------
        tags                        Optional List of String(s). Specify the altered
                                    user-provided tags.
        ======================      ===============================================

        :return: A dictionary with key "success" indicating True or False.
        """

        if self._gis.version >= [9, 2]:
            url = "%s/alter" % self._url
            if isinstance(trace_config, TraceConfiguration):
                trace_config = trace_config.to_dict()
            params = {
                "f": "json",
                "globalId": global_id,
                "name": name,
                "description": description,
                "traceType": trace_type,
                "traceConfiguration": trace_config,
                "resultTypes": result_types,
                "tags": tags,
            }
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
