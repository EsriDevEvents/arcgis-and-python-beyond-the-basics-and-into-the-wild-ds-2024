from __future__ import annotations
from typing import Any
from arcgis import env
from arcgis._impl.common._mixins import PropertyMap
from arcgis.features._utility import TraceConfigurationsManager
from arcgis.features._trace_configuration import TraceConfiguration
from arcgis.auth.tools import LazyLoader

geometry = LazyLoader("arcgis.geometry")


########################################################################
class TraceNetworkManager(object):
    """
    The Trace Network Service exposes analytic capabilities (tracing)
    as well as validation of network topology.

    =====================   ===========================================
    **Inputs**              **Description**
    ---------------------   -------------------------------------------
    url                     Required String. The web endpoint to the trace service.
    ---------------------   -------------------------------------------
    version                 Optional Version. The `Version` class where the branch version will take place.
    ---------------------   -------------------------------------------
    gis                     Optional GIS. The `GIS` connection object.
    =====================   ===========================================


    """

    _con = None
    _gis = None
    _url = None
    _version = None
    _property = None
    _version_guid = None
    _version_name = None

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
            self._version_guid = version._guid
            self._version_name = version.properties.versionName
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
        moment: str | None = None,
        configuration: dict | TraceConfiguration | None = None,
        result_types: list[dict] | None = None,
        run_async: bool = False,
    ) -> dict:
        """
        A trace refers to a preconfigured algorithm that systematically
        travels a network to return results. Multiple parameters and properties
        are provided with the trace operation that support various analytic workflows.
        All traces use the network topology to read cached information about network features.
        This can improve performance of complex traces on large networks.
        Trace results are not guaranteed to accurately represent a trace network when
        dirty areas are present. The network topology must be validated to ensure that it
        reflects the most recent edits or updates made to the network.

        .. note::
            The active portal account must be licensed with the ArcGIS Trace
            Network user type extention to use this operation.

        ====================    ==================================================
        **Parameter**           **Description**
        --------------------    --------------------------------------------------
        locations               Required list of dictionaries. The locations for
                                starting points and barriers. An empty array must
                                be used when performing a subnetwork trace if a
                                subnetworkName is provided as part of the
                                `configuration` - for example, `locations=[]`.


                                The location is ignored by the trace if the following
                                required properties are not defined:
                                * `percentAlong` : required for edge features and objects.


                                .. code-block:: python

                                    [{
                                        "traceLocationType" : "startingPoint" | "barrier",
                                        "globalId" : <guid>,
                                        “percentAlong” : <double>, // optional
                                    }]
        --------------------    --------------------------------------------------
        trace_type              Required string. Specifies the core algorithm that
                                will be executed to analyze the network. Can be
                                configured using the `configuration` parameter.

                                `Values: 'connected' | 'subnetwork' | 'subnetworkController' |
                                'upstream' | 'downstream' | 'loops' | 'shortestPath' |
                                'isolation'`
        --------------------    --------------------------------------------------
        moment                  Optional Integer. Specifies the session moment. This
                                should only be specified if you do not want to use
                                the current moment.

                                Example: moment = <Epoch time in milliseconds>
        --------------------    --------------------------------------------------
        configuration           Optional dictionary or instance of TraceConfiguration
                                class. Specifies the collection of
                                trace configuration properties. Depending on the
                                `trace_type`, some properties are required.

                                To see all configuration properties see:
                                `Trace Configuration Properties
                                <https://developers.arcgis.com/rest/services-reference/enterprise/trace-trace-network-server-.htm#GUID-F0C932FD-B403-4223-9B00-E44D156C7DF9/>`_
        --------------------    --------------------------------------------------
        result_types            Optional parameter specifying hte types of results
                                to return.

                                .. code-block::
                                    [{
                                        "type" : "elements" | "aggregatedGeometry" | "connectivity",
                                        "includeGeometry" : true | false,
                                        "includePropagatedValues": true | false,
                                        "networkAttributeNames" :["attribute1Name","attribute2Name",...],
                                        "diagramTemplateName": <value>,
                                        "resultTypeFields":[{"networkSourceId":<long>,"fieldname":<value>},...]
                                    },...]

                                .. note::
                                    ArcGIS Enterprise 10.9.1 or later is required when
                                    using the `connectivity` type.
        ====================    ==================================================

        :return: If `asynchronous = True` then the status URL is returned for the job. Otherwise,
        a Dictionary of the Trace Results is returned.
        """
        url = "%s/trace" % self._url
        if isinstance(configuration, TraceConfiguration):
            configuration = configuration.to_dict()
        params = {
            "f": "json",
            "traceType": trace_type,
            "moment": moment,
            "traceLocations": locations,
            "traceConfiguration": configuration,
            "resultTypes": result_types,
            "async": run_async,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_network_moments(
        self,
        moments_to_return: list[str] = ["all"],
        moment: str | None = None,
    ) -> dict:
        """
        The `query_network_moments` operation returns the moments related
        to the network topology and operations against the topology. This
        includes when the topology was initially enabled, when it was last
        validated, when the topology was last disabled (and later enabled),
        and when the definition of the trace network was last modified.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        moments_to_return                           Optional List of Strings. Represents the collection of validate moments to
                                                    return. Default is all.

                                                    `Values: ["initialEnableTopology" | "fullValidateTopology" |
                                                            "partialValidateTopology" | "enableTopology" | "disableTopology" |
                                                            "definitionModification" | "indexUpdate" | "all" ]`
        ------------------------------------        --------------------------------------------------------------------
        moment                                      Optional Epoch in Time in Seconds.
                                                    Example: `moment=1603109606`
        ====================================        ====================================================================

        """
        url = "%s/queryNetworkMoments" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "momentsToReturn": moments_to_return,
            "moment": moment,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def validate_topology(
        self,
        envelope: dict[str, Any] | geometry.Envelope,
        return_edits: bool = False,
    ) -> dict:
        """
        Validating the network topology for a trace network maintains
        consistency between feature editing space and network topology space.
        Validating a network topology may include all or a subset of the
        dirty areas present in the network. Validation of network topology
        is supported synchronously and asynchronously.

        ====================================        ====================================================================
        **Parameter**                                **Description**
        ------------------------------------        --------------------------------------------------------------------
        envelope                                    Required Dictionary or Envelope. The envelope of the area to validate.

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
        ====================================        ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        url = "%s/validateNetworkTopology" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "validateArea": envelope,
            "returnEdits": return_edits,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def trace_configurations(self) -> TraceConfigurationsManager:
        """
        The `trace_configurations` resource provides access to all trace
        configuration operations for a trace network.
        It is returned as an array of named trace configurations with the creator,
        name, and global ID for each.
        """

        if self._gis.version >= [9, 2]:
            url = "%s/traceConfigurations" % self._url
            return TraceConfigurationsManager(
                url, version=self._version, gis=self._gis, service_url=self._url
            )
