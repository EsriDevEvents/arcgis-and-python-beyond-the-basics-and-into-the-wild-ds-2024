from __future__ import annotations
from typing import Any, Optional, Union
from arcgis import env
from arcgis._impl.common._mixins import PropertyMap


########################################################################
class NetworkDiagramManager(object):
    """
    The Network Diagram service resource represents a network diagram
    service published with ArcGIS Server. The resource provides information
    about the service itself (name, type, default diagram template) and
    exposes various functions to access published network diagrams, create
    new network diagrams and store them, edit and maintain network diagrams, and so on.

    The Network Diagram service supports some operations which allow
    retrieving network diagrams, getting the characteristics of
    the diagrams you want (diagram info, consistency state), creating new network
    diagrams and deleting network diagrams.

    .. note::
        The active portal account must be licensed with the ArcGIS Utility Network
        user type extension or the ArcGIS Trace Network user type extension to
        use the utility network and network diagram services.

    =====================   ===========================================
    **Inputs**              **Description**
    ---------------------   -------------------------------------------
    url                     Required String. The web endpoint to the utility service.
    ---------------------   -------------------------------------------
    version                 Required Version. The `Version` class where the branch version will take place.
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
        self._version = version
        if version:
            self._version_guid = version._guid
            self._version_name = version.properties.versionName
        else:
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
    def find_diagram_names(
        self,
        moment: Optional[int] = None,
        extent: Optional[dict] = None,
        where: Optional[str] = None,
        features: Optional[dict[str, Any]] = None,
        exclude_system_diagrams: bool = False,
    ):
        """
        The find_diagram_names operation is performed on a Network Diagram Service resource.
        The result of this operation is an array of strings, each one corresponding to
        a network diagram's name.

        This operation is used to retrieve the set of diagrams that cover
        a given extent, verify a particular WHERE clause, or contain specific
        utility network features or diagram features.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        moment                              Optional integer. The session moment.
        ------------------------------      -----------------------------------------------------
        extent                              Optional dictionary representing the extent that you
                                            want the network extent of the resulting diagrams to
                                            intersect.

                                            Syntax:
                                            ```
                                            {
                                                "xmin": <xmin>,
                                                "ymin": <ymin>,
                                                "xmax": <xmax>,
                                                "ymax": <ymax>,
                                                "zmin": <zmin>,
                                                "zmax": <zmax>,
                                                "mmin": <mmin>,
                                                "mmax": <mmax>,
                                                "spatialReference": {<spatialReference>}
                                            }
                                            ```
        ------------------------------      -----------------------------------------------------
        where                               Optional string. Any legal SQL WHERE clause operating
                                            on some fields in the diagrams table is allowed. See
                                            table below for the exact list of field names that can
                                            be used.
        ------------------------------      -----------------------------------------------------
        features                            Optional dictionary. A set of utility network feature
                                            Global IDs, or network diagram feature Global IDs
                                            represented in a given diagram that are included in the
                                            resulting queried diagrams.

                                            The dictionary is composed of two keys:
                                            * `globalIDs` - An array of utility network feature
                                            Global IDs (case 1), or an array of network diagram
                                            feature Global IDs (case 2).
                                            * `diagram` - For case 1, NULL, or for case 2, the
                                            diagram name referencing the specified network diagram
                                            feature Global IDs.
        ------------------------------      -----------------------------------------------------
        exclude_system_diagrams             Optional bool. If True, the operation returns any
                                            diagrams except for the subnetwork system diagrams. If
                                            False (default), the operation returns any diagram.
        ==============================      =====================================================

        WHERE CLAUSE FIELDS:

        ================      ===================
        **Field**             **Type**
        ----------------      -------------------
        name                  String
        ----------------      -------------------
        tag                   String
        ----------------      -------------------
        creationDate          Date
        ----------------      -------------------
        creator               String
        ----------------      -------------------
        lastUpdateDate        Date
        ----------------      -------------------
        lastUpdateBy          String
        ================      ===================

        :return: An array of strings, each one corresponding to a diagram name.
        """
        url = "%s/findDiagramNames" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
            "extent": extent,
            "where": where,
            "features": features,
            "excludeSystemDiagrams": exclude_system_diagrams,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def create_diagram_from_features(self, template: str, initial_features: list[str]):
        """
        The create_diagram_from_features operation is performed on a Network Diagram
        Service resource. The result of this operation is a Diagram Information JSON object.

        It is used to create a new temporary network diagram.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        template                            Required string. The name of the diagram template the
                                            new network diagram will be based on.
        ------------------------------      -----------------------------------------------------
        initial_features                    Required list of strings. A list of utility network
                                            features Global IDs from which the new diagram is going
                                            to be built.
        ==============================      =====================================================

        :return: A dictionary of the diagram information.
        """
        url = "%s/createDiagramFromFeatures" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "template": template,
            "initialFeatures": initial_features,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def find_diagram_infos(
        self, diagrams_names: list[str], moment: Optional[int] = None
    ):
        """
        The find_diagram_infos operation is performed on a Network Diagram Service
        resource. The result of this operation is an array of Diagram Information JSON objects.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        diagram_names                       Required list of strings. Each string corresponds to a
                                            diagram name for which you want to get diagram information.
        ------------------------------      -----------------------------------------------------
        moment                              Optional Integer. A session moment.
        ==============================      =====================================================


        :return: Diagram info object for each of the diagram names specified in input.
        """
        url = "%s/findDiagramInfos" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "diagramNames": diagrams_names,
            "moment": moment,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_consistency_state(
        self, diagrams_names: list[str], moment: Optional[int] = None
    ):
        """
        The query_consistency_state operation is performed on a Network Diagram Service resource.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        diagram_names                       Required list of strings. Each string corresponds to a
                                            diagram name for which you want to get diagram information.
        ------------------------------      -----------------------------------------------------
        moment                              Optional Integer. A session moment.
        ==============================      =====================================================

        :return: It returns the consistency state for each of the diagram names specified in input.
        """
        url = "%s/queryConsistencyState" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "diagramNames": diagrams_names,
            "moment": moment,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def delete_diagram(self, name: str):
        """
        The delete_diagram operation is performed on a Network Diagram Service
        resource.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        name                                Required string. The name of the network diagram to delete.
        ==============================      =====================================================

        :return: The moment (date) the delete_diagram operation happens.

        """

        url = "%s/deleteDiagram" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "name": name,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def templates(self):
        """
        The Diagram Templates resource represents all the diagram templates
        under a Network Diagram service. It returns an array of template names.
        """
        url = "%s/templates" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def template(self, name):
        """
        The Template resource represents a single diagram template
        under a Network Diagram service. It is returned as a JSON
        object that provides the list of the diagram layouts and their
        property sets, which are preset for the template.

        The Template resource doesn't support any operation nor any child resource.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        name                                Required string. The name of the template.
        ==============================      =====================================================

        """
        url = "%s/templates/%s" % (self._url, name)
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def diagrams(self, moment: Optional[int] = None):
        """
        The Network Diagrams resource represents all the network diagrams
        under a Network Diagram service. It is returned as an array of network
        diagram names.
        By default, only the diagrams in Default are returned.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        moment                              Optional int. The session moment.
        ==============================      =====================================================

        :return: A list of diagram names.
        """

        url = "%s/diagrams" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def diagram(self, name: str):
        """
        The Diagram resource represents a single network diagram under
        a Network Diagram service.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        name                                Required string. The name of the network diagram.
        ==============================      =====================================================

        """
        url = "%s/diagrams/%s" % (self._url, name)
        return Diagram(url, self._version, self._gis)

    # ----------------------------------------------------------------------
    @property
    def diagram_dataset(self):
        """
        The Diagram Dataset resource regroups the info related to each diagram
        template under a Network Diagram service. It returns an array of diagram template info.
        """
        url = "%s/diagramDataset" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
        }
        return self._con.post(url, params)


# --------------------------------------------------------------------------
class Diagram(object):
    """
    The Diagram resource represents a single network diagram under a Network
    Diagram service. It is returned as a JSON Diagram Information object.

    It supports three child resources:
    * Diagram Map—Mimics a map service resource for the network diagram.
    * Dynamic Layers—Describes the sublayers under the diagram layer.
    * Layer Definitions—Details the layer and labeling properties that define each sublayer under the diagram layer.
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
        self._version = version
        if version:
            self._version_guid = version._guid
            self._version_name = version.properties.versionName
        else:
            self._version_guid = None
            self._version_name = None

    # ----------------------------------------------------------------------
    def _init(self):
        """initializer"""
        try:
            res = self._con.get(
                self._url,
                {
                    "f": "json",
                    "gdbVersion": self._version_name,
                    "sessionId": self._version_guid,
                },
            )
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
    def append_features(self, added_features: list[str]):
        """
        The append_features operation is performed on a Diagram resource.
        The result of this operation is a Diagram JSON Information object,
        and the moment (date) the appendFeatures operation
        happens for a stored diagram.

        It is used to append a set of utility network feature to the diagram resource.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        added_features                      Required list of strings. The strings are utility network
                                            feature Global IDs, the features being appended to the
                                            diagram resource.
        ==============================      =====================================================
        """

        url = "%s/appendFeatures" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "addedFeatures": added_features,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def apply_layout(
        self,
        layout_name: str,
        layer_params: Optional[dict[str, Any]] = None,
        junction_ids: list = None,
        container_ids: list = None,
        edge_ids: list = None,
        run_async: bool = False,
    ):
        """
        The apply_layout operation is performed on a Diagram resource.
        The result of this operation is a Diagram JSON Information object,
        and the moment (date) the applyLayout operation happens for a stored diagram.

        It is used to apply a specific diagram algorithm on all or parts of
        the resource diagram content.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        layout_name                         Required string. The name of the algorithm layout to
                                            execute.

                                            `Values: "AngleDirectedDiagramLayout" | "CompressionDiagramLayout" |
                                            "ForceDirectedDiagramLayout" | "GeoPositionsDiagramLayout" |
                                            "GridDiagramLayout" | "LinearDispatchDiagramLayout" |
                                            "MainLineTreeDiagramLayout" | "MainRingDiagramLayout" |
                                            "PartialOverlappingEdgesDiagramLayout" | "RadialTreeDiagramLayout" |
                                            "RelativeMainlineDiagramLayout" | "ReshapeEdgesDiagramLayout" |
                                            "RotateTreeDiagramLayout" | "SeparateOverlappingEdgesDiagramLayout" |
                                            "SmartTreeDiagramLayout" | "SpatialDispatchDiagramLayout"`
        ------------------------------      -----------------------------------------------------
        layout_params                       Optional dictionary. The algorithm layer parameters.

                                            `Specific Property Parameters <https://developers.arcgis.com/rest/services-reference/enterprise/appendix-diagram-layout-property-set-objects.htm\>`_

                                            Example:
                                               layoutParams = {
                                                   "type": "PropertySet",
                                                    "propertySetItems": [
                                                        "are_containers_preserved",
                                                        false,
                                                        "is_active",
                                                        false,
                                                        "iterations_number",
                                                        20,
                                                        "repel_factor",
                                                        1,
                                                        "degree_freedom",
                                                        1
                                                    ]
                                               }
        ------------------------------      -----------------------------------------------------
        junction_ids                        Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of junction object Ids
                                            that will be processed.
        ------------------------------      -----------------------------------------------------
        container_ids                       Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of container object Ids
                                            that will be processed.
        ------------------------------      -----------------------------------------------------
        edge_ids                            Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of edge object Ids
                                            that will be processed.
        ------------------------------      -----------------------------------------------------
        run_async                           Optional bool. Specify whether the layout algorithm will
                                            run synchronously or asynchronously.

                                            If False, the layout algorithm will run synchronously
                                            and can fail if its execution exceeds the service timeout—600
                                            seconds by default. This is the default.

                                            If True, the layout algorithm will run asynchronously.
                                            This option dedicates server resources to run the layout
                                            algorithm with a longer time-out. Running asynchronously can
                                            be interesting when executing layout that are time
                                            consuming—for example, Partial Overlapping Edges—and
                                            applying to large diagrams—more than 25,000 features.
        ==============================      =====================================================

        """
        url = "%s/applyLayout" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "layoutName": layout_name,
            "layoutParams": layer_params,
            "junctionObjectIDs": junction_ids,
            "containerObjectIDs": container_ids,
            "edgeObjectIDs": edge_ids,
        }
        if self._gis.version >= [9, 2]:
            params["async"] = run_async
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def apply_template_layouts(
        self,
        junction_ids: Optional[list] = None,
        container_ids: Optional[list] = None,
        edge_ids: Optional[list] = None,
    ):
        """
        The apply_template_layouts operation is performed on a Diagram resource.
        The result of this operation is a Diagram JSON Information object, and
        the moment the applyTemplateLayouts operation happens for a stored diagram.

        It is used to re-execute the list of the layout algorithms currently configured
        on the template the resource diagram is based on.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        junction_ids                        Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of junction object Ids
                                            that will be processed.
        ------------------------------      -----------------------------------------------------
        container_ids                       Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of container object Ids
                                            that will be processed.
        ------------------------------      -----------------------------------------------------
        edge_ids                            Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of edge object Ids
                                            that will be processed.
        ==============================      =====================================================
        """
        url = "%s/applyTemplateLayouts" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "junctionObjectIDs": junction_ids,
            "containerObjectIDs": container_ids,
            "edgeObjectIDs": edge_ids,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def clear_flags(self, flag_type: str):
        """
        The clear_flags operation is performed on a Diagram resource.
        This operation returns the moment (date) the clear_flags operation happens
        when it applies on a stored diagram.

        It is used to clear all root junction, end junction, pivot
        junction and barrier flags on the resource diagram.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        flag_type                           Required string. The type of flag you want to clear in
                                            the diagram.

                                            `Values: "esriDiagramRootJunction" | "esriDiagramEndJunction" |
                                            "esriDiagramPivotJunction" | "esriDiagramBarrierEdge" |
                                            "esriDiagramBarrierJunction"`
        ==============================      =====================================================
        """
        url = "%s/clearFlags" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "flagType": flag_type,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def extend(self, extend_type: str = None, from_features: list[str] = None):
        """
        The extend operation is performed on a Diagram resource.
        The result of this operation is a Diagram JSON Information object,
        and the moment the edit operation happens for a stored diagram.

        It is used to extend the diagram resource content one connectivity level,
        optionally regarding to the traversability over the network.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        extend_type                         Optional string. The type of extend you want to process.

                                            `Values: "esriDiagramExtendByAttachment" |
                                            "esriDiagramExtendByConnectivity" |
                                            "esriDiagramExtendByTraversability" |
                                            "esriDiagramExtendByContainment"`
        ------------------------------      -----------------------------------------------------
        from_features                       Optional list of strings. Diagram feature Global IDs,
                                            those diagram features being those from which the
                                            extend process will execute.
        ==============================      =====================================================
        """
        url = "%s/extend" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "extendType": extend_type,
            "fromFeatures": from_features,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def find_diagram_features(
        self,
        from_features: list[str],
        include_aggregations: bool,
        add_connectivity_associations: bool,
        add_structural_attachments: bool,
        from_diagram: Optional[str] = None,
        moment: Optional[int] = None,
    ):
        """
        The find_diagram_features operation is performed on a Diagram resource.
        It returns an array of diagram feature JSON objects for the input fromDiagram.

        This is the operation to use to search for the utility network features
        associated with a set of diagram features that are referenced in the diagram resource.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        from_features                       Required list of strings. Depending on whether you
                                            want to retrieve diagram features associated with
                                            utility network features (case#1) or with diagram
                                            features that are represented in another diagram (case#2),
                                            an array of utility network feature Global IDs (case#1),
                                            or an array of network diagram feature Global IDs (case#2).
        ------------------------------      -----------------------------------------------------
        include_aggregations                Required boolean.
                                            Case#1—When the fromFeatures reference utility
                                            network feature Global IDs:
                                            * `True`—The operation returns the diagram features that
                                            strictly represent those utility network features in the
                                            diagram resource, and the diagram features that are associated
                                            with those utility network features but not represented in
                                            the diagram resource where they are reduced or collapsed.
                                            * `False`—The operation only returns the diagram features
                                            associated with those utility network features that are
                                            not reduced nor collapsed in the diagram resource; that is,
                                            it only returns the diagram features associated with those
                                            utility network features that are visibly represented
                                            in the diagram resource.


                                            Case#2—When the fromFeatures reference diagram feature
                                            Global IDs represented in another diagram:
                                            * `True`—The operation returns the diagram features
                                            associated with the same utility network features
                                            those diagram features are, whether those features
                                            are reduced or collapsed in this other diagram and/or
                                            in the resource diagram.
                                            * `False`—The operation only returns the diagram features
                                            associated with the same utility network features that
                                            are visibly represented in this other diagram
                                            and in the resource diagram.
        ------------------------------      -----------------------------------------------------
        add_connectivity_associations       Required boolean. When the from_features reference utility
                                            network feature Global IDs:

                                            * `True`—The operation also adds any connectivity association
                                            diagram edges for which it has just retrieved both the "from"
                                            and "to" diagram junctions.
                                            * `False`—The operation doesn't add any connectivity
                                            association diagram edges represented in the diagram resource.
        ------------------------------      -----------------------------------------------------
        add_structural_attachments          Required boolean. When the from_features reference
                                            utility network feature Global IDs:

                                            * `True`—The operation also adds any structural attachment
                                            diagram edges for which it has just retrieved both the
                                            "from" and "to" diagram junctions.
                                            * `False`—The operation doesn't add any structural
                                            attachment diagram edges represented in the
                                            diagram resource.
        ------------------------------      -----------------------------------------------------
        from_diagram                        Optional string. The name of the diagram the feature
                                            Global IDs specified for the from_features parameters
                                            belong to, or null when the from_features parameter
                                            references utility network features.
        ------------------------------      -----------------------------------------------------
        moment                              Optional integer. The session moment.
        ==============================      =====================================================
        """
        url = "%s/findDiagramFeatures" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
            "fromFeatures": from_features,
            "fromDiagram": from_diagram,
            "includeAggregations": include_aggregations,
            "addConnectivityAssociations": add_connectivity_associations,
            "addStructuralAttachments": add_structural_attachments,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def find_initial_network_objects(self, moment: Optional[int] = None):
        """
        The find_initial_network_objects operation is performed on a Diagram resource.
        It returns an array of network feature globalIDs.

        This is the operation to use to search for the set of network
        features used as input for the diagram build.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        moment                              Optional integer. A session moment.
        ==============================      =====================================================
        """
        url = "%s/findInitialNetworkObjects" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def find_network_features(
        self,
        from_features: list[str],
        include_aggregations: bool,
        moment: Optional[int] = None,
    ):
        """
        The find_network_features operation is performed on a Diagram resource.
        It returns an array of network feature global IDs.

        This is the operation to use to search for the diagram features referenced
        in the diagram resource that are associated with a set of utility
        network features or a set of diagram features
        represented in another network diagram.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        from_features                       Required list of strings. A list of diagram network
                                            feature global IDs.
        ------------------------------      -----------------------------------------------------
        include_aggregations                Required boolean.
                                            * `True`—The operation returns all the network features
                                            associated with the diagram features specified in the
                                            from_features parameter, whether those features
                                            are reduced or collapsed in the diagram resource.
                                            * `False`—The operation only returns the network features
                                            associated with diagram features specified in the
                                            from_features parameter that are not reduced or
                                            collapsed in the diagram resource; that is, it only
                                            returns the network features associated with the specified
                                            diagram features that are visibly represented in the diagrams.
        ------------------------------      -----------------------------------------------------
        moment                              Optional integer. A session moment.
        ==============================      =====================================================

        """
        url = "%s/findNetworkFeatures" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
            "fromFeatures": from_features,
            "includeAggregations": include_aggregations,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def get_aggregations(self, moment: Optional[int] = None):
        """
        The get_aggregations operation is performed on a Diagram resource.
        The result of this operation is an array of Diagram Aggregation JSON objects.

        It returns all the diagram aggregations in the diagram resource.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        moment                              Optional integer. A session moment.
        ==============================      =====================================================
        """
        url = "%s/getAggregations" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def get_flags(
        self,
        flag_type: str,
        moment: Optional[int] = None,
        out_sr: Optional[Union[str, dict]] = None,
    ):
        """
        The get_flags operation is performed on a Diagram resource.
        The result of this operation is a JSON Information object that
        returns the list of diagram element IDs bringing a flag, with its
        flag type and its location.

        It is used to get the root junction, end junction, pivot junction,
        and barrier flag on a particular diagram feature.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        flag_type                           Required string. The type of flag you want to search
                                            for on the diagram resource.

                                            `Values: "esriDiagramRootJunction" | "esriDiagramEndJunction" |
                                            "esriDiagramPivotJunction" | "esriDiagramBarrierEdge" |
                                            "esriDiagramBarrierJunction"`
        ------------------------------      -----------------------------------------------------
        moment                              Optional integer. A session moment.
        ------------------------------      -----------------------------------------------------
        out_sr                              Optional dictionary or string to specify the spatial
                                            reference of the returned geometry.
        ==============================      =====================================================
        """
        url = "%s/getFlags" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
            "flagType": flag_type,
            "outSR": out_sr,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def manage_flag(self, flag_type: str, flag_id: str, action: str):
        """
        The manage_flag operation is performed on a Diagram resource.
        This operation returns the moment the manageFlag operation happens
        when it applies on a stored diagram.

        It is used to add or remove root junction, end junction, pivot
        junction and barrier flags on a particular diagram feature.


        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        flag_type                           Required string. The type of flag you want to search
                                            for on the diagram resource.

                                            `Values: "esriDiagramRootJunction" | "esriDiagramEndJunction" |
                                            "esriDiagramPivotJunction" | "esriDiagramBarrierEdge" |
                                            "esriDiagramBarrierJunction"`
        ------------------------------      -----------------------------------------------------
        flag_id                             Required string. The diagram element id of the diagram
                                            feature to which you want the flag to be added or from
                                            which you want the flag to be removed.
        ------------------------------      -----------------------------------------------------
        action                              Required string. The flag operation to execute:
                                            * "add" - to add a new flag on a particular diagram feature.
                                            * "remove" - to remove a flag from a particular
                                            diagram feature.
        ==============================      =====================================================
        """
        url = "%s/manageFlag" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "flagID": flag_id,
            "flagType": flag_type,
            "action": action,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def overwrite_from_features(self, initial_features: list[str]):
        """
        The overwrite_from_features operation is performed on a Diagram resource.
        The result of this operation is a Diagram JSON Information object,
        and the moment (date) the overwriteFromFeatures operation happens for a stored diagram.

        It is used to overwrite the diagram resource content from a set of
        utility network feature Global IDs.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        initial_features                    Required list of strings. A list of utility network
                                            feature Global IDs, those features being the ones used
                                            to overwrite the diagram resource.
        ==============================      =====================================================
        """
        url = "%s/overwriteFromFeatures" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "initialFeatures": initial_features,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_diagram_content(
        self,
        moment: Optional[int] = None,
        add_diagram_info: bool = False,
        add_geometries: bool = False,
        add_attributes: bool = False,
        add_aggregations: bool = False,
        use_value_names: bool = False,
        out_sr: Optional[Union[str, dict]] = None,
    ):
        """
        The query_diagram_content operation is performed on a diagram resource.
        It returns the diagram content in a simple format that reflects basic
        connectivity. Other optional information can also be returned,
        such as diagram feature geometry, network element attributes with their
        string descriptions rather than raw values for coded domain values, aggregated
        elements, and diagram properties.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        moment                              Optional int. The session moment.
        ------------------------------      -----------------------------------------------------
        add_diagram_info                    Optional bool.
                                            * `True`—The operation also returns some extra
                                            information on the diagram, such as its template,
                                            the diagram statistics, its creation and last update dates,
                                            and the diagram extent.
                                            * `False`—The operation doesn't return extra
                                            information on the diagram (default).
        ------------------------------      -----------------------------------------------------
        add_geometries                      Optional bool.
                                            * `True`—The operation returns each diagram feature with its geometry.
                                            * `False`—The operation doesn't return the diagram feature geometries (default).
        ------------------------------      -----------------------------------------------------
        add_attributes                      Optional bool.
                                            * `True`—The operation also returns the attributes
                                            of the utility network feature associated with
                                            each diagram feature.
                                            * `False`—The operation doesn't
                                            return any attributes (default).
        ------------------------------      -----------------------------------------------------
        add_aggregations                    Optional bool.
                                            * `True`—The operation returns each diagram feature
                                            with the list of the utility network elements it
                                            aggregates with their asset group and asset type values.
                                            * `False`—The operation doesn't return any aggregations (default)
        ------------------------------      -----------------------------------------------------
        use_value_names                     Optional bool.
                                            For the cases in which the associated network feature
                                            attributes or aggregations are exported, that is,
                                            `add_attributes = True` or `add_aggregations = True`,
                                            the method to use to export coded domain and subtype values:

                                            * `True`—Coded domain and subtype values will be
                                            exported using their string descriptions rather than raw values.
                                            * `False`—Coded domain and subtype values will be
                                            exported as raw values. This is the default.
        ------------------------------      -----------------------------------------------------
        out_sr                              Optional dictionary or string. The spatial reference of the
                                            returned geometry.
        ==============================      =====================================================

        """
        url = "%s/queryDiagramContent" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "moment": moment,
            "addDiagramInfo": add_diagram_info,
            "addGeometries": add_geometries,
            "addAttributes": add_attributes,
            "addAggregations": add_aggregations,
            "useValueNames": use_value_names,
            "outSR": out_sr,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_elements_by_extent(
        self,
        extent: Optional[dict] = None,
        moment: Optional[int] = None,
        add_contents: bool = False,
        return_junctions: bool = True,
        return_edges: bool = True,
        return_containers: bool = True,
        return_geometry: bool = True,
        out_sr: Optional[Union[str, dict]] = None,
    ):
        """
        The query_elements_by_extent operation is performed on a diagram resource.
        The result of this operation is a JSON object composed of three arrays—one
        for the resulting diagram junctions, another for the resulting diagram edges,
        and the third one for the diagram containers.

        It returns the set of diagram features (diagram junctions, diagram edges,
        or diagram containers) represented in the diagram resource that intersects
        a specified envelope—this one being optionally enlarged to include any
        containers that may be partially within it.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        extent                              Optional dictionary. Represent the extent you want the
                                            resulting diagram features to intersect.

                                            Without specifying this parameter, the operation returns
                                            all the diagram features in the resource diagram.

                                            .. code-block:: python

                                                {
                                                    "xmin": <xmin>,
                                                    "ymin": <ymin>,
                                                    "xmax": <xmax>,
                                                    "ymax": <ymax>,
                                                    "zmin": <zmin>,
                                                    "zmax": <zmax>,
                                                    "mmin": <mmin>,
                                                    "mmax": <mmax>,
                                                    "spatialReference": <spatialReference>
                                                }
        ------------------------------      -----------------------------------------------------
        moment                              Optional int. The session moment.
        ------------------------------      -----------------------------------------------------
        add_contents                        Optional bool.
                                            * `False` —To return the diagram features which strictly
                                            intersect the specified envelope (default)
                                            * `True`—To enlarge the searching envelope so it
                                            includes the extent of any containers that are partially
                                            within the specified envelope
        ------------------------------      -----------------------------------------------------
        return_junctions                    Optional bool.
                                            * `True`—To return the diagram junctions (default)
                                            * `False`—To not return the diagram junctions
        ------------------------------      -----------------------------------------------------
        return_edges                        Optional bool.
                                            * `True`—To return the diagram edges (default)
                                            * `False`—To not return the edges junctions
        ------------------------------      -----------------------------------------------------
        return_containers                   Optional bool.
                                            * `True`—To return the diagram containers (default)
                                            * `False`—To not return the diagram containers
        ------------------------------      -----------------------------------------------------
        return_geometry                     Optional bool.
                                            * `True`—To return each queried diagram feature with its
                                            geometry (default)
                                            * `False`—To not return the geometry.
        ------------------------------      -----------------------------------------------------
        out_sr                              Optional dictionary or string representing the spatial
                                            reference of the returned geometry.
        ==============================      =====================================================
        """
        url = "%s/queryDiagramElementsByExtent" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "extent": extent,
            "moment": moment,
            "addContents": add_contents,
            "returnJunctions": return_junctions,
            "returnEdges": return_edges,
            "returnContainers": return_containers,
            "returnGeometry": return_geometry,
            "outSR": out_sr,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_elements_by_ids(
        self,
        junction_ids: Optional[list] = None,
        container_ids: Optional[list] = None,
        edge_ids: Optional[list] = None,
        moment: Optional[int] = None,
        add_connected: bool = False,
        return_geometry: bool = True,
        out_sr: Optional[Union[dict, str]] = None,
    ):
        """
        The query_elements_by_ids operation is performed on a diagram resource.
        The result of this operation is a JSON object composed of three
        arrays—one for the resulting diagram junctions, another for the
        resulting diagram edges, and the third one for the diagram containers.

        It returns the set of diagram features—that is, diagram junctions,
        diagram edges, or diagram containers—represented in the diagram
        resource with the specified ObjectIDs. Then when specifying diagram edge
        ObjectIDs, optionally, extend the resulting set of diagram elements to the
        junctions they are connected to or when specifying diagram container ObjectIDs,
        optionally, extend the resulting set of diagram elements to the edge, junction,
        or container diagram elements they contain.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        junction_ids                        Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of junction object Ids
                                            that will be processed.
        ------------------------------      -----------------------------------------------------
        container_ids                       Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of container object Ids
                                            that will be processed.
        ------------------------------      -----------------------------------------------------
        edge_ids                            Optional list. For the case you want the layout algorithm
                                            to execute on a diagram part. A list of edge object Ids
                                            that will be processed.
        ------------------------------      -----------------------------------------------------
        moment                              Optional integer. The session moment.
        ------------------------------      -----------------------------------------------------
        add_connected                       Optional bool.
                                            * `False`-Doesn't enlarge the returned diagram junction to
                                            those that are from or to junctions of the specified
                                            edge_ids. (default)
                                            * `True`-Enlarges the returned diagram junctions to those
                                            that are from or to junctions of the specifies edge_ids.
        ------------------------------      -----------------------------------------------------
        return_geometry                     Optional bool.
                                            * `False`-Return each queried feature without its geometry.
                                            * `True`-Return each queried feature with its geometry. (default)
        ------------------------------      -----------------------------------------------------
        out_sr                              Optional dictionary or string representing the spatial
                                            reference for the returned geometry.
        ==============================      =====================================================
        """
        url = "%s/queryDiagramElementsByObjectIDs" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "junctionObjectIDs": junction_ids,
            "containerObjectIDs": container_ids,
            "edgeObjectIDs": edge_ids,
            "moment": moment,
            "addConnected": add_connected,
            "returnGeometry": return_geometry,
            "outSR": out_sr,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def save_layout(
        self,
        junctions: Optional[list[dict]] = None,
        containers: Optional[list[dict]] = None,
        edges: Optional[list[dict]] = None,
    ):
        """
        The save_layout operation is performed on a Diagram resource.
        The result of this operation is a Diagram JSON Information object,
        and the moment (date) the saveLayout operation happens for a stored diagram.

        It is used to apply and save new geometries that may have been computed for a
        set of diagram features represented in the diagram resource.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        junctions                           Optional list of dictionaries. Provide the new geometry
                                            for each edited junction, those diagram junctions being
                                            identified thanks to their Diagram Element ID.

                                            Syntax:
                                            junctions=[{"ID": <jctDEID1>, "geometry": <geometry1>}, ...]
        ------------------------------      -----------------------------------------------------
        containers                          Optional list of dictionaries. Provide the new geometry
                                            for each edited container, those diagram containers being
                                            identified thanks to their Diagram Element ID.

                                            Syntax:
                                            containers=[{"ID": <containerDEID1>, "geometry": <geometry1>}, ...]
        ------------------------------      -----------------------------------------------------
        edges                               Optional list of dictionaries. Provide the new geometry
                                            for each edited edge, those diagram edges being
                                            identified thanks to their Diagram Element ID.

                                            Syntax:
                                            containers=[{"ID": <edgeDEID1>, "geometry": <geometry1>}, ...]
        ==============================      =====================================================
        """
        url = "%s/saveLayout" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "junctions": junctions,
            "containers": containers,
            "edges": edges,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def store(self, name: str, access: str):
        """
        The store operation is performed on a Diagram resource.
        The result of this operation is a Diagram JSON Information object,
        and the moment the store operation happens.

        It is used to store the temporary network diagram resource in the database.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        name                                Required string. The name of the network diagram to
                                            be stored.
        ------------------------------      -----------------------------------------------------
        access                              Required string. The access right level you want to
                                            set for the stored diagram.

                                            Values:
                                            * "esriDiagramPublicAccess" - Anyone will have full
                                            access rights on the diagram; that is,
                                            view/edit/update/overwrite/delete permissions on it.
                                            * "esriDiagramProtectedAccess" - Anyone will have view
                                            and read access rights on the diagram, but they will have
                                            no permission for editing/update/overwriting/nor
                                            deleting the diagram.
                                            * "esriDiagramPrivateAccess" - No access to the
                                            diagram—nor to view nor to edit it—for anyone except its owner
        ==============================      =====================================================
        """
        url = "%s/store" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "name": name,
            "access": access,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def update(self):
        """
        The update operation is performed on a Diagram resource.
        The result of this operation is a Diagram JSON Information object,
        and the moment the update operation happens for a stored diagram.

        It is used to update the diagram resource content; that is, synchronize
        its content from the network features used to initially generate it,
        and so reflect any changes that may have impacted those network
        features into the diagram.

        """
        url = "%s/update" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def dynamic_layers(self, all_layers: bool = False):
        """
        The Dynamic Layers resource describes the sublayers under the diagram layer.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        all_layers                          Optional bool. If True, all layers are cached whether
                                            they contain diagram features or not.
                                            If False, only the layers that contain diagram features
                                            are cached.
        ==============================      =====================================================

        :return:
            An array of JSON object layers with their SQL query.
            Each JSON object layer item in the array provides the following information:
            * "id"—The ID of the layer
            * "source"—The layer source internal information:
            * "type"—"workspaceLayer"
            * "workspaceID"—"Diagram"
            * "layerID"—The internal layer ID
            * "definitionExpression"-The filtering expression based on the diagram GUID

        """
        url = "%s/dynamicLayers" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "allLayers": all_layers,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def layer_definitions(self, all_layers: bool = False, moment: Optional[int] = None):
        """
        The Layers Definitions resource details all the layer and labeling
        properties that define each sublayer under the diagram layer.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        all_layers                          Optional bool. If True, all layers are cached whether
                                            they contain diagram features or not.
                                            If False, only the layers that contain diagram features
                                            are cached.
        ------------------------------      -----------------------------------------------------
        moment                              Optional int. The session moment.
        ==============================      =====================================================
        """
        url = "%s/layerDefinitions" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
            "allLayers": all_layers,
            "moment": moment,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def diagram_map(self):
        """
        The Diagram Map resource mimics a map service resource. It is returned as a Map service by the REST API.

        .. note::
            This map REST endpoint is the one you can add as content to your web maps.
        """
        url = "%s/map" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "sessionId": self._version_guid,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def export_diagram_map(
        self,
        size: str,
        dpi: Optional[str] = None,
        bbox: Optional[str] = None,
        format: str = "PNGS",
        transparent: bool = False,
        map_scale: Optional[str] = None,
        rotation: Optional[int] = 0,
        bbox_sr: Optional[str] = None,
        image_sr: Optional[str] = None,
        moment: Optional[int] = None,
    ):
        """
        The Export operation is performed on a Diagram Map resource.
        The result of this operation is a map image that represents the diagram specified in the URL.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        size                                Required string. The size ("width, height") of the
                                            exported image in pixels.

                                            Example: size="600, 600"
        ------------------------------      -----------------------------------------------------
        dpi                                 Optional string. The device resolution of the exported
                                            image (dots per inch). If the dpi is not specified,
                                            an image with a default of 96 will be exported.

                                            Example: dpi="600, 600"
        ------------------------------      -----------------------------------------------------
        bbox                                Optional string. The extent(bounding box) of the exported
                                            image. Unless the bbox_sr parameter has been specified,
                                            the bbox assumed to be in the spatial reference of
                                            the map.

                                            The bbox coordinates should always use a period as a
                                            decimal separator, even in countries where traditionally
                                            a comma is used.

                                            Example: bbox="-104, 35.6, -94.32, 41"
        ------------------------------      -----------------------------------------------------
        format                              Optional string. The format of the exported image.

                                            `Values: "PNG32" | "PNG24" | "PNG" | "JPG" | "DIB" |
                                            "TIFF" | "EMF" | "PS" | "PDF" | "GIF" | "SVG" | "SVGZ" | "BMP"`
        ------------------------------      -----------------------------------------------------
        transparent                         Optional bool.  If true, the image will be exported
                                            with the background color of the map set as its
                                            transparent color. The default is false.
                                            Only the "PNG" and "GIF" formats support transparency.

                                            Internet Explorer 6 does not display transparency
                                            correctly for png24 image formats.
        ------------------------------      -----------------------------------------------------
        map_scale                           Optional int. Use this parameter to export a map image
                                            at a specific scale, with the map centered around the
                                            center of the specified bbox.
        ------------------------------      -----------------------------------------------------
        rotation                            Optional int. Use this parameter to export a map image
                                            rotated at a specific angle, with the map centered
                                            around the center of the specified bounding box (bbox).
                                            It could be a positive or negative number.
        ------------------------------      -----------------------------------------------------
        bbox_sr                             Optional string. The well-known ID that is the spatial
                                            reference of the bbox.

                                            If none is specified, the spatial reference of the
                                            map is used.
        ------------------------------      -----------------------------------------------------
        image_sr                            Optional string. The well-known ID that is the spatial
                                            reference of the exported image.

                                            If none is specified, the spatial reference of the
                                            map is used.
        ------------------------------      -----------------------------------------------------
        moment                              Optional int. The session moment.
        ==============================      =====================================================
        """
        url = "%s/map/export" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "moment": moment,
            "size": size,
            "dpi": dpi,
            "imageSR": image_sr,
            "bboxSR": bbox_sr,
            "format": format,
            "transparent": transparent,
            "mapScale": map_scale,
            "rotation": rotation,
            "bbox": bbox,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def identify_diagram_map(
        self,
        geometry: Union[str, dict],
        tolerance: int,
        image_display: str,
        map_extent: str,
        geometry_precision: Optional[int] = None,
        return_field_name: bool = False,
        return_unformatted_values: bool = False,
        return_z: bool = False,
        return_m: bool = False,
        return_geometry: bool = True,
        max_allowable_offset: Optional[int] = None,
        geometry_type: str = "esriGeometryPoint",
        sr: Optional[str] = None,
        moment: Optional[int] = None,
    ):
        """
        The Identify operation is performed on a Diagram Map resource.
        The result of this operation is an identify results resource.

        ==============================      =====================================================
        **Parameter**                       **Description**
        ------------------------------      -----------------------------------------------------
        geometry                            Required string or dictionary.
                                            The geometry to identify on. The type of the geometry is specified by the geometryType parameter. The structure of the geometries is the same as the structure of the JSON geometry objects returned by the ArcGIS REST API. In addition to the JSON structures, for points and envelopes, you can specify the geometries with a simpler comma-separated syntax.

                                            Syntax:

                                            * JSON structures: geometry=<geometryType>&geometry={ geometry}
                                            * Point simple syntax: geometry=esriGeometryPoint&geometry=<x>,<y>
                                            * Envelope simple syntax: geometry=esriGeometryEnvelope&geometry=<xmin>,<ymin>,<xmax>,<ymax>


                                            Example:

                                                * JSON structures: geometry=esriGeometryPoint&geometry={x: -104, y: 35.6}
                                                * Point simple syntax:geometry=esriGeometryPoint&geometry=-104,35.6
                                                * Envelope simple syntax: geometry=esriGeometryEnvelope&geometry=-104,35.6,-94.32,41
        ------------------------------      -----------------------------------------------------
        tolerance                           Required int. The distance in screen pixels from the
                                            specified geometry within which the identify should be
                                            performed.
        ------------------------------      -----------------------------------------------------
        image_display                       Required string. The screen image display parameters
                                            (width, height, and DPI) of the map being currently viewed.

                                            The mapExtent and the imageDisplay parameters are
                                            used by the server to determine the layers visible
                                            in the current extent. They are also used to calculate
                                            the distance on the map to search based on the tolerance
                                            in screen pixels.

                                            Syntax: image_display="<width>,<heigth>,<dpi>"
        ------------------------------      -----------------------------------------------------
        map_extent                          Required string. The extent or bbox fo the map currently
                                            being viewed. Unless the sr parameter has been specified,
                                            the map_extent is assumed to be in the spatial reference
                                            of the map.

                                            The `map_extent` and `image_display` parameters are
                                            used by the server to determine the layers visible in
                                            the current extent. They are also used to calculate
                                            the distance on the map to seach based on the `tolerance`
                                            in screen pixels.

                                            Syntax: map_extent="<xmin>, <ymin>, <xmax>, <ymax>"
        ------------------------------      -----------------------------------------------------
        geometry_precision                  Optional int. Specify the number of decimal places in
                                            the response geometries returned by the operation.
                                            Does not apply to m and z values.
        ------------------------------      -----------------------------------------------------
        return_field_name                   Optional bool. If True, field names will be returned
                                            instead of field aliases.
        ------------------------------      -----------------------------------------------------
        return_unformatted_values           Optional bool. If True, the values in the result will
                                            not be formatted. Numbers will be returned as is and
                                            dates will be returned as epoch values.
        ------------------------------      -----------------------------------------------------
        return_z                            Optional bool. If True, z value will be included in
                                            the results if the features have z-values.

                                            Only applies if `return_geometry=True`
        ------------------------------      -----------------------------------------------------
        return_m                            Optional bool. If True, m value will be included in
                                            the results if the features have m-values.

                                            Only applies if `return_geometry=True`
        ------------------------------      -----------------------------------------------------
        return_geometry                     Optional bool. If True, the result set will include
                                            the geometries associated with each result.
        ------------------------------      -----------------------------------------------------
        max_allowable_offset                Optional int. This option can be used to specify the
                                            maximum allowable offset to be used for generalizing
                                            geometries returned by the identify operation.
                                            The max_allowable_offset is in the units of the sr.
                                            If sr is not specified, max_allowable_offset is
                                            assumed to be in the unit of the spatial reference of the map.
        ------------------------------      -----------------------------------------------------
        geometry type                       Optional string. The type of geometry specified by the
                                            geometry parameter. The geometry type could be a point, line, polygon, or an envelope.
                                            The default geometry type is a point ("esriGeometryPoint").

                                            `Values: "esriGeometryPoint" | "esriGeometryMultipoint" |
                                            "esriGeometryPolyline" | "esriGeometryPolygon" | "esriGeometryEnvelope"`
        ------------------------------      -----------------------------------------------------
        sr                                  Optional string. The well-known ID fo the spatial
                                            reference of the input and output geometries as well
                                            as the map_extent.

                                            If sr is not specified, the geometry and the map_extent
                                            are assumed to be in the spatial reference of the map,
                                            and the output geometries are also in the spatial reference
                                            of the map.
        ------------------------------      -----------------------------------------------------
        moment                              Optional int. The session moment.
        ==============================      =====================================================
        """
        url = "%s/map/identify" % self._url
        params = {
            "f": "json",
            "gdbVersion": self._version_name,
            "moment": moment,
            "geometry": geometry,
            "tolerance": tolerance,
            "imageDisplay": image_display,
            "mapExtent": map_extent,
            "geometryPrecision": geometry_precision,
            "returnFieldName": return_field_name,
            "returnUnformattedValues": return_unformatted_values,
            "returnZ": return_z,
            "returnM": return_m,
            "returnGeometry": return_geometry,
            "maxAllowableOffset": max_allowable_offset,
            "geometryType": geometry_type,
            "sr": sr,
        }
        return self._con.post(url, params)
