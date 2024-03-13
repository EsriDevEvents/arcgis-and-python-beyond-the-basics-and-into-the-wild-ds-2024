from __future__ import annotations
from arcgis.auth.tools import LazyLoader
from typing import Generator
from arcgis.geometry import Geometry
import copy
import datetime

try:
    import arcgis.graph._arcgisknowledge as _kgparser

    HAS_KG = True
except ImportError as e:
    HAS_KG = False
_isd = LazyLoader("arcgis._impl.common._isd")
from typing import List, Any


class KnowledgeGraph:
    """
    Provides access to the Knowledge Graph service data model and properties, as well as
    methods to search and query the graph.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Knowledge Graph service URL
    ------------------     --------------------------------------------------------------------
    gis                    an authenticated :class:`arcgis.gis.GIS` object.
    ==================     ====================================================================

    .. code-block:: python

        # Connect to a Knowledge Graph service:

        gis = GIS(url="url",username="username",password="password")

        knowledge_graph = KnowledgeGraph(url, gis=gis)

    """

    _gis = None
    _url = None
    _properties = None

    def __init__(self, url: str, *, gis=None):
        """initializer"""
        self._url = url
        self._gis = gis

    def _validate_import(self):
        if HAS_KG == False:
            raise ImportError(
                "An error occured with importing the Knowledge Graph libraries. Please ensure you "
                "are using Python 3.9, 3.10 or 3.11 on Windows or Linux platforms."
            )

    def _getInputQuantParams(self, inputQuantParams: dict):
        clientCoreQuantParams = _kgparser.InputQuantizationParameters()
        clientCoreQuantParams.xy_resolution = inputQuantParams["xyResolution"]
        clientCoreQuantParams.x_false_origin = inputQuantParams["xFalseOrigin"]
        clientCoreQuantParams.y_false_origin = inputQuantParams["yFalseOrigin"]
        clientCoreQuantParams.z_resolution = inputQuantParams["zResolution"]
        clientCoreQuantParams.z_false_origin = inputQuantParams["zFalseOrigin"]
        clientCoreQuantParams.m_resolution = inputQuantParams["mResolution"]
        clientCoreQuantParams.m_false_origin = inputQuantParams["mFalseOrigin"]
        return clientCoreQuantParams

    @classmethod
    def fromitem(cls, item):
        """Returns the Knowledge Graph service from an Item"""
        if item.type != "Knowledge Graph":
            raise ValueError(
                "Invalid item type, please provide a 'Knowledge Graph' item."
            )
        return cls(url=item.url, gis=item._gis)

    @property
    def properties(self) -> _isd.InsensitiveDict:
        """Returns the properties of the Knowledge Graph service"""
        if self._properties is None:
            resp = self._gis._con.get(self._url, {"f": "json"})
            self._properties = _isd.InsensitiveDict(resp)
        return self._properties

    def search(self, search: str, category: str = "both") -> List[dict]:
        """
        Allows for the searching of the properties of entities,
        relationships, or both in the graph using a full-text index.

        `Learn more about searching a knowledge graph <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-graph-search.htm>`_

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        search              Required String. The search to perform on the Knowledge Graph.
        ----------------    ---------------------------------------------------------------
        category            Optional String.  The category is the location of the full
                            text search.  This can be isolated to either the `entities` or
                            the `relationships`.  The default is to look in `both`.

                            The allowed values are: both, entities, relationships,
                            both_entity_relationship, and meta_entity_provenance. Both and
                            both_entity_relationship are functionally the same.
        ================    ===============================================================

        .. note::
            Check the `service definition for the Knowledge Graph service <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-hosted-server.htm>`_
            for valid values of category. Not all services support both and both_entity_relationship.

        .. code-block:: python

            #Perform a search on the knowledge graph
            search_result = knowledge_graph.search("cat")

            # Perform a search on only entities in the knowledge graph
            searchentities_result = knowledge_graph.search("cat", "entities")

        :return: List[list]

        """
        url = self._url + "/graph/search"
        cat_lu = {
            "both": _kgparser.esriNamedTypeCategory.both,
            "both_entity_relationship": _kgparser.esriNamedTypeCategory.both_entity_relationship,
            "relationships": _kgparser.esriNamedTypeCategory.relationship,
            "entities": _kgparser.esriNamedTypeCategory.entity,
            "meta_entity_provenance": _kgparser.esriNamedTypeCategory.meta_entity_provenance,
        }
        assert str(category).lower() in cat_lu.keys()
        r_enc = _kgparser.GraphSearchRequestEncoder()
        r_enc.search_query = search
        r_enc.return_geometry = True
        r_enc.max_num_results = self.properties["maxRecordCount"]
        r_enc.type_category_filter = cat_lu[category.lower()]
        r_enc.encode()
        error = r_enc.get_encoding_result().error
        if error.error_code != 0:
            raise Exception(error.error_message)
        query_dec = _kgparser.GraphQueryDecoder()

        session = self._gis._con._session
        response = session.post(
            url=url,
            params={"f": "pbf"},
            data=r_enc.get_encoding_result().byte_buffer,
            stream=True,
            headers={"Content-Type": "application/octet-stream"},
        )
        rows = []
        query_dec = _kgparser.GraphQueryDecoder()
        query_dec.data_model = self._datamodel
        for chunk in response.iter_content(8192):
            did_push = query_dec.push_buffer(chunk)
            while query_dec.next_row():
                rows.append(query_dec.get_current_row())
        return rows

    def update_search_index(self, adds: dict = None, deletes: dict = None) -> dict:
        """
        Allows users to add or delete search index properties for different entities and
        relationships from the graph's data model. Can only be existent properties for a given
        entity/relationship. Note that an empty dictionary result indicates success.

        =========================   ===============================================================
        **Parameter**                **Description**
        -------------------------   ---------------------------------------------------------------
        adds                        Optional dict. See below for structure. The properties to add
                                    to the search index, specified by entity/relationship.
        -------------------------   ---------------------------------------------------------------
        deletes                     Optional dict. See below for structure. The properties to
                                    delete from the search index, specified by entity/relationship.
        =========================   ===============================================================

        .. code-block:: python

            # example of an adds or deletes dictionary
            {
                "Entity1" : { "property_names": ["prop1", "prop2"]},
                "Entity2" : {"property_names": ["prop1"]},
                "RelationshipType1" : { "property_names": ["prop1", "prop2"]},
                "RelationshipType2" : {"property_names": ["prop1"]},
            }

        :return: A `dict`. Empty dict indicates success, errors will be returned in the dict.

        """

        self._validate_import()
        url = self._url + "/dataModel/searchIndex/update"
        params = {
            "f": "pbf",
            "token": self._gis._con.token,
        }
        headers = {"Content-Type": "application/octet-stream"}

        enc = _kgparser.GraphUpdateSearchIndexRequestEncoder()
        if adds:
            enc.insert_add_search_property(adds)
        if deletes:
            enc.insert_delete_search_property(deletes)

        enc.encode()
        enc_result = enc.get_encoding_result()
        error = enc_result.error
        if error.error_code != 0:
            raise Exception(error.error_message)

        session = self._gis._con._session
        response = session.post(
            url=url,
            params=params,
            data=enc_result.byte_buffer,
            stream=True,
            headers=headers,
        )

        content = response.content
        dec = _kgparser.GraphUpdateSearchIndexResponseDecoder()
        dec.decode(content)

        results = dec.get_results()
        return results

    def query(self, query: str) -> List[dict]:
        """
        Queries the Knowledge Graph using openCypher

        `Learn more about querying a knowledge graph <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-graph-query.htm>`_

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        query               Required String. Allows you to return the entities and
                            relationships in a graph, as well as the properties of those
                            entities and relationships, by providing an openCypher query.
        ================    ===============================================================

        .. code-block:: python

            # Perform an openCypher query on the knowledge graph
            query_result = knowledge_graph.query("MATCH path = (n)-[r]-(n2) RETURN path LIMIT 5")


        :return: List[list]

        """
        self._validate_import()
        url = f"{self._url}/graph/query"
        params = {
            "f": "pbf",
            "token": self._gis._con.token,
            "openCypherQuery": query,
        }

        data = self._gis._con.get(url, params, return_raw_response=True, try_json=False)
        buffer_dm = data.content
        gqd = _kgparser.GraphQueryDecoder()
        gqd.push_buffer(buffer_dm)
        gqd.data_model = self._datamodel
        rows = []
        while gqd.next_row():
            r = gqd.get_current_row()
            rows.append(r)
        return rows

    def query_streaming(
        self,
        query: str,
        input_transform: dict[str, Any] = None,
        bind_param: dict[str, Any] = None,
        include_provenance: bool = False,
    ):
        """
        Query the graph using an openCypher query. Allows for more customization than the base
        `query()` function. Creates a generator of the query results, from which users can
        access each row or add them to a list. See below for example usage.


        ===================    ===============================================================
        **Parameter**           **Description**
        -------------------    ---------------------------------------------------------------
        query                  Required String. Allows you to return the entities and
                               relationships in a graph, as well as the properties of those
                               entities and relationships, by providing an openCypher query.
        -------------------    ---------------------------------------------------------------
        input_transform        Optional dict. Allows a user to specify custom quantization
                               parameters for input geometry, which dictate how geometries are
                               compressed and transferred to the server. Defaults to lossless
                               WGS84 quantization.
        -------------------    ---------------------------------------------------------------
        bind_param             Optional dict. The bind parameters used to filter
                               query results. Key of each pair is the string name for it,
                               which is how the parameter can be referenced in the query. The
                               value can be any "primitive" type value that may be found as
                               an attribute of an entity or relationship (e.g., string,
                               double, boolean, etc.), a list, an anonymous object (a dict),
                               or a geometry.

                               Anonymous objects and geometries can be passed
                               in as either their normal Python forms, or following the
                               format found in Knowledge Graph entries (containing an
                               "_objectType" key, and "_properties" for anonymous objects).

                               Note: Including bind parameters not used in the query will
                               cause queries to yield nothing on ArangoDB based services,
                               while Neo4j based services will still produce results.
        -------------------    ---------------------------------------------------------------
        include_provenance     Optional boolean. When `True`, provenance entities (metadata)
                               will be included in the query results. Defaults to `False`.
        ===================    ===============================================================

        .. code-block:: python

            # Get a list of all query results
            query_gen = knowledge_graph.query_streaming("MATCH path = (n)-[r]-(n2) RETURN path LIMIT 5")
            results = list(gen)

            # Grab one result at a time
            query_gen = knowledge_graph.query_streaming("MATCH path = (n)-[r]-(n2) RETURN path LIMIT 5")
            first_result = next(query_gen)
            second_result = next(query_gen)


        """

        self._validate_import()
        url = f"{self._url}/graph/query"
        params = {
            "f": "pbf",
            "token": self._gis._con.token,
        }
        headers = {"Content-Type": "application/octet-stream"}

        # initialize encoder
        r_enc = _kgparser.GraphQueryRequestEncoder()
        r_enc.open_cypher_query = query

        # set quant params
        if input_transform:
            quant_params = self._getInputQuantParams(input_transform)
        else:
            quant_params = _kgparser.InputQuantizationParameters.WGS84_lossless()
        r_enc.input_quantization_parameters = quant_params

        # set bind parameters
        if bind_param:

            def convert_to_properties(dictionary, last_key):
                if not isinstance(dictionary, dict):
                    return dictionary

                properties_dict = {}
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        if key != "_properties" and last_key == False:
                            properties_dict[key] = {
                                "_objectType": "object",
                                "_properties": convert_to_properties(value, False),
                            }
                        elif key != "properties" and last_key == True:
                            properties_dict[key] = convert_to_properties(value, False)
                        else:
                            properties_dict[key] = convert_to_properties(value, True)
                    else:
                        properties_dict[key] = value

                return properties_dict

            for k, v in bind_param.items():
                if isinstance(v, Geometry):
                    if "_objectType" not in v.keys():
                        copy_dict = copy.deepcopy(v)
                        copy_dict["_objectType"] = "geometry"
                        converted = _kgparser.from_value_object(copy_dict)
                    else:
                        converted = _kgparser.from_value_object(v)
                    r_enc.set_param_key_value(k, converted)

                elif isinstance(v, dict):
                    copy_dict = copy.deepcopy(v)
                    if "_properties" not in copy_dict.keys():
                        changed = {
                            "_objectType": "object",
                            "_properties": convert_to_properties(copy_dict, False),
                        }
                        converted = _kgparser.from_value_object(changed)
                    else:
                        if "_objectType" not in copy_dict.keys():
                            copy_dict["_objectType"] = "object"
                        copy_dict["_properties"] = convert_to_properties(
                            copy_dict["_properties"], True
                        )
                        converted = _kgparser.from_value_object(copy_dict)
                    r_enc.set_param_key_value(k, converted)

                elif isinstance(
                    v,
                    (
                        datetime.date,
                        datetime.time,
                        datetime.datetime,
                        datetime.timedelta,
                    ),
                ):
                    r_enc.set_param_key_value(k, v)
                else:
                    converted = _kgparser.from_value_object(v)
                    r_enc.set_param_key_value(k, converted)

        # set provenance behavior
        if include_provenance == True:
            r_enc.provenance_behavior = _kgparser.ProvenanceBehavior.include
        else:
            r_enc.provenance_behavior = _kgparser.ProvenanceBehavior.exclude

        r_enc.encode()
        error = r_enc.get_encoding_result().error
        if error.error_code != 0:
            raise Exception(error.error_message)
        query_dec = _kgparser.GraphQueryDecoder()

        session = self._gis._con._session
        response = session.post(
            url=url,
            params=params,
            data=r_enc.get_encoding_result().byte_buffer,
            stream=True,
            headers=headers,
        )

        for chunk in response.iter_content(8192):
            did_push = query_dec.push_buffer(chunk)
            while query_dec.next_row():
                yield query_dec.get_current_row()

    @property
    def _datamodel(self) -> object:
        """
        Returns the datamodel for the Knowledge Graph service
        """
        self._validate_import()
        url = f"{self._url}/dataModel/queryDataModel"
        params = {
            "f": "pbf",
        }
        r_dm = self._gis._con.get(
            url, params=params, return_raw_response=True, try_json=False
        )
        buffer_dm = r_dm.content
        dm = _kgparser.decode_data_model_from_protocol_buffer(buffer_dm)
        return dm

    @property
    def datamodel(self) -> dict:
        """
        Returns the datamodel for the Knowledge Graph service
        """
        self._validate_import()
        url = f"{self._url}/dataModel/queryDataModel"
        params = {
            "f": "pbf",
        }
        r_dm = self._gis._con.get(
            url, params=params, return_raw_response=True, try_json=False
        )
        buffer_dm = r_dm.content
        dm = _kgparser.decode_data_model_from_protocol_buffer(buffer_dm)
        return dm.to_value_object()

    def apply_edits(
        self,
        adds: list[dict[str, Any]] = [],
        updates: list[dict[str, Any]] = [],
        deletes: list[dict[str, Any]] = [],
        input_transform: dict[str, Any] = None,
        cascade_delete: bool = False,
        cascade_delete_provenance: bool = False,
    ) -> dict:
        """
        Allows users to add new graph entities/relationships, update existing
        entities/relationships, or delete existing entities/relationships. For details on how the
        dictionaries for each of these operations should be structured, please refer to the samples
        further below.

        .. note::
            objectid values are not supported in dictionaries for apply_edits

        =========================   ===============================================================
        **Parameter**                **Description**
        -------------------------   ---------------------------------------------------------------
        adds                        Optional list of dicts. The list of objects to add to the
                                    graph, represented in dictionary format.
        -------------------------   ---------------------------------------------------------------
        updates                     Optional list of dicts. The list of existent graph objects that
                                    are to be updated, represented in dictionary format.
        -------------------------   ---------------------------------------------------------------
        deletes                     Optional list of dicts. The list of existent objects to remove
                                    from the graph, represented in dictionary format.
        -------------------------   ---------------------------------------------------------------
        input_transform             Optional dict. Allows a user to specify custom quantization
                                    parameters for input geometry, which dictate how geometries are
                                    compressed and transferred to the server. Defaults to lossless
                                    WGS84 quantization.
        -------------------------   ---------------------------------------------------------------
        cascade_delete              Optional boolean. When `True`, relationships connected to
                                    entities that are being deleted will automatically be deleted
                                    as well. When `False`, these relationships must be deleted
                                    manually first. Defaults to `False`.
        -------------------------   ---------------------------------------------------------------
        cascade_delete_provenance   Optional boolean. When `True`, deleting entities/relationships
                                    or setting their property values to null will result in
                                    automatic deletion of associated provenance records. When
                                    `False`, `apply_edits()` will fail if there are provenance
                                    records connected to entities/relationships intended for
                                    deletion or having their properties set to null.
        =========================   ===============================================================

        .. code-block:: python

            # example of an add dictionary- include all properties
            {
                "_objectType": "entity",
                "_typeName": "Person",
                "_id": "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXXX}"
                "_properties": {
                    "name": "PythonAPILover",
                    "hometown": "Redlands",
                }
            }

            # update dictionary- include only properties being changed
            {
                "_objectType": "entity",
                "_typeName": "Person",
                "_id": "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXXX}"
                "_properties": {
                    "hometown": "Lisbon",
                }
            }

            # delete dictionary- pass a list of id's to be deleted
            {
                "_objectType": "entity",
                "_typeName": "Person",
                "_ids": ["{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXXX}"]
            }

        :return: A `dict` showing the results of the edits.

        """

        url = self._url + "/graph/applyEdits"

        if input_transform:
            quant_params = self._getInputQuantParams(input_transform)
        else:
            quant_params = _kgparser.InputQuantizationParameters.WGS84_lossless()

        # now, make our encoder, and specify the edits to it
        enc = _kgparser.GraphApplyEditsEncoder(
            self._datamodel.spatial_reference,
            quant_params,
        )

        for edit in adds:
            enc.add(edit)
        for edit in updates:
            enc.update(edit)
        for edit in deletes:
            enc.delete_from_ids(edit)
        enc.cascade_delete = cascade_delete
        enc.cascade_delete_provenance = cascade_delete_provenance

        # encode and prepare for the post request
        enc.encode()
        res = enc.get_encoding_result()

        if res.error.error_code != 0:
            raise Exception(res.error.error_message)

        pbf_params = {
            "f": "pbf",
            "token": self._gis._con.token,
        }
        headers = {"Content-Type": "application/octet-stream"}

        # post and decode the response
        session = self._gis._con._session
        request_response = session.post(
            url,
            params=pbf_params,
            headers=headers,
            data=res.byte_buffer,
            stream=True,
        )
        apply_edits_response = request_response.content

        dec = _kgparser.GraphApplyEditsDecoder()
        dec.decode(apply_edits_response)
        results_dict = dec.get_results()

        return results_dict

    def named_object_type_adds(
        self,
        entity_types: list[dict[str, Any]] = [],
        relationship_types: list[dict[str, Any]] = [],
    ) -> dict:
        """
        Adds entity and relationship types to the data model

        `Learn more about adding named types to a knowledge graph <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-datamodel-edit-namedtypes-add.htm>`_

        ==================  ===============================================================
        **Parameter**        **Description**
        ------------------  ---------------------------------------------------------------
        entity_types        Optional list of dicts. The list of entity types to add to the
                            data model, represented in dictionary format.
        ------------------  ---------------------------------------------------------------
        relationship_types  Optional list of dicts. The list of relationship types to add
                            to the data model, represented in dictionary format.
        ==================  ===============================================================

        .. code-block:: python

            # example of a named type to be added to the data model
            {
                "name": "Person",
                "alias": "Person",
                "role": "esriGraphNamedObjectRegular",
                "strict": False,
                "properties": {
                    "Name": {
                        "name": "Name",
                        "alias": "Name",
                        "fieldType": "esriFieldTypeString",
                        "editable": True,
                        "visible": True,
                        "required": False,
                        "isSystemMaintained": False,
                        "role": "esriGraphPropertyRegular"
                    },
                    "Nickname": {
                        "name": "Nickname",
                        "alias": "Nickname",
                        "fieldType": "esriFieldTypeString",
                        "editable": True,
                        "visible": True,
                        "required": False,
                        "isSystemMaintained": False,
                        "role": "esriGraphPropertyRegular"
                    }
                }
            }


        :return: A `dict` showing the results of the named type adds.

        """
        self._validate_import()
        url = f"{self._url}/dataModel/edit/namedTypes/add"

        r_enc = _kgparser.GraphNamedObjectTypeAddsRequestEncoder()
        for entity_type in entity_types:
            r_enc.add_entity_type(entity_type)
        for relationship_type in relationship_types:
            r_enc.add_relationship_type(relationship_type)

        r_enc.encode()
        error = r_enc.get_encoding_result().error
        if error.error_code != 0:
            raise Exception(error.error_message)
        r_dec = _kgparser.GraphNamedObjectTypeAddsResponseDecoder()

        session = self._gis._con._session
        response = session.post(
            url=url,
            params={"f": "pbf"},
            data=r_enc.get_encoding_result().byte_buffer,
            stream=True,
            headers={"Content-Type": "application/octet-stream"},
        )
        r_response = response.content

        r_dec.decode(r_response)
        results_dict = r_dec.get_results()

        return results_dict

    def named_object_type_update(
        self, type_name: str, named_type_update: dict[str, Any], mask: dict[str, Any]
    ) -> dict:
        """
        Updates an entity or relationship type in the data model

        `Learn more about updating named types in a knowledge graph <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-datamodel-edit-namedtypes-type-update.htm>`_

        =================   ===============================================================
        **Parameter**        **Description**
        -----------------   ---------------------------------------------------------------
        type_name           Required string. The named type to be updated.
        -----------------   ---------------------------------------------------------------
        named_type_update   Required dict. The entity or relationship type to be updated,
                            represented in dictionary format.
        -----------------   ---------------------------------------------------------------
        mask                Required dict. A dictionary representing the properties of the
                            named type to be updated.
        =================   ===============================================================

        .. code-block:: python

            # example of a named type to be updated
            {
                "name": "Person",
                "alias": "Person",
                "role": "esriGraphNamedObjectRegular",
                "strict": False
            }

            # update the named type's alias:
            {
                "update_alias": True
            }
            # OR
            {
                "update_name": False,
                "update_alias": True,
                "update_role": False,
                "update_strict": False
            }


        :return: A `dict` showing the results of the named type update.

        """
        self._validate_import()
        url = f"{self._url}/dataModel/edit/namedTypes/{type_name}/update"

        data_model = self._datamodel
        entity_type = data_model.query_entity_type(type_name)
        relationship_type = data_model.query_relationship_type(type_name)

        r_enc = _kgparser.GraphNamedObjectTypeUpdateRequestEncoder()
        if entity_type is not None:
            r_enc.update_entity_type(named_type_update, mask)
        elif relationship_type is not None:
            r_enc.update_relationship_type(named_type_update, mask)

        r_enc.encode()
        error = r_enc.get_encoding_result().error
        if error.error_code != 0:
            raise Exception(error.error_message)
        r_dec = _kgparser.GraphNamedObjectTypeUpdateResponseDecoder()

        session = self._gis._con._session
        response = session.post(
            url=url,
            params={"f": "pbf"},
            data=r_enc.get_encoding_result().byte_buffer,
            stream=True,
            headers={"Content-Type": "application/octet-stream"},
        )
        r_response = response.content

        r_dec.decode(r_response)
        results_dict = r_dec.get_results()

        return results_dict

    def named_object_type_delete(self, type_name: str) -> dict:
        """
        Deletes an entity or relationship type in the data model

        `Learn more about deleting named types in a knowledge graph <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-datamodel-edit-namedtypes-type-delete.htm>`_

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        type_name           Required string. The named type to be deleted.
        ================    ===============================================================

        .. code-block:: python

            # Delete a named type in the data model
            delete_result = knowledge_graph.named_object_type_delete("Person")


        :return: A `dict` showing the results of the named type delete.

        """
        self._validate_import()
        url = f"{self._url}/dataModel/edit/namedTypes/{type_name}/delete"

        r_dec = _kgparser.GraphNamedObjectTypeUpdateResponseDecoder()

        session = self._gis._con._session
        response = session.post(
            url=url,
            params={"f": "pbf"},
            stream=True,
            headers={"Content-Type": "application/octet-stream"},
        )
        r_response = response.content

        r_dec.decode(r_response)
        results_dict = r_dec.get_results()

        return results_dict

    def graph_property_adds(
        self, type_name: str, graph_properties: list[dict[str, Any]]
    ) -> dict:
        """
        Adds properties to a named type in the data model

        `Learn more about adding properties in a knowledge graph <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-datamodel-edit-namedtypes-type-fields-add.htm>`_

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        type_name           Required string. The entity or relationship type to which the
                            properties will be added.
        ----------------    ---------------------------------------------------------------
        graph_properties    Required list of dicts. The list of properties to add
                            to the named type, represented in dictionary format.
        ================    ===============================================================

        .. code-block:: python

            # example of a shape property to be added to a named type
            {
                "name": "MyPointGeometry",
                "alias": "MyPointGeometry",
                "fieldType": "esriFieldTypeGeometry",
                "geometryType": "esriGeometryPoint",
                "hasZ": False,
                "hasM": False,
                "nullable": True,
                "editable": True,
                "visible": True,
                "required": False,
                "isSystemMaintained": False,
                "role": "esriGraphPropertyRegular"
            }

            # example of an integer property to be added to a named type
            {
                "name": "MyInt",
                "alias": "MyInt",
                "fieldType": "esriFieldTypeInteger",
                "nullable": True,
                "editable": True,
                "defaultValue": 123,
                "visible": True,
                "required": False,
                "isSystemMaintained": False,
                "role": "esriGraphPropertyRegular",
                "domain": "MyIntegerDomain"
            }


        :return: A `dict` showing the results of the property adds.

        """
        self._validate_import()
        url = f"{self._url}/dataModel/edit/namedTypes/{type_name}/fields/add"

        r_enc = _kgparser.GraphPropertyAddsRequestEncoder()
        for prop in graph_properties:
            r_enc.add_property(prop)

        r_enc.encode()
        error = r_enc.get_encoding_result().error
        if error.error_code != 0:
            raise Exception(error.error_message)
        r_dec = _kgparser.GraphPropertyAddsResponseDecoder()

        session = self._gis._con._session
        response = session.post(
            url=url,
            params={"f": "pbf"},
            data=r_enc.get_encoding_result().byte_buffer,
            stream=True,
            headers={"Content-Type": "application/octet-stream"},
        )
        r_response = response.content

        r_dec.decode(r_response)
        results_dict = r_dec.get_results()

        return results_dict

    def graph_property_update(
        self,
        type_name: str,
        property_name: str,
        graph_property: dict[str, Any],
        mask: dict[str, Any],
    ) -> dict:
        """
        Updates a property for a named type in the data model

        `Learn more about updating properties in a knowledge graph <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-datamodel-edit-namedtypes-type-fields-update.htm>`_

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        type_name           Required string. The entity or relationship type containing
                            the property to be updated.
        ----------------    ---------------------------------------------------------------
        property_name       Required string. The property to be updated.
        ----------------    ---------------------------------------------------------------
        graph_property      Required dict. The graph property to be updated,
                            represented in dictionary format.
        ----------------    ---------------------------------------------------------------
        mask                Required dict. A dictionary representing the properties of the
                            field to be updated.
        ================    ===============================================================

        .. code-block:: python

            # example of a shape property to be updated
            {
                "name": "MyPointGeometry",
                "alias": "MyPointGeometry",
                "fieldType": "esriFieldTypeGeometry",
                "geometryType": "esriGeometryPoint",
                "hasZ": False,
                "hasM": False,
                "nullable": True,
                "editable": True,
                "visible": True,
                "required": False,
                "isSystemMaintained": False,
                "role": "esriGraphPropertyRegular"
            }

            # example: update the property's alias
            {
                "update_alias": True
            }
            # OR
            {
                "update_name": False,
                "update_alias": True,
                "update_field_type": False,
                "update_geometry_type": False,
                "update_default_value": False,
                "update_nullable": False,
                "update_editable": False,
                "update_visible": False,
                "update_required": False,
                "update_has_z": False,
                "update_has_m": False,
                "update_domain:" False
            }


        :return: A `dict` showing the results of the property update.

        """
        self._validate_import()
        url = f"{self._url}/dataModel/edit/namedTypes/{type_name}/fields/update"

        r_enc = _kgparser.GraphPropertyUpdateRequestEncoder()
        r_enc.update_property(graph_property, mask)
        r_enc.name = property_name

        r_enc.encode()
        error = r_enc.get_encoding_result().error
        if error.error_code != 0:
            raise Exception(error.error_message)
        r_dec = _kgparser.GraphPropertyUpdateResponseDecoder()

        session = self._gis._con._session
        response = session.post(
            url=url,
            params={"f": "pbf"},
            data=r_enc.get_encoding_result().byte_buffer,
            stream=True,
            headers={"Content-Type": "application/octet-stream"},
        )
        r_response = response.content

        r_dec.decode(r_response)
        results_dict = r_dec.get_results()

        return results_dict

    def graph_property_delete(self, type_name: str, property_name: str) -> dict:
        """
        Delete a property for a named type in the data model

        `Learn more about deleting properties in a knowledge graph <https://developers.arcgis.com/rest/services-reference/enterprise/kgs-datamodel-edit-namedtypes-type-fields-delete.htm>`_

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        type_name           Required string. The entity or relationship type containing
                            the property to be deleted.
        ----------------    ---------------------------------------------------------------
        property_name       Required string. The property to be deleted.
        ================    ===============================================================

        .. code-block:: python

            # Delete a named type's property in the data model
            delete_result = knowledge_graph.graph_property_delete("Person", "Address")


        :return: A `dict` showing the results of the property delete.

        """
        self._validate_import()
        url = f"{self._url}/dataModel/edit/namedTypes/{type_name}/fields/delete"

        r_enc = _kgparser.GraphPropertyDeleteRequestEncoder()
        r_enc.name = property_name

        r_enc.encode()
        error = r_enc.get_encoding_result().error
        if error.error_code != 0:
            raise Exception(error.error_message)
        r_dec = _kgparser.GraphPropertyDeleteResponseDecoder()

        session = self._gis._con._session
        response = session.post(
            url=url,
            params={"f": "pbf"},
            data=r_enc.get_encoding_result().byte_buffer,
            stream=True,
            headers={"Content-Type": "application/octet-stream"},
        )
        r_response = response.content

        r_dec.decode(r_response)
        results_dict = r_dec.get_results()

        return results_dict
