"""
Schematics are simplified representations of networks, intended to explain their structure and make the way they operate
understandable. The arcgis.schematics module contains the types and functions for working with schematic layers and
datasets.

"""

from arcgis.gis import Layer

"""This class provides access to diagrams and schematic layers, as well as diagram templates."""


class SchematicLayers(Layer):
    def __init__(self, url, gis=None):
        super(SchematicLayers, self).__init__(url, gis)
        try:
            from arcgis.gis.server._service._adminfactory import AdminServiceGen

            self.service = AdminServiceGen(service=self, gis=gis)
        except:
            pass

    @property
    def diagrams(self):
        """
        The Schematic Diagrams resource represents all the schematic diagrams
        under a schematic service. It is returned as an array of `Schematic
        Diagram resources <https://developers.arcgis.com/rest/services-reference/enterprise/schematic-diagram.htm>`_
        by the REST API.
        """
        params = {"f": "json"}
        exportURL = self._url + "/diagrams"
        return self._con.get(path=exportURL, params=params, token=self._token)

    # ----------------------------------------------------------------------
    @property
    def folders(self):
        """
        The Schematic Folders resource represents the set of schematic folders
        in the schematic dataset(s) related to the schematic layers under a
        schematic service. It is returned as an array of `Schematic Folder Objects <https://developers.arcgis.com/documentation/common-data-types/schematic-folder-object.htm>`_
        by the REST API.
        """
        params = {"f": "json"}
        exportURL = self._url + "/folders"
        return self._con.get(path=exportURL, params=params, token=self._token)

    # ----------------------------------------------------------------------
    @property
    def layers(self):
        """
        The Schematic Layers resource represents all the schematic layers
        under a schematic service published by ArcGIS Server. It is returned
        as an array of `Schematic Layer resources <https://developers.arcgis.com/rest/services-reference/enterprise/schematic-layer.htm>`_
        by the REST API.
        """
        params = {"f": "json"}
        exportURL = self._url + "/schematicLayers"
        return self._con.get(path=exportURL, params=params, token=self._token)

    # ----------------------------------------------------------------------
    @property
    def templates(self):
        """
        The Schematic Diagram Templates represents all the schematic diagram
        templates related to the published schematic layers under a schematic
        service. It is returned as an array of `Schematic Diagram Template
        resources <https://developers.arcgis.com/rest/services-reference/enterprise/schematic-diagram-template.htm>`_
        by the REST API.
        """
        params = {"f": "json"}
        exportURL = self._url + "/templates"
        return self._con.get(path=exportURL, params=params, token=self._token)

    # ----------------------------------------------------------------------
    def search_diagrams(
        self, whereClause=None, relatedObjects=None, relatedSchematicObjects=None
    ):
        """
        The Schematic Search Diagrams operation is performed on the schematic
        service resource. The result of this operation is an array of Schematic
        Diagram Information Object.

        It is used to search diagrams in the schematic service by criteria;
        that is, diagrams filtered out via a where clause on any schematic
        diagram class table field, diagrams that contain schematic features
        associated with a specific set of GIS features/objects, or diagrams
        that contain schematic features associated with the same GIS features/
        objects related to another set of schematic features.

        See `Schematic Search Diagrams <https://developers.arcgis.com/rest/services-reference/enterprise/schematic-search-diagrams.htm>`_
        for full details.

        =======================      =======================================================================
        **Parameter**                 **Description**
        -----------------------      -----------------------------------------------------------------------
        whereClause                  A where clause for the query filter. Any legal SQL where clause
                                     operating on the fields in the schematic diagram class table is allowed.
                                     See the `Schematic diagram class fields  <https://developers.arcgis.com/rest/services-reference/enterprise/schematic-search-diagrams.htm>`_
                                     table for the exact list of field names that can be used in this where
                                     clause.
        -----------------------      -----------------------------------------------------------------------
        relatedObjects               An array containing the list of the GIS features/objects IDs per
                                     feature class/table name that are in relation with schematic features
                                     in the resulting queried diagrams. Each GIS feature/object ID
                                     corresponds to a value of the OBJECTID field in the GIS feature
                                     class/table.
        -----------------------      -----------------------------------------------------------------------
        relatedSchematicObjects      An array containing the list of the schematic feature names per
                                     schematic feature class ID that have the same associated GIS
                                     features/objects with schematic features in the resulting queried
                                     diagrams. Each schematic feature name corresponds to a value of the
                                     ``SCHEMATICTID`` field in the schematic feature class.
        =======================      =======================================================================
        """
        params = {"f": "json"}
        if whereClause:
            params["where"] = whereClause
        if relatedObjects:
            params["relatedObjects"] = relatedObjects
        if relatedSchematicObjects:
            params["relatedSchematicObjects"] = relatedSchematicObjects

        exportURL = self._url + "/searchDiagrams"
        return self._con.get(path=exportURL, params=params, token=self._token)
