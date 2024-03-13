from __future__ import annotations

import os
import json
from arcgis.gis import Item
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


class NBService:
    """
    A single ArcGIS Notebook Geoprocessing Tool.
    """

    _gis = None
    _url = None

    def __init__(self, url, gis):
        """initializer"""
        self._url = url
        self._gis = gis

    @property
    def properties(self):
        """Returns the manager's properties"""
        return PropertyMap(self._gis._con.get(self._url, {"f": "json"}))

    def delete(self) -> bool:
        """
        Deletes the notebook service

        :returns: Boolean
        """
        url = f"{self._url}/delete"
        params = {"f": "json"}
        res = self._gis._con.post(url, params)
        return res.get("status", "failed") == "success"


class NBServicesManager:
    """
    The `NBServicesManager` is used to manage the container of services published on the notebook server. An object of this
    class can be created using :meth:`~arcgis.gis.nb.NotebookServer.services` method of the
    :class:`~arcgis.gis.nb.NotebookServer` class
    """

    _properties = None
    _gis = None
    _nbs = None
    _url = None

    def __init__(self, url: str, gis: GIS, nbs: "NotebookServer"):
        """initializer"""
        self._url = url
        self._gis = gis
        self._nbs = nbs

    @property
    def properties(self) -> dict:
        """Returns the manager's properties"""
        return PropertyMap(self._gis._con.get(self._url, {"f": "json"}))

    @property
    def types(self) -> dict:
        """

        The types resource provides metadata and extensions that can be
        enabled on GPServer service types supported in ArcGIS Notebook
        Server. The services framework uses this information to validate a
        service and construct the objects in the service. The metadata
        contains identifiers for each object, a default list of
        capabilities, properties, and other resource information. Type
        information for a specific service type can be accessed by
        appending the type name (GPServer, for example) to this URL.

        :returns: Dict
        """
        url = f"{self._url}/types"
        params = {"f": "json"}
        return self._gis._con.get(url, params)

    @property
    def services(self) -> tuple[NBService]:
        """
        Returns a tuple of all :class:`~arcgis.gis.nb._services.NBService` created by the Notebook Server.

        :returns: tuple

        """
        service_list = []
        for service in self.properties["services"]:
            url = f"{self._url}/{service['id']}.{service['type']}"
            service_list.append(NBService(url=url, gis=self._gis))
        return tuple(service_list)

    def create(self, item: Item, title: str, description: str = None) -> Item:
        """
        ArcGIS Notebook Server supports publishing a geoprocessing service
        from a notebook. The `create` operation creates a service when a
        JSON representation of the service is submitted to it.

        To publish a service on Notebook Server, you must be an
        administrator or a user with Notebook and Publish Web Tools
        privileges. The notebook must belong to the user publishing the service.

        A notebook-to-web tool relationship is created for maintaining the
        relation between the notebook and the associated web tool created
        for the service. This relationship ensures that ownership and
        sharing permissions are the same for both. When a notebook is
        deleted, the associated web tool is also deleted.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The notebook Item to create a service from.
        ------------------     --------------------------------------------------------------------
        title                  Required string. The name of the GP tool
        ------------------     --------------------------------------------------------------------
        description            Required string. The description of the tool.
        ==================     ====================================================================

        :return:
            :class:`~arcgis.gis.Item` of the tool.

        """

        assert isinstance(item, Item) and item.type.lower() == "notebook"
        if description is None:
            description = ""

        params = {
            "description": f"{description}",
            "provider": "notebooks",
            "type": "GPServer",
            "jsonProperties": {
                "title": f"{title}",
                "notebookId": item.id,
                "tasks": [{"type": "notebook", "name": f"{item.title}"}],
            },
        }
        params = {"serviceProperties": params}
        url = f"{self._url}/createService"
        res = self._gis._con.post(url, params)
        item_id = res.get("itemId", None)
        if item_id:
            return Item(gis=self._gis, itemid=item_id)
        else:
            return res
