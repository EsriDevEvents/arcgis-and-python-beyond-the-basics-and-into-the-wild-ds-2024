from __future__ import annotations
import json
import csv
from datetime import datetime
from arcgis.gis.kubernetes._admin._base import _BaseKube
from collections import OrderedDict
from urllib.request import HTTPError
from arcgis.gis._impl._con import Connection
from arcgis.gis import GIS, Item
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._isd import InsensitiveDict
from typing import Dict, Any, List, Tuple


class DataStore(_BaseKube):
    _parent = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis, parent, initialize=False):
        """Constructor


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Required GIS. The :class:`~arcgis.gis.GIS` object.
        ------------------     --------------------------------------------------------------------
        parent                 Required :class:`~arcgis.gis.admin.kubernetes.DataStores`.
                               The Kubernetes datastore manager class.
        ==================     ====================================================================

        """
        super(DataStore, self).__init__(url=url, gis=gis, parent=parent)
        self._url = url
        self._gis = gis
        self._con = gis._con
        self._parent = parent
        if initialize:
            self._init(gis._con)

    # ----------------------------------------------------------------------
    def delete(self, force=True):
        """
        Removes the datastore from the Kubernetes Site

        :return: Boolean

        """
        if self.properties.systemManaged == False:
            return self._parent._unregister_data_item(
                path=self.properties.path, force=force
            )
        else:
            raise Exception("System Managed DataStore cannot be removed.")
        return False

    # ----------------------------------------------------------------------
    @property
    def status(self):
        """
        This resource returns health information for a **relational data store** only.

        :return: Dict
        """
        url = self._url + "/status"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def switch_role(self):
        """
        This operation promotes a standby relational data store to act as
        primary while also downgrading the existing primary to act as
        standby. This operation may take some time to complete.

        :return: Boolean
        """
        url = self._url + "/switchRole"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res


###########################################################################
class DataStores(_BaseKube):
    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis, initialize=False):
        """Constructor


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object..
        ==================     ====================================================================

        """
        super(DataStores, self).__init__(gis=gis, url=url)
        self._url = url
        self._gis = gis
        self._con = gis._con
        if initialize:
            self._init(gis._con)

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = InsensitiveDict(result)
            else:
                self._json_dict = {}
                self._properties = InsensitiveDict({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        returns the object properties
        """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def __getattr__(self, name):
        """adds dot notation to any class"""
        if self._properties is None:
            self._init()
        try:
            return self._properties.__getitem__(name)
        except:
            for k, v in self._json_dict.items():
                if k.lower() == name.lower():
                    return v
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    # ----------------------------------------------------------------------
    def __getitem__(self, key):
        """helps make object function like a dictionary object"""
        try:
            return self._properties.__getitem__(key)
        except KeyError:
            for k, v in self._json_dict.items():
                if k.lower() == key.lower():
                    return v
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, key)
            )
        except:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, key)
            )

    # ----------------------------------------------------------------------
    @property
    def url(self):
        """gets/sets the service url"""
        return self._url

    # ----------------------------------------------------------------------
    @url.setter
    def url(self, value):
        """gets/sets the service url"""
        self._url = value
        self.refresh()

    # ----------------------------------------------------------------------
    def __iter__(self):
        """creates iterable for classes properties"""
        for k, v in self._json_dict.items():
            yield k, v

    # ----------------------------------------------------------------------
    def _refresh(self):
        """reloads all the properties of a given service"""
        self._init()

    # ----------------------------------------------------------------------
    @property
    def stores(self):
        """
        returns a list of all datastores in the enterprise
        """
        stores = []
        for ds in self.properties["items"]:
            url = f"{self._url}/{ds['id']}"
            stores.append(DataStore(url, gis=self._gis, parent=self))
        return stores

    # ----------------------------------------------------------------------
    def add(
        self,
        item: str | Item,
        options: dict[str, Any] | None = None,
        sync: bool | None = None,
    ) -> dict[str, Any] | None:
        """
        Registers a new data item with the data store.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required string. The dictionary representing the data item.
                               See http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#//02r3000001s9000000
        ==================     ====================================================================


        :return:
            The data item if registered successfully, None otherwise.

        """
        if isinstance(item, Item):
            item = item.id
        res = self._register_data_item(item=item, options=options, sync=sync)
        if res["status"] == "success" or res["status"] == "exists":
            url = self._url + f"/{res['id']}"
            return DataStore(url, self._gis, self)
        elif "jobsUrl" in res:
            from arcgis.gis.kubernetes._admin._jobs import Job

            return Job(url=res["JobsUrl"], gis=self._gis)
        else:
            return res

    def validate(self, item: Dict[str, Any]) -> bool:
        """
        Validates that the path (for file shares) or connection string (for
        databases) for a specific data item is accessible to every server
        node in the site by checking against the JSON representing the data
        item, ensuring that the data item can be registered and used
        successfully within the server's data store.

        Validating a data item does not automatically register it for you.
        You need to explicitly register your data item by invoking the
        register operation.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required string. The JSON representing the data item.
                               See http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#//02r3000001s9000000
        ==================     ====================================================================
        """
        params = {"item": item, "f": "json"}
        url = self._url + "/validateDataItem"
        return (
            self._con.post(path=url, postdata=params).get("status", "failed")
            == "success"
        )

    # ----------------------------------------------------------------------
    def _register_data_item(
        self, item: Item, options: dict[str, Any] = None, sync: bool = None
    ):
        """
        Registers a new data item with the server's data store.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required string. The JSON representing the data item.
                               See http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#//02r3000001s9000000
        ==================     ====================================================================

        :return:
            A response
        """
        params = {"item": item, "f": "json"}
        if options:
            params["options"] = options
        if sync:
            params["async"] = sync
        url = self._url + "/registerItem"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def _unregister_data_item(self, path):
        """
        Unregisters a data item that has been previously registered with
        the server's data store.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        path                   Required string. The path to the share folder.
        ==================     ====================================================================

        :return:
            Bool

            .. code-block:: python

            EXAMPLE:

            path = r"/fileShares/folder_share"
            print data.unregisterDataItem(path)

        """
        url = self._url + "/unregisterItem"
        params = {"f": "json", "itempath": path, "force": True}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        elif "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def search(
        self,
        parent_path=None,
        ancestor_path=None,
        types=None,
        id=None,
        is_managed=None,
        json=False,
        **kwargs,
    ):
        """
        Use this operation to search through the various data items that are registered in the server's data store.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        parent_path            Optional string. The path of the parent under which to find items.
        ------------------     --------------------------------------------------------------------
        ancestor_path          Optional string. The path of the ancestor under which to find items.
        ------------------     --------------------------------------------------------------------
        types                  Optional string. A filter for the type of the items (for example, fgdb or folder or egdb).
        ------------------     --------------------------------------------------------------------
        id                     Optional string. A filter to search by the ID of the item.
        ------------------     --------------------------------------------------------------------
        is_managed             Optional Boolean.  Specifies if the data store is system managed.
        ------------------     --------------------------------------------------------------------
        json                   Optional Boolean. If `True`, the results will be returned as the raw
                               JSON response. `False`, the response will be a list of `DataStore`
                               objects.  The default is `False`.
        ==================     ====================================================================


        :return:
            A list of the items found matching the search criteria.

        """

        """ jenn note: list of possible types """
        params = {
            "f": "json",
        }
        if parent_path is not None:
            params["parentPath"] = parent_path
        if ancestor_path is not None:
            params["ancestorPath"] = ancestor_path
        if types is not None:
            params["types"] = types
        if id is not None:
            params["id"] = id
        if "decrypt" in kwargs.keys():
            params["decrypt"] = kwargs["decrypt"]
        if is_managed is not None:
            params["isManaged"] = is_managed
        url = self._url + "/findItems"
        res = self._con.post(path=url, postdata=params)
        if "items" in res and json == False:
            stores = []
            for ds in res["items"]:
                url = f"{self._url}/{ds['id']}"
                stores.append(DataStore(url, gis=self._gis, parent=self))
            return stores
        elif "items" in res and json:
            return res["items"]
        return res

    # ----------------------------------------------------------------------
    @property
    def config(self):
        """
        Gets/Sets information on the data store's configuration properties
        that affect the behavior of the data holdings of the server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        config                 Required string. A JSON string containing the data store configuration.
        ==================     ====================================================================


        :return: dict
        """
        url = self._url + "/config"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @config.setter
    def config(self, config):
        """
        Gets/Sets information on the data store's configuration properties
        that affect the behavior of the data holdings of the server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        config                 Required string. A JSON string containing the data store configuration.
        ==================     ====================================================================

        :return:
           Dict

        """
        if config is None:
            config = {}
        params = {"f": "json", "datastoreConfig": config}
        url = self._url + "/config/update"
        return self._con.post(path=url, postdata=params)
