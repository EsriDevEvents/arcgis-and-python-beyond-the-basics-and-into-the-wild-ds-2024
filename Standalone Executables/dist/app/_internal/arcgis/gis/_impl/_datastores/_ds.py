from __future__ import annotations
import time as _time
from typing import Union, Any
import uuid
from arcgis.gis import GIS
from .._con import Connection
from arcgis._impl.common._mixins import PropertyMap
from arcgis import env as _env
from arcgis.gis._impl._jb import StatusJob
from arcgis.gis import Item
import concurrent.futures

###########################################################################


class PortalDataStore(object):
    """

    The ``PortalDatastore`` object provides access to operations that allow you
    to manage and work with user-managed `data store items <https://enterprise.arcgis.com/en/portal/latest/use/manage-data-store-items.htm>`_ .
    Access an instance of this class using the `datastore` property of a
    :class:`~arcgis.gis.GIS` object.

    .. code-block:: python

        >>> gis = GIS("organzation url", "username", "<password>")
        >>>
        >>> portal_dstore = gis.datastore
        >>> type(portal_dstore)

        <class 'arcgis.gis._impl._datastores._ds.PortalDataStore'>

    With this class you can:

    - Validate a data store item against your server.
    - Register a data store item to your server.
    - Retrieve a list of servers your data store item is registered to.
    - Refresh your data store registration information on the server.
    - Unregister a data store from your server.
    - Publish and retrieve layers in bulk, and delete bulk-published layers.

    See `User-managed data stores <https://enterprise.arcgis.com/en/portal/latest/use/data-store-items.htm>`_
    for detailed explanation of the item types this class manages. Also,
    see `Data Item <https://developers.arcgis.com/rest/enterprise-administration/server/dataitem.htm>`_
    for technical details on managing server side components.

    .. note::
        This class provides different functionality than the :class:`~arcgis.gis.server.Datastore` object
        which is used for data stores registered directly with a :class:`~arcgis.gis.server.Server`.
    """

    _con = None
    _gis = None
    _url = None
    _properties = None
    # ----------------------------------------------------------------------

    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< PortalDataStore @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< PortalDataStore @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def describe(self, item, server_id, path, store_type="datastore"):
        """
        The ``describe`` method is used to list the contents of a data store
        added to Enterprise as a data store item. A client can use this
        method multiple times to discover the contents of the
        data store incrementally. For example, the client can request a
        description of the root, and then request sub-folders.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The data store :class:`~arcgis.gis.Item` to describe.
        ------------------     --------------------------------------------------------------------
        server_id              Required String. The unique id of the server the item is registered
                               to.

                               .. note::
                                   You can retrieve the `server_id` value from the dictionary
                                   returned by the federation.

                                   .. code-block:: python

                                       # Retrieve server ids
                                       >>> gis_federation = gis.admin.federation.servers
                                       >>>
                                       >>> server_list = gis_federation["servers"]
                                       >>> server_list
                                       >>>
                                       >>> [{'id': 'W2ozgK50Le9CjrI4',
                                            'name': 'mymachine.domain.com:6443',
                                            'url': 'https://mymachine.domain.com/mywebadaptor_name',
                                            'isHosted': True,
                                            'adminUrl': 'https://mymachine.domain.com:6443/arcgis',
                                            'serverRole': 'HOSTING_SERVER',
                                            'serverFunction': '',
                                            'webgisServerTrustKey': ''}]
                                       >>>
                                       >>> host_id = server_list[0]["id"]
        ------------------     --------------------------------------------------------------------
        path                   Required String. The path to any data store's root ("/"), or the
                               path to a sub-folder or entity inside the root.
        ------------------     --------------------------------------------------------------------
        store_type             Required String. For root resource the object type should be
                               `datastore`, and for sub-entities, the value depends upon data type
                               of the data store. Value can be determined by looking at
                               ``type`` values returned by :meth:`~PortalDataStore.describe`
        ==================     ====================================================================

        .. code-block:: python

            # Usage Example: Data store item added using an enterprise geodatabase

            >>> ds_items = gis.content.search("*", item_type="Data Store")
            >>>
            >>> for ds_item in ds_items:
            >>>     print(f"{ds_item.title:33}{ds_item.id}{' '*2}{ds_item.get_data()['type']}")
            >>>
               City_Service                     ea9a628b037........7cf29eda554d3  bigDataFileShare
               CityParks2 Datastore Item API    a188c1f5b42........eb1f8dea03a8e  egdb
               various_imagery                  c6971f24a25........fe22d7142f835  folder
            >>>
            >>> egdb_dsitem = ds_items[1]
            >>>
            >>> portal_ds = gis.datastore
            >>>
            >>> describe_job = portal_ds.describe(item=egdb_dsitem,
                                                  server_id=host_id,
                                                  path="/",
                                                  store_type="datastore")
            >>>
            >>> while describe_job.status == "processing":
            >>>     continue
            >>> if describe_job.status == "succeeded":
            >>>     describe_egdb = describe_job.result()
            >>> else:
            >>>     print(f"Job status: {describe_job.status}")
            >>>     print(f"Job messages: {describe_job.messages}")
            >>>
            >>> describe_egdb

                {'id': '6bd436223f2b433285f58cf1df5bccef',
                 'definition': {'operation': 'describe',
                  'datastoreId': 'a25ae2f4c4674bf4799eb2f4fdae3b8f',
                  'serverId': 'K2ozgC10aL6ECdm2',
                  'path': '/',
                  'type': 'datastore'},
                 'status': 'succeeded',
                 'created': 166512637...1,
                 'modified': 16574263...8,
                 'messages': [],
                 'result': {'datastore': {'name': 'world_db_0klt50rsl52d4',
                   'type': 'egdb',
                   'path': '/',
                   'datastoreId': 'a25ae2f4c4674bf4799eb2f4fdae3b8f'},
                  'children': [{'name': 'world.DATAo.World_Data',
                    'displayName': 'World_Data',
                    'type': 'featureDataset',
                    'datastoreId': 'a25ae2f4c4674bf4799eb2f4fdae3b8f',
                    'path': '/world.DATAo.World_Data'},
                   {'name': 'world.DATAo.Forest_Landscapes',
                    'displayName': 'Forest_Landscapes',
                    'type': 'featureDataset',
                    'datastoreId': 'a25ae2f4c4674bf4799eb2f4fdae3b8f',
                    'path': '/world.DATAo.Forest_Landscapes'}]}}
            >>>
            >>> # describe sub-entity of the database data store
            >>>
            >>> describe_job_sub = portal_ds.describe(item=egdb_dsitem,
                                                      server_id=host_id,
                                                      path="/world.DATAo.Forest_Landscapes'",
                                                      store_type="featureDataset")

        :return:
            A :class:`~arcgis.gis._impl._jb.StatusJob` object

        """
        if isinstance(item, Item):
            item = item.id
        params = {
            "datastoreId": item,
            "serverId": server_id,
            "path": path,
            "type": store_type,
            "f": "json",
        }
        url = f"{self._url}/describe"
        res = self._con.get(url, params)
        _time.sleep(0.5)
        executor = concurrent.futures.ThreadPoolExecutor(1)
        futureobj = executor.submit(
            self._status, **{"job_id": res["jobId"], "key": res["key"]}
        )
        executor.shutdown(False)
        return StatusJob(
            future=futureobj,
            op="Describe DataStore",
            jobid=res["jobId"],
            gis=self._gis,
            notify=_env.verbose,
            extra_marker="",
            key=res.get("key", None),
        )

    # ----------------------------------------------------------------------
    def _status(self, job_id, key=None):
        """
        Checks the status of an export job

        :return: dict
        """
        params = {}
        if job_id:
            url = f"{self._gis._portal.resturl}portals/self/jobs/{job_id}"
            params["f"] = "json"
            if key:
                params["key"] = key
            res = self._con.post(url, params)
            while res["status"] not in [
                "completed",
                "complete",
                "succeeded",
            ]:
                res = self._con.post(url, params)
                if res["status"] == "failed":
                    raise Exception(res)
            count = 0
            while (
                res["status"] in ["completed", "complete", "succeeded"]
                and not "result" in res
                and count < 10
            ):
                res = self._con.post(url, params)
                if "result" in res:
                    return res
                count += 1
            return res
        else:
            raise Exception(res)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        The ``properties`` property retrieves the properties of the current
        :class:`~PortalDataStore` object

        :return:
           A dictionary of properties pertaining to the :class:`~PortalDataStore`
        """
        if self._properties is None:
            params = {"f": "json"}
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        return self._properties

    # ----------------------------------------------------------------------
    def register(self, item, server_id, bind=False):
        """

        The ``register`` method allows for a data store :class:`~arcgis.gis.Item`
        that has been added to the Enterprise portal to be registered with
        Enterprise servers.

        .. note::
            Before registering a data store :class:`~arcgis.gis.Item`, it is
            recommended that you :meth:`~PortalDataStore.validate` it with
            the server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required data store :class:`~arcgis.gis.Item` or Item Id
                               string.

                               .. note::
                                   A data store :class:`~arcgis.gis.Item` can be registered on
                                   multiple servers.
        ------------------     --------------------------------------------------------------------
        server_id              Required String. The unique id of the server you want to register
                               the data store item with.
        ------------------     --------------------------------------------------------------------
        bind                   Optional Boolean. Specifies whether to bind the data store item to
                               the federated server. For more information about binding a data
                               store to additional federated servers, see the note below.
                               The default value is ``False``.
        ==================     ====================================================================

        .. code-block:: python

            # Usage Example: adding data store item sourced by an enterprise
            # geodatabase and registering with server

            >>> # Get the server id value for registration
            >>> servers_dict = gis.admin.federation.servers
            >>> server_list = servers_dict["servers"]
            >>> host_id = [srvr["id"]
            >>>            for srvr in server_list
            >>>            if srvr["serverRole"] == "HOSTING_SERVER"][0]
            >>>
            >>> # Get the host server's DataStoreManager to create an
            >>> # encrypted password string for the database
            >>> host = gis.admin.servers.get(role="HOSTING_SERVER")[0]
            >>> host_dsmgr = host.datastores
            >>> conn_file_sql = r"/pathway/to/connection_file/your_connection.sde"
            >>> conn_string = host_dsmgr.generate_connection_string(conn_file_sql)
            >>>
            >>> # Add the data store item to the Enterprise Portal
            >>> text_param = {"info": {"isManaged": False,
            >>>                        "dataStoreConnectionType": "shared",
            >>>                        "connectionString": conn_string},
            >>>               "type": "egdb",
            >>>               "path": "/enterpriseDatabases/sql_server_datastore"}
            >>> item_properties = {"title": "SqlServer Datastore Item API",
            >>>                    "type": "Data Store",
            >>>                    "tags": "api_created,datastore_item,bulk_publishing",
            >>>                    "snippet": "Adding a datastore item to use api for management."}
            >>>
            >>> ds_item = gis.content.add(item_properties=item_properties,
            >>>                           text=text_param)
            >>>
            >>> # Get the Enteprises PortalDataStore and register with the server
            >>> portal_ds = gis.datastore
            >>>
            >>> portal_ds.register(item=ds_item,
            >>>                    server_id=host_id,
            >>>                    bind=False)

        :return:
            A boolean indicating success (True), or failure (False)

        .. note::
            To create a data store :class:`~arcgis.gis.Item` from a data store previously registered
            with an Enterprise Server, see the `Create a data store item from an existing data store <https://enterprise.arcgis.com/en/portal/latest/administer/windows/create-item-from-existing-data-store.htm#ESRI_SECTION1_58D081604CF841AC80D527D34A67660C>`_ page in the Enterprise Portal documentation.

        """
        if isinstance(item, Item):
            item_id = item.id
        elif isinstance(item, str):
            item_id = item
        url = "{base}/addToServer".format(base=self._url)
        params = {"f": "json", "datastoreId": item_id, "serverId": server_id}
        if not bind is None:
            params["bindToServerDatastore"] = bind
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    @property
    def _all_datasets(self):
        """
        The _all_datasets resource page provides access to bulk publishing
        operations. These operations allow users to publish and synchronize
        datasets from a given data store, return a list of published layers,
        and remove all previously published layers in preparation for the
        deregistration of a data store.

        :return: Boolean

        """
        url = "{base}/allDatasets".format(base=self._url)
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        elif "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def delete_layers(self, item):
        """
        The ``delete_layers`` method removes all layers published from a database
        data store :class:`~arcgis.gis.Item`.

        .. note::
            Before a data store :class:`~arcgis.gis.Item` can be unregistered
            from a server, all of its bulk-published layers must be deleted.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The database data store
                               :class:`~arcgis.gis.Item` from which to delete all
                               the published layers.
        ==================     ====================================================================

        .. code-block:: python

            # Usage Example

            >>> # Get data store item sourced by a database
            >>> db_dsitem = gis.content.get("<item_id of data store item>")
            >>>
            >>> portal_ds = gis.datastore
            >>> portal_ds.delete_layers(db_dsitem)

        :return:
            A boolean indicating success (``True``), or failure (``False``)


        """
        if isinstance(item, Item):
            item_id = item.id
        else:
            item_id = item
            item = self._gis.content.get(item_id)
        params = {"f": "json", "datastoreId": item_id}
        url = f"{self._url}/allDatasets/deleteLayers"
        res = self._con.post(url, params)
        if res["success"] == True:
            status = item.status()
            while status["status"].lower() != "completed":
                status = item.status()
                if status["status"].lower() == "failed":
                    return False
                else:
                    _time.sleep(2)
            return True
        return False

    # ----------------------------------------------------------------------
    def layers(self, item):
        """
        The ``layers`` operation returns a list of layers bulk published from
        a database data store item using the
        :meth:`~PortalDataStore.publish_layers` method.

        .. note::
            The ``layers`` method returns a list of dictionaries. Each
            dictionary contains a `layer` key and a `dataset` key for each
            layer created during publishing.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The data store :class:`~arcgis.gis.Item` to list all
                               published layers and registered datasets.
        ==================     ====================================================================

        .. code-block:: python

            # Usage Example
            >>> ds_items = gis.content.get("*", item_type="Data Store")
            >>>
            >>> db_dsitem = [ds
            >>>              for ds in ds_items
            >>>              if ds.get_data()["type"] == "egdb"][0]
            >>>
            >>> portal_ds = gis.datastore
            >>>
            >>> portal_ds.layers(db_dsitem)

            [{'layer': {'id': '059...711',
                        'title': 'Boundary_GTM',
                        'created': 1667433642607,
                        'modified': 1667433732631,
                        'url': 'https://<url>/<webadaptor>/rest/services/<folder_name>/Boundary_GTM/FeatureServer',
                        'type': 'Feature Service'},
              'dataset': {'name': 'world.dto.CountryBoundary_GTM',
                          'displayName': 'CountryBoundary_Guatemala',
                          'type': 'featureClass',
                          'datastoreId': 'ad38...01b8e',
                          'path': '/world.dto.GTM_Datasets/world.dto.Boundary_GTM'}},
                          .
                          .
                          .
             {'layer': {'id': '0d5a6e6f2003483ba987eb044ca4dec6',
                        'title': 'dk_lau_2016',
                        'created': 1667433476227,
                        'modified': 1667433605914,
                        'url': 'https://<url>/<webadaptor>/rest/services/<folder_name>/dk_lau_2016/MapServer',
                        'type': 'Map Service'},
              'dataset': {'name': 'world.dto.dk_lau_2016',
                          'displayName': 'dk_lau_2016',
                          'type': 'table',
                          'datastoreId': 'ad38...01b8e',
                          'path': '/world.dto.dk_lau_2016'}}]

        :return:
            A list of dictionaries. For each layer, a dictionary of information about the
            layer and the source dataset from which it was published.
        """

        if isinstance(item, Item):
            item_id = item.id
        else:
            item_id = item
            item = self._gis.content.get(item_id)

        url = "{base}/allDatasets/getLayers".format(base=self._url)
        params = {"f": "json", "datastoreId": item_id}
        res = self._con.post(url, params)
        if "layerAndDatasets" in res:
            return res["layerAndDatasets"]
        return []

    # ----------------------------------------------------------------------
    def publish(
        self,
        config: dict,
        server_id,
        folder=None,
        description=None,
        tags: Union[list, str] = None,
    ):
        """
        The ``publish`` operation is used to publish services by reference from
        specific datasets in a data store.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        config                 Required Dictionary.  This is the service configuration property
                               and it must contain the reference to the data in the data store. It
                               specifies the data store Id and the path of the data.  A client can
                               discover the proper paths of the data by using the
                               :meth:`~PortalDataStore.describe` method.

                               .. note::
                                   The ``cacheStoreId`` value for this dictionary correpsonds to the
                                   datastore id value.

                               .. code-block:: python

                                   # Example format

                                   >>> config = {"type":"SceneServer",
                                   >>>           "serviceName":"sonoma",
                                   >>>           "properties":{"pathInCachedStore":"/v17_i3s/SONOMA_LiDAR.i3srest",
                                   >>>                         "cacheStoreId":"d7b072...00d9"}}
        ------------------     --------------------------------------------------------------------
        server_id              Required String. The unique Id of the server to publish to.

                               .. note::
                                   Any :class:`~arcgis.gis.server.Server` id can be obtained from
                                   the dictionary returned by by the
                                   :attr:`~arcgis.gis.admin.Federation.servers` property of
                                   :class:`~arcgis.gis.admin.Federation` objects.
        ------------------     --------------------------------------------------------------------
        folder                 Optional String. The name of the folder on the server to store the
                               service. If none is provided, it is placed in the root.
        ------------------     --------------------------------------------------------------------
        description            Optional String. An optional string to attach to the
                               generated :class:`~arcgis.gis.Item`.
        ------------------     --------------------------------------------------------------------
        tags                   Optional list. An array of descriptive words that describes the
                               newly published :class:`~arcgis.gis.Item`.
        ==================     ====================================================================

        .. code-block:: python

            # Usage Example

            >>> portal_ds = gis.datastore
            >>>
            >>> service_config = {"type":"SceneServer",
            >>>                   "serviceName":"sonoma",
            >>>                   "properties":{"pathInCachedStore":"/v17_i3s/SONOMA_LiDAR.i3srest",
            >>>                                 "cacheStoreId":"d7b072...00d9"}}
            >>>
            >>> server_list = gis.admin.Federation.servers["servers"]
            >>> gis_server_id = [s["id"]
            >>>                  for s in server_list
            >>>                  if s["serverRole"] == "HOSTING_SERVER"]
            >>>
            >>> pub_job = portal_ds.publish(config= service_config,
            >>>                             server_id= gis_server_id)
            >>>
            >>> if pub_job.status == "succeeded":
            >>>     published_item = pub_job.result()

        :return:
            A :class:`~arcgis.gis._impl._jb.StatusJob` object

        """
        url = f"{self._url}/publish"
        if isinstance(tags, list):
            tags = ",".join([str(t) for t in tags])

        params = {
            "serviceConfiguration": config,
            "serverId": server_id,
            "serverFolder": folder,
            "description": description,
            "tags": tags,
            "f": "json",
        }
        res = self._con.post(url, params)
        executor = concurrent.futures.ThreadPoolExecutor(1)
        futureobj = executor.submit(
            self._status, **{"job_id": res["jobId"], "key": res["key"]}
        )
        executor.shutdown(False)
        return StatusJob(
            future=futureobj,
            op="Publish",
            jobid=res["jobId"],
            gis=self._gis,
            notify=_env.verbose,
            extra_marker="",
            key=res.get("key", None),
        )

    # ----------------------------------------------------------------------
    def servers(self, item):
        """
        The ``servers`` property returns a list of your servers that a given
        data store has been registered to. This operation returns the
        serverId, the server name, both the server and admin URLs, and
        whether or not the server is hosted.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The data store :class:`~arcgis.gis.Item` to
                               for which to list all registered servers.
        ==================     ====================================================================

        .. code-block:: python

            # Usage Example

            >>> # Get a data store item
            >>> ds_item = gis.content.search("*", item_type="Data Store)[0]
            >>>
            >>> server_list = gis.datastore.servers(ds_item)
            >>> server_list[0]

            [{'id': 'T4omqC59LaH4vLi0',
              'name': 'jserver.domain.com:6443',
              'adminUrl': 'https://server.domain.com:6443/arcgis',
              'url': 'https://jserver.domain.com/server',
              'isHosted': True,
              'serverType': 'ArcGIS',
              'serverRole': 'HOSTING_SERVER',
              'serverFunction': ''}]

        :return:
            A list of dictionaries with metadata about each server the item is
            registered with.

        """

        if isinstance(item, Item):
            item_id = item.id
        else:
            item_id = item
            item = self._gis.content.get(item_id)

        url = self._url + "/getServers"
        params = {"f": "json", "datastoreId": item_id}
        res = self._con.post(url, params)
        if "servers" in res:
            return res["servers"]
        return res

    # ----------------------------------------------------------------------
    def publish_layers(
        self,
        item: Item,
        srv_config: dict[str, Any],
        server_id: str,
        *,
        folder: str | None = None,
        server_folder: str | None = None,
        sync_metadata: bool | None = None,
        use_config: bool | None = None,
    ):
        """
        The ``publish_layers`` operation publishes, or syncs, the datasets from a
        database data store :class:`~arcgis.gis.Item`. This results in at least
        one layer per dataset.

        .. note::
            When this operation is called for the first time, an **argument for
            every parameter must be provided**. On subsequent calls, this method
            will synchronize the datasets in the data store with the layers
            creted in the Enterprise, which includes both publishing layers
            from newly added datasets and removing layers for datasets no
            longer found in the data store.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The data store :class:`~arcgis.gis.Item` holding
                               the content to publish.
        ------------------     --------------------------------------------------------------------
        srv_config             Required Dict. The JSON that will be used as a template for all the
                               services that will be published or synced. This JSON can be used to
                               change the properties of the output map and feature services.

                               .. note::
                                   Only map service configurations with feature services enabled are
                                   supported by this parameter.
        ------------------     --------------------------------------------------------------------
        server_id              Required String. The serverId that the datasets will be published to.

                               .. note::
                                   Use `gis.admin.federation.servers` to get the id values for
                                   individual servers federated with the Enterprise portal.
        ------------------     --------------------------------------------------------------------
        folder                 Required String. The folder to which the datasets will be published.

                               .. note::
                                   This folder must exist in the Enteprise portal.
        ------------------     --------------------------------------------------------------------
        server_folder          Required String. The name of the server folder.

                               .. note::
                                   If this folder does not exist, the method will
                                   create it.
        ------------------     --------------------------------------------------------------------
        sync_metadata          Optional bool. Determines if item info details are updated using the
                               metadata of the source dataset when a sync is performed. The default
                               is false.
        ------------------     --------------------------------------------------------------------
        use_config             Optional bool. When true, the new `srv_config` will be applied to
                               all layers.
        ==================     ====================================================================

        :return:
            Boolean object.

        .. code-block:: python

            # Usage Example: bulk publishing from enterprise geodatabase data store item

            >>> ds_items = gis.content.search("*", item_type="Data Store")
            >>>
            >>> db_dsitem = [ds for ds in ds_items if ds.get_data()["type"] == "egdb"][0]
            >>>
            >>> portal_folderid = [f["id"]
            >>>                    for f in gis.users.me.folders
            >>>                    if f["title"] == "My_Bulk_Layers_Folder"]
            >>>
            >>> service_template = {"serviceName": None,
            >>>                     "type": "MapServer",
            >>>                     "capabilities":"Map,Query",
            >>>                     "extensions": [{"typeName": "FeatureServer",
            >>>                                     "capabilities":"Query,Create,Update,Delete",
            >>>                                     "enabled": "true",
            >>>                                     "properties": {"maxRecordCount": 3500}}]}
            >>>
            >>>
            >>> portal_ds = gis.datastore
            >>>
            >>> bulk_publish_job = portal_ds.publish_layers(item = db_dsitem,
            >>>                                             srv_config = service_template,
            >>>                                             server_id = host_id,
            >>>                                             folder = portal_folderid,
            >>>                                             server_folder="bulk_egdb_layers")
            >>> bulk_publish_job
            True

        """
        if server_folder is None:
            base = "buld_pub_"
            if isinstance(item, Item):
                base = item.title.lower().replace(" ", "")
            server_folder = f"{base}{uuid.uuid4().hex[:3]}"
        if folder is None:
            from arcgis.gis import UserManager, User, ContentManager

            isinstance(self._gis, GIS)
            cm = self._gis.content
            folder = cm.create_folder(folder=f"srvc_folder_{uuid.uuid4().hex[:5]}")[
                "id"
            ]
        if isinstance(item, Item):
            item_id = item.id
        else:
            item_id = item
            item = self._gis.content.get(item_id)
        url = self._url + "/allDatasets/publishLayers"
        params = {
            "f": "json",
            "datastoreId": item_id,
            "templateSvcConfig": srv_config,
            "portalFolderId": folder,
            "serverId": server_id,
            "serverFolder": server_folder,
        }
        if not sync_metadata is None and isinstance(sync_metadata, bool):
            params["syncItemInfo"] = sync_metadata
        if not use_config is None and isinstance(use_config, bool):
            params["applySvcConfigChanges"] = use_config
        res = self._con.post(url, params)
        if res["success"] == True:
            status = item.status()
            while status["status"].lower() != "completed":
                status = item.status()
                if status["status"].lower() == "failed":
                    return False
                else:
                    _time.sleep(2)
            return True
        return False

    # ----------------------------------------------------------------------
    def unregister(self, item, server_id):
        """
        The ``unregister`` method removes the data store from the list of data
        stores on the server.

        .. code-block:: python

            # Usage Example
            >>> ds_item = gis.content.get("<item id of data store>")
            >>>
            >>> server_info = gis.admin.federation.servers["servers"][0]
            >>> server_id = server_info["id"]
            >>>
            >>> gis.datastore.unregister(item = dsitem,
            >>>                          server_id = server_id)

        :return:
            A boolean indicating success (True), or failure (False)

        """
        if isinstance(item, Item):
            item_id = item.id
        elif isinstance(item, str):
            item_id = item
        url = self._url + "/removeFromServer"
        params = {"f": "json", "datastoreId": item_id, "serverId": server_id}
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def refresh_server(self, item, server_id):
        """
        The ``refresh_server`` method updates the server with information that
        changed in the data store. See
        `Manage data store items <https://enterprise.arcgis.com/en/portal/latest/use/manage-data-store-items.htm>`_
        for more information.

        .. note::
            After a data store :class:`~arcgis.gis.Item` has been registered, there
            may be times in which the registration information may be changed. When
            changes like these occur, the server will need to be updated with
            the newly configured information so that your users will still be
            able to access the data store item without interruption.

        .. warning::
            This operation can only be performed after the data
            store information has been updated.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required data store :class:`~arcgis.gis.Item` or item id (as string).
                               The data store to register with the server. Note that a data store
                               can be registered on multiple servers.
        ------------------     --------------------------------------------------------------------
        server_id              Required String. The unique id of the server you want to register
                               the datastore with.
        ==================     ====================================================================

        .. code-block:: python

            # Usage Example
            >>> portal_ds = gis.datastore
            >>>
            >>> ds_item = gis.content.search("*", item_type="Data Store")[0]
            >>> server_list = gis.admin.federation.servers["servers]
            >>>
            >>> host_id = [s["id"]
            >>>            for s in server_list
            >>>            if s["serverRole"] == "HOSTING_SERVER][0]
            >>>
            >>> portal_ds.refresh_server(item = ds_item, server_id = host_id)

        :return:
            A boolean indicating success (True), or failure (False)

        """
        if isinstance(item, Item):
            item_id = item.id
        elif isinstance(item, str):
            item_id = item

        url = self._url + "/refreshServer"
        params = {"f": "json", "datastoreId": item_id, "serverId": server_id}
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def validate(self, server_id, item=None, config=None, future=False):
        """
        The ``validate`` method ensures that your ArcGIS Server can connect and use
        use the datasets accessed by a given data store item. The data store
        :class:`~arcgis.gis.Item` can be validated by using either the `id`
        property, or an item object tself as the `item` argument.

        .. note::
            While this operation can be called before or after the data store item
            has been registered, it is recommended to validate before
            registration on the server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        server_id              Required String. The unique id of the server with which you want to
                               register the data store.
        ------------------     --------------------------------------------------------------------
        item                   Optional. The item id or data store
                               :class:`~arcgis.gis.Item` to validate. Required if no ``config``
                               provided.

                               .. note::
                                   A single data store item can be registered on multiple servers.
        ------------------     --------------------------------------------------------------------
        config                 Optional dict. The connection information for a new datastore.
                               Required if no ``item`` provided.
        ------------------     --------------------------------------------------------------------
        future                 Optional bool. Indicates whether to run the validate operation
                               asynchronously. The default is `False`.
        ==================     ====================================================================

        .. code-block:: python

            # Usage Example: Validating an added item against the Enterprise Raster Analytics server

            >>> gis = GIS(url="<url to GIS>", username="<username>", password="<password>")
            >>>
            >>> ds_item = gis.content.search("*", item_type="data store")[0]
            >>>
            >>> server_list = gis.admin.federation.servers["servers]
            >>> raster_id = [srv["id"]
            >>>              for srv in server_list
            >>>              if srv["function"] == "RasterAnalytics"][0]
            >>>
            >>> portal_ds = gis.datastore
            >>> portal_ds.validate(server_id = raster_id,
            >>>                    item = ds_item)

        :return:
            A boolean indicating success (True), or failure (False)

        """

        if item and isinstance(item, Item):
            item_id = item.id
        elif item and isinstance(item, str):
            item_id = item
        else:
            item_id = None

        url = self._url + "/validate"
        params = {"f": "json", "serverId": server_id}
        if item:
            params["datastoreId"] = item_id
        elif config:
            import json

            params["datastore"] = json.dumps(config)
        else:
            raise ValueError("Invalid parameters, an item or config is required.")
        res = self._con.post(url, params)
        if "status" in res:
            if res["status"] == "success":
                return True
            return res["status"]
        return res
