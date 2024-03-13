from __future__ import annotations
import json
from arcgis.gis import Layer, _GISResource, Item
from arcgis.geoprocessing import import_toolbox
from arcgis.geometry import Geometry
from arcgis.auth.tools import LazyLoader

_services = LazyLoader("arcgis.gis.server.admin._services")


class SceneLayerManager(_GISResource):
    """
    The ``SceneLayerManager`` class allows administration (if access permits) of ArcGIS Online hosted scene layers.
    A :class:`~arcgis.mapping.SceneLayerManager` offers access to map and layer content.
    """

    def __init__(self, url, gis=None, scene_lyr=None):
        if url.split("/")[-1].isdigit():
            url = url.replace(f"/{url.split('/')[-1]}", "")
        super(SceneLayerManager, self).__init__(url, gis)
        self._sl = scene_lyr
        # Scene Layers published from Scene Layer Package are read only.
        if "layers" in self.properties:
            self._source_type = (
                "Feature Service"
                if "updateEnabled" in self.properties.layers[0]
                else "Scene Layer Package"
            )
        else:
            # No layers are present so we will not have cache
            self._source_type = "Scene Layer Package"

    # ----------------------------------------------------------------------
    def refresh(self):
        """
        The ``refresh`` operation refreshes a service, which clears the web
        server cache for the service.
        """
        if self._source_type == "Scene Layer Package":
            url = self._url + "SceneServer/refresh"
            params = {"f": "json"}

            res = self._con.post(url, params)

            super(SceneLayerManager, self)._refresh()

            return res
        return None

    # ----------------------------------------------------------------------
    def swap(self, target_service_name: str):
        """
        The swap operation replaces the current service cache with an existing one.

        .. note::
            The ``swap`` operation is for ArcGIS Online only.

        ====================        ====================================================
        **Parameter**                **Description**
        --------------------        ----------------------------------------------------
        target_service_name         Required string. Name of service you want to swap with.
        ====================        ====================================================

        :returns: dictionary indicating success or error

        """
        if self._source_type == "Scene Layer Package":
            url = self._url + "/swap"
            params = {"f": "json", "targetServiceName": target_service_name}
            return self._con.post(url, params)
        return None

    # ----------------------------------------------------------------------
    def jobs(self):
        """
        The tile service job summary (jobs) resource represents a
        summary of all jobs associated with a vector tile service.
        Each job contains a jobid that corresponds to the specific
        jobid run and redirects you to the Job Statistics page.

        """
        if self._source_type == "Scene Layer Package":
            url = self._url + "/jobs"
            params = {"f": "json"}
            return self._con.get(url, params)
        return None

    # ----------------------------------------------------------------------
    def cancel_job(self, job_id: str):
        """
        The ``cancel_job`` operation supports cancelling a job while update
        tiles is running from a hosted feature service. The result of this
        operation is a response indicating success or failure with error
        code and description.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        job_id              Required String. The job id to cancel.
        ===============     ====================================================

        """
        if self._source_type == "Scene Layer Package":
            url = self._url + "/jobs/%s/cancel" % job_id
            params = {"f": "json"}
            return self._con.post(url, params)
        return None

    # ----------------------------------------------------------------------
    def job_statistics(self, job_id: str):
        """
        Returns the job statistics for the given jobId

        """
        if self._source_type == "Scene Layer Package":
            url = self._url + "/jobs/%s" % job_id
            params = {"f": "json"}
            return self._con.post(url, params)
        return None

    # -----------------------------------------------------------------------
    def rerun_job(self, job_id: str, code: str):
        """
        The ``rerun_job`` operation supports re-running a canceled job from a
        hosted map service. The result of this operation is a response
        indicating success or failure with error code and description.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        code                Required string, parameter used to re-run a given
                            jobs with a specific error
                            code: ``ALL | ERROR | CANCELED``
        ---------------     ----------------------------------------------------
        job_id              Required string, job to reprocess
        ===============     ====================================================

        :return:
           A boolean or dictionary
        """
        if self._source_type == "Scene Layer Package":
            url = self._url + "/jobs/%s/rerun" % job_id
            params = {"f": "json", "rerun": code}
            return self._con.post(url, params)
        return None

    # ----------------------------------------------------------------------
    def import_package(self, item: str | Item):
        """
        The ``import`` method imports from an :class:`~arcgis.gis.Item` object.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        item                Required ItemId or :class:`~arcgis.gis.Item` object. The TPK file's item id.
                            This TPK file contains to-be-extracted bundle files
                            which are then merged into an existing cache service.
        ===============     ====================================================

        :return:
            A dictionary

        """
        if self._source_type == "Scene Layer Package":
            params = {
                "f": "json",
                "sourceItemId": None,
            }
            if isinstance(item, str):
                params["sourceItemId"] = item
            elif isinstance(item, Item):
                params["sourceItemId"] = item.itemid
            else:
                raise ValueError("The `item` must be a string or Item")
            url = self._url + "/import"
            res = self._con.post(url, params)
            return res
        return None

    # ----------------------------------------------------------------------
    def update(self):
        """
        The ``update`` method starts update generation for ArcGIS Online. It updates
        the underlying source dataset for the service, essentially refreshing the
        underlying package data.

        :return:
           Dictionary.
        """
        if self._gis._portal.is_arcgisonline:
            url = "%s/update" % self._url
            params = {"f": "json"}
            return self._con.post(url, params)
        return None

    # ----------------------------------------------------------------------
    def edit(self, item: str | Item):
        """
        The ``edit`` method edits from an :class:`~arcgis.gis.Item` object.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        item                Required ItemId or :class:`~arcgis.gis.Item` object. The TPK file's item id.
                            This TPK file contains to-be-extracted bundle files
                            which are then merged into an existing cache service.
        ===============     ====================================================

        :return:
            A dictionary

        """
        if self._source_type == "Scene Layer Package":
            params = {
                "f": "json",
                "sourceItemId": None,
            }
            if isinstance(item, str):
                params["sourceItemId"] = item
            elif isinstance(item, Item):
                params["sourceItemId"] = item.itemid
            else:
                raise ValueError("The `item` must be a string or Item")
            url = self._url + "/edit"
            res = self._con.post(url, params)
            return res
        return None

    # ----------------------------------------------------------------------
    def rebuild_cache(self, layers: int | list[int]):
        """
        The rebuild_cache operation update the scene layer cache to reflect
        any changes made to the feature layer used to publish this scene layer.
        The results of the operation is a response indicating success, which
        redirects you to the Job Statistics page, or failure.

        =====================       ====================================================
        **Parameter**                **Description**
        ---------------------       ----------------------------------------------------
        layers                      Required int or list of int. Comma seperated values indicating
                                    the id of the layers to rebuild in the cache.

                                    Ex: [0,1,2]
        =====================       ====================================================
        """
        if self._source_type == "Feature Service":
            url = self._url + "/rebuildCache"
            params = {"f": "json", "layers": layers}
            return self._con.post(url, params)
        return None

    # ----------------------------------------------------------------------
    def update_cache(self, layers: int | list[int]):
        """
        Update Cache is a "light rebuild" where attributes and geometries of
        the layers selected are updated and can be used for change tracking on
        the feature layer to only update nodes with dirty tiles.
        The results of the operation is a response indicating success, which
        redirects you to the Job Statistics page, or failure.

        =====================       ====================================================
        **Parameter**                **Description**
        ---------------------       ----------------------------------------------------
        layers                      Required int or list of int. Comma seperated values indicating
                                    the id of the layers to update in the cache.

                                    Ex: [0,1,2]
        =====================       ====================================================
        """
        if self._source_type == "Feature Service":
            url = self._url + "/updateCache"
            params = {"f": "json", "layers": layers}
            return self._con.post(url, params)
        return None

    # ----------------------------------------------------------------------
    def update_attribute(self, layers: int | list[int]):
        """
        Update atrribute is a "light rebuild" where attributes of
        the layers selected are updated and can be used for change tracking.
        The results of the operation is a response indicating success, which
        redirects you to the Job Statistics page, or failure.

        =====================       ====================================================
        **Parameter**                **Description**
        ---------------------       ----------------------------------------------------
        layers                      Required int or list of int. Comma seperated values indicating
                                    the id of the layers to update in the cache.

                                    Ex: [0,1,2]
        =====================       ====================================================
        """
        if self._source_type == "Feature Service":
            url = self._url + "/updateAttribute"
            params = {"f": "json", "layers": layers}
            return self._con.post(url, params)
        return None


###########################################################################
class EnterpriseSceneLayerManager(_GISResource):
    """
    The ``EnterpriseSceneLayerManager`` class allows administration (if access permits) of ArcGIS Enterprise hosted scene layers.
    A :class:`~arcgis.mapping.SceneLayer` offers access to layer content.

    .. note:: Url must be admin url such as: ``https://services.myserver.com/arcgis/rest/admin/services/serviceName/SceneServer/``
    """

    _gptbx = None

    def __init__(self, url, gis=None, scene_lyr=None):
        if url.split("/")[-1].isdigit():
            url = url.replace(f"/{url.split('/')[-1]}", "")
        super(EnterpriseSceneLayerManager, self).__init__(url, gis)
        self._sl = scene_lyr

    # ----------------------------------------------------------------------
    def edit(self, service_dictionairy: dict):
        """
        To edit a service, you need to submit the complete JSON
        representation of the service, which includes the updates to the
        service properties. Editing a service causes the service to be
        restarted with updated properties.

        ===================     ====================================================================
        **Parameter**            **Description**
        -------------------     --------------------------------------------------------------------
        service_dictionairy     Required dict. The service JSON as a dictionary.
        ===================     ====================================================================


        :return: boolean
        """
        sl_service = _services.Service(self.url, self._gis)
        return sl_service.edit(service_dictionairy)

    # ----------------------------------------------------------------------
    def start(self):
        """starts the specific service"""
        sl_service = _services.Service(self.url, self._gis)
        return sl_service.start()

    # ----------------------------------------------------------------------
    def stop(self):
        """stops the specific service"""
        sl_service = _services.Service(self.url, self._gis)
        return sl_service.stop()

    # ----------------------------------------------------------------------
    def change_provider(self, provider: str):
        """
        Allows for the switching of the service provide and how it is hosted on the ArcGIS Server instance.

        Values:

        + 'ArcObjects' means the service is running under the ArcMap runtime i.e. published from ArcMap
        + 'ArcObjects11': means the service is running under the ArcGIS Pro runtime i.e. published from ArcGIS Pro
        + 'DMaps': means the service is running in the shared instance pool (and thus running under the ArcGIS Pro provider runtime)

        :return: Boolean

        """
        sl_service = _services.Service(self.url, self._gis)
        return sl_service.change_provider(provider)

    # ----------------------------------------------------------------------
    def delete(self):
        """deletes a service from arcgis server"""
        sl_service = _services.Service(self.url, self._gis)
        return sl_service.delete()

    # ----------------------------------------------------------------------
    @property
    def _tbx(self):
        """gets the toolbox"""
        if self._gptbx is None:
            self._gptbx = import_toolbox(
                url_or_item=self._gis.hosting_servers[0].url
                + "/System/SceneCachingControllers/GPServer",
                gis=self._gis,
            )
            self._gptbx._is_fa = True
        return self._gptbx

    # ----------------------------------------------------------------------
    def rebuild_cache(
        self,
        layer: list[int] | None = None,
        extent: dict | None = None,
        area_of_interest: dict | None = None,
    ):
        """
        The rebuild_cache operation update the scene layer cache to reflect
        any changes made to the feature layer used to publish this scene layer.
        The results of the operation is the url to the scene service once it is
        done rebuilding.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        layer                               Optional list of integers. The list of layers to cook.
        -------------------------------     --------------------------------------------------------------------
        extent                              Optional dict. The updated extent to be used. If nothing is specified,
                                            the default extent is used.
        -------------------------------     --------------------------------------------------------------------
        area_of_interest                    Optional dict representing a feature. Specify the updated area
                                            of interest.

                                            Syntax:
                                                {
                                                    "displayFieldName": "",
                                                    "geometryType": "esriGeometryPolygon",
                                                    "spatialReference": {
                                                    "wkid": 54051,
                                                    "latestWkid": 54051
                                                    },
                                                    "fields": [
                                                    {
                                                    "name": "OID",
                                                    "type": "esriFieldTypeOID",
                                                    "alias": "OID"
                                                    },
                                                    {
                                                    "name": "updateGeom_Length",
                                                    "type": "esriFieldTypeDouble",
                                                    "alias": "updateGeom_Length"
                                                    },
                                                    {
                                                    "name": "updateGeom_Area",
                                                    "type": "esriFieldTypeDouble",
                                                    "alias": "updateGeom_Area"
                                                    }
                                                    ],
                                                    "features": [],
                                                    "exceededTransferLimit": False
                                                }
        ===============================     ====================================================================

        :return: If successful, the url to the scene service

        """
        if layer is None:
            layer = {}
        elif layer is not None:
            layer = {layer}
        if extent is None:
            extent = "DEFAULT"
        if area_of_interest is None:
            return self._tbx.manage_scene_cache(
                service_url=self._sl.url,
                num_of_caching_service_instances=2,
                layer=layer,
                update_mode="RECREATE_ALL_NODES",
                update_extent=extent,
            )
        else:
            return self._tbx.manage_scene_cache(
                service_url=self._sl.url,
                num_of_caching_service_instances=2,
                layer=layer,
                update_mode="RECREATE_ALL_NODES",
                update_extent=extent,
                area_of_interest=area_of_interest,
            )

    # ----------------------------------------------------------------------
    def update_cache(
        self,
        layer: list[int] | None = None,
        extent: dict | None = None,
        area_of_interest: dict | None = None,
    ):
        """
        Update Cache is a "light rebuild" where attributes and geometries of
        the layers selected are updated and can be used for change tracking on
        the feature layer to only update nodes with dirty tiles,.
        The results of the operation is the url to the scene service once it is
        done updating.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        layer                               Optional list of integers. The list of layers to cook.
        -------------------------------     --------------------------------------------------------------------
        extent                              Optional dict. The updated extent to be used. If nothing is specified,
                                            the default extent is used.
        -------------------------------     --------------------------------------------------------------------
        area_of_interest                    Optional dict representing a feature. Specify the updated area
                                            of interest.

                                            Syntax:
                                                {
                                                    "displayFieldName": "",
                                                    "geometryType": "esriGeometryPolygon",
                                                    "spatialReference": {
                                                    "wkid": 54051,
                                                    "latestWkid": 54051
                                                    },
                                                    "fields": [
                                                    {
                                                    "name": "OID",
                                                    "type": "esriFieldTypeOID",
                                                    "alias": "OID"
                                                    },
                                                    {
                                                    "name": "updateGeom_Length",
                                                    "type": "esriFieldTypeDouble",
                                                    "alias": "updateGeom_Length"
                                                    },
                                                    {
                                                    "name": "updateGeom_Area",
                                                    "type": "esriFieldTypeDouble",
                                                    "alias": "updateGeom_Area"
                                                    }
                                                    ],
                                                    "features": [],
                                                    "exceededTransferLimit": False
                                                }
        ===============================     ====================================================================

        :return: If successful, the url to the scene service

        """
        if layer is None:
            layer = {}
        elif layer is not None:
            layer = {layer}
        if extent is None:
            extent = "DEFAULT"
        if area_of_interest is None:
            return self._tbx.manage_scene_cache(
                service_url=self._sl.url,
                num_of_caching_service_instances=2,
                layer=layer,
                update_mode="PARTIAL_UPDATE_NODES",
                update_extent=extent,
            )
        else:
            return self._tbx.manage_scene_cache(
                service_url=self._sl.url,
                num_of_caching_service_instances=2,
                layer=layer,
                update_mode="PARTIAL_UPDATE_NODES",
                update_extent=extent,
                area_of_interest=area_of_interest,
            )

    # ----------------------------------------------------------------------
    def update_attribute(
        self,
        layer: list[int] | None = None,
        extent: dict | None = None,
        area_of_interest: dict | None = None,
    ):
        """
        Update atrribute is a "light rebuild" where attributes of
        the layers selected are updated and can be used for change tracking.
        The results of the operation is the url to the scene service once it is
        done updating.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        layer                               Optional list of integers. The list of layers to cook.
        -------------------------------     --------------------------------------------------------------------
        extent                              Optional dict. The updated extent to be used. If nothing is specified,
                                            the default extent is used.
        -------------------------------     --------------------------------------------------------------------
        area_of_interest                    Optional dict representing a feature. Specify the updated area
                                            of interest.

                                            Syntax:
                                                {
                                                    "displayFieldName": "",
                                                    "geometryType": "esriGeometryPolygon",
                                                    "spatialReference": {
                                                    "wkid": 54051,
                                                    "latestWkid": 54051
                                                    },
                                                    "fields": [
                                                    {
                                                    "name": "OID",
                                                    "type": "esriFieldTypeOID",
                                                    "alias": "OID"
                                                    },
                                                    {
                                                    "name": "updateGeom_Length",
                                                    "type": "esriFieldTypeDouble",
                                                    "alias": "updateGeom_Length"
                                                    },
                                                    {
                                                    "name": "updateGeom_Area",
                                                    "type": "esriFieldTypeDouble",
                                                    "alias": "updateGeom_Area"
                                                    }
                                                    ],
                                                    "features": [],
                                                    "exceededTransferLimit": false
                                                }
        ===============================     ====================================================================

        :return: If successful, the url to the scene service
        """
        if layer is None:
            layer = {}
        elif layer is not None:
            layer = {layer}
        if extent is None:
            extent = "DEFAULT"
        if area_of_interest is None:
            return self._tbx.manage_scene_cache(
                service_url=self._sl.url,
                num_of_caching_service_instances=2,
                layer=layer,
                update_mode="PARTIAL_UPDATE_ATTRIBUTES",
                update_extent=extent,
            )
        else:
            return self._tbx.manage_scene_cache(
                service_url=self._sl.url,
                num_of_caching_service_instances=2,
                layer=layer,
                update_mode="PARTIAL_UPDATE_ATTRIBUTES",
                update_extent=extent,
                area_of_interest=area_of_interest,
            )


###########################################################################
class Object3DLayer(Layer):
    """
    The ``Object3DLayer`` represents a Web scene 3D Object layer.

    .. note::
        Web scene layers are cached web layers that are optimized for displaying a large amount of 2D and 3D features.
        See the :class:`~arcgis.mapping.SceneLayer` class for more information.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`  object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.Point3DLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """

    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(Object3DLayer, self).__init__(url, gis)
        self._admin = None

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": "SceneLayer", "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": "SceneLayer", "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def manager(self):
        """
        The ``manager`` property returns an instance of :class:`~arcgis.mapping.SceneLayerManager` class
        or :class:`~arcgis.mapping.EnterpriseSceneLayerManager` class
        which provides methods and properties for administering this service.
        """
        if self._admin is None:
            if self._gis._portal.is_arcgisonline:
                rd = {"/rest/services/": "/rest/admin/services/"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = SceneLayerManager(adminURL, self._gis, self)
            else:
                rd = {"/rest/": "/admin/", "/SceneServer": ".SceneServer"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = EnterpriseSceneLayerManager(adminURL, self._gis, self)
        return self._admin

    # ----------------------------------------------------------------------
    def _str_replace(self, mystring, rd):
        """Replaces a value based on a key/value pair where the
        key is the text to replace and the value is the new value.

        The find/replace is case insensitive.

        """
        import re

        patternDict = {}
        for key, value in rd.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            patternDict[value] = pattern
        for key in patternDict:
            regex_obj = patternDict[key]
            mystring = regex_obj.sub(key, mystring)
        return mystring


###########################################################################
class IntegratedMeshLayer(Layer):
    """
    The ``IntegratedMeshLayer`` class represents a Web scene Integrated Mesh layer.

    .. note::
        Web scene layers are cached web layers that are optimized for displaying a large amount of 2D and 3D features.
        See the :class:`~arcgis.mapping.SceneLayer` class for more information.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`  object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.Point3DLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """

    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(IntegratedMeshLayer, self).__init__(url, gis)
        self._admin = None

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": "IntegratedMeshLayer", "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": "IntegratedMeshLayer", "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def manager(self):
        """
        The ``manager`` property returns an instance of :class:`~arcgis.mapping.SceneLayerManager` class
        or :class:`~arcgis.mapping.EnterpriseSceneLayerManager` class
        which provides methods and properties for administering this service.
        """
        if self._admin is None:
            if self._gis._portal.is_arcgisonline:
                rd = {"/rest/services/": "/rest/admin/services/"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = SceneLayerManager(adminURL, self._gis, self)
            else:
                rd = {"/rest/": "/admin/", "/SceneServer": ".SceneServer"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = EnterpriseSceneLayerManager(adminURL, self._gis, self)
        return self._admin

    # ----------------------------------------------------------------------
    def _str_replace(self, mystring, rd):
        """Replaces a value based on a key/value pair where the
        key is the text to replace and the value is the new value.

        The find/replace is case insensitive.

        """
        import re

        patternDict = {}
        for key, value in rd.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            patternDict[value] = pattern
        for key in patternDict:
            regex_obj = patternDict[key]
            mystring = regex_obj.sub(key, mystring)
        return mystring


###########################################################################


class VoxelLayer(Layer):
    """
    The ``VoxelLayer`` class represents a Web Scene Voxel layer.

    .. note::
        Web scene layers are cached web layers that are optimized for displaying
        a large amount of 2D and 3D features. See the
        :class:`~arcgis.mapping.SceneLayer` class for more information.

    ==================     =============================================================
    **Parameter**           **Description**
    ------------------     -------------------------------------------------------------
    url                    Required string, specify the url ending in ``/SceneServer/``
    ------------------     -------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS` object. If not specified,
                           the active GIS connection is used.
    ==================     =============================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.VoxelLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """

    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(VoxelLayer, self).__init__(url, gis)
        self._admin = None

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": "VoxelLayer", "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": "VoxelLayer", "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def manager(self):
        """
        The ``manager`` property returns an instance of
        :class:`~arcgis.mapping.SceneLayerManager` class
        or :class:`~arcgis.mapping.EnterpriseSceneLayerManager` class
        which provides methods and properties for administering this service.
        """
        if self._admin is None:
            if self._gis._portal.is_arcgisonline:
                rd = {"/rest/services/": "/rest/admin/services/"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = SceneLayerManager(adminURL, self._gis, self)
            else:
                rd = {"/rest/": "/admin/", "/SceneServer": ".SceneServer"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = EnterpriseSceneLayerManager(adminURL, self._gis, self)
        return self._admin

    # ----------------------------------------------------------------------
    def _str_replace(self, mystring, rd):
        """Replaces a value based on a key/value pair where the
        key is the text to replace and the value is the new value.

        The find/replace is case insensitive.

        """
        import re

        patternDict = {}
        for key, value in rd.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            patternDict[value] = pattern
        for key in patternDict:
            regex_obj = patternDict[key]
            mystring = regex_obj.sub(key, mystring)
        return mystring


###########################################################################
class Point3DLayer(Layer):
    """
    The ``Point3DLayer`` class represents a Web scene 3D Point layer.

    .. note::
        Web scene layers are cached web layers that are optimized for displaying a large amount of 2D and 3D features.
        See the :class:`~arcgis.mapping.SceneLayer` class for more information.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`  object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.Point3DLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """

    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(Point3DLayer, self).__init__(url, gis)
        self._admin = None

    # ----------------------------------------------------------------------
    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": "SceneLayer", "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": "SceneLayer", "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def manager(self):
        """
        The ``manager`` property returns an instance of :class:`~arcgis.mapping.SceneLayerManager` class
        or :class:`~arcgis.mapping.EnterpriseSceneLayerManager` class
        which provides methods and properties for administering this service.
        """
        if self._admin is None:
            if self._gis._portal.is_arcgisonline:
                rd = {"/rest/services/": "/rest/admin/services/"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = SceneLayerManager(adminURL, self._gis, self)
            else:
                rd = {"/rest/": "/admin/", "/SceneServer": ".SceneServer"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = EnterpriseSceneLayerManager(adminURL, self._gis, self)
        return self._admin

    # ----------------------------------------------------------------------
    def _str_replace(self, mystring, rd):
        """Replaces a value based on a key/value pair where the
        key is the text to replace and the value is the new value.

        The find/replace is case insensitive.

        """
        import re

        patternDict = {}
        for key, value in rd.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            patternDict[value] = pattern
        for key in patternDict:
            regex_obj = patternDict[key]
            mystring = regex_obj.sub(key, mystring)
        return mystring


###########################################################################
class PointCloudLayer(Layer):
    """
    The ``PointCloudLayer`` class represents a Web scene Point Cloud layer.

    .. note::
        Point Cloud layers are cached web layers that are optimized for displaying a large amount of 2D and 3D features.
        See the :class:`~arcgis.mapping.SceneLayer` class for more information.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`  object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.PointCloudLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """

    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(PointCloudLayer, self).__init__(url, gis)
        self._admin = None

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": "PointCloudLayer", "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": "PointCloudLayer", "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def manager(self):
        """
        The ``manager`` property returns an instance of :class:`~arcgis.mapping.SceneLayerManager` class
        or :class:`~arcgis.mapping.EnterpriseSceneLayerManager` class
        which provides methods and properties for administering this service.
        """
        if self._admin is None:
            if self._gis._portal.is_arcgisonline:
                rd = {"/rest/services/": "/rest/admin/services/"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = SceneLayerManager(adminURL, self._gis, self)
            else:
                rd = {"/rest/": "/admin/", "/SceneServer": ".SceneServer"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = EnterpriseSceneLayerManager(adminURL, self._gis, self)
        return self._admin

    # ----------------------------------------------------------------------
    def _str_replace(self, mystring, rd):
        """Replaces a value based on a key/value pair where the
        key is the text to replace and the value is the new value.

        The find/replace is case insensitive.

        """
        import re

        patternDict = {}
        for key, value in rd.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            patternDict[value] = pattern
        for key in patternDict:
            regex_obj = patternDict[key]
            mystring = regex_obj.sub(key, mystring)
        return mystring


###########################################################################
class BuildingLayer(Layer):
    """
    The ``BuildingLayer`` class represents a Web building layer.

    .. note::
        Web scene layers are cached web layers that are optimized for displaying a large amount of 2D and 3D features.
        See the :class:`~arcgis.mapping.SceneLayer` class for more information.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`  object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.BuildingLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """

    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(BuildingLayer, self).__init__(url, gis)
        self._admin = None

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": "BuildingSceneLayer", "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": "BuildingSceneLayer", "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def manager(self):
        """
        The ``manager`` property returns an instance of :class:`~arcgis.mapping.SceneLayerManager` class
        or :class:`~arcgis.mapping.EnterpriseSceneLayerManager` class
        which provides methods and properties for administering this service.
        """
        if self._admin is None:
            if self._gis._portal.is_arcgisonline:
                rd = {"/rest/services/": "/rest/admin/services/"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = SceneLayerManager(adminURL, self._gis, self)
            else:
                rd = {"/rest/": "/admin/", "/SceneServer": ".SceneServer"}
                adminURL = self._str_replace(self._url, rd)
                if adminURL.split("/")[-1].isdigit():
                    adminURL = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
                self._admin = EnterpriseSceneLayerManager(adminURL, self._gis, self)
        return self._admin

    # ----------------------------------------------------------------------
    def _str_replace(self, mystring, rd):
        """Replaces a value based on a key/value pair where the
        key is the text to replace and the value is the new value.

        The find/replace is case-insensitive.

        """
        import re

        patternDict = {}
        for key, value in rd.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            patternDict[value] = pattern
        for key in patternDict:
            regex_obj = patternDict[key]
            mystring = regex_obj.sub(key, mystring)
        return mystring


###########################################################################
class _SceneLayerFactory(type):
    """
    Factory that generates the Scene Layers

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`  object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.PointCloudLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """

    def __call__(cls, url, gis=None):
        lyr = Layer(url=url, gis=gis)
        props = lyr.properties
        if "sublayers" in props:
            return BuildingLayer(url=url, gis=gis)
        elif "layerType" in props:
            lt = props.layerType
        else:
            lt = props.layers[0].layerType
        if str(lt).lower() == "pointcloud":
            return PointCloudLayer(url=url, gis=gis)
        elif str(lt).lower() == "point":
            return Point3DLayer(url=url, gis=gis)
        elif str(lt).lower() == "3dobject":
            return Object3DLayer(url=url, gis=gis)
        elif str(lt).lower() == "building":
            return BuildingLayer(url=url, gis=gis)
        elif str(lt).lower() == "IntegratedMesh".lower():
            return IntegratedMeshLayer(url=url, gis=gis)
        elif str(lt).lower() == "voxel":
            return VoxelLayer(url=url, gis=gis)
        return lyr


###########################################################################
class SceneLayer(Layer, metaclass=_SceneLayerFactory):
    """
    The ``SceneLayer`` class represents a Web scene layer.

    .. note::
        Web scene layers are cached web layers that are optimized for displaying a large amount of 2D and 3D features.

    .. note::
        Web scene layers can be used to represent 3D points, point clouds, 3D objects and
        integrated mesh layers.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`  object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.PointCloudLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """

    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(SceneLayer, self).__init__(url, gis)
