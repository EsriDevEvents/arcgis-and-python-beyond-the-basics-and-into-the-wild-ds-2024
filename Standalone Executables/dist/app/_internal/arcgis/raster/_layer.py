from __future__ import annotations
import json
from typing import Any, Optional, Union
import uuid
import datetime
import tempfile
import numbers
import warnings
import copy
import requests as _requests
from six import b

from arcgis._impl.common._utils import _date_handler
from arcgis.gis import GIS, Layer, Item
from arcgis.geometry import (
    Geometry,
    Envelope,
    SpatialReference,
    Point,
    Polygon,
    Polyline,
    MultiPoint,
)
from arcgis.features import FeatureSet
from arcgis.gis import _GISResource
import logging
import arcgis as _arcgis
import base64
from collections import defaultdict
from ._RasterInfo import RasterInfo

try:
    import numpy as np
except:
    pass

_LOGGER = logging.getLogger(__name__)

try:
    from arcgis.auth.tools import LazyLoader

    arcpy = LazyLoader("arcpy")
except:
    pass


def _get_rendering_service_layer(layer):
    if layer._rendering_service_layer:
        return layer._rendering_service_layer
    else:
        from .functions.utility import _generate_layer_token

        token = _generate_layer_token(layer, layer.url)
        newlyr = ImageryLayer(
            {"input_raster": layer.url + "?token=" + token}, layer._gis
        )
        layer._rendering_service_layer = newlyr
        return newlyr


def _find_and_replace_mosaic_rule(fnarg_ra, mosaic_rule, url):
    for key, value in fnarg_ra.items():
        if key == "Raster" and isinstance(value, dict) and not (value.keys() & {"url"}):
            return _find_and_replace_mosaic_rule(
                value["rasterFunctionArguments"], fnarg_ra
            )
        if key == "Rasters":
            if isinstance(value, list):
                for each_element in value:
                    return _find_and_replace_mosaic_rule(
                        each_element["rasterFunctionArguments"], fnarg_ra
                    )
        elif key == "Raster" or key == "Rasters":
            if isinstance(value, dict):
                if value.keys() & {"url"}:
                    value["mosaicRule"] = mosaic_rule
            else:
                fnarg_ra[key] = {}
                fnarg_ra[key]["url"] = url
                fnarg_ra[key]["mosaicRule"] = mosaic_rule

    return fnarg_ra


###########################################################################
class ImageryLayerCacheManager(_GISResource):
    """
    The ``ImageryLayerCacheManager`` class allows for administration of ArcGIS Online hosted image layers.
    """

    def __init__(self, url, gis=None, img_lyr=None):
        super(ImageryLayerCacheManager, self).__init__(url, gis)
        self._img_lyr = img_lyr
        self._gis = gis
        self._url = url

    # ----------------------------------------------------------------------
    def refresh(self):
        """
        The ``refresh`` operation refreshes a service, which clears the web
        server cache for the service.

        """
        url = self._url + "/refresh"
        params = {"f": "json"}
        res = self._con.post(url, params, timeout=None)
        if "success" in res:
            return res["success"]
        return res

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
        job_id              Required string, the job id to cancel.
        ===============     ====================================================

        :return:
            A boolean indicating cancellation (True), or not (False)
        """
        url = self._url + "/jobs/%s/cancel" % job_id
        params = {"f": "json"}
        return self._con.post(url, params, timeout=None)

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
        url = self._url + "/swap"
        params = {"f": "json", "targetServiceName": target_service_name}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def jobs(self):
        """
        Get a list of all the jobs on the tile server

        :return:
            A list
        """
        url = self._url + "/jobs"
        params = {"f": "json"}
        res = self._con.post(url, params, timeout=None)
        if "jobs" in res:
            return res["jobs"]
        return res

    # ----------------------------------------------------------------------
    def job_status(self, job_id: str):
        """
        The ``job_status`` method retrieves the current Job Status.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        job_id                required String. The unique identifier of the job in question.
        =================     ====================================================================


        :return: A dictionary
        """
        url = self._url + "/jobs/%s/status" % job_id
        params = {"f": "json"}
        res = self._con.get(url, params)
        return res

    # ----------------------------------------------------------------------
    def job_statistics(self, job_id: str):
        """
        The ``job_statistics`` method retrieves the job statistics for the given ``job_id``.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        job_id                required String. The unique identifier of the job in question.
        =================     ====================================================================


        :return: A dictionary

        """
        url = self._url + "/jobs/%s" % job_id
        params = {"f": "json"}
        return self._con.post(url, params, timeout=None)

    # ----------------------------------------------------------------------
    def import_tiles(
        self,
        item: Union[str, Item],
        levels: Optional[Union[str, list[int]]] = None,
        extent: Optional[Union[str, dict[str, int]]] = None,
        merge: bool = False,
        replace: bool = False,
    ):
        """
        The ``import_tiles`` method imports cache from a new ``ImageLayer Tile Package``.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        item                Required ItemId or :class:`~arcgis.gis.Item`. The TPK file's item id.
                            This TPK file contains to-be-extracted bundle files
                            which are then merged into an existing cache service.
        ---------------     ----------------------------------------------------
        levels              Optional String / List of integers, The level of details
                            to update. Example: "1,2,10,20" or [1,2,10,20]
        ---------------     ----------------------------------------------------
        extent              Optional String / Dict. The area to update as Xmin, YMin, XMax, YMax
                            example: "-100,-50,200,500" or
                            {'xmin':100, 'ymin':200, 'xmax':105, 'ymax':205}
        ---------------     ----------------------------------------------------
        merge               Optional Boolean. Default is false and applicable to
                            compact cache storage format. It controls whether
                            the bundle files from the TPK file are merged with
                            the one in the existing cached service. Otherwise,
                            the bundle files are overwritten.
        ---------------     ----------------------------------------------------
        replace             Optional Boolean. Default is false, applicable to
                            compact cache storage format and used when
                            merge=true. It controls whether the new tiles will
                            replace the existing ones when merging bundles.
        ===============     ====================================================

        :return: A dictionary

        .. code-block:: python

            from arcgis.mapping import ImageryLayer
            from arcgis.gis import GIS

            # Example Usage
            gis = GIS(url, username, password)
            img_lyr = ImageryLayer("<url>", gis)
            img_lyr_cache_manager = img_lyr.cache_manager
            imported_tiles = img_lyr_cache_manager.import_tiles(item = item1,
                                                                levels = "11-20",
                                                                extent = {
                                                                            "xmin":6224324.092137296,
                                                                            "ymin":487347.5253569535,
                                                                            "xmax":11473407.698535524,
                                                                            "ymax":4239488.369818687
                                                                         },
                                                                merge = True
                                                                )
        """
        params = {
            "f": "json",
            "sourceItemId": None,
            "extent": extent,
            "levels": levels,
            "mergeBundle": merge,
            "replaceTiles": replace,
        }
        if isinstance(item, str):
            params["sourceItemId"] = item
        elif isinstance(item, Item):
            params["sourceItemId"] = item.itemid
        else:
            raise ValueError("The `item` must be a string or Item")
        url = self._url + "/importTiles"
        res = self._con.post(url, params, timeout=None)
        return res

    # ----------------------------------------------------------------------
    def update_tiles(
        self,
        levels: Optional[Union[str, list[int]]] = None,
        extent: Optional[Union[str, dict[str, int]]] = None,
        merge: bool = False,
        replace: bool = False,
    ):
        """
        The ``update_tiles`` method starts tile generation for ArcGIS Online.  The levels of detail
        and the extent are needed to determine the area where tiles need
        to be rebuilt.

        .. note::
            The ``update_tiles`` operation is for ArcGIS Online only.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        levels              Optional String / List of integers, The level of details
                            to update. Example: "1,2,10,20" or [1,2,10,20]
        ---------------     ----------------------------------------------------
        extent              Optional String / Dict. The area to update as Xmin, YMin, XMax, YMax
                            example: "-100,-50,200,500" or
                            {'xmin':100, 'ymin':200, 'xmax':105, 'ymax':205}
        ---------------     ----------------------------------------------------
        merge               Optional Boolean. Default is `False`. When true the updated
                            cache is merged with the existing cache.
        ---------------     ----------------------------------------------------
        replace             Optional Boolean.  The default is False.  The updated
                            tiles will remove the existing tiles.
        ===============     ====================================================

        :return:
           JSON dictionary that indicates 'success' or 'error'. If the product is not ArcGIS Online tile service, the
           result will be None.

        .. code-block:: python

            from arcgis.mapping import ImageryLayer
            from arcgis.gis import GIS

            # Example Usage
            gis = GIS(url, username, password)
            img_lyr = ImageryLayer("<url>", gis)
            img_lyr_cache_manager = img_lyr.cache_manager
            updated_tiles = img_lyr_cache_manager.update_tiles(levels = "11-20",
                                                               extent = {
                                                                            "xmin":6224324.092137296,
                                                                            "ymin":487347.5253569535,
                                                                            "xmax":11473407.698535524,
                                                                            "ymax":4239488.369818687
                                                                        },
                                                               merge = True
                                                               )
        """
        if self._gis._portal.is_arcgisonline:
            url = "%s/updateTiles" % self._url
            params = {
                "f": "json",
                "mergeBundle": json.dumps(merge),
                "replaceTiles": json.dumps(replace),
            }
            if levels:
                if isinstance(levels, list):
                    levels = ",".join(str(e) for e in levels)
                params["levels"] = levels
            if extent:
                if isinstance(extent, dict):
                    extent2 = "{},{},{},{}".format(
                        extent["xmin"],
                        extent["ymin"],
                        extent["xmax"],
                        extent["ymax"],
                    )
                    extent = extent2
                params["extent"] = extent
            return self._con.post(url, params, timeout=None)
        return None

    # ----------------------------------------------------------------------
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
                            code: ``ALL`` | ``ERROR`` | ``CANCELED``
        ---------------     ----------------------------------------------------
        job_id              Required string, job to reprocess
        ===============     ====================================================

        :return:
           JSON dictionary indicating 'success' or 'error'
        """
        url = self._url + "/jobs/%s/rerun" % job_id
        params = {"f": "json", "rerun": code}
        return self._con.post(url, params, timeout=None)

    # ----------------------------------------------------------------------
    def edit_tile_service(
        self,
        service_definition: Optional[str] = None,
        min_scale: Optional[int] = None,
        max_scale: Optional[int] = None,
        source_item_id: Optional[str] = None,
        export_tiles_allowed: bool = False,
        max_export_tile_count: int = 100000,
    ):
        """
        The ``edit_tile_service`` operation updates a Tile Service's properties.

        =====================     ====================================================
        **Parameter**              **Description**
        ---------------------     ----------------------------------------------------
        service_definition        Optional String.
                                  Updates a service definition
        ---------------------     ----------------------------------------------------
        min_scale                 Optional integer
                                  Sets the services minimum scale for caching
        ---------------------     ----------------------------------------------------
        max_scale                 Optional integer.
                                  Sets the service's maximum scale for caching
        ---------------------     ----------------------------------------------------
        source_item_id            Optional String.
                                  The Source Item ID is the GeoWarehouse Item ID of the map service
        ---------------------     ----------------------------------------------------
        export_tiles_allowed      Optional bool. Sets the value to let users export tiles
        ---------------------     ----------------------------------------------------
        max_export_tile_count     Optional integer.
                                  Sets the maximum amount of tiles to be exported from a single call.
                                  Deletes tiles for the current cache
        =====================     ====================================================

        :return:
           A boolean indicating success (True), or failure (False)
        """
        params = {
            "f": "json",
        }
        if not service_definition is None:
            params["serviceDefinition"] = service_definition
        if not min_scale is None:
            params["minScale"] = float(min_scale)
        if not max_scale is None:
            params["maxScale"] = float(max_scale)
        if not source_item_id is None:
            params["sourceItemId"] = source_item_id
        if not export_tiles_allowed is None:
            params["exportTilesAllowed"] = export_tiles_allowed
        if not max_export_tile_count is None:
            params["maxExportTileCount"] = int(max_export_tile_count)
        url = self._url + "/edit"
        res = self._con.post(url, params, timeout=None)
        if "success" in res:
            if res["success"]:
                self._img_lyr._hydrated = False
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def delete_tiles(self, levels: str, extent: Optional[dict[str, Any]] = None):
        """
        The ``delete_tiles`` method deletes tiles for the current cache.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        levels              Required string, The level to delete.
                            Example, 0-5,10,11-20 or 1,2,3 or 0-5
        ---------------     ----------------------------------------------------
        extent              Optional dictionary,  If specified, the tiles within
                            this extent will be deleted or will be deleted based
                            on the service's full extent.

                            Example:
                                | 6224324.092137296,487347.5253569535,
                                | 11473407.698535524,4239488.369818687
                                | the minx, miny, maxx, maxy values or,
                                | {"xmin":6224324.092137296,"ymin":487347.5253569535,
                                 "xmax":11473407.698535524,"ymax":4239488.369818687}
                                | "spatialReference":{"wkid":102100}} the JSON
                                | representation of the Extent object.
        ===============     ====================================================

        :return:
           A dictionary

        .. code-block:: python

            from arcgis.mapping import ImageryLayer
            from arcgis.gis import GIS

            # Example Usage
            gis = GIS(url, username, password)
            img_lyr = ImageryLayer("<url>", gis)
            img_lyr_cache_manager = img_lyr.cache_manager
            deleted_tiles = img_lyr_cache_manager.delete_tiles(levels = "11-20",
                                                               extent = {
                                                                            "xmin":6224324.092137296,
                                                                            "ymin":487347.5253569535,
                                                                            "xmax":11473407.698535524,
                                                                            "ymax":4239488.369818687
                                                                        }
                                                               )

        """
        params = {
            "f": "json",
            "levels": levels,
        }
        if extent:
            params["extent"] = extent
        url = self._url + "/deleteTiles"
        return self._con.post(url, params, timeout=None)


###########################################################################
class _RasterRenderingService(Layer):
    def __init__(self, gis=None):
        self.gis = gis
        url = None
        if gis is not None:
            if gis._con._product == "AGOL":
                ra_url = gis.properties.helperServices.get("rasterAnalytics", {}).get(
                    "url", ""
                )
                url = (
                    ra_url.replace("rasteranalysis", "rasterutils")
                    .replace("RasterAnalysisTools", "RasterRendering")
                    .replace("GPServer", "ImageServer")
                )
            else:
                image_hosting_server_url = None
                raster_analytics_server_url = None
                hosting_server_url = None

                for ds in gis._datastores:
                    if (
                        "serverFunction" in ds._server.keys()
                    ) and "ImageHosting" in ds._server["serverFunction"]:
                        image_hosting_server_url = ds._server["url"]
                        break
                    elif (
                        "serverFunction" in ds._server.keys()
                    ) and "RasterAnalytics" in ds._server["serverFunction"]:
                        raster_analytics_server_url = ds._server["url"]
                    elif ("serverFunction" in ds._server.keys()) and ds._server[
                        "serverFunction"
                    ] == "":
                        hosting_server_url = ds._server["url"]
                if image_hosting_server_url:
                    url = (
                        image_hosting_server_url
                        + "/rest/services/System/RasterRendering/ImageServer"
                    )
                elif raster_analytics_server_url:
                    url = (
                        raster_analytics_server_url
                        + "/rest/services/System/RasterRendering/ImageServer"
                    )
                elif hosting_server_url:
                    url = (
                        hosting_server_url
                        + "/rest/services/System/RasterRendering/ImageServer"
                    )
        self.url = url
        if url:
            self.token = gis._con._create_token(url)

        else:
            self.token = None


class ImageryLayer(Layer):
    """

    The ``ImageryLayer`` class can be used to represent an image service resource as a layer.

    An ``ImageryLayer`` object retrieves and displays data from image services. ``ImageryLayer``
    allows you to apply server-defined or client-defined raster functions (e.g. remap, colormap), and mosaic rules.

    ``ImageryLayer`` objects can also be created using raster datasets or raster products present in datastores registered with the server/active GIS
    (types: ``fileShares``, ``cloudStores``, ``rasterStores``).
    To learn more about datastores, visit the
    `What is ArcGIS Data Store? <https://enterprise.arcgis.com/en/portal/latest/administer/windows/what-is-arcgis-data-store.htm>`_
    page in the ArcGIS Enterprise documentation.

    Usage: ``arcgis.raster.ImageryLayer(url, gis=gis)``

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    url                                      Required string. The input raster path

                                             Example:

                                                url = "https://myserver/arcgis/rest/services/ImageServiceName/ImageServer"

                                                url = "/fileShares/file_share_name/path/to/raster"

                                                url = "/cloudStores/cloud_store_name/path/to/raster"

                                                url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/M/BP/2021/6/S2A_43MBP_20210622_0_L2A/B08.tif"

                                             .. note::
                                                When working with datastore rasters or non image service urls,
                                                ``RasterRendering`` service
                                                should be enabled in the active GIS connection

    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. :class:`~arcgis.gis.GIS` of the ImageryLayer object.
    ====================================     ====================================================================

    .. code-block:: python

        # Example Usage

        # Imagery layer items are available as content in the GIS. Items can be searched using gis.content.search()
        # This snippet creates an imagery layer using the 'layers' property of the searched Imagery Layer Item
        img_lyr = gis.content.search("my_image_service", item_type="Imagery Layer")[0].layers[0]

        # Create an imagery layer from an image service url
        img_lyr = ImageryLayer("https://myserver/arcgis/rest/services/ImageServiceName/ImageServer", gis=gis)

        # Create an imagery layer from a .tif file present in user's registered fileShare datastore
        # (Requires RasterRendering service to be enabled in the active GIS)
        img_lyr = ImageryLayer("/fileShares/data/Amberg.tif", gis=gis)

        # Create an imagery layer from a publicly accesible Cloud-Optimized GeoTIFF
        # (Requires RasterRendering service to be enabled in the active GIS)
        img_lyr = ImageryLayer("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/M/BP/2021/6/S2A_43MBP_20210622_0_L2A/B08.tif",
                                gis=gis)

        # Overlay an imagery layer on the 'MapView' widget
        map = gis.map()
        map.add_layer(img_lyr)

    """

    _ilm = None
    _rendering_service_object = None

    def __init__(self, url: str, gis: Optional[GIS] = None):
        self._datastore_raster = False
        self._uri = None
        if isinstance(url, bytes):
            url = base64.b64decode(url)
            url = url.decode("UTF-8")
            import ast

            url = ast.literal_eval(url)
        if isinstance(url, str) or isinstance(url, dict) or isinstance(url, bytes):
            if (
                "/fileShares/" in url
                or "/rasterStores/" in url
                or "/cloudStores/" in url
                or isinstance(url, dict)
                or "/vsi" in url
                or isinstance(url, bytes)
                or "ImageServer" not in url
            ):
                self._gis = _arcgis.env.active_gis if gis is None else gis
                self._datastore_raster = True
                self._uri = url
                if isinstance(url, dict):
                    raster_url = None
                    raster_url = url.get("input_raster", None)
                    if raster_url:
                        self._uri = raster_url
                    else:
                        encoded_dict = str(self._uri).encode("utf-8")
                        self._uri = base64.b64encode(encoded_dict)
                gis = _arcgis.env.active_gis if gis is None else gis
                if gis is not None:
                    if ImageryLayer._rendering_service_object is None or (
                        (
                            (ImageryLayer._rendering_service_object is not None)
                            and ImageryLayer._rendering_service_object.gis is not None
                        )
                        and ImageryLayer._rendering_service_object.gis.url != gis.url
                    ):
                        ImageryLayer._rendering_service_object = (
                            _RasterRenderingService(gis)
                        )
                        url = ImageryLayer._rendering_service_object.url
                    else:
                        if ImageryLayer._rendering_service_object.gis is not None:
                            if (
                                ImageryLayer._rendering_service_object.gis.url
                                == gis.url
                            ):
                                url = ImageryLayer._rendering_service_object.url
                                self._lazy_token = (
                                    ImageryLayer._rendering_service_object.token
                                )

        super(ImageryLayer, self).__init__(url, gis)
        self._spatial_filter = None
        self._temporal_filter = None
        self._where_clause = "1=1"
        self._fn = None
        self._fnra = None
        self._filtered = False
        self._mosaic_rule = self._set_mosaic_rule()
        if self._mosaic_rule:
            self._using_default_mosaic_rule = True
        self._extent = None
        self._uses_gbl_function = False
        self._other_outputs = {}
        self._raster_info = {}
        self._tiles_only = None
        self._extent_set = False
        self._original_info = {}
        self._rendering_rule_from_item = False
        self._rendering_service_layer = None

    def refresh_service(self, options: str = None, future: bool = True) -> str:
        """
        Refresh Service is a task in the existing out-of-the-box
        Publishing Tools geoprocessing service used by the service publisher
        to refresh a GIS service to reflect back-end data changes.
        """
        return self._gis._tools.system_service.refresh_service(
            self, options=options, future=future
        )

    @property
    def rasters(self):
        """
        The ``rasters`` property creates a Raster Manager for this layer.
        Used to create an instance of the manager and help perform edit methods on the
        ImageryLayer.

        :return:
        An instance of :class:`~arcgis.raster.RasterManager` for the ImageryLayer
        """
        if str(self.properties["capabilities"]).lower().find("edit") > -1:
            return RasterManager(self)
        else:
            return None

    @property
    def cache_manager(self):
        """
        The ``cache_manager`` property provides access to the tools to update, add, and remove cache on the
        :class:`~arcgis.raster.ImageryLayer` object.

        :return: :class:`~arcgis.raster.ImageryLayerCacheManager` or None
        """

        def _str_replace(mystring, rd):
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

        if self._ilm is None:
            if self._gis._portal.is_arcgisonline:
                rd = {"/rest/services/": "/rest/admin/services/"}
                adminurl = _str_replace(mystring=self.url, rd=rd)
                self._ilm = ImageryLayerCacheManager(
                    url=adminurl, gis=self._gis, img_lyr=self
                )
        return self._ilm

    @property
    def tiles(self):
        """
        The ``tiles`` property creates an ImageryTileManager for this layer.
        This manager helps perform edits on the tiles of the Image Layer.

        :return: An :class:`~arcgis.raster.ImageryTileManager` for this Image Layer
        """
        if "tileInfo" in self.properties:
            return ImageryTileManager(self)
        else:
            return None

    @property
    def service(self):
        """
        The ``service`` property represents the service backing this imagery layer (if user can administer the service).
        """
        try:
            from arcgis.gis.server._service._adminfactory import (
                AdminServiceGen,
            )

            return AdminServiceGen(service=self, gis=self._gis)
        except:
            return None

    # ----------------------------------------------------------------------
    def catalog_item(self, id: int):
        """
        The ``catalog_item`` method returns a single raster catalog item associated with the specified ID

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        id                    Required integer. The 'raster ID'.
        =================     ====================================================================

        :return:
            :class:`~arcgis.raster.RasterCatalogItem` associated with the ID
        """
        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        if str(self.properties["capabilities"]).lower().find("catalog") == -1:
            return None
        return RasterCatalogItem(url="%s/%s" % (self._url, id), imglyr=self)

    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": type(self).__name__, "url": url}

        options_dict = {"imageServiceParameters": {}}

        if self.tiles_only:
            lyr_dict.update({"capabilities": "tilesOnly"})
        else:
            lyr_dict.update({"capabilities": "dynamic"})
        if self._fn is not None or self._mosaic_rule is not None:
            if self._fn is not None:
                options_dict["imageServiceParameters"]["renderingRule"] = self._fn

            if self._mosaic_rule is not None:
                options_dict["imageServiceParameters"]["mosaicRule"] = self._mosaic_rule

        if self._datastore_raster:
            options_dict["imageServiceParameters"]["raster"] = self._uri
            if isinstance(self._uri, bytes):
                if "renderingRule" in options_dict["imageServiceParameters"]:
                    del options_dict["imageServiceParameters"]["renderingRule"]
                options_dict["imageServiceParameters"]["raster"] = self._fnra

        if options_dict["imageServiceParameters"] != {}:
            lyr_dict.update({"options": json.dumps(options_dict)})
        lyr_dict.update({"uses_gbl": self._uses_gbl_function})
        return lyr_dict

    @classmethod
    def fromitem(cls, item: Item):
        """
        The ``fromitem`` method creates an Image Layer from a GIS :class:`~arcgis.gis.Item`.

        :return:
            An :class:`~arcgis.raster.ImageryLayer` object
        """
        if not item.type == "Image Service":
            raise TypeError("item must be a type of Image Service, not " + item.type)

        return cls(item.url, item._gis)

    @property
    def extent(self):
        """
        Get/Set the area of interest.
        Used for displaying the :class:`~arcgis.raster.ImageryLayer` when queried.

        :return: A dictionary
        """
        if self._extent is None:
            if "initialExtent" in self.properties:
                self._extent = self.properties.initialExtent
            elif "extent" in self.properties:
                self._extent = self.properties.extent
        return self._extent

    @property
    def pixel_type(self):
        """
        Get the pixel type of the :class:`~arcgis.raster.ImageryLayer`

        :return:
            The pixel type
        """
        pixel_type = self.properties.pixelType
        return pixel_type

    @property
    def width(self):
        """
        Get the width of the :class:`~arcgis.raster.ImageryLayer` object.

        :return:
            A float
        """
        width = (
            self.properties.initialExtent["xmax"]
            - self.properties.initialExtent["xmin"]
        )
        return width

    @property
    def height(self):
        """
        Get the height of the :class:`~arcgis.raster.ImageryLayer` object.

        :return:
            A float
        """
        height = (
            self.properties.initialExtent["ymax"]
            - self.properties.initialExtent["ymin"]
        )
        return height

    @property
    def columns(self):
        """
        Get the number of columns in the :class:`~arcgis.raster.ImageryLayer`.

        :return: An Integer
        """
        number_of_columns = (
            self.properties.initialExtent["xmax"]
            - self.properties.initialExtent["xmin"]
        ) / self.properties.pixelSizeX
        return number_of_columns

    @property
    def rows(self):
        """
        Get the number of rows in the :class:`~arcgis.raster.ImageryLayer`.

        :return: An Integer
        """
        number_of_rows = (
            self.properties.initialExtent["ymax"]
            - self.properties.initialExtent["ymin"]
        ) / self.properties.pixelSizeY
        return number_of_rows

    @property
    def band_count(self):
        """
        Get the band count of the :class:`~arcgis.raster.ImageryLayer`.

        :return: An Integer
        """
        band_count = self.properties.bandCount
        return band_count

    @property
    def tiles_only(self):
        """
        The ``tiles_only`` property returns ``True`` if the layer is a Tiled Imagery Layer.

        :return:
            A boolean indicating a Tiled Imagery Layer (True), or not (False).
        """
        if self._tiles_only != None:
            return self._tiles_only
        else:
            self._tiles_only = False
            props = self._get_service_info()
            if "capabilities" in props.keys():
                if "TilesOnly" in props["capabilities"]:
                    self._tiles_only = True
            return self._tiles_only

    @property
    def histograms(self):
        """
        Get the histograms of each band in the :class:`~arcgis.raster.ImageryLayer`
        as a list of dictionaries corresponding to each band.

        .. note::
            If histograms is not found, returns None. In this case, call the
            :attr:`~arcgis.raster.ImageryLayer.compute_histograms` method.

        :Syntax:

            my_hist = imagery_layer.histograms

        :return:
            | #Structure of the return value for a two band imagery layer
            | [
            |  {#band 1
            |  "size":256,
            |  "min":560,
            |  "max":24568,
            |  "counts": [10,99,56,42200,125,....] #length of this list corresponds 'size'
            |  },
            |  {#band 2
            |  "size":256,
            |  "min":8000,
            |  "max":15668,
            |  "counts": [45,9,690,86580,857,....] #length of this list corresponds 'size'
            |  }
            | ]

        """
        if self.properties.hasHistograms:
            # proceed
            url = self._url + "/histograms"
            params = {"f": "json"}
            if self._datastore_raster:
                params["Raster"] = self._uri
            hist_return = self._con.post(url, params, token=self._token, timeout=None)

            # process this into a dict
            return hist_return["histograms"]
        else:
            return None

    @property
    def raster_info(self):
        """
        Get the information about the :class:`~arcgis.raster.ImageryLayer` such as
        ``bandCount``, ``extent``, ``pixelSizeX``, ``pixelSizeY``, and ``pixelType``.

        :return: A dictionary of the raster information
        """
        if self._raster_info != {}:
            return self._raster_info
        if "extent" in self.properties:
            self._raster_info.update({"extent": dict(self.properties.extent)})

        if "bandCount" in self.properties:
            self._raster_info.update({"bandCount": self.properties.bandCount})

        if "pixelType" in self.properties:
            self._raster_info.update({"pixelType": self.properties.pixelType})

        if "pixelSizeX" in self.properties:
            self._raster_info.update({"pixelSizeX": self.properties.pixelSizeX})

        if "pixelSizeY" in self.properties:
            self._raster_info.update({"pixelSizeY": self.properties.pixelSizeY})

        if "compressionType" in self.properties:
            self._raster_info.update(
                {"compressionType": self.properties.compressionType}
            )

        if "blockHeight" in self.properties:
            self._raster_info.update({"blockHeight": self.properties.blockHeight})

        if "blockWidth" in self.properties:
            self._raster_info.update({"blockWidth": self.properties.blockWidth})

        if "noDataValues" in self.properties:
            self._raster_info.update({"noDataValues": self.properties.noDataValues})

        return self._raster_info

    @extent.setter
    def extent(self, value):
        self._extent_set = True
        self._extent = value

    # ----------------------------------------------------------------------
    def attribute_table(
        self, rendering_rule: Optional[str] = None, as_df: bool = False
    ):
        """
        The ``attribute_table`` method returns categorical mapping of pixel
        values (for example, a ``class``, ``group``, ``category``, or ``membership``).

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        rendering_rule        Specifies the rendering rule for how the requested image should be
                              processed.

                              .. note::
                                The response is updated Layer info that reflects a
                                custom processing as defined by the rendering rule. For example, if
                                ``renderingRule`` contains an attributeTable function, the response
                                will indicate "hasRasterAttributeTable": true; if the renderingRule
                                contains functions that alter the number of bands, the response will
                                indicate a correct bandCount value.
        =================     ====================================================================

        :return: A dictionary

        """
        if (
            "hasRasterAttributeTable" in self.properties
            and self.properties["hasRasterAttributeTable"]
        ):
            url = "%s/rasterAttributeTable" % self._url
            params = {"f": "json"}
            if rendering_rule is not None:
                params["renderingRule"] = rendering_rule
            elif self._fn is not None:
                params["renderingRule"] = self._fn

            if self._datastore_raster:
                params["Raster"] = self._uri
                if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                    del params["renderingRule"]
                    params["Raster"] = self._uri

            rat = self._con.post(path=url, postdata=params, timeout=None)

            if as_df:
                import pandas as pd

                if "features" in rat:
                    df1 = pd.DataFrame(rat["features"])
                    if "attributes" in df1.columns:
                        attributes_list = df1["attributes"].tolist()
                        rat_df = pd.DataFrame(attributes_list)
                        rat_df = rat_df.style.set_properties(**{"text-align": "left"})
                        rat_df = rat_df.set_table_styles(
                            [
                                dict(
                                    selector="th",
                                    props=[("text-align", "left")],
                                )
                            ]
                        )
                        return rat_df
                    else:
                        return None
                else:
                    return None
            else:
                return rat

        return None

    # ----------------------------------------------------------------------
    @property
    def multidimensional_info(self):
        """
        Get the multidimensional
        information of the Image Layer. This property is supported if the
        ``hasMultidimensions`` property of the Layer is True.

        .. note::
            Common data sources for multidimensional image services are mosaic
            datasets created from netCDF, GRIB, and HDF data.
        """
        if (
            "hasMultidimensions" in self.properties
            and self.properties["hasMultidimensions"] == True
        ) or self._datastore_raster:
            url = "%s/multiDimensionalInfo" % self._url
            params = {"f": "json"}
            if self._fn is not None:
                params["renderingRule"] = self._fn

            if self._datastore_raster:
                params["Raster"] = self._uri
                if isinstance(self._uri, bytes):
                    del params["renderingRule"]
                    params["Raster"] = self._uri
            return self._con.post(path=url, params=params, timeout=None)

    # ----------------------------------------------------------------------
    @property
    def item_info(self):
        """Get the Image Service's item's infomation"""
        url = "{url}/info/iteminfo".format(url=self._url)
        params = {"f": "json"}
        if self._datastore_raster:
            params["Raster"] = self._uri
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def metadata(self):
        """Get the Image Service's XML metadata file"""
        url = "{url}/info/metadata".format(url=self._url)
        params = {"f": "json"}
        if self._datastore_raster:
            params["Raster"] = self._uri
        return self._con.get(url, params, try_json=False)

    # ----------------------------------------------------------------------
    def project(
        self,
        geometries: list,
        in_sr: Union[str, dict[str, Any], SpatialReference],
        out_sr: Union[str, dict[str, Any], SpatialReference],
    ):
        """
        The ``project`` method is performed on an :class:`~arcgis.raster.ImageryLayer` object.
        This operation projects a list of input geometries from the input
        spatial reference to the output spatial reference. The response
        order of geometries is in the same order as they were requested.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        geometries            Required list. The list of geometries to be projected.
        -----------------     --------------------------------------------------------------------
        in_sr                 Required string, dictionary, :class:`~arcgis.geometry.SpatialReference`. The ``in_sr``
                              can accept a
                              multitudes of values.  These can be a WKID, image coordinate system
                              (ICSID), or image coordinate system in json/dict format.
                              Additionally the arcgis.geometry.SpatialReference object is also a
                              valid entry.

                              .. note::

                                    An image coordinate system ID can be specified
                                    using 0:icsid; for example, 0:64. The extra 0: is used to avoid
                                    conflicts with wkid
        -----------------     --------------------------------------------------------------------
        out_sr                Required string, dictionary, :class:`~arcgis.geometry.SpatialReference`. The ``out_sr``
                              can accept a
                              multitudes of values.  These can be a WKID, image coordinate system
                              (ICSID), or image coordinate system in json/dict format.
                              Additionally the arcgis.geometry.SpatialReference object is also a
                              valid entry.

                              .. note::

                                    An image coordinate system ID can be specified
                                    using 0:icsid; for example, 0:64. The extra 0: is used to avoid
                                    conflicts with wkid
        =================     ====================================================================

        :return: A dictionary


        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)
            except:
                raise RuntimeError(
                    "Failed to perform project operation on the TilesOnly service"
                )

        url = "%s/project" % self._url
        params = {
            "f": "json",
            "inSR": in_sr,
            "outSR": out_sr,
            "geometries": geometries,
        }
        if self._datastore_raster:
            params["Raster"] = self._uri
        return self._con.post(path=url, postdata=params, timeout=None)

    # ----------------------------------------------------------------------
    def identify(
        self,
        geometry: Union[dict[str, Any], Polygon, Point, MultiPoint, Envelope],
        mosaic_rule: Optional[Union[str, dict]] = None,
        rendering_rules: Optional[Union[list[str], dict[str, Any]]] = None,
        pixel_size: Optional[Union[str, dict[str, int]]] = None,
        time_extent: Optional[Union[list[datetime.datetime], datetime.datetime]] = None,
        return_geometry: bool = False,
        return_catalog_items: bool = True,
        return_pixel_values: bool = True,
        max_item_count: Optional[int] = None,
        slice_id: Optional[int] = None,
        process_as_multidimensional: bool = False,
    ):
        """

        The ``identify`` method identifies the content of an image layer for a given location
        and a given mosaic rule. The location can be a :class:`~arcgis.geometry.Point` or a
        :class:`~arcgis.geometry.Polygon` or a
        :class:`~arcgis.geometry.Envelope` or a
        :class:`~arcgis.geometry.MultiPoint`

        .. note::
            The ``identify`` operation is supported by both mosaic dataset and
            raster dataset image services.

        The result of this operation includes the pixel value of the mosaic
        for a given mosaic rule, a resolution (pixel size), and a set of
        catalog items that overlap the given geometry. The single pixel
        value is that of the mosaic at the centroid of the specified
        location. If there are multiple rasters overlapping the location,
        the visibility of a raster is determined by the order of the
        rasters defined in the mosaic rule. It also contains a set of
        catalog items that overlap the given geometry. The catalog items
        are ordered based on the mosaic rule. A list of catalog item
        visibilities gives the percentage contribution of the item to
        overall mosaic.

        ============================    ====================================================================
        **Parameter**                   **Description**
        ----------------------------    --------------------------------------------------------------------
        geometry                        Required dictionary/Point/Polygon/MultiPoint/Envelope. A :class:`~arcgis.geometry.Geometry` that
                                        defines the location to be identified.

                                        .. note::
                                            The location can be a point or polygon or envelope or multipoint.
                                              - Support for envelope was added at 10.9.1.
                                              - Support for multipoint was added at 11.0.
        ----------------------------    --------------------------------------------------------------------
        mosaic_rule                     Optional string or dict. Specifies the mosaic rule when defining how
                                        individual images should be mosaicked. When a mosaic rule is not
                                        specified, the default mosaic rule of the image layer will be used
                                        (as advertised in the root resource: defaultMosaicMethod,
                                        mosaicOperator, sortField, sortValue).
        ----------------------------    --------------------------------------------------------------------
        rendering_rules                 Optional dictionary/list. Specifies the rendering rule for how the
                                        requested image should be rendered.
        ----------------------------    --------------------------------------------------------------------
        pixel_size                      Optional string or dict. The pixel level being identified (or the
                                        resolution being looked at).
                                        Syntax:
                                          - dictionary structure: pixel_size={point}
                                          - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                          - pixel_size={"x": 0.18, "y": 0.18}
                                          - pixel_size='0.18,0.18'
        ----------------------------    --------------------------------------------------------------------
        time_extent                     Optional list of datetime objects or datetime object.  The time
                                        instant or time extent of the raster to be identified. This
                                        parameter is only valid if the image layer supports time.
        ----------------------------    --------------------------------------------------------------------
        return_geometry                 Optional boolean. Default is False.  Indicates whether or not to
                                        return the raster catalog item's footprint. Set it to false when the
                                        catalog item's footprint is not needed to improve the identify
                                        operation's response time.
        ----------------------------    --------------------------------------------------------------------
        return_catalog_items            Optional boolean.  Indicates whether or not to return raster catalog
                                        items. Set it to false when catalog items are not needed to improve
                                        the identify operation's performance significantly. When set to
                                        false, neither the geometry nor attributes of catalog items will be
                                        returned.
        ----------------------------    --------------------------------------------------------------------
        return_pixel_values             Optional boolean.  Indicates whether to return the pixel values of
                                        all mosaicked raster catalog items under the requested geometry.

                                        Set it to false when only the pixel value of mosaicked output is
                                        needed at requested geometry.

                                        The default value of this parameter is ``True``.

                                        .. note::
                                            This parameter was added at 10.6.1.
        ----------------------------    --------------------------------------------------------------------
        max_item_count                  Optional int. If the return_catalog_items parameter is set to true,
                                        this parameter will take effect. The default behavior is to return
                                        all raster catalog items within the requested geometry.
                                        Otherwise, the number of items returned will be the value specified in the
                                        max_item_count or all eligible items, whichever is smaller.

                                        Example:
                                        max_item_count = 5

                                        .. note::
                                            The parameter was added at 10.6.1.

        ----------------------------    --------------------------------------------------------------------
        slice_id                        Optional int. The slice ID of multidimensional raster. The identify
                                        operation will be performed for the specified slice. To get the slice
                                        ID use slices method on the ``ImageryLayer`` object.


                                        .. note::
                                            The parameter was added at 10.9 for image services which use
                                            ``ArcObjects11`` or ``ArcObjectsRasterRendering``
                                            as the service provider.
        ----------------------------    --------------------------------------------------------------------
        process_as_multidimensional     Optional boolean. Specifies whether to process the image service as a
                                        multidimensional image service.

                                            - ``False`` - Pixel values of the specified rendering rules and mosaic\
                                            rule at the specified geometry will be returned. This is the\
                                            default.
                                            - ``True`` - The image service is treated as a multidimensional raster,\
                                            and pixel values from all slices, along with additional properties\
                                            describing the slices, will be returned.

                                        .. note::
                                            The parameter was added at 10.9 for image services which use
                                            ``ArcObjects11`` or ``ArcObjectsRasterRendering``
                                            as the service provider.
        ============================    ====================================================================

        :return: A dictionary

        .. code-block:: python

            # Example Usage
            img_layer = gis.content.search("my_image_service", item_type="Imagery Layer")[0].layers[0]
            identified = img_layer.identify(geometry=polygon_obj,
                                            pixel_size="0.18,0.18",
                                            return_geometry=True,
                                            return_pixel_values=False,
                                            max_item_count=5,
                                            slice_id=1,
                                            )
        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)
            except:
                raise RuntimeError(
                    "Failed to perform identify operation on the TilesOnly service"
                )

        url = "%s/identify" % self._url
        params = {"f": "json", "geometry": dict(geometry)}
        from arcgis.geometry._types import (
            Point,
            Polygon,
            Envelope,
            MultiPoint,
        )
        from arcgis._impl.common._mixins import PropertyMap

        if isinstance(geometry, Point):
            params["geometryType"] = "esriGeometryPoint"
        elif isinstance(geometry, Polygon):
            params["geometryType"] = "esriGeometryPolygon"
        elif isinstance(geometry, (Envelope, PropertyMap)):
            params["geometryType"] = "esriGeometryEnvelope"
        elif isinstance(geometry, MultiPoint):
            params["geometryType"] = "esriGeometryMultipoint"
        elif isinstance(geometry, dict):
            if "x" in geometry:
                params["geometryType"] = "esriGeometryPoint"
            elif "points" in geometry:
                params["geometryType"] = "esriGeometryMultipoint"
            elif "xmin" in geometry:
                params["geometryType"] = "esriGeometryEnvelope"
            else:
                params["geometryType"] = "esriGeometryPolygon"

        if mosaic_rule is not None:
            params["mosaicRule"] = mosaic_rule
        elif self._mosaic_rule is not None:
            params["mosaicRule"] = self._mosaic_rule

        if rendering_rules is not None:
            if isinstance(rendering_rules, dict):
                params["renderingRule"] = rendering_rules
            elif isinstance(rendering_rules, list):
                params["renderingRules"] = rendering_rules
            else:
                raise ValueError(
                    "Invalid Rendering Rules - It can be only be a dictionary or a list type object"
                )
        elif self._fn:
            params["renderingRule"] = self._fn

        if pixel_size is not None:
            params["pixelSize"] = pixel_size
        if time_extent is not None:
            if isinstance(time_extent, datetime.datetime):
                time_extent = "%s" % int(time_extent.timestamp() * 1000)
            elif isinstance(time_extent, list):
                time_extent = "%s,%s" % (
                    int(time_extent[0].timestamp() * 1000),
                    int(time_extent[1].timestamp() * 1000),
                )
            params["time"] = time_extent
        elif time_extent is None and self._temporal_filter is not None:
            params["time"] = self._temporal_filter
        if isinstance(return_geometry, bool):
            params["returnGeometry"] = return_geometry
        if isinstance(return_catalog_items, bool):
            params["returnCatalogItems"] = return_catalog_items
        if isinstance(return_pixel_values, bool):
            params["returnPixelValues"] = return_pixel_values

        if max_item_count is not None:
            params["maxItemCount"] = max_item_count

        if slice_id is not None:
            params["sliceId"] = slice_id

        if isinstance(process_as_multidimensional, bool):
            params["processAsMultidimensional"] = process_as_multidimensional

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]
                params["Raster"] = self._uri

        return self._con.post(path=url, postdata=params, timeout=None)

    # ----------------------------------------------------------------------
    def measure(
        self,
        from_geometry: Union[Geometry, dict[str, Any]],
        to_geometry: Optional[Union[Geometry, dict[str, Any]]] = None,
        measure_operation: Optional[Union[str, dict[str, str]]] = None,
        pixel_size: Optional[Union[str, dict[str, float]]] = None,
        mosaic_rule: Optional[Union[str, dict[str, str]]] = None,
        linear_unit: Optional[str] = None,
        angular_unit: Optional[str] = None,
        area_unit: Optional[str] = None,
    ):
        """
        The ``measure`` method lets a user measure distance, direction, area,
        perimeter, and height from an image layer. The result of this
        operation includes the name of the raster dataset being used,
        sensor name, and measured values.

        .. note::
            The measure operation can be supported by image services from
            raster datasets and mosaic datasets. Spatial reference is required
            to perform basic measurement (distance, area, and so on). Sensor
            metadata (geodata transformation) needs to be present in the data
            source used by an image layer to enable height measurement (for
            example, imagery with RPCs). The mosaic dataset or Layer needs to
            include DEM to perform 3D measure.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        from_geometry         Required :class:`~arcgis.geometry.Geometry` or dictionary.
                              A geometry that defines the ``from`` location of the measurement.
        -----------------     --------------------------------------------------------------------
        to_geometry           Optional :class:`~arcgis.geometry.Geometry` or dictionary. A geometry that defines
                              the ``to``
                              location of the measurement. The type of geometry must be the same
                              as ``from_geometry``.
        -----------------     --------------------------------------------------------------------
        measure_operation     Optional string or dict. Specifies the type of measure being
                              performed.

                              Values: `Point, DistanceAndAngle, AreaAndPerimeter, HeightFromBaseAndTop,
                              HeightFromBaseAndTopShadow,
                              HeightFromTopAndTopShadow, Centroid,
                              Point3D, DistanceAndAngle3D,
                              AreaAndPerimeter3D, Centroid3D`

                              Different measureOperation types require different from and to
                              geometries:
                               - Point and Point3D-Require only \
                                 from_geometry, type: {Point}
                               - DistanceAndAngle, DistanceAndAngle3D, \
                               HeightFromBaseAndTop, \
                               HeightFromBaseAndTopShadow, and \
                               HeightFromTopAndTopShadow - Require both \
                               from_geometry and to_geometry, type: {Point}
                               - AreaAndPerimeter, \
                                 AreaAndPerimeter3D, Centroid, and \
                                 Centroid3D - Require only from_geometry, \
                                 type: {Polygon}, {Envelope}
                              Supported measure operations can be derived from the
                              mensurationCapabilities in the image layer root resource.
                              Basic capability supports Point,
                              DistanceAndAngle, AreaAndPerimeter,
                              and Centroid.
                              Basic and 3Dcapabilities support Point3D,
                              DistanceAndAngle3D,AreaAndPerimeter3D,
                              and Centroid3D.
                              Base-Top Height capability supports
                              HeightFromBaseAndTop.
                              Top-Top Shadow Height capability supports
                              HeightFromTopAndTopShadow.
                              Base-Top Shadow Height capability supports
                              HeightFromBaseAndTopShadow.
        -----------------     --------------------------------------------------------------------
        pixel_size            Optional string or dict. The pixel level (resolution) being
                              measured. If pixel size is not specified, pixel_size will default to
                              the base resolution of the image layer. The raster at the specified pixel
                              size in the mosaic dataset will be used for measurement.
                              Syntax:
                                - dictionary structure: pixel_size={point}
                                - Point simple syntax: pixel_size='<x>,<y>'
                              Examples:
                                - pixel_size={"x": 0.18, "y": 0.18}
                                - pixel_size='0.18,0.18'
        -----------------     --------------------------------------------------------------------
        mosaic_rule           Optional string or dict. Specifies the mosaic rule when defining how
                              individual images should be mosaicked. When a mosaic rule is not
                              specified, the default mosaic rule of the image layer will be used
                              (as advertised in the root resource: defaultMosaicMethod,
                              mosaicOperator, sortField, sortValue). The first visible image is
                              used by measure.
        -----------------     --------------------------------------------------------------------
        linear_unit           Optional string. The linear unit in which height, length, or
                              perimeters will be calculated. It can be any of the following
                              U constant. If the unit is not specified, the default is
                              Meters. The list of valid Units constants include:
                              `Inches,Feet,Yards,Miles,NauticalMiles,
                              Millimeters,Centimeters,Decimeters,Meters,
                              Kilometers`
        -----------------     --------------------------------------------------------------------
        angular_unit          Optional string. The angular unit in which directions of line
                              segments will be calculated. It can be one of the following
                              DirectionUnits constants:
                              `DURadians, DUDecimalDegrees`

                              .. note::
                                If the unit is not specified, the default is `DUDecimalDegrees`.
        -----------------     --------------------------------------------------------------------
        area_unit             Optional string. The area unit in which areas of polygons will be
                              calculated. It can be any AreaUnits constant. If the unit is not
                              specified, the default is SquareMeters. The list of valid
                              AreaUnits constants include:
                              `SquareInches,SquareFeet,SquareYards,Acres,
                              SquareMiles,SquareMillimeters,SquareCentimeters,
                              SquareDecimeters,SquareMeters,Ares,Hectares,
                              SquareKilometers`
        =================     ====================================================================

        :return: A dictionary

        .. code-block:: python

            # Example Usage
            img_layer = gis.content.search("my_image_service", item_type="Imagery Layer")[0].layers[0]
            measured = img_layer.measure(from_geometry=point1,
                                         to_geometry=point2,
                                         measure_operation="HeightFromTopAndTopShadow")
        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)

            except:
                raise RuntimeError(
                    "Failed to perform measure operation on the TilesOnly service"
                )

        if linear_unit is not None:
            linear_unit = "esri%s" % linear_unit
        if angular_unit is not None:
            angular_unit = "esri%s" % angular_unit
        if area_unit is not None:
            area_unit = "esri%s" % area_unit
        measure_operation = "esriMensuration%s" % measure_operation
        url = "%s/measure" % self._url
        params = {"f": "json", "fromGeometry": from_geometry}
        if self._datastore_raster:
            params["Raster"] = self._uri
        from arcgis.geometry._types import Polygon, Point, Envelope
        from arcgis._impl.common._mixins import PropertyMap

        if isinstance(from_geometry, Polygon):
            params["geometryType"] = "esriGeometryPolygon"
        elif isinstance(from_geometry, Point):
            params["geometryType"] = "esriGeometryPoint"
        elif isinstance(from_geometry, (Envelope, PropertyMap)):
            params["geometryType"] = "esriGeometryEnvelope"
        elif isinstance(from_geometry, dict):
            if "x" in from_geometry:
                params["geometryType"] = "esriGeometryPoint"
            elif "xmin" in from_geometry:
                params["geometryType"] = "esriGeometryEnvelope"
            else:
                params["geometryType"] = "esriGeometryPolygon"
        if to_geometry:
            params["toGeometry"] = to_geometry
        if measure_operation is not None:
            params["measureOperation"] = measure_operation
        if mosaic_rule is not None:
            params["mosaicRule"] = mosaic_rule
        elif self._mosaic_rule is not None:
            params["mosaicRule"] = self._mosaic_rule
        if pixel_size:
            params["pixelSize"] = pixel_size
        if linear_unit:
            params["linearUnit"] = linear_unit
        if area_unit:
            params["areaUnit"] = area_unit
        if angular_unit:
            params["angularUnit"] = angular_unit
        return self._con.post(path=url, postdata=params, timeout=None)

    def set_filter(
        self,
        where: Optional[str] = None,
        geometry: Optional[dict] = None,
        time: Optional[Union[datetime.datetime, datetime.date, list[int], str]] = None,
        lock_rasters: bool = False,
        clear_filters: bool = False,
    ):
        """
        The ``set_filters`` method filters the rasters that will be used for applying raster functions.

        If ``lock_rasters`` is set to ``True``, the LockRaster mosaic rule will be applied to the layer,
        unless overridden.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        where                 Optional string. A where clause on this layer to filter the imagery
                              layer by the selection sql statement. Any legal SQL where clause
                              operating on the fields in the raster
        -----------------     --------------------------------------------------------------------
        geometry              Optional arcgis.geometry.filters. To filter results by a spatial
                              relationship with another geometry
        -----------------     --------------------------------------------------------------------
        time                  Optional datetime, date, or timestamp. A temporal filter to this
                              layer to filter the imagery layer by time using the specified time
                              instant or the time extent.

                              Syntax: time_filter=<timeInstant>

                              Time extent specified as list of [<startTime>, <endTime>]
                              For time extents one of <startTime> or <endTime> could be None. A
                              None value specified for start time or end time will represent
                              infinity for start or end time respectively.
                              Syntax: time_filter=[<startTime>, <endTime>] ; specified as
                              datetime.date, datetime.datetime or timestamp in milliseconds
        -----------------     --------------------------------------------------------------------
        lock_rasters          Optional boolean. If True, the LockRaster mosaic rule will be
                              applied to the layer, unless overridden
        -----------------     --------------------------------------------------------------------
        clear_filters         Optional boolean. If True, the applied filters are cleared
        =================     ====================================================================

        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        if clear_filters:
            self._filtered = False
            self._where_clause = None
            self._temporal_filter = None
            self._spatial_filter = None
            self._mosaic_rule = None
        else:
            self._filtered = True
            if where is not None:
                self._where_clause = where

            if geometry is not None:
                self._spatial_filter = geometry

            if time is not None:
                self._temporal_filter = time

            if lock_rasters:
                oids = self.query(
                    where=self._where_clause,
                    time_filter=self._temporal_filter,
                    geometry_filter=self._spatial_filter,
                    return_ids_only=True,
                )["objectIds"]
                self._mosaic_rule = {
                    "mosaicMethod": "esriMosaicLockRaster",
                    "lockRasterIds": oids,
                    "ascending": True,
                    "mosaicOperation": "MT_FIRST",
                }

    def filter_by(
        self,
        where: Optional[str] = None,
        geometry: Optional[dict] = None,
        time: Optional[Union[datetime.datetime, datetime.date, list[int], str]] = None,
        lock_rasters: bool = True,
    ):
        """
        The ``filter_by`` method filters the layer by where clause, geometry and temporal filters.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        where                 Optional string. A where clause on this layer to filter the imagery
                              layer by the selection sql statement. Any legal SQL where clause
                              operating on the fields in the raster
        -----------------     --------------------------------------------------------------------
        geometry              Optional arcgis.geometry.filters. To filter results by a spatial
                              relationship with another geometry
        -----------------     --------------------------------------------------------------------
        time                  Optional datetime, date, or timestamp. A temporal filter to this
                              layer to filter the imagery layer by time using the specified time
                              instant or the time extent.

                              Syntax: time_filter=<timeInstant>

                              Time extent specified as list of [<startTime>, <endTime>]
                              For time extents one of <startTime> or <endTime> could be None. A
                              None value specified for start time or end time will represent
                              infinity for start or end time respectively.
                              Syntax: time_filter=[<startTime>, <endTime>] ; specified as
                              datetime.date, datetime.datetime or timestamp in milliseconds
        -----------------     --------------------------------------------------------------------
        lock_rasters          Optional boolean. If True, the LockRaster mosaic rule will be
                              applied to the layer, unless overridden
        =================     ====================================================================

        :return: :class:`~arcgis.raster.ImageryLayer` with filtered images meeting the filter criteria

        .. code-block:: python

            # Example Usage
            img_lyr = gis.content.search("my_image_service", item_type="Imagery Layer")[0].layers[0]
            filtered_img_lyr = img_lyr.filter_by(time=[1627417239146],
                                                lock_rasters=True)

        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        newlyr = self._clone_layer()

        newlyr._where_clause = where
        newlyr._spatial_filter = geometry
        newlyr._temporal_filter = time

        if lock_rasters:
            oids = self.query(
                where=where,
                time_filter=time,
                geometry_filter=geometry,
                return_ids_only=True,
            )["objectIds"]
            newlyr._mosaic_rule = {
                "mosaicMethod": "esriMosaicLockRaster",
                "lockRasterIds": oids,
                "ascending": True,
                "mosaicOperation": "MT_FIRST",
            }

        newlyr._filtered = True
        return newlyr

    def _clone_layer(self):
        if type(self).__name__ == "Raster" or type(self).__name__ == "RasterCollection":
            newlyr = Raster(
                self._url,
                is_multidimensional=self._is_multidimensional,
                gis=self._gis,
            )

        elif type(self).__name__ == "ImageryLayer":
            if self._datastore_raster:
                newlyr = ImageryLayer(self._uri, self._gis)
            else:
                newlyr = ImageryLayer(self._url, self._gis)
        newlyr._lazy_properties = self.properties
        newlyr._hydrated = True
        newlyr._lazy_token = self._token

        newlyr._fn = self._fn
        newlyr._fnra = self._fnra
        newlyr._mosaic_rule = self._mosaic_rule
        newlyr._extent = self._extent
        newlyr._extent_set = self._extent_set

        newlyr._where_clause = self._where_clause
        newlyr._spatial_filter = self._spatial_filter
        newlyr._temporal_filter = self._temporal_filter
        newlyr._filtered = self._filtered

        return newlyr

    def filtered_rasters(self):
        """
        The ``filtered_rasters`` method retrieves object ids of the filtered rasters in this
        :class:`~arcgis.raster.ImageryLayer`, by applying the where clause, spatial and
        temporal filters.

        :return:
            A list of object ids. If no rasters are filtered, returns None.
            If all rasters are filtered, returns empty list.
        """

        if self._filtered:
            oids = self.query(
                where=self._where_clause,
                time_filter=self._temporal_filter,
                geometry_filter=self._spatial_filter,
                return_ids_only=True,
            )["objectIds"]
            return oids  # ['$' + str(x) for x in oids]
        else:
            return None  # return '$$'

    def export_image(
        self,
        bbox: Optional[Union[dict[str, float], str]] = None,
        image_sr: Optional[Union[int, dict[str, Any], SpatialReference]] = None,
        bbox_sr: Optional[Union[int, dict[str, Any], SpatialReference]] = None,
        size: Optional[list[int]] = None,
        time: Optional[Union[datetime.date, datetime.datetime, list[int], str]] = None,
        export_format: str = "jpgpng",
        pixel_type: Optional[str] = None,
        no_data: Optional[float] = None,
        no_data_interpretation: str = "esriNoDataMatchAny",
        interpolation: Optional[str] = None,
        compression: Optional[str] = None,
        compression_quality: Optional[int] = None,
        band_ids: Optional[list[int]] = None,
        mosaic_rule: Optional[dict[str, Any]] = None,
        rendering_rule: Optional[dict[str, Any]] = None,
        f: str = "json",
        save_folder: Optional[str] = None,
        save_file: Optional[str] = None,
        compression_tolerance: Optional[float] = None,
        adjust_aspect_ratio: Optional[bool] = None,
        lerc_version: Optional[int] = None,
        slice_id: Optional[int] = None,
    ):
        """
        The ``export_image`` operation is performed on an :class:`~arcgis.raster.ImageryLayer` object.
        The result of this operation is an image method. This method
        provides information about the exported image, such as its URL,
        extent, width, and height.

        .. note::
            In addition to the usual response formats of HTML and JSON, you can
            also request the image format while performing this operation. When
            you perform an export with the image format , the server responds
            by directly streaming the image bytes to the client. With this
            approach, you don't get any information associated with the
            exported image other than the image itself.

        ======================  ====================================================================
        **Parameter**           **Description**
        ----------------------  --------------------------------------------------------------------
        bbox                    Optional dict or string. The extent (bounding box) of the exported
                                image. Unless the bbox_sr parameter has been specified, the bbox is
                                assumed to be in the spatial reference of the imagery layer.

                                The bbox should be specified as an arcgis.geometry.Envelope object,
                                it's json representation or as a list or string with this
                                format: '<xmin>, <ymin>, <xmax>, <ymax>'
                                If omitted, the extent of the imagery layer is used
        ----------------------  --------------------------------------------------------------------
        image_sr                Optional string, :class:`~arcgis.geometry.SpatialReference`.
                                The spatial reference of the
                                exported image. The spatial reference can be specified as either a
                                well-known ID, it's json representation or as an
                                :class:`~arcgis.geometry.SpatialReference` object.
                                If the ``image_sr`` is not specified, the image will be exported in the
                                spatial reference of the imagery layer.
        ----------------------  --------------------------------------------------------------------
        bbox_sr                 Optional string, SpatialReference. The spatial reference of the
                                bbox.
                                The spatial reference can be specified as either a well-known ID,
                                it's json representation or as an arcgis.geometry.SpatialReference
                                object.
                                If the image_sr is not specified, bbox is assumed to be in the
                                spatial reference of the imagery layer.
        ----------------------  --------------------------------------------------------------------
        size                    Optional list. The size (width * height) of the exported image in
                                pixels. If size is not specified, an image with a default size of
                                1200*450 will be exported.
                                Syntax: list of [width, height]
        ----------------------  --------------------------------------------------------------------
        time                    Optional datetime.date, datetime.datetime or timestamp string. The
                                time instant or the time extent of the exported image.
                                Time instant specified as datetime.date, datetime.datetime or
                                timestamp in milliseconds since epoch
                                Syntax: time=<timeInstant>

                                Time extent specified as list of [<startTime>, <endTime>]
                                For time extents one of <startTime> or <endTime> could be None. A
                                None value specified for start time or end time will represent
                                infinity for start or end time respectively.
                                Syntax: time=[<startTime>, <endTime>] ; specified as
                                datetime.date, datetime.datetime or timestamp
        ----------------------  --------------------------------------------------------------------
        export_format           Optional string. The format of the exported image. The default
                                format is jpgpng. The jpgpng format returns a JPG if there are no
                                transparent pixels in the requested extent; otherwise, it returns a
                                PNG (png32).

                                Values: jpgpng,png,png8,png24,jpg,bmp,gif,tiff,png32,bip,bsq,lerc
        ----------------------  --------------------------------------------------------------------
        pixel_type              Optional string. The pixel type, also known as data type, pertains
                                to the type of values stored in the raster, such as signed integer,
                                unsigned integer, or floating point. Integers are whole numbers,
                                whereas floating points have decimals.
        ----------------------  --------------------------------------------------------------------
        no_data                 Optional float. The pixel value representing no information.
        ----------------------  --------------------------------------------------------------------
        no_data_interpretation  Optional string. Interpretation of the no_data setting. The default
                                is NoDataMatchAny when no_data is a number, and NoDataMatchAll when
                                no_data is a comma-delimited string: NoDataMatchAny,NoDataMatchAll.
        ----------------------  --------------------------------------------------------------------
        interpolation           Optional string. The resampling process of extrapolating the pixel
                                values while transforming the raster dataset when it undergoes
                                warping or when it changes coordinate space.
                                One of: RSP_BilinearInterpolation, RSP_CubicConvolution,
                                RSP_Majority, RSP_NearestNeighbor
        ----------------------  --------------------------------------------------------------------
        compression             Optional string. Controls how to compress the image when exporting
                                to TIFF format: None, JPEG, LZ77. It does not control compression on
                                other formats.
        ----------------------  --------------------------------------------------------------------
        compression_quality     Optional integer. Controls how much loss the image will be subjected
                                to by the compression algorithm. Valid value ranges of compression
                                quality are from 0 to 100.
        ----------------------  --------------------------------------------------------------------
        band_ids                Optional list. If there are multiple bands, you can specify a single
                                band to export, or you can change the band combination (red, green,
                                blue) by specifying the band number. Band number is 0 based.
                                Specified as list of ints, eg [2,1,0]
        ----------------------  --------------------------------------------------------------------
        mosaic_rule             Optional dict. Specifies the mosaic rule when defining how
                                individual images should be mosaicked. When a mosaic rule is not
                                specified, the default mosaic rule of the image layer will be used
                                (as advertised in the root resource: defaultMosaicMethod,
                                mosaicOperator, sortField, sortValue).
        ----------------------  --------------------------------------------------------------------
        rendering_rule          Optional dict. Specifies the rendering rule for how the requested
                                image should be rendered.
        ----------------------  --------------------------------------------------------------------
        f                       Optional string. The response format.  default is json
                                Values: json,image,kmz,numpy_array

                                **Note:** If f="numpy_array" and if the raster is a single or multiband raster,
                                the dimensions of the array will be rows, columns, and number of bands.
                                If the raster is a multidimensional raster, the dimensions of the array
                                will be number of slices, rows, columns, and number of bands.
                                LERC needs to be installed to export image service as numpy array.

                                If f="image", the bytes of the exported image are
                                returned unless save_folder and save_file parameters are also
                                passed, in which case the image is written to the specified file
        ----------------------  --------------------------------------------------------------------
        save_folder             Optional string. The folder in which the exported image is saved
                                when f=image
        ----------------------  --------------------------------------------------------------------
        save_file               Optional string. The file in which the exported image is saved when
                                f=image
        ----------------------  --------------------------------------------------------------------
        compression_tolerance   Optional float. Controls the tolerance of the lerc compression
                                algorithm. The tolerance defines the maximum possible error of pixel
                                values in the compressed image.
                                Example: compression_tolerance=0.5 is loseless for 8 and 16 bit
                                images, but has an accuracy of +-0.5 for floating point data. The
                                compression tolerance works for the LERC format only.
        ----------------------  --------------------------------------------------------------------
        adjust_aspect_ratio     Optional boolean. Indicates whether to adjust the aspect ratio or
                                not. By default adjust_aspect_ratio is true, that means the actual
                                bbox will be adjusted to match the width/height ratio of size
                                paramter, and the response image has square pixels.
        ----------------------  --------------------------------------------------------------------
        lerc_version            Optional integer. The version of the Lerc format if the user sets
                                the format as lerc.
                                Values: 1 or 2
                                If a version is specified, the server returns the matching version,
                                or otherwise the highest version available.
        ----------------------  --------------------------------------------------------------------
        slice_id                Optional integer. Exports the given slice of a multidimensional raster.
                                To get the slice index use slices method on the ImageryLayer object.
        ======================  ====================================================================

        :return: Dictionary of the data that is exported or a string indicating the file path.

        .. code-block:: python

            # Usage Example: Exports an ImageryLayer object (created using Image Service) to a local location in tiff format

            imagery_layer = ImageryLayer("https://myserver/arcgis/rest/services/ImageServiceName/ImageServer", gis=gis)
            imagery_layer.export_image(size=[1400, 600],
                                       export_format="tiff",
                                       f="image",
                                       save_folder=r"/path/to/save_folder",
                                       save_file="my_raster.tif")

        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)

            except:
                raise RuntimeError(
                    "Failed to perform export image operation on the TilesOnly service"
                )

        import datetime

        no_data_interpretation = "esri%s" % no_data_interpretation
        if size is None:
            size = [1200, 450]

        params = {
            "size": "%s,%s" % (size[0], size[1]),
        }

        if bbox is not None:
            if type(bbox) == str:
                params["bbox"] = bbox
            elif type(bbox) == list:
                params["bbox"] = "%s,%s,%s,%s" % (
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                )
            else:  # json dict or Geometry Envelope object
                if bbox_sr is None:
                    if "spatialReference" in bbox:
                        bbox_sr = bbox["spatialReference"]

                bbox = "%s,%s,%s,%s" % (
                    bbox["xmin"],
                    bbox["ymin"],
                    bbox["xmax"],
                    bbox["ymax"],
                )
                params["bbox"] = bbox

        else:
            params["bbox"] = self.extent  # properties.initialExtent
            if bbox_sr is None:
                if "spatialReference" in self.extent:
                    bbox_sr = self.extent["spatialReference"]

        if image_sr is not None:
            params["imageSR"] = image_sr
        if bbox_sr is not None:
            params["bboxSR"] = bbox_sr
        if pixel_type is not None:
            params["pixelType"] = pixel_type

        url = self._url + "/exportImage"
        __allowedFormat = [
            "jpgpng",
            "png",
            "png8",
            "png24",
            "jpg",
            "bmp",
            "gif",
            "tiff",
            "png32",
            "bip",
            "bsq",
            "lerc",
        ]
        __allowedPixelTypes = [
            "C128",
            "C64",
            "F32",
            "F64",
            "S16",
            "S32",
            "S8",
            "U1",
            "U16",
            "U2",
            "U32",
            "U4",
            "U8",
            "UNKNOWN",
        ]
        __allowednoDataInt = ["esriNoDataMatchAny", "esriNoDataMatchAll"]
        __allowedInterpolation = [
            "RSP_BilinearInterpolation",
            "RSP_CubicConvolution",
            "RSP_Majority",
            "RSP_NearestNeighbor",
        ]
        __allowedCompression = ["JPEG", "LZ77"]
        if mosaic_rule is not None:
            params["mosaicRule"] = mosaic_rule
        elif self._mosaic_rule is not None:
            params["mosaicRule"] = self._mosaic_rule

        if export_format in __allowedFormat:
            params["format"] = export_format

        if self._temporal_filter is not None:
            time = self._temporal_filter

        if time is not None:
            if type(time) is list:
                starttime = _date_handler(time[0])
                endtime = _date_handler(time[1])
                if starttime is None:
                    starttime = "null"
                if endtime is None:
                    endtime = "null"
                params["time"] = "%s,%s" % (starttime, endtime)
            else:
                params["time"] = _date_handler(time)

        if (
            interpolation is not None
            and interpolation in __allowedInterpolation
            and isinstance(interpolation, str)
        ):
            params["interpolation"] = interpolation

        if pixel_type is not None and pixel_type in __allowedPixelTypes:
            params["pixelType"] = pixel_type

        if no_data_interpretation in __allowedInterpolation:
            params["noDataInterpretation"] = no_data_interpretation

        if no_data is not None:
            params["noData"] = no_data

        if compression is not None and compression in __allowedCompression:
            params["compression"] = compression

        if band_ids is not None and isinstance(band_ids, list):
            params["bandIds"] = ",".join([str(x) for x in band_ids])

        if rendering_rule is not None:
            if "function_chain" in rendering_rule:
                params["renderingRule"] = rendering_rule["function_chain"]
            else:
                params["renderingRule"] = rendering_rule

        elif self._fn is not None:
            if not self._uses_gbl_function:
                params["renderingRule"] = self._fn
            else:
                _LOGGER.warning(
                    """Imagery layer object containing global functions in the function chain cannot be used for dynamic visualization.
                                   \nThe layer output must be saved as a new image service before it can be visualized. Use save() method of the layer object to create the processed output."""
                )
                return None

        if compression_tolerance is not None:
            params["compressionTolerance"] = compression_tolerance

        if compression_quality is not None:
            params["compressionQuality"] = compression_quality

        if adjust_aspect_ratio is not None:
            if adjust_aspect_ratio is True:
                params["adjustAspectRatio"] = "true"
            else:
                params["adjustAspectRatio"] = "false"

        params["f"] = f

        if lerc_version:
            params["lercVersion"] = lerc_version

        if slice_id is not None:
            params["sliceId"] = slice_id

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]
                params["Raster"] = self._uri
        if f == "json":
            return self._con.post(url, params, token=self._token, timeout=None)
        elif f == "image":
            if save_folder is not None and save_file is not None:
                return self._con.post(
                    url,
                    params,
                    out_folder=save_folder,
                    try_json=False,
                    file_name=save_file,
                    token=self._token,
                    timeout=None,
                )
            else:
                return self._con.post(
                    url,
                    params,
                    try_json=False,
                    force_bytes=True,
                    token=self._token,
                    timeout=None,
                )
        elif f == "kmz":
            return self._con.post(
                url,
                params,
                out_folder=save_folder,
                file_name=save_file,
                token=self._token,
                timeout=None,
            )

        elif f == "numpy_array":
            params["f"] = "image"
            params["format"] = "lerc"
            params["lercVersion"] = 2
            res = self._con.post(
                url,
                params,
                try_json=False,
                force_bytes=True,
                token=self._token,
            )

            try:
                import lerc
            except ImportError:
                raise ImportError(
                    "lerc not found. Install lerc to export image service as numpy array (pip install pylerc)"
                )

            if not isinstance(res, bytes):
                raise RuntimeError(res)
            result, data, valid_mask = lerc.decode(res)
            if result != 0:
                raise RuntimeError("decoding bytes from imagery service failed.")

            # transpose
            if "hasMultidimensions" in self.properties:
                is_multidimensional = self.properties.hasMultidimensions
            if is_multidimensional:
                if len(data.shape) == 2:
                    data = np.expand_dims(np.expand_dims(data, axis=2), axis=0)
                elif len(data) == 3:
                    if len(self.slices) == 1:
                        data = np.expand_dims(np.transpose(data, [1, 2, 0]), axis=0)
                    else:
                        data = np.expand_dims(np.transpose(data, [2, 0, 1]), axis=3)
                else:
                    assert len(data.shape) == 4
                    data = np.transpose(data, [3, 1, 2, 0])
            else:
                if len(data.shape) == 2:
                    data = np.expand_dims(data, axis=2)
                else:
                    data = np.transpose(data, axes=[1, 2, 0])

            return data

        else:
            print("Unsupported output format")

    # ----------------------------------------------------------------------
    def query(
        self,
        where: Optional[str] = None,
        out_fields: Union[str, list[str]] = "*",
        time_filter: Optional[
            Union[datetime.date, datetime.datetime, list[int], str]
        ] = None,
        geometry_filter: Optional[dict] = None,
        return_geometry: bool = True,
        return_ids_only: bool = False,
        return_count_only: bool = False,
        pixel_size: Optional[float] = None,
        order_by_fields: Optional[str] = None,
        return_distinct_values: Optional[bool] = None,
        out_statistics: Optional[Union[dict[str, Any], str]] = None,
        group_by_fields_for_statistics: Optional[Union[str, dict[str, Any]]] = None,
        out_sr: Optional[Union[dict[str, Any], int, SpatialReference]] = None,
        return_all_records: bool = False,
        object_ids: Optional[str] = None,
        multi_dimensional_def: Optional[dict[str, Any]] = None,
        result_offset: Optional[int] = None,
        result_record_count: Optional[int] = None,
        max_allowable_offset: Optional[int] = None,
        true_curves: bool = False,
        as_df: bool = False,
        raster_query: Optional[str] = None,
        return_extent_only: bool = False,
    ):
        """
        The ``query`` method queries an :class:`~arcgis.raster.ImageryLayer` by applying the filter specified by
        the user. The result of this operation is
        either a set of features or an array of raster IDs (if return_ids_only is set to True),
        count (if return_count_only is set to True), or a set of field statistics (if out_statistics is used).

        ==============================  ====================================================================
        **Parameter**                   **Description**
        ------------------------------  --------------------------------------------------------------------
        where                           Optional string. A where clause on this layer to filter the imagery
                                        layer by the selection sql statement. Any legal SQL where clause
                                        operating on the fields in the raster
        ------------------------------  --------------------------------------------------------------------
        out_fields                      Optional string. The attribute fields to return, comma-delimited
                                        list of field names.
        ------------------------------  --------------------------------------------------------------------
        time_filter                     Optional datetime.date, datetime.datetime or timestamp in
                                        milliseconds. The time instant or the time extent to query.

                                        Syntax: time_filter=<timeInstant>

                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time_filter=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp in milliseconds
        ------------------------------  --------------------------------------------------------------------
        geometry_filter                 Optional arcgis.geometry.filters. Spatial filter from
                                        arcgis.geometry.filters module to filter results by a spatial
                                        relationship with another geometry.
        ------------------------------  --------------------------------------------------------------------
        return_geometry                 Optional boolean. True means a geometry will be returned, else just
                                        the attributes
        ------------------------------  --------------------------------------------------------------------
        return_ids_only                 Optional boolean. False is default.  True means only OBJECTIDs will
                                        be returned
        ------------------------------  --------------------------------------------------------------------
        return_count_only               Optional boolean. If True, then an integer is returned only based on
                                        the sql statement
        ------------------------------  --------------------------------------------------------------------
        return_extent_only              optional boolean. If True, then only the extent is returned.
                                        This parameter is available from 10.8.1 onwards.
        ------------------------------  --------------------------------------------------------------------
        pixel_size                      optional dict or string. Query visible rasters at a given pixel size.
                                        If pixel_size is not specified, rasters at all resolutions can be
                                        queried.
                                        Syntax:
                                            - dictionary structure: pixel_size={point}
                                            - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                            - pixel_size={"x": 0.18, "y": 0.18}
                                            - pixel_size='0.18,0.18'
        ------------------------------  --------------------------------------------------------------------
        order_by_fields                 Optional string. Order results by one or more field names. Use ASC
                                        or DESC for ascending or descending order, respectively.
        ------------------------------  --------------------------------------------------------------------
        return_distinct_values          Optional boolean. If true, returns distinct values based on the
                                        fields specified in out_fields. This parameter applies only if the
                                        supportsAdvancedQueries property of the image layer is true.
        ------------------------------  --------------------------------------------------------------------
        out_statistics                  Optional dict or string. The definitions for one or more field-based
                                        statistics to be calculated.
        ------------------------------  --------------------------------------------------------------------
        group_by_fields_for_statistics  Optional dict/string. One or more field names using the
                                        values that need to be grouped for calculating the
                                        statistics.
        ------------------------------  --------------------------------------------------------------------
        out_sr                          Optional dict, SpatialReference. If the returning geometry needs to
                                        be in a different spatial reference, provide the function with the
                                        desired WKID.
        ------------------------------  --------------------------------------------------------------------
        return_all_records              Optional boolean. If True(default) all records will be returned.
                                        False means only the limit of records will be returned.
        ------------------------------  --------------------------------------------------------------------
        object_ids                      Optional string. The object IDs of this raster catalog to be
                                        queried. When this parameter is specified, any other filter
                                        parameters (including where) are ignored.
                                        When this parameter is specified, setting return_ids_only=true is
                                        invalid.
                                        Syntax: objectIds=<objectId1>, <objectId2>
                                        Example: objectIds=37, 462
        ------------------------------  --------------------------------------------------------------------
        multi_dimensional_def           Optional dict. The filters defined by multiple dimensional
                                        definitions.
        ------------------------------  --------------------------------------------------------------------
        result_offset                   Optional integer. This option fetches query results by skipping a
                                        specified number of records. The query results start from the next
                                        record (i.e., resultOffset + 1). The Default value is None.
        ------------------------------  --------------------------------------------------------------------
        result_record_count             Optional integer. This option fetches query results up to the
                                        resultRecordCount specified. When resultOffset is specified and this
                                        parameter is not, image layer defaults to maxRecordCount. The
                                        maximum value for this parameter is the value of the layer's
                                        maxRecordCount property.
                                        max_allowable_offset - This option can be used to specify the
                                        max_allowable_offset to be used for generalizing geometries returned
                                        by the query operation. The max_allowable_offset is in the units of
                                        the out_sr. If outSR is not specified, max_allowable_offset is
                                        assumed to be in the unit of the spatial reference of the Layer.
        ------------------------------  --------------------------------------------------------------------
        true_curves                     Optional boolean. If true, returns true curves in output geometries,
                                        otherwise curves get converted to densified polylines or polygons.
        ------------------------------  --------------------------------------------------------------------
        as_df                           Optional boolean. Returns the query result as a dataframe object
        ------------------------------  --------------------------------------------------------------------
        raster_query                    Optional string.  Make query based on key properties of each
                                        raster catalog item. Any legal SQL where clause operating on the
                                        key properties of raster catalog items is allowed.

                                        Example: LANDSAT_WRS_PATH >= 150 AND LANDSAT_WRS_PATH<= 165

                                        This option was added at 10.8.1.
        ==============================  ====================================================================

        :return: A :class:`~arcgis.features.FeatureSet` containing the footprints (features) matching the query when
                  return_geometry is ``True``, else a dictionary containing the expected return
                  type.

        .. code-block:: python

            # Usage Example

            img_lyr = gis.content.search("my_image_service", item_type="Imagery Layer")[0].layers[0]
            search_res = img_lyr.query(where="OBJECTID=1")

        """

        def _feat_to_row(feature):
            from arcgis.geometry import Geometry

            attribute = {}
            attribute.update(feature["attributes"])
            attribute["SHAPE"] = Geometry(feature["geometry"])
            return attribute

        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )

        params = {
            "f": "json",
            "outFields": out_fields,
            "returnGeometry": return_geometry,
            "returnIdsOnly": return_ids_only,
            "returnCountOnly": return_count_only,
            "returnExtentOnly": return_extent_only,
        }
        if object_ids:
            params["objectIds"] = object_ids
        if multi_dimensional_def:
            params["multidimensionalDefinition"] = multi_dimensional_def
        if result_offset:
            params["resultOffset"] = result_offset
        if result_record_count:
            params["resultRecordCount"] = result_record_count
        if max_allowable_offset:
            params["maxAllowableOffset"] = max_allowable_offset
        if true_curves:
            params["returnTrueCurves"] = true_curves
        if where is not None:
            params["where"] = where
        elif self._where_clause is not None:
            params["where"] = self._where_clause
        else:
            params["where"] = "1=1"

        if not group_by_fields_for_statistics is None:
            params["groupByFieldsForStatistics"] = group_by_fields_for_statistics
        if not out_statistics is None:
            params["outStatistics"] = out_statistics

        if self._temporal_filter is not None:
            time_filter = self._temporal_filter

        if time_filter is not None:
            if type(time_filter) is list:
                starttime = _date_handler(time_filter[0])
                endtime = _date_handler(time_filter[1])
                if starttime is None:
                    starttime = "null"
                if endtime is None:
                    endtime = "null"
                params["time"] = "%s,%s" % (starttime, endtime)
            else:
                params["time"] = _date_handler(time_filter)

        if self._spatial_filter is not None:
            geometry_filter = self._spatial_filter

        if not geometry_filter is None and isinstance(geometry_filter, dict):
            gf = geometry_filter
            params["geometry"] = gf["geometry"]
            params["geometryType"] = gf["geometryType"]
            params["spatialRel"] = gf["spatialRel"]
            if "inSR" in gf:
                params["inSR"] = gf["inSR"]

        if pixel_size is not None:
            params["pixelSize"] = pixel_size
        if order_by_fields is not None:
            params["orderByFields"] = order_by_fields
        if return_distinct_values is not None:
            params["returnDistinctValues"] = return_distinct_values
        if out_sr is not None:
            params["outSR"] = out_sr
        if raster_query is not None:
            params["rasterQuery"] = raster_query

        url = self._url + "/query"
        if return_all_records and return_count_only == False:
            oids = self.query(
                where=where,
                out_fields=out_fields,
                time_filter=time_filter,
                geometry_filter=geometry_filter,
                return_geometry=False,
                return_ids_only=True,
            )
            count = self.query(
                where=where,
                geometry_filter=geometry_filter,
                time_filter=time_filter,
                return_count_only=True,
            )
            if count > self.properties.maxRecordCount:
                n = count // self.properties.maxRecordCount
                if (count % self.properties.maxRecordCount) > 0:
                    n += 1
                records = None
                for i in range(n):
                    oid_sub = [
                        str(o)
                        for o in oids["objectIds"][
                            self.properties.maxRecordCount
                            * i : self.properties.maxRecordCount
                            * (i + 1)
                        ]
                    ]
                    oid_joined = ",".join(oid_sub)
                    sql = "{name} in ({oids})".format(
                        name=oids["objectIdFieldName"], oids=oid_joined
                    )
                    params["where"] = sql
                    if records is None:
                        records = self._con.post(
                            path=url,
                            postdata=params,
                            token=self._token,
                            timeout=None,
                        )

                    else:
                        res = self._con.post(
                            path=url,
                            postdata=params,
                            token=self._token,
                            timeout=None,
                        )
                        records["features"].extend(res["features"])
                result = records
            else:
                result = self._con.post(
                    path=url,
                    postdata=params,
                    token=self._token,
                    timeout=None,
                )
        else:
            result = self._con.post(
                path=url, postdata=params, token=self._token, timeout=None
            )
        if "error" in result:
            raise ValueError(result)

        if return_count_only:
            return result["count"]
        elif return_ids_only:
            return result
        elif return_extent_only:
            return result
        elif return_geometry:
            if as_df:
                if "features" in result:
                    import pandas as pd

                    rows = [_feat_to_row(feat) for feat in result["features"]]
                    df = pd.DataFrame(rows)
                    df.spatial.name
                    return df
                return result
            else:
                return FeatureSet.from_dict(result)
        else:
            return result

    # ----------------------------------------------------------------------
    def get_download_info(
        self,
        raster_ids: str,
        polygon: Optional[Polygon] = None,
        extent: Optional[str] = None,
        out_format: Optional[str] = None,
    ):
        """
        The ``get_download_info`` method returns information (the file ID)
        that can be used to download the raw raster files that are
        associated with a specified set of rasters in the raster catalog.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        raster_ids            Required string. A comma-separated list of raster IDs whose files
                              are to be downloaded.
        -----------------     --------------------------------------------------------------------
        polygon               Optional :class:`~arcgis.geometry.Polygon` object. The geometry to apply for clipping
        -----------------     --------------------------------------------------------------------
        extent                Optional string. The geometry to apply for clipping
                              example: "-104,35.6,-94.32,41"
        -----------------     --------------------------------------------------------------------
        out_format            Optional string. The format of the rasters returned. If not
                              specified, the rasters will be in their native format.
                              The format applies when the clip geometry is also specified, and the
                              format will be honored only when the raster is clipped.

                              To force the Download Rasters operation to convert source images to
                              a different format, append :Conversion after format string.
                              Valid formats include: TIFF, Imagine Image, JPEG, BIL, BSQ, BIP,
                              ENVI, JP2, GIF, BMP, and PNG.
                              Example: out_format='TIFF'
        =================     ====================================================================

        :return: A dictionary
        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )

        url = "%s/download" % self._url
        if self.properties["capabilities"].lower().find("download") == -1:
            return
        params = {
            "f": "json",
            "rasterIds": raster_ids,
        }
        if polygon is not None:
            params["geometry"] = polygon
            params["geometryType"] = "esriGeometryPolygon"
        if extent is not None:
            params["geometry"] = extent
            params["geometryType"] = "esriGeometryEnvelope"
        if out_format is not None:
            params["format"] = out_format
        return self._con.post(path=url, postdata=params, timeout=None)

    # ----------------------------------------------------------------------
    def get_raster_file(
        self, download_info: dict[str, Any], out_folder: Optional[str] = None
    ):
        """
        The ``get_raster_file`` method returns a list of raw raster files. The
        ``download_info`` is obtained by using the :attr:`~arcgis.raster.ImageryLayer.get_download_info` operation.


        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        download_info         Required dictionary. This is derived from the
                              :attr:`~arcgis.raster.ImageryLayer.get_download_info` method.
        -----------------     --------------------------------------------------------------------
        out_folder            Optional string. Path to the file save location. If the value is
                              None, the OS temporary directory is used.
        =================     ====================================================================

        :return: A list of the files downloaded
        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )

        import os
        import tempfile

        cap = self.properties["capabilities"].lower()
        if cap.find("download") == -1 or cap.find("catalog") == -1:
            return None

        if out_folder is None:
            out_folder = tempfile.gettempdir()
        if out_folder and os.path.isdir(out_folder) == False:
            os.makedirs(out_folder, exist_ok=True)
        url = "%s/file" % self._url
        params = {"f": "json"}
        p = []
        files = []
        if "rasterFiles" in download_info:
            for f in download_info["rasterFiles"]:
                params = {"f": "json"}
                params["id"] = f["id"]
                for rid in f["rasterIds"]:
                    params["rasterId"] = rid
                    files.append(
                        self._con.get(
                            path=url,
                            params=params,
                            out_folder=out_folder,
                            file_name=os.path.basename(params["id"]),
                        )
                    )
                del f
        return files

    # ----------------------------------------------------------------------
    def compute_pixel_location(
        self,
        raster_id: int,
        geometries: list[Geometry],
        spatial_reference: Union[str, dict[str, Any], SpatialReference],
    ):
        """

        The ``compute_pixel_location`` method calculates pixel location
        in column and row on specific raster catalog item based on given input geometry.

        .. note::
            A prerequisite is that the raster catalog item has valid ``icsToPixel`` resource.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        raster_id             Required integer. Specifies the objectId of image service's raster
                              catalog. This integer rasterId number will determine which raster's
                              image coordinate system will be used during the calculation and
                              which raster does the column and row of results represent.
        -----------------     --------------------------------------------------------------------
        geometries            Required list of geometries for computing pixel locations.
                              All geometries in this list should be of the type defined by ``geometryType``.

        -----------------     --------------------------------------------------------------------
        spatial_reference     Required string, dictionary,
                              This specifies the spatial reference of the Geometries parameter above.
                              It can accept a multitudes of values.  These can be a WKID,
                              image coordinate system (ICSID), or image coordinate system in json/dict format.
                              Additionally the :class:`~arcgis.geometry.SpatialReference` object is also a
                              valid entry.

                              .. note::

                                    An image coordinate system ID can be specified
                                    using 0:icsid; for example, 0:64. The extra 0: is used to avoid
                                    conflicts with wkid
        =================     ====================================================================

        :return: A dictionary. The result of this operation includes x and y values for the column
                 and row of each input geometry. It also includes a z value for the height at given
                 location based on elevation info that the catalog raster item has.


        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        url = "%s/computePixelLocation" % self._url
        params = {
            "f": "json",
            "rasterId": raster_id,
            "geometries": geometries,
            "spatialReference": spatial_reference,
        }
        return self._con.post(path=url, postdata=params, timeout=None)

    def slices(self, muldidef: Optional[list[str]] = None):
        """
        The ``slices`` method is used to query slice ID and multidimensional information of a multidimensional image
        service.

        .. note::
            The ``slices`` operation is available in ArcGIS Image Server 10.8.1 and higher.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        muldidef              Optional list. Multidimensional definition used for querying
                              dimensional slices of the input image service.

                              See the
                              `Multidimensional definition <https://developers.arcgis.com/documentation/common-data-types/multidimensional-definition.htm>`_
                              page in the ArcGIS REST API documentation for more information.
        =================     ====================================================================

        :return: A dictionary containing the list of slice definitions.

        .. code-block:: python

            # Usage Example: This example returns the slice ID and multidimensional information of slices with
            # "salinity" variable at "StdZ" dimension with a value of "-5000".

            multidimensional_definition = [{"variableName":"salinity","dimensionName":"StdZ","values":[-5000]}]
            multidimensional_lyr_input.slices(multidimensional_definition)

        """
        url = self._url + "/slices"

        params = {"f": "json"}

        if muldidef is not None:
            params["multidimensionalDefinition"] = muldidef

        if self._datastore_raster:
            params["Raster"] = self._uri

        return self._con.post(path=url, postdata=params, timeout=None)

    def statistics(
        self,
        variable: Optional[str] = None,
        rendering_rule: Optional[dict] = None,
    ):
        """
        The ``statistics`` method retrieves the statistics of the raster.

        .. note::
            The ``statistics`` method is available in ArcGIS Image Server 10.8.1 and higher.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        variable              Optional string. For an image service that has multidimensional
                              information, this parameter can be used to request statistics for
                              each variable. If not specified, it will return statistics for the
                              whole image service. Eligible variable names can be queried from
                              multidimensional_info property of the Imagery Layer object.
        -----------------     --------------------------------------------------------------------
        rendering_rule        Optional dictionary. Specifies the rendering rule for how the requested image should be rendered.

                              In the context of accessing image service statistics resource,
                              this parameter is used to retrieve statistics info in attached
                              predefined raster function templates (inside a StatisticsHistogram function).

                              This parameter is available from 10.9.1
        =================     ====================================================================

        :return: A dictionary containing the statistics.

        .. code-block:: python

            # Usage Example: This example returns the statistics of an Imagery Layer object.

            lyr_input.statistics()



        """
        url = self._url + "/statistics"

        params = {"f": "json"}

        if variable is not None:
            params["variable"] = variable

        if rendering_rule is not None:
            params["renderingRule"] = rendering_rule
        elif self._fn is not None:
            params["renderingRule"] = self._fn

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]
                params["Raster"] = self._uri

        return self._con.post(
            path=url, postdata=params, token=self._token, timeout=None
        )

    def get_histograms(
        self,
        variable: Optional[str] = None,
        rendering_rule: Optional[dict] = None,
    ):
        """
        The ``get_histograms`` method retrieves the histograms of each band in the :class:`~arcgis.raster.ImageryLayer`
        as a list of dictionaries corresponding to each band.

        .. note::
            * ``get_histograms`` can return a histogram for each variable if used with multidimensional ImageryLayer \
            object by specifying value for variable parameter.
            * If histogram is not found, returns None. In this case, call the :attr:`~arcgis.raster.ImageryLayer.compute_histograms` method.
            * ``get_histograms`` is an enhanced version of the :attr:`~arcgis.raster.ImageryLayer.histograms` property on the ``ImageryLayer`` class with additional variable parameter.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        variable              Optional string. For an image service that has multidimensional
                              information, this parameter can be used to request histograms for
                              each variable. It will return histograms for the whole ImageryLayer
                              if not specified.
                              This parameter is available from 10.8.1
        -----------------     --------------------------------------------------------------------
        rendering_rule        Optional dictionary. Specifies the rendering rule for how the requested image should be rendered.

                              In the context of accessing image service histograms resource,
                              this parameter is used to retrieve histograms info in attached
                              predefined raster function templates (inside a StatisticsHistogram function).

                              This parameter is available from 10.9.1
        =================     ====================================================================

        :return: A list

        Example Structure of the return value:

            | [{"size":256,
            |   "min":560,
            |   "max":24568,
            |   "counts": [10,99,56,42200,125,....]}] # length of this list corresponds "size"

        .. code-block:: python
            
            # Example Usage
            my_hist = imagery_layer.get_histograms(variable="water_temp")

        """
        if self.properties.hasHistograms:
            # proceed
            url = self._url + "/histograms"
            params = {"f": "json"}
            if variable is not None:
                params["variable"] = variable
            if rendering_rule is not None:
                params["renderingRule"] = rendering_rule
            elif self._fn is not None:
                params["renderingRule"] = self._fn

            if self._datastore_raster:
                params["Raster"] = self._uri
                if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                    del params["renderingRule"]
                    params["Raster"] = self._uri
            hist_return = self._con.post(url, params, token=self._token, timeout=None)

            # process this into a dict
            return hist_return["histograms"]
        else:
            return None

    # ----------------------------------------------------------------------
    def thumbnail(self, out_path: Optional[str] = None):
        """
        The ``thumbnail`` method downloads the image service's thumbnail image to local disk.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        out_path              Optional string. Represents the path to which the image needs to be downloaded.
        =================     ====================================================================

        .. code-block:: python

            # Usage Example: This example returns the thumbnail of an Imagery Layer object.

            lyr_input.thumbnail()

        :return: string representing path to the downloaded thumbnail.

        """
        if out_path is None:
            out_path = tempfile.gettempdir()
        url = "{url}/info/thumbnail".format(url=self._url)
        params = {"f": "json"}
        if self._datastore_raster:
            params["Raster"] = self._uri
        return self._con.get(
            url, params, out_folder=out_path, file_name="thumbnail.png"
        )

    # ----------------------------------------------------------------------
    def _add_rasters(
        self,
        raster_type,
        item_ids=None,
        service_url=None,
        compute_statistics=False,
        build_pyramids=False,
        build_thumbnail=False,
        minimum_cell_size_factor=None,
        maximum_cell_size_factor=None,
        attributes=None,
        geodata_transforms=None,
        geodata_transform_apply_method="esriGeodataTransformApplyAppend",
    ):
        """
        This operation is supported at 10.1 and later.
        The Add Rasters operation is performed on an image layer method.
        The Add Rasters operation adds new rasters to an image layer
        (POST only).
        The added rasters can either be uploaded items, using the item_ids
        parameter, or published services, using the service_url parameter.
        If item_ids is specified, uploaded rasters are copied to the image
        Layer's dynamic image workspace location; if the service_url is
        specified, the image layer adds the URL to the mosaic dataset no
        raster files are copied. The service_url is required input for the
        following raster types: Image Layer, Map Service, WCS, and WMS.

        Inputs:

        item_ids - The upload items (raster files) to be added. Either
         item_ids or service_url is needed to perform this operation.
            Syntax: item_ids=<itemId1>,<itemId2>
            Example: item_ids=ib740c7bb-e5d0-4156-9cea-12fa7d3a472c,
                             ib740c7bb-e2d0-4106-9fea-12fa7d3a482c
        service_url - The URL of the service to be added. The image layer
         will add this URL to the mosaic dataset. Either item_ids or
         service_url is needed to perform this operation. The service URL is
         required for the following raster types: Image Layer, Map
         Service, WCS, and WMS.
            Example: service_url=http://myserver/arcgis/services/Portland/ImageServer
        raster_type - The type of raster files being added. Raster types
         define the metadata and processing template for raster files to be
         added. Allowed values are listed in image layer resource.
            Example: Raster Dataset,CADRG/ECRG,CIB,DTED,Image Layer,Map Service,NITF,WCS,WMS
        compute_statistics - If true, statistics for the rasters will be
         computed. The default is false.
            Values: false,true
        build_pyramids - If true, builds pyramids for the rasters. The
         default is false.
                Values: false,true
        build_thumbnail	 - If true, generates a thumbnail for the rasters.
         The default is false.
                Values: false,true
        minimum_cell_size_factor - The factor (times raster resolution) used
         to populate the MinPS field (maximum cell size above which the
         raster is visible).
                Syntax: minimum_cell_size_factor=<minimum_cell_size_factor>
                Example: minimum_cell_size_factor=0.1
        maximum_cell_size_factor - The factor (times raster resolution) used
         to populate MaxPS field (maximum cell size below which raster is
         visible).
                Syntax: maximum_cell_size_factor=<maximum_cell_size_factor>
                Example: maximum_cell_size_factor=10
        attributes - Any attribute for the added rasters.
                Syntax:
                {
                  "<name1>" : <value1>,
                  "<name2>" : <value2>
                }
                Example:
                {
                  "MinPS": 0,
                  "MaxPS": 20;
                  "Year" : 2002,
                  "State" : "Florida"
                }
        geodata_transforms - The geodata transformations applied on the
         added rasters. A geodata transformation is a mathematical model
         that performs a geometric transformation on a raster; it defines
         how the pixels will be transformed when displayed or accessed.
         Polynomial, projective, identity, and other transformations are
         available. The geodata transformations are applied to the dataset
         that is added.
                Syntax:
                [
                {
                  "geodataTransform" : "<geodataTransformName1>",
                  "geodataTransformArguments" : {<geodataTransformArguments1>}
                  },
                  {
                  "geodataTransform" : "<geodataTransformName2>",
                  "geodataTransformArguments" : {<geodataTransformArguments2>}
                  }
                ]
         The syntax of the geodataTransformArguments property varies based
         on the specified geodataTransform name. See Geodata Transformations
         documentation for more details.
        geodata_transform_apply_method - This parameter defines how to apply
         the provided geodataTransform. The default is
         esriGeodataTransformApplyAppend.
                Values: esriGeodataTransformApplyAppend |
                esriGeodataTransformApplyReplace |
                esriGeodataTransformApplyOverwrite
        """
        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )

        url = self._url + "/add"
        params = {"f": "json"}
        if item_ids is None and service_url is None:
            raise Exception("An itemId or service_url must be provided")

        if isinstance(item_ids, (list, tuple)):
            item_ids = ",".join(item_ids)

        params["geodataTransformApplyMethod"] = geodata_transform_apply_method
        params["rasterType"] = raster_type
        params["buildPyramids"] = build_pyramids
        params["buildThumbnail"] = build_thumbnail
        params["minimumCellSizeFactor"] = minimum_cell_size_factor
        params["computeStatistics"] = compute_statistics
        params["maximumCellSizeFactor"] = maximum_cell_size_factor
        params["attributes"] = attributes
        params["geodataTransforms"] = geodata_transforms

        if not item_ids is None:
            params["itemIds"] = item_ids
        if not service_url is None:
            params["serviceUrl"] = service_url
        return self._con.post(url, params, token=self._token, timeout=None)

    # ----------------------------------------------------------------------
    def _delete_rasters(self, raster_ids):
        """
        The Delete Rasters operation deletes one or more rasters in an image layer.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        raster_ids            Required string. The object IDs of a raster catalog items to be
                              removed. This is a comma seperated string.
                              example 1: raster_ids='1,2,3,4' # Multiple IDs
                              example 2: raster_ids='10' # single ID
        =================     ====================================================================

        :return: dictionary
        """
        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        params = {"f": "json", "rasterIds": raster_ids}
        url = "%s/delete" % self._url
        return self._con.post(path=url, postdata=params, timeout=None)

    # ----------------------------------------------------------------------
    def _update_raster(
        self,
        raster_id,
        files=None,
        item_ids=None,
        service_url=None,
        compute_statistics=False,
        build_pyramids=False,
        build_thumbnail=False,
        minimum_cell_size_factor=None,
        maximum_cell_size_factor=None,
        attributes=None,
        footprint=None,
        geodata_transforms=None,
        apply_method="esriGeodataTransformApplyAppend",
    ):
        """
        The Update Raster operation updates rasters (attributes and
        footprints, or replaces existing raster files) in an image layer.
        In most cases, this operation is used to update attributes or
        footprints of existing rasters in an image layer. In cases where
        the original raster needs to be replaced, the new raster can either
        be items uploaded using the items parameter or URLs of published
        services using the serviceUrl parameter.

        =========================       ====================================================================
        **Parameter**                    **Description**
        -------------------------       --------------------------------------------------------------------
        raster_ids                      Required integer. The object IDs of a raster catalog items to be
                                        updated.
        -------------------------       --------------------------------------------------------------------
        files                           Optional list. Local source location to the raster to replace the
                                        dataset with.
                                        Example: [r"<path>\data.tiff"]
        -------------------------       --------------------------------------------------------------------
        item_ids                        Optional string.  The uploaded items (raster files) being used to
                                        replace existing raster.
        -------------------------       --------------------------------------------------------------------
        service_url                     Optional string. The URL of the layer to be uploaded to replace
                                        existing raster data. The image layer will add this URL to the
                                        mosaic dataset. The serviceUrl is required for the following raster
                                        types: Image Layer, Map Service, WCS, and WMS.
        -------------------------       --------------------------------------------------------------------
        compute_statistics              Optional bool. If true, statistics for the uploaded raster will be computed.
                                        The default is false.
        -------------------------       --------------------------------------------------------------------
        build_pyramids                  Optional boolean. If true, builds pyramids for the uploaded raster.
                                        The default is false.
        -------------------------       --------------------------------------------------------------------
        build_thumbnail                 Optional boolean. If true, generates a thumbnail for the uploaded
                                        raster. The default is false.
        -------------------------       --------------------------------------------------------------------
        minimum_cell_size_factor        Optional float. The factor (times raster resolution) used to
                                        populate MinPS field (minimum cell size above which raster is visible).
        -------------------------       --------------------------------------------------------------------
        maximum_cell_size_factor        Optional float. The factor (times raster resolution) used to
                                        populate MaxPS field (maximum cell size below which raster is visible).
        -------------------------       --------------------------------------------------------------------
        footprint                       Optional Polygon.  A JSON 2D polygon object that defines the
                                        footprint of the raster. If the spatial reference is not defined, it
                                        will default to the image layer's spatial reference.
        -------------------------       --------------------------------------------------------------------
        attributes                      Optional dictionary.  Any attribute for the uploaded raster.
        -------------------------       --------------------------------------------------------------------
        geodata_transforms              Optional string. The geodata transformations applied on the updated
                                        rasters. A geodata transformation is a mathematical model that
                                        performs geometric transformation on a raster. It defines how the
                                        pixels will be transformed when displayed or accessed, such as
                                        polynomial, projective, or identity transformations. The geodata
                                        transformations will be applied to the updated dataset.
        -------------------------       --------------------------------------------------------------------
        apply_method                    Optional string. Defines how to apply the provided geodataTransform.
                                        The default is esriGeodataTransformApplyAppend.
                                        Values: esriGeodataTransformApplyAppend,
                                                esriGeodataTransformApplyReplace,
                                                esriGeodataTransformApplyOverwrite
        =========================       ====================================================================

        :return: dictionary
        """
        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        url = "%s/update" % self._url
        ids = []
        if files:
            for f in files:
                u = self._upload(fp=f)
                if u:
                    ids.append(u)
            item_ids = ",".join(ids)
        params = {
            "f": "json",
            "rasterId": raster_id,
        }
        if item_ids is not None:
            params["itemIds"] = item_ids
        if service_url is not None:
            params["serviceUrl"] = service_url
        if compute_statistics is not None:
            params["computeStatistics"] = compute_statistics
        if build_pyramids is not None:
            params["buildPyramids"] = build_pyramids
        if build_thumbnail is not None:
            params["buildThumbnail"] = build_thumbnail
        if minimum_cell_size_factor is not None:
            params["minimumCellSizeFactor"] = minimum_cell_size_factor
        if maximum_cell_size_factor is not None:
            params["maximumCellSizeFactor"] = maximum_cell_size_factor
        if footprint is not None:
            params["footprint"] = footprint
        if attributes is not None:
            params["attributes"] = attributes
        if geodata_transforms is not None:
            params["geodataTransforms"] = geodata_transforms
        if apply_method is not None:
            params["geodataTransformApplyMethod"] = apply_method
        return self._con.post(path=url, postdata=params, timeout=None)

    # ----------------------------------------------------------------------
    def _upload(self, fp, description=None):
        """uploads a file to the image layer"""
        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        url = "%s/uploads/upload" % self._url
        params = {"f": "json"}
        if description:
            params["description"] = description
        files = {"file": fp}
        res = self._con.post(path=url, postdata=params, files=files, timeout=None)
        if "success" in res and res["success"]:
            return res["item"]["itemID"]
        return None

    # ----------------------------------------------------------------------
    def compute_stats_and_histograms(
        self,
        geometry: Union[Geometry, Polygon, Envelope],
        mosaic_rule: Optional[dict[str, str]] = None,
        rendering_rule: Optional[dict[str, str]] = None,
        pixel_size: Optional[Union[dict[str, float], str]] = None,
        time: Optional[Union[str, list[int], datetime.date, datetime.datetime]] = None,
        process_as_multidimensional: bool = False,
    ):
        """
        The ``compute_stats_and_histograms`` method computes both statistics and histograms for the
        :class:`~arcgis.raster.ImageryLayer` object from the given extent.

        ============================    ====================================================================
        **Parameter**                    **Description**
        ----------------------------    --------------------------------------------------------------------
        geometry                        Required :class:`~arcgis.geometry.Geometry` (:class:`~arcgis.geometry.Polygon`
                                        or :class:`~arcgis.geometry.Envelope`).
                                        A geometry that defines the geometry within which the statistics and histograms are computed.
        ----------------------------    --------------------------------------------------------------------
        mosaic_rule                     Optional dictionary.  Specifies the mosaic rule when defining how
                                        individual images should be mosaicked. When a mosaic rule is not
                                        specified, the default mosaic rule of the image layer will be used
                                        (as advertised in the root resource: defaultMosaicMethod,
                                        mosaicOperator, sortField, sortValue).
        ----------------------------    --------------------------------------------------------------------
        rendering_rule                  Optional dictionary. Specifies the rendering rule for how the
                                        requested image should be rendered.
        ----------------------------    --------------------------------------------------------------------
        pixel_size                      Optional string or dict. The pixel level being used (or the
                                        resolution being looked at). If pixel size is not specified, then
                                        pixel_size will default to the base resolution of the dataset. The
                                        raster at the specified pixel size in the mosaic dataset will be
                                        used for histogram calculation.
                                        
                                        Syntax:
                                          - dictionary structure: pixel_size={point}
                                          - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                          - pixel_size={"x": 0.18, "y": 0.18}
                                          - pixel_size='0.18,0.18'
        ----------------------------    --------------------------------------------------------------------
        time                            Optional datetime.date, datetime.datetime or timestamp string. The
                                        time instant or the time extent to compute statistics and histograms.
                                        Time instant specified as datetime.date, datetime.datetime or
                                        timestamp in milliseconds since epoch
                                        Syntax: time=<timeInstant>
                                        
                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp
                                        
                                        .. note::
                                            The parameter was added at 10.8.
        ----------------------------    --------------------------------------------------------------------
        process_as_multidimensional     Optional boolean. Specifies whether to process the image service as 
                                        a multidimensional image service.
                                        
                                            - ``False`` - Statistics and histograms of pixel values from only the \
                                                      first slice is computed. This is the default.
                                            - ``True`` - The image service is treated as a multidimensional raster, \
                                                     and statistics and histograms of pixel values from all selected \
                                                     slices are computed.
                                        
                                        .. note::
                                            Added at 10.9 for image services which use
                                            ``ArcObjects11`` or ``ArcObjectsRasterRendering``
                                            as the service provider.
        ============================    ====================================================================

        :return: dictionary

        .. code-block:: python

            # Usage Example 1: Compute the stats and histograms in the specified area of interest for a time instant.

            aoi = {
                    "spatialReference": {"wkid": 32610},
                    "xmax": 725000,
                    "xmin": 720000,
                    "ymax": 4300000,
                    "ymin": 4250000
                  }

            aoi_geometry = Geometry(aoi)

            comp_stats_hist_01 = img_lyr.compute_stats_and_histograms(geometry=aoi,
                                                                      rendering_rule={"rasterFunction":None},
                                                                      time="1326650400000")

        .. code-block:: python

            # Usage Example 2: Compute the stats and histograms in the specified area of interest for a time extent.

            aoi = {
                    "spatialReference": {"wkid": 32610},
                    "xmax": 725000,
                    "xmin": 720000,
                    "ymax": 4300000,
                    "ymin": 4250000
                  }

            aoi_geometry = Geometry(aoi)

            # If the datetime object is not in the UTC timezone, the API will internally convert it to the UTC timezone.
            start = datetime.datetime(2012,1,15,18,0,0, tzinfo=datetime.timezone.utc)
            end = datetime.datetime(2012,1,15,21,0,0, tzinfo=datetime.timezone.utc)
            
            comp_stats_hist_02 = img_lyr.compute_stats_and_histograms(geometry=aoi,
                                                                      rendering_rule={"rasterFunction":None},
                                                                      time=[start,end])

        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)

            except:
                raise RuntimeError(
                    "Failed to perform compute statistics and histograms operation on the TilesOnly service"
                )

        import datetime

        url = "%s/computeStatisticsHistograms" % self._url
        from arcgis.geometry import Polygon

        if isinstance(geometry, Polygon):
            gt = "esriGeometryPolygon"
        else:
            gt = "esriGeometryEnvelope"
        params = {"f": "json", "geometry": geometry, "geometryType": gt}
        if pixel_size is not None:
            params["pixelSize"] = pixel_size
        if rendering_rule is not None:
            params["renderingRule"] = rendering_rule
        elif self._fn is not None:
            params["renderingRule"] = self._fn
        if mosaic_rule is not None:
            params["mosaicRule"] = mosaic_rule
        elif self._mosaic_rule is not None:
            params["mosaicRule"] = self._mosaic_rule

        from ._util import _set_time_param

        if time is not None:
            params["time"] = _set_time_param(time)

        if isinstance(process_as_multidimensional, bool):
            params["processAsMultidimensional"] = process_as_multidimensional

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]

        return self._con.post(path=url, postdata=params, timeout=None)

    # ----------------------------------------------------------------------
    def compute_tie_points(self, raster_id: int, geodata_transforms: dict[str, Any]):
        """
        The ``compute_tie_points`` method retrieves tie points that can be used
        to match the source image to the reference image. The reference
        image is configured by the image layer publisher.

        .. note::
            For more information, see `Fundamentals for georeferencing` a raster dataset.

        ==================    ====================================================================
        **Parameter**          **Description**
        ------------------    --------------------------------------------------------------------
        raster_id             Required integer. Source raster ID.
        ------------------    --------------------------------------------------------------------
        geodata_transforms    Required dictionary. The geodata transformation that provides a
                              rough fit of the source image to the reference image.
                              For example, a
                              first order polynomial transformation that fits the source image to
                              the expected location.
        ==================    ====================================================================

        :return: dictionary
        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        url = "%s/computeTiePoints" % self._url
        params = {
            "f": "json",
            "rasterId": raster_id,
            "geodataTransform": geodata_transforms,
        }
        return self._con.post(path=url, postdata=params, timeout=None)

    # ----------------------------------------------------------------------
    def legend(
        self,
        band_ids: Optional[str] = None,
        rendering_rule: Optional[dict[str, Any]] = None,
        as_html: bool = False,
    ):
        """
        The ``legend`` method retrieves the legend information. The information includes the symbol images and labels for
        each symbol.

        .. note::
            Each symbol is generally an image of size 20 x 20
            pixels at 96 DPI. Symbol sizes may vary slightly for some renderer
            types (e.g., Vector Field Renderer). Additional information in the
            legend response will include the layer name, layer type, label,
            and content type.

        The legend symbols include the base64 encoded ``imageData``. The
        symbols returned in response to an image layer legend request
        reflect the default renderer of the image layer or the renderer
        defined by the rendering rule and band Ids.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        band_ids              Optional string. If there are multiple bands, you can specify a
                              single band, or you can change the band combination (red, green,
                              blue) by specifying the band ID. Band ID is 0 based.
                              Example: bandIds=2,1,0
        -----------------     --------------------------------------------------------------------
        rendering_rule        Optional dictionary. Specifies the rendering rule for how the
                              requested image should be rendered.
        -----------------     --------------------------------------------------------------------
        as_html               Optional bool. Returns an HTML table if True
        =================     ====================================================================

        :return: A legend as a dictionary by default, or as an HTML table if as_html is True
        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)

            except:
                raise RuntimeError(
                    "Failed to perform legend operation on the TilesOnly service"
                )

        url = "%s/legend" % self._url
        params = {"f": "json"}
        if band_ids is not None:
            params["bandIds"] = band_ids
        if rendering_rule is not None:
            params["renderingRule"] = rendering_rule
        elif self._fn is not None:
            params["renderingRule"] = self._fn

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]

        legend = self._con.post(path=url, postdata=params, timeout=None)
        if as_html is True:
            legend_type = legend["layers"][0]["legendType"]
            if legend_type.lower() == "stretched":
                table_and_cell_style = (
                    "border:none!important; background-color: #ffffff;"
                )
                legend_table = (
                    f"<table style='{table_and_cell_style} border-collapse: collapse;'>"
                )
                img_td_style = "text-align:left; vertical-align: top; position: relative; top: 10px; padding: 0px;"
                label_td_style = (
                    "text-align:left; padding: 0; position: relative; left: 7px; "
                )
            else:
                legend_table = "<table>"

            for idx, legend_element in enumerate(legend["layers"][0]["legend"]):
                thumbnail = "data:{0};base64,{1}".format(
                    legend_element["contentType"],
                    legend_element["imageData"],
                )
                width = legend_element["width"]
                height = legend_element["height"]
                imgtag = "<img src='{0}' width='{1}' height='{2}' />".format(
                    thumbnail, width, height
                )

                if legend_type.lower() == "stretched":
                    img_row = f"<tr style='padding: 0; position: relative; {'top: 3.5px;' if idx==0 else 'bottom: 0.5px;'}'><td style='{table_and_cell_style} {img_td_style}'>"
                    label_row = f"</td><td style='{table_and_cell_style} {label_td_style} {'display: none' if idx==1 else ''}'>"
                else:
                    img_row = "<tr><td>"
                    label_row = "</td><td style='text-align:left'>"

                legend_table += (
                    img_row
                    + imgtag
                    + label_row
                    + legend_element["label"]
                    + "</td></tr>"
                )
            legend_table += "</table>"

            from IPython.display import HTML

            return HTML(legend_table)
        else:
            return legend

    # ----------------------------------------------------------------------
    def colormap(
        self,
        rendering_rule: Optional[dict[str, Any]] = None,
        variable: Optional[str] = None,
    ):
        """
        The ``colormap`` method returns RGB color representation of pixel
        values.

        .. note::
            This method is supported if the ``hasColormap`` property of the layer is ``True``.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        rendering_rule      Optional dictionary. Specifies the rendering rule for how the
                            requested image should be rendered.
                            See the
                            `Raster function objects <https://developers.arcgis.com/documentation/common-data-types/raster-function-objects.htm>`_
                            page in the ArcGIS REST API documentation for JSON syntax and examples.
        ---------------     --------------------------------------------------------------------
        variable            Optional string. This parameter can be used to request a
                            colormap for each variable for an image service that has
                            multidimensional information. It will return a colormap
                            for the whole image service if not specified. Eligible variable names
                            can be queried from :attr:`~arcgis.raster.ImageryLayer.multidimensional_info` property of the
                            :class:`~arcgis.raster.ImageryLayer` object.

                            .. note::
                                This parameter is available from 10.8.1 onwards.

        ===============     ====================================================================

        :return: A dictionary
        """
        if self.properties.hasColormap:
            url = self._url + "/colormap"
            params = {"f": "json"}
            if rendering_rule is not None:
                params["renderingRule"] = rendering_rule
            if variable is not None:
                params["variable"] = variable
            if self._datastore_raster:
                params["Raster"] = self._uri
                if isinstance(self._uri, bytes):
                    if (
                        "renderingRule" in params.keys()
                        and "renderingRule" in params.keys()
                    ):
                        del params["renderingRule"]
            return self._con.get(url, params, token=self._token)
        else:
            return None

    # ----------------------------------------------------------------------
    def compute_class_stats(
        self,
        descriptions: dict[str, Any],
        mosaic_rule: str = "defaultMosaicMethod",
        rendering_rule: Optional[dict[str, str]] = None,
        pixel_size: Optional[Union[str, dict[str, float]]] = None,
    ):
        """
        The ``compute_class_stats`` method computes the class statistics signatures (used by the maximum likelihood
        classifier)

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        descriptions        Required dict. Class descriptions are training site polygons and
                            their class descriptions. The structure of the geometry is the same
                            as the structure of the JSON geometry objects returned by the
                            ArcGIS REST API.

                            :Syntax:
                            | {
                            |     "classes":  [  // A list of classes
                            |           {
                            |             "id" : <id>,
                            |             "name" : "<name>",
                            |             "geometry" : <geometry> //polygon
                            |           },
                            |           {
                            |             "id" : <id>,
                            |             "name" : "<name>",
                            |             "geometry" : <geometry>  //polygon
                            |           }
                            |           ...
                            |                 ]
                            | }
        ---------------     --------------------------------------------------------------------
        mosaic_rule         Optional string. Specifies the mosaic rule when defining how
                            individual images should be mosaicked. When a mosaic rule is not
                            specified, the default mosaic rule of the image layer will be used
                            (as advertised in the root resource: ``defaultMosaicMethod``,
                            ``mosaicOperator``, ``sortField``, ``sortValue``).
                            See the
                            `Mosaic rule objects <https://developers.arcgis.com/documentation/common-data-types/mosaic-rules.htm>`_
                            help for more information.
        ---------------     --------------------------------------------------------------------
        rendering_rule      Optional dictionary. Specifies the rendering rule for how the
                            requested image should be rendered.
                            See the
                            `raster function objects <https://developers.arcgis.com/documentation/common-data-types/raster-function-objects.htm>`_
                            page in the ArcGIS REST API documentation for JSON syntax and examples.
        ---------------     --------------------------------------------------------------------
        pixel_size          Optional string or dictionary. The pixel level being used (or the
                            resolution being looked at). If pixel size is not specified, then
                            pixel_size will default to the base resolution of the dataset.
                            The structure of the pixel_size parameter is the same as the
                            structure of the point object returned by the ArcGIS REST API.
                            In addition to the dictionary structure, you can specify the pixel size
                            with a comma-separated syntax.

                              Syntax:
                                - dictionary structure: pixel_size={point}
                                - Point simple syntax: pixel_size='<x>,<y>'
                              Examples:
                                - pixel_size={"x": 0.18, "y": 0.18}
                                - pixel_size='0.18,0.18'
        ===============     ====================================================================

        :return: A dictionary

        .. code-block:: python

            # Example Usage

            img_lyr = gis.content.search("my_image_service", item_type="Imagery Layer")[0].layers[0]

            stats = img_lyr.compute_class_stats(descriptions={"classes": [
                                                                    {
                                                                        "id" : <id1>,
                                                                        "name" : "<name1>",
                                                                        "geometry" : <polygon1>
                                                                    },
                                                                    {
                                                                        "id" : <id2>,
                                                                        "name" : "<name2>",
                                                                        "geometry" : <polygon2>
                                                                    }
                                                                        ]
                                                            },
                                                pixel_size = {"x": 0.18, "y": 0.18}
                                            )

        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)

            except:
                raise RuntimeError(
                    "Failed to perform compute class stats operation on the TilesOnly service"
                )

        url = self._url + "/computeClassStatistics"

        params = {
            "f": "json",
            "classDescriptions": descriptions,
            "mosaicRule": mosaic_rule,
        }
        if self._mosaic_rule is not None and mosaic_rule is None:
            params["mosaicRule"] = self._mosaic_rule
        if rendering_rule is not None:
            params["renderingRule"] = rendering_rule
        if pixel_size is not None:
            params["pixelSize"] = pixel_size

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]

        return self._con.post(path=url, postdata=params, timeout=None)

    # ----------------------------------------------------------------------
    def compute_histograms(
        self,
        geometry: Union[Geometry, Polygon, Envelope],
        mosaic_rule: Optional[str] = None,
        rendering_rule: Optional[dict[str, Any]] = None,
        pixel_size: Optional[Union[dict[str, float], str]] = None,
        time: Optional[Union[str, list[int], datetime.date, datetime.datetime]] = None,
        process_as_multidimensional: bool = False,
    ):
        """
        The ``compute_histograms`` method returns a list of histograms for all raster bands computed for the
        :class:`~arcgis.raster.ImageryLayer` from the given extent.

        ============================    ====================================================================
        **Parameter**                   **Description**
        ----------------------------    --------------------------------------------------------------------
        geometry                        Required :class:`~arcgis.geometry.Geometry` (:class:`~arcgis.geometry.Polygon`
                                        or :class:`~arcgis.geometry.Envelope`).
                                        A geometry that defines the geometry within which the histogram is computed.
        ----------------------------    --------------------------------------------------------------------
        mosaic_rule                     Optional string. Specifies the mosaic rule when defining how
                                        individual images should be mosaicked. When a mosaic rule is not
                                        specified, the default mosaic rule of the image layer will be used
                                        (as advertised in the root resource: defaultMosaicMethod,
                                        mosaicOperator, sortField, sortValue).
                                        See the
                                        `Mosaic rule objects <https://developers.arcgis.com/documentation/common-data-types/mosaic-rules.htm>`_
                                        help for more information.
        ----------------------------    --------------------------------------------------------------------
        rendering_rule                  Optional rule. Specifies the rendering rule for how the requested image 
                                        should be processed. The response is updated Layer info that reflects a
                                        custom processing as defined by the rendering rule. For example, if
                                        renderingRule contains an attributeTable function, the response
                                        will indicate "hasRasterAttributeTable": true; if the renderingRule
                                        contains functions that alter the number of bands, the response will
                                        indicate a correct bandCount value.
        ----------------------------    --------------------------------------------------------------------
        pixel_size                      Optional string or dictionary. The pixel level being used (or the
                                        resolution being looked at). If pixel size is not specified, then
                                        pixel_size will default to the base resolution of the dataset.
                                        The structure of the pixel_size parameter is the same as the
                                        structure of the point object returned by the ArcGIS REST API.
                                        In addition to the dictionary structure, you can specify the pixel size
                                        with a comma-separated string.
                                        
                                        Syntax:
                                          - dictionary structure: pixel_size={point}
                                          - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                          - pixel_size={"x": 0.18, "y": 0.18}
                                          - pixel_size='0.18,0.18'
        ----------------------------    --------------------------------------------------------------------
        time                            Optional datetime.date, datetime.datetime or timestamp string. The
                                        time instant or the time extent to compute the histogram.
                                        Time instant specified as datetime.date, datetime.datetime or
                                        timestamp in milliseconds since epoch
                                        Syntax: time=<timeInstant>
                                        
                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp
                                        
                                        .. note::
                                            This parameter was added at 10.8.

        ----------------------------    --------------------------------------------------------------------
        process_as_multidimensional     Optional boolean. Specifies whether to process the image service as a 
                                        multidimensional image service.
                                        
                                            - ``False`` - The histogram of pixel values from only the first slice \
                                                      is computed. This is the default.
                                            - ``True`` - The image service is treated as a multidimensional raster, \
                                                     and histograms of pixel values from all selected slices are computed.
                                        
                                        .. note::
                                            Added at 10.9 for image services which use
                                            ``ArcObjects11`` or ``ArcObjectsRasterRendering``
                                            as the service provider.
        ============================    ====================================================================

        :return: A dictionary

        .. code-block:: python

            # Usage Example 1: Compute the histograms in the specified area of interest for a time instant.
            
            aoi = {
                    "spatialReference": {"wkid": 32610},
                    "xmax": 725000,
                    "xmin": 720000,
                    "ymax": 4300000,
                    "ymin": 4250000
                  }

            aoi_geometry = Geometry(aoi)

            comp_hist_01 = img_lyr.compute_histograms(geometry=aoi,
                                                      rendering_rule={"rasterFunction":None},
                                                      time="1326650400000")

        .. code-block:: python

            # Usage Example 2: Compute the histograms in the specified area of interest for a time extent.

            aoi = {
                    "spatialReference": {"wkid": 32610},
                    "xmax": 725000,
                    "xmin": 720000,
                    "ymax": 4300000,
                    "ymin": 4250000
                  }

            aoi_geometry = Geometry(aoi)

            # If the datetime object is not in the UTC timezone, the API will internally convert it to the UTC timezone.
            start = datetime.datetime(2012,1,15,18,0,0, tzinfo=datetime.timezone.utc)
            end = datetime.datetime(2012,1,15,21,0,0, tzinfo=datetime.timezone.utc)
            
            comp_hist_02 = img_lyr.compute_histograms(geometry=aoi,
                                                      rendering_rule={"rasterFunction":None},
                                                      time=[start, end])

        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)

            except:
                raise RuntimeError(
                    "Failed to perform compute histograms operation on the TilesOnly service"
                )

        import datetime

        url = self._url + "/computeHistograms"
        params = {
            "f": "json",
            "geometry": geometry,
        }

        if "xmin" in geometry:
            params["geometryType"] = "esriGeometryEnvelope"
        else:
            params["geometryType"] = "esriGeometryPolygon"

        if mosaic_rule is not None:
            params["mosaicRule"] = mosaic_rule
        elif self._mosaic_rule is not None:
            params["mosaicRule"] = self._mosaic_rule

        if not rendering_rule is None:
            params["renderingRule"] = rendering_rule
        elif self._fn is not None:
            params["renderingRule"] = self._fn

        if not pixel_size is None:
            params["pixelSize"] = pixel_size

        from ._util import _set_time_param

        if time is not None:
            params["time"] = _set_time_param(time)

        if isinstance(process_as_multidimensional, bool):
            params["processAsMultidimensional"] = process_as_multidimensional

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]

        return self._con.post(url, params, token=self._token, timeout=None)

        # ----------------------------------------------------------------------

    def get_samples(
        self,
        geometry: Union[Geometry, Point, MultiPoint, Polyline, Polygon, Envelope],
        geometry_type: Optional[str] = None,
        sample_distance: Optional[float] = None,
        sample_count: Optional[int] = None,
        mosaic_rule: Optional[dict[str, str]] = None,
        pixel_size: Optional[Union[dict[str, float], str]] = None,
        return_first_value_only: Optional[bool] = None,
        interpolation: Optional[str] = None,
        out_fields: Optional[str] = None,
        slice_id: Optional[int] = None,
        time: Optional[
            Union[
                str,
                list[datetime.date],
                list[datetime.datetime],
                datetime.date,
                datetime.datetime,
            ]
        ] = None,
    ):
        """
        The ``get_samples`` operation is supported by both mosaic dataset and raster
        dataset imagery layers.
        The result of this operation includes sample point locations, pixel
        values, and corresponding spatial resolutions of the source data for a
        given geometry.

        .. note::
            * When the input geometry is a :class:`~arcgis.geometry.Polyline`, \
            :class:`~arcgis.geometry.Envelope`, or \
            :class:`~arcgis.geometry.Polygon`, sampling is based on ``sample_count`` or ``sample_distance``; when the \
            input geometry is a :class:`~arcgis.geometry.Point` or :class:`~arcgis.geometry.MultiPoint`,\
            the point or points are used directly.

            * The number of sample locations in the response is based on the \
            ``sample_distance`` or ``sample_count`` parameter and cannot exceed the limit of \
            the image layer (the default is 1000, which is an approximate limit).

        =======================  =======================================================================
        **Parameter**             **Description**
        -----------------------  -----------------------------------------------------------------------
        geometry                 Required :class:`~arcgis.geometry.Geometry` object that defines the location(s)
                                 to be sampled. The structure of the geometry is the same as the structure of 
                                 the JSON geometry objects returned by the ArcGIS REST API. Applicable geometry
                                 types are ``point``, ``multipoint``, ``polyline``, ``polygon``, and ``envelope``. When
                                 spatial reference is omitted in the input geometry, it will be assumed
                                 to be the spatial reference of the image layer.
        -----------------------  -----------------------------------------------------------------------
        geometry_type            Optional string. The type of geometry specified by the geometry
                                 parameter.
                                 Applicable geometry
                                 types are ``point``, ``multipoint``, ``polyline``, ``polygon``, and ``envelope``.
        -----------------------  -----------------------------------------------------------------------
        sample_distance          Optional float. The distance interval used to sample points from
                                 the provided path. The unit is the same as the input geometry. If
                                 neither sample_count nor sample_distance is provided, no
                                 densification can be done for paths (polylines), and a default
                                 sample_count (100) is used for areas (polygons or envelopes).
        -----------------------  -----------------------------------------------------------------------
        sample_count             Optional integer. The approximate number of sample locations from
                                 the provided path. If neither sample_count nor sample_distance is
                                 provided, no densification can be done for paths (polylines), and a
                                 default sample_count (100) is used for areas (polygons or envelopes).
        -----------------------  -----------------------------------------------------------------------
        mosaic_rule              Optional dictionary.  Specifies the mosaic rule when defining how
                                 individual images should be mosaicked. When a mosaic rule is not
                                 specified, the default mosaic rule of the image layer will be used
                                 (as advertised in the root resource: defaultMosaicMethod,
                                 mosaicOperator, sortField, sortValue).
        -----------------------  -----------------------------------------------------------------------
        pixel_size               Optional string or dict. The pixel level being used (or the
                                 resolution being looked at). If pixel size is not specified, then
                                 pixel_size will default to the base resolution of the dataset. The
                                 raster at the specified pixel size in the mosaic dataset will be
                                 used for histogram calculation.

                                 Syntax:
                                    - dictionary structure: pixel_size={point}
                                    - Point simple syntax: pixel_size='<x>,<y>'
                                 Examples:
                                    - pixel_size={"x": 0.18, "y": 0.18}
                                    - pixel_size='0.18,0.18'
        -----------------------  -----------------------------------------------------------------------
        return_first_value_only  Optional boolean. Indicates whether to return all values at a
                                 point, or return the first non-NoData value based on the current
                                 mosaic rule.
                                 The default is true.
        -----------------------  -----------------------------------------------------------------------
        interpolation            Optional string. The resampling method. Default is nearest neighbor.
                                 Values: RSP_BilinearInterpolation, RSP_CubicConvolution, RSP_Majority,
                                 RSP_NearestNeighbor
        -----------------------  -----------------------------------------------------------------------
        out_fields               Optional string. The list of fields to be included in the response.
                                 This list is a comma-delimited list of field names. You can also
                                 specify the wildcard character (*) as the value of this parameter to
                                 include all the field values in the results.
        -----------------------  -----------------------------------------------------------------------
        slice_id                 Optional integer. The slice ID of a multidimensional raster. The operation 
                                 will be performed for the specified slice.
                                 This parameter is available from 10.9 onwards.
        -----------------------  -----------------------------------------------------------------------
        time                     Optional datetime.date, datetime.datetime or timestamp string.
                                 The time instant or time extent of the raster to be sampled.
                                 Time instant specified as datetime.date, datetime.datetime or
                                 timestamp in milliseconds since epoch
                                 Syntax: time=<timeInstant>
                                        
                                 Time extent specified as list of [<startTime>, <endTime>]
                                 For time extents one of <startTime> or <endTime> could be None. A
                                 None value specified for start time or end time will represent
                                 infinity for start or end time respectively.
                                 Syntax: time=[<startTime>, <endTime>] ; specified as
                                 datetime.date, datetime.datetime or timestamp
                                 This parameter is available from 10.9 onwards.
        =======================  =======================================================================

        :return:
            A list of samples
        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)

            except:
                raise RuntimeError(
                    "Failed to perform get samples operation on the TilesOnly service"
                )

        if not isinstance(geometry, Geometry):
            geometry = Geometry(geometry)

        geometry_type_value_list = [
            "point",
            "multipoint",
            "polyline",
            "polygon",
            "envelope",
        ]
        if geometry_type is None:
            geometry_type = "esriGeometry" + geometry.type
        elif geometry_type.lower() in geometry_type_value_list:
            geometry_type = "esriGeometry" + geometry_type.lower().capitalize()

        url = self._url + "/getSamples"
        params = {
            "f": "json",
            "geometry": geometry,
            "geometryType": geometry_type,
        }

        if not sample_distance is None:
            params["sampleDistance"] = sample_distance
        if not sample_count is None:
            params["sampleCount"] = sample_count
        if not mosaic_rule is None:
            params["mosaicRule"] = mosaic_rule
        elif self._mosaic_rule is not None:
            params["mosaicRule"] = self._mosaic_rule
        if not pixel_size is None:
            params["pixelSize"] = pixel_size
        if not return_first_value_only is None:
            params["returnFirstValueOnly"] = return_first_value_only
        if not interpolation is None:
            params["interpolation"] = interpolation
        if not out_fields is None:
            params["outFields"] = out_fields
        if slice_id is not None:
            params["sliceId"] = slice_id

        from ._util import _set_time_param

        if time is not None:
            params["time"] = _set_time_param(time)

        if self._datastore_raster:
            params["Raster"] = self._uri

        sample_data = self._con.post(
            path=url, postdata=params, token=self._token, timeout=None
        )["samples"]
        from copy import deepcopy

        new_sample_data = deepcopy(sample_data)
        # region: Try to convert values to list of numbers if it makes sense
        try:
            for element in new_sample_data:
                if "value" in element and isinstance(element["value"], str):
                    pix_values_numbers = [float(s) for s in element["value"].split(" ")]
                    element["values"] = pix_values_numbers
            sample_data = new_sample_data
        except:
            pass  # revert and return the original data as is.

        # endregion
        return sample_data

    def key_properties(self, rendering_rule: Optional[dict[str, Any]] = None):
        """
        The ``key_properties`` method retrieves the key properties of the :class:`~arcgis.raster.ImageryLayer`,
        such as band properties.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        rendering_rule        Optional dictionary. Specifies the rendering rule for how the
                              requested image should be rendered.
        =================     ====================================================================

        :return:
            A dictionary representing the key properties of the :class:`~arcgis.raster.ImageryLayer`
        """
        url = self._url + "/keyProperties"
        params = {"f": "json"}

        if rendering_rule is not None:
            params["renderingRule"] = rendering_rule
        elif self._fn is not None:
            params["renderingRule"] = self._fn

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]

        return self._con.post(
            path=url, postdata=params, token=self._token, timeout=None
        )

    def mosaic_by(
        self,
        method: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_val: Optional[str] = None,
        lock_rasters: Optional[list[int]] = None,
        viewpt: Optional[Point] = None,
        asc: bool = True,
        where: Optional[str] = None,
        fids: Optional[list[int]] = None,
        muldidef: Optional[list] = None,
        op: str = "first",
        item_rendering_rule: Optional[str] = None,
    ):
        """
        The ``mosaic_by`` method defines how individual images in this layer should be mosaicked.

        .. note::
            It specifies selection, mosaic method, sort order, overlapping pixel resolution, etc.
            Mosaic rules are for mosaicking rasters in
            the mosaic dataset.

        A mosaic rule is used to define:

            * The selection of rasters that will participate in the mosaic (using where clause).
            * The mosaic method, e.g. how the selected rasters are ordered.
            * The mosaic operation, e.g. how overlapping pixels at the same location are resolved.

        .. note::
            See the `Understanding the mosaicking rules for a mosaic dataset <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/understanding-the-mosaicking-rules-for-a-mosaic-dataset.htm#ESRI_SECTION1_ABDC9F3F6F724A4F8079051565DC59E>`_
            page for more inforamation on mosaic datasets. See the
            `mosaic rule <https://developers.arcgis.com/documentation/common-data-types/mosaic-rules.htm>`_ page for
            more information on mosaic rules.

        =======================  =======================================================================
        **Parameter**             **Description**
        -----------------------  -----------------------------------------------------------------------
        method                   Optional string. Determines how the selected rasters are ordered.
                                 str, can be none,center,nadir,northwest,seamline,viewpoint,
                                 attribute,lock-raster
                                 required if method is: center,nadir,northwest,seamline, optional
                                 otherwise. If no method is passed "none" method is used, which uses
                                 the order of records to sort
                                 If sort_by and optionally sort_val parameters are specified,
                                 ``attribute`` method is used

                                 .. note::
                                    If lock_rasters are specified, "lock-raster" method is used.
                                    If a viewpt parameter is passed, "viewpoint" method is used.
        -----------------------  -----------------------------------------------------------------------
        sort_by                  Optional string. field name when sorting by attributes
        -----------------------  -----------------------------------------------------------------------
        sort_val                 Optional string. A constant value defining a reference or base value
                                 for the sort field when sorting by attributes
        -----------------------  -----------------------------------------------------------------------
        lock_rasters             Optional list, an array of raster Ids. All the rasters with the given
                                 list of raster Ids are selected to participate in the mosaic. The
                                 rasters will be visible at all pixel sizes regardless of the minimum
                                 and maximum pixel size range of the locked rasters.
        -----------------------  -----------------------------------------------------------------------
        viewpt                   Optional point, used as view point for viewpoint mosaicking method
        -----------------------  -----------------------------------------------------------------------
        asc                      Optional bool, indicate whether to use ascending or descending
                                 order. Default is ascending order.
        -----------------------  -----------------------------------------------------------------------
        where                    Optional string. where clause to define a subset of rasters used in
                                 the mosaic, be aware that the rasters may not be visible at all
                                 scales
        -----------------------  -----------------------------------------------------------------------
        fids                     Optional list of objectids, use the raster id list to define a
                                 subset of rasters used in the mosaic, be aware that the rasters may
                                 not be visible at all scales.
        -----------------------  -----------------------------------------------------------------------
        muldidef                 Optional array. multidemensional definition used for filtering by
                                 variable/dimensions.
                                 See the
                                 `Multidimensional definition <https://developers.arcgis.com/documentation/common-data-types/multidimensional-definition.htm>`_
                                 page in the ArcGIS REST API documentation for more information.
        -----------------------  -----------------------------------------------------------------------
        op                       Optional string, first,last,min,max,mean,blend,sum mosaic operation
                                 to resolve overlap pixel values: from first or last raster, use the
                                 min, max or mean of the pixel values, or blend them.
        -----------------------  -----------------------------------------------------------------------
        item_rendering_rule      Optional item rendering rule, applied on items before mosaicking.
        =======================  =======================================================================

        :return:
            A mosaic rule.

        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        mosaic_rule = {
            "mosaicMethod": "esriMosaicNone",
            "ascending": asc,
            "mosaicOperation": "MT_" + op.upper(),
        }

        if where is not None:
            mosaic_rule["where"] = where

        if fids is not None:
            mosaic_rule["fids"] = fids

        if muldidef is not None:
            mosaic_rule["multidimensionalDefinition"] = muldidef

        if method in ["none", "center", "nadir", "northwest", "seamline"]:
            mosaic_rule["mosaicMethod"] = "esriMosaic" + method.title()

        if viewpt is not None:
            if not isinstance(viewpt, Geometry):
                viewpt = Geometry(viewpt)
            mosaic_rule["mosaicMethod"] = "esriMosaicViewpoint"
            mosaic_rule["viewpoint"] = viewpt

        if sort_by is not None:
            mosaic_rule["mosaicMethod"] = "esriMosaicAttribute"
            mosaic_rule["sortField"] = sort_by
            if sort_val is not None:
                mosaic_rule["sortValue"] = sort_val

        if lock_rasters is not None:
            mosaic_rule["mosaicMethod"] = "esriMosaicLockRaster"
            mosaic_rule["lockRasterIds"] = lock_rasters

        if item_rendering_rule is not None:
            mosaic_rule["itemRenderingRule"] = item_rendering_rule

        if self._fnra is not None:
            self._fnra["rasterFunctionArguments"] = _find_and_replace_mosaic_rule(
                self._fnra["rasterFunctionArguments"], mosaic_rule, self._url
            )
        self._mosaic_rule = mosaic_rule

    def validate(
        self,
        rendering_rule: Optional[dict[str, Any]] = None,
        mosaic_rule: Optional[dict[str, Any]] = None,
    ):
        """
        The ``validate`` method validates rendering rule and/or mosaic rule of an image service.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        rendering_rule        Optional dictionary. Specifies the rendering rule to be validated
        -----------------     --------------------------------------------------------------------
        mosaic_rule           Optional dictionary. Specifies the mosaic rule to be validated
        =================     ====================================================================

        :return: A dictionary showing whether the specified rendering rule and/or mosaic rule is valid
        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        url = self._url + "/validate"

        params = {"f": "json"}
        if mosaic_rule is not None:
            params["mosaicRule"] = mosaic_rule
        if rendering_rule is not None:
            params["renderingRule"] = rendering_rule

        if self._datastore_raster:
            params["Raster"] = self._uri
            if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                del params["renderingRule"]

        return self._con.post(path=url, postdata=params, timeout=None)

    def compute_cache_info(self, out_sr: Optional[int] = None):
        """
        The ``compute_cache_info`` method computes and generates new image service tile cache
        schemes for image services.
        If the corresponding image tile cache scheme is missing, it will also create a new set of
        cached image tiles in the cache directory of the image service. This operation only
        generates image tile cache schemes based on raster tiles (LERC2D format).

        .. note::
            This applies to image services that have dynamic service caching capability enabled.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        out_sr                Optional integer. The spatial reference of the boundary's geometry.
                              The spatial reference can be specified as a well-known ID.
                              If the ``out_sr`` is not specified, it will use the spatial
                              reference of the image service.
        =================     ====================================================================

        :returns: A dictionary with the image tile cache scheme information.

        .. code-block:: python

            # Example Usage: Compute the cache info for a given spatial reference.

            cache_info_op = img_lyr.compute_cache_info(out_sr=3857)

        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        url = self._url + "/computeCacheInfo"
        params = {"f": "json"}
        if out_sr is not None:
            params["outSR"] = out_sr

        return self._con.post(path=url, postdata=params, timeout=None)

    def compute_angles(
        self,
        raster_id: int,
        point: Optional[Union[dict, Point]] = None,
        angle_name: Optional[str] = None,
        spatial_reference: Optional[dict] = None,
    ):
        """
        The ``compute_angles`` method computes the rotation angle of a raster for a user-specified
        angle direction, and optionally, a user-specified rotation point and user-specified
        spatial reference.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        raster_id             Required integer. Specifies the object ID of the raster catalog which
                              will determine the raster and image coordinate system to use in a
                              mosaic dataset.
        -----------------     --------------------------------------------------------------------
        point                 Optional dictionary or :class:`~arcgis.geometry.Point` object.
                              The point geometry that defines the reference point of rotation to 
                              compute the angle direction. By default, takes the centroid of image
                              as point of rotation.
        -----------------     --------------------------------------------------------------------
        angle_name            Optional string. Specifies the name (or names) of the rotation
                              angle to be computed.

                              Possible options are: 

                              - "up": The computed angle after rotating the map so the top of the image is always \
                              oriented to the direction of the sensor when it acquired the image.
                              - "north": The computed angle after rotating so the top of the image is always \
                              toward north.
                              
                              You can specify multiple angle names by separating the names with a
                              comma. By default, angles are computed for all directions.
        -----------------     --------------------------------------------------------------------
        spatial_reference     Optional dictionary. Specifies the spatial reference to be used by
                              the image. By default, the spatial reference of the image is used.
        =================     ====================================================================

        :returns: A dictionary with the computed rotation angle of a raster.

        .. code-block:: python

            # Example Usage: Compute angles for a given point and rotation angle.

            my_point = {
                        "x": 7952916.33, 
                        "y": 3869525.96,
                        "spatialReference": {"wkid": 3857}
                       }

            my_point_object = Point(my_point)
            
            compute_angles_op = img_lyr.compute_angles(raster_id=1,
                                                       point=my_point_object,
                                                       angle_name="north",
                                                       spatial_reference={"wkid": 54004})

        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        url = self._url + "/computeAngles"
        params = {"rasterId": raster_id, "f": "json"}
        if point is not None:
            params["point"] = point
        if angle_name is not None:
            params["angleName"] = angle_name
        if spatial_reference is not None:
            params["spatialReference"] = spatial_reference

        return self._con.post(path=url, postdata=params, timeout=None)

    def calculate_volume(
        self,
        geometries: Union[list[Polygon], list[Envelope]],
        base_type: Optional[int] = None,
        mosaic_rule: Optional[dict[str, Any]] = None,
        constant_z: Optional[int] = None,
        pixel_size: Optional[Union[str, dict[str, float]]] = None,
    ):
        """
        The ``calculate_volume`` method performs volumetric calculation on an elevation service.

        .. note::

            * If a service does not have vertical spatial reference and z unit is not in meters, user \
            needs to apply a conversion factor when interpreting results. \
            The results are always in square meters (area) and cubic meters (volume).

            * The ``calculate_volume`` method is only available in 10.7+.

        =================       ====================================================================
        **Parameter**            **Description**
        -----------------       --------------------------------------------------------------------
        geometries              A required list of :class:`~arcgis.geometry.Polygon` geometry objects or a
                                list of :class:`~arcgis.geometry.Envelope` geometry objects.
                                A geometry that defines the geometry
                                within which the volume is computed.
        -----------------       --------------------------------------------------------------------
        base_type               Optional integer.
                                0 - constant z;
                                1 - best fitting plane;
                                2 - lowest elevation on the perimeter;
                                3 - highest elevation on the perimeter;
                                4 - average elevation on the perimeter
        -----------------       --------------------------------------------------------------------
        mosaic_rule             Optional dictionary. Used to select different DEMs in a mosaic dataset
        -----------------       --------------------------------------------------------------------
        constant_z              Optional integer. parameter to specify constant z value
        -----------------       --------------------------------------------------------------------
        pixel_size              Optional string or dictionary. Defines the spatial resolution at which volume calculation is performed
                                Syntax:
                                    - dictionary structure: pixel_size={point}
                                    - Point simple syntax: pixel_size='<x>,<y>'
                                Examples:
                                    - pixel_size={"x": 0.18, "y": 0.18}
                                    - pixel_size='0.18,0.18'
        =================       ====================================================================

        :return: A dictionary showing volume values for each geometry in the input geometries array

        """
        if self.tiles_only:
            try:
                self = _get_rendering_service_layer(self)

            except:
                raise RuntimeError(
                    "Failed to perform calculate volume operation on the TilesOnly service"
                )

        if self.properties.serviceDataType == "esriImageServiceDataTypeElevation":
            url = "%s/calculateVolume" % self._url
            from arcgis.geometry import Polygon

            if isinstance(geometries, list):
                geometry = geometries[0]
                if geometry:
                    if isinstance(geometry, Polygon):
                        gt = "esriGeometryPolygon"
                    else:
                        gt = "esriGeometryEnvelope"
            else:
                raise RuntimeError(
                    "Invalid geometries - required an array of Polygon geometry object or an array of envelope geometry object"
                )
            params = {
                "f": "json",
                "geometries": geometries,
                "geometryType": gt,
            }
            if base_type is not None:
                params["baseType"] = base_type

            if mosaic_rule is not None:
                params["mosaicRule"] = mosaic_rule
            elif self._mosaic_rule is not None:
                params["mosaicRule"] = self._mosaic_rule

            if constant_z is not None:
                params["constantZ"] = constant_z

            if pixel_size is not None:
                params["pixelSize"] = pixel_size

            if self._datastore_raster:
                params["Raster"] = self._uri

            return self._con.post(path=url, postdata=params, timeout=None)

        return None

    def query_boundary(
        self,
        out_sr: Optional[Union[int, dict[str, Any], SpatialReference]] = None,
    ):
        """
        The ``query_boundary`` operation is supported by image services based on mosaic datasets
        or raster datasets.

        For an image service based on a mosaic dataset, the result of this operation
        includes the geometry shape of the mosaicked items' boundary and area of
        coverage in square meters.

        For an image service based on a raster dataset, the result of this operation
        includes the geometry shape of the dataset's envelope boundary and area of
        coverage in square meters.

        .. note::
            The ``query_boundary`` method was added at 10.6.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        out_sr                The spatial reference of the boundary's geometry.

                              The spatial reference can be specified as either a well-known ID or
                              as a spatial reference JSON object.

                              If the ``out_SR`` is not specified, the boundary will be reported in the
                              spatial reference of the image service.

                              Example:
                                4326
        =================     ====================================================================

        :return: A dictionary representing the boundary geometry shape.

        """
        if (
            (hasattr(self, "_do_not_hydrate")) and not self._do_not_hydrate
        ) or not hasattr(self, "_do_not_hydrate"):
            if self.tiles_only:
                try:
                    self = _get_rendering_service_layer(self)
                except:
                    raise RuntimeError(
                        "Failed to perform query boundary operation on the TilesOnly service"
                    )

        url = self._url + "/queryBoundary"

        params = {"f": "json"}
        if out_sr is not None:
            params["outSR"] = out_sr

        if self._datastore_raster:
            params["Raster"] = self._uri

        return self._con.post(path=url, postdata=params, timeout=None)

    def query_gps_info(
        self,
        where: Optional[str] = None,
        object_ids: Optional[list[int]] = None,
        time_filter: Optional[
            Union[datetime.date, datetime.datetime, list[int], str]
        ] = None,
        geometry_filter: Optional[dict] = None,
    ):
        """
        The ``query_gps_info`` method queries an :class:`~arcgis.raster.ImageryLayer` by applying the filter specified by
        the user. The result of this operation is the gps and orientation information for image collections created by
        OrthoMapping REST/Python API or Ortho Maker.

        .. note::
            The ``query_gps_info`` operation is supported at 11.2 and later.

        ==============================  ====================================================================
        **Parameter**                   **Description**
        ------------------------------  --------------------------------------------------------------------
        where                           Optional string. A where clause on this layer to filter the imagery
                                        layer by the selection sql statement. Any legal SQL where clause
                                        operating on the fields in the raster
        ------------------------------  --------------------------------------------------------------------
        object_ids                      Optional list of objectids, use the raster id list to define a
                                        subset of rasters.
        ------------------------------  --------------------------------------------------------------------
        time_filter                     Optional datetime.date, datetime.datetime or timestamp in
                                        milliseconds. The time instant or the time extent to query.

                                        Syntax: time_filter=<timeInstant>

                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time_filter=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp in milliseconds
        ------------------------------  --------------------------------------------------------------------
        geometry_filter                 Optional arcgis.geometry.filters. Spatial filter from
                                        arcgis.geometry.filters module to filter results by a spatial
                                        relationship with another geometry.
        ==============================  ====================================================================

        :return: A dict containing the gps and camera information for the image collection.

        .. code-block:: python

            # Usage Example

            img_lyr = gis.content.search("my_image_service", item_type="Imagery Layer")[0].layers[0]
            gps_info = img_lyr.query_gps_info(where="OBJECTID=1")

            # Usage Example 2

            img_lyr = gis.content.search("my_image_service", item_type="Imagery Layer")[0].layers[0]
            aoi_intersects = arcgis.geometry.filters.intersects(geometry=geometry_obj)
            gps_info = img_lyr.query_gps_info(geometry_filter=aoi_intersects)

        """

        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )

        params = {"f": "json"}
        if object_ids:
            params["objectIds"] = object_ids

        if where is not None:
            params["where"] = where
        elif self._where_clause is not None:
            params["where"] = self._where_clause
        else:
            params["where"] = "1=1"

        if self._temporal_filter is not None:
            time_filter = self._temporal_filter

        if time_filter is not None:
            if type(time_filter) is list:
                starttime = _date_handler(time_filter[0])
                endtime = _date_handler(time_filter[1])
                if starttime is None:
                    starttime = "null"
                if endtime is None:
                    endtime = "null"
                params["time"] = "%s,%s" % (starttime, endtime)
            else:
                params["time"] = _date_handler(time_filter)

        if self._spatial_filter is not None:
            geometry_filter = self._spatial_filter

        if not geometry_filter is None and isinstance(geometry_filter, dict):
            gf = geometry_filter
            params["geometry"] = gf["geometry"]
            params["geometryType"] = gf["geometryType"]
            params["spatialRel"] = gf["spatialRel"]
            if "inSR" in gf:
                params["inSR"] = gf["inSR"]

        url = self._url + "/queryGPSInfo"
        res = self._con.post(path=url, postdata=params, timeout=None)

        return res

    def _compute_multidimensional_info(
        self,
        where=None,
        object_ids=None,
        time_filter=None,
        geometry_filter=None,
        pixel_size=None,
        raster_query=None,
        variable_field_name=None,
        dimension_field_names=None,
    ):
        """
        Opertion to get the multidimensional info.
        ==============================  ====================================================================
        **Parameter**                   **Description**
        ------------------------------  --------------------------------------------------------------------
        where                           optional string. A where clause on this layer to filter the imagery
                                        layer by the selection sql statement. Any legal SQL where clause
                                        operating on the fields in the raster
        ------------------------------  --------------------------------------------------------------------
        out_fields                      optional string. The attribute fields to return, comma-delimited
                                        list of field names.
        ------------------------------  --------------------------------------------------------------------
        time_filter                     optional datetime.date, datetime.datetime or timestamp in
                                        milliseconds. The time instant or the time extent to compute mdim info.

                                        Syntax: time_filter=<timeInstant>

                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time_filter=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp in milliseconds
        ------------------------------  --------------------------------------------------------------------
        geometry_filter                 optional arcgis.geometry.filters. Spatial filter from
                                        arcgis.geometry.filters module to filter results by a spatial
                                        relationship with another geometry.
        ------------------------------  --------------------------------------------------------------------
        pixel_size                      optional dict or string. Query visible rasters at a given pixel size.
                                        If pixel_size is not specified, rasters at all resolutions can be
                                        queried.
                                        Syntax:
                                            - dictionary structure: pixel_size={point}
                                            - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                            - pixel_size={"x": 0.18, "y": 0.18}
                                            - pixel_size='0.18,0.18'
        ------------------------------  --------------------------------------------------------------------
        variable_field_name             Variable names
        ------------------------------  --------------------------------------------------------------------
        dimension_field_names           Dimension Names
        ==============================  ====================================================================

        :return: A dict representing the md info
        """
        if self.tiles_only:
            raise RuntimeError(
                "This operation cannot be performed on a TilesOnly Service"
            )

        url = self._url + "/computeMultidimensionalInfo"

        params = {"f": "json"}
        if object_ids:
            params["objectIds"] = object_ids

        if where is not None:
            params["where"] = where
        elif self._where_clause is not None:
            params["where"] = self._where_clause
        else:
            params["where"] = "1=1"

        if self._temporal_filter is not None:
            time_filter = self._temporal_filter

        if time_filter is not None:
            if type(time_filter) is list:
                starttime = _date_handler(time_filter[0])
                endtime = _date_handler(time_filter[1])
                if starttime is None:
                    starttime = "null"
                if endtime is None:
                    endtime = "null"
                params["time"] = "%s,%s" % (starttime, endtime)
            else:
                params["time"] = _date_handler(time_filter)

        if self._spatial_filter is not None:
            geometry_filter = self._spatial_filter

        if not geometry_filter is None and isinstance(geometry_filter, dict):
            gf = geometry_filter
            params["geometry"] = gf["geometry"]
            params["geometryType"] = gf["geometryType"]
            params["spatialRel"] = gf["spatialRel"]
            if "inSR" in gf:
                params["inSR"] = gf["inSR"]

        if pixel_size is not None:
            params["pixelSize"] = pixel_size

        if variable_field_name is not None:
            params["variableFieldName"] = variable_field_name

        if dimension_field_names is not None:
            params["dimensionFieldNames"] = dimension_field_names

        if self._datastore_raster:
            params["Raster"] = self._uri

        res = self._con.post(
            path=url, postdata=params, token=self._token, timeout=None
        )["multidimensionalInfo"]
        return res

    @property
    def mosaic_rule(self):
        """
        The ``mosaic_rule`` property is used by the :class:`~arcgis.raster.ImageryLayer` to define:

                - The selection of rasters that will participate in the mosaic
                - The mosaic method, e.g. how the selected rasters are ordered.
                - The mosaic operation, e.g. how overlapping pixels at the same location are resolved.

        Set by calling the :attr:`~arcgis.raster.ImageryLayer.mosaic_by` or
        :attr:`~arcgis.raster.ImageryLayer.filter_by` methods on the layer.

        :return: A mosaic rule
        """
        return self._mosaic_rule

    @mosaic_rule.setter
    def mosaic_rule(self, value):
        self._mosaic_rule = value

    def _mosaic_operation(self, op):
        """
        Sets how overlapping pixels at the same location are resolved

        :param op: string, one of first,last,min,max,mean,blend,sum

        :return: this imagery layer with mosaic operation set to op
        """

        if self._datastore_raster:
            raise RuntimeError(
                "This operation cannot be performed on a datastore raster"
            )
        newlyr = self._clone_layer()
        if self._mosaic_rule is not None:
            newlyr._mosaic_rule["mosaicOperation"] = "MT_" + op.upper()
        return newlyr

    def first(self):
        """
        The ``first`` method returns this :class:`~arcgis.raster.ImageryLayer` with mosaic operation set to ``first`` (overlapping
        pixels at the same location are resolved by picking the first image)

        :return: An :class:`~arcgis.raster.ImageryLayer` object
        """
        return self._mosaic_operation("first")

    def last(self):
        """
        The ``last`` method returns this :class:`~arcgis.raster.ImageryLayer` with mosaic operation set to ``last`` (overlapping
        pixels at the same location are resolved by picking the last image)

        :return: An :class:`~arcgis.raster.ImageryLayer` object
        """
        return self._mosaic_operation("last")

    def min(self):
        """
        The ``min`` method returns this :class:`~arcgis.raster.ImageryLayer` with mosaic operation set to ``min`` (overlapping
        pixels at the same location are resolved by picking the min pixel value)

        :return: An :class:`~arcgis.raster.ImageryLayer` object
        """
        return self._mosaic_operation("min")

    def max(self):
        """
        The ``max`` method returns this :class:`~arcgis.raster.ImageryLayer` with mosaic operation set to ``max`` (overlapping
        pixels at the same location are resolved by picking the max pixel value)

        :return: An :class:`~arcgis.raster.ImageryLayer` object
        """
        return self._mosaic_operation("max")

    def mean(self):
        """
        The ``mean`` method returns this :class:`~arcgis.raster.ImageryLayer` with mosaic operation set to ``mean`` (overlapping
        pixels at the same location are resolved by choosing the mean of all overlapping pixels)

        :return: An :class:`~arcgis.raster.ImageryLayer` object
        """
        return self._mosaic_operation("mean")

    def blend(self):
        """
        The ``blend`` method returns this :class:`~arcgis.raster.ImageryLayer` with mosaic operation set to ``blend`` (overlapping
        pixels at the same location are resolved by blending all overlapping pixels)

        :return: An :class:`~arcgis.raster.ImageryLayer` object
        """
        return self._mosaic_operation("blend")

    def sum(self):
        """
        The ``sum`` method returns this :class:`~arcgis.raster.ImageryLayer` with mosaic operation set to ``sum`` (overlapping
        pixels at the same location are resolved by adding up all overlapping pixel values)

        :return: An :class:`~arcgis.raster.ImageryLayer` object
        """
        return self._mosaic_operation("sum")

    def save(
        self,
        output_name: Optional[str] = None,
        for_viz: bool = False,
        process_as_multidimensional: Optional[bool] = None,
        build_transpose: Optional[bool] = None,
        context: Optional[dict] = None,
        *,
        gis: Optional[GIS] = None,
        future: bool = False,
        **kwargs,
    ):
        """
        The ``save`` method saves this imagery layer to the GIS as an :class:`~arcgis.raster.ImageryLayer` item.
        If ``for_viz`` is ``True``, a new :class:`~arcgis.gis.Item` is created that
        uses the applied raster functions for visualization at display resolution using on-the-fly image processing.
        If ``for_viz`` is ``False``, distributed raster analysis is used for generating a new raster information product by
        applying raster functions at source resolution across the extent of the output imagery layer.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        output_name                              Optional string. If not provided, an Imagery Layer item is created
                                                 by the method and used as the output.
                                                 You can pass in the name of the output Imagery Layer that should be
                                                 created by this method to be used as the output for the tool.
                                                 Alternatively, if for_viz is False, you can pass in an existing
                                                 Image Layer Item from your GIS to use that instead.
                                                 A RuntimeError is raised if a layer by that name already exists
        ------------------------------------     --------------------------------------------------------------------
        for_viz                                  Optional boolean. If True, a new Item is created that uses the
                                                 applied raster functions for visualization at display resolution
                                                 using on-the-fly image processing.
                                                 If for_viz is False, distributed raster analysis is used for
                                                 generating a new raster information product for use in analysis and
                                                 visualization by applying raster functions at source resolution
                                                 across the extent of the output imagery layer.
        ------------------------------------     --------------------------------------------------------------------
        process_as_multidimensional              Optional bool.  If the input is multidimensional raster, the output
                                                 will be processed as multidimensional if set to True
        ------------------------------------     --------------------------------------------------------------------
        build_transpose                          Optional bool, if set to true, transforms the output
                                                 multidimensional raster. Valid only if process_as_multidimensional
                                                 is set to True
        ------------------------------------     --------------------------------------------------------------------
        context                                  context contains additional settings that affect task execution.

                                                 context parameter overwrites values set through arcgis.env parameter

                                                 This function has the following settings:

                                                  - Extent (extent): A bounding box that defines the analysis area.

                                                    Example:
                                                        {"extent": {"xmin": -122.68,
                                                        "ymin": 45.53,
                                                        "xmax": -122.45,
                                                        "ymax": 45.6,
                                                        "spatialReference": {"wkid": 4326}}}

                                                  - Output Spatial Reference (outSR): The output raster will be
                                                    projected into the output spatial reference.

                                                    Example:
                                                        {"outSR": {spatial reference}}

                                                  - Snap Raster (snapRaster): The output raster will have its
                                                    cells aligned with the specified snap raster.

                                                    Example:
                                                        {'snapRaster': {'url': '<image_service_url>'}}

                                                  - Mask (mask): Only cells that fall within the analysis
                                                    mask will be considered in the operation.

                                                    Example:
                                                        {"mask": {"url": "<image_service_url>"}}

                                                  - Cell Size (cellSize): The output raster will have the resolution
                                                    specified by cell size.

                                                    Example:
                                                        {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                                  - Parallel Processing Factor (parallelProcessingFactor): controls
                                                    Raster Processing (CPU) service instances.

                                                    Example:
                                                        Syntax example with a specified number of processing instances:

                                                        {"parallelProcessingFactor": "2"}

                                                        Syntax example with a specified percentage of total
                                                        processing instances:

                                                        {"parallelProcessingFactor": "60%"}

                                                  - Resampling Method (resamplingMethod): The output raster will be
                                                    resampled to method specified.
                                                    The supported values are: BILINEAR, NEAREST, CUBIC.

                                                    Example:
                                                        {'resamplingMethod': "NEAREST"}
        ------------------------------------     --------------------------------------------------------------------
        gis                                      Optional :class:`~arcgis.gis.GIS` object.
                                                 The GIS to be used for saving the
                                                 output. Keyword only parameter.
        ------------------------------------     --------------------------------------------------------------------
        future                                   Optional boolean. If True, a future object will be returned and the process
                                                 will not wait for the task to complete. The default is False, which means wait for results.
        ------------------------------------     --------------------------------------------------------------------
        folder                                   Optional string or dictionary. Creates a folder in the portal, if it does
                                                 not exist, with the given folder name and persists the output in this folder.
                                                 The dictionary returned by the gis.content.create_folder() can also be passed in as input.

                                                 Example:
                                                    {'username': 'user1', 'id': '6a3b77c187514ef7873ba73338cf1af8', 'title': 'trial'}
        ------------------------------------     --------------------------------------------------------------------
        tiles_only                               In ArcGIS Online, the default output image service for this function
                                                 would be a Tiled Imagery Layer.

                                                 To create Dynamic Imagery Layer as output on ArcGIS Online, set
                                                 ``tiles_only`` parameter to ``False``.

                                                 Function will not honor tiles_only parameter in ArcGIS Enterprise and
                                                 will generate Dynamic Imagery Layer by default.
        ====================================     ====================================================================

        :return: output_raster - :class:`~arcgis.raster.ImageryLayer` item

        .. code-block:: python

            # Usage Example

            img_lyr.save(output_name="saved_imagery_layer",
                         process_as_multidimensional=True,
                         build_transpose=True,
                         folder="my_imagery_layers",
                         gis=gis)
        """
        g = _arcgis.env.active_gis if gis is None else gis
        layer_extent_set = False
        gr_output = None

        if for_viz:
            if g._con._auth.lower() != "ANON".lower() and g._con._auth is not None:
                text_data = {
                    "id": "resultLayer",
                    "visibility": True,
                    "bandIds": [],
                    "opacity": 1,
                    "title": output_name,
                    "timeAnimation": False,
                    "renderingRule": self._fnra,
                    "mosaicRule": self._mosaic_rule,
                }
                ext = self.properties.initialExtent

                item_properties = {
                    "title": output_name,
                    "type": "Image Service",
                    "url": self._url,
                    "description": self.properties.description,
                    "tags": "imagery",
                    "extent": "{},{},{},{}".format(
                        ext["xmin"], ext["ymin"], ext["xmax"], ext["ymax"]
                    ),
                    "spatialReference": json.dumps(
                        dict(self.properties.spatialReference)
                    ),
                    "text": json.dumps(text_data),
                }

                return g.content.add(item_properties)
            else:
                raise RuntimeError("You need to be signed in to a GIS to create Items")
        else:
            from .analytics import is_supported, generate_raster, _save_ra

            if self._fnra is None:
                from .functions import identity

                identity_layer = identity(self)
                if isinstance(identity_layer, Raster):
                    if hasattr(identity_layer, "_engine_obj"):
                        self._fnra = identity_layer._engine_obj._fnra
                else:
                    self._fnra = identity_layer._fnra

            if is_supported(g):
                if self._extent is not None and _arcgis.env.analysis_extent is None:
                    _arcgis.env.analysis_extent = dict(self._extent)
                    layer_extent_set = True
                try:
                    if (self._uses_gbl_function) and (
                        ("use_ra" in self._other_outputs.keys())
                        and self._other_outputs["use_ra"] == True
                    ):
                        gr_output = _save_ra(
                            self._fnra,
                            output_name=output_name,
                            other_outputs=self._other_outputs,
                            gis=g,
                            future=future,
                            **kwargs,
                        )
                    else:
                        gr_output = generate_raster(
                            self._fnra,
                            output_name=output_name,
                            process_as_multidimensional=process_as_multidimensional,
                            build_transpose=build_transpose,
                            gis=g,
                            future=future,
                            context=context,
                            **kwargs,
                        )
                except Exception:
                    if layer_extent_set:
                        _arcgis.env.analysis_extent = None
                        layer_extent_set = False
                    raise

                if layer_extent_set:
                    _arcgis.env.analysis_extent = None
                    layer_extent_set = False
                if gr_output is not None:
                    return gr_output
            else:
                raise RuntimeError("This GIS does not support raster analysis.")

    def to_features(
        self,
        field: str = "Value",
        output_type: str = "Polygon",
        simplify: bool = True,
        output_name: Optional[str] = None,
        create_multipart_features: bool = False,
        max_vertices_per_feature: Optional[int] = None,
        context: Optional[dict] = None,
        *,
        gis: Optional[GIS] = None,
        future: bool = False,
        **kwargs,
    ):
        """
        The ``to_features`` method converts this raster to a persisted :class:`~arcgis.features.FeatureLayer`
        of the specified type using Raster Analytics.

        .. note::
            Distributed raster analysis is used for generating a new feature layer by
            applying raster functions at source resolution across the extent of the raster
            and performing a raster to features conversion.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        field                                    Optional string - field that specifies which value will be used for the conversion.
                                                 It can be any integer or a string field.

                                                 A field containing floating-point values can only be used if the output is to a point dataset.

                                                 Default is "Value"
        ------------------------------------     --------------------------------------------------------------------
        output_type                              Optional string.

                                                 One of the following: ['Point', 'Line', 'Polygon']
        ------------------------------------     --------------------------------------------------------------------
        simplify                                 Optional bool, This option that specifies how the features should be smoothed. It is
                                                 only available for line and polygon output.

                                                 True, then the features will be smoothed out. This is the default.

                                                 if False, then The features will follow exactly the cell boundaries of the raster dataset.
        ------------------------------------     --------------------------------------------------------------------
        output_name                              Optional. If not provided, an Feature layer is created by the method and used as the output

                                                 You can pass in an existing Feature Service Item from your GIS to use that instead.

                                                 Alternatively, you can pass in the name of the output Feature Service that should be created by this method
                                                 to be used as the output for the tool.

                                                 A RuntimeError is raised if a service by that name already exists
        ------------------------------------     --------------------------------------------------------------------
        create_multipart_features                Optional boolean. Specifies whether the output polygons will consist of
                                                 single-part or multipart features.

                                                 True: Specifies that multipart features will be created based on polygons that have the same value.

                                                 False: Specifies that individual features will be created for each polygon. This is the default.
        ------------------------------------     --------------------------------------------------------------------
        max_vertices_per_feature                 Optional int. The vertex limit used to subdivide a polygon into smaller polygons.
        ------------------------------------     --------------------------------------------------------------------
        context                                  context contains additional settings that affect task execution.

                                                 context parameter overwrites values set through arcgis.env parameter

                                                 This function has the following settings:

                                                  - Extent (extent): A bounding box that defines the analysis area.

                                                    Example:
                                                        {"extent": {"xmin": -122.68,
                                                        "ymin": 45.53,
                                                        "xmax": -122.45,
                                                        "ymax": 45.6,
                                                        "spatialReference": {"wkid": 4326}}}

                                                  - Output Spatial Reference (outSR): The output raster will be
                                                    projected into the output spatial reference.

                                                    Example:
                                                        {"outSR": {spatial reference}}
        ------------------------------------     --------------------------------------------------------------------
        gis                                      Optional GIS object. If not speficied, the currently active connection
                                                 is used.
        ------------------------------------     --------------------------------------------------------------------
        future                                   Optional boolean. If True, a future object will be returned and the process
                                                 will not wait for the task to complete. The default is False, which means wait for results.
        ====================================     ====================================================================

        :return: A :class:`~arcgis.features.FeatureLayer` item.

        .. code-block:: python

            # Usage Example

            feature_layer = img_lyr.to_features(output_type="Polygon",
                                                simplify = False,
                                                output_name="new_feature_layer",
                                                create_multipart_features = True,
                                                )

        """
        g = _arcgis.env.active_gis if gis is None else gis

        from arcgis.raster.analytics import convert_raster_to_feature

        return convert_raster_to_feature(
            self,
            field,
            output_type,
            simplify,
            output_name,
            create_multipart_features=create_multipart_features,
            max_vertices_per_feature=max_vertices_per_feature,
            gis=g,
            future=future,
            context=context,
            **kwargs,
        )

    def draw_graph(
        self, show_attributes: bool = False, graph_size: str = "14.25, 15.25"
    ):
        """
        The ``draw_graph`` method displays a structural representation of the function chain and it's
        raster input values. If ``show_attributes`` is set to ``True``, then the ``draw_graph`` method also
        displays the attributes
        of all the functions in the function chain, representing the rasters in a blue rectangular
        box, attributes in green rectangular box and the raster function names in yellow.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        show_attributes       Optional boolean. If ``True``, the graph displayed includes all the
                              attributes of the function and not only it's function name and raster
                              inputs
                              Set to False by default, to display only the raster function name and
                              the raster inputs to it.
        -----------------     --------------------------------------------------------------------
        graph_size            Optional string. Maximum width and height of drawing, in inches,
                              seperated by a comma. If only a single number is given, this is used
                              for both the width and the height. If defined and the drawing is
                              larger than the given size, the drawing is uniformly scaled down so
                              that it fits within the given size.
        =================     ====================================================================

        :return: A Graph

        .. code-block:: python

            # Usage Example: Draws the function chain applied on the ImageryLayer object created from an Image service.

            imagery_layer = ImageryLayer("https://myserver/arcgis/rest/services/ImageServiceName/ImageServer", gis=gis)
            grayscale_layer = grayscale(raster=imagery_layer)
            invert_layer = boolean_not(rasters=[grayscale_layer])
            invert_layer.draw_graph(show_attributes=True)

        """
        import re
        import numbers
        from operator import eq

        try:
            from graphviz import Digraph
        except:
            print("Graphviz needs to be installed. pip install graphviz")
        from .functions.utility import _find_object_ref

        global nodenumber, root
        nodenumber = root = 0
        function_dictionary = self._fnra

        global dict_arg
        dict_arg = {}

        if function_dictionary is None:
            return "No raster function has been applied on the Imagery Layer"

        def _raster_slicestring(slice_string, **kwargs):
            try:
                subString = re.search(
                    "/services/(.+?)/ImageServer", slice_string
                ).group(1)
            except AttributeError:
                if slice_string.startswith("$"):
                    if "url" in kwargs.keys():
                        return _raster_slicestring(kwargs["url"])
                elif (
                    "/fileShares/" in slice_string
                    or "/rasterStores/" in slice_string
                    or "/cloudStores/" in slice_string
                    or "/vsi" in slice_string
                ):
                    slice_string = slice_string.rsplit("/", 1)[1]
                elif "\\" in slice_string:
                    slice_string = slice_string.rsplit("\\", 1)[1]
                if "https" in slice_string and any(
                    x in slice_string for x in ["MapServer", "FeatureServer"]
                ):
                    slice_string = None
                subString = slice_string
            return subString

        hidden_inputs = [
            "ToolName",
            "PrimaryInputParameterName",
            "OutputRasterParameterName",
        ]
        G = Digraph(
            comment="Raster Function Chain", format="svg"
        )  # To declare the graph
        G.clear()  # clear all previous cases of the same named
        G.attr(
            rankdir="LR",
            len="1",
            splines="ortho",
            nodesep="0.5",
            size=graph_size,
        )  # Display graph from Left to Right

        def _draw_graph(
            self,
            show_attributes,
            function_dictionary=None,
            G=None,
            dg_nodenumber=None,
            dg_root=None,
            **kwargs,
        ):  # regular fnra
            global nodenumber, root

            if dg_nodenumber:
                nodenumber = dg_nodenumber

            if dg_root:
                root = dg_root

            def _toolname_slicestring(slice_string):
                try:
                    subString = re.search("(.+?)_sa", slice_string).group(1)
                except AttributeError:
                    subString = slice_string
                return subString

            def _raster_function_graph(rfa_value, rfa_key, connect, **kwargs):
                global nodenumber
                if isinstance(rfa_value, dict):
                    if "rasterFunction" in rfa_value.keys():
                        _function_graph(rfa_value, rfa_key, connect, **kwargs)

                    if "url" in rfa_value.keys():
                        nodenumber += 1
                        rastername = _raster_slicestring(str(rfa_value["url"]))
                        if rastername is not None:
                            G.node(
                                str(nodenumber),
                                rastername,
                                style=("filled"),
                                shape="note",
                                color="darkseagreen2",
                                fillcolor="darkseagreen2",
                                fontname="sans-serif",
                            )
                            G.edge(
                                str(nodenumber),
                                str(connect),
                                color="#BEBEBE",
                                arrowsize="0.9",
                                penwidth="1",
                            )

                    if "uri" in rfa_value.keys():
                        nodenumber += 1
                        rastername = _raster_slicestring(str(rfa_value["uri"]))
                        if rastername is not None:
                            G.node(
                                str(nodenumber),
                                rastername,
                                style=("filled"),
                                shape="note",
                                color="darkseagreen2",
                                fillcolor="darkseagreen2",
                                fontname="sans-serif",
                            )
                            G.edge(
                                str(nodenumber),
                                str(connect),
                                color="#BEBEBE",
                                arrowsize="0.9",
                                penwidth="1",
                            )

                    elif "function" in rfa_value.keys():
                        _rft_draw_graph(G, rfa_value, rfa_key, connect, show_attributes)

                elif isinstance(rfa_value, list):
                    for rfa_value_search_dict in rfa_value:
                        if isinstance(rfa_value_search_dict, dict):
                            for rfa_value_search_key in rfa_value_search_dict.keys():
                                if rfa_value_search_key == "rasterFunction":
                                    _function_graph(
                                        rfa_value_search_dict,
                                        rfa_key,
                                        connect,
                                        **kwargs,
                                    )

                        elif isinstance(rfa_value_search_dict, numbers.Number):
                            nodenumber += 1
                            rastername = str(rfa_value_search_dict)
                            G.node(
                                str(nodenumber),
                                rastername,
                                style=("filled"),
                                fixedsize="shape",
                                width=".75",
                                shape="circle",
                                color="darkslategray2",
                                fillcolor="darkslategray2",
                                fontname="sans-serif",
                            )
                            G.edge(
                                str(nodenumber),
                                str(connect),
                                color="#BEBEBE",
                                arrowsize="0.9",
                                penwidth="1",
                            )
                        else:
                            nodenumber += 1
                            rastername = _raster_slicestring(str(rfa_value_search_dict))
                            G.node(
                                str(nodenumber),
                                rastername,
                                style=("filled"),
                                shape="note",
                                color="darkseagreen2",
                                fillcolor="darkseagreen2",
                                fontname="sans-serif",
                            )
                            G.edge(
                                str(nodenumber),
                                str(connect),
                                color="#BEBEBE",
                                arrowsize="0.9",
                                penwidth="1",
                            )

                elif isinstance(rfa_value, int) or isinstance(rfa_value, float):
                    nodenumber += 1
                    rastername = str(rfa_value)
                    G.node(
                        str(nodenumber),
                        rastername,
                        style=("filled"),
                        fixedsize="shape",
                        width=".75",
                        shape="circle",
                        color="darkslategray2",
                        fillcolor="darkslategray2",
                        fontname="sans-serif",
                    )
                    G.edge(
                        str(nodenumber),
                        str(connect),
                        color="#BEBEBE",
                        arrowsize="0.9",
                        penwidth="1",
                    )

                elif isinstance(rfa_value, str):
                    nodenumber += 1
                    if "url" in kwargs.keys():
                        rastername = _raster_slicestring(rfa_value, url=kwargs["url"])
                    else:
                        rastername = _raster_slicestring(rfa_value)
                    if rastername is not None:
                        G.node(
                            str(nodenumber),
                            rastername,
                            style=("filled"),
                            shape="note",
                            color="darkseagreen2",
                            fillcolor="darkseagreen2",
                            fontname="sans-serif",
                        )
                        G.edge(
                            str(nodenumber),
                            str(connect),
                            color="#BEBEBE",
                            arrowsize="0.9",
                            penwidth="1",
                        )

            def _attribute_function_graph(rfa_value, rfa_key, root):
                global nodenumber
                nodenumber += 1
                rastername = rfa_key + " = " + str(rfa_value)
                G.node(
                    str(nodenumber),
                    rastername,
                    style=("filled"),
                    shape="rectangle",
                    color="antiquewhite",
                    fillcolor="antiquewhite",
                    fontname="sans-serif",
                )
                G.edge(
                    str(nodenumber),
                    str(root),
                    color="#BEBEBE",
                    arrowsize="0.9",
                    penwidth="1",
                )

            def _function_graph(dictionary, childnode, connect, **kwargs):
                global nodenumber, root
                if isinstance(dictionary, dict):
                    for dkey, dvalue in dictionary.items():
                        if dkey == "rasterFunction" and dvalue != "GPAdapter":
                            if (
                                dvalue == "Identity"
                                and "renderingRule"
                                in dictionary["rasterFunctionArguments"]["Raster"]
                            ):
                                if (
                                    "rasterFunction"
                                    in dictionary["rasterFunctionArguments"]["Raster"][
                                        "renderingRule"
                                    ]
                                ):
                                    _function_graph(
                                        dictionary["rasterFunctionArguments"]["Raster"][
                                            "renderingRule"
                                        ],
                                        "Raster",
                                        connect,
                                        url=dictionary["rasterFunctionArguments"][
                                            "Raster"
                                        ]["url"],
                                    )

                            else:
                                nodenumber += 1
                                G.node(
                                    str(nodenumber),
                                    dvalue,
                                    style=("rounded, filled"),
                                    shape="box",
                                    color="lightgoldenrod1",
                                    fillcolor="lightgoldenrod1",
                                    fontname="sans-serif",
                                )
                                if connect > 0:
                                    G.edge(
                                        str(nodenumber),
                                        str(connect),
                                        color="#BEBEBE",
                                        arrowsize="0.9",
                                        penwidth="1",
                                    )
                                connect = nodenumber
                                for (
                                    dkey,
                                    dvalue,
                                ) in (
                                    dictionary.items()
                                ):  # Check dictionary again for rasterFunctionArguments
                                    if dkey == "rasterFunctionArguments":
                                        for key, value in dvalue.items():
                                            if (
                                                key == "Raster"
                                                or key == "Raster2"
                                                or key == "Rasters"
                                                or key == "PanImage"
                                                or key == "MSImage"
                                            ):
                                                _raster_function_graph(
                                                    value,
                                                    key,
                                                    connect,
                                                    **kwargs,
                                                )

                                            elif show_attributes == True:
                                                _attribute_function_graph(
                                                    value, key, connect
                                                )

                        elif (
                            dkey == "rasterFunction" and dvalue == "GPAdapter"
                        ):  # To handle global function arguments
                            for rf_key, rf_value in dictionary.items():
                                if rf_key == "rasterFunctionArguments":
                                    for (
                                        gbl_key,
                                        gbl_value,
                                    ) in rf_value.items():
                                        if gbl_key == "toolName":
                                            toolname = _toolname_slicestring(gbl_value)
                                            nodenumber += 1
                                            G.node(
                                                str(nodenumber),
                                                toolname,
                                                style=("rounded, filled"),
                                                shape="box",
                                                color="lightgoldenrod1",
                                                fillcolor="lightgoldenrod1",
                                                fontname="sans-serif",
                                            )
                                            G.edge(
                                                str(nodenumber),
                                                str(connect),
                                                color="#BEBEBE",
                                                arrowsize="0.9",
                                                penwidth="1",
                                            )
                                            connect = nodenumber
                                        elif (
                                            gbl_key.endswith("_raster")
                                            or gbl_key.endswith("_data")
                                            or gbl_key.endswith("_features")
                                        ):  # To check if rasterFunctionArguments has rasters in it
                                            _raster_function_graph(
                                                gbl_value,
                                                gbl_key,
                                                connect,
                                                **kwargs,
                                            )

                                        elif (
                                            show_attributes == True
                                            and gbl_key != "PrimaryInputParameterName"
                                            and gbl_key != "OutputRasterParameterName"
                                        ):
                                            _attribute_function_graph(
                                                gbl_value, gbl_key, connect
                                            )
                        elif dkey == "function":
                            _rft_draw_graph(
                                G,
                                dictionary,
                                nodenumber,
                                connect,
                                show_attributes,
                            )

                # To find first rasterFunction

            for dkey, dvalue in function_dictionary.items():
                if (
                    dkey == "rasterFunction" and dvalue != "GPAdapter"
                ):  # To find first rasterFunction
                    if (
                        dvalue == "Identity"
                        and "renderingRule"
                        in function_dictionary["rasterFunctionArguments"]["Raster"]
                    ):
                        if (
                            "rasterFunction"
                            in function_dictionary["rasterFunctionArguments"]["Raster"][
                                "renderingRule"
                            ]
                        ):  # if the first raster function is a rendering rule applied on an image service
                            _function_graph(
                                function_dictionary["rasterFunctionArguments"][
                                    "Raster"
                                ]["renderingRule"],
                                None,
                                root,
                                url=function_dictionary["rasterFunctionArguments"][
                                    "Raster"
                                ]["url"],
                            )
                        else:
                            return "No raster function applied"
                    else:
                        root += 1
                        G.node(
                            str(root),
                            dvalue,
                            style=("rounded, filled"),
                            shape="box",
                            color="lightgoldenrod1",
                            fillcolor="lightgoldenrod1",
                            fontname="sans-serif",
                        )  # create first rasterFunction graph node
                        nodenumber = root
                        if (root - 1) > 0:
                            G.edge(
                                str(root),
                                str(dg_root),
                                color="#BEBEBE",
                                arrowsize="0.9",
                                penwidth="1",
                            )
                            temproot = root
                        for rf_key, rf_value in function_dictionary.items():
                            if (
                                rf_key == "rasterFunctionArguments"
                            ):  # To check dictionary again for rasterFunctionArguments
                                for rfa_key, rfa_value in rf_value.items():
                                    if (
                                        rfa_key == "rasterFunction"
                                    ):  # To check if rasterFunctionArguments has another rasterFunction chain in it
                                        _function_graph(rfa_value, rfa_key, nodenumber)
                                    elif (
                                        rfa_key == "Raster"
                                        or rfa_key == "Raster2"
                                        or rfa_key == "Rasters"
                                        or rfa_key == "PanImage"
                                        or rfa_key == "MSImage"
                                    ):  # To check if rasterFunctionArguments includes raster inputs in it
                                        temproot = root
                                        _raster_function_graph(rfa_value, rfa_key, root)
                                    elif show_attributes == True:
                                        temproot = root
                                        _attribute_function_graph(
                                            rfa_value, rfa_key, temproot
                                        )

                elif (
                    dkey == "rasterFunction" and dvalue == "GPAdapter"
                ):  # To handle global function arguments
                    for rf_key, rf_value in function_dictionary.items():
                        if rf_key == "rasterFunctionArguments":
                            for gbl_key, gbl_value in rf_value.items():
                                if gbl_key == "toolName":
                                    toolname = _toolname_slicestring(gbl_value)
                                    # To check if rasterFunctionArguments has another rasterFunction chain in it
                                    root += 1
                                    G.node(
                                        str(root),
                                        toolname,
                                        style=("rounded, filled"),
                                        shape="box",
                                        color="lightgoldenrod1",
                                        fillcolor="lightgoldenrod1",
                                        fontname="sans-serif",
                                    )
                                    nodenumber = root
                                    if (root - 1) > 0:
                                        G.edge(
                                            str(root),
                                            str(dg_root),
                                            color="#BEBEBE",
                                            arrowsize="0.9",
                                            penwidth="1",
                                        )
                                    nodenumber = root
                                elif (
                                    gbl_key.endswith("_raster")
                                    or gbl_key.endswith("_data")
                                    or gbl_key.endswith("_features")
                                ):  # To check if rasterFunctionArguments includes raster inputs in it
                                    _raster_function_graph(gbl_value, gbl_key, root)
                                elif (
                                    show_attributes == True
                                    and gbl_key != "PrimaryInputParameterName"
                                    and gbl_key != "OutputRasterParameterName"
                                ):
                                    _attribute_function_graph(gbl_value, gbl_key, root)
                elif dkey == "function":
                    _rft_draw_graph(
                        G,
                        function_dictionary,
                        nodenumber,
                        root,
                        show_attributes,
                        **kwargs,
                    )

            return G

        def _rft_draw_graph(
            G, gdict, gnodenumber, groot, show_attributes, **kwargs
        ):  # rft fnra
            global nodenumber, connect, root
            global dict_arg

            def _rft_function_create(value, childnode, **kwargs):
                global nodenumber
                dict_temp_arg = {}
                check_empty_graph = Digraph()
                list_arg = []
                flag = 0
                # save function chain in order to avoid function chain duplicating
                for k_func, v_func in value["function"].items():
                    if k_func == "name":
                        list_arg.append(k_func + str(v_func))
                for k_arg, v_arg in value["arguments"].items():
                    list_arg.append(k_arg + str(v_arg))

                list_arg.sort()
                list_arg_str = str(list_arg)
                if (
                    dict_arg is not None
                ):  # if function chain is repeating connect to respective node
                    for k_check in dict_arg.keys():
                        if k_check == list_arg_str:
                            G.edge(
                                str(dict_arg.get(k_check)),
                                str(childnode),
                                color="#BEBEBE",
                                arrowsize="0.9",
                                penwidth="1",
                            )
                            flag = 1

                if flag == 0:  # New function chain
                    nodenumber += 1
                    G.node(
                        str(nodenumber),
                        value["function"]["name"],
                        style=("rounded, filled"),
                        shape="box",
                        color="lightgoldenrod1",
                        fillcolor="lightgoldenrod1",
                        fontname="sans-serif",
                    )

                    if nodenumber > 0:
                        G.edge(
                            str(nodenumber),
                            str(childnode),
                            color="#BEBEBE",
                            arrowsize="0.9",
                            penwidth="1",
                        )

                    connect = nodenumber
                    dict_temp_arg = {list_arg_str: connect}
                    dict_arg.update(dict_temp_arg)
                    if "isDataset" in value["arguments"].keys():
                        if value["arguments"]["isDataset"] == False:
                            for arg_element in value["arguments"]["value"]["elements"]:
                                _rft_raster_function_graph(
                                    arg_element, connect, **kwargs
                                )
                        elif value["arguments"]["isDataset"] == True:
                            _rft_raster_function_graph(
                                value["arguments"], connect, **kwargs
                            )  # Rf which only have 1 parameter

                    _rft_function_graph(value["arguments"], connect, **kwargs)

            def _rft_raster_function_graph(
                raster_dict, childnode, **kwargs
            ):  # If isDataset=True
                global nodenumber, connect
                if "rasterFunction" in raster_dict.keys():
                    _draw_graph(
                        self,
                        show_attributes,
                        raster_dict,
                        G,
                        nodenumber,
                        childnode,
                    )
                elif "value" in raster_dict.keys():
                    if raster_dict["value"] is not None:
                        if (
                            isinstance(raster_dict["value"], numbers.Number)
                            or "value" in raster_dict["value"]
                        ):  # ***Handling Scalar rasters***
                            if isinstance(raster_dict["value"], numbers.Number):
                                nodenumber += 1
                                G.node(
                                    str(nodenumber),
                                    str(raster_dict["value"]),
                                    style=("filled"),
                                    fontsize="12",
                                    shape="circle",
                                    fixedsize="shape",
                                    color="darkslategray2",
                                    fillcolor="darkslategray2",
                                    fontname="sans-serif",
                                )
                                G.edge(
                                    str(nodenumber),
                                    str(childnode),
                                    color="#BEBEBE",
                                    arrowsize="0.9",
                                    penwidth="1",
                                )
                            elif isinstance(
                                raster_dict["value"]["value"], numbers.Number
                            ):
                                nodenumber += 1
                                G.node(
                                    str(nodenumber),
                                    str(raster_dict["value"]["value"]),
                                    style=("filled"),
                                    fontsize="12",
                                    shape="circle",
                                    fixedsize="shape",
                                    color="darkslategray2",
                                    fillcolor="darkslategray2",
                                    fontname="sans-serif",
                                )
                                G.edge(
                                    str(nodenumber),
                                    str(childnode),
                                    color="#BEBEBE",
                                    arrowsize="0.9",
                                    penwidth="1",
                                )

                        elif "elements" in raster_dict["value"]:
                            ev_list = "n"
                            if "elements" in raster_dict["value"]:
                                ev_list = raster_dict["value"]["elements"]
                            else:
                                ev_list = raster_dict["value"]
                            for e in ev_list:
                                if isinstance(e, dict):
                                    if (
                                        "function" in e.keys()
                                    ):  # if function template inside
                                        _rft_function_graph(e, childnode)
                                    elif (
                                        "url" in e.keys()
                                        or "uri" in e.keys()
                                        or ("type" in e and e["type"] == "Scalar")
                                    ):
                                        _rft_raster_function_graph(e, childnode)
                                    else:  # if raster dataset inside raster array
                                        _rft_raster_function_graph(e, childnode)
                                else:
                                    nodenumber += 1
                                    G.node(
                                        str(nodenumber),
                                        str(e),
                                        style=("filled"),
                                        fontsize="12",
                                        shape="circle",
                                        fixedsize="shape",
                                        color="darkslategray2",
                                        fillcolor="darkslategray2",
                                        fontname="sans-serif",
                                    )
                                    G.edge(
                                        str(nodenumber),
                                        str(childnode),
                                        color="#BEBEBE",
                                        arrowsize="0.9",
                                        penwidth="1",
                                    )

                            if (
                                ev_list == "n"
                            ):  # if no value in rasters when the rft was made
                                nodenumber += 1
                                G.node(
                                    str(nodenumber),
                                    str(raster_dict["name"]),
                                    style=("filled"),
                                    shape="note",
                                    color="darkseagreen2",
                                    fillcolor="darkseagreen2",
                                    fontname="sans-serif",
                                )
                                G.edge(
                                    str(nodenumber),
                                    str(childnode),
                                    color="#BEBEBE",
                                    arrowsize="0.9",
                                    penwidth="1",
                                )
                                # If elements is empty i.e Rasters has no value when rft was created

                        elif "function" in raster_dict["value"]:
                            _rft_function_graph(raster_dict, childnode)
                        elif (
                            "name" in raster_dict["value"]
                        ):  # if raster properties are preserved
                            nodenumber += 1
                            G.node(
                                str(nodenumber),
                                str(raster_dict["value"]["name"]),
                                style=("filled"),
                                shape="note",
                                color="darkseagreen2",
                                fillcolor="darkseagreen2",
                                fontname="sans-serif",
                            )
                            G.edge(
                                str(nodenumber),
                                str(childnode),
                                color="#BEBEBE",
                                arrowsize="0.9",
                                penwidth="1",
                            )
                        elif (
                            "url" in raster_dict["value"]
                        ):  # if raster properties are preserved
                            nodenumber += 1
                            rastername = _raster_slicestring(
                                str(raster_dict["value"]["url"])
                            )
                            if rastername is not None:
                                G.node(
                                    str(nodenumber),
                                    rastername,
                                    style=("filled"),
                                    shape="note",
                                    color="darkseagreen2",
                                    fillcolor="darkseagreen2",
                                    fontname="sans-serif",
                                )
                                G.edge(
                                    str(nodenumber),
                                    str(childnode),
                                    color="#BEBEBE",
                                    arrowsize="0.9",
                                    penwidth="1",
                                )
                        elif (
                            "uri" in raster_dict["value"]
                        ):  # if raster properties are preserved
                            nodenumber += 1
                            rastername = _raster_slicestring(
                                str(raster_dict["value"]["uri"])
                            )
                            if rastername is not None:
                                G.node(
                                    str(nodenumber),
                                    rastername,
                                    style=("filled"),
                                    shape="note",
                                    color="darkseagreen2",
                                    fillcolor="darkseagreen2",
                                    fontname="sans-serif",
                                )
                                G.edge(
                                    str(nodenumber),
                                    str(childnode),
                                    color="#BEBEBE",
                                    arrowsize="0.9",
                                    penwidth="1",
                                )
                        elif (
                            "datasetName" in raster_dict["value"]
                        ):  # local image location
                            if "name" in raster_dict["value"]["datasetName"]:
                                nodenumber += 1
                                G.node(
                                    str(nodenumber),
                                    str(raster_dict["value"]["datasetName"]["name"]),
                                    style=("filled"),
                                    shape="note",
                                    color="darkseagreen2",
                                    fillcolor="darkseagreen2",
                                    fontname="sans-serif",
                                )
                                G.edge(
                                    str(nodenumber),
                                    str(childnode),
                                    color="#BEBEBE",
                                    arrowsize="0.9",
                                    penwidth="1",
                                )

                        elif isinstance(
                            raster_dict["value"], list
                        ):  # raster_dict"value" does not have "value" or "elements" in it (ArcMap scalar rft case)
                            for x in raster_dict["value"]:
                                if isinstance(
                                    x, numbers.Number
                                ):  # Check if scalar float value
                                    nodenumber += 1
                                    G.node(
                                        str(nodenumber),
                                        str(x),
                                        style=("filled"),
                                        fontsize="12",
                                        shape="circle",
                                        fixedsize="shape",
                                        color="darkslategray2",
                                        fillcolor="darkslategray2",
                                        fontname="sans-serif",
                                    )
                                    G.edge(
                                        str(nodenumber),
                                        str(childnode),
                                        color="#BEBEBE",
                                        arrowsize="0.9",
                                        penwidth="1",
                                    )

                                elif isinstance(x, dict):
                                    if (
                                        "url" in x
                                        or "uri" in x
                                        or ("type" in x and x["type"] == "Scalar")
                                        or ("isDataset" in x and x["isDataset"] == True)
                                        or (
                                            "value" in x
                                            and isinstance(x["value"], dict)
                                        )
                                    ):
                                        _rft_raster_function_graph(
                                            x, childnode, **kwargs
                                        )
                                    else:
                                        _rft_function_graph(x, childnode, **kwargs)

                elif "url" in raster_dict.keys():  # Handling Raster
                    nodenumber += 1
                    rastername = _raster_slicestring(str(raster_dict["url"]))
                    if rastername is not None:
                        G.node(
                            str(nodenumber),
                            rastername,
                            style=("filled"),
                            shape="note",
                            color="darkseagreen2",
                            fillcolor="darkseagreen2",
                            fontname="sans-serif",
                        )
                        G.edge(
                            str(nodenumber),
                            str(childnode),
                            color="#BEBEBE",
                            arrowsize="0.9",
                            penwidth="1",
                        )
                elif "uri" in raster_dict.keys():
                    nodenumber += 1
                    rastername = _raster_slicestring(str(raster_dict["uri"]))
                    if rastername is not None:
                        G.node(
                            str(nodenumber),
                            rastername,
                            style=("filled"),
                            shape="note",
                            color="darkseagreen2",
                            fillcolor="darkseagreen2",
                            fontname="sans-serif",
                        )
                        G.edge(
                            str(nodenumber),
                            str(childnode),
                            color="#BEBEBE",
                            arrowsize="0.9",
                            penwidth="1",
                        )
                elif (
                    "datasetName" in raster_dict.keys()
                    and "name" in raster_dict["datasetName"]
                ):  # if RasterInfo rf has data in it
                    rastername = str(raster_dict["datasetName"]["name"])
                    nodenumber += 1
                    G.node(
                        str(nodenumber),
                        rastername,
                        style=("filled"),
                        shape="note",
                        color="darkseagreen2",
                        fillcolor="darkseagreen2",
                        fontname="sans-serif",
                    )
                    G.edge(
                        str(nodenumber),
                        str(childnode),
                        color="#BEBEBE",
                        arrowsize="0.9",
                        penwidth="1",
                    )
                elif "name" in raster_dict:
                    rastername = str(raster_dict["name"])  # Handling Raster
                    nodenumber += 1
                    G.node(
                        str(nodenumber),
                        rastername,
                        style=("filled"),
                        shape="note",
                        color="darkseagreen2",
                        fillcolor="darkseagreen2",
                        fontname="sans-serif",
                    )
                    G.edge(
                        str(nodenumber),
                        str(childnode),
                        color="#BEBEBE",
                        arrowsize="0.9",
                        penwidth="1",
                    )

            def _rft_function_graph(dictionary, childnode, **kwargs):
                global nodenumber, connect
                count = 0
                if "function" in dictionary:
                    _rft_function_create(dictionary, childnode)

                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        if "isDataset" in value.keys():
                            if (
                                (value["isDataset"] == True)
                                or key == "Raster"
                                or key == "Raster2"
                                or key == "Rasters"
                            ):
                                _rft_raster_function_graph(value, childnode)
                            elif (
                                value["isDataset"] == False
                            ) and show_attributes == True:  # Show Parameters
                                if "value" in value.keys():
                                    if isinstance(value["value"], dict):
                                        if "elements" not in value["value"]:
                                            nodenumber += 1
                                            if "value" in value:
                                                if value[
                                                    "value"
                                                ] is not None or isinstance(
                                                    value["value"], bool
                                                ):
                                                    atrr_name = (
                                                        str(value["name"])
                                                        + " = "
                                                        + str(value["value"])
                                                    )
                                            else:
                                                atrr_name = str(value["name"])
                                                G.node(
                                                    str(nodenumber),
                                                    atrr_name,
                                                    style=("filled"),
                                                    shape="rectangle",
                                                    color="antiquewhite",
                                                    fillcolor="antiquewhite",
                                                    fontname="sans-serif",
                                                )
                                                G.edge(
                                                    str(nodenumber),
                                                    str(childnode),
                                                    color="#BEBEBE",
                                                    arrowsize="0.9",
                                                    penwidth="1",
                                                )
                                    else:
                                        if (
                                            "name" in value
                                            and value["name"] not in hidden_inputs
                                        ):
                                            nodenumber += 1
                                            if value["value"] is not None or isinstance(
                                                value["value"], bool
                                            ):
                                                atrr_name = (
                                                    str(value["name"])
                                                    + " = "
                                                    + str(value["value"])
                                                )
                                            else:
                                                atrr_name = str(value["name"])

                                            G.node(
                                                str(nodenumber),
                                                atrr_name,
                                                style=("filled"),
                                                shape="rectangle",
                                                color="antiquewhite",
                                                fillcolor="antiquewhite",
                                                fontname="sans-serif",
                                            )
                                            G.edge(
                                                str(nodenumber),
                                                str(childnode),
                                                color="#BEBEBE",
                                                arrowsize="0.9",
                                                penwidth="1",
                                            )
                                else:
                                    nodenumber += 1
                                    atrr_name = str(value["name"])
                                    G.node(
                                        str(nodenumber),
                                        atrr_name,
                                        style=("filled"),
                                        shape="rectangle",
                                        color="antiquewhite",
                                        fillcolor="antiquewhite",
                                        fontname="sans-serif",
                                    )
                                    G.edge(
                                        str(nodenumber),
                                        str(childnode),
                                        color="#BEBEBE",
                                        arrowsize="0.9",
                                        penwidth="1",
                                    )

                        elif "datasetName" in value.keys():
                            _rft_raster_function_graph(value, childnode)
                        elif "url" in value.keys():
                            _rft_raster_function_graph(value, childnode)
                        elif "function" in value.keys():  # Function Chain inside Raster
                            _rft_function_create(value, childnode)
                        elif "rasterFunction" in value.keys():
                            _draw_graph(
                                self,
                                show_attributes,
                                value,
                                G,
                                nodenumber,
                                childnode,
                            )  # regular fnra

            # nodenumber=gnodenumber
            if "function" in gdict.keys():
                if groot == 0:  # Check if graph is empty
                    flag_graph = 1
                    root = groot + 1
                    # print(gdict["function"])
                    if "name" in gdict["function"]:
                        G.node(
                            str(root),
                            gdict["function"]["name"],
                            style=("rounded, filled"),
                            shape="box",
                            color="lightgoldenrod1",
                            fillcolor="lightgoldenrod1",
                            fontname="sans-serif",
                        )
                        nodenumber = root + 1

                else:
                    flag_graph = 2
                    root = groot
                if "isDataset" in gdict["arguments"]:
                    if gdict["arguments"]["isDataset"] == False:
                        if "value" in gdict["arguments"]:
                            if "elements" in gdict["arguments"]["value"]:
                                if gdict["arguments"]["value"]["elements"]:
                                    for arg_element in gdict["arguments"]["value"][
                                        "elements"
                                    ]:
                                        _rft_function_graph(arg_element, root, **kwargs)
                            else:
                                _rft_raster_function_graph(gdict["arguments"], root)
                        else:
                            _rft_raster_function_graph(gdict["arguments"], root)
                    else:
                        _rft_function_graph(gdict["arguments"]["value"], root, **kwargs)
                elif "datasetName" in gdict.keys():
                    _rft_raster_function_graph(gdict, root)

                if flag_graph == 1:
                    _rft_function_graph(
                        gdict["arguments"], root, **kwargs
                    )  # send only arguments of the first function to be processed
                else:
                    _rft_function_graph(
                        gdict, root, **kwargs
                    )  # Send entire dictionary back to be processed

            return G

        return _draw_graph(self, show_attributes, function_dictionary, G)

    def temporal_profile(
        self,
        points: list[Point] = [],
        time_field: Optional[str] = None,
        variables: list[str] = [],
        bands: list[int] = [0],
        time_extent: Optional[list[datetime.datetime]] = None,
        dimension: Optional[list[str]] = None,
        dimension_values: Optional[list[float]] = [],
        show_values: bool = False,
        trend_type: Optional[str] = None,
        trend_order: Optional[int] = None,
        plot_properties: dict = {},
    ):
        """
        The ``temporal_profile`` method creates a temporal profile.
        A temporal profile serves as a basic analysis tool for imagery data in a time series.
        Visualizing change over time with the temporal profile allows trends to be displayed
        and compared with variables, bands, or values from other dimensions simultaneously.

        Using the functionality in temporal profile charts, you can perform trend analysis, gain insight into
        multidimensional raster data at given locations, and plot values that are changing over time
        in the form of a line graph.

        Temporal profile charts can be used in various scientific applications involving time series
        analysis of raster data, and the graphical output of results can be used directly as
        input for strategy management and decision making.

        The x-axis of the temporal profile displays the time in continuous time intervals. The time field is
        obtained from the timeInfo of the image service.

        The y-axis of the temporal profile displays the variable value.


        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        points                                   Required list of :class:`~arcgis.geometry.Point` objects.
        ------------------------------------     --------------------------------------------------------------------
        time_field                               Required string. The time field that will be used for plotting
                                                 temporal profile.

                                                 If not specified the time field is obtained from the timeInfo of
                                                 the image service.
        ------------------------------------     --------------------------------------------------------------------
        variables                                Required list of strings. The variables that will be used for plotting temporal profile.
                                                 For non multidimensional data, the variable would be name of the Sensor.
                                                 To plot the graph against all sensors specify - "ALL_SENSORS"
        ------------------------------------     --------------------------------------------------------------------
        bands                                    Optional list of integers. Band indices to be used for plotting temporal profile.
                                                 By default takes the first band (band index - 0).
                                                 For a multiband data, you can compare the time change of different
                                                 bands over different locations.
        ------------------------------------     --------------------------------------------------------------------
        time_extent                              Optional list of datetime objects. This represents the time extent.
        ------------------------------------     --------------------------------------------------------------------
        dimension                                Optional list of strings. The dimension names that will be used for plotting temporal profile.
                                                 This option works specifically on multidimensional data containing a time dimension and other dimensions.

                                                 The temporal profile is created based on the specific values in other
                                                 dimensions, such as depth at the corresponding time value. For example,
                                                 soil moisture data usually includes both a time dimension and vertical
                                                 dimension below the earth's surface, resulting in a temporal profile
                                                 at 0.1, 0.2, and 0.3 meters below the ground.
        ------------------------------------     --------------------------------------------------------------------
        dimension_values                         Optional list of floats. This parameter can be used to specify
                                                 the values of dimension parameter other than the time dimension (dimension
                                                 name specified using dimension parameter)
        ------------------------------------     --------------------------------------------------------------------
        show_values                              Optional boolean. Default False.
                                                 Set this parameter to True to display the values at each point in the line graph.
        ------------------------------------     --------------------------------------------------------------------
        trend_type                               Optional string. Default None.
                                                 Set the trend_type parameter to either linear or harmonic to draw the trend line.
                                                 linear : Fits the pixel values for a variable along a linear trend line.
                                                 harmonic : Fits the pixel values for a variable along a harmonic trend line.
        ------------------------------------     --------------------------------------------------------------------
        trend_order                              Optional integer. The frequency number to use in the trend fitting.
                                                 This parameter specifies the frequency of cycles in a year.
                                                 The default value is 1, or one harmonic cycle per year.

                                                 This parameter is only included in the trend analysis for a harmonic regression.
        ------------------------------------     --------------------------------------------------------------------
        plot_properties                          Optional dictionary. This parameter can be used to set the figure
                                                 properties. These are the `matplotlib.pyplot.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_
                                                 parameters and values specified in dictionary format.

                                                 eg: {"figsize":(15,15)}
        ====================================     ====================================================================

        :return:
            None

        """
        from arcgis.raster._charts import temporal_profile

        return temporal_profile(
            self,
            points=points,
            time_field=time_field,
            variables=variables,
            bands=bands,
            time_extent=time_extent,
            dimension=dimension,
            dimension_values=dimension_values,
            show_values=show_values,
            trend_type=trend_type,
            trend_order=trend_order,
            plot_properties=plot_properties,
        )

    def render_tilesonly_layer(
        self, level: Optional[int] = None, slice_id: Optional[int] = None
    ):
        """
        The ``render_tilesonly_layer`` method renders a tiles-only :class:`~arcgis.raster.ImageryLayer` at a given
        level.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        level                                    Optional integer. Level to be used for rendering.
                                                 Default value is 0.
        ------------------------------------     --------------------------------------------------------------------
        slice_id                                 Optional integer. Renders the given slice of a multidimensional raster.
                                                 To get the slice index use slices method on the ImageryLayer object.
        ====================================     ====================================================================

        :return:
            None
        """
        if self.tiles_only:
            dataSourceExtent = self.extent
            tinfo = self.properties.tileInfo
            origin = tinfo["origin"]
            tw = tinfo["cols"]
            th = tinfo["rows"]
            if "lods" in tinfo.keys():
                if len(tinfo["lods"]) >= 1:
                    if level is None:
                        level = 0
                    resolution = {
                        "x": tinfo["lods"][level]["resolution"],
                        "y": tinfo["lods"][level]["resolution"],
                    }
                    # level = tinfo["lods"][resolution]["level"]
            import math

            colStart = math.floor(
                (dataSourceExtent["xmin"] - origin["x"]) / resolution["x"] / tw
            )
            colEnd = math.ceil(
                (dataSourceExtent["xmax"] - origin["x"] - resolution["x"])
                / resolution["x"]
                / tw
            )
            rowStart = math.floor(
                (origin["y"] - dataSourceExtent["ymax"]) / resolution["y"] / th
            )
            rowEnd = math.ceil(
                (origin["y"] - dataSourceExtent["ymin"] - resolution["y"])
                / resolution["y"]
                / th
            )
            from matplotlib import pyplot as plt
            from numpy import ma

            img = []
            numarray = None
            numpylist = []
            mask_array = None
            masklist = []
            for i in range(rowStart, rowEnd):
                for j in range(colStart, colEnd):
                    num, valid_mask = self._read_tilesonly_layer(
                        level, i, j, slice_id, as_numpy=True
                    )
                    if numarray is None:
                        numarray = num
                    else:
                        numarray = np.ma.concatenate((numarray, num), axis=1)
                    if mask_array is None:
                        mask_array = valid_mask
                    else:
                        mask_array = np.concatenate((mask_array, valid_mask), axis=1)
                numpylist.append(numarray)
                masklist.append(mask_array)
                numarray = None
                mask_array = None
                # img.append(((lyr.tiles.image_tile(2,i,j, as_numpy = True))))

            for index, ele in enumerate(numpylist):
                if index == 0:
                    numarray = ele
                else:
                    # imgnew = plt.imshow(ele)
                    numarray = np.ma.concatenate((numarray, ele), axis=0)

            for index, ele in enumerate(masklist):
                if index == 0:
                    mask_array = ele
                else:
                    # imgnew = plt.imshow(ele)
                    mask_array = np.concatenate((mask_array, ele), axis=0)
            num_bands = self.band_count
            try:
                if numarray.dtype != "uint8" or (
                    numarray.dtype == "float"
                    and (numarray.min() < 0 or 1 < numarray.max())
                ):
                    band_arr_list = []
                    render_bands = (
                        1
                        if (num_bands == 1 and numarray.ndim == 2)
                        else numarray.shape[2]
                    )
                    for i in range(render_bands):
                        if num_bands == 1 and numarray.ndim == 2:
                            band_arr = numarray
                        else:
                            band_arr = numarray[:, :, i]

                        # percent clip stretching
                        p005 = np.percentile(band_arr, 0.5)
                        p995 = np.percentile(band_arr, 99.5)
                        r = 255.0 / (p995 - p005 + 2)
                        out = np.round(r * (band_arr - p005 + 1)).astype("uint8")
                        out[band_arr < p005] = 0
                        out[band_arr > p995] = 255
                        band_arr_list.append(out)

                    if num_bands == 1 and numarray.ndim == 2:
                        stretched_img = band_arr_list[0]
                    else:
                        stretched_img = np.ma.dstack(band_arr_list)
                    numarray = stretched_img
            except:
                pass

            # numarray = numarray[np.ix_(mask_array.any(1), mask_array.any(0))]
            custom_cmap = None
            if num_bands == 1:
                colormap_list = []
                colormap_dict = self.colormap()
                if colormap_dict is not None:
                    if "colormap" in colormap_dict.keys():
                        colormap_list = colormap_dict["colormap"]

                    if colormap_list is not None and colormap_list != []:
                        import matplotlib.colors

                        cmap_np = np.array(colormap_list)
                        colors = cmap_np[:, 1:] / 255
                        custom_cmap = matplotlib.colors.ListedColormap(colors)

            if numarray.mask.ndim == 2:
                mask = numarray.mask == False
            else:
                mask = numarray.mask[:, :, 1] == False
            numarray = numarray[np.ix_(mask.any(1), mask.any(0))]

            if custom_cmap is not None:
                imgnew = plt.imshow(numarray, cmap=custom_cmap)
            else:
                if numarray.dtype == "uint8":
                    numarray[numarray.mask] = 255
                elif numarray.dtype == "float32":
                    numarray[numarray.mask] = np.nan
                if (
                    "hasMultidimensions" in self.properties
                    and self.properties["hasMultidimensions"]
                ):
                    imgnew = plt.imshow(numarray)
                else:
                    imgnew = plt.imshow(numarray, cmap="Greys_r")
            plt.axis("off")
            imgnew.axes.get_xaxis().set_visible(False)
            imgnew.axes.get_yaxis().set_visible(False)
            plt.close(imgnew.figure)
            return imgnew.figure

    def _read_tilesonly_layer(self, level, row, column, slice_id=None, as_numpy=False):
        import tempfile, uuid

        fname = "%s.jpg" % uuid.uuid4().hex
        out_folder = tempfile.gettempdir()
        params = {}

        if slice_id is not None:
            params["sliceId"] = slice_id
        url = "%s/tile/%s/%s/%s" % (self._url, level, row, column)
        if self.tiles_only:
            res = self._con.get(
                path=url, params=params, try_json=False, force_bytes=True
            )

            try:
                import lerc
            except:
                _LOGGER.warning(
                    "lerc needs to be installed, to render Tiled Imagery Layer (pip install pylerc)"
                )
            if not isinstance(res, bytes):
                raise RuntimeError(res)
            result, data, valid_mask = lerc.decode(res)
            if result != 0:
                raise RuntimeError("decoding bytes from imagery service failed.")
            data, valid_mask = np.broadcast_arrays(data, valid_mask)
            data.setflags(write=True)
            valid_mask = valid_mask == False
            from numpy import ma

            data = ma.masked_array(data, valid_mask)
            if data.shape[0] > 3 and len(data.shape) == 3:
                data = data[0:3]  # Extract first 3 bands
            if len(data) == 2:
                data = np.expand_dims(data, axis=2)
            elif len(data) == 3:
                data = np.transpose(data, axes=[1, 2, 0])

            # data = data[np.ix_(valid_mask.any(1), valid_mask.any(0))]
            return data, valid_mask

    def _get_service_info(self, rendering_rule=None):
        if self._original_info != {}:
            return self._original_info
        else:
            url = self._url

            params = {"f": "json"}
            if rendering_rule is not None:
                params["renderingRule"] = rendering_rule

            if self._datastore_raster:
                params["Raster"] = self._uri
                if isinstance(self._uri, bytes) and "renderingRule" in params.keys():
                    del params["renderingRule"]

            dictdata = {}
            token = None
            try:
                dictdata = self._con.post(self.url, params, timeout=None)
            except Exception as e:
                try:
                    if (
                        (hasattr(self, "_lazy_token")) and self._lazy_token is None
                    ) or not hasattr(self, "_lazy_token"):
                        from .functions.utility import _generate_layer_token

                        token = _generate_layer_token(self, self._url)
                        if token is not None:
                            self._lazy_token = token
                except Exception as e:
                    token = self._token
                try:
                    dictdata = self._con.post(
                        self.url, params, token=token, timeout=None
                    )
                except Exception as e:
                    if hasattr(e, "msg") and e.msg == "Method Not Allowed":
                        dictdata = self._con.get(self.url, params, token=token)
                    elif str(e).lower().find("token required") > -1:
                        dictdata = self._con.get(self.url, params)
                    elif str(e).lower().find("invalid token") > -1:
                        dictdata = self._con.post(
                            self.url, params, token=None, timeout=None
                        )
                        self._lazy_token = None  # got info as public service
                    else:
                        raise e
            self._original_info = dictdata
            return self._original_info

    def plot_histograms(
        self,
        geometry: Optional[Union[Geometry, Polygon, Envelope]] = None,
        pixel_size: Optional[Union[str, dict[str, Any]]] = None,
        time: Optional[Union[datetime.date, datetime.datetime, str, list[int]]] = None,
        bands: list[int] = [],
        display_stats: bool = True,
        plot_properties: Optional[dict[str, Any]] = None,
        subplot_properties: Optional[list[dict[str, Any], dict[str, Any]]] = None,
    ):
        """
        The ``plot_histograms`` method is used to plot the histograms for the :class:`~arcgis.raster.ImageryLayer`.
        
        Image histograms visually summarize the distribution of a continuous numeric variable by measuring
        the frequency at which certain values appear in the image. The x-axis in the image histogram is a 
        number line that displays the range of image pixel values that has been split into number ranges, 
        or bins. A bar is drawn for each bin, and the width of the bar represents the density number range 
        of the bin; the height of the bar represents the number of pixels that fall into that range. 
        Understanding the distribution of your data is an important step in the data exploration process.

        This method can be used for plotting the band-wise image histogram charts of any
        :class:`~arcgis.raster.ImageryLayer` published with mosaic datasets or a raster dataset.

        ============================    ====================================================================
        **Parameter**                   **Description**
        ----------------------------    --------------------------------------------------------------------
        geometry                        Optional :class:`~arcgis.geometry.Geometry` (:class:`~arcgis.geometry.Polygon`
                                        or :class:`~arcgis.geometry.Envelope`).
                                        A geometry that defines the geometry within which the histogram is computed.

                                        .. note::
                                            If not provided, then the full extent of the
                                            raster will be used for the computation.
        ----------------------------    --------------------------------------------------------------------
        pixel_size                      Optional string or dictionary. The pixel level being used (or the
                                        resolution being looked at). If pixel size is not specified, then
                                        pixel_size will default to the base resolution of the dataset.
                                        The structure of the pixel_size parameter is the same as the
                                        structure of the point object returned by the ArcGIS REST API.
                                        In addition to the dictionary structure, you can specify the pixel size
                                        with a comma-separated string.
                                        
                                        Syntax:
                                          - dictionary structure: pixel_size={point}
                                          - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                          - pixel_size={"x": 0.18, "y": 0.18}
                                          - pixel_size='0.18,0.18'
        ----------------------------    --------------------------------------------------------------------
        time                            Optional datetime.date, datetime.datetime or timestamp string. The
                                        time instant or the time extent to compute statistics and histograms.
                                        Time instant specified as datetime.date, datetime.datetime or
                                        timestamp in milliseconds since epoch
                                        Syntax: time=<timeInstant>
                                        
                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp
                                        
                                        Available in 10.8+
        ----------------------------    --------------------------------------------------------------------
        bands                           Optional list of band indices. By default takes the first band (band index - 0).
                                        Image histogram charts are plotted for these specific bands.

                                        Example:
                                            - [0,2,3]
        ----------------------------    --------------------------------------------------------------------
        display_stats                   Optional boolean. Specifies whether to plot the band-wise statistics 
                                        along with the histograms.

                                        Some basic descriptive statistics are calculated and displayed on 
                                        histograms. The mean and median are displayed with one line each, and 
                                        one standard deviation above and below the mean is displayed using two lines.

                                            - False - The statistics will not be displayed along with the histograms.
                                            - True - The statistics will be displayed along with the histograms. \
                                                     This is the default.
        ----------------------------    --------------------------------------------------------------------
        plot_properties                 Optional dictionary. This parameter can be used to set the figure 
                                        properties. These are the
                                        `matplotlib.pyplot.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_
                                        parameters and values specified in dictionary format.

                                        Example:
                                         {"figsize":(15,15)}
        ----------------------------    --------------------------------------------------------------------
        subplot_properties              Optional list or dictionary. This parameter can be used to set band-wise 
                                        histogram (subplot) display properties. These are the `matplotlib.axes.Axes.bar() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib-axes-axes-bar>`__
                                        parameters and values specified in dictionary format.

                                        .. note::
                                            `matplotlib.axes.Axes.bar() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib-axes-axes-bar>`_
                                            parameters: ''x', 'height' or 'align' cannot be passed into ``subplot_properties``.

                                        Example:

                                        subplot_properties = [
                                                 {"color":"r"},
                                                 {"color":"g"},
                                                 {"color":"b","edgecolor":"w"}
                                                            ]
        ============================    ====================================================================

        .. tip::

            When working with multidimensional imagery layers, you can use the
            `multidimensional_filter() <https://developers.arcgis.com/python/api-reference/arcgis.raster.functions.html#multidimensional-filter>`__
            raster function on the layer for slicing the data along defined variables and dimensions.
            ``plot_histograms`` can then be used on the output layer returned upon applying the filter.
        
        :return: None

        .. code-block:: python

            # Usage Example: Plots histograms of the raster with specified resolution and bands

            raster1.plot_histograms(pixel_size="0.18, 0.18",
                                    bands=[1, 2, 3],
                                    plot_properties={"figsize":(15,15)},
                                    subplot_properties=[
                                                        {"color":"r"},
                                                        {"color":"g"},
                                                        {"color":"b","edgecolor":"w"}
                                                       ],
                                    )
        
        """
        from arcgis.raster._charts import plot_histograms

        return plot_histograms(
            self,
            geometry=geometry,
            pixel_size=pixel_size,
            time=time,
            bands=bands,
            display_stats=display_stats,
            plot_properties=plot_properties,
            subplot_properties=subplot_properties,
        )

    def spectral_profile(
        self,
        points: list[Point] = [],
        show_values: bool = False,
        plot_properties: dict[str, Any] = {},
    ):
        """
        The ``spectral_profile`` method can be used to create spectral profile charts.

        Spectral profile charts allow you to select areas of interest or ground features on the image and review the spectral information of all bands in a chart format.

        The x-axis of the spectral profile displays the band names

        The y-axis of the spectral profile displays the spectral values.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        points                                   Required list of :class:`~arcgis.geometry.Point` objects.
        ------------------------------------     --------------------------------------------------------------------
        show_values                              Optional boolean. Default is False.
                                                 Set this parameter to True to display the values at each point in the line graph.
        ------------------------------------     --------------------------------------------------------------------
        plot_properties                          Optional dictionary. This parameter can be used to set the figure
                                                 properties. These are the `matplotlib.pyplot.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_
                                                 parameters and values specified in dictionary format.

                                                 eg: {"figsize":(15,15)}
        ====================================     ====================================================================

        :return:
            None

        """
        from arcgis.raster._charts import spectral_profile

        return spectral_profile(
            self,
            points=points,
            show_values=show_values,
            plot_properties=plot_properties,
        )

    def dimension_profile(
        self,
        points: list[Point],
        dimension: str,
        time: datetime.datetime,
        variables: list[str] = [],
        show_values: bool = False,
        show_trend_line: bool = False,
        plot_properties: dict[str, Any] = {},
    ):
        """
        Dimension profile chart visualizes change along a vertical dimension, such as depth or height,
        using a multidimensional raster dataset with a z-dimension.
        Dimension Profile is only available for multidimensional datasets that contain a z-dimension.

        Change is plotted in the form of a line graph for a given location and date or time. This allows
        trends in two variables to be displayed and compared simultaneously, while taking into account
        different unit scales.

        The x-axis of the dimension profile displays the values of the variable. Default minimum and
        maximum x-axis bounds are set based on the range of data values represented on the axis.

        The y-axis of the dimension profile displays the vertical dimension value.


        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        points                                   Required list of :class:`~arcgis.geometry.Point` objects.
        ------------------------------------     --------------------------------------------------------------------
        dimension                                Required string. The dimension name that will be used for plotting dimension profile. Use this parameter to set the field that
                                                 represents the dimension field in the image service.
        ------------------------------------     --------------------------------------------------------------------
        time                                     Required datetime object or timestamp in milliseconds. The time slice that will be used for plotting dimension profile.
        ------------------------------------     --------------------------------------------------------------------
        variables                                Required list of strings. The variables that will be used for plotting dimension profile.
                                                 The dimension profile chart allows a maximum of two variables to be displayed.
        ------------------------------------     --------------------------------------------------------------------
        show_values                              Optional boolean. Default value is False.
                                                 Set this parameter to True to display the values at each point in the line graph.
        ------------------------------------     --------------------------------------------------------------------
        show_trend_line                          Optional boolean. Default value is False.
                                                 Set this parameter to True to add a linear trend line to the dimension profile chart.
                                                 One trend line will be drawn for each location when charting multiple locations,
                                                 or each variable when charting multiple variables.
        ------------------------------------     --------------------------------------------------------------------
        plot_properties                          Optional dictionary. This parameter can be used to set the figure
                                                 properties. These are the `matplotlib.pyplot.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_
                                                 parameters and values specified in dictionary format.

                                                 eg: {"figsize":(15,15)}
        ====================================     ====================================================================

        :return:
            None

        """
        from arcgis.raster._charts import dimension_profile

        return dimension_profile(
            self,
            points=points,
            dimension=dimension,
            time=time,
            variables=variables,
            show_values=show_values,
            show_trend_line=show_trend_line,
            plot_properties=plot_properties,
        )

    def _repr_jpeg_(self):
        if self._uses_gbl_function:
            return self._repr_svg_()

        if self.tiles_only:
            fig = self.render_tilesonly_layer()
            try:
                from IPython.core.pylabtools import print_figure

                data = print_figure(fig, "jpeg")
                from matplotlib import pyplot as plt

                plt.close(fig)
                return data
            except:
                pass

        else:
            bbox_sr = None
            if "spatialReference" in self.extent:
                bbox_sr = self.extent["spatialReference"]
            if not self._uses_gbl_function:
                return self.export_image(
                    bbox=self._extent,
                    bbox_sr=bbox_sr,
                    size=[1200, 450],
                    export_format="jpeg",
                    f="image",
                )

    def _repr_svg_(self):
        if self._uses_gbl_function:
            graph = self.draw_graph()
            svg_graph = graph.pipe().decode("utf-8")
            return svg_graph
        else:
            return None

    def _set_mosaic_rule(self):
        mosaic_method_mapping = {
            "none": "esriMosaicNone",
            "center": "esriMosaicCenter",
            "northwest": "esriMosaicNorthwest",
            "nadir": "esriMosaicNadir",
            "viewpoint": "esriMosaicViewpoint",
            "byattribute": "esriMosaicAttribute",
            "lockraster": "esriMosaicLockRaster",
            "seamline": "esriMosaicSeamline",
        }

        mosaic_rule = {}
        try:
            if type(self) == ImageryLayer:
                if (
                    "capabilities" in self.properties
                    and str(self.properties["capabilities"]).lower().find("catalog")
                    == -1
                ):
                    return None
                if (
                    "defaultMosaicMethod" in self.properties.keys()
                ) and self.properties["defaultMosaicMethod"] != None:
                    if (
                        self.properties["defaultMosaicMethod"].lower()
                        in mosaic_method_mapping.keys()
                    ):
                        mosaic_rule.update(
                            {
                                "mosaicMethod": mosaic_method_mapping[
                                    self.properties["defaultMosaicMethod"].lower()
                                ]
                            }
                        )
                if ("sortField" in self.properties.keys()) and self.properties[
                    "sortField"
                ] != None:
                    mosaic_rule.update({"sortField": self.properties["sortField"]})
                if ("sortValue" in self.properties.keys()) and self.properties[
                    "sortValue"
                ] != None:
                    mosaic_rule.update({"sortValue": self.properties["sortValue"]})
                if ("mosaicOperator" in self.properties.keys()) and self.properties[
                    "mosaicOperator"
                ] != None:
                    mosaic_rule.update(
                        {
                            "mosaicOperation": "MT_"
                            + self.properties["mosaicOperator"].upper()
                        }
                    )
                if ("sortAscending" in self.properties.keys()) and self.properties[
                    "sortAscending"
                ] != None:
                    mosaic_rule.update({"ascending": self.properties["sortAscending"]})
        except:
            pass

        return mosaic_rule

    def __sub__(self, other):
        from arcgis.raster.functions import minus

        return minus([self, other])

    def __rsub__(self, other):
        from arcgis.raster.functions import minus

        return minus([other, self])

    def __add__(self, other):
        from arcgis.raster.functions import plus

        return plus([self, other])

    def __radd__(self, other):
        from arcgis.raster.functions import plus

        return plus([other, self])

    def __mul__(self, other):
        from arcgis.raster.functions import times

        return times([self, other])

    def __rmul__(self, other):
        from arcgis.raster.functions import times

        return times([other, self])

    def __div__(self, other):
        from arcgis.raster.functions import divide

        return divide([self, other])

    def __rdiv__(self, other):
        from arcgis.raster.functions import divide

        return divide([other, self])

    def __pow__(self, other):
        from arcgis.raster.functions import power

        return power([self, other])

    def __rpow__(self, other):
        from arcgis.raster.functions import power

        return power([other, self])

    def __abs__(self):
        from arcgis.raster.functions import abs

        return abs([self])

    def __lshift__(self, other):
        from arcgis.raster.functions import bitwise_left_shift

        return bitwise_left_shift([self, other])

    def __rlshift__(self, other):
        from arcgis.raster.functions import bitwise_left_shift

        return bitwise_left_shift([other, self])

    def __rshift__(self, other):
        from arcgis.raster.functions import bitwise_right_shift

        return bitwise_right_shift([self, other])

    def __rrshift__(self, other):
        from arcgis.raster.functions import bitwise_right_shift

        return bitwise_right_shift([other, self])

    def __floordiv__(self, other):
        from arcgis.raster.functions import floor_divide

        return floor_divide([self, other])

    def __rfloordiv__(self, other):
        from arcgis.raster.functions import floor_divide

        return floor_divide([other, self])

    def __truediv__(self, other):
        from arcgis.raster.functions import float_divide

        return float_divide([self, other])

    def __rtruediv__(self, other):
        from arcgis.raster.functions import float_divide

        return float_divide([other, self])

    def __mod__(self, other):
        from arcgis.raster.functions import mod

        return mod([self, other])

    def __rmod__(self, other):
        from arcgis.raster.functions import mod

        return mod([other, self])

    def __neg__(self):
        from arcgis.raster.functions import negate

        return negate([self])

    def __invert__(self):
        from arcgis.raster.functions import boolean_not

        return boolean_not([self])

    def __and__(self, other):
        from arcgis.raster.functions import boolean_and

        return boolean_and([self, other])

    def __rand__(self, other):
        from arcgis.raster.functions import boolean_and

        return boolean_and([other, self])

    def __xor__(self, other):
        from arcgis.raster.functions import boolean_xor

        return boolean_xor([self, other])

    def __rxor__(self, other):
        from arcgis.raster.functions import boolean_xor

        return boolean_xor([other, self])

    def __or__(self, other):
        from arcgis.raster.functions import boolean_or

        return boolean_or([self, other])

    def __ror__(self, other):
        from arcgis.raster.functions import boolean_or

        return boolean_or([other, self])

    def __ne__(self, other):
        if isinstance(other, (ImageryLayer, Raster, numbers.Number)):
            from arcgis.raster.functions import not_equal

            return not_equal([self, other])
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (ImageryLayer, Raster, numbers.Number)):
            from arcgis.raster.functions import equal_to

            return equal_to([self, other])
        else:
            return NotImplemented

    def __gt__(self, other):
        from arcgis.raster.functions import greater_than

        return greater_than([self, other])

    def __ge__(self, other):
        from arcgis.raster.functions import greater_than_equal

        return greater_than_equal([self, other])

    def __lt__(self, other):
        from arcgis.raster.functions import less_than

        return less_than([self, other])

    def __le__(self, other):
        from arcgis.raster.functions import less_than_equal

        return less_than_equal([self, other])

    def __deepcopy__(self, memo=None):
        newlyr = self._clone_layer()
        return newlyr

        # Raster.Raster.__pos__ = unaryPos         # +v


# Raster.Raster.__abs__ = Functions.Abs    # abs(v)
#
# Raster.Raster.__add__  = Functions.Plus  # +
# Raster.Raster.__radd__ = lambda self, lhs: Functions.Plus(lhs, self)
# Raster.Raster.__sub__  = Functions.Minus # -
# # TODO Huh?
# Raster.Raster.__rsub__ = Functions.Minus
# # Raster.Raster.__rsub__ = lambda self, lhs: Functions.Minus(lhs, self)
# Raster.Raster.__mul__  = Functions.Times # *
# Raster.Raster.__rmul__ = lambda self, lhs: Functions.Times(lhs, self)
# Raster.Raster.__pow__  = Functions.Power # **
# Raster.Raster.__rpow__ = lambda self, lhs: Functions.Power(lhs, self)
#
# Raster.Raster.__lshift__  = Functions.BitwiseLeftShift  # <<
# Raster.Raster.__rlshift__ = lambda self, lhs: Functions.BitwiseLeftShift(lhs, self)
# Raster.Raster.__rshift__  = Functions.BitwiseRightShift # >>
# Raster.Raster.__rrshift__ = lambda self, lhs: Functions.BitwiseRightShift(lhs, self)
#
# Raster.Raster.__div__       = Functions.Divide     # /
# Raster.Raster.__rdiv__      = lambda self, lhs: Functions.Divide(lhs, self)

# Raster.Raster.__floordiv__  = Functions.FloorDivide # //
# Raster.Raster.__rfloordiv__ = lambda self, lhs: Functions.FloorDivide(lhs, self)

# Raster.Raster.__truediv__   = Functions.FloatDivide # /
# Raster.Raster.__rtruediv__  = lambda self, lhs: Functions.FloatDivide(lhs, self)


# Raster.Raster.__mod__       = Functions.Mod        # %
# Raster.Raster.__rmod__      = lambda self, lhs: Functions.Mod(lhs, self)
# Raster.Raster.__divmod__    = returnNotImplemented # divmod()
# Raster.Raster.__rdivmod__   = returnNotImplemented
#
# # The Python bitwise operators are used for Raster boolean operators.
# Raster.Raster.__invert__ = Functions.BooleanNot # ~
# Raster.Raster.__and__    = Functions.BooleanAnd # &
# Raster.Raster.__rand__   = lambda self, lhs: Functions.BooleanAnd(lhs, self)
# Raster.Raster.__xor__    = Functions.BooleanXOr # ^
# Raster.Raster.__rxor__   = lambda self, lhs: Functions.BooleanXOr(lhs, self)
# Raster.Raster.__or__     = Functions.BooleanOr  # |
# Raster.Raster.__ror__    = lambda self, lhs: Functions.BooleanOr(lhs, self)
#
# # Python will use the non-augmented versions of these.
# Raster.Raster.__iadd__      = returnNotImplemented # +=
# Raster.Raster.__isub__      = returnNotImplemented # -=
# Raster.Raster.__imul__      = returnNotImplemented # *=
# Raster.Raster.__idiv__      = returnNotImplemented # /=
# Raster.Raster.__itruediv__  = returnNotImplemented # /=
# Raster.Raster.__ifloordiv__ = returnNotImplemented # //=
# Raster.Raster.__imod__      = returnNotImplemented # %=
# Raster.Raster.__ipow__      = returnNotImplemented # **=
# Raster.Raster.__ilshift__   = returnNotImplemented # <<=
# Raster.Raster.__irshift__   = returnNotImplemented # >>=
# Raster.Raster.__iand__      = returnNotImplemented # &=
# Raster.Raster.__ixor__      = returnNotImplemented # ^=
# Raster.Raster.__ior__       = returnNotImplemented # |=

from .. import raster

from arcgis.raster._util import (
    _to_datetime,
    _datetime2ole,
    _ole2datetime,
    _iso_to_datetime,
    _check_if_iso_format,
    _epoch_to_iso,
)


# def _get_class(path, is_multidimensional=False,  format="server", gis=None):
#    if format == 'server':
#        return _ImageServerRaster(path, is_multidimensional, format, gis)
#    elif format == 'arcpy':
#        return _ArcpyRaster(path, is_multidimensional, format, gis)
#    else:
#        raise ValueError(format)


def _get_engine(engine):
    """
    Function to get the engine that will be used to process the Raster object.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    engine                                   Required string.
                                                Possible options:
                                                "arcpy" : Returns arcpy engine
                                                "image_server" : Returns image server engine
    ------------------------------------     --------------------------------------------------------------------
    """

    engine_dict = {"arcpy": _ArcpyRaster, "image_server": _ImageServerRaster}
    if isinstance(engine, str):
        return engine_dict[engine]
    return engine


class Raster:
    """
    A ``Raster`` object is a variable that references a raster.
    It can be used to query the properties of the raster dataset.

    Usage: ``arcgis.raster.Raster(path, is_multidimensional=False,  engine=None, gis=None)``

    The ``Raster`` class can work with arcpy engine or image server engine. By default,
    if the path is a local path, then the Raster class uses the arcpy engine
    else it will use ``image_server`` engine.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    path                                     Required string. The input raster.

                                             Example:
                                                path = r"/path/to/raster"

                                                path = "https://myserver/arcgis/rest/services/ImageServiceName/ImageServer"

                                                path = "/fileShares/file_share_name/path/to/raster"

                                                path = "/cloudStores/cloud_store_name/path/to/raster"

                                                path = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/M/BP/2021/6/S2A_43MBP_20210622_0_L2A/B08.tif"

                                             .. note::
                                             When working with datastore rasters or non image service urls, RasterRendering service
                                             should be enabled in the active GIS connection
    ------------------------------------     --------------------------------------------------------------------
    is_multidimensional                      Optional boolean. Determines whether the input raster will be
                                             treated as multidimensional.

                                             Specify True if the input is multidimensional and should be
                                             processed as multidimensional, where processing occurs for every
                                             slice in the dataset. Specify False if the input is not
                                             multidimensional, or if it is multidimensional and should not be
                                             processed as multidimensional.

                                             Default is False
    ------------------------------------     --------------------------------------------------------------------
    extent                                   Optional dict. If the input raster's extent cannot be automatically
                                             inferred, pass in a dictionary representing the raster's extent
                                             for when viewing on a :class:`~arcgis.widgets.MapView` widget.

                                             Example:
                                                | { "xmin" : -74.22655,
                                                |   "ymin" : 40.712216,
                                                |   "xmax" : -74.12544,
                                                |   "ymax" : 40.773941,
                                                |   "spatialReference" :
                                                |       { "wkid" : 4326 }
                                                | }
    ------------------------------------     --------------------------------------------------------------------
    cmap                                     Optional str. When displaying a 1 band raster in a
                                             :class:`~arcgis.widgets.MapView` widget, what matplotlib colormap
                                             to apply to the raster. See :meth:`arcgis.mapping.symbol.display_colormaps`
                                             for a list of compatible values.
    ------------------------------------     --------------------------------------------------------------------
    opacity                                  Optional number. When displaying a raster in a
                                             :class:`~arcgis.widgets.MapView` widget, what opacity to apply. 0
                                             is completely transparent, 1 is completely opaque.
                                             Default: 1
    ------------------------------------     --------------------------------------------------------------------
    engine                                   Optional string. The backend engine to be used.
                                             Possible options:
                                                - "arcpy" : Use the arcpy engine for processing.

                                                - "image_server" : Use the Image Server engine for processing.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. :class:`~arcgis.gis.GIS` of the Raster object.
    ====================================     ====================================================================

    .. code-block:: python

        # Example Usage

        map = gis.map()

        # Overlay an image service on the 'MapView' widget
        service_url = gis.content.search("my_image_service", item_type="Imagery Layer")[0].url
        raster = Raster(path=service_url, gis=gis)
        map.add_layer(raster)

        # Overlay .tif file present in user's registered fileShare datastore
        # (Requires RasterRendering service to be enabled in the active GIS)
        raster = Raster("/fileShares/data/Amberg.tif", gis=gis)
        map.add_layer(raster)

        # Overlay a publicly accesible Cloud-Optimized GeoTIFF
        # (Requires RasterRendering service to be enabled in the active GIS)
        raster = Raster("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/M/BP/2021/6/S2A_43MBP_20210622_0_L2A/B08.tif",
                        gis=gis)
        map.add_layer(raster)

        # Overlay a local .tif file
        raster = Raster(r"./data/Amberg.tif")
        map.add_layer(raster)

        # Overlay a 1-channel .gdb file with the "Orange Red" colormap at 85% opacity
        raster = Raster("./data/madison_wi.gdb/Impervious_Surfaces",
                        cmap = "OrRd",
                        opacity = 0.85)
        map.add_layer(raster)

        # Overlay a local .jpg file by manually specifying its extent
        raster = Raster("./data/newark_nj_1922.jpg",
                        extent = {"xmin":-74.22655,
                                  "ymin":40.712216,
                                  "xmax":-74.12544,
                                  "ymax":40.773941,
                                  "spatialReference":{"wkid":4326}})
        map.add_layer(raster)

    """

    def __init__(
        self,
        path: str,
        is_multidimensional: bool = False,
        extent: Optional[dict] = None,
        cmap: Optional[str] = None,
        opacity: Optional[float] = None,
        engine: Optional[str] = None,
        gis: Optional[GIS] = None,
    ):
        self._engine_obj = None
        if not isinstance(is_multidimensional, bool):
            raise TypeError("is_multidimensional must be boolean type")
        if engine is not None:
            if engine not in [
                "arcpy",
                "image_server",
                _ArcpyRaster,
                _ImageServerRaster,
            ]:
                raise RuntimeError(
                    'Invalid engine: parameter engine should be either "arcpy" or "image_server"'
                )
            engine = _get_engine(engine)
            self._engine = engine
        else:
            if isinstance(path, RasterInfo):
                engine = _ArcpyRaster
            if isinstance(path, str):
                if (
                    "https://" not in path
                    and "http://" not in path
                    and "/fileShares/" not in path
                    and "/rasterStores/" not in path
                    and "/cloudStores/" not in path
                    and not isinstance(path, dict)
                    and "/vsi" not in path
                ):  # local raster case
                    if engine is None:
                        engine = _ArcpyRaster
            elif not isinstance(path, Raster):
                try:
                    import arcpy

                    if isinstance(path, arcpy.Raster):
                        engine = _ArcpyRaster
                except:
                    pass
            if engine is None:
                engine = _ImageServerRaster
            self._engine = engine
        if self._engine_obj is None and self._engine is not None:
            self._engine_obj = self._engine(path, is_multidimensional, gis)

        if extent:
            self.extent = extent
        if cmap:
            self.cmap = cmap
        if opacity:
            self.opacity = opacity

    # def __iter__(self):
    #    return(self._engine_obj.__iter__())

    def __getitem__(self, item):
        return self._engine_obj.__getitem__(item)

    def __setitem__(self, idx, value):
        return self._engine_obj.__setitem__(idx, value)

    def refresh_service(self, options: str = None, future: bool = True) -> str:
        """
        Refresh Service is a task in the existing out-of-the-box
        Publishing Tools geoprocessing service used by the service publisher
        to refresh a GIS service to reflect back-end data changes.
        """
        try:
            gis = self._engine_obj._gis
            return gis._tools.system_service.refresh_service(
                self, options=options, future=future
            )

        except:
            raise ValueError(
                "Unsupported Raster type. refresh_service method is only available on Raster objects created from an image service."
            )

    def set_engine(self, engine):
        """
        The ``set_engine`` method can be used to change the back end engine of the :class:`~arcgis.raster.Raster`
        object.
        """
        return Raster(
            path=self._engine_obj._path,
            is_multidimensional=self._engine_obj._is_multidimensional,
            engine=engine,
            gis=self._engine_obj._gis,
        )

    @property
    def extent(self):
        """
        Get/Set the area of interest. Used for displaying the :class:`~arcgis.raster.Raster` when queried.

        :return: The extent of the Raster as a dictionary

        """
        return self._engine_obj.extent

    @extent.setter
    def extent(self, value: dict):
        self._engine_obj.extent = value

    _cmap = None

    @property
    def cmap(self):
        """
        Get/Set what matplotlib colormap to apply to the raster (when displaying a 1 band raster
        in a :class:`~arcgis.widgets.MapView` widget).

        .. note::
            The ``cmap`` value must be a string. See :attr:`arcgis.mapping.symbol.display_colormaps`
            for a list of compatible values.
        """
        return self._cmap

    @cmap.setter
    def cmap(self, value: str):
        if isinstance(value, str):
            self._cmap = value
        else:
            raise Exception("`cmap` must be of type `str`")

    _vmin = None

    @property
    def vmin(self):
        """
        When displaying a 1 band raster with the ``cmap`` argument specified
        on a MapView, ``vmin`` and ``vmax`` define the data range that the colormap covers.
        The ``vmin`` property is the lower end of that range.
        """
        if self._vmin is None:
            self._vmin = self._attempt_infer_vmin()
        return self._vmin

    @vmin.setter
    def vmin(self, value: int):
        self._vmin = value

    def _attempt_infer_vmin(self):
        # only tested against _ArcpyRaster engines..
        try:
            return self._engine_obj._raster.minimum
        except Exception:
            return None

    _vmax = None

    @property
    def vmax(self):
        """
        When displaying a 1 band raster with the ``cmap`` argument specified
        on a MapView, ``vmin`` and ``vmax`` define the data range that the colormap covers.
        The ``vmax`` property is the upper end of that range.
        """
        if self._vmax is None:
            self._vmax = self._attempt_infer_vmax()
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        self._vmax = value

    def _attempt_infer_vmax(self):
        # only tested against _ArcpyRaster engines..
        try:
            return self._engine_obj._raster.maximum
        except Exception:
            return None

    _opacity = 1

    @property
    def opacity(self):
        """
        Get/Set what opacity to apply when displaying the raster in a
        :class:`~arcgis.widgets.MapView` widget.

        .. note::
            0 is completely transparent, 1 is completely opaque. The default value of ``opacity`` is 1.

        """
        return self._opacity

    @opacity.setter
    def opacity(self, value: float):
        self._opacity = value

    @property
    def pixel_type(self):
        """
        The ``pixel_type`` property returns the pixel type of the :class:`~arcgis.raster.Raster` object.

        :return:
            The pixel type
        """
        return self._engine_obj.pixel_type

    @property
    def width(self):
        """
        The ``width`` property returns width of the raster in the units of its spatial reference.

        :return:
            A float representing the width
        """
        return self._engine_obj.width

    @property
    def height(self):
        """
        The ``height`` property returns height of the raster in the units of its spatial reference.

        :return:
            A float representing the height
        """
        return self._engine_obj.height

    @property
    def columns(self):
        """
        The ``columns`` property returns the number of columns in the raster.

        :return:
            An integer
        """
        return self._engine_obj.columns

    @property
    def rows(self):
        """
        The ``rows`` property returns the number of rows in the raster.

        :return:
            An integer
        """
        return self._engine_obj.rows

    @property
    def band_count(self):
        """
        The ``band_count`` property returns the band count of the raster

        :return:
            An integer
        """
        return self._engine_obj.band_count

    @property
    def catalog_path(self):
        """
        The ``catalog_path`` property represents the full path and the name of the referenced raster.

        :return:
            A String
        """
        return self._engine_obj.catalog_path

    @property
    def catalog_paths(self):
        """
        The ``catalog_paths`` property represents the full paths and the names of each item comprising a mosaic dataset.

        :return:
            A list of paths of each item comprising a mosaic dataset.
        """
        return self._engine_obj.catalog_paths

    @property
    def path(self):
        """
        The ``path`` property represents the full path and name of the referenced raster.

        :return:
            A string
        """
        return self._engine_obj.path

    @property
    def name(self):
        """
        The ``name`` property returns the name of the raster.

        :return:
            A String
        """
        return self._engine_obj.name

    @property
    def names(self):
        """
        The ``names`` property returns the names of each item comprising a mosaic dataset.

        :return:
            A list of names of each item comprising a mosaic dataset.
        """
        return self._engine_obj.names

    @property
    def has_RAT(self):
        """
        The ``has_RAT`` property identifies if there is an associated attribute table:
        ``True`` if an attribute table exists, or ``False`` if no attribute table exists.

        :return:
            A boolean
        """
        return self._engine_obj.has_RAT

    @property
    def is_multidimensional(self):
        """
        The ``is_multidimensional`` property returns ``True`` if the raster is multidimensional.

        :return:
            A boolean
        """
        return self._engine_obj.is_multidimensional

    @property
    def is_temporary(self):
        """
        The ``is_temporary`` property returns ``True`` if the raster is temporary, or ``False`` if it is permanent.

        :return:
            A boolean
        """
        return self._engine_obj.is_temporary

    @property
    def mean_cell_width(self):
        """
        The ``mean_cell_width`` property returns the cell size in the x direction.

        :return:
            A float
        """
        return self._engine_obj.mean_cell_width

    @property
    def mean_cell_height(self):
        """
        The ``mean_cell_height`` property returns the cell size in the y direction.

        :return:
            A float
        """
        return self._engine_obj.mean_cell_height

    @property
    def multidimensional_info(self):
        """
        The ``multidimensional_info`` property returns the multidimensional information of the raster dataset,
        including variable names, descriptions and units, and dimension names, units, intervals, units, and ranges.
        """
        return self._engine_obj.multidimensional_info

    @property
    def minimum(self):
        """
        The ``minimum`` property returns minimum value in the referenced raster.

        :return:
            A float
        """
        return self._engine_obj.minimum

    @property
    def maximum(self):
        """
        The ``maximum`` property returns the maximum value in the referenced raster.

        :return:
            A float
        """
        return self._engine_obj.maximum

    @property
    def mean(self):
        """
        The ``mean`` property returns the mean value in the referenced raster.

        :return:
            A float
        """
        return self._engine_obj.mean

    @property
    def standard_deviation(self):
        """
        The ``standard_deviation`` property returns the standard deviation of the values in the referenced raster.

        :return:
            A float
        """
        return self._engine_obj.standard_deviation

    @property
    def spatial_reference(self):
        """
        The ``spatial_reference`` property returns the spatial reference of the referenced raster.

        :return:
            A spatial reference
        """
        return self._engine_obj.spatial_reference

    @property
    def variable_names(self):
        """
        The ``variable_names`` property returns the variable names in the multidimensional raster.

        :return:
            A List of variable names
        """
        return self._engine_obj.variable_names

    @property
    def variables(self):
        """
        The ``variables`` property returns the variable names and their dimensions in the multidimensional raster
        dataset.

        For example, a multidimensional raster containing temperature data over 24 months would
        return the following: ['temp(StdTime=24)']

        :return:
            A list of names and dimensions.
        """
        return self._engine_obj.variables

    @property
    def slices(self):
        """
        The ``slices`` property returns the attribute information of each slice,
        including its variable name, dimension names, and
        dimension values returned as a list of dictionaries.

        :return:
            A list of dictionaries
        """
        return self._engine_obj.slices

    @property
    def band_names(self):
        """
        The ``band_names`` property returns the band names of the raster

        :return:
            The band names as a List of Strings
        """
        return self._engine_obj.band_names

    @property
    def block_size(self):
        """
        The ``block_size`` property returns the block size of the raster

        :return:
            A tuple
        """
        return self._engine_obj.block_size

    @property
    def compression_type(self):
        """
        The ``compression_type`` property returns the compression type of the raster.

        :return:
            A string
        """
        return self._engine_obj.compression_type

    @property
    def format(self):
        """
        The ``format`` property returns the raster format.

        :return:
            A string
        """
        return self._engine_obj.format

    @property
    def no_data_value(self):
        """
        The ``no_data_value`` property returns the ``NoData`` value of the raster.

        :return:
            The ``NoData`` value
        """
        return self._engine_obj.no_data_value

    @property
    def no_data_values(self):
        """
        The ``no_data_values`` property returns the ``NoData`` value for each band in the multiband raster.

        :return:
            A list of ``NoData`` values
        """
        return self._engine_obj.no_data_values

    @property
    def uncompressed_size(self):
        """
        The ``uncompressed_size`` property returns the size of the referenced raster dataset on disk.

        :return:
            A float
        """
        return self._engine_obj.uncompressed_size

    @property
    def is_integer(self):
        """
        The ``is_integer`` property returns ``True`` if the raster has integer type.

        :return:
            A boolean
        """
        return self._engine_obj.is_integer

    @property
    def properties(self):
        """
        The ``properties`` property returns the property name and value pairs in the referenced raster.

        :return:
            A dictionary
        """
        return self._engine_obj.key_properties

    @property
    def read_only(self):
        """
        The ``read_only`` property returns whether the raster cell values are writable or not using the [row, column]
        notation.
        When this property is ``True``, they are not writable. Otherwise, they are writable.

        :return:
            A boolean
        """
        return self._engine_obj.read_only

    @property
    def RAT(self):
        """
        The ``RAT`` property returns the attribute table as a dictionary, if the table exists.

        :return:
            A dictionary
        """
        return self._engine_obj.RAT

    @property
    def raster_info(self):
        """
        The ``raster_info`` property returns information about the :class:`~arcgis.raster.Raster` such as
        ``bandCount``, ``extent``, ``pixelSizeX``, ``pixelSizeY``, ``pixelType``.

        :return:
            The raster info
        """
        return self._engine_obj.raster_info

    @staticmethod
    def from_stac_item(
        stac_item: Union[str, Item],
        request_params: Optional[dict[str, Any]] = None,
        engine: Optional[dict[str, Any]] = None,
        *,
        gis: Optional[GIS] = None,
    ):
        """
        The ``from_stac_item`` method creates a Raster object from a
        `SpatioTemporal Asset Catalog (STAC) Item <https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md>`__.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        stac_item             Required string or `pystac.Item <https://pystac.readthedocs.io/en/latest/api.html#item>`__ object. If string, then it should be
                              the URL of the STAC item. It can be a Static STAC item URL or a STAC
                              API Item URL.

                              .. note::

                                STAC items from the following STAC APIs are supported:

                                    - https://planetarycomputer.microsoft.com/api/stac/v1 (Following collections are supported: daymet-annual-pr, daymet-daily-hi, \
                                        3dep-seamless, 3dep-lidar-dsm, sentinel-1-rtc, gridmet, daymet-annual-na, daymet-monthly-na, daymet-annual-hi, \
                                        daymet-monthly-hi, daymet-monthly-pr, hgb, cop-dem-glo-30, cop-dem-glo-90, terraclimate, gnatsgo-rasters, 3dep-lidar-hag, \
                                        3dep-lidar-intensity, 3dep-lidar-pointsourceid, mtbs, noaa-c-cap, alos-fnf-mosaic, 3dep-lidar-returns, mobi, landsat-c2-l2, \
                                        chloris-biomass, daymet-daily-pr, 3dep-lidar-dtm-native, 3dep-lidar-classification, 3dep-lidar-dtm, gap, alos-dem, jrc-gsw, \
                                        hrea, sentinel-2-l2a, daymet-daily-na, nrcan-landcover, ecmwf-forecast, noaa-mrms-qpe-24h-pass2, sentinel-1-grd, nasadem, \
                                        io-lulc, landsat-c2-l1, drcog-lulc, chesapeake-lc-7, chesapeake-lc-13, chesapeake-lu, noaa-mrms-qpe-1h-pass1, \
                                        noaa-mrms-qpe-1h-pass2, noaa-nclimgrid-monthly, usda-cdl, esa-cci-lc, esa-cci-lc-netcdf, noaa-climate-normals-netcdf, \
                                        noaa-climate-normals-gridded, io-lulc-9-class, io-biodiversity, naip, noaa-cdr-sea-surface-temperature-whoi, \
                                        noaa-cdr-ocean-heat-content, noaa-cdr-sea-surface-temperature-whoi-netcdf, sentinel-3-olci-wfr-l2-netcdf, \
                                        noaa-cdr-ocean-heat-content-netcdf, sentinel-3-synergy-v10-l2-netcdf, sentinel-3-olci-lfr-l2-netcdf, \
                                        sentinel-3-slstr-lst-l2-netcdf, sentinel-3-slstr-wst-l2-netcdf, sentinel-3-synergy-syn-l2-netcdf, \
                                        sentinel-3-synergy-vgp-l2-netcdf, sentinel-3-synergy-vg1-l2-netcdf, esa-worldcover)
                                    - https://earth-search.aws.element84.com/v0 (All collections are suported)
                                    - https://earth-search.aws.element84.com/v1 (All collections are suported)
                                    - https://services.sentinel-hub.com/api/v1/catalog (All collections are suported)

                              Example:
                                    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/tx_m_2609719_se_14_060_20201217"
        -----------------     --------------------------------------------------------------------
        request_params        Optional dictionary. This parameter can be used to set the properties
                              for making the STAC Item request. These are the `requests.get() method <https://requests.readthedocs.io/en/master/api/#requests.get>`__
                              parameters and values will be specified in dictionary format.

                              | This parameter is honoured when the stac_item parameter is set to a string (URL).

                              Example:
                                    {"verify":False}
        -----------------     --------------------------------------------------------------------
        engine                Optional string. The backend engine to be used for Raster processing.

                              Possible options:
                                - "arcpy" : Use the arcpy engine for processing. 

                                - "image_server" : Use the Image Server engine for processing (This is the default).
                              
                              Example:
                                    "image_server"

                              .. note::
                                When using ``image_server`` engine, RasterRendering service should be enabled \
                                in the active GIS connection.
        -----------------     --------------------------------------------------------------------
        gis                   Optional :class:`~arcgis.gis.GIS` object. The GIS of the Raster object.
        =================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object

        .. code-block:: python

            # Usage Example: Creating a Raster object from a STAC Item.

            ras = Raster.from_stac_item(stac_item=stac_item_url,
                                        gis=gis)

        """
        if isinstance(stac_item, str):
            if request_params is None:
                request_params = {}
            if not isinstance(request_params, dict):
                raise RuntimeError("request_params should be of type dictionary")
            if any(key in request_params for key in ["url", "params", "data", "json"]):
                raise RuntimeError(
                    "request_params cannot contain these keys : url, params, data or json"
                )

            data = _requests.get(stac_item, **request_params)
            if data.status_code != 200 or data.headers.get("content-type") not in [
                "application/json",
                "application/geo+json",
                "application/json;charset=utf-8",
                "application/geo+json; charset=utf-8",
            ]:
                raise RuntimeError(
                    f"Invalid Response: Please verify that the stac_item URL is correct-\n{data.text}"
                )

            json_data = data.json()
        else:
            try:
                import pystac

                json_data = stac_item.to_dict()
            except ImportError:
                raise ImportError(
                    "pystac not found, parameter stac_item accepts either a STAC Item URL or a pystac.Item object"
                )
            except Exception:
                raise RuntimeError(f"Invalid/Unsupported STAC Item-\n{stac_item}")

        zarr_datasets = [
            "daymet-annual-pr",
            "daymet-daily-hi",
            "gridmet",
            "daymet-annual-na",
            "daymet-monthly-na",
            "daymet-annual-hi",
            "daymet-monthly-hi",
            "daymet-monthly-pr",
            "terraclimate",
            "daymet-daily-pr",
            "daymet-daily-na",
        ]

        if "type" not in json_data or (
            json_data["type"] != "Feature"
            and (
                json_data["type"] == "Collection"
                and json_data["id"] not in zarr_datasets
            )
        ):
            raise RuntimeError(f"Invalid STAC Item-\n{json_data}")

        item = json_data

        from ._util import _get_stac_metadata_file
        from arcgis.raster.functions import composite_band

        metadata_file = _get_stac_metadata_file(item)
        if not metadata_file:
            raise RuntimeError("STAC Item not supported")

        ras = (
            composite_band(
                rasters=[Raster(file, engine=engine, gis=gis) for file in metadata_file]
            )
            if isinstance(metadata_file, list)
            else Raster(metadata_file, engine=engine, gis=gis)
        )
        return ras

    def get_raster_bands(self, band_ids_or_names: Optional[list[str]] = None):
        """
        The ``get_raster_bands`` method returns a :class:`~arcgis.raster.Raster` object for each band specified in a
        multiband raster.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        band_ids_or_names     Required list. The index number or names of the bands to return as
                              Raster objects. If not specified, all bands will be extracted.
        =================     ====================================================================

        :return:
            A list of :class:`~arcgis.raster.Raster` objects (or a single :class:`~arcgis.raster.Raster` object if only one band is specified)

        .. code-block:: python

            # Usage Example: Generates the raster pertaining to the first band

            raster1 = Raster(r"./data/Amberg.tif")
            raster1.get_raster_bands(band_ids_or_names=[0])

        """
        return self._engine_obj.get_raster_bands(band_ids_or_names)

    def get_variable_attributes(self, variable_name: str):
        """
        The ``get_variable_attributes`` method returns the attribute information of a variable, such as
        description, unit, etc.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        variable_name         Required string. the name of the variable
        =================     ====================================================================

        :return:
            A dictionary. The attribute information of the given variable.

        .. code-block:: python

            # Usage Example: Returns variable attribute information

            raster1.get_variable_attributes(variable_name="variable_name")

        """
        return self._engine_obj.get_variable_attributes(variable_name)

    def get_dimension_names(self, variable_name: str):
        """
        The ``get_dimension_names`` method retrieves a list of the dimension names that the variable contains.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        variable_name         Required string. the name of the variable
        =================     ====================================================================

        :return:
            A list. The dimension names that the given variable contains.

        .. code-block:: python

            # Usage Example: Returns the list of the dimension names that the variable "variable_name" contains

            raster1.get_dimension_names(variable_name="variable_name")

        """
        return self._engine_obj.get_dimension_names(variable_name)

    def get_dimension_values(
        self,
        variable_name: str,
        dimension_name: str,
        return_as_datetime_object: bool = False,
    ):
        """
        The ``get_dimension_values`` method retrieves a list of values along the given dimension within the
        given variable.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        variable_name                            Required string. the name of the variable
        ------------------------------------     --------------------------------------------------------------------
        dimension_name                           Required string. the name of the dimension
        ------------------------------------     --------------------------------------------------------------------
        return_as_datetime_object                Set to True, to return the dimension values as datetime object.
                                                 Valid only if the dimension name is StdTime
        ====================================     ====================================================================

        :return:
            A list. The dimension values along the given dimension within the given variable.

        .. code-block:: python

            # Usage Example: Returns the values of a given dimension associated with the given variable.

            raster1.get_dimension_values(variable_name="variable_name",
                                         dimension_name="dimension_name")

        """
        return self._engine_obj.get_dimension_values(
            variable_name, dimension_name, return_as_datetime_object
        )

    def get_dimension_attributes(self, variable_name: str, dimension_name: str):
        """
         The ``get_dimension_attributes`` method retrieves the attribute information of a dimension within a variable,
         such as min value, max value, unit, etc.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        variable_name         Required string. the name of the variable
        -----------------     --------------------------------------------------------------------
        dimension_name        Required string. the name of the dimension
        =================     ====================================================================

        :return:
            A dictionary. The attribute information of the given dimension within the given variable.

        .. code-block:: python

            # Usage Example: Returns specified dimension attribute dictionary for given variable

            raster1.get_dimension_attributes(variable_name="variable_name",
                                             dimension_name="dimension_name")

        """
        return self._engine_obj.get_dimension_attributes(variable_name, dimension_name)

    def rename_variable(self, current_variable_name: str, new_variable_name: str):
        """
        The ``rename_variable`` method renames the given variable name.

        .. note::
            The ``rename_variable`` operation is not supported on image services.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        current_variable_name                    Required string. the name of the variable to be renamed
        ------------------------------------     --------------------------------------------------------------------
        new_variable_name                        Required string. the new variable name
        ====================================     ====================================================================

        :return:
            A List. The dimension names that the given variable contains

        .. code-block:: python

            # Usage Example: Rename variable name

            raster1.rename_variable(current_variable_name="current_variable_name",
                                    new_variable_name="new_variable_name")

        """
        return self._engine_obj.rename_variable(
            current_variable_name, new_variable_name
        )

    def set_property(self, property_name: str, property_value: str):
        """
        The ``set_property`` method adds a customized property to the raster. If the property name exists,
        the existing property value will be overwritten.

        .. note::
            The ``set_property`` operation is not supported on image services

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        property_name         Required string. The property name of the raster
        -----------------     --------------------------------------------------------------------
        property_value        Required string. The value to assign to the property.
        =================     ====================================================================

        :return: None

        .. code-block:: python

            # Usage Example: Add user-defined property name and value to raster

            raster1.set_property(property_name="property_name",
                                 property_value="property_value")

        """
        return self._engine_obj.set_property(property_name, property_value)

    def get_property(self, property_name: str):
        """
        The ``get_property`` method returns the value of the given property.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        property_name         Required string. the name of the property
        =================     ====================================================================

        :return: A string.

        .. code-block:: python

            # Usage Example 1: Returns value of the property

            raster2.get_property(property_name="property_name")

        """
        return self._engine_obj.get_property(property_name)

    def read(
        self,
        upper_left_corner: tuple[int] = (0, 0),
        origin_coordinate: tuple[int] = None,
        ncols: int = 0,
        nrows: int = 0,
        nodata_to_value: Optional[int] = None,
        cell_size: Optional[tuple[int]] = None,
    ):
        """
        The ``read`` method reads a numpy array from the calling :class:`~arcgis.raster.Raster` object.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        upper_left_corner     2-D tuple. a tuple with 2 values representing the number of pixels along x and y
                              direction relative to the origin_coordinate. E.g., (2, 0), means that
                              the real origin to extract the array is 2 pixels away in x
                              direction from the origin_coordinate
        -----------------     --------------------------------------------------------------------
        origin_coordinate     2-d tuple (X, Y). The x and y values are in map units.
                              If no value is specified, the top left corner of the calling raster,
        -----------------     --------------------------------------------------------------------
        ncols                 Integer. the number of columns from the real origin in the calling
                              raster to convert to the NumPy array.
                              If no value is specified, the number of columns of the calling raster
                              will be used. Default: None
        -----------------     --------------------------------------------------------------------
        nrows                 Integer. the number of rows from the real origin in the calling
                              raster to convert to the NumPy array.
                              If no value is specified, the number of rows of the calling raster
                              will be used. Default: None
        -----------------     --------------------------------------------------------------------
        nodata_to_value       Numeric. pixels with nodata values in the raster would be assigned
                              with the given value in  the NumPy array. If no value is specified,
                              the NoData value of the calling raster will be used. Default: None
        -----------------     --------------------------------------------------------------------
        cell_size             2-D tuple. a tuple with 2 values shows the x_cell_size and y_cell_size,
                              e.g., cell_size = (2, 2).
                              if no value is specified, the original cell size of the calling raster
                              will be used. Otherwise, pixels would be resampled to the requested cell_size
        =================     ====================================================================

        :return:
            A numpy.ndarray. If self is a multidimensional raster, the array has shape (slices, height, width, bands)

        .. code-block:: python

            # Usage Example: Reads a numpy array from (2, 2) pixels away from the origin of the raster

            raster1.read(upper_left_corner=(2, 2))

        """
        return self._engine_obj.read(
            upper_left_corner,
            origin_coordinate,
            ncols,
            nrows,
            nodata_to_value,
            cell_size,
        )

    def write(
        self,
        array: np.ndarray,
        upper_left_corner: tuple[int] = (0, 0),
        origin_coordinate: Optional[tuple[int]] = None,
        value_to_nodata: Optional[int] = None,
    ):
        """
        The ``write`` method writes a numpy array to the calling :class:`~arcgis.raster.Raster` object.

        .. note::
            The ``write`` operation is not supported on image services.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        array                 Required numpy.ndarray. the array must be in the shape of (slices,
                              height, width, bands) for writing a multidimensional raster and
                              (height, width bands) for writing a normal raster
        -----------------     --------------------------------------------------------------------
        upper_left_corner     2-D tuple.a tuple with 2 values representing the number of pixels
                              along x and y direction that shows the position relative to the
                              origin_coordinate. E.g., (2, 0), means that the position from which the
                              numpy array will be written into the calling Raster is 2 pixels away
                              in x direction from the origin_coordinate.
                              Default value is (0, 0)
        -----------------     --------------------------------------------------------------------
        origin_coordinate     2-D tuple (X, Y) from where the numpy array will be written
                              into the calling Raster. The x- and y-values are in map units.
                              If no value is specified, the top left corner of the
                              calling raster,
        -----------------     --------------------------------------------------------------------
        value_to_nodata       Numeric. The value in the numpy array assigned to be the
                              NoData values in the calling Raster.


                              If no value is specified, the NoData value of the calling Raster will
                              be used. Default None
        =================     ====================================================================

        :return: None

        .. code-block:: python

            # Usage Example: Write a numpy array (2, 2) pixels away from the origin to the raster

            raster1.write(upper_left_corner=(2, 2))

        """
        return self._engine_obj.write(
            array, upper_left_corner, origin_coordinate, value_to_nodata
        )

    def remove_variables(self, variable_names: list[str]):
        """
        The ``remove_variables`` method removes the given variables.

        .. note::
            The ``remove_variables`` operation is not supported on image services.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        variable_names        Required list. the list of variables to be removed
        =================     ====================================================================

        :return: A List. A list of all variables.

        .. code-block:: python

            # Usage Example: Removes specified variable and returns the list of remaining variables in the raster dataset.

            raster1.remove_variables(variable_names=["variable_name_1", "variable_name_2"])
        """
        return self._engine_obj.remove_variables(variable_names)

    def add_dimension(
        self,
        variable: str,
        new_dimension_name: str,
        dimension_value: str,
        dimension_attributes: Optional[str] = None,
    ):
        """
        The ``add_dimension`` method adds a new dimension to a given variable.

        .. note::
            The ``add_dimension`` operation is not supported on image services

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        variable                                 Required string. variable to which the new dimension is to be added
        ------------------------------------     --------------------------------------------------------------------
        new_dimension_name                       Required string. name of the new dimension to be added
        ------------------------------------     --------------------------------------------------------------------
        dimension_value                          Required string. dimension value
        ------------------------------------     --------------------------------------------------------------------
        dimension_attributes                     Optional attributes of the new dimension like Description, Unit etc.
        ====================================     ====================================================================

        :return: The variable names and their dimensions in the multidimensional raster

        .. code-block:: python

            # Usage Example: Adds a new dimension to the specified variable of multidimensional raster

            raster1 = Raster(r"/path/to/mult_dim.crf")

            raster1.add_dimension(variable="variable_name",
                                  new_dimension_name="new_dimension_name",
                                  dimension_value="dimension_value")

        """
        return self._engine_obj.add_dimension(
            variable,
            new_dimension_name,
            dimension_value,
            dimension_attributes,
        )

    def get_colormap(self, variable_name: Optional[str] = None):
        """
        The ``get_colormap`` method retrieves the color map of the raster.

        .. note::
            If the raster is multidimensional, returns the color map of a variable.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        variable_name                            Optional string. The variable name of the multidimensional raster.
                                                 If a variable is not specified and the raster is multidimensional,
                                                 the color map of the first variable will be returned.
        ====================================     ====================================================================

        :return:
            A dictionary representing the colormap of the raster or the given variable.

        .. code-block:: python

            # Usage Example: Returns the colormap of specificied variable of a multidimensional raster

            raster1.get_colormap(variable_name="variable_name")

        """

        return self._engine_obj.get_colormap(variable_name)

    def set_colormap(
        self,
        color_map: Union[str, dict[str, Any]],
        variable_name: Optional[str] = None,
    ):
        """
        The ``set_colormap`` method sets the color map for the raster.

        .. note::

            * If the raster is multidimensional, it sets the color map for a variable.

            * The ``set_colormap`` operation is not supported on image services

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        color_map                                Optional (string, dict): The color map to apply to the raster. This
                                                 can be a string indicating the name of the color map or color ramp
                                                 to use, for example, NDVI or Yellow To Red, respectively. This can
                                                 also be a Python dictionary with a custom color map or color ramp
                                                 object.

                                                 For example:

                                                 customized colormap object, e.g., {'values': [0, 1, 2, 3, 4, 5, 6], 'colors': ['#000000', '#DCFFDF', '#B8FFBE', '#85FF90', '#50FF60','#00AB10', '#006B0A']}

                                                 colorramp name, e.g., "Yellow To Red"

                                                 colormap name, e.g., "NDVI"

                                                 customized colorramp object, e.g., {"type": "algorithmic", "fromColor": [115, 76, 0, 255],"toColor": [255, 25, 86, 255], "algorithm": "esriHSVAlgorithm"}
        ------------------------------------     --------------------------------------------------------------------
        variable_name                            Optional string. The variable name of the multidimensional raster dataset.
                                                 If a variable is not specified and the raster is multidimensional, the color
                                                 map of the first variable will be set.
        ====================================     ====================================================================

        :return: None

        .. code-block:: python

            # Usage Example: Sets "NDVI" color map for the raster

            raster1.set_colormap(color_map="NDVI")

        """
        return self._engine_obj.set_colormap(color_map, variable_name)

    def get_statistics(self, variable_name: Optional[str] = None):
        """
        The ``get_statistics`` method retrieves the statistics of the raster.

        .. note::
            If the raster is multidimensional, returns the statistics of a variable.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        variable_name                            Optional string. The variable name of the multidimensional raster dataset.
                                                 If a variable is not specified and the raster is multidimensional,
                                                 the statistics of the first variable will be returned.
        ====================================     ====================================================================

        :return:
            A dictionary. The statistics of the raster or the given variable.

        .. code-block:: python

            # Usage Example: Returns the statistics of the raster

            raster1.get_statistics()

        """

        return self._engine_obj.get_statistics(variable_name)

    def set_statistics(
        self,
        statistics_obj: Optional[list[dict[str, float]]],
        variable_name: Optional[str] = None,
    ):
        """
        The ``set_statistics`` method sets the statistics for the raster.
        If the raster is multiband, it sets the statistics for each band.
        If the raster is multidimensional, it sets the statistics for a variable.

        .. note::
            The ``set_statistics`` operation is not supported on image services.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        statistics_obj                           Optional list of statistics objects. A list of Python dictionaries containing statistics and corresponding
                                                 values to set. For example, [{'min': 10, 'max': 20}] sets the minimum
                                                 and maximum pixel values.

                                                 If the raster is multiband, the statistics for each band will be set with
                                                 each dictionary in the list. The first band will use the statistics in the
                                                 first dictionary. The second band will use the statistics in the second
                                                 dictionary, and so on.

                                                 min - The minimum pixel value
                                                 max - The maximum pixel value
                                                 mean - The mean pixel value
                                                 median - The median pixel value
                                                 standardDeviation - The standard deviation of the pixel values
                                                 count - The total number of pixels
                                                 skipX - The horizontal skip factor
                                                 skipY - The vertical skip factor

                                                 For example:

                                                 [{'min': val, 'max': val, 'mean': val, 'standardDeviation': val,
                                                 'median': val, 'mode': val, 'count': val}, ...]
        ------------------------------------     --------------------------------------------------------------------
        variable_name                            Optional string. The variable name of the multidimensional raster.
                                                 If a variable is not specified and the raster is multidimensional,
                                                 the statistics of the first variable will be set.
        ====================================     ====================================================================

        :return: None

        .. code-block:: python

            # Usage Example: Sets statistics of the raster

            raster1.set_statistics(statistics_obj=[{"min": val,
                                                    "max": val,
                                                    "mean": val,
                                                    "standardDeviation": val}],
                                   variable_name="variable_name")

        """

        return self._engine_obj.set_statistics(statistics_obj, variable_name)

    def get_histograms(self, variable_name: Optional[str] = None):
        """
        The ``get_histograms`` method retrieves the histograms of the raster.

        .. note::
            If the raster is multidimensional, it returns the histogram of a variable.
            If the raster is multiband, it returns the histogram of each band.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        variable_name                            Optional string. The variable name of the multidimensional raster dataset.
                                                 If a variable is not specified and the raster is multidimensional,
                                                 the histogram of the first variable will be returned.
        ====================================     ====================================================================

        :return:
            A list of dictionaries. The histogram values of the raster or variable.

        .. code-block:: python

            # Usage Example: Returns the histograms of the raster

            raster1.get_histograms()

        """

        return self._engine_obj.get_histograms(variable_name)

    def set_histograms(
        self,
        histogram_obj: list[dict[str, float]],
        variable_name: Optional[str] = None,
    ):
        """
        The ``set_histograms`` method sets the histogram for the raster or a given variable, if the raster is
        multidimensional.

        .. note::
            The ``set_histograms`` operation is not supported on image services

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        histogram_obj                            Optional list of histogram objects(dict),

                                                 If the raster is multiband, the histogram for each band will be set
                                                 with each dictionary in the list. The first band will use the histogram
                                                 in the first dictionary. The second band will use the histogram in
                                                 the second dictionary, and so on.

                                                 size - The number of bins in the histogram

                                                 min - The minimum pixel value

                                                 max - The maximum pixel value

                                                 counts - A list containing the number of pixels in each bin, in the order of bins

                                                 For example:

                                                 [{'size': number_of_bins, 'min': min_val, 'max': max_val, 'counts': [pixel_count_at_each_bin, ...]}, ...]
        ------------------------------------     --------------------------------------------------------------------
        variable_name                            Optional string. The variable name of the multidimensional raster dataset.
                                                 If a variable is not specified and the raster is multidimensional,
                                                 the histogram will be set for the first variable.
        ====================================     ====================================================================

        :return: None

        .. code-block:: python

            # Usage Example: Sets specified histograms for the raster

            raster1.set_histograms(histogram_obj=[{"size": number_of_bins,
                                                   "min": min_val,
                                                   "max": max_val,
                                                   "counts": [pixel_count_at_each_bin]}])
        """

        return self._engine_obj.set_histograms(histogram_obj, variable_name)

    def append_slices(self, md_raster: Optional[Raster] = None):
        """
        The ``append_slices`` method appends the slices from another multidimensional raster.

        .. note::
            The ``add_dimension`` operation is not supported on image services

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        md_raster             Required multidimensional raster. The multidimensional raster containing
                              the slices to be appended.

                              This raster must have the same variables, with the same dimension names,
                              as the target raster. The cell sizes, extents, and spatial reference
                              systems must also match.

                              The slices in this raster must be for dimension values that follow
                              the dimension values of the slices in the target raster.

                              If a variable has two dimensions, slices will be appended along
                              one dimension. The other dimension must have the same number of
                              slices as the dimension in the target raster.

                              For example, if a salinity variable contains slices over time and
                              depth dimensions, time slices can be appended to another salinity
                              multidimensional raster but only if the same number of depth slices
                              exist in both rasters.
        =================     ====================================================================

        :return (string): A string containing the variable names and the associated dimensions in the multidimensional raster.
                           For example, if the resulting raster has 10 time slices with precipitation data, it will return 'prcp(StdTime=10)'.

        .. code-block:: python

            # Usage Example: Append slices to target raster from source multidimensional raster

            target_raster.append_slices(md_raster=source_raster_obj)

        """
        return self._engine_obj.append_slices(md_raster)

    def set_variable_attributes(
        self, variable_name: str, variable_attributes: dict[str, Any]
    ):
        """
        The ``set_variable_attributes`` method sets the attribute information of a variable in a multidimensional raster.
        For example, description, unit, and so on.

        .. note::
            The ``set_variable_attributes`` operation is not supported on image services

        ====================================     ====================================================================
        **Parameter**                            **Description**
        ------------------------------------     --------------------------------------------------------------------
        variable_name                            Required string. The variable name of the multidimensional raster dataset.
        ------------------------------------     --------------------------------------------------------------------
        variable_attributes                      Required dict that contains attribute information to replace the current
                                                 attribute information of the variable.

                                                 For example:

                                                 {"Description": "Daily total precipitation", "Unit": "mm/day"}.
        ====================================     ====================================================================

        :return:
            The attribute information of the variable as a dictionary

        .. code-block:: python

            # Usage Example: Sets variable attributes to the specified variable

            raster1.set_variable_attributes(variable_name="variable_name",
                                            variable_attributes={"attribute_1": "value_1",
                                                                 "attribute_2": "value_2"})

        """
        return self._engine_obj.set_variable_attributes(
            variable_name, variable_attributes
        )

    def summarize(
        self,
        geometry: Union[Geometry, Polygon, Envelope],
        pixel_size: Optional[Union[str, dict[str, float]]] = None,
    ):
        """
        The ``summarize`` method retrieves the statistics of a :class:`~arcgis.raster.Raster` for a given geometry.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        geometry              Required :class:`~arcgis.geometry.Geometry` (:class:`~arcgis.geometry.Polygon`
                              or :class:`~arcgis.geometry.Envelope`).
                              A geometry that defines the geometry within which the statistics are computed.
        -----------------     --------------------------------------------------------------------
        pixel_size            Optional string or dict. The pixel level being used (or the
                              resolution being looked at). If pixel size is not specified, then
                              pixel_size will default to the base resolution of the dataset. The
                              raster at the specified pixel size in the mosaic dataset will be
                              used for histogram calculation.

                              Syntax:
                                - dictionary structure: pixel_size={point}
                                - Point simple syntax: pixel_size='<x>,<y>'
                              Examples:
                                - pixel_size={"x": 0.18, "y": 0.18}
                                - pixel_size='0.18,0.18'
        =================     ====================================================================

        :return: A dictionary. (Dictionary at each index represents the statistics of the corresponding band.)

                      | [{
                      |     "min": 0,
                      |     "max": 9,
                      |     "mean": 3.271703916996627,
                      |     "standardDeviation": 1.961013669880657,
                      |     "median": 4,
                      |     "mode": 4,
                      |     "skipX": 1,
                      |     "skipY": 1,
                      |     "count": 2004546
                      |   }]

        .. code-block:: python

            # Usage Example: Summarize a raster at an area.

            stats = raster.summarize(geometry=geom_obj)
            mean_of_first_band = stats[0]["mean"]

        """
        return self._engine_obj.summarize(geometry=geometry, pixel_size=pixel_size)

    @property
    def _lyr_json(self):
        return self._engine_obj._lyr_json

    def save(
        self,
        output_name: Optional[str] = None,
        for_viz: bool = False,
        process_as_multidimensional: Optional[bool] = None,
        build_transpose: Optional[bool] = None,
        gis: Optional[GIS] = None,
        future: bool = False,
        **kwargs,
    ):
        """
        When run using image_server engine, ``save`` method persists this :class:`~arcgis.raster.Raster` to the
        :class:`~arcgis.gis.GIS` as an :class:`~arcgis.raster.ImageryLayer` item.

        If ``for_viz`` is ``True``, a new :class:`~arcgis.gis.Item` is created that uses the applied raster
        functions for visualization at display resolution using on-the-fly image processing.
        If ``for_viz`` is ``False``, distributed raster analysis is used for generating a new raster information
        product by applying raster functions at source resolution across the extent of the output imagery layer.

        .. note::
            When run using arcpy engine, ``save`` persists this raster to location specified in ``output_name``.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        output_name                              Optional string.

                                                 When run using image_server engine, specify output name.
                                                 If not provided, an Imagery Layer item is created
                                                 by the method and used as the output.
                                                 You can pass in the name of the output raster that should be
                                                 created by this method to be used as the output for the tool.
                                                 Alternatively, if for_viz is False, you can pass in an existing
                                                 Image Layer Item from your GIS to use that instead.
                                                 A RuntimeError is raised if a layer by that name already exists

                                                 When run using arcpy engine, output_name is the name string
                                                 representing the output location.
        ------------------------------------     --------------------------------------------------------------------
        for_viz                                  Optional boolean. If True, a new Item is created that uses the
                                                 applied raster functions for visualization at display resolution
                                                 using on-the-fly image processing.
                                                 If for_viz is False, distributed raster analysis is used for
                                                 generating a new raster information product for use in analysis and
                                                 visualization by applying raster functions at source resolution
                                                 across the extent of the output raster.

                                                 (Available only when image_server engine is used)
        ------------------------------------     --------------------------------------------------------------------
        process_as_multidimensional              Optional bool.  If the input is multidimensional raster, the output
                                                 will be processed as multidimensional if set to True
        ------------------------------------     --------------------------------------------------------------------
        build_transpose                          Optional bool, if set to true, transforms the output
                                                 multidimensional raster. Valid only if process_as_multidimensional
                                                 is set to True
        ------------------------------------     --------------------------------------------------------------------
        gis                                      Optional arcgis.gis.GIS object. The GIS to be used for saving the
                                                 output. Keyword only parameter.

                                                 (Available only when image_server engine is used)
        ------------------------------------     --------------------------------------------------------------------
        future                                   Optional boolean. If True, the result will be a GPJob object and
                                                 results will be returned asynchronously. Keyword only parameter.

                                                 (Available only when image_server engine is used)
        ------------------------------------     --------------------------------------------------------------------
        folder                                   Optional string or dictionary. Creates a folder in the portal, if it does
                                                 not exist, with the given folder name and persists the output in this folder.
                                                 The dictionary returned by the gis.content.create_folder() can also be passed in as input.

                                                 (Available only when image_server engine is used)

                                                 Example:
                                                    {'username': 'user1', 'id': '6a3b77c187514ef7873ba73338cf1af8', 'title': 'trial'}
        ------------------------------------     --------------------------------------------------------------------
        tiles_only                               In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                                 To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                                 Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.

                                                 (Available only when image_server engine is used)
        ====================================     ====================================================================

        :return: A String representing the location of the output data

        .. code-block:: python

            # Usage Example 1: Saves the local raster output to a new location (usecase for arcpy engine rasters)

            raster1.save(output_name=r"/path/to/output_location/raster.crf",
                         process_as_multidimensional=True)

            # Usage Example 2: Saves the raster to the active GIS as an Imagery Layer Item (usecase for image_server engine rasters)

            raster2.save(output_name="output_imagery_layer_name",
                         folder="my_rasters",
                         gis=gis)

        """
        return self._engine_obj.save(
            output_name,
            for_viz,
            process_as_multidimensional,
            build_transpose,
            gis,
            future,
            **kwargs,
        )

    def _repr_png_(self):
        return self._engine_obj._repr_png_()

    def _repr_jpeg_(self):
        return self._engine_obj._repr_jpeg_()

    def plot_histograms(
        self,
        geometry: Optional[Union[Geometry, Polygon, Envelope]] = None,
        pixel_size: Optional[Union[str, dict[str, float]]] = None,
        time: Optional[Union[datetime.date, datetime.datetime, str, list[int]]] = None,
        bands: list[int] = [],
        display_stats: bool = True,
        plot_properties: Optional[dict[str, Any]] = None,
        subplot_properties: Optional[
            Union[list[dict[str, Any]], dict[str, Any]]
        ] = None,
    ):
        """
        The ``plot_histograms`` method plots the image histograms of a raster.
        
        Image histograms visually summarize the distribution of a continuous numeric variable by measuring
        the frequency at which certain values appear in the image. The x-axis in the image histogram is a 
        number line that displays the range of image pixel values that has been split into number ranges, 
        or bins. A bar is drawn for each bin, and the width of the bar represents the density number range 
        of the bin; the height of the bar represents the number of pixels that fall into that range. 
        Understanding the distribution of your data is an important step in the data exploration process.
    
        ``plot_histograms`` can be used for plotting the band-wise image histogram charts of any
        :class:`~arcgis.raster.Raster` object.
    
        ============================    ====================================================================
        **Parameter**                   **Description**
        ----------------------------    --------------------------------------------------------------------
        geometry                        Optional :class:`~arcgis.geometry.Geometry` (:class:`~arcgis.geometry.Polygon`
                                        or :class:`~arcgis.geometry.Envelope`).
                                        A geometry that defines the geometry within which the histogram is computed. 
                                        
                                        If not provided, then the full extent of the 
                                        raster will be used for the computation.

                                        **Note:** This parameter is honoured if the raster uses ``image_server`` engine.
        ----------------------------    --------------------------------------------------------------------
        pixel_size                      Optional string or dictionary. The pixel level being used (or the
                                        resolution being looked at). If pixel size is not specified, then
                                        pixel_size will default to the base resolution of the dataset.
                                        The structure of the pixel_size parameter is the same as the
                                        structure of the point object returned by the ArcGIS REST API.
                                        In addition to the dictionary structure, you can specify the pixel size
                                        with a comma-separated string.
                                        
                                        Syntax:
                                        - dictionary structure: pixel_size={point}
                                        - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                        - pixel_size={"x": 0.18, "y": 0.18}
                                        - pixel_size='0.18,0.18'

                                        **Note:** This parameter is honoured if the raster uses "image_server" engine.
        ----------------------------    --------------------------------------------------------------------
        time                            Optional datetime.date, datetime.datetime or timestamp string. The
                                        time instant or the time extent to compute statistics and histograms.
                                        Time instant specified as datetime.date, datetime.datetime or
                                        timestamp in milliseconds since epoch
                                        Syntax: time=<timeInstant>
                                        
                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp
                                        
                                        Added at 10.8

                                        **Note:** This parameter is honoured if the raster uses "image_server" engine.
        ----------------------------    --------------------------------------------------------------------
        bands                           Optional list of band indices. By default takes the first band (band index - 0).
                                        Image histogram charts are plotted for these specific bands.
    
                                        Example:
                                            - [0,2,3]
        ----------------------------    --------------------------------------------------------------------
        display_stats                   Optional boolean. Specifies whether to plot the band-wise statistics 
                                        along with the histograms.
    
                                        Some basic descriptive statistics are calculated and displayed on 
                                        histograms. The mean and median are displayed with one line each, and 
                                        one standard deviation above and below the mean is displayed using two lines.
    
                                            - False - The statistics will not be displayed along with the histograms.
                                            - True - The statistics will be displayed along with the histograms. \
                                                    This is the default.
        ----------------------------    --------------------------------------------------------------------
        plot_properties                 Optional dictionary. This parameter can be used to set the figure 
                                        properties. These are the `matplotlib.pyplot.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`__ 
                                        parameters and values specified in dictionary format.
    
                                        Example:
                                            - {"figsize":(15,15)}
        ----------------------------    --------------------------------------------------------------------
        subplot_properties              Optional list or dictionary. This parameter can be used to set band-wise 
                                        histogram (subplot) display properties. These are the `matplotlib.axes.Axes.bar() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib-axes-axes-bar>`__
                                        parameters and values specified in dictionary format.
    
                                        Example:
                                            - | [
                                            |  {"color":"r"},
                                            |  {"color":"g"},
                                            |  {"color":"b","edgecolor":"w"}
                                            | ]
                                            
                                        **Note:** `matplotlib.axes.Axes.bar() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib-axes-axes-bar>`__
                                        parameters: ''x', 'height' or 'align' cannot be passed into subplot_properties.
        ============================    ====================================================================

        .. tip::
        
            When working with multidimensional rasters, you can use the
            `multidimensional_filter() <https://developers.arcgis.com/python/api-reference/arcgis.raster.functions.html#multidimensional-filter>`__
            raster function on the Raster object for slicing the data along defined variables and dimensions.
            ``plot_histograms`` can then be used on the output raster returned upon applying the filter.
        
        :return: None

        .. code-block:: python

            # Usage Example: Plots histograms of the raster with specified resolution and bands

            raster1.plot_histograms(pixel_size="0.18, 0.18", bands=[1, 2, 3])
    
        """
        return self._engine_obj.plot_histograms(
            geometry=geometry,
            pixel_size=pixel_size,
            time=time,
            bands=bands,
            display_stats=display_stats,
            plot_properties=plot_properties,
            subplot_properties=subplot_properties,
        )

    def export_image(
        self,
        bbox: Optional[Union[str, dict[str, Any]]] = None,
        image_sr: Optional[Union[str, SpatialReference]] = None,
        bbox_sr: Optional[Union[str, SpatialReference]] = None,
        size: Optional[list[int]] = None,
        time: Optional[Union[datetime.date, datetime.datetime, str, list[int]]] = None,
        export_format: str = "jpgpng",
        pixel_type: Optional[str] = None,
        no_data: Optional[float] = None,
        no_data_interpretation: str = "esriNoDataMatchAny",
        interpolation: Optional[str] = None,
        compression: Optional[str] = None,
        compression_quality: Optional[int] = None,
        band_ids: Optional[list[int]] = None,
        mosaic_rule: Optional[dict[str, Any]] = None,
        rendering_rule: Optional[dict[str, Any]] = None,
        f: str = "image",
        save_folder: Optional[str] = None,
        save_file: Optional[str] = None,
        compression_tolerance: Optional[float] = None,
        adjust_aspect_ratio: Optional[bool] = None,
        lerc_version: Optional[int] = None,
    ):
        """
        The ``export_image`` operation is performed on a raster layer to visualise it.

        ======================  ====================================================================
        **Parameter**           **Description**
        ----------------------  --------------------------------------------------------------------
        bbox                    Optional dict or string. The extent (bounding box) of the exported
                                image. Unless the bbox_sr parameter has been specified, the bbox is
                                assumed to be in the spatial reference of the raster layer.
                                The bbox should be specified as an arcgis.geometry.Envelope object,
                                it's json representation or as a list or string with this
                                format: '<xmin>, <ymin>, <xmax>, <ymax>'
                                If omitted, the extent of the raster layer is used
        ----------------------  --------------------------------------------------------------------
        image_sr                Optional string, SpatialReference. The spatial reference of the
                                exported image. The spatial reference can be specified as either a
                                well-known ID, it's json representation or as an
                                arcgis.geometry.SpatialReference object.
                                If the image_sr is not specified, the image will be exported in the
                                spatial reference of the raster.
        ----------------------  --------------------------------------------------------------------
        bbox_sr                 Optional string, SpatialReference. The spatial reference of the
                                bbox.
                                The spatial reference can be specified as either a well-known ID,
                                it's json representation or as an arcgis.geometry.SpatialReference
                                object.
                                If the image_sr is not specified, bbox is assumed to be in the
                                spatial reference of the raster.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        size                    Optional list. The size (width * height) of the exported image in
                                pixels. If size is not specified, an image with a default size of
                                400*450 will be exported.
                                Syntax: list of [width, height]
        ----------------------  --------------------------------------------------------------------
        time                    Optional datetime.date, datetime.datetime or timestamp string. The
                                time instant or the time extent of the exported image.
                                Time instant specified as datetime.date, datetime.datetime or
                                timestamp in milliseconds since epoch
                                Syntax: time=<timeInstant>
                                Time extent specified as list of [<startTime>, <endTime>]
                                For time extents one of <startTime> or <endTime> could be None. A
                                None value specified for start time or end time will represent
                                infinity for start or end time respectively.
                                Syntax: time=[<startTime>, <endTime>] ; specified as
                                datetime.date, datetime.datetime or timestamp
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        export_format           Optional string. The format of the exported image. The default
                                format is jpgpng. The jpgpng format returns a JPG if there are no
                                transparent pixels in the requested extent; otherwise, it returns a
                                PNG (png32).
                                Values: jpgpng,png,png8,png24,jpg,bmp,gif,tiff,png32,bip,bsq,lerc
        ----------------------  --------------------------------------------------------------------
        pixel_type              Optional string. The pixel type, also known as data type, pertains
                                to the type of values stored in the raster, such as signed integer,
                                unsigned integer, or floating point. Integers are whole numbers,
                                whereas floating points have decimals.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        no_data                 Optional float. The pixel value representing no information.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        no_data_interpretation  Optional string. Interpretation of the no_data setting. The default
                                is NoDataMatchAny when no_data is a number, and NoDataMatchAll when
                                no_data is a comma-delimited string: NoDataMatchAny,NoDataMatchAll.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        interpolation           Optional string. The resampling process of extrapolating the pixel
                                values while transforming the raster dataset when it undergoes
                                warping or when it changes coordinate space.
                                One of: RSP_BilinearInterpolation, RSP_CubicConvolution,
                                RSP_Majority, RSP_NearestNeighbor
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        compression             Optional string. Controls how to compress the image when exporting
                                to TIFF format: None, JPEG, LZ77. It does not control compression on
                                other formats.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        compression_quality     Optional integer. Controls how much loss the image will be subjected
                                to by the compression algorithm. Valid value ranges of compression
                                quality are from 0 to 100.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        band_ids                Optional list. If there are multiple bands, you can specify a single
                                band to export, or you can change the band combination (red, green,
                                blue) by specifying the band number. Band number is 0 based.
                                Specified as list of ints, eg [2,1,0]
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        mosaic_rule             Optional dict. Specifies the mosaic rule when defining how
                                individual images should be mosaicked. When a mosaic rule is not
                                specified, the default mosaic rule of the image layer will be used
                                (as advertised in the root resource: defaultMosaicMethod,
                                mosaicOperator, sortField, sortValue).
        ----------------------  --------------------------------------------------------------------
        rendering_rule          Optional dict. Specifies the rendering rule for how the requested
                                image should be rendered.
        ----------------------  --------------------------------------------------------------------
        f                       Optional string. The response format.  default is json
                                Values: json,image,kmz,numpy_array

                                **Note:** If f="numpy_array" and if the raster is a single or multiband raster,
                                the dimensions of the array will be rows, columns, and number of bands.
                                If the raster is a multidimensional raster, the dimensions of the array
                                will be number of slices, rows, columns, and number of bands.
                                LERC needs to be installed to export image service as numpy array.

                                If f="image", the bytes of the exported image are
                                returned unless save_folder and save_file parameters are also
                                passed, in which case the image is written to the specified file
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        save_folder             Optional string. The folder in which the exported image is saved
                                when f=image
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        save_file               Optional string. The file in which the exported image is saved when
                                f=image
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        compression_tolerance   Optional float. Controls the tolerance of the lerc compression
                                algorithm. The tolerance defines the maximum possible error of pixel
                                values in the compressed image.
                                Example: compression_tolerance=0.5 is loseless for 8 and 16 bit
                                images, but has an accuracy of +-0.5 for floating point data. The
                                compression tolerance works for the LERC format only.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        adjust_aspect_ratio     Optional boolean. Indicates whether to adjust the aspect ratio or
                                not. By default adjust_aspect_ratio is true, that means the actual
                                bbox will be adjusted to match the width/height ratio of size
                                paramter, and the response image has square pixels.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        lerc_version            Optional integer. The version of the Lerc format if the user sets
                                the format as lerc.
                                Values: 1 or 2
                                If a version is specified, the server returns the matching version,
                                or otherwise the highest version available.
                                (Available only when image_server engine is used)
        ======================  ====================================================================

        :return: The raw raster data

        .. code-block:: python

            # Usage Example: Creates Raster object from a local dataset location and exports the image.

            raster_source = Raster(r"/path/to/raster")
            raster_source.export_image(size=[1000, 1000])

        """

        return self._engine_obj.export_image(
            bbox,
            image_sr,
            bbox_sr,
            size,
            time,
            export_format,
            pixel_type,
            no_data,
            no_data_interpretation,
            interpolation,
            compression,
            compression_quality,
            band_ids,
            mosaic_rule,
            rendering_rule,
            f,
            save_folder,
            save_file,
            compression_tolerance,
            adjust_aspect_ratio,
            lerc_version,
        )

    def draw_graph(
        self, show_attributes: bool = False, graph_size: str = "14.25, 15.25"
    ):
        """
        The ``draw_graph`` method displays a structural representation of the function chain and it's raster input
        values. If
        ``show_attributes`` is set to ``True``, then the ``draw_graph`` function also displays the attributes
        of all the functions in the function chain, representing the rasters in a blue rectangular
        box, attributes in green rectangular box and the raster function names in yellow.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        show_attributes       Optional boolean. If True, the graph displayed includes all the
                              attributes of the function and not only it's function name and raster
                              inputs
                              Set to False by default, to display only the raster function name and
                              the raster inputs to it.
        -----------------     --------------------------------------------------------------------
        graph_size            Optional string. Maximum width and height of drawing, in inches,
                              seperated by a comma. If only a single number is given, this is used
                              for both the width and the height. If defined and the drawing is
                              larger than the given size, the drawing is uniformly scaled down so
                              that it fits within the given size.
        =================     ====================================================================

        :return: A Graph

        .. code-block:: python

            # Usage Example 1: Draws the function chain applied on the Raster object created from an Image service.

            service_url = gis.content.search("my_image_service", item_type="Imagery Layer")[0].url
            raster = Raster(service_url, gis=gis)
            grayscale_raster = grayscale(raster=raster)
            invert_raster = boolean_not(rasters=[grayscale_raster])
            invert_raster.draw_graph(show_attributes=True)

            # Usage Example 2:  Draws the function chain applied on the Raster object created from local dataset

            raster = Raster(r"/path/to/raster")
            ndvi_raster = ndvi(raster=raster, band_indexes="5 6")
            ndvi_raster.draw_graph(show_attributes=True)

        """

        return self._engine_obj.draw_graph(show_attributes, graph_size)

    def __sub__(self, other):
        return self._engine_obj.__sub__(other)

    def __rsub__(self, other):
        return self._engine_obj.__rsub__(other)

    def __add__(self, other):
        return self._engine_obj.__add__(other)

    def __radd__(self, other):
        return self._engine_obj.__radd__(other)

    def __mul__(self, other):
        return self._engine_obj.__mul__(other)

    def __rmul__(self, other):
        return self._engine_obj.__rmul__(other)

    def __div__(self, other):
        return self._engine_obj.__div__(other)

    def __rdiv__(self, other):
        return self._engine_obj.__rdiv__(other)

    def __pow__(self, other):
        return self._engine_obj.__pow__(other)

    def __rpow__(self, other):
        return self._engine_obj.__rpow__(other)

    def __abs__(self):
        return self._engine_obj.__abs__()

    def __lshift__(self, other):
        return self._engine_obj.__lshift__(other)

    def __rlshift__(self, other):
        return self._engine_obj.__rlshift__(other)

    def __rshift__(self, other):
        return self._engine_obj.__rshift__(other)

    def __rrshift__(self, other):
        return self._engine_obj.__rrshift__(other)

    def __floordiv__(self, other):
        return self._engine_obj.__floordiv__(other)

    def __rfloordiv__(self, other):
        return self._engine_obj.__rfloordiv__(other)

    def __truediv__(self, other):
        return self._engine_obj.__truediv__(other)

    def __rtruediv__(self, other):
        return self._engine_obj.__rtruediv__(other)

    def __mod__(self, other):
        return self._engine_obj.__mod__(other)

    def __rmod__(self, other):
        return self._engine_obj.__rmod__(other)

    def __neg__(self):
        return self._engine_obj.__neg__()

    def __invert__(self):
        return self._engine_obj.__invert__()

    def __and__(self, other):
        return self._engine_obj.__and__(other)

    def __rand__(self, other):
        return self._engine_obj.__rand__(other)

    def __xor__(self, other):
        return self._engine_obj.__xor__(other)

    def __rxor__(self, other):
        return self._engine_obj.__rxor__(other)

    def __or__(self, other):
        return self._engine_obj.__or__(other)

    def __ror__(self, other):
        return self._engine_obj.__ror__(other)

    def __ne__(self, other):
        return self._engine_obj.__ne__(other)

    def __eq__(self, other):
        return self._engine_obj.__eq__(other)

    def __gt__(self, other):
        return self._engine_obj.__gt__(other)

    def __ge__(self, other):
        return self._engine_obj.__ge__(other)

    def __lt__(self, other):
        return self._engine_obj.__lt__(other)

    def __le__(self, other):
        return self._engine_obj.__le__(other)

        # bbox_sr = None
        # if not isinstance(bbox, arcpy.arcobjects.Extent):
        #     bbox_list = []
        #     arcpy_sr = None
        #     if bbox is not None:
        #         if type(bbox) == str:
        #             bbox_list = bbox.split(",")
        #         elif isinstance(bbox, list):
        #             bbox_list = bbox
        #         elif isinstance(bbox, dict):
        #             bbox_list.append(bbox['xmin'])
        #             bbox_list.append(bbox['ymin'])
        #             bbox_list.append(bbox['xmax'])
        #             bbox_list.append(bbox['ymax'])
        #
        #             if ('spatialReference' in bbox.keys()):
        #                 if 'wkt' in bbox['spatialReference'].keys():
        #                     bbox_sr = bbox['spatialReference']['wkt']
        #                 elif 'wkid' in bbox['spatialReference'].keys():
        #                     bbox_sr = bbox['spatialReference']['wkid']
        #     if bbox_sr is not None:
        #         if not isinstance(bbox_sr, arcpy.arcobjects.SpatialReference):
        #             try:
        #                 arcpy_sr = arcpy.SpatialReference(bbox_sr)
        #             except:
        #                 arcpy_sr = arcpy.SpatialReference()
        #                 arcpy_sr.loadFromString(bbox_sr)
        #         else:
        #             arcpy_sr = bbox_sr
        #     if (bbox_list and arcpy_sr):
        #         bbox = arcpy.Extent(bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3], spatial_reference=arcpy_sr)
        #     elif bbox_list:
        #         bbox = arcpy.Extent(bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3])
        #
        # columns = 400
        # rows = 400
        #
        # if size is not None and isinstance(size, list):
        #     columns = size[0]
        #     rows = size[1]
        #
        # if mosaic_rule is not None:
        #     mosaic_rule = mosaic_rule
        # elif self._mosaic_rule is not None:
        #     mosaic_rule = self._mosaic_rule
        #
        # __allowedFormat = ["jpgpng", "png",
        #                    "png8", "png24",
        #                    "jpg", "bmp",
        #                    "gif", "tiff",
        #                    "png32", "bip", "bsq", "lerc"]
        #
        # if export_format in __allowedFormat:
        #     export_format = export_format
        #
        # if rendering_rule is not None:
        #     if 'function_chain' in rendering_rule:
        #         rendering_rule = rendering_rule['function_chain']
        #     else:
        #         rendering_rule = rendering_rule
        # elif self._fn is not None:
        #     if not self._uses_gbl_function:
        #         rendering_rule = self._fn
        #     else:
        #         _LOGGER.warning("""Imagery layer object containing global functions in the function chain cannot be used for dynamic visualization.
        #                             \nThe layer output must be saved as a new image service before it can be visualized. Use save() method of the layer object to create the processed output.""")
        #         return None
        #
        # return self._raster.export(w=columns, h=rows, bbox=bbox, sr=image_sr, rr=rendering_rule, mr=mosaic_rule,
        #                            f=export_format)

    '''
    def _repr_jpeg_(self):
        if self._remote:
            bbox_sr = None
            if 'spatialReference' in self.extent:
                bbox_sr = self.extent['spatialReference']
            if not self._uses_gbl_function:
                return super().export_image(bbox=self._extent, bbox_sr=bbox_sr, size=[1200, 450],
                                            export_format='jpgpng', f='image')
        return self.export_image(bbox=self._extent, size=[400, 400], export_format='jpgpng')

    def mosaic_by(self, method=None, sort_by=None, sort_val=None, lock_rasters=None, viewpt=None, asc=True, where=None,
                  fids=None,
                  muldidef=None, op="first", item_rendering_rule=None):
        """
        Defines how individual images in this layer should be mosaicked. It specifies selection,
        mosaic method, sort order, overlapping pixel resolution, etc. Mosaic rules are for mosaicking rasters in
        the mosaic dataset. A mosaic rule is used to define:
        * The selection of rasters that will participate in the mosaic (using where clause).
        * The mosaic method, e.g. how the selected rasters are ordered.
        * The mosaic operation, e.g. how overlapping pixels at the same location are resolved.
        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        method                optional string. Determines how the selected rasters are ordered.
                              str, can be none,center,nadir,northwest,seamline,viewpoint,
                              attribute,lock-raster
                              required if method is: center,nadir,northwest,seamline, optional
                              otherwise. If no method is passed "none" method is used, which uses
                              the order of records to sort
                              If sort_by and optionally sort_val parameters are specified,
                              "attribute" method is used
                              If lock_rasters are specified, "lock-raster" method is used
                              If a viewpt parameter is passed, "viewpoint" method is used.
        -----------------     --------------------------------------------------------------------
        sort_by               optional string. field name when sorting by attributes
        -----------------     --------------------------------------------------------------------
        sort_val              optional string. A constant value defining a reference or base value
                              for the sort field when sorting by attributes
        -----------------     --------------------------------------------------------------------
        lock_rasters          optional, an array of raster Ids. All the rasters with the given
                              list of raster Ids are selected to participate in the mosaic. The
                              rasters will be visible at all pixel sizes regardless of the minimum
                              and maximum pixel size range of the locked rasters.
        -----------------     --------------------------------------------------------------------
        viewpt                optional point, used as view point for viewpoint mosaicking method
        -----------------     --------------------------------------------------------------------
        asc                   optional bool, indicate whether to use ascending or descending
                              order. Default is ascending order.
        -----------------     --------------------------------------------------------------------
        where                 optional string. where clause to define a subset of rasters used in
                              the mosaic, be aware that the rasters may not be visible at all
                              scales
        -----------------     --------------------------------------------------------------------
        fids                  optional list of objectids, use the raster id list to define a
                              subset of rasters used in the mosaic, be aware that the rasters may
                              not be visible at all scales.
        -----------------     --------------------------------------------------------------------
        muldidef              optional array. multidemensional definition used for filtering by
                              variable/dimensions.
                              See https://developers.arcgis.com/documentation/common-data-types/multidimensional-definition.htm
        -----------------     --------------------------------------------------------------------
        op                    optional string, first,last,min,max,mean,blend,sum mosaic operation
                              to resolve overlap pixel values: from first or last raster, use the
                              min, max or mean of the pixel values, or blend them.
        -----------------     --------------------------------------------------------------------
        item_rendering_rule   optional item rendering rule, applied on items before mosaicking.
        =================     ====================================================================
        :return: a mosaic rule defined in the format at
            https://developers.arcgis.com/documentation/common-data-types/mosaic-rules.htm
        Also see http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/understanding-the-mosaicking-rules-for-a-mosaic-dataset.htm#ESRI_SECTION1_ABDC9F3F6F724A4F8079051565DC59E
        """
        if self._datastore_raster:
            raise RuntimeError("This operation cannot be performed on a datastore raster")
        mosaic_rule = {
            "mosaicMethod": "esriMosaicNone",
            "ascending": asc,
            "mosaicOperation": 'MT_' + op.upper()
        }

        if where is not None:
            mosaic_rule['where'] = where

        if fids is not None:
            mosaic_rule['fids'] = fids

        if muldidef is not None:
            mosaic_rule['multidimensionalDefinition'] = muldidef

        if method in ['none', 'center', 'nadir', 'northwest', 'seamline']:
            mosaic_rule['mosaicMethod'] = 'esriMosaic' + method.title()

        if viewpt is not None:
            if not isinstance(viewpt, Geometry):
                viewpt = Geometry(viewpt)
            mosaic_rule['mosaicMethod'] = 'esriMosaicViewpoint'
            mosaic_rule['viewpoint'] = viewpt

        if sort_by is not None:
            mosaic_rule['mosaicMethod'] = 'esriMosaicAttribute'
            mosaic_rule['sortField'] = sort_by
            if sort_val is not None:
                mosaic_rule['sortValue'] = sort_val

        if lock_rasters is not None:
            mosaic_rule['mosaicMethod'] = 'esriMosaicLockRaster'
            mosaic_rule['lockRasterIds'] = lock_rasters

        if item_rendering_rule is not None:
            mosaic_rule['itemRenderingRule'] = item_rendering_rule

        if self._fnra is not None:
            self._fnra["rasterFunctionArguments"] = _find_and_replace_mosaic_rule(self._fnra["rasterFunctionArguments"],
                                                                                  mosaic_rule, self._url)
        self._mosaic_rule = mosaic_rule
    '''


class _ImageServerRaster(ImageryLayer, Raster):
    def __init__(self, path, is_multidimensional=None, gis=None):
        ImageryLayer.__init__(self, path, gis=gis)
        self._gis = gis
        # self._url = path
        self._is_multidimensional = is_multidimensional
        self._engine = _ImageServerRaster
        self._path = path
        self._do_not_hydrate = False
        self._mdinfo = None
        self._extent = None
        self._extent_set = False
        self._service_url = self._path

    @property
    def extent(self):
        if self._extent is None:
            return super().extent
        else:
            return self._extent

    @extent.setter
    def extent(self, value):
        self._extent_set = True
        self._extent = value

    @property
    def pixel_type(self):
        """returns pixel type of the imagery layer"""
        return super().pixel_type

    @property
    def width(self):
        """returns width of the raster in the units of its spatial reference"""
        return super().width

    @property
    def height(self):
        """returns height of the raster in the units of its spatial reference"""
        return super().height

    @property
    def columns(self):
        """returns number of columns in the raster"""
        return super().columns

    @property
    def rows(self):
        """returns number of rows in the raster"""
        return super().rows

    @property
    def band_count(self):
        """returns the band count of the raster"""
        return super().band_count

    @property
    def catalog_path(self):
        import os

        if self._datastore_raster:
            path = self._uri
            path = (
                (
                    path.rpartition("\\")[0]
                    if "\\" in path and not os.path.exists(path)
                    else path
                )
                if isinstance(path, str)
                else self._service_url
            )
            return path
        return self._url

    @property
    def catalog_paths(self):
        if self._datastore_raster:
            return [self.catalog_path]
        return [self._url]

    @property
    def path(self):
        import os

        path_val = os.path.dirname(self.catalog_path)
        if path_val.startswith("http"):
            return None
        else:
            return path_val

    @property
    def name(self):
        return super().properties.name

    @property
    def names(self):
        return [super().properties.name]

    @property
    def has_RAT(self):
        return super().properties.hasRasterAttributeTable

    @property
    def is_multidimensional(self):
        if "hasMultidimensions" in super().properties:
            return super().properties["hasMultidimensions"]

    @property
    def is_temporary(self):
        if self._fn is not None:
            return True
        else:
            return False

    @property
    def mean_cell_width(self):
        return super().properties.pixelSizeX

    @property
    def mean_cell_height(self):
        return super().properties.pixelSizeY

    @property
    def multidimensional_info(self):
        mdinfo = super().multidimensional_info
        if mdinfo is not None:
            for index, ele in enumerate(mdinfo["multidimensionalInfo"]["variables"]):
                # if (ele['name'] == variable_name):
                for index_dim, ele_dim in enumerate(ele["dimensions"]):
                    if (
                        ele_dim["name"] == "StdTime"
                        or ele_dim["name"].lower() == "time"
                        or ele_dim["name"].lower() == "date"
                        or ele_dim["name"].lower() == "acquisitiondate"
                        or ele_dim["unit"] == "ISO8601"
                    ):
                        if ("recurring" in ele_dim.keys()) and ele_dim["recurring"]:
                            continue
                        val = ele_dim["values"]
                        # if (('unit' in ele_dim.keys()) and ele_dim['unit'] == 'ISO8601'):
                        val_list = []
                        for val_ele in val:
                            if isinstance(val_ele, list):
                                val_list.append(
                                    [
                                        _epoch_to_iso(val_ele[0]),
                                        _epoch_to_iso(val_ele[1]),
                                    ]
                                )
                            else:
                                val_list.append(_epoch_to_iso(val_ele))
                        ele["dimensions"][index_dim]["values"] = val_list
                        if "extent" in ele_dim.keys():
                            ele["dimensions"][index_dim]["extent"] = [
                                _epoch_to_iso(ele_dim["extent"][0]),
                                _epoch_to_iso(ele_dim["extent"][1]),
                            ]

        return mdinfo

    @property
    def minimum(self):
        if super().properties.minValues != []:
            return super().properties.minValues[0]
        else:
            return None

    @property
    def maximum(self):
        if super().properties.maxValues != []:
            return super().properties.maxValues[0]
        else:
            return None

    @property
    def mean(self):
        if super().properties.meanValues != []:
            return super().properties.meanValues[0]
        else:
            return None

    @property
    def standard_deviation(self):
        if super().properties.stdvValues != []:
            return super().properties.stdvValues[0]
        else:
            return None

    @property
    def spatial_reference(self):
        return super().properties.spatialReference

    @property
    def variable_names(self):
        if "hasMultidimensions" in super().properties:
            variable_names_list = []
            for ele in super().multidimensional_info["multidimensionalInfo"][
                "variables"
            ]:
                variable_names_list.append(ele["name"])
            return variable_names_list
        else:
            return None

    @property
    def variables(self):
        if "hasMultidimensions" in super().properties:
            variable_list = []
            for ele in self.multidimensional_info["multidimensionalInfo"]["variables"]:
                if "dimensions" in ele.keys() and "name" in ele.keys():
                    for dim_ele in ele["dimensions"]:
                        variable_list.append(
                            ele["name"]
                            + "("
                            + dim_ele["name"]
                            + "="
                            + str(len(dim_ele["values"]))
                            + ")"
                        )
            return variable_list
        else:
            return None

    @property
    def slices(self):
        # if "hasMultidimensions" in super().properties:
        #    mdim_slices=super().slices()
        #    slice_list = []
        #    for slice in mdim_slices["slices"]:
        #        slice_list.append({"variable":slice["elements"][0]["variableName"]})
        #    i=0
        #    for slice in mdim_slices["slices"]:
        #        for ele in slice["elements"]:
        #            if ele["values"][0][0]==ele["values"][0][1]:
        #                if ele['dimensionName'] == 'StdTime' or ele['dimensionName'].lower() == 'time' or ele['dimensionName'].lower() == 'date' or  ele['dimensionName'].lower() == 'acquisitiondate' or ele['dimensionName'] == 'ISO8601':
        #                    slice_list[i].update({ele["dimensionName"]:_epoch_to_iso(ele["values"][0][0])})
        #                else:
        #                    slice_list[i].update({ele["dimensionName"]:ele["values"][0][0]})
        #            else:
        #                if ele['dimensionName'] == 'StdTime' or ele_dim['name'].lower() == 'time':
        #                    values = ele["values"]
        #                    if isinstance(values[0],list):
        #                        values=values[0]
        #                    for j in range(0,len(values)):
        #                        values[j]=_epoch_to_iso(values[j])
        #                    slice_list[i].update({ele["dimensionName"]:values})
        #                else:
        #                    slice_list[i].update({ele["dimensionName"]:ele["values"]})
        #        i=i+1
        #    return slice_list
        #    #slice_list = []
        #    #for ele in self.multidimensional_info['multidimensionalInfo']['variables']:
        #    #    if "dimensions" in ele.keys() and "name" in ele.keys():
        #    #        for dim_ele in (ele["dimensions"]):
        #    #            for dim_slice in dim_ele["values"]:
        #    #                slice_list.append({"variable": ele['name'], dim_ele["name"]: dim_slice})
        #    #return slice_list
        # else:
        #    return None
        if "hasMultidimensions" in super().properties:
            mdim_slices = super().slices()
            slice_list = []
            for slice in mdim_slices["slices"]:
                slice_list.append(
                    {"variable": slice["multidimensionalDefinition"][0]["variableName"]}
                )
            i = 0
            for slice in mdim_slices["slices"]:
                for ele in slice["multidimensionalDefinition"]:
                    # if ele["values"][0][0]==ele["values"][0][1]:
                    #    if ele['dimensionName'] == 'StdTime' or ele['dimensionName'].lower() == 'time' or ele['dimensionName'].lower() == 'date' or  ele['dimensionName'].lower() == 'acquisitiondate' or ele['dimensionName'] == 'ISO8601':
                    #        slice_list[i].update({ele["dimensionName"]:_epoch_to_iso(ele["values"][0][0])})
                    #    else:
                    #        slice_list[i].update({ele["dimensionName"]:ele["values"][0][0]})
                    # else:
                    if (
                        ele["dimensionName"] == "StdTime"
                        or ele["dimensionName"].lower() == "time"
                        or ele["dimensionName"].lower() == "date"
                        or ele["dimensionName"].lower() == "acquisitiondate"
                    ):
                        # if ele['dimensionName'] == 'StdTime' or ele['dimensionName'].lower() == 'time':
                        values = ele["values"]
                        if isinstance(values[0], list):
                            values = values[0]
                        for j in range(0, len(values)):
                            values[j] = _epoch_to_iso(values[j])
                        if (isinstance(values, list)) and len(values) == 1:
                            slice_list[i].update({ele["dimensionName"]: (values[0],)})
                        else:
                            slice_list[i].update({ele["dimensionName"]: tuple(values)})
                    else:
                        values = ele["values"]
                        if (isinstance(values, list)) and len(values) == 1:
                            slice_list[i].update({ele["dimensionName"]: (values[0],)})
                        else:
                            slice_list[i].update({ele["dimensionName"]: tuple(values)})

                i = i + 1
            return slice_list
            # slice_list = []
            # for ele in self.multidimensional_info['multidimensionalInfo']['variables']:
            #    if "dimensions" in ele.keys() and "name" in ele.keys():
            #        for dim_ele in (ele["dimensions"]):
            #            for dim_slice in dim_ele["values"]:
            #                slice_list.append({"variable": ele['name'], dim_ele["name"]: dim_slice})
            # return slice_list
        else:
            return None

    @property
    def band_names(self):
        if "bandNames" in super().properties:
            return super().properties.bandNames
        else:
            return None

    @property
    def block_size(self):
        block_width = None
        block_height = None
        if "blockWidth" in super().properties:
            block_width = super().properties.blockWidth
        if "blockHeight" in super().properties:
            block_height = super().properties.blockHeight
        return (block_width, block_height)

    @property
    def compression_type(self):
        if "compressionType" in super().properties:
            return super().properties.compressionType
        else:
            return None

    @property
    def is_integer(self):
        if "pixelType" in super().properties:
            pixel_type = super().properties.pixelType
            if (
                pixel_type == "U1"
                or pixel_type == "U2"
                or pixel_type == "U4"
                or pixel_type == "U8"
                or pixel_type == "S8"
                or pixel_type == "U16"
                or pixel_type == "S16"
                or pixel_type == "U32"
                or pixel_type == "S32"
            ):
                return True
            else:
                return False

    @property
    def format(self):
        if "datasetFormat" in super().properties:
            return super().properties.datasetFormat
        else:
            return None

    @property
    def no_data_value(self):
        if "noDataValue" in super().properties:
            return super().properties.noDataValue
        else:
            return None

    @property
    def no_data_values(self):
        if "noDataValues" in super().properties:
            return super().properties.noDataValues
        else:
            return None

    @property
    def uncompressed_size(self):
        if "uncompressedSize" in super().properties:
            return super().properties.uncompressedSize
        else:
            return None

    @property
    def key_properties(self):
        return super().key_properties()

    @property
    def read_only(self):
        return True

    @property
    def RAT(self):
        return super().attribute_table()

    @property
    def raster_info(self):
        """
        Returns information about the ImageryLayer such as
        bandCount, extent , pixelSizeX, pixelSizeY, pixelType
        """
        return super().raster_info

    def get_raster_bands(self, band_ids_or_names=None):
        if band_ids_or_names is None or (
            isinstance(band_ids_or_names, list) and len(band_ids_or_names) == 0
        ):
            band_count = super().band_count
            band_ids_or_names = [i + 1 for i in range(band_count)]

        from arcgis.raster.functions import extract_band

        return_list = []
        if isinstance(band_ids_or_names, list):
            for ele in band_ids_or_names:
                if isinstance(ele, int):
                    return_list.append(extract_band(self, band_ids=[ele]))
                else:
                    return_list.append(extract_band(self, band_names=[ele]))
            return return_list[0] if len(return_list) == 1 else return_list
        else:
            raise RuntimeError("band_ids_or_names should be of type list")

    def get_variable_attributes(self, variable_name):
        attribute_info = {}
        for ele in self.multidimensional_info["multidimensionalInfo"]["variables"]:
            if ele["name"] == variable_name:
                if "description" in ele.keys():
                    attribute_info.update({"Description": ele["description"]})
                if "unit" in ele.keys():
                    attribute_info.update({"Unit": ele["unit"]})
                # attribute_info.update({"Unit": ele['unit'], "Description": ele['description']})
                break
        return attribute_info

    def get_dimension_names(self, variable_name):
        dim_list = []
        for ele in self.multidimensional_info["multidimensionalInfo"]["variables"]:
            if ele["name"] == variable_name:
                for ele_dim in ele["dimensions"]:
                    dim_list.append(ele_dim["name"])
        return dim_list

    def get_dimension_values(
        self, variable_name, dimension_name, return_as_datetime_object=False
    ):
        val = []
        val_list = []
        for index, ele in enumerate(
            self.multidimensional_info["multidimensionalInfo"]["variables"]
        ):
            if ele["name"] == variable_name:
                for ele_dim in ele["dimensions"]:
                    if ele_dim["name"] == dimension_name:
                        val = ele_dim["values"]
                        if isinstance(val, list):
                            for ele in val:
                                if isinstance(ele, list):
                                    val_list.append(tuple(ele))
                                else:
                                    val_list.append(ele)
                        else:
                            val_list = val
                        break
        # unit=''
        # if ('Unit' in self.get_dimension_attributes(variable_name, dimension_name).keys()):
        #    unit = self.get_dimension_attributes(variable_name, dimension_name)["Unit"]
        if (
            dimension_name == "StdTime"
            or dimension_name.lower() == "time"
            or dimension_name.lower() == "date"
            or dimension_name.lower() == "acquisitiondate"
        ) and return_as_datetime_object is True:
            # if (dimension_name == 'StdTime' and return_as_datetime_object is True):
            val_list = []
            for val_ele in val:
                if isinstance(val_ele, list):
                    val_list.append(
                        (
                            _iso_to_datetime(val_ele[0]),
                            _iso_to_datetime(val_ele[1]),
                        )
                    )
                else:
                    val_list.append(_iso_to_datetime(val_ele))
        return val_list

    def get_dimension_attributes(self, variable_name, dimension_name):
        attribute_info = {}
        for index, ele in enumerate(
            self.multidimensional_info["multidimensionalInfo"]["variables"]
        ):
            if ele["name"] == variable_name:
                for ele_dim in ele["dimensions"]:
                    if ele_dim["name"] == dimension_name:
                        if "interval" in ele_dim.keys():
                            attribute_info["Interval"] = ele_dim["interval"]
                        else:
                            attribute_info["Interval"] = ""
                        if "intervalUnit" in ele_dim.keys():
                            attribute_info["IntervalUnit"] = ele_dim["intervalUnit"]
                        else:
                            attribute_info["IntervalUnit"] = ""
                        if "hasRegularIntervals" in ele_dim.keys():
                            attribute_info["HasRegularIntervals"] = ele_dim[
                                "hasRegularIntervals"
                            ]
                        else:
                            attribute_info["HasRegularIntervals"] = ""
                        if "hasRanges" in ele_dim.keys():
                            attribute_info["HasRanges"] = ele_dim["hasRanges"]
                        else:
                            attribute_info["HasRanges"] = ""
                        if "extent" in ele_dim.keys():
                            attribute_info.update(
                                {
                                    "Minimum": ele_dim["extent"][0],
                                    "Maximum": ele_dim["extent"][1],
                                }
                            )
                        else:
                            attribute_info["Minimum"] = ""
                            attribute_info["Maximum"] = ""
                        # attribute_info.update({"Interval": ele_dim["interval"],
                        #                        "IntervalUnit": ele_dim["intervalUnit"],
                        #                        "HasRegularIntervals": ele_dim["hasRegularIntervals"],
                        #                        "hasRanges": ele_dim["hasRanges"], "Minimum": ele_dim["extent"][0],
                        #                        "Maximum": ele_dim["extent"][1]})
                        if "unit" in ele_dim.keys():
                            attribute_info.update({"Unit": ele_dim["unit"]})
                        else:
                            attribute_info["Unit"] = ""
                        if "description" in ele_dim.keys():
                            attribute_info.update(
                                {"Description": ele_dim["description"]}
                            )
                        else:
                            attribute_info["Description"] = ""
        return attribute_info

    def rename_variable(self, current_variable_name, new_variable_name):
        raise RuntimeError("Operation is not supported on image services")

    def set_property(self, property_name, property_value):
        raise RuntimeError("Operation is not supported on image services")

    def get_property(self, property_name):
        key_properties = self.key_properties
        if key_properties is not None:
            if property_name in key_properties.keys():
                return key_properties[property_name]
        else:
            return None

    def read(
        self,
        upper_left_corner: tuple[int] = (0, 0),
        origin_coordinate: Optional[tuple[int]] = None,
        ncols: int = 0,
        nrows: int = 0,
        nodata_to_value: Optional[int] = None,
        cell_size: Optional[tuple[int]] = None,
    ):
        """
        read a numpy array from the calling raster

        :param upper_left_corner: 2-D tuple. a tuple with 2 values representing the number of pixels along x and y
        direction relative to the origin_coordinate. E.g., (2, 0), means that the real origin to extract the array
        is 2 pixels away in x direction from the origin_coordinate

        :param origin_coordinate: arcpy.Point or 2-d tuple (X, Y). The x and y values are in map units.
        If no value is specified, the top left corner of the calling raster, i.e., arcpy.Point(XMin, YMax) will be used

        :param ncols: integer. the number of columns from the real origin in the calling raster to convert to the NumPy array.
        If no value is specified, the number of columns of the calling raster will be used. Default: None

        :param nrows: integer. the number of rows from the real origin in the calling raster to convert to the NumPy array.
        If no value is specified, the number of rows of the calling raster will be used. Default: None

        :param nodata_to_value: numeric. pixels with nodata values in the raster would be assigned with the given value in
        the NumPy array. If no value is specified, the NoData value of the calling raster will be used. Default: None

        :param cell_size: 2-D tuple. a tuple with 2 values shows the x_cell_size and y_cell_size, e.g., cell_size = (2, 2).
        if no value is specified, the original cell size of the calling raster will be used. Otherwise, pixels would be
        resampled to the requested cell_size

        :return: numpy.ndarray. If self is a multidimensional raster, the array has shape (slices, height, width, bands)
        """
        import lerc

        extent = self.extent
        xmin, ymin, xmax, ymax = (
            extent["xmin"],
            extent["ymin"],
            extent["xmax"],
            extent["ymax"],
        )
        if cell_size is None:
            cell_size = (self.mean_cell_width, self.mean_cell_height)
        if origin_coordinate is None:
            origin_coordinate = (xmin, ymax)
        if nrows == 0:
            nrows = self.height
        if ncols == 0:
            ncols = self.width

        xmin = origin_coordinate[0]
        ymin = origin_coordinate[1] - cell_size[1] * nrows
        xmax = origin_coordinate[0] + cell_size[0] * ncols
        ymax = origin_coordinate[1]
        size = (
            round((xmax - xmin) / cell_size[0]),
            round((ymax - ymin) / cell_size[1]),
        )

        if (not self._do_not_hydrate) and self.is_multidimensional:
            multidimensional_info = super().multidimensional_info[
                "multidimensionalInfo"
            ]
            mosaic_rule = {
                "mosaicMethod": "esriMosaicAttribute",
                "ascending": False,
                "multidimensionalDefinition": [],
            }
            for variable in multidimensional_info["variables"]:
                variable_name = variable["name"]
                for dimension in variable["dimensions"]:
                    dimension_name = dimension["name"]
                    extent = dimension["extent"]

                    mosaic_rule["multidimensionalDefinition"].append(
                        {
                            "variableName": variable_name,
                            "dimensionName": dimension_name,
                            "values": [extent],
                        }
                    )
        else:
            mosaic_rule = None

        res = super().export_image(
            bbox=",".join(map(str, [xmin, ymin, xmax, ymax])),
            f="image",
            export_format="lerc",
            size=size,
            no_data=nodata_to_value,
            mosaic_rule=mosaic_rule,
            lerc_version=2,
        )

        if not isinstance(res, bytes):
            raise RuntimeError(res)
        result, data, valid_mask = lerc.decode(res)
        if result != 0:
            raise RuntimeError("decoding bytes from imagery service failed.")

        # transpose
        if (not self._do_not_hydrate) and self.is_multidimensional:
            if len(data) == 2:
                data = np.expand_dims(np.expand_dims(data, axis=2), axis=0)
            elif len(data) == 3:
                if len(self.slices) == 1:
                    data = np.expand_dims(np.transpose(data, [1, 2, 0]), axis=0)
                else:
                    data = np.expand_dims(np.transpose(data, [2, 0, 1]), axis=3)
            else:
                assert len(data.shape) == 4
                data = np.transpose(data, [3, 1, 2, 0])
        else:
            if len(data) == 2:
                data = np.expand_dims(data, axis=2)
            else:
                data = np.transpose(data, axes=[1, 2, 0])

        return data

    def write(
        self,
        array,
        upper_left_corner=(0, 0),
        origin_coordinate=None,
        value_to_nodata=None,
    ):
        raise RuntimeError("Operation is not supported on image services")

    def remove_variables(self, variable_names):
        raise RuntimeError("Operation is not supported on image services")

    def add_dimension(
        self,
        variable,
        new_dimension_name,
        dimension_value,
        dimension_attributes=None,
    ):
        raise RuntimeError("Operation is not supported on image services")

    def append_slices(self, md_raster=None):
        raise RuntimeError("Operation is not supported on image services")

    def set_variable_attributes(self, variable_name, variable_attributes):
        raise RuntimeError("Operation is not supported on image services")

    def get_colormap(self, variable_name=None):
        colormap = super().colormap(variable_name)
        if (isinstance(colormap, dict)) and "colormap" in colormap.keys():
            return {"values": colormap["colormap"]}
        else:
            return colormap

    def summarize(self, geometry=None, pixel_size=None):
        stats_histograms = super().compute_stats_and_histograms(
            geometry=geometry, rendering_rule=self._fn, pixel_size=pixel_size
        )
        return stats_histograms["statistics"]

    def set_colormap(self, color_map, variable_name=None):
        raise RuntimeError("Operation is not supported on image services")

    def get_statistics(self, variable_name=None):
        statistics = super().statistics(variable_name)
        if (isinstance(statistics, dict)) and "statistics" in statistics.keys():
            return statistics["statistics"]
        else:
            return statistics

    def set_statistics(self, statistics_obj, variable_name=None):
        raise RuntimeError("Operation is not supported on image services")

    def get_histograms(self, variable_name=None):
        return super().get_histograms(variable_name)

    def set_histograms(self, histogram_obj, variable_name=None):
        raise RuntimeError("Operation is not supported on image services")

    @property
    def _lyr_json(self):
        _lyr_json = super()._lyr_json
        _lyr_json["type"] = "ImageryLayer"
        return _lyr_json

    def save(
        self,
        output_name: Optional[str] = None,
        for_viz: bool = False,
        process_as_multidimensional: Optional[bool] = None,
        build_transpose=None,
        gis: Optional[GIS] = None,
        future: bool = False,
        **kwargs,
    ):
        """
        Persists this imagery layer to location specified in outpath as an Imagery Layer item.
        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        outpath               Required string.
        -----------------     --------------------------------------------------------------------
        for_viz               The output raster format. The default format will be derived from
                              the file extension that was specified in the outpath.
        =================     ====================================================================
        :return: String representing the location of the output data
        """
        return super().save(
            output_name=output_name,
            for_viz=for_viz,
            process_as_multidimensional=process_as_multidimensional,
            build_transpose=build_transpose,
            gis=gis,
            future=future,
            **kwargs,
        )

    def _repr_png_(self):
        if super().tiles_only:
            fig = super().render_tilesonly_layer()
            try:
                from IPython.core.pylabtools import print_figure

                data = print_figure(fig, "png")
                from matplotlib import pyplot as plt

                plt.close(fig)
                return data
            except:
                pass
        else:
            bbox_sr = None
            if "spatialReference" in self.extent:
                bbox_sr = self.extent["spatialReference"]

            if not self._uses_gbl_function:
                return super().export_image(
                    bbox=self._extent,
                    bbox_sr=bbox_sr,
                    size=[1200, 450],
                    export_format="png32",
                    f="image",
                )

    def _repr_jpeg_(self):
        return None

    def plot_histograms(
        self,
        geometry: Optional[Union[Polygon, Envelope]] = None,
        pixel_size: Optional[Union[str, dict[str, float]]] = None,
        time: Optional[Union[datetime.date, datetime.datetime, str, list[int]]] = None,
        bands: list[int] = [],
        display_stats: bool = True,
        plot_properties: Optional[dict[str, Any]] = None,
        subplot_properties: Optional[Union[list[dict], dict[str, Any]]] = None,
    ):
        """
        Image histograms visually summarize the distribution of a continuous numeric variable by measuring 
        the frequency at which certain values appear in the image. The x-axis in the image histogram is a 
        number line that displays the range of image pixel values that has been split into number ranges, 
        or bins. A bar is drawn for each bin, and the width of the bar represents the density number range 
        of the bin; the height of the bar represents the number of pixels that fall into that range. 
        Understanding the distribution of your data is an important step in the data exploration process.
    
        ``plot_histograms()`` can be used for plotting the band-wise image histogram charts of any Raster object.
    
        ============================    ====================================================================
        **Parameter**                   **Description**
        ----------------------------    --------------------------------------------------------------------
        geometry                        Optional Polygon or Extent. A geometry that defines the geometry
                                        within which the histogram is computed. The geometry can be an
                                        envelope or a polygon. If not provided, then the full extent of the 
                                        raster will be used for the computation.

                                        **Note:** This parameter is honoured if the raster uses "image_server" engine.
        ----------------------------    --------------------------------------------------------------------
        pixel_size                      Optional string or dictionary. The pixel level being used (or the
                                        resolution being looked at). If pixel size is not specified, then
                                        pixel_size will default to the base resolution of the dataset.
                                        The structure of the pixel_size parameter is the same as the
                                        structure of the point object returned by the ArcGIS REST API.
                                        In addition to the dictionary structure, you can specify the pixel size
                                        with a comma-separated string.
                                        
                                        Syntax:
                                        - dictionary structure: pixel_size={point}
                                        - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                        - pixel_size={"x": 0.18, "y": 0.18}
                                        - pixel_size='0.18,0.18'

                                        **Note:** This parameter is honoured if the raster uses "image_server" engine.
        ----------------------------    --------------------------------------------------------------------
        time                            Optional datetime.date, datetime.datetime or timestamp string. The
                                        time instant or the time extent to compute statistics and histograms. .
                                        Time instant specified as datetime.date, datetime.datetime or
                                        timestamp in milliseconds since epoch
                                        Syntax: time=<timeInstant>
                                        
                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp
                                        
                                        Added at 10.8

                                        **Note:** This parameter is honoured if the raster uses "image_server" engine.
        ----------------------------    --------------------------------------------------------------------
        bands                           Optional list of band indices. By default takes the first band (band index - 0).
                                        Image histogram charts are plotted for these specific bands.
    
                                        Example:
                                            - [0,2,3]
        ----------------------------    --------------------------------------------------------------------
        display_stats                   Optional boolean. Specifies whether to plot the band-wise statistics 
                                        along with the histograms.
    
                                        Some basic descriptive statistics are calculated and displayed on 
                                        histograms. The mean and median are displayed with one line each, and 
                                        one standard deviation above and below the mean is displayed using two lines.
    
                                            - False - The statistics will not be displayed along with the histograms.
                                            - True - The statistics will be displayed along with the histograms. \
                                                    This is the default.
        ----------------------------    --------------------------------------------------------------------
        plot_properties                 Optional dictionary. This parameter can be used to set the figure 
                                        properties. These are the `matplotlib.pyplot.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`__ 
                                        parameters and values specified in dict format.
    
                                        Example:
                                            - {"figsize":(15,15)}
        ----------------------------    --------------------------------------------------------------------
        subplot_properties              Optional list or dictionary. This parameter can be used to set band-wise 
                                        histogram (subplot) display properties. These are the `matplotlib.axes.Axes.bar() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib-axes-axes-bar>`__
                                        parameters and values specified in dictionary format.
    
                                        Example:
                                            - | [
                                            |  {"color":"r"},
                                            |  {"color":"g"},
                                            |  {"color":"b","edgecolor":"w"}
                                            | ]
                                            
                                        **Note:** `matplotlib.axes.Axes.bar() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib-axes-axes-bar>`__
                                        parameters: ''x', 'height' or 'align' cannot be passed into subplot_properties.
        ============================    ====================================================================

        .. tip::
        When working with multidimensional rasters, you can use the `multidimensional_filter() <https://developers.arcgis.com/python/api-reference/arcgis.raster.functions.html#multidimensional-filter>`__
        raster function on the Raster object for slicing the data along defined variables and dimensions.
        `plot_histograms()` can then be used on the output raster returned upon applying the filter.
        
        :return: None
    
        """
        return super().plot_histograms(
            geometry=geometry,
            pixel_size=pixel_size,
            time=time,
            bands=bands,
            display_stats=display_stats,
            plot_properties=plot_properties,
            subplot_properties=subplot_properties,
        )

    def export_image(
        self,
        bbox: Optional[Union[dict[str, Any], str]] = None,
        image_sr: Optional[Union[str, SpatialReference]] = None,
        bbox_sr: Optional[Union[str, SpatialReference]] = None,
        size: Optional[list[int]] = None,
        time: Optional[Union[datetime.date, datetime.datetime, str, list[int]]] = None,
        export_format: str = "jpgpng",
        pixel_type: Optional[str] = None,
        no_data: Optional[float] = None,
        no_data_interpretation: str = "esriNoDataMatchAny",
        interpolation: Optional[str] = None,
        compression: Optional[str] = None,
        compression_quality: Optional[int] = None,
        band_ids: Optional[list[int]] = None,
        mosaic_rule: Optional[dict[str, Any]] = None,
        rendering_rule: Optional[dict[str, Any]] = None,
        f: str = "image",
        save_folder: Optional[str] = None,
        save_file: Optional[str] = None,
        compression_tolerance: Optional[float] = None,
        adjust_aspect_ratio: Optional[bool] = None,
        lerc_version: Optional[int] = None,
    ):
        """
        The export_image operation is performed on a raster layer to visualise it.
        ======================  ====================================================================
        **Parameter**           **Description**
        ----------------------  --------------------------------------------------------------------
        bbox                    Optional dict or string. The extent (bounding box) of the exported
                                image. Unless the bbox_sr parameter has been specified, the bbox is
                                assumed to be in the spatial reference of the raster layer.
                                The bbox should be specified as an arcgis.geometry.Envelope object,
                                it's json representation or as a list or string with this
                                format: '<xmin>, <ymin>, <xmax>, <ymax>'
                                If omitted, the extent of the raster layer is used
        ----------------------  --------------------------------------------------------------------
        image_sr                Optional string, SpatialReference. The spatial reference of the
                                exported image. The spatial reference can be specified as either a
                                well-known ID, it's json representation or as an
                                arcgis.geometry.SpatialReference object.
                                If the image_sr is not specified, the image will be exported in the
                                spatial reference of the imagery layer.
        ----------------------  --------------------------------------------------------------------
        size                    Optional list. The size (width * height) of the exported image in
                                pixels. If size is not specified, an image with a default size of
                                400*450 will be exported.
                                Syntax: list of [width, height]
        ----------------------  --------------------------------------------------------------------
        export_format           Optional string. The format of the exported image. The default
                                format is jpgpng. The jpgpng format returns a JPG if there are no
                                transparent pixels in the requested extent; otherwise, it returns a
                                PNG (png32).
                                Values: jpgpng,png,png8,png24,jpg,bmp,gif,tiff,png32,bip,bsq,lerc
        ----------------------  --------------------------------------------------------------------
        mosaic_rule             Optional dict. Specifies the mosaic rule when defining how
                                individual images should be mosaicked. When a mosaic rule is not
                                specified, the default mosaic rule of the image layer will be used
                                (as advertised in the root resource: defaultMosaicMethod,
                                mosaicOperator, sortField, sortValue).
        ----------------------  --------------------------------------------------------------------
        rendering_rule          Optional dict. Specifies the rendering rule for how the requested
                                image should be rendered.
        ----------------------  --------------------------------------------------------------------
        :return: The raw raster data
        """
        result = super().export_image(
            bbox,
            image_sr,
            bbox_sr,
            size,
            time,
            export_format,
            pixel_type,
            no_data,
            no_data_interpretation,
            interpolation,
            compression,
            compression_quality,
            band_ids,
            mosaic_rule,
            rendering_rule,
            f,
            save_folder,
            save_file,
            compression_tolerance,
            adjust_aspect_ratio,
            lerc_version,
        )
        if f == "image" and save_folder is None and save_file is None:
            from IPython.display import Image

            return Image(result)
        else:
            return result

    def draw_graph(
        self, show_attributes: bool = False, graph_size: str = "14.25, 15.25"
    ):
        """
        Displays a structural representation of the function chain and it's raster input values. If
        show_attributes is set to True, then the draw_graph function also displays the attributes
        of all the functions in the function chain, representing the rasters in a blue rectangular
        box, attributes in green rectangular box and the raster function names in yellow.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        show_attributes       Optional boolean. If True, the graph displayed includes all the
                              attributes of the function and not only it's function name and raster
                              inputs
                              Set to False by default, to display only he raster function name and
                              the raster inputs to it.
        -----------------     --------------------------------------------------------------------
        graph_size            Optional string. Maximum width and height of drawing, in inches,
                              seperated by a comma. If only a single number is given, this is used
                              for both the width and the height. If defined and the drawing is
                              larger than the given size, the drawing is uniformly scaled down so
                              that it fits within the given size.
        =================     ====================================================================

        :return: Graph
        """
        return super().draw_graph(show_attributes, graph_size)

    def __sub__(self, other):
        from arcgis.raster.functions import minus

        return minus([self, other])

    def __rsub__(self, other):
        from arcgis.raster.functions import minus

        return minus([other, self])

    def __add__(self, other):
        from arcgis.raster.functions import plus

        return plus([self, other])

    def __radd__(self, other):
        from arcgis.raster.functions import plus

        return plus([other, self])

    def __mul__(self, other):
        from arcgis.raster.functions import times

        return times([self, other])

    def __rmul__(self, other):
        from arcgis.raster.functions import times

        return times([other, self])

    def __div__(self, other):
        from arcgis.raster.functions import divide

        return divide([self, other])

    def __rdiv__(self, other):
        from arcgis.raster.functions import divide

        return divide([other, self])

    def __pow__(self, other):
        from arcgis.raster.functions import power

        return power([self, other])

    def __rpow__(self, other):
        from arcgis.raster.functions import power

        return power([other, self])

    def __abs__(self):
        from arcgis.raster.functions import abs

        return abs([self])

    def __lshift__(self, other):
        from arcgis.raster.functions import bitwise_left_shift

        return bitwise_left_shift([self, other])

    def __rlshift__(self, other):
        from arcgis.raster.functions import bitwise_left_shift

        return bitwise_left_shift([other, self])

    def __rshift__(self, other):
        from arcgis.raster.functions import bitwise_right_shift

        return bitwise_right_shift([self, other])

    def __rrshift__(self, other):
        from arcgis.raster.functions import bitwise_right_shift

        return bitwise_right_shift([other, self])

    def __floordiv__(self, other):
        from arcgis.raster.functions import floor_divide

        return floor_divide([self, other])

    def __rfloordiv__(self, other):
        from arcgis.raster.functions import floor_divide

        return floor_divide([other, self])

    def __truediv__(self, other):
        from arcgis.raster.functions import float_divide

        return float_divide([self, other])

    def __rtruediv__(self, other):
        from arcgis.raster.functions import float_divide

        return float_divide([other, self])

    def __mod__(self, other):
        from arcgis.raster.functions import mod

        return mod([self, other])

    def __rmod__(self, other):
        from arcgis.raster.functions import mod

        return mod([other, self])

    def __neg__(self):
        from arcgis.raster.functions import negate

        return negate([self])

    def __invert__(self):
        from arcgis.raster.functions import boolean_not

        return boolean_not([self])

    def __and__(self, other):
        from arcgis.raster.functions import boolean_and

        return boolean_and([self, other])

    def __rand__(self, other):
        from arcgis.raster.functions import boolean_and

        return boolean_and([other, self])

    def __xor__(self, other):
        from arcgis.raster.functions import boolean_xor

        return boolean_xor([self, other])

    def __rxor__(self, other):
        from arcgis.raster.functions import boolean_xor

        return boolean_xor([other, self])

    def __or__(self, other):
        from arcgis.raster.functions import boolean_or

        return boolean_or([self, other])

    def __ror__(self, other):
        from arcgis.raster.functions import boolean_or

        return boolean_or([other, self])


class _ArcpyRaster(Raster, ImageryLayer):
    def __init__(self, path, is_multidimensional=None, gis=None):
        ImageryLayer.__init__(self, str(path), gis=gis)
        self._gis = gis
        self._is_multidimensional = is_multidimensional
        self._engine = _ArcpyRaster
        import arcpy

        if isinstance(path, RasterInfo):
            ri = arcpy.RasterInfo()
            rinfo = path.to_dict()
            if "geodataXform" not in rinfo.keys():
                if ("extent" in rinfo.keys()) and "spatialReference" in rinfo[
                    "extent"
                ].keys():
                    if ("wkid" in rinfo["extent"]["spatialReference"].keys()) and rinfo[
                        "extent"
                    ]["spatialReference"]["wkid"] is not None:
                        rinfo.update(
                            {
                                "geodataXform": {
                                    "spatialReference": rinfo["extent"][
                                        "spatialReference"
                                    ],
                                    "type": "IdentityXform",
                                }
                            }
                        )
                    else:
                        rinfo.update({"geodataXform": {"type": "IdentityXform"}})
            ri.fromJSONString(json.dumps(rinfo))
            self._raster = arcpy.ia.Raster(ri, is_multidimensional)
            self._uri = str(self._raster)
            self._path = str(self._raster)
        else:
            if isinstance(path, str):
                if (
                    "https://" in path or "http://" in path
                ):  # To provide access to secured service
                    if "ImageServer" in path and self._token is not None:
                        self._raster = arcpy.ia.Raster(
                            path + "?token=" + self._token,
                            is_multidimensional,
                        )
                    else:
                        self._raster = arcpy.ia.Raster(path, is_multidimensional)
                else:
                    self._raster = arcpy.ia.Raster(path, is_multidimensional)
            else:
                self._raster = path

            self._uri = str(path)
            self._path = str(path)
        self._datastore_raster = True
        self._do_not_hydrate = False
        self._extent_set = False
        self._extent = None

    @property
    def _lyr_dict(self):
        url = self._path

        lyr_dict = {"type": type(self).__name__, "url": url}
        if "https://" in url or "http://" in url:
            if "ImageServer" in url and self._token is not None:
                lyr_dict["serviceToken"] = self._token

        return lyr_dict

    def __iter__(self):
        return self._raster.__iter__()

    def __getitem__(self, item):
        return self._raster.__getitem__(item)

    def __setitem__(self, idx, value):
        return self._raster.__setitem__(idx, value)

    @property
    def extent(self):
        if self._extent:
            return self._extent
        else:
            return json.loads(self._raster.extent.JSON)

    @extent.setter
    def extent(self, value):
        self._extent_set = True
        self._extent = value

    @property
    def pixel_type(self):
        """returns pixel type of the imagery layer"""
        pixel_type = self._raster.pixelType
        return pixel_type

    @property
    def width(self):
        """returns width of the raster in the units of its spatial reference"""
        width = self._raster.extent.width
        return width

    @property
    def height(self):
        """returns height of the raster in the units of its spatial reference"""
        return self._raster.extent.height

    @property
    def columns(self):
        """returns number of columns in the raster"""
        return self._raster.width

    @property
    def rows(self):
        """returns number of rows in the raster"""
        return self._raster.height

    @property
    def band_count(self):
        """returns the band count of the raster"""
        return self._raster.bandCount

    @property
    def catalog_path(self):
        return self._raster.catalogPath

    @property
    def catalog_paths(self):
        return self._raster.catalogPaths

    @property
    def path(self):
        return self._raster.path

    @property
    def name(self):
        return self._raster.name

    @property
    def names(self):
        return self._raster.names

    @property
    def has_RAT(self):
        return self._raster.hasRAT

    @property
    def is_multidimensional(self):
        return self._raster.isMultidimensional

    @property
    def is_temporary(self):
        return self._raster.isTemporary

    @property
    def mean_cell_width(self):
        return self._raster.meanCellWidth

    @property
    def mean_cell_height(self):
        return self._raster.meanCellHeight

    @property
    def multidimensional_info(self):
        if self._raster.mdinfo is not None:
            return json.loads(self._raster.mdinfo)
        else:
            return None

    @property
    def minimum(self):
        return self._raster.minimum

    @property
    def maximum(self):
        return self._raster.maximum

    @property
    def mean(self):
        return self._raster.mean

    @property
    def standard_deviation(self):
        return self._raster.standardDeviation

    @property
    def spatial_reference(self):
        return self._raster.spatialReference.exportToString()

    @property
    def variable_names(self):
        return self._raster.variableNames

    @property
    def variables(self):
        return self._raster.variables

    @property
    def slices(self):
        return self._raster.slices

    @property
    def band_names(self):
        return self._raster.bandNames

    @property
    def block_size(self):
        return self._raster.blockSize

    @property
    def compression_type(self):
        return self._raster.compressionType

    @property
    def is_integer(self):
        return self._raster.isInteger

    @property
    def format(self):
        return self._raster.format

    @property
    def no_data_value(self):
        return self._raster.noDataValue

    @property
    def no_data_values(self):
        return self._raster.noDataValues

    @property
    def uncompressed_size(self):
        return self._raster.uncompressedSize

    @property
    def key_properties(self):
        return self._raster.properties

    @property
    def read_only(self):
        return self._raster.readOnly

    @property
    def RAT(self):
        return self._raster.RAT

    @property
    def raster_info(self):
        """
        Returns information about the ImageryLayer such as
        bandCount, extent , pixelSizeX, pixelSizeY, pixelType
        """
        if self._raster_info != {}:
            return self._raster_info
        try:
            ras_info = self._raster.getRasterInfo()
            ras_info_dict = json.loads(ras_info.toJSONString())
            self._raster_info = ras_info_dict
        except:
            # if getRasterInfo fails, get the info from the properties
            if self.extent is not None:
                self._raster_info.update({"extent": dict(self.extent)})

            if self.band_count is not None:
                self._raster_info.update({"bandCount": self.band_count})

            if self.pixel_type is not None:
                self._raster_info.update({"pixelType": self.pixel_type})

            if self.mean_cell_width is not None:
                self._raster_info.update({"pixelSizeX": self.mean_cell_width})

            if self.mean_cell_height is not None:
                self._raster_info.update({"pixelSizeY": self.mean_cell_height})

            if self.compression_type is not None:
                self._raster_info.update({"compressionType": self.compression_type})

            if self.block_size is not None:
                self._raster_info.update({"blockHeight": self.block_size[1]})

            if self.block_size is not None:
                self._raster_info.update({"blockWidth": self.block_size[0]})

            if self.no_data_values is not None:
                self._raster_info.update({"noDataValues": self.no_data_values})

        return self._raster_info

    def get_raster_bands(self, band_ids_or_names=None):
        if band_ids_or_names is None or (
            isinstance(band_ids_or_names, list) and len(band_ids_or_names) == 0
        ):
            band_count = self.band_count
            band_ids_or_names = [i + 1 for i in range(band_count)]
        from arcgis.raster.functions import extract_band

        return_list = []
        if isinstance(band_ids_or_names, list):
            for ele in band_ids_or_names:
                if isinstance(ele, int):
                    return_list.append(extract_band(self, band_ids=[ele]))
                else:
                    return_list.append(extract_band(self, band_names=[ele]))
            return return_list[0] if len(return_list) == 1 else return_list
        else:
            raise RuntimeError("band_ids_or_names should be of type list")

    def get_variable_attributes(self, variable_name):
        return self._raster.getVariableAttributes(variable_name)

    def get_dimension_names(self, variable_name):
        return self._raster.getDimensionNames(variable_name)

    def get_dimension_values(
        self, variable_name, dimension_name, return_as_datetime_object=False
    ):
        val_list = []
        # unit = None

        val = self._raster.getDimensionValues(variable_name, dimension_name)
        # if ("Unit" in self._raster.getDimensionAttributes(variable_name, dimension_name).keys()):
        #    unit = self._raster.getDimensionAttributes(variable_name, dimension_name)["Unit"]
        if (
            dimension_name == "StdTime"
            or dimension_name.lower() == "time"
            or dimension_name.lower() == "date"
            or dimension_name.lower() == "acquisitiondate"
        ) and return_as_datetime_object is True:
            val_list = []
            for val_ele in val:
                if isinstance(val_ele, list) or isinstance(val_ele, tuple):
                    val_list.append(
                        (
                            _iso_to_datetime(val_ele[0]),
                            _iso_to_datetime(val_ele[1]),
                        )
                    )
                else:
                    val_list.append(_iso_to_datetime(val_ele))
            return val_list
        else:
            return val

    def get_dimension_attributes(self, variable_name, dimension_name):
        return self._raster.getDimensionAttributes(variable_name, dimension_name)

    def rename_variable(self, current_variable_name, new_variable_name):
        return self._raster.renameVariable(current_variable_name, new_variable_name)

    def set_property(self, property_name, property_value):
        return self._raster.setProperty(property_name, property_value)

    def get_property(self, property_name):
        return self._raster.getProperty(property_name)

    def read(
        self,
        upper_left_corner: tuple[int] = (0, 0),
        origin_coordinate: Optional[tuple[int]] = None,
        ncols: int = 0,
        nrows: int = 0,
        nodata_to_value: Optional[int] = None,
        cell_size: Optional[tuple[int]] = None,
    ):
        """
        read a numpy array from the calling raster

        :param upper_left_corner: 2-D tuple. a tuple with 2 values representing the number of pixels along x and y
        direction relative to the origin_coordinate. E.g., (2, 0), means that the real origin to extract the array
        is 2 pixels away in x direction from the origin_coordinate

        :param origin_coordinate: arcpy.Point or 2-d tuple (X, Y). The x and y values are in map units.
        If no value is specified, the top left corner of the calling raster, i.e., arcpy.Point(XMin, YMax) will be used

        :param ncols: integer. the number of columns from the real origin in the calling raster to convert to the NumPy array.
        If no value is specified, the number of columns of the calling raster will be used. Default: None

        :param nrows: integer. the number of rows from the real origin in the calling raster to convert to the NumPy array.
        If no value is specified, the number of rows of the calling raster will be used. Default: None

        :param nodata_to_value: numeric. pixels with nodata values in the raster would be assigned with the given value in
        the NumPy array. If no value is specified, the NoData value of the calling raster will be used. Default: None

        :param cell_size: 2-D tuple. a tuple with 2 values shows the x_cell_size and y_cell_size, e.g., cell_size = (2, 2).
        if no value is specified, the original cell size of the calling raster will be used. Otherwise, pixels would be
        resampled to the requested cell_size

        :return: numpy.ndarray. If self is a multidimensional raster, the array has shape (slices, height, width, bands)
        """
        return self._raster.read(
            upper_left_corner=upper_left_corner,
            origin_coordinate=origin_coordinate,
            ncols=ncols,
            nrows=nrows,
            nodata_to_value=nodata_to_value,
            cell_size=cell_size,
        )

    def write(
        self,
        array,
        upper_left_corner=(0, 0),
        origin_coordinate=None,
        value_to_nodata=None,
    ):
        return self._raster.write(
            array=array,
            upper_left_corner=upper_left_corner,
            origin_coordinate=origin_coordinate,
            value_to_nodata=value_to_nodata,
        )

    def remove_variables(self, variable_names):
        return self._raster.removeVariables(variable_names)

    def add_dimension(
        self,
        variable,
        new_dimension_name,
        dimension_value,
        dimension_attributes=None,
    ):
        return self._raster.addDimension(
            variable,
            new_dimension_name,
            dimension_value,
            dimension_attributes,
        )

    @property
    def _lyr_json(self):
        _lyr_json = super()._lyr_json
        _lyr_json["type"] = "ImageryLayer"
        return _lyr_json

    def save(
        self,
        output_name: Optional[str] = None,
        for_viz: bool = False,
        process_as_multidimensional: Optional[bool] = None,
        build_transpose=None,
        gis: Optional[GIS] = None,
        future: bool = False,
        **kwargs,
    ):
        """
        Persists this imagery layer to location specified in outpath as an Imagery Layer item.
        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        outpath               Required string.
        -----------------     --------------------------------------------------------------------
        for_viz               The output raster format. The default format will be derived from
                              the file extension that was specified in the outpath.
        =================     ====================================================================
        :return: String representing the location of the output data
        """

        fnra_set = False
        if self._fnra is None:
            from .functions import identity

            identity_layer = identity(self)
            self._fnra = identity_layer._engine_obj._fnra
            fnra_set = True
        # use arcpy generate raster to save the raster with the function chain

        if process_as_multidimensional is not None:
            if (
                type(process_as_multidimensional) == bool
                and not process_as_multidimensional
            ):
                process_as_multidimensional = "CURRENT_SLICE"
            else:
                process_as_multidimensional = "ALL_SLICES"
        else:
            process_as_multidimensional = "ALL_SLICES"

        if build_transpose is not None:
            if build_transpose and type(build_transpose) == bool:
                build_transpose = "TRANSPOSE"
            else:
                build_transpose = "NO_TRANSPOSE"
        else:
            build_transpose = "NO_TRANSPOSE"

        ext_dict = None
        if self._extent_set and _arcgis.env.analysis_extent is None:
            ext_dict = dict(self.extent)
        else:
            ext_dict = _arcgis.env.analysis_extent

        if ext_dict is not None:
            outext, extsr = _get_extent(ext_dict)
            arcpy.env.extent = outext

        result = arcpy.GenerateRasterFromRasterFunction_management(
            json.dumps(self._fnra),
            output_name,
            process_as_multidimensional=process_as_multidimensional,
        )
        uri = result.getOutput(0)
        if (
            uri
            and process_as_multidimensional == "ALL_SLICES"
            and build_transpose == "TRANSPOSE"
        ):
            arcpy.management.BuildMultidimensionalTranspose(uri)
        if fnra_set == True:
            self._fnra = None
        return uri

    def _repr_png_(self):
        bbox_sr = None
        if "spatialReference" in self.extent:
            bbox_sr = self.extent["spatialReference"]
        if self._fn is None:
            return self.export_image(
                bbox=self.extent,
                bbox_sr=bbox_sr,
                size=[400, 400],
                export_format="png32",
            ).data
        else:
            # rendered_ras= Raster(arcpy.ia.Render(self._raster,{"rft":self._fn}), engine=_ArcpyRaster, is_multidimensional= self._is_multidimensional, gis=self._gis)

            return self.export_image(
                bbox=self.extent,
                bbox_sr=bbox_sr,
                size=[400, 400],
                export_format="png32",
            ).data

    def _repr_jpeg_(self):
        return None

    def plot_histograms(
        self,
        geometry: Optional[Union[Polygon, Envelope]] = None,
        pixel_size: Optional[Union[str, dict[str, float]]] = None,
        time: Optional[Union[datetime.date, datetime.datetime, str, list[int]]] = None,
        bands: list[int] = [],
        display_stats: bool = True,
        plot_properties: Optional[dict[str, Any]] = None,
        subplot_properties: Optional[Union[list[dict], dict[str, Any]]] = None,
    ):
        """
        Image histograms visually summarize the distribution of a continuous numeric variable by measuring 
        the frequency at which certain values appear in the image. The x-axis in the image histogram is a 
        number line that displays the range of image pixel values that has been split into number ranges, 
        or bins. A bar is drawn for each bin, and the width of the bar represents the density number range 
        of the bin; the height of the bar represents the number of pixels that fall into that range. 
        Understanding the distribution of your data is an important step in the data exploration process.
    
        ``plot_histograms()`` can be used for plotting the band-wise image histogram charts of any Raster object.
    
        ============================    ====================================================================
        **Parameter**                   **Description**
        ----------------------------    --------------------------------------------------------------------
        geometry                        optional Polygon or Extent. A geometry that defines the geometry
                                        within which the histogram is computed. The geometry can be an
                                        envelope or a polygon. If not provided, then the full extent of the 
                                        raster will be used for the computation.

                                        **Note:** This parameter is honoured if the raster uses "image_server" engine.
        ----------------------------    --------------------------------------------------------------------
        pixel_size                      optional string or dictionary. The pixel level being used (or the
                                        resolution being looked at). If pixel size is not specified, then
                                        pixel_size will default to the base resolution of the dataset.
                                        The structure of the pixel_size parameter is the same as the
                                        structure of the point object returned by the ArcGIS REST API.
                                        In addition to the dictionary structure, you can specify the pixel size
                                        with a comma-separated string.
                                        
                                        Syntax:
                                        - dictionary structure: pixel_size={point}
                                        - Point simple syntax: pixel_size='<x>,<y>'
                                        Examples:
                                        - pixel_size={"x": 0.18, "y": 0.18}
                                        - pixel_size='0.18,0.18'

                                        **Note:** This parameter is honoured if the raster uses "image_server" engine.
        ----------------------------    --------------------------------------------------------------------
        time                            optional datetime.date, datetime.datetime or timestamp string. The
                                        time instant or the time extent to compute statistics and histograms.
                                        Time instant specified as datetime.date, datetime.datetime or
                                        timestamp in milliseconds since epoch
                                        Syntax: time=<timeInstant>
                                        
                                        Time extent specified as list of [<startTime>, <endTime>]
                                        For time extents one of <startTime> or <endTime> could be None. A
                                        None value specified for start time or end time will represent
                                        infinity for start or end time respectively.
                                        Syntax: time=[<startTime>, <endTime>] ; specified as
                                        datetime.date, datetime.datetime or timestamp
                                        
                                        Added at 10.8

                                        **Note:** This parameter is honoured if the raster uses "image_server" engine.
        ----------------------------    --------------------------------------------------------------------
        bands                           optional list of band indices. By default takes the first band (band index - 0).
                                        Image histogram charts are plotted for these specific bands.
    
                                        Example:
                                            - [0,2,3]
        ----------------------------    --------------------------------------------------------------------
        display_stats                   optional boolean. Specifies whether to plot the band-wise statistics 
                                        along with the histograms.
    
                                        Some basic descriptive statistics are calculated and displayed on 
                                        histograms. The mean and median are displayed with one line each, and 
                                        one standard deviation above and below the mean is displayed using two lines.
    
                                            - False - The statistics will not be displayed along with the histograms.
                                            - True - The statistics will be displayed along with the histograms. \
                                                    This is the default.
        ----------------------------    --------------------------------------------------------------------
        plot_properties                 optional dictionary. This parameter can be used to set the figure 
                                        properties. These are the `matplotlib.pyplot.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`__ 
                                        parameters and values specified in dict format.
    
                                        Example:
                                            - {"figsize":(15,15)}
        ----------------------------    --------------------------------------------------------------------
        subplot_properties              optional list or dictionary. This parameter can be used to set band-wise 
                                        histogram (subplot) display properties. These are the `matplotlib.axes.Axes.bar() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib-axes-axes-bar>`__
                                        parameters and values specified in dictionary format.
    
                                        Example:
                                            - | [
                                            |  {"color":"r"},
                                            |  {"color":"g"},
                                            |  {"color":"b","edgecolor":"w"}
                                            | ]
                                            
                                        **Note:** `matplotlib.axes.Axes.bar() <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib-axes-axes-bar>`__
                                        parameters: ''x', 'height' or 'align' cannot be passed into subplot_properties.
        ============================    ====================================================================

        .. tip::
        When working with multidimensional rasters, you can use the `multidimensional_filter() <https://developers.arcgis.com/python/api-reference/arcgis.raster.functions.html#multidimensional-filter>`__
        raster function on the Raster object for slicing the data along defined variables and dimensions.
        `plot_histograms()` can then be used on the output raster returned upon applying the filter.
        
        :return: None
    
        """
        from arcgis.raster._charts import plot_histograms

        return plot_histograms(
            self,
            geometry=geometry,
            pixel_size=pixel_size,
            time=time,
            bands=bands,
            display_stats=display_stats,
            plot_properties=plot_properties,
            subplot_properties=subplot_properties,
        )

    def export_image(
        self,
        bbox=None,
        image_sr=None,
        bbox_sr=None,
        size=None,
        time=None,
        export_format="jpgpng",
        pixel_type=None,
        no_data=None,
        no_data_interpretation="esriNoDataMatchAny",
        interpolation=None,
        compression=None,
        compression_quality=None,
        band_ids=None,
        mosaic_rule=None,
        rendering_rule=None,
        f="json",
        save_folder=None,
        save_file=None,
        compression_tolerance=None,
        adjust_aspect_ratio=None,
        lerc_version=None,
    ):
        """
        The export_image operation is performed on a raster layer to visualise it.
        ======================  ====================================================================
        **Parameter**           **Description**
        ----------------------  --------------------------------------------------------------------
        bbox                    Optional dict or string. The extent (bounding box) of the exported
                                image. Unless the bbox_sr parameter has been specified, the bbox is
                                assumed to be in the spatial reference of the raster layer.
                                The bbox should be specified as an arcgis.geometry.Envelope object,
                                it's json representation or as a list or string with this
                                format: '<xmin>, <ymin>, <xmax>, <ymax>'
                                If omitted, the extent of the raster layer is used
        ----------------------  --------------------------------------------------------------------
        image_sr                optional string, SpatialReference. The spatial reference of the
                                exported image. The spatial reference can be specified as either a
                                well-known ID, it's json representation or as an
                                arcgis.geometry.SpatialReference object.
                                If the image_sr is not specified, the image will be exported in the
                                spatial reference of the raster.
        ----------------------  --------------------------------------------------------------------
        bbox_sr                 optional string, SpatialReference. The spatial reference of the
                                bbox.
                                The spatial reference can be specified as either a well-known ID,
                                it's json representation or as an arcgis.geometry.SpatialReference
                                object.
                                If the image_sr is not specified, bbox is assumed to be in the
                                spatial reference of the raster.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        size                    optional list. The size (width * height) of the exported image in
                                pixels. If size is not specified, an image with a default size of
                                400*450 will be exported.
                                Syntax: list of [width, height]
        ----------------------  --------------------------------------------------------------------
        time                    optional datetime.date, datetime.datetime or timestamp string. The
                                time instant or the time extent of the exported image.
                                Time instant specified as datetime.date, datetime.datetime or
                                timestamp in milliseconds since epoch
                                Syntax: time=<timeInstant>
                                Time extent specified as list of [<startTime>, <endTime>]
                                For time extents one of <startTime> or <endTime> could be None. A
                                None value specified for start time or end time will represent
                                infinity for start or end time respectively.
                                Syntax: time=[<startTime>, <endTime>] ; specified as
                                datetime.date, datetime.datetime or timestamp
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        export_format           optional string. The format of the exported image. The default
                                format is jpgpng. The jpgpng format returns a JPG if there are no
                                transparent pixels in the requested extent; otherwise, it returns a
                                PNG (png32).
                                Values: jpgpng,png,png8,png24,jpg,bmp,gif,tiff,png32,bip,bsq,lerc
        ----------------------  --------------------------------------------------------------------
        pixel_type              optional string. The pixel type, also known as data type, pertains
                                to the type of values stored in the raster, such as signed integer,
                                unsigned integer, or floating point. Integers are whole numbers,
                                whereas floating points have decimals.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        no_data                 optional float. The pixel value representing no information.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        no_data_interpretation  optional string. Interpretation of the no_data setting. The default
                                is NoDataMatchAny when no_data is a number, and NoDataMatchAll when
                                no_data is a comma-delimited string: NoDataMatchAny,NoDataMatchAll.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        interpolation           optional string. The resampling process of extrapolating the pixel
                                values while transforming the raster dataset when it undergoes
                                warping or when it changes coordinate space.
                                One of: RSP_BilinearInterpolation, RSP_CubicConvolution,
                                RSP_Majority, RSP_NearestNeighbor
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        compression             optional string. Controls how to compress the image when exporting
                                to TIFF format: None, JPEG, LZ77. It does not control compression on
                                other formats.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        compression_quality     optional integer. Controls how much loss the image will be subjected
                                to by the compression algorithm. Valid value ranges of compression
                                quality are from 0 to 100.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        band_ids                optional list. If there are multiple bands, you can specify a single
                                band to export, or you can change the band combination (red, green,
                                blue) by specifying the band number. Band number is 0 based.
                                Specified as list of ints, eg [2,1,0]
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        mosaic_rule             optional dict. Specifies the mosaic rule when defining how
                                individual images should be mosaicked. When a mosaic rule is not
                                specified, the default mosaic rule of the image layer will be used
                                (as advertised in the root resource: defaultMosaicMethod,
                                mosaicOperator, sortField, sortValue).
        ----------------------  --------------------------------------------------------------------
        rendering_rule          optional dict. Specifies the rendering rule for how the requested
                                image should be rendered.
        ----------------------  --------------------------------------------------------------------
        f                       optional string. The response format.  default is json
                                Values: json,image,kmz
                                If image format is chosen, the bytes of the exported image are
                                returned unless save_folder and save_file parameters are also
                                passed, in which case the image is written to the specified file
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        save_folder             optional string. The folder in which the exported image is saved
                                when f=image
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        save_file               optional string. The file in which the exported image is saved when
                                f=image
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        compression_tolerance   optional float. Controls the tolerance of the lerc compression
                                algorithm. The tolerance defines the maximum possible error of pixel
                                values in the compressed image.
                                Example: compression_tolerance=0.5 is loseless for 8 and 16 bit
                                images, but has an accuracy of +-0.5 for floating point data. The
                                compression tolerance works for the LERC format only.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        adjust_aspect_ratio     optional boolean. Indicates whether to adjust the aspect ratio or
                                not. By default adjust_aspect_ratio is true, that means the actual
                                bbox will be adjusted to match the width/height ratio of size
                                paramter, and the response image has square pixels.
                                (Available only when image_server engine is used)
        ----------------------  --------------------------------------------------------------------
        lerc_version            optional integer. The version of the Lerc format if the user sets
                                the format as lerc.
                                Values: 1 or 2
                                If a version is specified, the server returns the matching version,
                                or otherwise the highest version available.
                                (Available only when image_server engine is used)
        ======================  ====================================================================

        :return: The raw raster data
        """

        (width, height) = (0, 0) if size is None else (size[0], size[1])

        # convert extent and spatial reference
        extent, spatial_reference = None, None
        if isinstance(bbox_sr, dict):
            if "wkid" in bbox_sr.keys():
                if bbox_sr["wkid"] is None:
                    bbox_sr = None
        if bbox_sr is not None and not isinstance(
            bbox_sr, _arcgis.geometry.SpatialReference
        ):
            bbox_sr = _arcgis.geometry.SpatialReference(bbox_sr)
            bbox_sr = bbox_sr.as_arcpy

        if bbox is not None:
            if isinstance(bbox, dict):
                bbox = _arcgis.geometry.Envelope(bbox)
            if isinstance(bbox, _arcgis.geometry.Envelope):
                coordinates = bbox.coordinates()
                if bbox_sr is not None:
                    extent = arcpy.Extent(
                        coordinates[0],
                        coordinates[1],
                        coordinates[2],
                        coordinates[3],
                        spatial_reference=bbox_sr,
                    )
                else:
                    extent = arcpy.Extent(
                        coordinates[0],
                        coordinates[1],
                        coordinates[2],
                        coordinates[3],
                    )
            elif isinstance(bbox, list):
                extent = arcpy.Extent(
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    spatial_reference=bbox_sr,
                )
            elif isinstance(bbox, str):
                xmin, ymin, xmax, ymax = tuple(map(float, bbox.split(",")))
                extent = arcpy.Extent(xmin, ymin, xmax, ymax, spatial_reference=bbox_sr)
            else:
                raise TypeError("invalid bbox type")

        if image_sr is not None and not isinstance(
            image_sr, _arcgis.geometry.SpatialReference
        ):
            spatial_reference = _arcgis.geometry.SpatialReference(image_sr).as_arcpy

        return self._raster.exportImage(
            width,
            height,
            format=export_format,
            extent=extent,
            spatial_reference=spatial_reference,
            mosaic_rule=mosaic_rule,
        )

    def append_slices(self, md_raster=None):
        return self._raster.appendSlices(md_raster._engine_obj._raster)

    def set_variable_attributes(self, variable_name, variable_attributes):
        return self._raster.setVariableAttributes(variable_name, variable_attributes)

    def get_colormap(self, variable_name=None):
        if variable_name is None:
            variable_name = ""
        cmap = self._raster.getColormap(variable_name)
        if isinstance(cmap, dict):
            if ("type" in cmap.keys()) and cmap["type"] == "RasterColormap":
                del cmap["type"]
        return cmap

    def set_colormap(self, color_map, variable_name=None):
        if variable_name is None:
            variable_name = ""
        return self._raster.setColormap(color_map, variable_name)

    def get_statistics(self, variable_name=None):
        if variable_name is None:
            variable_name = ""
        return self._raster.getStatistics(variable_name)

    def set_statistics(self, statistics_obj, variable_name=None):
        if variable_name is None:
            variable_name = ""
        return self._raster.setStatistics(statistics_obj, variable_name)

    def get_histograms(self, variable_name=None):
        if variable_name is None:
            variable_name = ""
        return self._raster.getHistograms(variable_name)

    def set_histograms(self, histogram_obj, variable_name=None):
        if variable_name is None:
            variable_name = ""
        return self._raster.setHistograms(histogram_obj, variable_name)

    def summarize(self, geometry, pixel_size=None):
        raise RuntimeError("Operation is not supported on local rasters")

    def draw_graph(
        self, show_attributes: bool = False, graph_size: str = "14.25, 15.25"
    ):
        """
        Displays a structural representation of the function chain and it's raster input values. If
        show_attributes is set to True, then the draw_graph function also displays the attributes
        of all the functions in the function chain, representing the rasters in a blue rectangular
        box, attributes in green rectangular box and the raster function names in yellow.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        show_attributes       optional boolean. If True, the graph displayed includes all the
                              attributes of the function and not only it's function name and raster
                              inputs
                              Set to False by default, to display only he raster function name and
                              the raster inputs to it.
        -----------------     --------------------------------------------------------------------
        graph_size            optional string. Maximum width and height of drawing, in inches,
                              seperated by a comma. If only a single number is given, this is used
                              for both the width and the height. If defined and the drawing is
                              larger than the given size, the drawing is uniformly scaled down so
                              that it fits within the given size.
        =================     ====================================================================

        :return: Graph
        """
        return ImageryLayer.draw_graph(self, show_attributes, graph_size)

    def __sub__(self, other):
        from arcgis.raster.functions import minus

        return minus([self, other])

    def __rsub__(self, other):
        from arcgis.raster.functions import minus

        return minus([other, self])

    def __add__(self, other):
        from arcgis.raster.functions import plus

        return plus([self, other])

    def __radd__(self, other):
        from arcgis.raster.functions import plus

        return plus([other, self])

    def __mul__(self, other):
        from arcgis.raster.functions import times

        return times([self, other])

    def __rmul__(self, other):
        from arcgis.raster.functions import times

        return times([other, self])

    def __div__(self, other):
        from arcgis.raster.functions import divide

        return divide([self, other])

    def __rdiv__(self, other):
        from arcgis.raster.functions import divide

        return divide([other, self])

    def __pow__(self, other):
        from arcgis.raster.functions import power

        return power([self, other])

    def __rpow__(self, other):
        from arcgis.raster.functions import power

        return power([other, self])

    def __abs__(self):
        from arcgis.raster.functions import abs

        return abs([self])

    def __lshift__(self, other):
        from arcgis.raster.functions import bitwise_left_shift

        return bitwise_left_shift([self, other])

    def __rlshift__(self, other):
        from arcgis.raster.functions import bitwise_left_shift

        return bitwise_left_shift([other, self])

    def __rshift__(self, other):
        from arcgis.raster.functions import bitwise_right_shift

        return bitwise_right_shift([self, other])

    def __rrshift__(self, other):
        from arcgis.raster.functions import bitwise_right_shift

        return bitwise_right_shift([other, self])

    def __floordiv__(self, other):
        from arcgis.raster.functions import floor_divide

        return floor_divide([self, other])

    def __rfloordiv__(self, other):
        from arcgis.raster.functions import floor_divide

        return floor_divide([other, self])

    def __truediv__(self, other):
        from arcgis.raster.functions import float_divide

        return float_divide([self, other])

    def __rtruediv__(self, other):
        from arcgis.raster.functions import float_divide

        return float_divide([other, self])

    def __mod__(self, other):
        from arcgis.raster.functions import mod

        return mod([self, other])

    def __rmod__(self, other):
        from arcgis.raster.functions import mod

        return mod([other, self])

    def __neg__(self):
        from arcgis.raster.functions import negate

        return negate([self])

    def __invert__(self):
        from arcgis.raster.functions import boolean_not

        return boolean_not([self])

    def __and__(self, other):
        from arcgis.raster.functions import boolean_and

        return boolean_and([self, other])

    def __rand__(self, other):
        from arcgis.raster.functions import boolean_and

        return boolean_and([other, self])

    def __xor__(self, other):
        from arcgis.raster.functions import boolean_xor

        return boolean_xor([self, other])

    def __rxor__(self, other):
        from arcgis.raster.functions import boolean_xor

        return boolean_xor([other, self])

    def __or__(self, other):
        from arcgis.raster.functions import boolean_or

        return boolean_or([self, other])

    def __ror__(self, other):
        from arcgis.raster.functions import boolean_or

        return boolean_or([other, self])

    def __ne__(self, other):
        if isinstance(other, (ImageryLayer, Raster, numbers.Number)):
            from arcgis.raster.functions import not_equal

            return not_equal([self, other])
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (ImageryLayer, Raster, numbers.Number)):
            from arcgis.raster.functions import equal_to

            return equal_to([self, other])
        else:
            return NotImplemented

    def __gt__(self, other):
        from arcgis.raster.functions import greater_than

        return greater_than([self, other])

    def __ge__(self, other):
        from arcgis.raster.functions import greater_than_equal

        return greater_than_equal([self, other])

    def __lt__(self, other):
        from arcgis.raster.functions import less_than

        return less_than([self, other])

    def __le__(self, other):
        from arcgis.raster.functions import less_than_equal

        return less_than_equal([self, other])


def _get_raster_collection_engine(engine):
    """
    Function to get the engine that will be used to process the Raster object.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    engine                                   Required string.
                                                Possible options:
                                                "arcpy" : Returns arcpy engine
                                                "image_server" : Returns image server engine
    ------------------------------------     --------------------------------------------------------------------
    """

    engine_dict = {
        "arcpy": _ArcpyRasterCollection,
        "image_server": _ImageServerRasterCollection,
        "datastore": _LocalRasterCollection,
    }
    if isinstance(engine, str):
        return engine_dict[engine]
    return engine


from ._util import _local_function_template, _get_geometry, _get_extent

# class RasterCollectionEngineFactory():
#    def get_engine(self, engine):
#        engine_dict={
#            "arcpy":_ArcpyRasterCollection,
#            "image_server":_ImageServerRasterCollection,
#        "local_datastore_raster": _LocalRasterCollection}
#        return engine_dict[engine]


class RasterCollection:
    """
    The ``RasterCollection`` object allows a group of rasters to be sorted and
    filtered easily, and prepares a collection for additional processing and analysis.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    rasters                                  The input raster datasets. Supported inputs include a list of
                                             local or datastore rasters, a mosaic dataset, a multidimensional
                                             raster in Cloud Raster Format, a NetCDF file, or an image service.
                                             If you're using a list of raster datasets, all rasters must have
                                             the same cell size and spatial reference.

                                             arcpy should be available if the input is a local raster dataset.
    ------------------------------------     --------------------------------------------------------------------
    attribute_dict                           Optional dict. attribute information to be added to each raster,
                                             when the input is a list of rasters. For each key-value pair,
                                             the key is the attribute name and the value is a list of values
                                             that represent the attribute value for each raster. For example,
                                             to add a name field to a list of three rasters, use
                                             {"name": ["Landsat8_Jan", "Landsat8_Feb", "Landsat8_Mar"]}.
    ------------------------------------     --------------------------------------------------------------------
    where_clause                             Optional string. An expression that limits the records returned.
    ------------------------------------     --------------------------------------------------------------------
    query_geometry                           Optional. An object that filters the items such that
                                             only those that intersect with the object will be returned.
    ------------------------------------     --------------------------------------------------------------------
    engine                                   Optional string. The backend engine to be used.
                                             Possible options:
                                                - ``arcpy`` : Use the arcpy engine for processing.

                                                - ``image_server`` : Use the Image Server engine for processing.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` of the RasterCollection object.
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional. Additional properties to control the creation of RasterCollection.
                                             The context parameter would be honoured by all other collections created from this
                                             i.e., the map/filter outputs. The filter/map methods also support the context parameter
                                             which can be configured separately for each method.

                                             Currently available:

                                                 -  query_boundary:
                                                    The boolean value set to this option determines whether to add SHAPE field to the RasterCollection.
                                                    The value in the SHAPE field represents the boundary/geometry of the raster.
                                                    The query_boundary parameter is honoured only when the RasterCollection
                                                    is created from a list of Rasters.

                                                    - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                    - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                    By default, query_boundary is set to True, i.e, SHAPE field will be added.

                                                    Example:
                                                    {"query_boundary":True}
    ====================================     ====================================================================

    .. code-block:: python

        # Usage Example 1: Creates a raster collection from image service url
        service_url = gis.content.search('my_rasters', item_type="Imagery Layer")[0].url

        rc = RasterCollection(rasters=service_url, gis=gis)

    .. code-block:: python

        # Usage Example 2: Creates a raster collection from rasters stored locally

        ras1 = Raster(r"./data/ras1.tif")
        ras2 = Raster(r"./data/ras2.tif")
        ras3 = Raster(r"./data/ras3.tif")

        ras_list = [ras1, ras2, ras3]

        # Add attributes to the raster collection

        acquisition_date = ["2016-01-01T00:00:00", "2016-02-01T00:00:00", "2016-03-01T00:00:00"]
        name_list = ["Landsat8_Jan", "Landsat8_Feb", "Landsat8_Mar"]

        rc = RasterCollection(rasters=ras_list,
                              attribute_dict={"name": name_list,
                                              "AcquisitionDate": acquisition_date}
                             )

    """

    def __init__(
        self,
        rasters=None,
        attribute_dict: Optional[dict] = None,
        where_clause: Optional[str] = None,
        query_geometry: Optional[dict] = None,
        engine: Optional[str] = None,
        gis: Optional[GIS] = None,
        context: Optional[dict] = None,
    ):
        # self._remote = raster.use_server_engine
        # super().__init__(rasters, gis)

        # self._do_not_hydrate=False
        import pandas as pd

        local_class = True

        if engine is not None and engine not in [
            "arcpy",
            "image_server",
            _ArcpyRasterCollection,
            _ImageServerRasterCollection,
            _LocalRasterCollection,
        ]:
            raise RuntimeError(
                'Invalid engine: parameter engine should be either "arcpy" or "image_server"'
            )
        engine = _get_raster_collection_engine(engine)

        if context is None:
            self._context = {}
        else:
            self._context = context

        if engine is not None:
            self._ras_coll_engine = engine
            self._ras_coll_engine_obj = engine(
                rasters=rasters,
                attribute_dict=attribute_dict,
                where_clause=where_clause,
                query_geometry=query_geometry,
                engine=engine,
                gis=gis,
                context=context,
            )
        else:
            if isinstance(rasters, pd.DataFrame):
                rc_dict = rasters.to_dict("list")
                rasters = rc_dict.pop("Raster")
                attribute_dict = rc_dict

            if isinstance(rasters, str):
                if "https://" in rasters or "http://" in rasters:
                    self._ras_coll_engine = _ImageServerRasterCollection
                    self._ras_coll_engine_obj = _ImageServerRasterCollection(
                        rasters=rasters,
                        attribute_dict=attribute_dict,
                        where_clause=where_clause,
                        query_geometry=query_geometry,
                        engine=_ImageServerRasterCollection,
                        gis=gis,
                        context=context,
                    )
                else:
                    self._ras_coll_engine = _ArcpyRasterCollection
                    self._ras_coll_engine_obj = _ArcpyRasterCollection(
                        rasters=rasters,
                        attribute_dict=attribute_dict,
                        where_clause=where_clause,
                        query_geometry=query_geometry,
                        engine=_ArcpyRasterCollection,
                        gis=gis,
                        context=context,
                    )

            elif isinstance(rasters, list):
                for ele in rasters:
                    if isinstance(ele, Raster):
                        if isinstance(ele._engine_obj, _ArcpyRaster):
                            local_class = False
                        else:
                            continue
                    elif isinstance(ele, str):
                        if (
                            "/fileShares/" in ele
                            or "/rasterStores/" in ele
                            or "/cloudStores/" in ele
                            or "/vsi" in ele
                        ):
                            continue
                        else:
                            local_class = False
                    else:
                        local_class = False

                if local_class == True:
                    self._ras_coll_engine = _LocalRasterCollection
                    self._ras_coll_engine_obj = _LocalRasterCollection(
                        rasters=rasters,
                        attribute_dict=attribute_dict,
                        where_clause=where_clause,
                        query_geometry=query_geometry,
                        engine=_LocalRasterCollection,
                        gis=gis,
                        context=context,
                    )
                else:
                    self._ras_coll_engine = _ArcpyRasterCollection
                    self._ras_coll_engine_obj = _ArcpyRasterCollection(
                        rasters=rasters,
                        attribute_dict=attribute_dict,
                        where_clause=where_clause,
                        query_geometry=query_geometry,
                        engine=_ArcpyRasterCollection,
                        gis=gis,
                        context=context,
                    )

    def set_engine(self, engine: str):
        """
        The ``set_engine`` method can be used to change the back end engine of the
        :class:`~arcgis.raster.RasterCollection`.
        """
        return RasterCollection(
            rasters=self._ras_coll_engine_obj._rasters,
            attribute_dict=self._ras_coll_engine_obj._attribute_dict,
            where_clause=self._ras_coll_engine_obj._where_clause,
            query_geometry=self._ras_coll_engine_obj._spatial_filter,
            engine=engine,
            gis=self._ras_coll_engine_obj._gis,
            context=self._context,
        )

    @property
    def count(self):
        """
        The ``count`` property returns the count of items in the ``RasterCollection``.

        :return:
            An integer
        """
        return self._ras_coll_engine_obj.count

    @property
    def fields(self):
        """
        The ``fields`` property returns the fields available in the ``RasterCollection``.

        :return:
            A list of available fields
        """
        return self._ras_coll_engine_obj.fields

    @property
    def _rasters_list(self):
        return self._ras_coll_engine_obj._rasters_list

    def __iter__(self):
        return self._ras_coll_engine_obj.__iter__()

    def __len__(self):
        return self._ras_coll_engine_obj.__len__()

    def __getitem__(self, item):
        return self._ras_coll_engine_obj.__getitem__(item)

    @staticmethod
    def from_stac_api(
        stac_api: str,
        query: Optional[dict[str, Any]] = None,
        attribute_dict: Optional[dict[str, Any]] = None,
        request_method: str = "POST",
        request_params: Optional[dict[str, Any]] = None,
        engine: str = None,
        *,
        gis: Optional[GIS] = None,
    ):
        """
        The ``from_stac_api`` method creates  a :class:`~arcgis.raster.RasterCollection` object from a
        `SpatioTemporal Asset Catalog (STAC) API <https://github.com/radiantearth/stac-api-spec>`__
        `search <https://github.com/radiantearth/stac-api-spec/tree/master/item-search>`__ query.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        stac_api              Required string. URL of the STAC API root endpoint. The STAC API where
                              the search needs to be performed.

                              .. note::

                                The following STAC APIs are supported:

                                    - https://planetarycomputer.microsoft.com/api/stac/v1 (Following collections are supported: \
                                        3dep-seamless, 3dep-lidar-dsm, sentinel-1-rtc, hgb, cop-dem-glo-30, cop-dem-glo-90, gnatsgo-rasters, \
                                        3dep-lidar-hag, 3dep-lidar-intensity, 3dep-lidar-pointsourceid, mtbs, noaa-c-cap, alos-fnf-mosaic, 3dep-lidar-returns, \
                                        chloris-biomass, 3dep-lidar-dtm-native, 3dep-lidar-classification, 3dep-lidar-dtm, gap, alos-dem, jrc-gsw, \
                                        hrea, sentinel-2-l2a, nrcan-landcover, ecmwf-forecast, noaa-mrms-qpe-24h-pass2, sentinel-1-grd, nasadem, \
                                        io-lulc, landsat-c2-l1, drcog-lulc, chesapeake-lc-7, chesapeake-lc-13, chesapeake-lu, noaa-mrms-qpe-1h-pass1, \
                                        mobi, landsat-c2-l2, noaa-mrms-qpe-1h-pass2, noaa-nclimgrid-monthly, usda-cdl, esa-cci-lc, esa-cci-lc-netcdf, \
                                        noaa-climate-normals-netcdf, noaa-climate-normals-gridded, io-lulc-9-class, io-biodiversity, naip, \
                                        noaa-cdr-sea-surface-temperature-whoi, noaa-cdr-ocean-heat-content, noaa-cdr-sea-surface-temperature-whoi-netcdf, \
                                        sentinel-3-olci-wfr-l2-netcdf, noaa-cdr-ocean-heat-content-netcdf, sentinel-3-synergy-v10-l2-netcdf, \
                                        sentinel-3-olci-lfr-l2-netcdf, sentinel-3-slstr-lst-l2-netcdf, sentinel-3-slstr-wst-l2-netcdf, \
                                        sentinel-3-synergy-syn-l2-netcdf, sentinel-3-synergy-vgp-l2-netcdf, sentinel-3-synergy-vg1-l2-netcdf, esa-worldcover)
                                    - https://earth-search.aws.element84.com/v0 (All collections are suported)
                                    - https://earth-search.aws.element84.com/v1 (All collections are suported)
                                    - https://services.sentinel-hub.com/api/v1/catalog (All collections are suported)


                              Example:
                                    "https://planetarycomputer.microsoft.com/api/stac/v1"
        -----------------     --------------------------------------------------------------------
        query                 Optional dictionary. The GET/POST request query dictionary that can be
                              used to query a STAC API's search endpoint. (keys/values would depend
                              on the specification of the STAC API in use and the request_method
                              parameter value).

                              For the `bbox` query parameter, :class:`~arcgis.geometry.Envelope`
                              and :class:`~arcgis.geometry.Polygon`
                              objects are also accepted (in any spatial reference).

                              .. note:: 

                                ``limit`` key of the query should be explicitly set as ``None`` when \
                                trying to create a RasterCollection from all the matched items \
                                (retrieving them from all the pages). By default, the RasterCollection \
                                is created from the first page of matches.

                              Example:
                                    | {
                                    |   "collections": ["sentinel-2-l2a"],
                                    |   "bbox": [-110, 39.5, -105, 40.5],
                                    |   "query": {"eo:cloud_cover": {"lt": 0.5}},
                                    |   "datetime": "2020-10-05T00:00:00Z/2020-10-10T12:31:12Z",
                                    |   "limit": 100
                                    | }
        -----------------     --------------------------------------------------------------------
        attribute_dict        Optional dictionary. The attribute information to be added to each
                              (STAC Item) raster returned from the query. For each key-value pair, the key is
                              the attribute name, and the value is a list of values that represent
                              the attribute value for each raster.

                              Attribute values can also be collected from the STAC Items automatically
                              using the STAC Item metadata information. It can be done by specifying
                              the STAC Item property name for the Attribute of interest in this format:
                                    - key : value -> Attribute display name : STAC item property name

                              Example:
                                    | {
                                    |   "Name":"id",
                                    |   "Sensor":"platform",
                                    |   "StdTime":"datetime",
                                    |   "Cloud Cover":"eo:cloud_cover",
                                    |   "Spatial Reference":"proj:epsg",
                                    |   "Extent":"bbox"
                                    | }

                              .. note::
                                If 'Geometry' is not specified in the ``attribute_dict`` then it would
                                be automatically added for each Raster in the ``RasterCollection`` based
                                on its STAC Item 'geometry' property and would be in
                                Spatial reference: ``{'wkid':4326}``.
        -----------------     --------------------------------------------------------------------
        request_method        Optional string. The HTTP request method used with the STAC API for making the search.

                              Acceptable methods:

                                    - GET
                                    - POST (This is the default)

                              Example:
                                    "POST"
        -----------------     --------------------------------------------------------------------
        request_params        This parameter can be used to set the properties for making the STAC
                              API search request. These are the `requests.post() <https://requests.readthedocs.io/en/master/api/#requests.post>`__
                              or `requests.get() <https://requests.readthedocs.io/en/master/api/#requests.get>`__
                              method parameters and values will be specified in dictionary format.

                              Example:
                                    | {
                                    |   "verify":True,
                                    |   "headers":{"Authorization": "Bearer access_token_string"}
                                    | }
        -----------------     --------------------------------------------------------------------
        engine                Optional string. The backend engine to be used for Raster processing.

                              Possible options:
                                - "arcpy" : Use the arcpy engine for processing. 

                                - "image_server" : Use the Image Server engine for processing (This is the default).
                              
                              Example:
                                    "image_server"

                              .. note:: 
                              
                                When using ``image_server`` engine, RasterRendering service should be enabled \
                                in the active GIS connection.
        -----------------     --------------------------------------------------------------------
        gis                   Optional :class:`~arcgis.gis.GIS` object. The GIS of the RasterCollection object.
        =================     ====================================================================

        :return: A :class:`~arcgis.raster.RasterCollection` object

        .. code-block:: python

            # Usage Example: Creating a RasterCollection object from making a query to a STAC API.

            rc = RasterCollection.from_stac_api(stac_api=stac_api_url,
                                                query={
                                                        "collections": ["sentinel-2-l2a"],
                                                        "bbox": [-110, 39.5, -105, 40.5],
                                                        "query": {"eo:cloud_cover": {"lt": 0.5}},
                                                        "datetime": "2020-10-05T00:00:00Z/2020-10-10T12:31:12Z",
                                                        "limit": 100
                                                      },
                                                attribute_dict={
                                                                "Name":"id",
                                                                "Sensor":"platform",
                                                                "StdTime":"datetime",
                                                                "Cloud Cover":"eo:cloud_cover",
                                                                "Spatial Reference":"proj:epsg",
                                                                "Extent":"bbox"
                                                               },
                                                request_method="POST",
                                                gis=gis)

        """

        from ._util import _get_stac_metadata_file, _get_stac_api_search_items

        if not isinstance(stac_api, str):
            raise RuntimeError(f"Invalid STAC API URL-\n{stac_api}")
        api_search_endpoint = (
            stac_api + "search" if stac_api.endswith("/") else stac_api + "/search"
        )

        if request_params is None:
            request_params = {}
        if not isinstance(request_params, dict):
            raise RuntimeError("request_params should be of type dictionary")
        if any(key in request_params for key in ["url", "params", "data", "json"]):
            raise RuntimeError(
                "request_params cannot contain these keys : url, params, data or json"
            )

        if not isinstance(request_method, str) or request_method.upper() not in [
            "GET",
            "POST",
        ]:
            raise RuntimeError(
                "request_method can only be one of the following: GET or POST"
            )

        new_query = None
        if query is not None:
            if not isinstance(query, dict):
                raise RuntimeError("parameter query should be of type dictionary")
            new_query = copy.deepcopy(query)
            if "bbox" in new_query:
                query_extent = new_query["bbox"]
                from arcgis.geometry import Envelope, Polygon, project

                if isinstance(query_extent, (Envelope, Polygon)):
                    envelope_dict = json.loads(query_extent.envelope.JSON)
                    projected_envelope = None
                    if envelope_dict["spatialReference"] is None:
                        raise RuntimeError(
                            "Invalid bbox: Polygon/Envelope object should contain spatialReference"
                        )
                    try:
                        projected_envelope = project(
                            geometries=[envelope_dict],
                            in_sr=envelope_dict["spatialReference"],
                            out_sr=4326,
                        )
                    except Exception:
                        raise RuntimeError(
                            "Unsupported bbox: project operation failed for the given Polygon/Envelope object"
                        )
                    bbox_list = [
                        projected_envelope[0]["xmin"],
                        projected_envelope[0]["ymin"],
                        projected_envelope[0]["xmax"],
                        projected_envelope[0]["ymax"],
                    ]

                    if request_method.upper() == "GET":
                        bbox_str = ",".join(str(e) for e in bbox_list)
                        new_query["bbox"] = bbox_str
                    else:
                        new_query["bbox"] = bbox_list

        max_limit_map = {
            "planetarycomputer.microsoft.com/api/stac": 1000,
            "earth-search.aws.element84.com": 200,
            "services.sentinel-hub.com/api": 100,
        }

        stacs = list(max_limit_map.keys())

        search_stac = next(
            (stac for stac in stacs if stac in api_search_endpoint), None
        )

        if search_stac is None:
            raise RuntimeError("STAC API not supported")

        get_all_items = False

        if (
            new_query is not None
            and "limit" in new_query
            and new_query["limit"] is None
        ):
            new_query["limit"] = max_limit_map[search_stac]
            get_all_items = True

        items = _get_stac_api_search_items(
            api_search_endpoint,
            new_query,
            request_method,
            request_params,
            get_all_items,
        )

        if len(items) < 1:
            raise RuntimeError(f"No STAC items found. Please specify a better query")

        rc_attribute_dict = {}
        attribute_dict = {} if attribute_dict is None else attribute_dict
        for key in attribute_dict:
            if isinstance(attribute_dict[key], list):
                rc_attribute_dict[key] = attribute_dict[key]
            else:
                rc_attribute_dict[key] = [
                    item[attribute_dict[key]]
                    if attribute_dict[key] in item
                    else item["properties"][attribute_dict[key]]
                    if attribute_dict[key] in item["properties"]
                    else key
                    for item in items
                ]

        from arcgis.raster.functions import composite_band

        raster_list = []
        for item in items:
            metadata_file = _get_stac_metadata_file(item)
            if not metadata_file:
                raise RuntimeError(f"STAC Item not supported-\n{item}")
            ras = (
                composite_band(
                    rasters=[
                        Raster(file, engine=engine, gis=gis) for file in metadata_file
                    ]
                )
                if isinstance(metadata_file, list)
                else Raster(metadata_file, engine=engine, gis=gis)
            )
            raster_list.append(ras)

        if "Geometry" not in rc_attribute_dict:
            geometry_list = []
            for item in items:
                coordinates = item["geometry"]["coordinates"]
                polygon_geometry = Geometry(
                    {
                        "rings": coordinates,
                        "spatialReference": {"wkid": 4326},
                    }
                )
                geometry_list.append(polygon_geometry)
                rc_attribute_dict["Geometry"] = geometry_list

        rc = RasterCollection(
            raster_list,
            rc_attribute_dict,
            context={"query_boundary": False},
            gis=gis,
        )
        return rc

    @staticmethod
    def from_stac_catalog(
        stac_catalog,
        attribute_dict: Optional[dict[str, Any]] = None,
        request_params: Optional[dict[str, Any]] = None,
        engine: Optional[str] = None,
        *,
        gis: Optional[GIS] = None,
    ):
        """
        The ``from_stac_catalog`` method creates a :class:`~arcgis.raster.RasterCollection` object from a
        `Static SpatioTemporal Asset Catalog (STAC) <https://github.com/radiantearth/stac-spec/blob/master/catalog-spec/catalog-spec.md>`__.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        stac_catalog          Required string or `pystac.Catalog <https://pystac.readthedocs.io/en/latest/api.html#catalog>`__ object. If string, then it should
                              be the URL of the Static STAC (Catalog).

                              .. note::
                                Currently only Landsat-8 STAC (Catalogs) are supported for this method.

                              Example:
                                    "https://landsat-stac.s3.amazonaws.com/landsat-8-l1/010/117/catalog.json"
        -----------------     --------------------------------------------------------------------
        attribute_dict        Optional dictionary. The attribute information to be added to each
                              (STAC Item) raster of the catalog. For each key-value pair, the key is
                              the attribute name, and the value is a list of values that represent
                              the attribute value for each raster.

                              Attribute values can also be collected from the STAC Items automatically
                              using the STAC Item metadata information. It can be done by specifying
                              the STAC Item property name for the Attribute of interest in this format:
                                    - key : value -> Attribute display name : STAC item property name

                              Example:
                                    | {
                                    |   "Name":"id",
                                    |   "Sensor":"platform",
                                    |   "StdTime":"datetime",
                                    |   "Cloud Cover":"eo:cloud_cover",
                                    |   "Extent":"bbox"
                                    | }

                              .. note::
                                If 'Geometry' is not specified in the ``attribute_dict`` then it would
                                be automatically added for each Raster in the ``RasterCollection`` based
                                on its STAC Item 'geometry' property and would be in
                                Spatial reference: ``{'wkid':4326}``.
        -----------------     --------------------------------------------------------------------
        request_params        Optional dictionary. This parameter can be used to set the properties
                              for making the STAC Item/Catalog requests. These are the `requests.get() method <https://requests.readthedocs.io/en/master/api/#requests.get>`__
                              method parameters and values will be specified in dictionary format.

                              | This parameter is honoured when the stac_catalog parameter is set to a string (URL).

                              Example:
                                    {"verify":False}
        -----------------     --------------------------------------------------------------------
        engine                Optional string. The backend engine to be used for Raster processing.

                              Possible options:
                                - "arcpy" : Use the arcpy engine for processing. 

                                - "image_server" : Use the Image Server engine for processing (This is the default).
                              
                              Example:
                                    "image_server"

                              .. note:: 
                                    When using ``image_server`` engine, RasterRendering service should be enabled \
                                    in the active GIS connection.
        -----------------     --------------------------------------------------------------------
        gis                   Optional arcgis.gis.GIS object. The GIS of the RasterCollection object.
        =================     ====================================================================

        :return: A :class:`~arcgis.raster.RasterCollection` object

        .. code-block:: python

            # Usage Example: Creating a RasterCollection object from a Static STAC.

            rc = RasterCollection.from_stac_catalog(stac_catalog=stac_catalog_url,
                                                    attribute_dict={
                                                                    "Name":"id",
                                                                    "Sensor":"collection",
                                                                    "StdTime":"datetime",
                                                                    "Cloud Cover":"eo:cloud_cover",
                                                                    "Extent":"bbox"
                                                                   },
                                                    gis=gis)

        """
        is_pystac_cat = False
        if isinstance(stac_catalog, str):
            if request_params is None:
                request_params = {}
            if not isinstance(request_params, dict):
                raise RuntimeError("request_params should be of type dictionary")
            if any(key in request_params for key in ["url", "params", "data", "json"]):
                raise RuntimeError(
                    "request_params cannot contain these keys : url, params, data or json"
                )

            data = _requests.get(stac_catalog, **request_params)
            if data.status_code != 200 or data.headers.get("content-type") not in [
                "application/json",
                "application/geo+json",
                "application/json;charset=utf-8",
                "application/geo+json; charset=utf-8",
            ]:
                raise RuntimeError(
                    f"Invalid Response: Please verify that the stac_catalog URL is correct-\n{data.text}"
                )

            json_data = data.json()

            from ._util import _get_stac_links, _get_all_stac_catalog_items

            if not _get_stac_links(json_data, "item") and not _get_stac_links(
                json_data, "child"
            ):
                raise RuntimeError(f"Invalid STAC catalog-\n{stac_catalog}")

            items = _get_all_stac_catalog_items(json_data, request_params)
        else:
            try:
                import pystac

                items = stac_catalog.get_all_items()
                is_pystac_cat = True
            except ImportError:
                raise ImportError(
                    "pystac not found, parameter stac_catalog accepts either a Static STAC URL or a pystac.Catalog object"
                )
            except Exception:
                raise RuntimeError(f"Invalid/Unsupported STAC Catalog-\n{stac_catalog}")

        rc_attribute_dict = {}
        attribute_dict = {} if attribute_dict is None else attribute_dict
        for key in attribute_dict:
            if isinstance(attribute_dict[key], list):
                rc_attribute_dict[key] = attribute_dict[key]
            else:
                rc_attribute_dict[key] = []
        if "Geometry" not in rc_attribute_dict:
            rc_attribute_dict["Geometry"] = []

        from ._util import _get_stac_metadata_file

        raster_list = []
        for item in items:
            if is_pystac_cat:
                item_dict = item.to_dict()
            else:
                item_dict = item

            for key in attribute_dict:
                if isinstance(attribute_dict[key], list):
                    pass
                else:
                    if attribute_dict[key] in item_dict:
                        rc_attribute_dict[key].append(item_dict[attribute_dict[key]])
                    elif attribute_dict[key] in item_dict["properties"]:
                        rc_attribute_dict[key].append(
                            item_dict["properties"][attribute_dict[key]]
                        )
                    else:
                        rc_attribute_dict[key].append(key)

            metadata_file = _get_stac_metadata_file(item_dict)
            if not metadata_file:
                raise RuntimeError(f"STAC Item not supported-\n{item_dict}")

            ras = Raster(metadata_file, engine=engine, gis=gis)
            raster_list.append(ras)

            if "Geometry" not in attribute_dict:
                coordinates = item_dict["geometry"]["coordinates"]
                polygon_geometry = Geometry(
                    {
                        "rings": coordinates,
                        "spatialReference": {"wkid": 4326},
                    }
                )
                rc_attribute_dict["Geometry"].append(polygon_geometry)

        rc = RasterCollection(
            raster_list,
            rc_attribute_dict,
            context={"query_boundary": False},
            gis=gis,
        )
        return rc

    def filter_by(
        self,
        where_clause: Optional[str] = None,
        query_geometry_or_extent: Optional[Geometry] = None,
        raster_query: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ):
        """
        The ``filter_by`` method filters a :class:`~arcgis.raster.RasterCollection` based on attribute and/or spatial queries.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        where_clause                             Optional String. An SQL expression used to select a subset of rasters
        ------------------------------------     --------------------------------------------------------------------
        query_geometry_or_extent                 Optional Geometry object. Items in the collection that fails to
                                                 intersect the given geometry will be excluded
        ------------------------------------     --------------------------------------------------------------------
        raster_query                             Optional string. An SQL expression used to select a subset of rasters
                                                 by the raster's key properties.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.RasterCollection` object that only contains items satisfying the queries

        .. code-block:: python

            # Usage Example: Creates a Raster collection and filters rasters satisfying the raster query.

            service_url = gis.content.search('my_image_service')[0].url
            rc = RasterCollecton(service_url, gis=gis)

            filtered_rc = rc.filter_by(raster_query="raster_query")

        """
        return self._ras_coll_engine_obj.filter_by(
            where_clause=where_clause,
            query_geometry_or_extent=query_geometry_or_extent,
            raster_query=raster_query,
            context=context,
        )

    def filter_by_time(
        self,
        start_time: str = "",
        end_time: str = "",
        time_field_name: str = "StdTime",
        date_time_format: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ):
        """
        The ``filter_by_time`` method filters a :class:`~arcgis.raster.RasterCollection` by time.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        start_time                               Optional String representation of the start time.
        ------------------------------------     --------------------------------------------------------------------
        end_time                                 Optional String representation of the end time.
        ------------------------------------     --------------------------------------------------------------------
        time_field_name                          Optional string. the name of the field containing the time information
                                                 for each item. Default: "StdTime"
        ------------------------------------     --------------------------------------------------------------------
        date_time_format                         Optional string. the time format that is used to format the time field values.
                                                 Please ref the python date time standard for this argument (`See this <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`__).

                                                 Default is None and this means using the Pro standard time format
                                                 '%Y-%m-%dT%H:%M:%S' and ignoring the following sub-second.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return: a :class:`~arcgis.raster.RasterCollection` object that only contains items satisfying the filter

        .. code-block:: python

            # Usage Example: Filters the raster collection based on time parameters.

            filtered_rc_time = rc.filter_by_time(start_time="1990-01-01 00:00:00",
                                                 end_time="1999-12-31 00:00:00",
                                                 time_field_name="AcquisitionDate")

        """
        return self._ras_coll_engine_obj.filter_by_time(
            start_time=start_time,
            end_time=end_time,
            time_field_name=time_field_name,
            date_time_format=date_time_format,
            context=context,
        )

    def filter_by_calendar_range(
        self,
        calendar_field: str,
        start: int,
        end: Optional[int] = None,
        time_field_name: str = "StdTime",
        date_time_format: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ):
        """
        The ``filter_by_calendar_range`` method filters the :class:`~arcgis.raster.RasterCollection` by a
        ``calendar_field`` and its start and end value (inclusive).

        For example, if you would like to select all the rasters that have the time stamp on
        Monday, specify ``calendar_field`` as 'DAY_OF_WEEK' and put start and end to 1.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        calendar_field                           Required String, one of 'YEAR', 'MONTH', 'QUARTER', 'WEEK_OF_YEAR',
                                                 'DAY_OF_YEAR', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'HOUR'
        ------------------------------------     --------------------------------------------------------------------
        start                                    Required integer.
                                                 The start value of the calendar_field. For example, to filter all
                                                 items that were collected in January,

                                                 filtered_rc = rc.filter_by_calendar_range(calendar_field="MONTH", start=1).
        ------------------------------------     --------------------------------------------------------------------
        end                                      Optional integer.

                                                 The end value of the calendar_field. For example, to filter all items
                                                 that were collected in the first 5 days of each year,

                                                 filtered_rc = rc.filter_by_calendar_range(calendar_field="DAY_OF_YEAR",
                                                 start=1, end=5)
        ------------------------------------     --------------------------------------------------------------------
        time_field_name                          Optional string. The name of the field that contains the time attribute
                                                 for each item in the collection. The default is StdTime.
        ------------------------------------     --------------------------------------------------------------------
        date_time_format                         Optional string. The time format of the values in the time field.
                                                 For example, if the input time value is "1990-01-31", the
                                                 date_time_format is "%Y-%m-%d".
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return: a :class:`~arcgis.raster.RasterCollection` object that only contains items satisfying the filter

        .. code-block:: python

            # Usage Example 1: Filters the raster collection to hold rasters from the month of January.

            filtered_rc_month = rc2.filter_by_calendar_range(calendar_field="MONTH", start=1)

            # Usage Example 2: Filter the raster collection over the years 2015-2020.

            filtered_rc_years = rc2.filter_by_calendar_range(calendar_field="YEAR",
                                                             start=2015,
                                                             end=2020)

        """

        return self._ras_coll_engine_obj.filter_by_calendar_range(
            calendar_field=calendar_field,
            start=start,
            end=end,
            time_field_name=time_field_name,
            date_time_format=date_time_format,
            context=context,
        )

    def filter_by_geometry(
        self,
        query_geometry_or_extent: Union[Geometry, Raster, ImageryLayer],
        context: Optional[dict[str, Any]] = None,
    ):
        """
        The ``filter_by_geometry`` method filters the :class:`~arcgis.raster.RasterCollection` such that only those rasters that
        intersect with the geometry will be returned.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        query_geometry_or_extent                 Required object that filters the items such that only those that
                                                 intersect with the object will be returned. This can be specified
                                                 with a Geometry object, Raster object, ImageryLayer object.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return: a :class:`~arcgis.raster.RasterCollection` object that only contains items satisfying the filter

        .. code-block:: python

            # Usage Example: Filters the raster collection based on specified geometry.

            aoi = {
                    "spatialReference": {"wkid": 32610},
                    "xmax": 725000,
                    "xmin": 720000,
                    "ymax": 4300000,
                    "ymin": 4250000
                  }

            aoi_geometry = Geometry(aoi)

            filtered_rc_geom = rc.filter_by_geometry(query_geometry_or_extent=aoi_geometry)

        """

        return self._ras_coll_engine_obj.filter_by_geometry(
            query_geometry_or_extent=query_geometry_or_extent,
            context=context,
        )

    def filter_by_attribute(
        self,
        field_name: str,
        operator: str,
        field_values: Union[str, list[str], list[int], int],
        context: Optional[dict[str, Any]] = None,
    ):
        """
        The ``filter_by_attribute`` method filters the collection of raster items by an attribute query and returns a
        raster collection containing only the items that satisfy the query.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        field_name                               Required string. The field name to use in the filter.
        ------------------------------------     --------------------------------------------------------------------
        operator                                 Required string. The keyword to filter the attributes.
                                                 Keywords include the following:

                                                  - ``CONTAINS`` - The attribute in the field contains the specified string, list, or number.

                                                  - ``ENDS_WITH`` - The attribute ends with the specified string or number.

                                                  - ``EQUALS`` - The attribute equals the specified string, list, or number.

                                                  - ``GREATER_THAN`` - The attribute is greater than the specified number.

                                                  - ``IN`` - The attribute is one of the items in the specified list.

                                                  - ``LESS_THAN`` - The attribute is less than the specified number.

                                                  - ``NOT_CONTAINS`` - The attribute does not contain the specified string, list, or number.

                                                  - ``NOT_ENDS_WITH`` - The attribute does not end with the specified string or number.

                                                  - ``NOT_EQUALS`` - The attribute does not equal the specified string, list, or number.

                                                  - ``NOT_GREATER_THAN`` - The attribute is not greater than the specified number.

                                                  - ``NOT_IN`` - The attribute is not one of the items in the specified list.

                                                  - ``NOT_LESS_THAN`` - The attribute is not less than the specified number.

                                                  - ``NOT_STARTS_WITH`` - The attribute does not start with the specified string or number.

                                                  - ``STARTS_WITH`` - The attribute starts with the specified string or number.
        ------------------------------------     --------------------------------------------------------------------
        field_values                             Required object. The attribute value or values against which to compare.
                                                 This can be specified as a string, a list, or a number.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return: a :class:`~arcgis.raster.RasterCollection` object that only contains items satisfying the filter

        .. code-block:: python

            # Usage Example 1: Filters the raster collection based on matching field name.

            filtered_rc_attribute = rc.filter_by_attribute(field_name="Name",
                                                           operator="EQUALS",
                                                           field_values="field_values")

            # Usage Example 2: Filters the raster collection based on unmatching field name.

            filtered_rc_attribute2 = rc.filter_by_attribute(field_name="Name",
                                                            operator="NOT_EQUALS",
                                                            field_values="field_values")

        """
        return self._ras_coll_engine_obj.filter_by_attribute(
            field_name=field_name,
            operator=operator,
            field_values=field_values,
            context=context,
        )

    def filter_by_raster_property(
        self,
        property_name: str,
        operator: str,
        property_values: Union[str, list[str, list[int], int]],
        context: Optional[dict[str, Any]] = None,
    ):
        """
        The ``filter_by_raster_property`` method filters the :class:`~arcgis.raster.RasterCollection` by a raster
        property query and returns a :class:`~arcgis.raster.RasterCollection` containing only the
        items that satisfy the query.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        property_name                            Required string. The name of the property to use in the filter.
        ------------------------------------     --------------------------------------------------------------------
        operator                                 Required string. The keyword to filter the attributes.
                                                 Keywords include the following:

                                                  -  ``CONTAINS`` - The attribute in the field contains the specified string, list, or number.

                                                  - ``ENDS_WITH`` - The attribute ends with the specified string or number.

                                                  -  ``EQUALS`` - The attribute equals the specified string, list, or number.

                                                  -  ``GREATER_THAN`` - The attribute is greater than the specified number.

                                                  -  ``IN`` - The attribute is one of the items in the specified list.

                                                  -  ``LESS_THAN`` - The attribute is less than the specified number.

                                                  -  ``NOT_CONTAINS`` - The attribute does not contain the specified string, list, or number.

                                                  -  ``NOT_ENDS_WITH`` - The attribute does not end with the specified string or number.

                                                  -  ``NOT_EQUALS`` - The attribute does not equal the specified string, list, or number.

                                                  -  ``NOT_GREATER_THAN`` - The attribute is not greater than the specified number.

                                                  -  ``NOT_IN`` - The attribute is not one of the items in the specified list.

                                                  -  ``NOT_LESS_THAN`` - The attribute is not less than the specified number.

                                                  -  ``NOT_STARTS_WITH`` - The attribute does not start with the specified string or number.

                                                  -  ``STARTS_WITH`` - The attribute starts with the specified string or number.
        ------------------------------------     --------------------------------------------------------------------
        property_values                          Required object. The property value or values against which to compare.
                                                 This can be specified as a string, a list, or a number.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.RasterCollection` object that only contains items satisfying the filter

        .. code-block:: python

            # Usage Example: Filters out rasters with a band count of 3 from the raster collection

            filtered_rc = rc.filter_by_raster_property(property_name="BAND_COUNT",
                                                       operator="EQUALS",
                                                       property_values=3)

        """
        return self._ras_coll_engine_obj.filter_by_raster_property(
            property_name=property_name,
            operator=operator,
            property_values=property_values,
            context=context,
        )

    def sort(
        self,
        field_name: str,
        ascending: bool = True,
        context: Optional[dict[str, Any]] = None,
    ):
        """
        The ``sort`` method sorts the :class:`~arcgis.raster.RasterCollection` by a field name and returns
        a :class:`~arcgis.raster.RasterCollection` that is in the order specified.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        field_name                               Required string. The name of the field to use for sorting.
        ------------------------------------     --------------------------------------------------------------------
        ascending                                Optional bool. Specifies whether to sort in ascending or descending order.
                                                 (The default value is True)
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return: A sorted :class:`~arcgis.raster.RasterCollection` object

        """
        return self._ras_coll_engine_obj.sort(
            field_name=field_name, ascending=ascending, context=context
        )

    def get_field_values(self, field_name: str, max_count: int = 0):
        """
        The ``get_field_values`` method retrieves the values of a specified field from the
        :class:`~arcgis.raster.RasterCollection`.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        field_name                               Required string. The name of the field from which to extract values.
        ------------------------------------     --------------------------------------------------------------------
        max_count                                Optional integer. An integer that specifies the maximum number of
                                                 field values to be returned. The values will be returned in the
                                                 order that the raster items are ordered in the collection. If
                                                 no value is specified, all the field values for the given field
                                                 will be returned.
        ====================================     ====================================================================

        :return: A List of values of the specified field from the :class:`~arcgis.raster.RasterCollection`.

        """
        return self._ras_coll_engine_obj.get_field_values(
            field_name=field_name, max_count=max_count
        )

    def to_multidimensional_raster(
        self, variable_field_name: str, dimension_field_names: str
    ):
        """
        The ``to_multidimensional_raster`` method returns a multidimensional raster dataset, in which each item in the
        :class:`~arcgis.raster.RasterCollection` is a slice in the multidimensional raster.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        variable_field_name                      Required string. The name of the field that contains the variable names.
        ------------------------------------     --------------------------------------------------------------------
        dimension_field_names                    Required string. The name of the field or fields that contains the dimension names.
                                                 This can be specified as a single string or a list of strings.

                                                 For time-related dimensions, the field name must match one of the
                                                 following to be recognized as a time field: StdTime, Date, Time,
                                                 or AcquisitionDate. For nontime-related dimensions, the values in
                                                 those fields must be type Double. If there are two or more dimensions,
                                                 use a comma to separate the fields (for example,
                                                 dimension_field_names = ["Time", "Depth"]).
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object

        .. code-block:: python

            # Usage Example: Generates a multidimensional raster from the raster collection.

            multidim_raster = rc.to_multidimensional_raster(variable_field_name="Name",
                                                            dimension_field_name="AcquisitionDate")

        """
        return self._ras_coll_engine_obj.to_multidimensional_raster(
            variable_field_name=variable_field_name,
            dimension_field_names=dimension_field_names,
        )

    def max(
        self,
        ignore_nodata: bool = True,
        extent_type: str = "FirstOf",
        cellsize_type: str = "FirstOf",
    ):
        """
        The ``max`` method returns a :class:`~arcgis.raster.Raster` object in which each band contains the
        maximum pixel values for that band across all rasters in the raster collection.

        For example, if there are ten raster items in the :class:`~arcgis.raster.RasterCollection`, each with
        four bands, the max method will calculate the maximum pixel value that occurs across all raster
        items for band 1, band 2, band 3, and band 4; a four-band raster is returned.
        Band numbers are matched between raster items using the band index, so the items
        in the raster collection must follow the same band order.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored.

                                                    - True : The method will include all valid pixels and ignore any NoData pixels. This is the default.
                                                    - False : The method will result in NoData if there are any NoData values.
        ------------------------------------     --------------------------------------------------------------------
        extent_type                              Optional string. Specifies the extent to be used for the function.

                                                    - "FirstOf" - Use the extent of the first input raster to determine the processing extent. This is the default.

                                                    - "IntersectionOf" - Use the extent of the overlapping pixels to determine the processing extent.

                                                    - "UnionOf" - Use the extent of all the rasters to determine the processing extent.

                                                    - "LastOf" - Use the extent of the last input raster to determine the processing extent.
        ------------------------------------     --------------------------------------------------------------------
        cellsize_type                            Optional string. Specifies the cell size to be used for the function.

                                                    - "FirstOf" - Use the first cell size of the input rasters. This is the default.

                                                    - "MinOf" - Use the smallest cell size of all the input rasters.

                                                    - "MaxOf" - Use the largest cell size of all the input rasters.

                                                    - "MeanOf" - Use the mean cell size of all the input rasters.

                                                    - "LastOf" - Use the last cell size of the input rasters.
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object

        """
        return self._ras_coll_engine_obj.max(
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def min(
        self,
        ignore_nodata: bool = True,
        extent_type: str = "FirstOf",
        cellsize_type: str = "FirstOf",
    ):
        """
        The ``min`` method returns a :class:`~arcgis.raster.Raster` object in which each band contains the
        minimum pixel values for that band across all rasters in the :class:`~arcgis.raster.RasterCollection`.

        For example, if there are ten raster items in the raster collection, each with four bands,
        the min method will calculate the minimum pixel value that occurs across all raster
        items for band 1, band 2, band 3, and band 4; a four-band raster is returned.
        Band numbers are matched between raster items using the band index, so the items
        in the raster collection must follow the same band order.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored.

                                                    - True : The method will include all valid pixels and ignore any NoData pixels. This is the default.
                                                    - False : The method will result in NoData if there are any NoData values.
        ------------------------------------     --------------------------------------------------------------------
        extent_type                              Optional string. Specifies the extent to be used for the function.

                                                    - "FirstOf" - Use the extent of the first input raster to determine the processing extent. This is the default.

                                                    - "IntersectionOf" - Use the extent of the overlapping pixels to determine the processing extent.

                                                    - "UnionOf" - Use the extent of all the rasters to determine the processing extent.

                                                    - "LastOf" - Use the extent of the last input raster to determine the processing extent.
        ------------------------------------     --------------------------------------------------------------------
        cellsize_type                            Optional string. Specifies the cell size to be used for the function.

                                                    - "FirstOf" - Use the first cell size of the input rasters. This is the default.

                                                    - "MinOf" - Use the smallest cell size of all the input rasters.

                                                    - "MaxOf" - Use the largest cell size of all the input rasters.

                                                    - "MeanOf" - Use the mean cell size of all the input rasters.

                                                    - "LastOf" - Use the last cell size of the input rasters.
        ====================================     ====================================================================

        :return: a :class:`~arcgis.raster.Raster` object
        """
        return self._ras_coll_engine_obj.min(
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def median(
        self,
        ignore_nodata: bool = True,
        extent_type: str = "FirstOf",
        cellsize_type: str = "FirstOf",
    ):
        """
        The ``median`` method returns a :class:`~arcgis.raster.Raster` object in which each band contains the
        median pixel values for that band across all rasters in the :class:`~arcgis.raster.RasterCollection`.

        For example, if there are ten raster items in the raster collection,
        each with four bands, the median method will calculate the median pixel value
        that occurs across all raster items for band 1, for band 2, for band 3,
        and for band 4; a four-band raster is returned. Band numbers are matched
        between raster items using the band index, so the items in the raster
        collection must follow the same band order.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored.

                                                    - True : The method will include all valid pixels and ignore any NoData pixels. This is the default.
                                                    - False : The method will result in NoData if there are any NoData values.
        ------------------------------------     --------------------------------------------------------------------
        extent_type                              Optional string. Specifies the extent to be used for the function.

                                                    - "FirstOf" - Use the extent of the first input raster to determine the processing extent. This is the default.

                                                    - "IntersectionOf" - Use the extent of the overlapping pixels to determine the processing extent.

                                                    - "UnionOf" - Use the extent of all the rasters to determine the processing extent.

                                                    - "LastOf" - Use the extent of the last input raster to determine the processing extent.
        ------------------------------------     --------------------------------------------------------------------
        cellsize_type                            Optional string. Specifies the cell size to be used for the function.

                                                    - "FirstOf" - Use the first cell size of the input rasters. This is the default.

                                                    - "MinOf" - Use the smallest cell size of all the input rasters.

                                                    - "MaxOf" - Use the largest cell size of all the input rasters.

                                                    - "MeanOf" - Use the mean cell size of all the input rasters.

                                                    - "LastOf" - Use the last cell size of the input rasters.
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object
        """
        return self._ras_coll_engine_obj.median(
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def mean(
        self,
        ignore_nodata: bool = True,
        extent_type: str = "FirstOf",
        cellsize_type: str = "FirstOf",
    ):
        """
        The ``mean`` method returns a :class:`~arcgis.raster.Raster` object in which each band contains the average
        pixel values for that band across all rasters in the raster collection.

        For example, if there are ten raster items in the :class:`~arcgis.raster.RasterCollection`,
        each with four bands, the mean method will calculate the mean pixel value
        that occurs across all raster items for band 1, for band 2, for band 3,
        and for band 4; a four-band raster is returned. Band numbers are matched
        between raster items using the band index, so the items in the raster
        collection must follow the same band order.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored.

                                                    - True : The method will include all valid pixels and ignore any NoData pixels. This is the default.
                                                    - False : The method will result in NoData if there are any NoData values.
        ------------------------------------     --------------------------------------------------------------------
        extent_type                              Optional string. Specifies the extent to be used for the function.

                                                    - "FirstOf" - Use the extent of the first input raster to determine the processing extent. This is the default.

                                                    - "IntersectionOf" - Use the extent of the overlapping pixels to determine the processing extent.

                                                    - "UnionOf" - Use the extent of all the rasters to determine the processing extent.

                                                    - "LastOf" - Use the extent of the last input raster to determine the processing extent.
        ------------------------------------     --------------------------------------------------------------------
        cellsize_type                            Optional string. Specifies the cell size to be used for the function.

                                                    - "FirstOf" - Use the first cell size of the input rasters. This is the default.

                                                    - "MinOf" - Use the smallest cell size of all the input rasters.

                                                    - "MaxOf" - Use the largest cell size of all the input rasters.

                                                    - "MeanOf" - Use the mean cell size of all the input rasters.

                                                    - "LastOf" - Use the last cell size of the input rasters.
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object
        """
        return self._ras_coll_engine_obj.mean(
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def majority(
        self,
        ignore_nodata: bool = True,
        extent_type: str = "FirstOf",
        cellsize_type: str = "FirstOf",
    ):
        """
        The ``majority`` method returns a :class:`~arcgis.raster.Raster` object in which each band contains the pixel
        value that occurs most frequently for that band across all
        rasters in the raster collection.

        For example, if there are ten raster items in the :class:`~arcgis.raster.RasterCollection`,
        each with four bands, the majority method will determine the
        pixel value that occurs most frequently across all raster
        items for band 1, for band 2, for band 3, and for band 4;
        a four-band raster is returned. Band numbers are matched
        between raster items using the band index, so the items
        in the raster collection must follow the same band order.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored.

                                                    - True : The method will include all valid pixels and ignore any NoData pixels. This is the default.
                                                    - False : The method will result in NoData if there are any NoData values.
        ------------------------------------     --------------------------------------------------------------------
        extent_type                              Optional string. Specifies the extent to be used for the function.

                                                    - "FirstOf" - Use the extent of the first input raster to determine the processing extent. This is the default.

                                                    - "IntersectionOf" - Use the extent of the overlapping pixels to determine the processing extent.

                                                    - "UnionOf" - Use the extent of all the rasters to determine the processing extent.

                                                    - "LastOf" - Use the extent of the last input raster to determine the processing extent.
        ------------------------------------     --------------------------------------------------------------------
        cellsize_type                            Optional string. Specifies the cell size to be used for the function.

                                                    - "FirstOf" - Use the first cell size of the input rasters. This is the default.

                                                    - "MinOf" - Use the smallest cell size of all the input rasters.

                                                    - "MaxOf" - Use the largest cell size of all the input rasters.

                                                    - "MeanOf" - Use the mean cell size of all the input rasters.

                                                    - "LastOf" - Use the last cell size of the input rasters.
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object

        """
        return self._ras_coll_engine_obj.majority(
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def sum(
        self,
        ignore_nodata: bool = True,
        extent_type: str = "FirstOf",
        cellsize_type: str = "FirstOf",
    ):
        """
        The ``sum`` method returns a :class:`~arcgis.raster.Raster` object in which each band contains the sum
        of pixel values for that band across all rasters in the :class:`~arcgis.raster.RasterCollection`.

        For example, if there are ten raster items in the raster collection,
        each with four bands, the sum method will calculate the sum of pixel
        values for each pixel that occurs across all raster items for band 1,
        band 2, band 3, and band 4; a four-band raster is returned.
        Band numbers are matched between raster items using the band index,
        so the items in the raster collection must follow the same band order.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored.

                                                    - True : The method will include all valid pixels and ignore any NoData pixels. This is the default.
                                                    - False : The method will result in NoData if there are any NoData values.
        ------------------------------------     --------------------------------------------------------------------
        extent_type                              Optional string. Specifies the extent to be used for the function.

                                                    - "FirstOf" - Use the extent of the first input raster to determine the processing extent. This is the default.

                                                    - "IntersectionOf" - Use the extent of the overlapping pixels to determine the processing extent.

                                                    - "UnionOf" - Use the extent of all the rasters to determine the processing extent.

                                                    - "LastOf" - Use the extent of the last input raster to determine the processing extent.
        ------------------------------------     --------------------------------------------------------------------
        cellsize_type                            Optional string. Specifies the cell size to be used for the function.

                                                    - "FirstOf" - Use the first cell size of the input rasters. This is the default.

                                                    - "MinOf" - Use the smallest cell size of all the input rasters.

                                                    - "MaxOf" - Use the largest cell size of all the input rasters.

                                                    - "MeanOf" - Use the mean cell size of all the input rasters.

                                                    - "LastOf" - Use the last cell size of the input rasters.
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object
        """
        return self._ras_coll_engine_obj.sum(
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def std(
        self,
        ignore_nodata: bool = True,
        extent_type: str = "FirstOf",
        cellsize_type: str = "FirstOf",
    ):
        """
        The ``std`` method returns a :class:`~arcgis.raster.Raster` object in which each band contains the std
        of pixel values for that band across all rasters in the :class:`~arcgis.raster.RasterCollection`.

        For example, if there are ten raster items in the raster collection,
        each with four bands, the std method will calculate the std of pixel
        values for each pixel that occurs across all raster items for band 1,
        band 2, band 3, and band 4; a four-band raster is returned.
        Band numbers are matched between raster items using the band index,
        so the items in the raster collection must follow the same band order.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored.

                                                    - True : The method will include all valid pixels and ignore any NoData pixels. This is the default.
                                                    - False : The method will result in NoData if there are any NoData values.
        ------------------------------------     --------------------------------------------------------------------
        extent_type                              Optional string. Specifies the extent to be used for the function.

                                                    - "FirstOf" - Use the extent of the first input raster to determine the processing extent. This is the default.

                                                    - "IntersectionOf" - Use the extent of the overlapping pixels to determine the processing extent.

                                                    - "UnionOf" - Use the extent of all the rasters to determine the processing extent.

                                                    - "LastOf" - Use the extent of the last input raster to determine the processing extent.
        ------------------------------------     --------------------------------------------------------------------
        cellsize_type                            Optional string. Specifies the cell size to be used for the function.

                                                    - "FirstOf" - Use the first cell size of the input rasters. This is the default.

                                                    - "MinOf" - Use the smallest cell size of all the input rasters.

                                                    - "MaxOf" - Use the largest cell size of all the input rasters.

                                                    - "MeanOf" - Use the mean cell size of all the input rasters.

                                                    - "LastOf" - Use the last cell size of the input rasters.
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object
        """
        return self._ras_coll_engine_obj.std(
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def mosaic(self, mosaic_method: str = "FIRST"):
        """
        The ``mosaic`` method returns a :class:`~arcgis.raster.Raster` object in which all items in a
        :class:`~arcgis.raster.RasterCollection` have been mosaicked into a single raster.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        mosaic_method                            Optional string. The method used to handle overlapping areas
                                                 between adjacent raster items. Mosaic method options include the following:

                                                   - `FIRST` -  Determines the pixel value from the first raster that is overlapping.

                                                   - `LAST` - Determines the pixel value from the last raster that is overlapping.

                                                   - `MEAN` - Determines the average pixel value from the two rasters that are overlapping.

                                                   - `MIN` - Determines the lower pixel value from the two raster datasets that are overlapping.

                                                   - `MAX` - Determines the higher pixel value from the two raster datasets that are overlapping.

                                                   - `SUM` - Determines the sum of pixel values from the two rasters that are overlapping.


                                                 The default value is `FIRST`
        ====================================     ====================================================================

        :return: A :class:`~arcgis.raster.Raster` object

        """
        return self._ras_coll_engine_obj.mosaic(mosaic_method=mosaic_method)

    def quality_mosaic(
        self,
        quality_rc_or_list: Union[RasterCollection, list[Raster]],
        statistic_type: Optional[str] = None,
    ):
        """
        The ``quality_mosaic`` method returns a :class:`~arcgis.raster.Raster` object in which all items in a
        :class:`~arcgis.raster.RasterCollection` have been  mosaicked into a single raster based on a quality
        requirement.


        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        quality_rc_or_list                       Required. The :class:`~arcgis.raster.RasterCollection`
                                                 or list of rasters to be used as quality indicators.

                                                 For example, Landsat 8's Band 1 is the Coastal/Aerosol band, 
                                                 which can be used to estimate the concentration of fine aerosol 
                                                 particles such as smoke and haze in the atmosphere. For a 
                                                 collection of Landsat 8 images, use the select_bands method to 
                                                 return a RasterCollection object containing only Band 1 from 
                                                 each raster item. The number of raster items in the 
                                                 quality_rc_or_list must match the number of raster items 
                                                 in the raster collection to be mosaicked.
        ------------------------------------     --------------------------------------------------------------------
        statistic_type                           Required string. The statistic used to compare the input collection 
                                                 or list of quality rasters.

                                                    `MAX` - The highest pixel value in the input quality rasters will \
                                                          be the pixel value in the output raster. This is the default.

                                                    `MEDIAN` - The median pixel value in the input quality rasters \
                                                             will be the pixel value in the output raster.

                                                    `MIN` - The minimum pixel value in the input quality rasters \
                                                          will be the pixel value in the output raster.


                                                 .. note::
                                                    For example, to mosaic the input raster collection such that
                                                    those with the lowest aerosol content are on top,
                                                    use the ``MIN`` statistic type.

        ====================================     ====================================================================

        :return: a ``Raster`` object

        """
        return self._ras_coll_engine_obj.quality_mosaic(
            quality_rc_or_list=quality_rc_or_list,
            statistic_type=statistic_type,
        )

    def select_bands(
        self,
        band_ids_or_names: Union[str, int, list[str], list[int]],
        context: Optional[dict[str, Any]] = None,
    ):
        """
        The ``select_bands`` method selects a list of bands from every :class:`~arcgis.raster.Raster` item in a
        :class:`~arcgis.raster.RasterCollection` and returns a raster collection that contains :class:`~arcgis.raster.Raster` items with only
        the selected bands.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        band_ids_or_names                        Required. The names or index numbers of bands to be included in
                                                 the returned raster items. This can be specified with a single string,
                                                 integer, or a list of strings or integers.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return:
            A :class:`~arcgis.raster.RasterCollection` that contains
            :class:`~arcgis.raster.Raster` items with only the selected bands.

        """
        return self._ras_coll_engine_obj.select_bands(
            band_ids_or_names=band_ids_or_names, context=context
        )

    def map(self, func: dict[str, Any], context: Optional[dict[str, Any]] = None):
        """
        The ``map`` method maps a Python function over a raster collection.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        func                                     Required. The Python function to map over the raster collection.
                                                 The return value of the function must be a dictionary in which one
                                                 of the keys is raster. For example,
                                                 {"raster": output_raster_object, "name": input_item_name["name"]}.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return:
            A new :class:`~arcgis.raster.RasterCollection` created from the existing
            ``RasterCollection`` after applying the func on each item.

        .. code-block:: python

            # Usage Example: This snippet maps grayscale function to each raster item in the raster collection.

            rc_local = RasterCollection(r"./data/rasters.gdb/rasters")

            def apply_grayscale(item):
                raster = item["Raster"]
                gray = grayscale(raster)
                return {"raster": gray, "Name": item["Name"], "StdTime": item["AcquisitionDate"]}

            gray_rc = rc_local.map(func=apply_grayscale)

        """
        return self._ras_coll_engine_obj.map(func=func, context=context)

    def reduce(self, func, func_args=None):
        """
        The ``reduce`` method composites all the images in the collection to a single image based on a reducer function.

        ====================================     ====================================================================
        **Parameter**                            **Description**
        ------------------------------------     --------------------------------------------------------------------
        func                                     Required. The Python function to reduce the raster collection.
                                                 The function should accept a list of rasters and return a single reduced raster
        ------------------------------------     --------------------------------------------------------------------
        func_args                                Optional dictionary. Additional paramters to be passed to the reducer function.
        ====================================     ====================================================================

        :return: a ``Raster`` object

        .. code-block:: python

            # Usage Example 1: This snippet reduces a raster collection based on a reducer function from arcgis.raster.functions module that can accept a list of rasters.

            rc = RasterCollection("https://myserver/arcgis/rest/services/ImageServiceName/ImageServer")
            from arcgis.raster.functions import max
            max_raster = rc.reduce(func=max, func_args={"cellsize_type":"MinOf"})

            # Usage Example 2: This snippet reduces a raster collection based on a custom reducer function.

            rc = RasterCollection("https://myserver/arcgis/rest/services/ImageServiceName/ImageServer")

            def skewness(ras_list):
                from arcgis.raster.functions import mean, std, med
                cs_mean = mean(ras_list, process_as_multiband=True)
                cs_stddev = std(ras_list, process_as_multiband=True)
                cs_median = med(ras_list, process_as_multiband=True)
                out_skewness = 3*(cs_mean - cs_median)/cs_stddev
                return out_skewness

            skewness = rc.reduce(func=skewness)

        """
        if func_args is None:
            func_args = {}
        return self._ras_coll_engine_obj.reduce(func=func, func_args=func_args)

    def merge(self, collection2):
        """
        The ``merge`` method merges two `RasterCollections` into one. The output has all the items that were in either collection.

        ====================================     ====================================================================
        **Parameter**                            **Description**
        ------------------------------------     --------------------------------------------------------------------
        collection2                              RasterCollection object. The second collection to merge.
        ====================================     ====================================================================

        :return: a new Collection that has all the items that were in either collection.

        .. code-block:: python

            # Usage Example 1: merges two image collections rc1 and rc2 into one.

            rc1 = rc.filter_by_attribute("OBJECTID", "EQUALS", 1)
            rc2 = rc.filter_by_attribute("OBJECTID", "EQUALS", 2)
            new_rc = rc1.merge(rc2)

        """

        return self._ras_coll_engine_obj.merge(collection2._ras_coll_engine_obj)

    def summarize_field(self, field_name, summary_type="ALL"):
        """
        Summarizes a numeric field of the RasterCollection based on the specified summary_type

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        field_name                               Required string. The name of the field to be summarized
        ------------------------------------     --------------------------------------------------------------------
        summary_type                             Required string or list of strings representing the summary type.
                                                 Possible values - "COUNT", "COUNT_DISTINCT", "FIRST","HISTOGRAM", "MAX", "MEAN", "MIN",
                                                 "PRODUCT", "SAMPLE_SD", "SAMPLE_VAR", "SUM", "TOTAL_SD", "TOTAL_VAR", "ALL".
        ====================================     ====================================================================

        :return:
            A dictionary with key being the summary type and the value being the summary value.
        """

        property_values = self.get_field_values(field_name)
        summary_dict = {}
        import numbers

        if not isinstance(summary_type, list):
            summary_type = [summary_type]

        if "ALL" in map(str.upper, summary_type):
            summary_type = [
                "COUNT",
                "COUNT_DISTINCT",
                "FIRST",
                "HISTOGRAM",
                "MAX",
                "MEAN",
                "MIN",
                "PRODUCT",
                "SAMPLE_SD",
                "SAMPLE_VAR",
                "SUM",
                "TOTAL_SD",
                "TOTAL_VAR",
            ]

        from operator import is_not
        from functools import partial

        property_values_not_none = list(filter(partial(is_not, None), property_values))

        all_num = all(isinstance(x, numbers.Number) for x in property_values_not_none)
        if not all_num:
            raise RuntimeError("Only numeric fields can be summarized")
        try:
            for summary in summary_type:
                val = None
                summary = summary.lower()
                if summary == "count":
                    val = len(property_values_not_none)

                elif summary == "count_distinct":
                    val = len(np.unique(property_values_not_none))

                elif summary == "first":
                    val = property_values_not_none[0]

                elif summary == "histogram":
                    unique, counts = np.unique(
                        property_values_not_none, return_counts=True
                    )
                    val = dict(zip(unique, counts))

                elif summary == "max":
                    val = np.max(property_values_not_none)

                elif summary == "mean":
                    val = np.mean(property_values_not_none)

                elif summary == "min":
                    val = np.min(property_values_not_none)

                elif summary == "product":
                    val = np.prod(property_values_not_none)

                elif summary == "sample_sd":
                    val = np.std(property_values_not_none, ddof=1)

                elif summary == "sample_var":
                    val = np.var(property_values_not_none, ddof=1)

                elif summary == "sum":
                    val = np.sum(property_values_not_none)

                elif summary == "total_sd":
                    val = np.std(property_values_not_none)

                elif summary == "total_var":
                    val = np.var(property_values_not_none)

                else:
                    raise ValueError("invalid summary_type value")

                summary_dict.update({summary: val})
        except:
            raise RuntimeError("Failed to summarize the property")

        return summary_dict

    def add_field(
        self,
        field_name: str,
        field_values: list,
        context: Optional[dict[str, Any]] = None,
    ):
        """
        Adds a new field to the raster collection and populates it with values.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        field_name                               Required string. The name of the field to be added.
        ------------------------------------     --------------------------------------------------------------------
        field_values                             Required list. The list of values associated with the field name.
                                                 The length of the list should match the number of items in the raster collection
                                                 Providing only one value will set the same value for all rows.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return:
            A new :class:`~arcgis.raster.RasterCollection` that has the new field added.
        """

        return self._ras_coll_engine_obj.add_field(
            field_name, field_values, context=context
        )

    def group_by(self, field_name: str, context: Optional[dict[str, Any]] = None):
        """
        group_by method can be used to group the raster collection based on a field.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        field_name                               Required string. The name of the field that is used to group the raster collection.
                                                 Items with the same field values will be grouped together.
        ------------------------------------     --------------------------------------------------------------------
        context                                  Optional dictionary. Additional properties to control the creation of RasterCollection.
                                                 The default value for the context parameter would be the same as that of the
                                                 context settings applied to the parent collection.

                                                 Currently available:

                                                     -  query_boundary:
                                                        This boolean value set to this option determines whether to add SHAPE field
                                                        to the RasterCollection. The value in the SHAPE field represents the
                                                        boundary/geometry of the raster. The query_boundary parameter is honoured
                                                        only when the RasterCollection is created from a list of Rasters.

                                                        - True: Set query_boundary to True to add the SHAPE field to the RasterCollection.

                                                        - False: Set query_boundary to False to not add the SHAPE field to the RasterCollection. (Creation of RasterCollection would be faster)

                                                        Example:

                                                        {"query_boundary":True}
        ====================================     ====================================================================

        :return:
            A Dictionary. The dictionary that contains the grouped raster collections. The key of the dictionary is a
            field value of the field name that the grouping is based on. The value of the dictionary is a raster
            collection whose field name contains the same field value.

        .. code-block:: python

            # Usage Example 1: This example groups the raster collection into yearly data and creates a new raster collection using data from 1990.

            group_by_year = rc.group_by(field_name="Year", context=None)
            rc_1990 = group_by_year[1990]

        """

        return self._ras_coll_engine_obj.group_by(field_name, context=context)

    def _as_df(
        self,
        result_offset=None,
        result_record_count=None,
        return_all_records=False,
    ):
        """
        Returns the RasterCollection object as a dataframe

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        result_offset                            optional integer. This option fetches query results by skipping a
                                                 specified number of records. The query results start from the next
                                                 record (i.e., resultOffset + 1). The Default value is None.

                                                 only honoured when input is an image service
        ------------------------------           --------------------------------------------------------------------
        result_record_count                      optional integer. This option fetches query results up to the
                                                 resultRecordCount specified. When resultOffset is specified and this
                                                 parameter is not, image layer defaults to maxRecordCount. The
                                                 maximum value for this parameter is the value of the layer's
                                                 maxRecordCount property.
                                                 max_allowable_offset - This option can be used to specify the
                                                 max_allowable_offset to be used for generalizing geometries returned
                                                 by the query operation. The max_allowable_offset is in the units of
                                                 the out_sr. If outSR is not specified, max_allowable_offset is
                                                 assumed to be in the unit of the spatial reference of the Layer.
        ------------------------------           --------------------------------------------------------------------
        return_all_records                       Optional boolean. To return all records
        ====================================     ====================================================================

        :return: a dataframe object

        """

        return self._ras_coll_engine_obj._as_df(
            result_offset=result_offset,
            result_record_count=result_record_count,
            return_all_records=return_all_records,
        )

    # def display_image(self, item=None):
    #    if not raster.use_server:
    #        raise RuntimeError('Not available in non server env')
    #    else:
    #        if item is not None:
    #            newcollection = self._clone_raster_collection()
    #            newcollection._mosaic_rule = {
    #                "mosaicMethod": "esriMosaicLockRaster",
    #                "lockRasterIds": [item],
    #                "ascending": True,
    #                "mosaicOperation": "MT_FIRST"
    #            }
    #        else:
    #            newcollection = self

    #        bbox_sr = None
    #        if 'spatialReference' in newcollection.extent:
    #            bbox_sr = newcollection.extent['spatialReference']
    #        if not newcollection._uses_gbl_function:
    #            byte_array = (newcollection.export_image(bbox=newcollection._extent, bbox_sr=bbox_sr, size=[1200, 450], export_format='jpeg', f='image'))
    #            try:
    #                from IPython.display import Image
    #                return Image(byte_array)
    #            except:
    #                return byte_array

    # def _clone_raster_collection(self):
    #    return self._ras_coll_engine_obj.filter_by(where_clause=where_clause, query_geometry_or_extent=query_geometry_or_extent)

    # def _object_id_name(self):
    #    for ele in self.properties.fields:
    #        if ("type" in ele.keys()) and ele["type"]=="esriFieldTypeOID":
    #            return ele["name"]

    def _repr_html_(self):
        return self._ras_coll_engine_obj._repr_html_()

    def _repr_jpeg_(self):
        return None


class _ArcpyRasterCollection(RasterCollection, ImageryLayer):
    def __init__(
        self,
        rasters,
        attribute_dict=None,
        where_clause=None,
        query_geometry=None,
        engine=None,
        gis=None,
        context=None,
    ):
        ImageryLayer.__init__(self, str(rasters), gis=gis)

        self._mosaic_rule = None
        if gis is not None:
            self._gis = gis
        else:
            self._gis = None
        if where_clause is None:
            self._where_clause = "1=1"
        else:
            self._where_clause = where_clause
        if query_geometry is None:
            self._spatial_filter = None
        else:
            self._spatial_filter = query_geometry

        self._attribute_dict = attribute_dict

        self._local = False
        # self._do_not_hydrate=False
        self._rasters = rasters
        # self._ras_coll_engine=_ArcpyRasterCollection
        # self._engine=_ArcpyRaster
        # self._engine_obj=_ArcpyRaster(rasters, False, gis)
        self._ras_coll_engine = engine
        if context is None:
            self._context = {}
        else:
            self._context = context
        import arcpy

        try:
            arcpy.CheckOutExtension("ImageAnalyst")
            arcpy.CheckOutExtension("Spatial")
        except:
            pass
        if isinstance(rasters, str):
            if (
                "https://" in rasters or "http://" in rasters
            ):  # To provide access to secured service
                if "ImageServer" in rasters and self._token is not None:
                    self._raster_collection = arcpy.ia.RasterCollection(
                        rasters + "?token=" + self._token, attribute_dict
                    )
                else:
                    self._raster_collection = arcpy.ia.RasterCollection(
                        rasters, attribute_dict
                    )
            else:
                self._raster_collection = arcpy.ia.RasterCollection(
                    rasters, attribute_dict
                )
        if isinstance(rasters, list):
            arcpy_rasters_list = []
            for ele in rasters:
                if (isinstance(ele, Raster)) and isinstance(
                    ele._engine_obj, _ArcpyRaster
                ):
                    arcpy_rasters_list.append(ele._engine_obj._raster)

            if arcpy_rasters_list != []:
                rasters = arcpy_rasters_list
        self._raster_collection = arcpy.ia.RasterCollection(rasters, attribute_dict)
        self._df = self._as_df()

    @property
    def count(self):
        count = len(self._df.index)
        return count

    @property
    def fields(self):
        fields = self._raster_collection.fields
        return fields

    @property
    def _rasters_list(self):
        value_rasters = []
        for i in range(0, len(self)):
            value_rasters.append(Raster(self._raster_collection[i]["Raster"]))
        return value_rasters

    def __iter__(self):
        return iter(self._df.to_dict("records", into=dict))

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        return self._df.to_dict("records", into=dict)[item]

    # @property
    # def count(self):
    #    count = self._raster_collection.count
    #    return count

    # @property
    # def fields(self):
    #    fields = self._raster_collection.fields
    #    return fields

    # @property
    # def _rasters_list(self):
    #    ras_list=self.get_field_values("Raster")
    #    return ras_list

    # def __iter__(self):
    #    return(self._raster_collection.__iter__())

    # def __next__(self):
    #    return (self._raster_collection.__next__())

    # def __len__(self):
    #    return (self._raster_collection.__len__())

    # def __getitem__(self, item):
    #    return (self._raster_collection.__getitem__(item))

    def filter_by(
        self,
        where_clause="",
        query_geometry_or_extent=None,
        raster_query="",
        context=None,
    ):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        if isinstance(query_geometry_or_extent, _arcgis.geometry.Geometry):
            query_geometry_or_extent = query_geometry_or_extent.as_arcpy
        if where_clause is None:
            where_clause = ""
        if raster_query is None:
            raster_query = ""
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.filter(
                where_clause=where_clause,
                query_geometry_or_extent=query_geometry_or_extent,
                raster_query=raster_query,
            )
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def filter_by_time(
        self,
        start_time="",
        end_time="",
        time_field_name="StdTime",
        date_time_format=None,
        context=None,
    ):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.filterByTime(
                start_time=start_time,
                end_time=end_time,
                time_field_name=time_field_name,
            )
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def filter_by_calendar_range(
        self,
        calendar_field,
        start,
        end=None,
        time_field_name="StdTime",
        date_time_format=None,
        context=None,
    ):
        """
        filter the raster collection by a calendar_field and its start and end value (inclusive). i.e. if you would like
        to select all the rasters that have the time stamp on Monday, specify calendar_field as 'DAY_OF_WEEK' and put start and
        end to 1.

        :param calendar_field: string, one of 'YEAR', 'MONTH', 'QUARTER', 'WEEK_OF_YEAR', 'DAY_OF_YEAR', 'DAY_OF_MONTH',
         'DAY_OF_WEEK', 'HOUR'
        :param start: integer, the start time. inclusive.
        :param end: integer, default is None, if default is used, the end is set equal to start. inclusive.
        :param time_field_name: string, the time field anme, default is 'StdTime'.
        :param date_time_format: the time format that is used to format the time field values. Please ref the python
                                date time standard for this argument. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
                                Default is None and this means using the Pro standard time format '%Y-%m-%dT%H:%M:%S'
                                and ignoring the following sub-second.

        :return: a filtered raster collection.
        """
        # validation
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.filterByCalendarRange(
                calendar_field=calendar_field,
                start=start,
                end=end,
                time_field_name=time_field_name,
                date_time_format=date_time_format,
            )
        )
        # newcollection._ras_coll_engine_obj._raster_collection = self._raster_collection.filterByTime(self._raster_collection.filterByCalendarRange(calendar_field=calendar_field, start=start, end=end,time_field_name=time_field_name,date_time_format=date_time_format))
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def filter_by_geometry(self, query_geometry_or_extent, context=None):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        if isinstance(query_geometry_or_extent, _arcgis.geometry.Geometry):
            query_geometry_or_extent = query_geometry_or_extent.as_arcpy
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.filterByGeometry(
                query_geometry_or_extent=query_geometry_or_extent
            )
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def filter_by_attribute(self, field_name, operator, field_values, context=None):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.filterByAttribute(
                field_name=field_name,
                operator=operator,
                field_values=field_values,
            )
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def filter_by_raster_property(
        self, property_name, operator, property_values, context=None
    ):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.filterByRasterProperty(
                property_name=property_name,
                operator=operator,
                property_values=property_values,
            )
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def sort(self, field_name, ascending=True, context=None):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.sort(field_name=field_name, ascending=ascending)
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def get_field_values(self, field_name, max_count=0):
        df = self._df
        if max_count != 0:
            return df[field_name].tolist()[0:max_count]
        else:
            return df[field_name].tolist()

    def to_multidimensional_raster(self, variable_field_name, dimension_field_names):
        return Raster(
            self._raster_collection.toMultidimensionalRaster(
                variable_field_name=variable_field_name,
                dimension_field_names=dimension_field_names,
            )
        )

    def max(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        return Raster(
            self._raster_collection.max(
                ignore_nodata=ignore_nodata,
                extent_type=extent_type,
                cellsize_type=cellsize_type,
            )
        )

    def min(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        return Raster(
            self._raster_collection.min(
                ignore_nodata=ignore_nodata,
                extent_type=extent_type,
                cellsize_type=cellsize_type,
            )
        )

    def median(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        return Raster(
            self._raster_collection.median(
                ignore_nodata=ignore_nodata,
                extent_type=extent_type,
                cellsize_type=cellsize_type,
            )
        )

    def mean(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        return Raster(
            self._raster_collection.mean(
                ignore_nodata=ignore_nodata,
                extent_type=extent_type,
                cellsize_type=cellsize_type,
            )
        )

    def majority(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        return Raster(
            self._raster_collection.majority(
                ignore_nodata=ignore_nodata,
                extent_type=extent_type,
                cellsize_type=cellsize_type,
            )
        )

    def sum(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        return Raster(
            self._raster_collection.sum(
                ignore_nodata=ignore_nodata,
                extent_type=extent_type,
                cellsize_type=cellsize_type,
            )
        )

    def std(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        return Raster(
            self._raster_collection.std(
                ignore_nodata=ignore_nodata,
                extent_type=extent_type,
                cellsize_type=cellsize_type,
            )
        )

    def mosaic(self, mosaic_method):
        return Raster(self._raster_collection.mosaic(mosaic_method=mosaic_method))

    def quality_mosaic(self, quality_rc_or_list, statistic_type=None):
        return Raster(
            self._raster_collection.quality_mosaic(
                quality_rc_or_list=quality_rc_or_list,
                statistic_type=statistic_type,
            )
        )

    def select_bands(self, band_ids_or_names, context=None):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.selectBands(band_ids_or_names)
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def map(self, func, context=None):
        if context is None:
            context = self._context
        res = map(func, iter(self))
        rasters = []
        attribute_dict = defaultdict(list)
        for item in res:
            rasters.append(item["raster"])
            for key, value in item.items():
                if key != "raster":
                    attribute_dict[key].append(value)

        return RasterCollection(rasters, attribute_dict, context=context)

    def reduce(self, func, func_args={}):
        reduced_raster = func(self, **func_args)
        return reduced_raster

    def merge(self, collection2):
        newcollection = self._clone_raster_collection()

        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.merge(collection2._raster_collection)
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def add_field(self, field_name, field_values, context=None):
        """
         Adds a new field to the raster collection and populate it with values.
        :param field_name: Required string. The name of the field to be added.
        :param field_values: Required list. The list of values associated with the field name.
                             The length of the list should match the number of items in the raster collection
                             Providing only one value will set the same value for all rows.
        :return: Collection that has the new field added.
        """
        if context is None:
            context = self._context

        if field_name in self.fields:
            raise RuntimeError("Cannot add the field. The field name already exists.")

        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._raster_collection = (
            self._raster_collection.addField(
                field_name=field_name, field_values=field_values
            )
        )
        newcollection._ras_coll_engine_obj._df = (
            newcollection._ras_coll_engine_obj._as_df()
        )
        return newcollection

    def group_by(self, field_name, context=None):
        """
         group_by method can be used to group the raster collection based on a field.
        :param field_name: Required string.The name of the field that is used to group the raster collection. Items with the same field values will be grouped together.
        :return: Dictionary.The dictionary that contains the grouped raster collections. The key of the dictionary is a field value of the field name that the grouping is based on.
                 The value of the dictionary is a raster collection whose field name contains the same field value.
        """
        if context is None:
            context = self._context

        try:
            grouped_output_arcpy = self._raster_collection.groupBy(
                field_name=field_name
            )
            new_grouped_output = {}
            for item, value in grouped_output_arcpy.items():
                newcollection = self._clone_raster_collection(context=context)
                newcollection._ras_coll_engine_obj._raster_collection = value
                newcollection._ras_coll_engine_obj._df = (
                    newcollection._ras_coll_engine_obj._as_df()
                )
                new_grouped_output.update({item: newcollection})
            return new_grouped_output

        except:
            raise RuntimeError("group_by failed with the field_name - " + field_name)

    def _as_df(
        self,
        result_offset=None,
        result_record_count=None,
        return_all_records=False,
    ):
        import pandas as pd

        data = {}
        value_rasters = []
        value_geometries = []
        for index, field in enumerate(self.fields):
            try:
                value = self._raster_collection.getFieldValues(field)
                if field == "Raster":
                    for i, ele in enumerate(value):
                        value_rasters.append(
                            Raster(self._raster_collection[i]["Raster"])
                        )
                    data[field] = value_rasters
                elif field == "Shape":
                    for ele in value:
                        value_geometries.append(Geometry(ele.JSON))
                    data["Shape"] = value_geometries
                else:
                    data[field] = self._raster_collection.getFieldValues(field)
            except:
                continue
        return pd.DataFrame(data=data)

    def display_image(self, item=None):
        raise RuntimeError("Not available in non server env")

    def _clone_raster_collection(self, context=None):
        new_raster_collection = RasterCollection(
            self._rasters,
            self._attribute_dict,
            self._where_clause,
            self._spatial_filter,
            self._ras_coll_engine,
            self._gis,
            context=context,
        )
        new_raster_collection._fn = self._fn
        new_raster_collection._fnra = self._fnra
        new_raster_collection._mosaic_rule = self._mosaic_rule
        new_raster_collection._extent = self._extent
        return new_raster_collection

    def _object_id_name(self):
        for ele in self.properties.fields:
            if ("type" in ele.keys()) and ele["type"] == "esriFieldTypeOID":
                return ele["name"]

    def _repr_html_(self):
        return self._df._repr_html_()

    def _repr_jpeg_(self):
        return None


class _ImageServerRasterCollection(ImageryLayer, RasterCollection):
    def __init__(
        self,
        rasters=None,
        attribute_dict=None,
        where_clause=None,
        query_geometry=None,
        engine=None,
        gis=None,
        context=None,
    ):
        # self._remote = raster.use_server_engine
        ImageryLayer.__init__(self, rasters, gis=gis)
        self._mosaic_rule = None
        if gis is not None:
            self._gis = gis
        else:
            self._gis = None
        if where_clause is None:
            self._where_clause = "1=1"
        else:
            self._where_clause = where_clause
        if query_geometry is None:
            self._spatial_filter = None
        else:
            self._spatial_filter = query_geometry
        self._rasters = rasters
        self._attribute_dict = attribute_dict

        self._object_ids = None
        self._raster_query = None
        self._order_by_fields = None

        self._do_not_hydrate = False
        # self._ras_coll_engine=_ImageServerRasterCollection
        self._is_multidimensional = False
        self._engine = _ImageServerRaster
        self._engine_obj = _ImageServerRaster(rasters, False, gis)
        self._ras_coll_engine = engine
        self._df = None
        if context is None:
            self._context = {}
        else:
            self._context = context

        if str(self.properties["capabilities"]).lower().find("catalog") == -1:
            raise RuntimeError(
                "Image Service should have 'Catalog' capability to create a RasterCollection object."
            )
        self._df = self._as_df()
        self._start = 0
        self._lower_limit = self.properties.maxRecordCount
        self._upper_limit = 0
        self._max_rec_count = self.properties.maxRecordCount
        self._rep_df = None
        self._order_by_fields = self._object_id_name() + " ASC"
        self._count = None

    @property
    def count(self):
        if self._count is None:
            count = self.query(
                where=self._where_clause,
                geometry_filter=self._spatial_filter,
                object_ids=self._object_ids,
                return_count_only=True,
                raster_query=self._raster_query,
            )
            self._count = count
        else:
            count = self._count
        return count

    @property
    def fields(self):
        return tuple(self._df.columns.tolist())

    @property
    def _rasters_list(self):
        ras_list = self.get_field_values("Raster")
        return ras_list

    def __iter__(self):
        lim = True
        while lim:
            try:
                item = self[self._start]
            except IndexError:
                lim = False
                self._start = 0
            else:
                self._start += 1
                yield item

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        if item < self._max_rec_count:
            return self._df.to_dict("records", into=dict)[item]
        offset = self._max_rec_count
        i = 2
        while item >= offset:
            offset = self._max_rec_count * i
            i = i + 1
        offset = offset - self._max_rec_count
        if item >= self._upper_limit or item < self._lower_limit:
            self._rep_df = self._as_df(
                result_offset=offset, result_record_count=self._max_rec_count
            )
            self._upper_limit = offset + self._max_rec_count
            self._lower_limit = offset
        return (self._rep_df.to_dict("records", into=dict))[item - offset]

    def filter_by(
        self,
        where_clause=None,
        query_geometry_or_extent=None,
        raster_query=None,
        context=None,
    ):
        if context is None:
            context = self._context
        from arcgis.geometry.filters import intersects

        geometry_filter = None
        if query_geometry_or_extent is not None:
            query_geometry_or_extent = _get_geometry(query_geometry_or_extent)
            geometry_filter = intersects(query_geometry_or_extent)
        else:
            geometry_filter = self._spatial_filter
        # newcollection = self._clone_raster_collection()
        if where_clause is not None:
            where_clause = where_clause + " AND (" + self._where_clause + ")"
        else:
            where_clause = self._where_clause

        if raster_query is not None and self._raster_query is not None:
            raster_query = self._raster_query + " AND (" + raster_query + ")"
        elif raster_query is None:
            raster_query = self._raster_query
        oids = super().query(
            where=where_clause,
            geometry_filter=geometry_filter,
            return_ids_only=True,
            raster_query=raster_query,
        )["objectIds"]

        newcollection = RasterCollection(
            rasters=self._url,
            attribute_dict=self._attribute_dict,
            where_clause=where_clause,
            query_geometry=geometry_filter,
            gis=self._gis,
            context=context,
        )
        newcollection._ras_coll_engine_obj._mosaic_rule = {
            "mosaicMethod": "esriMosaicLockRaster",
            "lockRasterIds": oids,
            "ascending": True,
            "mosaicOperation": "MT_FIRST",
        }
        newcollection._ras_coll_engine_obj._raster_query = raster_query
        newcollection._ras_coll_engine_obj._object_ids = oids
        # newcollection._where_clause=where_clause
        # newcollection._spatial_filter=geometry_filter
        # newcollection._filtered =True
        newcollection._ras_coll_engine_obj._df = newcollection._as_df()
        return newcollection

    def filter_by_time(
        self,
        start_time="",
        end_time="",
        time_field_name="StdTime",
        date_time_format=None,
        context=None,
    ):
        if time_field_name not in self.fields:
            raise ValueError(
                "the time_field_name is not existed. Please input a valid name for time field."
            )

        sql_query1 = (
            time_field_name + " >= timestamp '" + start_time + "'" if start_time else ""
        )
        sql_query2 = (
            time_field_name + " <= timestamp '" + end_time + "'" if end_time else ""
        )

        if sql_query1 and sql_query2:
            sql_query = sql_query1 + " AND " + sql_query2
        else:
            sql_query = sql_query1 if sql_query1 else sql_query2

        return self.filter_by(sql_query, context=context)

    def filter_by_calendar_range(
        self,
        calendar_field,
        start,
        end=None,
        time_field_name="StdTime",
        date_time_format=None,
        context=None,
    ):
        """
        filter the raster collection by a calendar_field and its start and end value (inclusive). i.e. if you would like
        to select all the rasters that have the time stamp on Monday, specify calendar_field as 'DAY_OF_WEEK' and put start and
        end to 1.

        :param calendar_field: string, one of 'YEAR', 'MONTH', 'QUARTER', 'WEEK_OF_YEAR', 'DAY_OF_YEAR', 'DAY_OF_MONTH',
         'DAY_OF_WEEK', 'HOUR'
        :param start: integer, the start time. inclusive.
        :param end: integer, default is None, if default is used, the end is set equal to start. inclusive.
        :param time_field_name: string, the time field anme, default is 'StdTime'.
        :param date_time_format: the time format that is used to format the time field values. Please ref the python
                                date time standard for this argument. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
                                Default is None and this means using the Pro standard time format '%Y-%m-%dT%H:%M:%S'
                                and ignoring the following sub-second.

        :return: a filtered raster collection.
        """
        # validation
        if context is None:
            context = self._context

        calendar_field_types = [
            "YEAR",
            "MONTH",
            "QUARTER",
            "WEEK_OF_YEAR",
            "DAY_OF_YEAR",
            "DAY_OF_MONTH",
            "DAY_OF_WEEK",
            "HOUR",
        ]
        if not isinstance(calendar_field, str):
            raise TypeError("calender_field must be string type")

        calendar_field = calendar_field.upper()

        if calendar_field not in calendar_field_types:
            raise ValueError(
                "invalid calender_field, must be one of "
                + ", ".join(calendar_field_types)
            )

        if time_field_name not in self.fields:
            raise ValueError(
                "the time_field_name does not exist. Please input a valid name for time field."
            )

        if end is None:
            end = start
        # validate start and end type
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("start and end must be numeric.")
        if end < start:
            raise ValueError("end must be equal or larger than start.")
        # validate start and end value
        if calendar_field == "MONTH":
            if start < 1 or start > 12 or end < 1 or end > 12:
                raise ValueError(
                    "start and end must be between [1, 12] for MONTH filter."
                )
        elif calendar_field == "QUARTER":
            if start < 1 or start > 4 or end < 1 or end > 4:
                raise ValueError(
                    "start and end must be between [1, 4] for QUARTER filter."
                )
        elif calendar_field == "WEEK_OF_YEAR":
            if start < 1 or start > 53 or end < 1 or end > 53:
                raise ValueError(
                    "start and end must be between [1, 53] for WEEK_OF_YEAR filter."
                )
        elif calendar_field == "DAY_OF_YEAR":
            if start < 1 or start > 366 or end < 1 or end > 366:
                raise ValueError(
                    "start and end must be between [1, 366] for DAY_OF_YEAR filter."
                )
        elif calendar_field == "DAY_OF_MONTH":
            if start < 1 or start > 31 or end < 1 or end > 31:
                raise ValueError(
                    "start and end must be between [1, 31] for DAY_OF_MONTH filter."
                )
        elif calendar_field == "DAY_OF_WEEK":
            if start < 1 or start > 7 or end < 1 or end > 7:
                raise ValueError(
                    "start and end must be between [1, 7] for DAY_OF_WEEK filter."
                )
        elif calendar_field == "HOUR":
            if start < 1 or start > 24 or end < 1 or end > 24:
                raise ValueError(
                    "start and end must be between [1, 24] for HOUR filter."
                )

        oids_dict = super().query(
            where=self._where_clause,
            geometry_filter=self._spatial_filter,
            return_ids_only=True,
            raster_query=self._raster_query,
        )
        oid_name = oids_dict["objectIdFieldName"]
        oids = oids_dict["objectIds"]
        df = self._as_df()
        filtered_rasters_oids = []
        newcollection = self._clone_raster_collection(context=context)
        for index, row in df.iterrows():
            if date_time_format is None:
                date_time = datetime.datetime.strptime(
                    row[time_field_name], "%Y-%m-%dT%H:%M:%S"
                )
            else:
                date_time = datetime.datetime.strptime(
                    row[time_field_name], date_time_format
                )

            selected = False
            if calendar_field == "YEAR":
                if start <= date_time.year <= end:
                    selected = True
            elif calendar_field == "MONTH":
                if start <= date_time.month <= end:
                    selected = True
            elif calendar_field == "QUARTER":
                if start - 1 <= (date_time.month - 1) // 3 <= end - 1:
                    selected = True
            elif calendar_field == "WEEK_OF_YEAR":
                week_number = int(date_time.strftime("%U"))
                if start <= week_number + 1 <= end:
                    selected = True
            elif calendar_field == "DAY_OF_YEAR":
                yday = date_time.timetuple().tm_yday
                if start <= yday <= end:
                    selected = True
            elif calendar_field == "DAY_OF_MONTH":
                mday = date_time.timetuple().tm_mday
                if start <= mday <= end:
                    selected = True
            elif calendar_field == "DAY_OF_WEEK":
                wday = date_time.isoweekday()
                if start <= wday % 7 + 1 <= end:
                    selected = True
            elif calendar_field == "HOUR":
                hour = date_time.timetuple().tm_hour
                if start <= hour <= end:
                    selected = True

            if selected is True:
                filtered_rasters_oids.append(row[oid_name])

        where_clause = ""
        for ele in filtered_rasters_oids:
            where_clause += oid_name + " = " + str(ele) + " OR "
        newcollection._ras_coll_engine_obj._where_clause = where_clause[:-4]
        newcollection._ras_coll_engine_obj._mosaic_rule = {
            "mosaicMethod": "esriMosaicLockRaster",
            "lockRasterIds": filtered_rasters_oids,
            "ascending": True,
            "mosaicOperation": "MT_FIRST",
        }
        newcollection._ras_coll_engine_obj._filtered = True
        newcollection._ras_coll_engine_obj._df = newcollection._as_df()
        return newcollection

    def filter_by_geometry(self, query_geometry_or_extent, context=None):
        return self.filter_by(
            query_geometry_or_extent=query_geometry_or_extent,
            context=context,
        )

    def filter_by_attribute(self, field_name, operator, field_values, context=None):
        if not isinstance(field_name, str):
            raise TypeError("field_name should be string")

        from arcgis.raster._util import build_query_string

        query_string = build_query_string(field_name, operator.lower(), field_values)
        return self.filter_by(query_string, context=context)

    def filter_by_raster_property(
        self, property_name, operator, property_values, context=None
    ):
        if not isinstance(property_name, str):
            raise TypeError("property_name should be string")
        from arcgis.raster._util import build_query_string

        raster_query = build_query_string(
            property_name, operator.lower(), property_values
        )
        return self.filter_by(raster_query=raster_query, context=context)

    def sort(self, field_name, ascending=True, context=None):
        if context is None:
            context = self._context
        if ascending is True:
            order_by_fields_string = str(field_name) + " " + "ASC"
        else:
            order_by_fields_string = str(field_name) + " " + "DESC"
        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._order_by_fields = order_by_fields_string
        newcollection._ras_coll_engine_obj._df = newcollection._as_df()

        return newcollection

    def get_field_values(self, field_name, max_count=0):
        # if max_count ==0:
        #    max_count = self.count

        df = self._df
        if max_count != 0:
            return df[field_name].tolist()[0:max_count]
        else:
            return df[field_name].tolist()

    def to_multidimensional_raster(self, variable_field_name, dimension_field_names):
        md_info = super()._compute_multidimensional_info(
            where=self._where_clause,
            geometry_filter=self._spatial_filter,
            object_ids=self._object_ids,
            raster_query=self._raster_query,
            variable_field_name=variable_field_name,
            dimension_field_names=dimension_field_names,
        )

        from arcgis.raster.functions import _simple_collection

        lyr = _simple_collection(self, md_info)
        return lyr

    def max(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import max

        return max(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def min(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import min

        return min(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def median(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import med

        return med(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def mean(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import mean

        return mean(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def majority(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import majority

        return majority(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def sum(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import sum

        return sum(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def std(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import std

        return std(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def mosaic(self, mosaic_method):
        from arcgis.raster.functions import merge_rasters

        ras = merge_rasters(
            rasters=self._rasters_list, resolve_overlap_method=mosaic_method
        )
        return ras

    def quality_mosaic(self, quality_rc_or_list, statistic_type=None):
        from arcgis.raster.functions import arg_statistics, _pick

        if not isinstance(statistic_type, str) or statistic_type.upper() not in [
            "MAX",
            "MIN",
            "MEDIAN",
        ]:
            raise ValueError("invalid statistic_type value")

        statistic_type = statistic_type.upper()

        # statistics_type_code = {'MAX': 0, 'MIN': 1, 'MEDIAN': 2}

        if isinstance(quality_rc_or_list, RasterCollection):
            if quality_rc_or_list.count != self.count:
                raise ValueError(
                    "the quality collection must have same number of items with the calling collection"
                )

            if "Raster" in quality_rc_or_list.fields:
                rasters = quality_rc_or_list.get_field_values("Raster")
            # elif 'Path' in quality_rc_or_list.fields:
            # rasters = [arcpy.Raster(path) for path in quality_rc_or_list.getFieldValues('Path')]
        elif isinstance(quality_rc_or_list, list):
            if len(quality_rc_or_list) != self.count:
                raise ValueError(
                    "the quality raster list must have same number of items with the calling collection"
                )
            rasters = quality_rc_or_list
        else:
            raise ValueError("invalid quality_rc parameter")

        arg_statistics_result = arg_statistics(rasters, stat_type=statistic_type)
        arg_statistics_result = (
            arg_statistics_result + 1
        )  # the index in argstatistics output counts from 0, but Pick counts from 1

        if "Raster" in self.fields:
            rasters = self.get_field_values("Raster")
        elif "Path" in self.fields:
            rasters = self.get_field_values("Path")
        else:
            raise ValueError("invalid raster collection")

        inp_list = [arg_statistics_result]
        for ele in rasters:
            inp_list.append(ele)
        return _pick(inp_list)

    def select_bands(self, band_ids_or_names, context=None):
        if context is None:
            context = self._context
        from arcgis.raster.functions import (
            raster_collection_function,
            extract_band,
        )

        by_bandID_or_bandName = 0  # 1: by band id; 2: by band name
        if not (
            isinstance(band_ids_or_names, list)
            or isinstance(band_ids_or_names, str)
            or isinstance(band_ids_or_names, int)
        ):
            raise TypeError(
                "bands must be either a list, a single band ID or a single band Name"
            )
        if isinstance(band_ids_or_names, list):
            for band in band_ids_or_names:
                if not (isinstance(band, int) or isinstance(band, str)):
                    raise TypeError(
                        "elements in band_ids_or_names must be integer or string type"
                    )
                if isinstance(band, int) and by_bandID_or_bandName == 0:
                    by_bandID_or_bandName = 1
                elif isinstance(band, int) and by_bandID_or_bandName == 2:
                    raise TypeError(
                        "elements in band_ids_or_names should either all be integer or string type"
                    )
                if isinstance(band, str) and by_bandID_or_bandName == 0:
                    by_bandID_or_bandName = 2
                elif isinstance(band, str) and by_bandID_or_bandName == 1:
                    raise TypeError(
                        "elements in band_ids_or_names should either all be integer or string type"
                    )

        elif isinstance(band_ids_or_names, str):
            by_bandID_or_bandName = 2
        else:
            by_bandID_or_bandName = 1

        rasters = self.get_field_values("Raster")
        from arcgis.raster.functions import extract_band

        if by_bandID_or_bandName == 1:
            new_rasters = [
                extract_band(raster, band_ids=band_ids_or_names) for raster in rasters
            ]
        elif by_bandID_or_bandName == 2:
            new_rasters = [
                extract_band(raster, band_names=band_ids_or_names) for raster in rasters
            ]

        in_raster_collection_dict = {}
        for field_name in self.fields:
            if field_name == "Raster" or field_name == "Path":
                continue
            in_raster_collection_dict[field_name] = self.get_field_values(field_name)

        return RasterCollection(new_rasters, in_raster_collection_dict, context=context)

    def map(self, func, context=None):
        if context is None:
            context = self._context
        if isinstance(func, _arcgis.raster.functions.RFT) or isinstance(func, dict):
            from arcgis.raster.functions import raster_collection_function

            layer = raster_collection_function(self, item_function=func)

            newcollection = self._clone_raster_collection(context=context)
            newcollection._fn = layer._fn
            newcollection._fnra = layer._fnra
            return newcollection
        else:
            res = map(func, iter(self))
            rasters = []
            attribute_dict = defaultdict(list)
            for item in res:
                rasters.append(item["raster"])
                for key, value in item.items():
                    if key != "raster":
                        attribute_dict[key].append(value)

            return RasterCollection(rasters, attribute_dict, context=context)

    def reduce(self, func, func_args={}):
        rasters = self._rasters_list
        reduced_raster = func(rasters, **func_args)
        return reduced_raster

    def merge(self, collection2):
        import pandas as pd

        rc1 = self._as_df()
        rc2 = collection2._as_df()

        merged_collection = pd.concat([rc1, rc2], axis=0)
        merged_collection.reset_index(drop=True, inplace=True)

        return RasterCollection(merged_collection)

    def add_field(self, field_name, field_values, context=None):
        """
         Adds a new field to the raster collection and populate it with values.
        :param field_name: Required string. The name of the field to be added.
        :param field_values: Required list. The list of values associated with the field name.
                             The length of the list should match the number of items in the raster collection
                             Providing only one value will set the same value for all rows.
        :return: Collection that has the new field added.
        """
        if context is None:
            context = self._context

        if field_name in self.fields:
            raise RuntimeError("Cannot add the field. The field name already exists.")

        df = self._as_df()
        new_df = df.copy()

        if not isinstance(field_values, list):
            field_values = [field_values]

        if self.count != len(field_values):
            if len(field_values) == 1:
                field_values = field_values * self.count
            else:
                raise RuntimeError(
                    "Length of field_values does not match the raster collection count"
                )

        try:
            new_df[field_name] = field_values
        except:
            raise RuntimeError("Failed to add the field to the raster collection")

        return RasterCollection(new_df, context=context)

    def group_by(self, field_name, context=None):
        """
         group_by method can be used to group the raster collection based on a field.
        :param field_name: Required string.The name of the field that is used to group the raster collection. Items with the same field values will be grouped together.
        :return: Dictionary.The dictionary that contains the grouped raster collections. The key of the dictionary is a field value of the field name that the grouping is based on.
                 The value of the dictionary is a raster collection whose field name contains the same field value.
        """
        if context is None:
            context = self._context

        df = self._as_df()
        try:
            group_by_obj = df.groupby(field_name)
            groups = {}

            for key, val in group_by_obj.groups.items():
                groups.update(
                    {
                        key: RasterCollection(
                            group_by_obj.get_group(key), context=context
                        )
                    }
                )
            return groups
        except:
            raise RuntimeError("groupBy failed with the field_name - " + field_name)

    def _generate_raster_item_rft(self, raster_id):
        template_dict = {
            "rasterFunction": "RasterItem",
            "rasterFunctionArguments": {},
        }

        if self is not None and isinstance(self, ImageryLayer):
            url = self.url

            try:
                if (
                    (hasattr(self, "_lazy_token")) and self._lazy_token is None
                ) or not hasattr(self, "_lazy_token"):
                    from .functions.utility import _generate_layer_token

                    self._lazy_token = _generate_layer_token(self, url)
                if isinstance(self._lazy_token, str):
                    url = url + "?token=" + self._lazy_token
                    token = self._lazy_token
            except:
                url = self.url

            template_dict["rasterFunctionArguments"]["URL"] = url
            template_dict["rasterFunctionArguments"]["Raster"] = url

        if self is not None and isinstance(self, str):
            template_dict["rasterFunctionArguments"]["URL"] = self

        if raster_id is not None:
            template_dict["rasterFunctionArguments"]["RasterID"] = raster_id
        return template_dict

    def _as_df(
        self,
        result_offset=None,
        result_record_count=None,
        return_all_records=False,
    ):
        import pandas as pd

        df = super().query(
            where=self._where_clause,
            geometry_filter=self._spatial_filter,
            object_ids=self._object_ids,
            order_by_fields=self._order_by_fields,
            return_all_records=return_all_records,
            result_offset=result_offset,
            result_record_count=result_record_count,
            return_geometry=True,
            as_df=True,
            raster_query=self._raster_query,
        )
        date_field_names = []
        if len(df.index) > 0:
            for ele in self.properties.fields:
                if ("type" in ele.keys()) and ele["type"] == "esriFieldTypeDate":
                    if "name" in ele.keys():
                        date_field_names.append(ele["name"])
            for ele in date_field_names:
                df[ele] = pd.to_datetime(df[ele], unit="ms").dt.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )

            # pos = self.url.find("/ImageServer",0)
            # for i in range (0, len(df.index)):
            # df.loc[i, "Raster"] = self.url[0:(pos+12)]+"/"+str(df["OBJECTID"].loc[i])+self.url[(pos+12):]

            from arcgis.raster.functions import _raster_item

            oid_name = self._object_id_name()
            for i in range(0, len(df.index)):
                # self._do_not_hydrate=True
                rft = self._generate_raster_item_rft(int(df[oid_name].loc[i]))
                df.loc[i, "Raster"] = Raster(rft, gis=self._gis)
                df.loc[i, "Raster"]._engine_obj._fn = rft
                df.loc[i, "Raster"]._engine_obj._fnra = rft
                df.loc[i, "Raster"]._engine_obj._do_not_hydrate = True
                df.loc[i, "Raster"]._engine_obj._tiles_only = False
                df.loc[i, "Raster"]._engine_obj._lazy_token = self._lazy_token

        return df

    def display_image(self, item=None):
        if item is not None:
            newcollection = self._clone_raster_collection()
            newcollection._mosaic_rule = {
                "mosaicMethod": "esriMosaicLockRaster",
                "lockRasterIds": [item],
                "ascending": True,
                "mosaicOperation": "MT_FIRST",
            }
        else:
            newcollection = self

        bbox_sr = None
        if "spatialReference" in newcollection.extent:
            bbox_sr = newcollection.extent["spatialReference"]
        if not newcollection._uses_gbl_function:
            byte_array = newcollection.export_image(
                bbox=newcollection._extent,
                bbox_sr=bbox_sr,
                size=[1200, 450],
                export_format="jpeg",
                f="image",
            )
            try:
                from IPython.display import Image

                return Image(byte_array)
            except:
                return byte_array

    def _clone_raster_collection(self, context=None):
        new_raster_collection = RasterCollection(
            self._url,
            self._attribute_dict,
            self._where_clause,
            self._spatial_filter,
            self._ras_coll_engine,
            self._gis,
            context=context,
        )

        new_raster_collection._lazy_token = self._token
        new_raster_collection._fn = self._fn
        new_raster_collection._fnra = self._fnra
        new_raster_collection._mosaic_rule = self._mosaic_rule
        new_raster_collection._extent = self._extent
        new_raster_collection._raster_query = self._raster_query

        return new_raster_collection

    def _object_id_name(self):
        for ele in self.properties.fields:
            if ("type" in ele.keys()) and ele["type"] == "esriFieldTypeOID":
                return ele["name"]

    def _repr_html_(self):
        return self._df._repr_html_()

    def _repr_jpeg_(self):
        return None


def _get_shape(ele):
    boundary = Geometry(ele._engine_obj.query_boundary()["shape"])
    if isinstance(boundary, Envelope):
        boundary = boundary.polygon
    return boundary


class _LocalRasterCollection(ImageryLayer, RasterCollection):
    def __init__(
        self,
        rasters=None,
        attribute_dict=None,
        where_clause=None,
        query_geometry=None,
        engine=None,
        gis=None,
        context=None,
    ):
        # self._remote = raster.use_server_engine
        ImageryLayer.__init__(self, rasters, gis=gis)
        self._mosaic_rule = None
        if gis is not None:
            self._gis = gis
        else:
            self._gis = None
        if where_clause is None:
            self._where_clause = "1=1"
        else:
            self._where_clause = where_clause
        if query_geometry is None:
            self._spatial_filter = None
        else:
            self._spatial_filter = query_geometry

        # self._attribute_dict = attribute_dict
        self._rasters = rasters
        self._object_ids = None
        self._order_by_fields = None
        self._is_multidimensional = False
        # self._do_not_hydrate=False
        self._engine = _ImageServerRaster
        self._engine_obj = _ImageServerRaster(rasters, False, gis)

        self._ras_coll_engine = engine
        if context is None:
            self._context = {}
        else:
            self._context = context
        # self._engine=_ImageServerRaster
        # self._engine_obj=_ImageServerRaster(rasters, False, gis)
        ids = list(range(1, len(rasters) + 1))
        arcgis_rasters = []
        if attribute_dict is not None:
            if not isinstance(attribute_dict, dict):
                raise TypeError("attribute_dict must be dict type")

        shapes = []
        for ele in rasters:
            if not isinstance(ele, Raster):
                ele = Raster(ele)
            ele._engine_obj._do_not_hydrate = True
            # ele._engine_obj.token = self._token
            arcgis_rasters.append(ele)
        data = {"Raster": arcgis_rasters, "ID": ids}

        query_boundary = True
        if isinstance(self._context, dict):
            if "query_boundary" in self._context.keys():
                if isinstance(self._context["query_boundary"], bool):
                    query_boundary = self._context["query_boundary"]
                else:
                    _LOGGER.warning(
                        "Value for query_boundary key in context should be bool"
                    )

        if query_boundary:
            if "SHAPE" not in attribute_dict.keys():
                if not isinstance(ele._engine_obj, _ArcpyRaster):
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=len(rasters)
                    ) as executor:
                        future_to_url = (
                            executor.submit(_get_shape, ele) for ele in arcgis_rasters
                        )
                        for future in concurrent.futures.as_completed(future_to_url):
                            shapes.append(future.result())

                if shapes != []:
                    data.update({"SHAPE": shapes})
        if attribute_dict is not None:
            data.update(attribute_dict)
        self._attribute_dict = data
        import pandas as pd

        df_obj = pd.DataFrame(data)
        self._df = df_obj

    @property
    def count(self):
        count = len(self._df.index)
        return count

    @property
    def fields(self):
        return tuple(self._df.columns.tolist())

    @property
    def _rasters_list(self):
        ras_list = self.get_field_values("Raster")
        return ras_list

    def __iter__(self):
        return iter(self._df.to_dict("records", into=dict))

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        return self._df.to_dict("records", into=dict)[item]

    def filter_by(
        self,
        where_clause=None,
        query_geometry_or_extent=None,
        raster_query=None,
        context=None,
    ):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        if where_clause is not None:
            if "LIKE" in where_clause or "NOT LIKE" in where_clause:
                raise RuntimeError(
                    "Local RasterCollection does not support LIKE or NOT LIKE in where_clause."
                )
            newcollection._ras_coll_engine_obj._df = self._df.query(where_clause)
        if query_geometry_or_extent is not None:
            newcollection._ras_coll_engine_obj._df = (
                newcollection._ras_coll_engine_obj.filter_by_geometry(
                    query_geometry_or_extent
                )._as_df()
            )
        if raster_query is not None:
            props_list = []
            for ele in self:
                props_list.append(ele["Raster"].properties)
            import pandas as pd

            new_pd = pd.DataFrame.from_dict(props_list)
            df = self._as_df()
            frames = [df, new_pd]
            result = pd.concat(frames, axis=1)
            result.query(raster_query)
            # newcollection=self.filter_by_attribute("ID", "contains", list(result['ID']) )
            filtered_rasters = []
            attribute_dict = defaultdict(list)

            for item in iter(self):
                raster = self._get_raster_from_item(item)

                field_value = item["ID"]
                selected = False
                selected = field_value in list(result["ID"])
                if selected:
                    filtered_rasters.append(raster)
                    for key, value in item.items():
                        if key not in ["Raster", "Path"]:
                            attribute_dict[key].append(value)
            newcollection = (
                RasterCollection(filtered_rasters, attribute_dict, context=context)
                if filtered_rasters
                else None
            )
        return newcollection

    def filter_by_time(
        self,
        start_time="",
        end_time="",
        time_field_name="StdTime",
        date_time_format=None,
        context=None,
    ):
        if context is None:
            context = self._context
        if date_time_format is None:
            if start_time:
                start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
            if end_time:
                end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
        else:
            if start_time:
                start_time = datetime.datetime.strptime(start_time, date_time_format)
            if end_time:
                end_time = datetime.datetime.strptime(end_time, date_time_format)

        filtered_rasters = []
        attribute_dict = defaultdict(list)

        for item in iter(self):
            raster = self._get_raster_from_item(item)

            if date_time_format is None:
                date_time = datetime.datetime.strptime(
                    item[time_field_name], "%Y-%m-%dT%H:%M:%S"
                )
            else:
                date_time = datetime.datetime.strptime(
                    item[time_field_name], date_time_format
                )

            if start_time and date_time < start_time:
                continue
            elif end_time and date_time > end_time:
                continue
            else:
                filtered_rasters.append(raster)
                for key, value in item.items():
                    if key not in ["Raster", "Path"]:
                        attribute_dict[key].append(value)

        if not filtered_rasters:
            warnings.warn(
                "Warning: the output is None because no items have raster properties satisfying the query"
            )

        return (
            RasterCollection(filtered_rasters, attribute_dict, context=context)
            if filtered_rasters
            else None
        )

    def filter_by_calendar_range(
        self,
        calendar_field,
        start,
        end=None,
        time_field_name="StdTime",
        date_time_format=None,
        context=None,
    ):
        """
        filter the raster collection by a calendar_field and its start and end value (inclusive). i.e. if you would like
        to select all the rasters that have the time stamp on Monday, specify calendar_field as 'DAY_OF_WEEK' and put start and
        end to 1.

        :param calendar_field: string, one of 'YEAR', 'MONTH', 'QUARTER', 'WEEK_OF_YEAR', 'DAY_OF_YEAR', 'DAY_OF_MONTH',
         'DAY_OF_WEEK', 'HOUR'
        :param start: integer, the start time. inclusive.
        :param end: integer, default is None, if default is used, the end is set equal to start. inclusive.
        :param time_field_name: string, the time field anme, default is 'StdTime'.
        :param date_time_format: the time format that is used to format the time field values. Please ref the python
                                date time standard for this argument. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
                                Default is None and this means using the Pro standard time format '%Y-%m-%dT%H:%M:%S'
                                and ignoring the following sub-second.

        :return: a filtered raster collection.
        """
        # validation

        if context is None:
            context = self._context
        filtered_rasters = []
        attribute_dict = defaultdict(list)

        calendar_field_types = [
            "YEAR",
            "MONTH",
            "QUARTER",
            "WEEK_OF_YEAR",
            "DAY_OF_YEAR",
            "DAY_OF_MONTH",
            "DAY_OF_WEEK",
            "HOUR",
        ]
        if not isinstance(calendar_field, str):
            raise TypeError("calender_field must be string type")

        calendar_field = calendar_field.upper()

        if calendar_field not in calendar_field_types:
            raise ValueError(
                "invalid calender_field, must be one of "
                + ", ".join(calendar_field_types)
            )

        for item in iter(self):
            raster = self._get_raster_from_item(item)

            if date_time_format is None:
                date_time = datetime.datetime.strptime(
                    item[time_field_name], "%Y-%m-%dT%H:%M:%S"
                )
            else:
                date_time = datetime.datetime.strptime(
                    item[time_field_name], date_time_format
                )

            selected = False

            if calendar_field == "YEAR":
                if start <= date_time.year <= end:
                    selected = True
            elif calendar_field == "MONTH":
                if start <= date_time.month <= end:
                    selected = True
            elif calendar_field == "QUARTER":
                if start - 1 <= (date_time.month - 1) // 3 <= end - 1:
                    selected = True
            elif calendar_field == "WEEK_OF_YEAR":
                week_number = int(date_time.strftime("%U"))
                if start <= week_number + 1 <= end:
                    selected = True
            elif calendar_field == "DAY_OF_YEAR":
                yday = date_time.timetuple().tm_yday
                if start <= yday <= end:
                    selected = True
            elif calendar_field == "DAY_OF_MONTH":
                mday = date_time.timetuple().tm_mday
                if start <= mday <= end:
                    selected = True
            elif calendar_field == "DAY_OF_WEEK":
                wday = date_time.isoweekday()
                if start <= wday % 7 + 1 <= end:
                    selected = True
            elif calendar_field == "HOUR":
                hour = date_time.timetuple().tm_hour
                if start <= hour <= end:
                    selected = True

            if selected is True:
                filtered_rasters.append(raster)
                for key, value in item.items():
                    if key not in ["Raster", "Path"]:
                        attribute_dict[key].append(value)

        if not filtered_rasters:
            warnings.warn(
                "Warning: the output is None because no items have raster properties satisfying the query"
            )

        return (
            RasterCollection(filtered_rasters, attribute_dict, context=context)
            if filtered_rasters
            else None
        )

    def filter_by_geometry(self, query_geometry_or_extent, context=None):
        if context is None:
            context = self._context
        filtered_rasters = []
        attribute_dict = defaultdict(list)

        for item in iter(self):
            raster = self._get_raster_from_item(item)

            field_value = item["SHAPE"]
            if query_geometry_or_extent is not None:
                query_geometry_or_extent = _get_geometry(query_geometry_or_extent)
            selected = False
            selected = query_geometry_or_extent.contains(field_value, "BOUNDARY")

            if selected and isinstance(selected, bool):
                filtered_rasters.append(raster)
                for key, value in item.items():
                    if key not in ["Raster", "Path"]:
                        attribute_dict[key].append(value)

        if not filtered_rasters:
            warnings.warn(
                "Warning: the output is None because no items have raster properties satisfying the query"
            )
        return (
            RasterCollection(filtered_rasters, attribute_dict, context=context)
            if filtered_rasters
            else None
        )

    def filter_by_raster_property(
        self, property_name, operator, property_values, context=None
    ):
        if context is None:
            context = self._context
        values = property_values
        if not isinstance(values, str) and not isinstance(values, list):
            raise TypeError("values must be string of list of string")

        if isinstance(values, list):
            for v in values:
                if not isinstance(v, str):
                    raise TypeError("values must be string of list of string")

        filtered_rasters = []
        attribute_dict = defaultdict(list)

        for item in iter(self):
            raster = self._get_raster_from_item(item)

            raster_properties = raster.properties
            if property_name not in raster_properties:
                continue
            property_value = raster.get_property(property_name)

            operator = operator.lower()
            selected = False
            if operator == "equals":
                selected = property_value == values
            elif operator == "less_than":
                selected = property_value < values
            elif operator == "greater_than":
                selected = property_value > values
            elif operator == "not_equals":
                selected = property_value != values
            elif operator == "not_less_than":
                selected = property_value >= values
            elif operator == "not_greater_than":
                selected = property_value <= values
            elif operator == "starts_with":
                selected = property_value.startswith(values)
            elif operator == "ends_with":
                selected = property_value.endswith(values)
            elif operator == "not_starts_with":
                selected = not property_value.startswith(values)
            elif operator == "not_ends_with":
                selected = not property_value.endswith(values)
            elif operator == "contains":
                selected = values in property_value
            elif operator == "not_contains":
                selected = values not in property_value
            elif operator == "in":
                if not isinstance(values, list):
                    raise TypeError(
                        "Invalid values. Notet that values must be a list when operator is 'in'"
                    )
                selected = property_value in values
            elif operator == "not_in":
                if not isinstance(values, list):
                    raise TypeError(
                        "Invalid values. Notet that values must be a list when operator is 'not_in'"
                    )
                selected = property_value not in values
            else:
                raise ValueError("invalid operator")

            if selected:
                filtered_rasters.append(raster)
                for key, value in item.items():
                    if key not in ["Raster", "Path"]:
                        attribute_dict[key].append(value)

        if not filtered_rasters:
            warnings.warn(
                "Warning: the output is None because no items have raster properties satisfying the query"
            )
        return (
            RasterCollection(filtered_rasters, attribute_dict, context=context)
            if filtered_rasters
            else None
        )

    def filter_by_attribute(self, field_name, operator, field_values, context=None):
        if context is None:
            context = self._context
        filtered_rasters = []
        attribute_dict = defaultdict(list)

        for item in iter(self):
            raster = self._get_raster_from_item(item)

            field_value = item[field_name]

            operator = operator.lower()
            selected = False
            if operator == "equals":
                selected = field_value == field_values
            elif operator == "less_than":
                selected = field_value < field_values
            elif operator == "greater_than":
                selected = field_value > field_values
            elif operator == "not_equals":
                selected = field_value != field_values
            elif operator == "not_less_than":
                selected = field_value >= field_values
            elif operator == "not_greater_than":
                selected = field_value <= field_values
            elif operator == "starts_with":
                selected = field_value.startswith(field_values)
            elif operator == "ends_with":
                selected = field_value.endswith(field_values)
            elif operator == "not_starts_with":
                selected = not field_value.startswith(field_values)
            elif operator == "not_ends_with":
                selected = not field_value.endswith(field_values)
            elif operator == "contains":
                selected = field_values in field_value
            elif operator == "not_contains":
                selected = field_values not in field_value
            elif operator == "in":
                if not isinstance(field_values, list):
                    raise TypeError(
                        "Invalid field_values. Notet that field_values must be a list when operator is 'in'"
                    )
                selected = field_value in field_values
            elif operator == "not_in":
                if not isinstance(field_values, list):
                    raise TypeError(
                        "Invalid field_values. Notet that field_values must be a list when operator is 'not_in'"
                    )
                selected = field_value not in field_values
            else:
                raise ValueError("invalid operator")

            if selected:
                filtered_rasters.append(raster)
                for key, value in item.items():
                    if key not in ["Raster", "Path"]:
                        attribute_dict[key].append(value)

        if not filtered_rasters:
            warnings.warn(
                "Warning: the output is None because no items have raster properties satisfying the query"
            )
        return (
            RasterCollection(filtered_rasters, attribute_dict, context=context)
            if filtered_rasters
            else None
        )

    def sort(self, field_name, ascending=True, context=None):
        if context is None:
            context = self._context
        newcollection = self._clone_raster_collection(context=context)
        newcollection._ras_coll_engine_obj._df = self._df.sort_values(
            by=field_name, ascending=ascending
        )

        return newcollection

    def get_field_values(self, field_name, max_count=0):
        if max_count == 0:
            max_count = self.count

        df = self._df
        return df[field_name].tolist()[0:max_count]

    def to_multidimensional_raster(self, variable_field_name, dimension_field_names):
        if variable_field_name not in self.fields:
            raise ValueError("variable_field_name does not exist")

        if isinstance(dimension_field_names, list):
            for dimension_field_name in dimension_field_names:
                if dimension_field_name not in self.fields:
                    raise ValueError("the given dimension_field_name does not exist")
            dimension_field_names = ",".join(dimension_field_names)
        else:
            if dimension_field_names not in self.fields:
                raise ValueError("the given dimension_field_names does not exist")
        from arcgis.raster.functions import _simple_collection

        lyr = _simple_collection(self._rasters_list)
        rasters = lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"]
        lyr._engine_obj._fn["rasterFunctionArguments"]["Raster"] = {}
        lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"] = {}

        lyr._engine_obj._fn["rasterFunctionArguments"]["Raster"].update(
            {"rasters": rasters}
        )
        lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"].update(
            {"rasters": rasters}
        )
        for ele_field in self.fields:
            if ele_field == "Raster":
                continue
            lyr._engine_obj._fn["rasterFunctionArguments"]["Raster"].update(
                {ele_field: []}
            )
            lyr._engine_obj._fn["rasterFunctionArguments"]["Raster"].update(
                {"variable": variable_field_name}
            )
            lyr._engine_obj._fn["rasterFunctionArguments"]["Raster"].update(
                {"dimensions": dimension_field_names}
            )
            lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"].update(
                {ele_field: []}
            )
            lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"].update(
                {"variable": variable_field_name}
            )
            lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"].update(
                {"dimensions": dimension_field_names}
            )
            for ele in self[0 : self.count]:
                lyr._engine_obj._fn["rasterFunctionArguments"]["Raster"][
                    ele_field
                ].append(ele[ele_field])
                lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"][
                    ele_field
                ].append(ele[ele_field])
        fn = lyr._engine_obj._fn["rasterFunctionArguments"]["Raster"]
        fnra = lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"]
        lyr._engine_obj._fn["rasterFunctionArguments"]["Raster"] = json.dumps(fn)
        lyr._engine_obj._fnra["rasterFunctionArguments"]["Raster"] = json.dumps(fnra)
        return lyr

    def max(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import max

        return max(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def min(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import min

        return min(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def median(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import med

        return med(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def mean(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import mean

        return mean(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def majority(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import majority

        return majority(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def sum(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import sum

        return sum(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def std(
        self,
        ignore_nodata=True,
        extent_type="FirstOf",
        cellsize_type="FirstOf",
    ):
        from arcgis.raster.functions import std

        return std(
            self._rasters_list,
            ignore_nodata=ignore_nodata,
            extent_type=extent_type,
            cellsize_type=cellsize_type,
        )

    def mosaic(self, mosaic_method):
        from arcgis.raster.functions import merge_rasters

        ras = merge_rasters(
            rasters=self._rasters_list, resolve_overlap_method=mosaic_method
        )
        return ras

    def quality_mosaic(self, quality_rc_or_list, statistic_type=None):
        from arcgis.raster.functions import arg_statistics, _pick

        if not isinstance(statistic_type, str) or statistic_type.upper() not in [
            "MAX",
            "MIN",
            "MEDIAN",
        ]:
            raise ValueError("invalid statistic_type value")

        statistic_type = statistic_type.upper()

        # statistics_type_code = {'MAX': 0, 'MIN': 1, 'MEDIAN': 2}

        if isinstance(quality_rc_or_list, RasterCollection):
            if quality_rc_or_list.count != self.count:
                raise ValueError(
                    "the quality collection must have same number of items with the calling collection"
                )

            if "Raster" in quality_rc_or_list.fields:
                rasters = quality_rc_or_list.get_field_values("Raster")
            # elif 'Path' in quality_rc_or_list.fields:
            # rasters = [arcpy.Raster(path) for path in quality_rc_or_list.getFieldValues('Path')]
        elif isinstance(quality_rc_or_list, list):
            if len(quality_rc_or_list) != self.count:
                raise ValueError(
                    "the quality raster list must have same number of items with the calling collection"
                )
            rasters = quality_rc_or_list
        else:
            raise ValueError("invalid quality_rc parameter")

        arg_statistics_result = arg_statistics(rasters, stat_type=statistic_type)
        arg_statistics_result = (
            arg_statistics_result + 1
        )  # the index in argstatistics output counts from 0, but Pick counts from 1

        rasters = self._rasters_list

        inp_list = [arg_statistics_result]
        for ele in rasters:
            inp_list.append(ele)
        return _pick(inp_list)

    def select_bands(self, band_ids_or_names, context=None):
        if context is None:
            context = self._context
        from arcgis.raster.functions import (
            raster_collection_function,
            extract_band,
        )

        by_bandID_or_bandName = 0  # 1: by band id; 2: by band name
        if not (
            isinstance(band_ids_or_names, list)
            or isinstance(band_ids_or_names, str)
            or isinstance(band_ids_or_names, int)
        ):
            raise TypeError(
                "bands must be either a list, a single band ID or a single band Name"
            )
        if isinstance(band_ids_or_names, list):
            for band in band_ids_or_names:
                if not (isinstance(band, int) or isinstance(band, str)):
                    raise TypeError(
                        "elements in band_ids_or_names must be integer or string type"
                    )
                if isinstance(band, int) and by_bandID_or_bandName == 0:
                    by_bandID_or_bandName = 1
                elif isinstance(band, int) and by_bandID_or_bandName == 2:
                    raise TypeError(
                        "elements in band_ids_or_names should either all be integer or string type"
                    )
                if isinstance(band, str) and by_bandID_or_bandName == 0:
                    by_bandID_or_bandName = 2
                elif isinstance(band, str) and by_bandID_or_bandName == 1:
                    raise TypeError(
                        "elements in band_ids_or_names should either all be integer or string type"
                    )

        elif isinstance(band_ids_or_names, str):
            by_bandID_or_bandName = 2
        else:
            by_bandID_or_bandName = 1

        rasters = self._rasters_list

        from arcgis.raster.functions import extract_band

        if by_bandID_or_bandName == 1:
            new_rasters = [
                extract_band(raster, band_ids=band_ids_or_names) for raster in rasters
            ]
        elif by_bandID_or_bandName == 2:
            new_rasters = [
                extract_band(raster, band_names=band_ids_or_names) for raster in rasters
            ]

        in_raster_collection_dict = {}
        for field_name in self.fields:
            if field_name == "Raster" or field_name == "Path":
                continue
            in_raster_collection_dict[field_name] = self.get_field_values(field_name)

        return RasterCollection(new_rasters, in_raster_collection_dict, context=context)

    def map(self, func, context=None):
        if context is None:
            context = self._context
        res = map(func, iter(self))
        rasters = []
        attribute_dict = defaultdict(list)
        for item in res:
            rasters.append(item["raster"])
            for key, value in item.items():
                if key != "raster":
                    attribute_dict[key].append(value)

        return RasterCollection(rasters, attribute_dict, context=context)

    def reduce(self, func, func_args={}):
        rasters = self._rasters_list
        reduced_raster = func(rasters, **func_args)
        return reduced_raster

    def merge(self, collection2):
        import pandas as pd

        rc1 = self._as_df()
        rc2 = collection2._as_df()

        merged_collection = pd.concat([rc1, rc2], axis=0)
        merged_collection.reset_index(drop=True, inplace=True)

        return RasterCollection(merged_collection)

    def add_field(self, field_name, field_values, context=None):
        """
         Adds a new field to the raster collection and populate it with values.
        :param field_name: Required string. The name of the field to be added.
        :param field_values: Required list. The list of values associated with the field name.
                             The length of the list should match the number of items in the raster collection
                             Providing only one value will set the same value for all rows.
        :return: Collection that has the new field added.
        """

        if context is None:
            context = self._context

        if field_name in self.fields:
            raise RuntimeError("Cannot add the field. The field name already exists.")

        df = self._as_df()
        new_df = df.copy()

        if not isinstance(field_values, list):
            field_values = [field_values]

        if self.count != len(field_values):
            if len(field_values) == 1:
                field_values = field_values * self.count
            else:
                raise RuntimeError(
                    "Length of field_values does not match the raster collection count"
                )

        try:
            new_df[field_name] = field_values
        except:
            raise RuntimeError("Failed to add the field to the raster collection")

        return RasterCollection(new_df, context=context)

    def group_by(self, field_name, context=None):
        """
         group_by method can be used to group the raster collection based on a field.
        :param field_name: Required string.The name of the field that is used to group the raster collection. Items with the same field values will be grouped together.
        :return: Dictionary.The dictionary that contains the grouped raster collections. The key of the dictionary is a field value of the field name that the grouping is based on.
                 The value of the dictionary is a raster collection whose field name contains the same field value.
        """
        if context is None:
            context = self._context

        df = self._as_df()
        try:
            group_by_obj = df.groupby(field_name)
            groups = {}

            for key, val in group_by_obj.groups.items():
                groups.update(
                    {
                        key: RasterCollection(
                            group_by_obj.get_group(key), context=context
                        )
                    }
                )
            return groups
        except:
            raise RuntimeError("groupBy failed with the field_name - " + field_name)

    def _as_df(
        self,
        result_offset=None,
        result_record_count=None,
        return_all_records=False,
    ):
        return self._df

    # def display_image(self, item=None):
    #    if not raster.use_server:
    #        raise RuntimeError('Not available in non server env')

    def _get_raster_from_item(self, item):
        if "Raster" in item:
            raster = item["Raster"]
        else:
            raise RuntimeError("invalid raster collection")
        return raster

    def _clone_raster_collection(self, context=None):
        new_raster_collection = RasterCollection(
            self._rasters,
            self._attribute_dict,
            self._where_clause,
            self._spatial_filter,
            self._ras_coll_engine,
            self._gis,
            context=context,
        )
        return new_raster_collection

    def _object_id_name(self):
        for ele in self.properties.fields:
            if ("type" in ele.keys()) and ele["type"] == "esriFieldTypeOID":
                return ele["name"]

    def _repr_html_(self):
        return self._df._repr_html_()

    def _repr_jpeg_(self):
        return None


########################################################################
class ImageryTileManager(object):
    """
    Manages the tiles for Cached Imagery Layers.

    .. note :: This class is not created by users directly. An instance of this class, called
      tiles , is available as a property of an ImageryLayer object. Users call methods on this
      tiles  object to create and access tiles from an ImageryLayer.


    =================     ====================================================================
    **Parameter**          **Description**
    -----------------     --------------------------------------------------------------------
    imglyr                required ImageLayer. The imagery layer object that is cached.
    =================     ====================================================================



    """

    _service = None
    _url = None
    _con = None

    # ----------------------------------------------------------------------
    def __init__(self, imglyr):
        """Constructor"""
        if isinstance(imglyr, ImageryLayer):
            self._service = imglyr
            self._url = imglyr._url
            self._con = imglyr._con
        else:
            raise ValueError("service must be of type ImageLayer")

    def _status(self, url, res):
        """
        checks the status of the service for async operations
        """
        import time

        if "jobId" in res:
            url = url + "/jobs/%s" % res["jobId"]
            while res["jobStatus"] not in (
                "esriJobSucceeded",
                "esriJobFailed",
            ):
                res = self._con.get(path=url, params={"f": "json"})
                if res["jobStatus"] == "esriJobFailed":
                    return False, res
                if res["jobStatus"] == "esriJobSucceeded":
                    return True, res
                time.sleep(2)
        return True, res

    # ----------------------------------------------------------------------
    def export(
        self,
        tile_package: bool = False,
        extent: Optional[str] = None,
        optimize_for_size: bool = True,
        compression: int = 75,
        export_by: str = "LevelID",
        levels: Optional[str] = None,
        aoi: Optional[Polygon] = None,
    ):
        """
        The ``export`` method allows client applications to download map tiles
        from server for offline use.

        .. note::
            This operation is performed on a
            ``Imagery Layer`` that allows clients to export cache tiles. The result
            of this operation is an ``Image Layer Job``.

        ``export`` can be enabled in a layer by using ArcGIS Desktop or the
        ArcGIS Server Administrative Site Directory. In ArcGIS Desktop,
        make an admin or publisher connection to the server, go to layer
        properties and enable "Allow Clients to Export Cache Tiles" in
        advanced caching page of the layer Editor. You can also specify
        the maximum tiles clients will be allowed to download.

        .. note::
            The default maximum allowed tile count is 100,000. To enable this capability
            using the ArcGIS Servers Administrative Site Directory, edit the
            layer and set the properties ``exportTilesAllowed=True`` and
            ``maxExportTilesCount=100000``.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        tile_package          Optional boolean.   Allows exporting either a tile package or a
                              cache raster data set. If the value is true output will be in tile
                              package format and if the value is false Cache Raster data set is
                              returned. The default value is false
        -----------------     --------------------------------------------------------------------
        extent                Optional string. The extent (bounding box) of the tile package or
                              the cache dataset to be exported. If extent does not include a
                              spatial reference, the extent values are assumed to be in the
                              spatial reference of the map. The default value is full extent of
                              the tiled map service.

                              Syntax: <xmin>, <ymin>, <xmax>, <ymax>
                              Example: -104,35.6,-94.32,41
        -----------------     --------------------------------------------------------------------
        optimize_for_size     Optional boolean. Use this parameter to enable compression of JPEG
                              tiles and reduce the size of the downloaded tile package or the
                              cache raster data set. Compressing tiles slightly compromises on the
                              quality of tiles but helps reduce the size of the download. Try out
                              sample compressions to determine the optimal compression before
                              using this feature.
        -----------------     --------------------------------------------------------------------
        compression           Optional integer. When optimizeTilesForSize=true you can specify a
                              compression factor. The value must be between 0 and 100. Default is
                              75.
        -----------------     --------------------------------------------------------------------
        export_by             Optional string. The criteria that will be used to select the tile
                              service levels to export. The values can be Level IDs, cache scales
                              or the Resolution (in the case of image services).
                              Values: LevelID,Resolution,Scale
                              Default: LevelID
        -----------------     --------------------------------------------------------------------
        levels                Optional string. Specify the tiled service levels to export. The
                              values should correspond to Level IDs, cache scales or the
                              Resolution as specified in exportBy parameter. The values can be
                              comma separated values or a range.

                              Example 1: 1,2,3,4,5,6,7,8,9
                              Example 2: 1-4,7-9
        -----------------     --------------------------------------------------------------------
        aoi                   Optional :class:`~arcgis.geometry.Polygon`. The ``areaOfInterest`` Polygon allows
                              exporting tiles within the specified polygon areas.

                              .. note::
                                This parameter supersedes ``extent`` parameter.
        =================     ====================================================================

        :return:
            An ``Image Layer Job``

        .. code-block:: python

            # Example Usage
            exported = img_tile_manager.export(tile_package = True,
                                               levels = "11-20",
                                               extent = {
                                                       "xmin":6224324.092137296,
                                                       "ymin":487347.5253569535,
                                                       "xmax":11473407.698535524,
                                                       "ymax":4239488.369818687
                                                       },
                                               merge = True,
                                               optimize_for_size = True,
                                               export_by = "Scale"
                                               )
        """
        if self._service.properties["exportTilesAllowed"] == False:
            return None

        url = "%s/%s" % (self._url, "exportTiles")
        if export_by is None:
            export_by = "LevelID"
        params = {
            "f": "json",
            "tilePackage": tile_package,
            "exportExtent": extent,
            "optimizeTilesForSize": optimize_for_size,
            "compressionQuality": compression,
            "exportBy": export_by,
            "levels": levels,
        }

        if aoi:
            params["areaOfInterest"] = aoi

        res = self._con.post(path=url, postdata=params, timeout=None)
        sid = res["jobId"]
        success, res = self._status(url, res)
        if success == False:
            return res
        else:
            if "results" in res and "out_service_url" in res["results"]:
                rurl = url + "/jobs/%s/%s" % (
                    sid,
                    res["results"]["out_service_url"]["paramUrl"],
                )
                result_url = self._con.get(path=rurl, params={"f": "json"})["value"]
                dl_res = self._con.get(path=result_url, params={"f": "json"})
                if "files" in dl_res:
                    import tempfile

                    files = []
                    for f in dl_res["files"]:
                        files.append(
                            self._con.get(
                                path=f["url"],
                                try_json=False,
                                out_folder=tempfile.gettempdir(),
                                file_name=f["name"],
                            )
                        )
                        del f
                    return files
                return []
            return res

    # ----------------------------------------------------------------------
    def estimate_size(
        self,
        tile_package: bool = False,
        extent: Optional[str] = None,
        optimize_for_size: bool = True,
        compression: int = 75,
        export_by: str = "LevelID",
        levels: Optional[str] = None,
        aoi: Optional[Polygon] = None,
    ):
        """
        The ``estimate_size`` operation is an asynchronous task that
        allows estimation of the size of the tile package or the cache data
        set that you download using the Export Tiles operation. This
        operation can also be used to estimate the tile count in a tile
        package and determine if it will exceced the maxExportTileCount
        limit set by the administrator of the layer. The result of this
        operation is the response size. This job response contains
        reference to the Image Layer ``Result`` method that returns the total
        size of the cache to be exported (in bytes) and the number of tiles
        that will be exported.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        tile_package          Optional boolean.  If the value is true output will be in tile
                              package format and if the value is false Cache Raster data set is
                              returned. The default value is false
        -----------------     --------------------------------------------------------------------
        extent                Optional string. The extent (bounding box) of the tile package or
                              the cache dataset to be exported. If extent does not include a
                              spatial reference, the extent values are assumed to be in the
                              spatial reference of the map. The default value is full extent of
                              the tiled map service.

                              Syntax: <xmin>, <ymin>, <xmax>, <ymax>
                              Example: -104,35.6,-94.32,41
        -----------------     --------------------------------------------------------------------
        optimize_for_size     Optional boolean. Use this parameter to enable compression of JPEG
                              tiles and reduce the size of the downloaded tile package or the
                              cache raster data set. Compressing tiles slightly compromises on the
                              quality of tiles but helps reduce the size of the download. Try out
                              sample compressions to determine the optimal compression before
                              using this feature.
        -----------------     --------------------------------------------------------------------
        compression           Optional integer. When optimizeTilesForSize=true you can specify a
                              compression factor. The value must be between 0 and 100. Default is
                              75.
        -----------------     --------------------------------------------------------------------
        export_by             Optional string. The criteria that will be used to select the tile
                              service levels to export. The values can be Level IDs, cache scales
                              or the Resolution (in the case of image services).
                              Values: LevelID,Resolution,Scale
                              Default: LevelID
        -----------------     --------------------------------------------------------------------
        levels                Optional string. Specify the tiled service levels to export. The
                              values should correspond to Level IDs, cache scales or the
                              Resolution as specified in exportBy parameter. The values can be
                              comma separated values or a range.

                              Example 1: 1,2,3,4,5,6,7,8,9
                              Example 2: 1-4,7-9
        -----------------     --------------------------------------------------------------------
        aoi                   Optional :class:`~arcgis.geometry.Polygon`. The ``areaOfInterest`` Polygon allows
                              exporting tiles within the specified polygon areas.

                              .. note::
                                This parameter supersedes ``extent`` parameter.
        =================     ====================================================================

        :return: A dictionary

        .. code-block:: python

            # Example Usage
            size_estimate = img_tile_manager.estimate_size(tile_package = True,
                                                           levels = "11-20",
                                                           extent = {
                                                                        "xmin":6224324.092137296,
                                                                        "ymin":487347.5253569535,
                                                                        "xmax":11473407.698535524,
                                                                        "ymax":4239488.369818687
                                                                    },
                                                           merge = True,
                                                           optimize_for_size = True,
                                                           export_by = "Scale"
                                                           )
        """
        if self._service.properties["exportTilesAllowed"] == False:
            return None
        url = "%s/%s" % (self._url, "estimateExportTilesSize")
        if export_by is None:
            export_by = "LevelID"
        params = {
            "f": "json",
            "tilePackage": tile_package,
            "exportExtent": extent,
            "optimizeTilesForSize": optimize_for_size,
            "compressionQuality": compression,
            "exportBy": export_by,
            "levels": levels,
        }

        if aoi:
            params["areaOfInterest"] = aoi
        res = self._con.post(path=url, postdata=params, timeout=None)
        sid = res["jobId"]
        success, res = self._status(url, res)
        if success == False:
            return res
        else:
            if "results" in res and "out_service_url" in res["results"]:
                rurl = url + "/jobs/%s/%s" % (
                    sid,
                    res["results"]["out_service_url"]["paramUrl"],
                )
                result_url = self._con.get(path=rurl, params={"f": "json"})["value"]
                return result_url
            else:
                return res
        return res

    # ----------------------------------------------------------------------
    def _get_job(self, job_id):
        """
        Retrieves status and message information about a specific job.

        This is useful for checking jobs that have been launched manually.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        job_id                required string.  Unique ID of a job.
        =================     ====================================================================

        :return: dictionary
        """
        url = "%s/jobs/%s" % (self._url, job_id)
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def _get_job_inputs(self, job_id, parameter):
        """
        The Image Layer input method represents an input parameter for
        a Image Layer Job. It provides information about the input
        parameter such as its name, data type, and value. The value is the
        most important piece of information provided by this method.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        job_id                required string.  Unique ID of a job.
        -----------------     --------------------------------------------------------------------
        parameter             required string.  Name of the job parameter to retrieve.
        =================     ====================================================================

        :return: dictionary

        :Example Output Format:

        {"paramName" : "<paramName>","dataType" : "<dataType>","value" : <valueLiteralOrObject>}

        """
        url = "%s/jobs/%s/inputs/%s" % (self._url, job_id, parameter)
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def _get_job_result(self, job_id, parameter):
        """
        The Image Layer input method represents an input parameter for
        a Image Layer Job. It provides information about the input
        parameter such as its name, data type, and value. The value is the
        most important piece of information provided by this method.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        job_id                required string.  Unique ID of a job.
        -----------------     --------------------------------------------------------------------
        parameter             required string.  Name of the job parameter to retrieve.
        =================     ====================================================================

        :return: dictionary

        :Example Output Format:

        {"paramName" : "<paramName>","dataType" : "<dataType>","value" : <valueLiteralOrObject>}

        """
        url = "%s/jobs/%s/results/%s" % (self._url, job_id, parameter)
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def image_tile(self, level: int, row: int, column: int, blank_tile: bool = False):
        """
        For cached image services, the ``image_tile`` method represents a single cached
        tile for the image. The image bytes for the tile at the specified
        level, row, and column are directly streamed to the client.

        .. note::
            If the tile is not found, an HTTP status code of 404 is thrown.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        level                 Required integer. The level of detail ID.
        -----------------     --------------------------------------------------------------------
        row                   Required integer. The row of the cache to pull from.
        -----------------     --------------------------------------------------------------------
        column                Required integer. The column of the cache to pull from.
        -----------------     --------------------------------------------------------------------
        blank_tile            Optional boolean.  Default is False. This parameter applies only to
                              cached image services that are configured with the ability to return
                              blank or missing tiles for areas where cache is not available. When
                              False, the server will return a resource not found (HTTP 404)
                              response instead of a blank or missing tile. When this parameter is
                              not set, the response will contain the header blank-tile : true
                              for a blank/missing tile.
        =================     ====================================================================

        :return: None or file path (string)
        """
        import tempfile, uuid

        fname = "%s.jpg" % uuid.uuid4().hex
        params = {"blankTile": blank_tile}
        url = "%s/tile/%s/%s/%s" % (self._url, level, row, column)
        out_folder = tempfile.gettempdir()
        return self._con.get(
            path=url,
            out_folder=out_folder,
            file_name=fname,
            params=params,
            try_json=False,
        )


########################################################################
class RasterCatalogItem(object):
    """
    The ``RasterCatalogItem`` object represents a single catalog item on an ``ImageryLayer``. ``RasterCatalogItem``
    is only to be used with :class:`~arcgis.raster.ImageryLayer` objects that have ``Catalog`` in the layer's
    capabilities property.


    =================     ====================================================================
    **Parameter**          **Description**
    -----------------     --------------------------------------------------------------------
    url                   required string. Web address to the catalog item.
    -----------------     --------------------------------------------------------------------
    imglyr                required :class:`~arcgis.raster.ImageryLayer`. The imagery layer object.
    -----------------     --------------------------------------------------------------------
    initialize            optional boolean. Default is ``True``. If ``False``, the properties of the
                          item will not be loaded until requested.
    =================     ====================================================================

    """

    _properties = None
    _con = None
    _url = None
    _service = None
    _json_dict = None

    def __init__(self, url, imglyr, initialize=True):
        """class initializer"""
        self._url = url
        self._con = imglyr._con
        self._service = imglyr
        if initialize:
            self._init(self._con)

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        from arcgis._impl.common._mixins import PropertyMap
        from urllib.error import HTTPError

        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = PropertyMap(result)
            else:
                self._json_dict = {}
                self._properties = PropertyMap({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        The ``properties`` property retrieves the object's properties.

        :return:
            Raster properties
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
    def info(self):
        """
        The ``info`` property returns information about the associated raster
        such as its width, height, number of bands, and pixel type.

        :return:
            Raster properties
        """
        url = "%s/info" % self._url
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def key_properties(self):
        """
        The ``key_properties`` property returns key properties of the
        associated raster in an imagery layer.

        :return:
            Key properties of a :class:`~arcgis.raster.Raster` object
        """
        url = "%s/info/keyProperties" % self._url
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def thumbnail(self):
        """
        The ``thumbnail`` property returns a thumbnail of the current item.

        :return:
            A thumbnail
        """
        import tempfile

        folder = tempfile.gettempdir()
        url = "%s/thumbnail" % self._url
        params = {}
        return self._con.get(
            path=url,
            params={},
            try_json=False,
            out_folder=folder,
            file_name="thumbnail.png",
        )

    # ----------------------------------------------------------------------
    def image(
        self,
        bbox: str,
        return_format: str = "JSON",
        bbox_sr: Optional[Union[str, SpatialReference]] = None,
        size: Optional[list[int]] = None,
        image_sr: Optional[Union[str, SpatialReference]] = None,
        image_format: str = "png",
        pixel_type: Optional[str] = None,
        no_data: Optional[float] = None,
        interpolation: Optional[str] = None,
        compression: int = 75,
    ):
        """
        The ``image`` method returns a composite image for a single
        raster catalog item. You can use this method for generating
        dynamic images based on a single catalog item.
        This method provides information about the exported image, such
        as its URL, width and height, and extent.

        .. note::
            Apart from the usual response formats of html and json, you can
            also request a format called image for the image. When you specify
            image as the format, the server responds by directly streaming the
            image bytes to the client. With this approach, you don't get any
            information associated with the image other than the actual image.

        =================     ====================================================================
        **Parameter**         **Description**
        -----------------     --------------------------------------------------------------------
        bbox                  Required string. The extent (bounding box) of the exported image.
                              Unless the bbox_sr parameter has been specified, the bbox is assumed
                              to be in the spatial reference of the image layer.
                              Syntax: <xmin>, <ymin>, <xmax>, <ymax>
                              Example: bbox=-104,35.6,-94.32,41
        -----------------     --------------------------------------------------------------------
        return_format         Optional string.  The response can either be IMAGER or JSON. Image
                              will return the image file to disk where as the JSON value will
                              The default value is JSON.
        -----------------     --------------------------------------------------------------------
        bbox_sr               Optional string.  The spatial reference of the bbox.
        -----------------     --------------------------------------------------------------------
        size                  Optional string.The size (width * height) of the exported image in
                              pixels. If the size is not specified, an image with a default size
                              of 400 * 400 will be exported.
                              Syntax: <width>, <height>
                              Example: size=600,550
        -----------------     --------------------------------------------------------------------
        image_sr              Optional string/integer.  The spatial reference of the image.
        -----------------     --------------------------------------------------------------------
        format                Optional string. The format of the exported image. The default
                              format is png.
                              Values: `png, png8, png24, jpg, bmp, gif`
        -----------------     --------------------------------------------------------------------
        pixel_type            Optional string. The pixel type, also known as data type, that
                              pertains to the type of values stored in the raster, such as signed
                              integer, unsigned integer, or floating point. Integers are whole
                              numbers; floating points have decimals.
                              Values: `C128, C64, F32, F64, S16, S32, S8, U1, U16, U2, U32, U4,
                              U8, UNKNOWN`
        -----------------     --------------------------------------------------------------------
        no_data               Optional float. The pixel value representing no information.
        -----------------     --------------------------------------------------------------------
        interpolation         Optional string. The resampling process of extrapolating the pixel
                              values while transforming the raster dataset when it undergoes
                              warping or when it changes coordinate space.
                              Values: `RSP_BilinearInterpolation,
                              RSP_CubicConvolution, RSP_Majority, RSP_NearestNeighbor`
        -----------------     --------------------------------------------------------------------
        compression           Optional integer. Controls how much loss the image will be subjected
                              to by the compression algorithm. Valid value ranges of compression
                              quality are from 0 to 100.
        =================     ====================================================================

        :return:
            A composite image
        """
        import json

        try_json = True
        out_folder = None
        out_file = None
        url = "%s/image" % self._url
        if return_format is None:
            return_format = "json"
        elif return_format.lower() == "image":
            return_format = "image"
            out_folder = tempfile.gettempdir()
            if image_format is None:
                ext = "png"
            elif image_format.lower() in ("png", "png8", "png24"):
                ext = "png"
            else:
                ext = image_format
            try_json = False
            out_file = "%s.%s" % (uuid.uuid4().hex, ext)
        else:
            return_format = "json"
        params = {"f": return_format}
        if bbox is not None:
            params["bbox"] = bbox
        if bbox_sr is not None:
            params["bboxSR"] = bbox_sr
        if size is not None:
            params["size"] = size
        if image_sr is not None:
            params["imageSR"] = image_sr
        if image_format is not None:
            params["format"] = image_format
        if pixel_type is not None:
            params["pixelType"] = pixel_type
        if no_data is not None:
            params["noData"] = no_data

        return self._con.get(
            path=url,
            params=params,
            try_json=try_json,
            file_name=out_file,
            out_folder=out_folder,
        )

    # ----------------------------------------------------------------------
    @property
    def ics(self):
        """
        The raster ``ics`` property returns the image coordinate system of the
        associated raster in an image layer.

        .. note::
            The returned ics can be used as the SR parameter.


        """
        url = "%s/info/ics" % self._url
        return self._con.get(path=url, params={"f": "json"})

    # ----------------------------------------------------------------------
    @property
    def metadata(self):
        """
        The ``metadata`` property returns metadata of the imagery layer or a
        raster catalog item.

        .. note::
            The output format is always XML.

        :return:
            Metadata in XML format
        """
        url = "%s/info/metadata" % self._url
        out_folder = tempfile.gettempdir()
        out_file = "metadata.xml"
        return self._con.get(
            path=url,
            params={},
            try_json=False,
            file_name=out_file,
            out_folder=out_folder,
        )

    # ----------------------------------------------------------------------
    @property
    def image_support_data(self):
        """
        The ``image_support_data`` property returns  image support data of
        the NITF based raster catalog item. Specifically, the Image Support
        Data resource returns the NITF file structure and contents in XML
        format to provide more detailed information about a particular NITF file.
        """
        url = "%s/info/imageSupportData" % self._url
        out_folder = tempfile.gettempdir()
        out_file = "imageSupportData.xml"
        return self._con.get(
            path=url,
            params={},
            try_json=False,
            file_name=out_file,
            out_folder=out_folder,
        )

    # ----------------------------------------------------------------------
    @property
    def sensor(self):
        """
        The ``sensor`` property returns information about the sensor.
        Example:
        {
            "name": "IdentityXform",
            "sensor_provider": "esri"
        }
        """
        url = "%s/info/sensor" % self._url
        return self._con.get(path=url, params={"f": "json"})

    # ----------------------------------------------------------------------
    @property
    def ics_to_pixel(self):
        """
        The ``ics_to_pixel`` property returns coefficients to build up a mathematic model for geometric
        transformation.

        .. note::
            With this transformation, ICS coordinates based
            from the catalog item raster can be used to calculate the original
            column and row numbers on the corresponding image.

        """
        url = "%s/info/icsToPixel" % self._url
        return self._con.get(path=url, params={"f": "json"})


########################################################################
class RasterManager(object):
    """
    The ``RasterManager`` class allows users to update, add, and delete rasters to an
    :class:`~arcgis.raster.ImageryLayer` object.

    The functions are only available if the layer has 'Edit' on it's capabilities property.

    .. note ::
        This class is not created by users directly. An instance of this class, called ``rasters``,
        is available as a property of an ImageryLayer object. Users call methods on this ``rasters`` object
        to  update, add and delete rasters from an :class:`~arcgis.raster.ImageryLayer`.

    =================     ====================================================================
    **Parameter**          **Description**
    -----------------     --------------------------------------------------------------------
    imglyr                required :class:`~arcgis.raster.ImageryLayer` object.
                          The imagery layer object where 'Edit' is in the capabilities.
    =================     ====================================================================
    """

    _service = None

    # ----------------------------------------------------------------------
    def __init__(self, imglyr):
        """Constructor"""
        self._service = imglyr

    # ----------------------------------------------------------------------
    def add(
        self,
        raster_type,
        item_ids: Optional[str] = None,
        service_url: Optional[str] = None,
        compute_statistics: bool = False,
        build_pyramids: bool = False,
        build_thumbnail: bool = False,
        minimum_cell_size_factor: Optional[float] = None,
        maximum_cell_size_factor: Optional[float] = None,
        attributes: Optional[dict[str, Any]] = None,
        geodata_transforms: Optional[list[dict[str, Any]]] = None,
        geodata_transform_apply_method: str = "esriGeodataTransformApplyAppend",
    ):
        """
        The ``add`` operation is performed on an image layer method, adding new rasters to an image layer.
        (POST only).
        The added rasters can either be uploaded items, using the ``item_ids``
        parameter, or published services, using the ``service_url`` parameter.
        If ``item_ids`` is specified, uploaded rasters are copied to the image
        layer's dynamic image workspace location; if the ``service_url`` is
        specified, the image layer adds the URL to the mosaic dataset no
        raster files are copied. The ``service_url`` is required input for the
        following raster types: Image Layer, Map Service, WCS, and WMS.

        .. note::
            The ``add`` operation is supported at 10.1 and later.

        ===============================     ====================================================================
        **Parameter**                       **Description**
        -------------------------------     --------------------------------------------------------------------
        item_ids                            The upload items (raster files) to be added. Either item_ids or
                                            service_url is needed to perform this operation.
                                             
                                            Syntax:

                                                ``item_ids=<itemId1>,<itemId2>``

                                            Example:
                                                item_ids= "ib740c7bb-e5d0-4156-9cea-12fa7d3a472c,\
                                                           ib740c7bb-e2d0-4106-9fea-12fa7d3a482c"

        -------------------------------     --------------------------------------------------------------------
        service_url                         The URL of the service to be added. The image layer \
                                            will add this URL to the mosaic dataset. Either item_ids or \
                                            service_url is needed to perform this operation. The service URL is \
                                            required for the following raster types: Image Layer, Map Service, \
                                            WCS, and WMS.

                                            Example:
                                                service_url= "http://myserver/arcgis/services/Portland/ImageServer"
        -------------------------------     --------------------------------------------------------------------
        raster_type                         The type of raster files being added. Raster types \
                                            define the metadata and processing template for raster files to be \
                                            added. Allowed values are listed in image layer resource.

        -------------------------------     --------------------------------------------------------------------
        compute_statistics                  If True, statistics for the rasters will be computed. \
                                            The default is False.
                                            
                                            Values: 

                                                `False,True`
        -------------------------------     --------------------------------------------------------------------
        build_pyramids                      If True, builds pyramids for the rasters. The default is False.
                                            
                                            Values: 

                                                `False,True`
        -------------------------------     --------------------------------------------------------------------
        build_thumbnail                     If True, generates a thumbnail for the rasters. The default is False.
                                            
                                            Values:

                                                `False,True`
        -------------------------------     --------------------------------------------------------------------
        minimum_cell_size_factor            The factor (times raster resolution) used \
                                            to populate the MinPS field (maximum cell size above which the \
                                            raster is visible).
                                            
                                            Syntax:

                                                minimum_cell_size_factor=<minimum_cell_size_factor>

        -------------------------------     --------------------------------------------------------------------
        maximum_cell_size_factor            The factor (times raster resolution) used \
                                            to populate MaxPS field (maximum cell size below which raster is visible).
                                            
                                            Syntax:

                                                maximum_cell_size_factor=<maximum_cell_size_factor>

        -------------------------------     --------------------------------------------------------------------
        attributes                          Any attribute for the added rasters.

                                            Syntax:

                                              | {
                                              |   "<name1>" : <value1>,
                                              |   "<name2>" : <value2>
                                              | }
                

        -------------------------------     --------------------------------------------------------------------
        geodata_transforms                  The geodata transformations applied on the \
                                            added rasters. A geodata transformation is a mathematical model \
                                            that performs a geometric transformation on a raster; it defines \
                                            how the pixels will be transformed when displayed or accessed. \
                                            Polynomial, projective, identity, and other transformations are \
                                            available. The geodata transformations are applied to the dataset \
                                            that is added.

                                            Syntax:

                                               [
                                               {
                                                 "geodataTransform" : "<geodataTransformName1>",
                                                 "geodataTransformArguments" : {<geodataTransformArguments1>}
                                                 },
                                                 {
                                                 "geodataTransform" : "<geodataTransformName2>",
                                                 "geodataTransformArguments" : {<geodataTransformArguments2>}
                                                 }
                                               ]

                                            The syntax of the geodataTransformArguments property varies based \
                                            on the specified geodataTransform name. See Geodata Transformations \
                                            documentation for more details.
        -------------------------------     --------------------------------------------------------------------
        geodata_transform_apply_method      This parameter defines how to apply the provided geodataTransform. \
                                            The default is esriGeodataTransformApplyAppend.

                                            Values: 

                                                `esriGeodataTransformApplyAppend |
                                                esriGeodataTransformApplyReplace |
                                                esriGeodataTransformApplyOverwrite`
        ===============================     ====================================================================

        :return: A dictionary

        .. code-block:: python

            # Example Usage
            added = raster_manager.add(item_ids= "ib740c7bb-e5d0-4156-9cea-12fa7d3a472c,ib740c7bb-e2d0-4106-9fea-12fa7d3a482c",
                                       service_url = "http://myserver/arcgis/services/Portland/ImageServer",
                                       raster_type = "Raster Dataset",
                                       build_thumbnail = True,
                                       minimum_cell_size_factor = 0.1,
                                       maximum_cell_size_factor = 10,
                                       attributes = {
                                                        "MinPS": 0,
                                                        "MaxPS": 20;
                                                        "Year" : 2002,
                                                        "State" : "Florida"
                                                   },
                                       geodata_transforms = [
                                                                {
                                                                 "geodataTransform" : "<geodataTransformName1>",
                                                                 "geodataTransformArguments" : {<geodataTransformArguments1>}
                                                                },
                                                                {
                                                                 "geodataTransform" : "<geodataTransformName2>",
                                                                 "geodataTransformArguments" : {<geodataTransformArguments2>}
                                                                }
                                                           ],
                                       geodata_transform_apply_method = "esriGeodataTransformApplyOverwrite"
                                       )
        """

        return self._service._add_rasters(
            raster_type,
            item_ids,
            service_url,
            compute_statistics,
            build_pyramids,
            build_thumbnail,
            minimum_cell_size_factor,
            maximum_cell_size_factor,
            attributes,
            geodata_transforms,
            geodata_transform_apply_method,
        )

    # ----------------------------------------------------------------------
    def delete(self, raster_ids: str):
        """
        The ``delete`` operation deletes one or more rasters in an imagery layer.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        raster_ids            Required string. The object IDs of a raster catalog items to be
                              removed. This is a comma seperated string.

                              | example 1: raster_ids='1,2,3,4' # Multiple IDs
                              | example 2: raster_ids='10' # single ID
        =================     ====================================================================

        :return: A dictionary
        """
        return self._service._delete_rasters(raster_ids)

    # ----------------------------------------------------------------------
    def update(
        self,
        raster_id: int,
        files: Optional[list[str]] = None,
        item_ids: Optional[str] = None,
        service_url: Optional[str] = None,
        compute_statistics: bool = False,
        build_pyramids: bool = False,
        build_thumbnail: bool = False,
        minimum_cell_size_factor: Optional[float] = None,
        maximum_cell_size_factor: Optional[float] = None,
        attributes: Optional[dict[str, Any]] = None,
        footprint: Optional[Polygon] = None,
        geodata_transforms: Optional[list[dict[str, Any]]] = None,
        apply_method: str = "esriGeodataTransformApplyAppend",
    ):
        """
        The ``update`` operation updates rasters (attributes and
        footprints, or replaces existing raster files) in an image layer.
        In most cases, this operation is used to update attributes or
        footprints of existing rasters in an image layer.

        .. note::
            In cases where the original raster needs to be replaced, the new raster can either
            be items uploaded using the items parameter or URLs of published
            services using the serviceUrl parameter.

        ========================  ====================================================================
        **Parameter**              **Description**
        ------------------------  --------------------------------------------------------------------
        raster_id                 Required integer. The object IDs of a raster catalog items to be
                                  updated.
        ------------------------  --------------------------------------------------------------------
        files                     Optional list. Local source location to the raster to replace the
                                  dataset with.
                                  Example: [r"<path>\data.tiff"]
        ------------------------  --------------------------------------------------------------------
        item_ids                  Optional string.  The uploaded items (raster files) being used to
                                  replace existing raster.
        ------------------------  --------------------------------------------------------------------
        service_url               Optional string. The URL of the layer to be uploaded to replace
                                  existing raster data. The image layer will add this URL to the
                                  mosaic dataset. The serviceUrl is required for the following raster
                                  types: Image Layer, Map Service, WCS, and WMS.
        ------------------------  --------------------------------------------------------------------
        compute_statistics        If true, statistics for the uploaded raster will be computed. The
                                  default is false.
        ------------------------  --------------------------------------------------------------------
        build_pyramids            Optional boolean. If true, builds pyramids for the uploaded raster.
                                  The default is false.
        ------------------------  --------------------------------------------------------------------
        build_thumbnail           Optional boolean. If true, generates a thumbnail for the uploaded
                                  raster. The default is false.
        ------------------------  --------------------------------------------------------------------
        minimum_cell_size_factor  Optional float. The factor (times raster resolution) used to
                                  populate MinPS field (minimum cell size above which raster is
                                  visible).
        ------------------------  --------------------------------------------------------------------
        maximum_cell_size_factor  Optional float. The factor (times raster resolution) used to
                                  populate MaxPS field (maximum cell size below which raster is
                                  visible).
        ------------------------  --------------------------------------------------------------------
        footprint                 Optional Polygon.  A JSON 2D polygon object that defines the
                                  footprint of the raster. If the spatial reference is not defined, it
                                  will default to the image layer's spatial reference.
        ------------------------  --------------------------------------------------------------------
        attributes                Optional dictionary.  Any attribute for the uploaded raster.
        ------------------------  --------------------------------------------------------------------
        geodata_transforms        Optional list. The geodata transformations applied on the updated
                                  rasters. A geodata transformation is a mathematical model that
                                  performs geometric transformation on a raster. It defines how the
                                  pixels will be transformed when displayed or accessed, such as
                                  polynomial, projective, or identity transformations. The geodata
                                  transformations will be applied to the updated dataset.
        ------------------------  --------------------------------------------------------------------
        apply_method              Optional string. Defines how to apply the provided geodataTransform. \
                                  The default is esriGeodataTransformApplyAppend.
                                  
                                  Values: `esriGeodataTransformApplyAppend, \
                                      esriGeodataTransformApplyReplace, \
                                      esriGeodataTransformApplyOverwrite`
        ========================  ====================================================================

        :return: A dictionary

        .. code-block:: python

            # Example Usage
            updated = raster_manager.update(raster_id = 087631,
                                            item_ids = "ib740c7bb-e5d0-4156-9cea-12fa7d3a472c,ib740c7bb-e2d0-4106-9fea-12fa7d3a482c",
                                            service_url = "http://myserver/arcgis/services/Portland/ImageServer",
                                            build_thumbnail = True,
                                            minimum_cell_size_factor = 0.1,
                                            maximum_cell_size_factor = 10,
                                            attributes = {
                                                            "MinPS": 0,
                                                            "MaxPS": 20;
                                                            "Year" : 2002,
                                                            "State" : "Florida"
                                                         },
                                            geodata_transforms = [
                                                                      {
                                                                       "geodataTransform" : "<geodataTransformName1>",
                                                                       "geodataTransformArguments" : {<geodataTransformArguments1>}
                                                                      },
                                                                      {
                                                                       "geodataTransform" : "<geodataTransformName2>",
                                                                       "geodataTransformArguments" : {<geodataTransformArguments2>}
                                                                      }
                                                                 ],
                                            apply_method = "esriGeodataTransformApplyOverwrite"
                                        )

        """
        return self._service._update_raster(
            raster_id=raster_id,
            files=files,
            item_ids=item_ids,
            service_url=service_url,
            compute_statistics=compute_statistics,
            build_pyramids=build_pyramids,
            build_thumbnail=build_thumbnail,
            minimum_cell_size_factor=minimum_cell_size_factor,
            maximum_cell_size_factor=maximum_cell_size_factor,
            attributes=attributes,
            footprint=footprint,
            geodata_transforms=geodata_transforms,
            apply_method=apply_method,
        )
