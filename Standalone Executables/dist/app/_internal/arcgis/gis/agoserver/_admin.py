import logging
import urllib.parse
from typing import Union, List
from functools import lru_cache

from cachetools import cached, TTLCache
from arcgis.auth.tools import LazyLoader
from arcgis._impl.common._isd import InsensitiveDict
from arcgis.gis import GIS

_scenemgr = LazyLoader("arcgis.mapping._scenelyrs._lyrs")
_featuremgr = LazyLoader("arcgis.features.managers")
_mapservermgr = LazyLoader("arcgis.mapping._types")
_imagemgr = LazyLoader("arcgis.raster._layer")

_log = logging.getLogger()


###########################################################################
class AGOLServerManager:
    """
    Represents a Single AGO Server

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required String. The url string to the ArcGIS Online Server
    ------------------     --------------------------------------------------------------------
    gis                    Required GIS. The connection to ArcGIS Online.
    ==================     ====================================================================

    """

    _gis = None
    _url = None
    _properties = None

    def __init__(self, url: str, gis: GIS):
        """initializer"""
        self._url = url
        self._gis = gis

    # ---------------------------------------------------------------------
    def __str__(self):
        return f"< AGOLServerManager @ {self._url} >"

    # ---------------------------------------------------------------------
    def __repr__(self):
        return f"< AGOLServerManager @ {self._url} >"

    @property
    @lru_cache(maxsize=100)
    def is_tile_server(self) -> bool:
        """
        Returns if the server if hosting tiles or not

        :returns: bool
        """
        return self._url.lower().find("/tiles/") > -1

    @property
    @cached(cache=TTLCache(maxsize=10, ttl=25))
    def properties(self) -> InsensitiveDict:
        """
        Returns the server's properties. This call is cached for 25 seconds.

        :return: Dict
        """
        resp = self._gis._con.get(self._url, {"f": "json"})
        return InsensitiveDict(resp)

    @lru_cache(maxsize=50)
    def get(
        self, name: str
    ) -> Union[
        _mapservermgr.VectorTileLayerManager,
        _imagemgr.ImageryLayerCacheManager,
        _scenemgr.SceneLayerManager,
        _featuremgr.FeatureLayerCollectionManager,
        _mapservermgr.MapImageLayerManager,
    ]:
        """
        Returns a single service manager.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required String. The name of the service.
        ==================     ====================================================================

        :returns: Union[:class:`~arcgis.mapping.VectorTileLayer`,
                        :class:`~arcgis.raster.ImageryLayerCacheManager`,
                        :class:`~arcgis.mapping.SceneLayerManager`,
                        :class:`~arcgis.features.managers.FeatureLayerCollectionManager`,
                        :class:`~arcgis.mapping.MapImageLayerManager`]
        """

        if self.is_tile_server == False:
            for service in self.services:
                if service.properties.adminServiceInfo.name.lower() == name.lower():
                    return service
        else:
            for service in self.services:
                if service.properties.name.lower() == name.lower():
                    return service

        return

    def status(self, name: str) -> str:
        """
        Returns the status of a given service by name.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required String. The name of the service.
        ==================     ====================================================================

        :returns: string
        """
        found = False
        if self.is_tile_server == False:
            if "services" in self.properties:
                for service in self.properties["services"]:
                    if "adminServiceInfo" in service:
                        service = service["adminServiceInfo"]
                    if service["name"].lower() == name.lower():
                        found = True
                        return service["status"]
        else:
            for service in self.services:
                if service.properties.name.lower() == name.lower():
                    found = True
                    if hasattr(service, "status"):
                        return service.status
                    elif "status" in service.properties:
                        return service.properties.status
                    else:
                        return "UNKNOWN"
        if found == False:
            raise Exception("Service not found.")

    @property
    def services(self) -> list:
        """Returns the Administrative Endpoints

        :returns: list
        """
        services = []
        properties = self.properties
        if "services" in properties:
            for service in properties["services"]:
                if "adminServiceInfo" in service:
                    service = service["adminServiceInfo"]
                name = urllib.parse.quote(service["name"])
                if self.is_tile_server:
                    url = f"{self._url}/{name}/{service['type']}"
                else:
                    url = f"{self._url}/{name}.{service['type']}"
                serivce_type = service["type"].lower()
                if serivce_type == "mapserver":
                    services.append(
                        _mapservermgr.MapImageLayerManager(url=url, gis=self._gis)
                    )
                elif serivce_type == "featureserver":
                    services.append(
                        _featuremgr.FeatureLayerCollectionManager(
                            url=url, gis=self._gis
                        )
                    )

                elif serivce_type.find("vector") > -1:
                    services.append(
                        _mapservermgr.VectorTileLayerManager(url=url, gis=self._gis)
                    )
                elif serivce_type == "sceneserver":
                    services.append(_scenemgr.SceneLayerManager(url=url, gis=self._gis))
                elif serivce_type == "imageserver":
                    services.append(
                        _imagemgr.ImageryLayerCacheManager(url, gis=self._gis)
                    )
                else:
                    _log.warning(f"No manager found for service: {url}.")
        return services


###########################################################################
class AGOLServersManager:
    """
    This class allows users to work with hosted tile and feature services on
    ArcGIS Online.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    gis                    Required GIS. The connection to ArcGIS Online.
    ==================     ====================================================================


    """

    _gis = None

    def __init__(self, gis: GIS):
        """initializer"""
        if gis._portal.is_arcgisonline == False:
            raise ValueError("Invalid GIS")
        self._gis = gis

    @property
    def properties(self) -> InsensitiveDict:
        """
        Returns the properties of the server

        :returns: InsensitiveDict
        """
        return InsensitiveDict(self._urls(gis=self._gis))

    @lru_cache(maxsize=254)
    def _urls(self, gis: GIS) -> dict:
        """returns the parsed urls"""
        info = gis._registered_servers()
        tile_urls = set(info["urls"].get("tiles", {}).get("https", []))
        feature_urls = set(info["urls"].get("features", {}).get("https", []))
        tile_urls = set(info["urls"].get("tiles", {}).get("https", []))
        pid = gis.properties.id
        tile_urls = [
            f"https://{url}/tiles/{pid}/arcgis/rest/admin/services"
            for url in tile_urls
            if url not in feature_urls
        ]
        feature_urls = [
            f"https://{url}/{pid}/ArcGIS/admin/services" for url in feature_urls
        ]
        return {"tile": tile_urls, "feature": feature_urls}

    @property
    def tile_server(self) -> List[AGOLServerManager]:
        """returns a list of Tile Administrative Servers"""
        return [
            AGOLServerManager(url, gis=self._gis)
            for url in self._urls(self._gis)["tile"]
        ]

    @property
    def feature_server(self) -> List[AGOLServerManager]:
        """returns a list of Feature Administrative Servers"""
        return [
            AGOLServerManager(url, gis=self._gis)
            for url in self._urls(self._gis)["feature"]
        ]

    @lru_cache(maxsize=254)
    def list(self) -> List[AGOLServerManager]:
        """
        Returns a list of all server managers

        :returns: List[:class:`~arcgis.gis.agoserver.AGOLServerManager`]
        """
        return self.tile_server + self.feature_server
