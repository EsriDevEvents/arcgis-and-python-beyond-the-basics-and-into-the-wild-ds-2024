from arcgis.auth.tools import LazyLoader
import os
import functools
import urllib.parse
from arcgis.gis import GIS

logging = LazyLoader("logging")
_isd = LazyLoader("arcgis._impl.common._isd")
layer = LazyLoader("arcgis.features.layer")
geocoding = LazyLoader("arcgis.geocoding")
gptool = LazyLoader("arcgis.geoprocessing._tool")
geommodule = LazyLoader("arcgis._impl.tools")
network_dataset = LazyLoader("arcgis.network")
_gis = LazyLoader("arcgis.gis")
mapping = LazyLoader("arcgis.mapping")
raster = LazyLoader("arcgis.raster")
schematics = LazyLoader("arcgis.schematics")

_log = logging.getLogger()


@functools.lru_cache(maxsize=250)
def _create_service(url: str, layer_type: str, gis: GIS, name: str = None):
    has_layer = False
    digit = None
    if name:
        url = url.replace(name, urllib.parse.quote(name))
    if os.path.basename(url).isdigit():
        digit = os.path.basename(url)
        url = os.path.dirname(url)
        has_layer = True
    if layer_type.lower() == "mapserver":
        if has_layer:
            return mapping.MapServiceLayer(url=f"{url}/{digit}", gis=gis)
        else:
            return mapping.MapImageLayer(url=url, gis=gis)
    elif layer_type.lower() == "featureserver":
        if has_layer:
            return layer.FeatureLayer(url=url, gis=gis)
        else:
            return layer.FeatureLayerCollection(url=url, gis=gis)
    elif layer_type.lower() == "imageserver":
        return raster.ImageryLayer(url=url, gis=gis)
    elif layer_type.lower() == "gpserver":
        return gptool.Toolbox(url=url, gis=gis)
    elif layer_type.lower() == "geometryserver":
        return geommodule.GeometryService(url=url, gis=gis)
    elif layer_type.lower() == "geocodeserver":
        return geocoding.Geocoder(location=url, gis=gis)
    elif layer_type.lower() == "naserver":
        return network_dataset.NetworkDataset(url=url, gis=gis)
    elif layer_type.lower() == "vectortileserver":
        return mapping.VectorTileLayer(url=url, gis=gis)
    elif layer_type.lower() == "sceneserver":
        return mapping._scenelyrs._lyrs.SceneLayer(url=url, gis=gis)
    else:
        return _gis.Layer(url=url, gis=gis)

    return None


###########################################################################
class AGOLServicesDirectory:
    """
    The ArcGIS Online Services Directory displays the hosted services for
    a site.

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

    # ---------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS):
        """initializer"""
        self._gis = gis
        self._url = url

    # ---------------------------------------------------------------------
    def __str__(self):
        return f"< AGOLServicesDirectory @ {self._url} >"

    # ---------------------------------------------------------------------
    def __repr__(self):
        return f"< AGOLServicesDirectory @ {self._url} >"

    # ---------------------------------------------------------------------
    @functools.lru_cache(maxsize=255)
    def _org_id(self, gis: GIS) -> str:
        return gis.properties.id

    # ---------------------------------------------------------------------
    @property
    def properties(self) -> _isd.InsensitiveDict:
        """
        Returns the server's properties

        :returns: InsensitiveDict
        """
        resp = self._gis._con.get(self._url, {"f": "json"})
        return _isd.InsensitiveDict(resp)

    # ---------------------------------------------------------------------
    @property
    def folders(self) -> list:
        """
        Returns a list of folder names

        :returns: List
        """
        return []

    # ---------------------------------------------------------------------
    @property
    def services(self) -> list:
        """returns a list of services hosted on ArcGIS Online Server"""
        services = []

        if "services" in self.properties:
            for s in self.properties["services"]:
                try:
                    services.append(
                        _create_service(
                            url=s["url"],
                            layer_type=s["type"],
                            gis=self._gis,
                            name=s.get("name", None),
                        )
                    )
                except Exception as e:
                    msg = "URL: %s is throwing error:  %s" % (s["url"], e)
                    _log.warning(msg)
                    _log.warning("Could not load service: %s" % s["url"])
        return services
