"""
Generates Layer Types from the given inputs.

"""
from __future__ import absolute_import
import os
from urllib.parse import urlparse
from arcgis.gis import GIS
from arcgis.features.layer import FeatureLayer, FeatureLayerCollection
from arcgis.geocoding import Geocoder
from arcgis.geoprocessing._tool import Toolbox
from arcgis.geoprocessing import import_toolbox as _import_toolbox
from arcgis._impl.tools import _GeometryService as GeometryService
from arcgis.network import NetworkDataset
from arcgis.gis import Layer
from arcgis.mapping import VectorTileLayer
from arcgis.mapping import MapImageLayer, MapServiceLayer
from arcgis.raster import ImageryLayer
from arcgis.schematics import SchematicLayers
from arcgis.mapping._scenelyrs import SceneLayer
from ..._impl._con import Connection
from ._geodataservice import GeoData


class ServiceFactory(type):
    """
    Generates a geometry object from a given set of
    JSON (dictionary or iterable)
    """

    def __call__(cls, url=None, item=None, server=None, initialize=False):
        """generates the proper type of layer from a given url"""
        from .. import ServicesDirectory

        hasLayer = False
        if url is None and item is None:
            raise ValueError("A URL to the service or an arcgis.Item is required.")
        elif url is None and item is not None:
            url = item.url

        if isinstance(server, Connection) or hasattr(server, "token"):
            connection = server
        elif isinstance(server, (GIS, ServicesDirectory)):
            connection = server._con
        else:
            try:
                parsed = urlparse(url)
                site_url = "{scheme}://{nl}/{wa}".format(
                    scheme=parsed.scheme,
                    nl=parsed.netloc,
                    wa=parsed.path[1:].split("/")[0],
                )
                connection = Connection(baseurl=site_url)  # anonymous connection
                server = ServicesDirectory(url=site_url)
            except:
                parsed = urlparse(url)
                site_url = "https://{nl}/rest/services".format(
                    scheme=parsed.scheme, nl=parsed.netloc
                )
                connection = Connection(
                    baseurl=site_url, all_ssl=parsed.scheme == "https"
                )  # anonymous connection
                server = ServicesDirectory(url=site_url)
        base_name = os.path.basename(url)
        if base_name.isdigit():
            base_name = os.path.basename(url.replace("/" + base_name, ""))
            hasLayer = True
        if base_name.lower() == "mapserver":
            if hasLayer:
                return MapServiceLayer(url=url, gis=server)
            else:
                return MapImageLayer(url=url, gis=server)
        elif base_name.lower() == "featureserver":
            if hasLayer:
                return FeatureLayer(url=url, gis=server)
            else:
                return FeatureLayerCollection(url=url, gis=server)
        elif base_name.lower() == "imageserver":
            return ImageryLayer(url=url, gis=server)
        elif base_name.lower() == "gpserver":
            from arcgis.geoprocessing import import_toolbox as _import_toolbox

            res = _import_toolbox(url, server)
            return res
        elif base_name.lower() == "geometryserver":
            return GeometryService(url=url, gis=server)
        elif base_name.lower() == "mobileserver":
            return Layer(url=url, gis=server)
        elif base_name.lower() == "geocodeserver":
            return Geocoder(location=url, gis=server)
        elif base_name.lower() == "globeserver":
            if hasLayer:
                return Layer(url=url, gis=server)
            return Layer(url=url, gis=server)
        elif base_name.lower() == "geodataserver":
            return GeoData(url=url, connection=connection)
        elif base_name.lower() == "naserver":
            return NetworkDataset(url=url, gis=server)
        elif base_name.lower() == "sceneserver":
            return SceneLayer(url=url, gis=server)
        elif base_name.lower() == "schematicsserver":
            return SchematicLayers(url=url, gis=server)
        elif base_name.lower() == "vectortileserver":
            return VectorTileLayer(url=url, gis=server)
        else:
            return Layer(url=url, gis=server)
        return type.__call__(cls, url, connection, item, initialize)


###########################################################################
class Service(object, metaclass=ServiceFactory):
    """
    The Layer class allows users to pass a url, connection or other object
    to the class and get back properties and functions specifically related
    to the service.

    Inputs:
       url - internet address to the service
       server - Server class
       item - Portal or AGOL Item class
    """

    def __init__(self, url, item=None, server=None):
        if iterable is None:
            iterable = ()
        super(Layer, self).__init__(url, item, server)
