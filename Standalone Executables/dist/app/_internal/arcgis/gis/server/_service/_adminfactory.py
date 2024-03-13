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
from arcgis._impl.tools import _GeometryService as GeometryService
from arcgis.network import NetworkDataset
from arcgis.gis import Layer
from arcgis.mapping import VectorTileLayer
from arcgis.mapping import MapImageLayer
from arcgis.raster import ImageryLayer
from arcgis.schematics import SchematicLayers
from arcgis.mapping._scenelyrs import SceneLayer
from ..._impl._con import Connection
from ._geodataservice import GeoData
from ._layerfactory import Service
from ..admin._services import Service as AdminService


def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""

    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get("__slots__")
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop("__dict__", None)
        orig_vars.pop("__weakref__", None)
        if hasattr(cls, "__qualname__"):
            orig_vars["__qualname__"] = cls.__qualname__
        return metaclass(cls.__name__, cls.__bases__, orig_vars)

    return wrapper


def _str_replace(mystring, rd):
    """replaces a value based on a key/value pair where the
    key is the text to replace and the value is the new value.

    The find/replace is case insensitive.

    """
    import re

    patternDict = {}
    myDict = {}
    for key, value in rd.items():
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        patternDict[value] = pattern
    for key in patternDict:
        regex_obj = patternDict[key]
        mystring = regex_obj.sub(key, mystring)
    return mystring


class AdminServiceFactory(type):
    """
    Generates an Administrative Service Object from a url or service object
    """

    def __call__(cls, service, gis, initialize=False):
        """generates the proper type of layer from a given url"""

        url = service._url
        if isinstance(service, FeatureLayer) or os.path.basename(url).isdigit():
            parent = Service(url=os.path.dirname(url), server=gis)
            return AdminServiceGen(parent, gis)
        elif isinstance(service, (NetworkDataset)):
            rd = {"naserver", "MapServer"}
            url = _str_replace(url, rd)
            parent = Service(url=url, server=gis)
            return AdminServiceGen(parent, gis)
        else:
            rd = {"/rest/": "/admin/"}
            connection = service._con
            admin_url = "%s.%s" % (
                _str_replace(os.path.dirname(url), rd),
                os.path.basename(url),
            )
            return AdminService(url=admin_url, gis=gis)


###########################################################################
@add_metaclass(AdminServiceFactory)
class AdminServiceGen(object):
    """
    The Layer class allows users to pass a url, connection or other object
    to the class and get back properties and functions specifically related
    to the service.

    Inputs:
       url - internet address to the service
       server - Server class
       item - Enterprise or Online Item class
    """

    def __init__(self, service, gis):
        iterable = None
        if iterable is None:
            iterable = ()
        super(AdminService, self).__init__(service, gis)
