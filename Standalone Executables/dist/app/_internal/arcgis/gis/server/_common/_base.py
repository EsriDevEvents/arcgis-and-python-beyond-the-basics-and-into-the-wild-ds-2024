"""
Contains the base class that all server object inherit from.
"""
from __future__ import absolute_import
from __future__ import annotations
import json
import functools
from collections import OrderedDict
from urllib.request import HTTPError
from ..._impl._con import Connection
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


###########################################################################
class BaseServer(object):
    """class most server object inherit from"""

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    def __init__(self, url, gis=None, initialize=True, **kwargs):
        """class initializer"""
        if gis is None and "connection" in kwargs:
            connection = kwargs["connection"]
            gis = kwargs.pop("connection", None)
        super(BaseServer, self).__init__()
        self._url = url

        # gis = kwargs.pop('gis', None)
        if gis is None and isinstance(gis, GIS):
            gis = gis._portal.con

        if isinstance(gis, Connection):
            self._con = gis
        elif hasattr(gis, "_con"):
            self._gis = gis._con
        else:
            raise ValueError("gis must be of type SiteConnection")
        if initialize:
            self._init(gis)

    @functools.lru_cache(maxsize=10)
    def _server_version(self) -> list[float]:
        """returns the server version number"""
        params: dict[str, Any] = {"f": "json"}
        url: str = self._url.split("/admin/")[0] + "/admin"
        return [
            int(v) for v in str(self._con.get(url, params)["currentVersion"]).split(".")
        ]

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
        self._refresh()

    # ----------------------------------------------------------------------
    def __iter__(self):
        """creates iterable for classes properties"""
        for k, v in self._json_dict.items():
            yield k, v

    # ----------------------------------------------------------------------
    def _refresh(self):
        """reloads all the properties of a given service"""
        self._init()
