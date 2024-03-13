"""
Contains the base class that all server object inherit from.
"""
from __future__ import annotations
from __future__ import absolute_import
import json
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, Generator
from urllib.request import HTTPError
from arcgis.gis._impl._con import Connection
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


###########################################################################
class _BaseKube(object):
    """class most server object inherit from"""

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    def __init__(
        self,
        url: str,
        gis: Optional[GIS] = None,
        initialize: Optional[bool] = True,
        **kwargs: Optional[Any],
    ) -> None:
        """class initializer"""
        if gis is None and "connection" in kwargs:
            connection = kwargs["connection"]
            gis = kwargs.pop("connection", None)
        super(_BaseKube, self).__init__()
        self._url = url

        # gis = kwargs.pop('gis', None)
        if not gis is None and isinstance(gis, GIS):
            gis = gis._portal.con

        if isinstance(gis, Connection):
            self._con = gis
        elif hasattr(gis, "_con"):
            self._gis = gis._con
        else:
            raise ValueError("gis must be of type Connection or GIS")
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    def _init(self, connection: Optional[Connection] = None) -> None:
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
    def __str__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> Dict[str, Any]:
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
    def url(self) -> str:
        """gets/sets the service url"""
        return self._url

    # ----------------------------------------------------------------------
    @url.setter
    def url(self, value) -> None:
        """gets/sets the service url"""
        self._url = value
        self.refresh()

    # ----------------------------------------------------------------------
    def __iter__(self):
        """creates iterable for classes properties"""
        for k, v in self._json_dict.items():
            yield k, v

    # ----------------------------------------------------------------------
    def _refresh(self) -> None:
        """reloads all the properties of a given service"""
        self._init()
