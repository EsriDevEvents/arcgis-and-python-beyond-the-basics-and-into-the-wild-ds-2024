from __future__ import annotations
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


class BaseMissionServer(object):
    """
    Base Class for all Mission Server Classes
    """

    _url = None
    _gis = None
    _con = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None):
        self._url = url
        if gis is None:
            from arcgis import env

            gis = env.active_gis
        if gis is None:
            raise ValueError("A GIS could not be obtained.")
        self._gis = gis
        self._con = self._gis._con

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        self._properties = None
        res = self._con.get(self._url, {"f": "json"})
        self._properties = PropertyMap(res)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """gets the service properties"""
        if self._properties is None:
            self._init()
        return self._properties
