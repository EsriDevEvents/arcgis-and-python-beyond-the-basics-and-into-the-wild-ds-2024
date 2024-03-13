"""
Contains the base class that all portaladmin object inherit from.
"""
from __future__ import absolute_import
import json
from ...gis._impl._con import Connection
from ...gis import GIS
from ..._impl.common._mixins import PropertyMap


###########################################################################
class BasePortalAdmin(object):
    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    def __init__(self, url, gis=None, initialize=True, **kwargs):
        """class initializer"""
        super(BasePortalAdmin, self).__init__()
        self._url = url
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(connection=self._con)

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        result = connection.get(path=self._url, params=params)
        try:
            self._json_dict = result
            self._properties = PropertyMap(self._json_dict)
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the object"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def url(self):
        """gets/sets the service url"""
        return self._url

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __iter__(self):
        """creates iterable for classes properties"""
        for k, v in self._json_dict.items():
            yield k, v

    # ----------------------------------------------------------------------
    def _refresh(self):
        """reloads all the properties of a given service"""
        self._init()
