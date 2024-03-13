"""
Represents messages left by users on a given Item in the GIS
"""
from __future__ import absolute_import
import json
from ..gis._impl._con import Connection
from ..gis import GIS, Item
from urllib.parse import unquote


########################################################################
class Comment(dict):
    """
    Represents messages left by users on a given Item in the GIS
    """

    _gis = None
    _portal = None
    _item = None
    _con = None
    _json_dict = None
    _json = None
    _hydrated = None

    def __init__(self, url, item, data=None, initialize=True, **kwargs):
        """
        class initializer

        Parameters:
         :url: web address to the resource
         :item: arcgis.gis.Item class where the comment originate from
         :data: allows the object to be pre-populated with information from
          a dictionary
         :initialize: if True, on creation, the object will hydrate itself.
        """
        super(Comment, self).__init__()
        self._url = url
        self._gis = item._gis
        self._portal = self._gis._portal
        if isinstance(item._gis, Connection):
            self._con = item._gis
        elif isinstance(item._gis, GIS):
            self._gis = item._gis
            self._con = item._gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if data and isinstance(data, dict):
            for k, v in data.items():
                self[k] = v
            self.__dict__.update(data)
        if initialize:
            self._init(connection=self._con)
            self._hydrated = True
        else:
            self._hydrated = False

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not attr.startswith("_")
        ]
        params = {"f": "json"}
        result = connection.get(path=self._url, params=params)
        self._json_dict = result
        for k, v in result.items():
            if k in attributes:
                setattr(self, "_" + k, v)
                self[k] = v
            else:
                self[k] = v
        self.__dict__.update(result)

    # ----------------------------------------------------------------------
    def __getattr__(self, name):
        if not self._hydrated and not name.startswith("_"):
            self._init()
        try:
            if name.lower() == "comment" and name.lower() in [
                g.lower() for g in self.__dict__.keys()
            ]:
                try:
                    if "%u" in self.__dict__[name]:
                        return (
                            unquote(self.__dict__[name])
                            .replace("%u", "\\u")
                            .encode()
                            .decode("unicode_escape")
                        )
                    return self.__dict__[name]
                except:
                    return self.__dict__[name]
            return self.__dict__[name]
        except:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    # ----------------------------------------------------------------------
    def __getitem__(self, k):
        try:
            if k.lower() == "comment" and k in [
                g.lower() for g in self.__dict__.keys()
            ]:
                try:
                    if "%u" in self.__dict__[k]:
                        return (
                            unquote(self.__dict__[k])
                            .replace("%u", "\\u")
                            .encode()
                            .decode("unicode_escape")
                        )
                    return self.__dict__[k]
                except:
                    return self.__dict__[k]
            return self.__dict__[k]
        except KeyError:
            if not self._hydrated and not k.startswith("_"):
                self._init()
            return self.__dict__[k]

    # ----------------------------------------------------------------------
    def update(self, comment):
        """
        Updates a comment. Available only to the authenticated user who
        created the comment.

        Parameters:
         :comment: updated comment text
        :return:
         On successful update, the comment Id
         On unsuccessful update, JSON response message
        """
        url = "%s/update" % self._url
        params = {"f": "json", "comment": comment}
        res = self._con.post(path=url, postdata=params)
        if "commentId" in res:
            self._init()
            return res["commentId"]
        return res

    # ----------------------------------------------------------------------
    def delete(self):
        """
        removes a given comment
        """
        url = "%s/delete" % self._url
        params = {"f": "json"}
        res = self._con.post(path=url, postdata=params)
        if "success" in res:
            return res["success"]
        return res
