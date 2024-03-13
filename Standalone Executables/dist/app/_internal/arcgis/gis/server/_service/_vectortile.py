"""

.. module:: _vectortile.py
   :platform: Windows, Linux
   :synopsis: Represents functions/classes that represents a vector tile
              service.
.. moduleauthor:: Esri

"""
from __future__ import absolute_import
import tempfile
from typing import Optional
from .._common._base import BaseServer


########################################################################
class VectorTile(BaseServer):
    """
    The vector tile service resource represents a vector service published
    with ArcGIS Server. The resource provides information about the service
    such as the tile info, spatial reference, initial and full extents.
    """

    _con = None
    _url = None
    _json = None
    _json_dict = None

    def __init__(self, url, connection=None, initialize=True):
        """class initializer"""
        super(VectorTile, self).__init__(url=url, connection=connection)
        self._url = url
        self._con = connection
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    def tile_fonts(
        self, fontstack: str, stack_range: str, out_folder: Optional[str] = None
    ):
        """This resource returns glyphs in PBF format. The template url for
        this fonts resource is represented in Vector Tile Style resource."""
        url = "{url}/resources/fonts/{fontstack}/{stack_range}.pbf".format(
            url=self._url, fontstack=fontstack, stack_range=stack_range
        )
        params = {}
        if out_folder is None:
            out_folder = tempfile.gettempdir()
        return self._con.get(path=url, params=params, out_folder=out_folder)

    # ----------------------------------------------------------------------
    def vector_tile(
        self, level: str, row: str, column: str, out_folder: Optional[str] = None
    ):
        """This resource represents a single vector tile for the map. The
        bytes for the tile at the specified level, row and column are
        returned in PBF format. If a tile is not found, an HTTP status code
        of 404 (Not found) is returned."""
        url = "{url}/tile/{level}/{row}/{column}.pbf".format(
            url=self._url, level=level, row=row, column=column
        )
        params = {}

        if out_folder is None:
            out_folder = tempfile.gettempdir()
        return self._con.get(
            path=url, params=params, out_folder=out_folder, try_json=False
        )

    # ----------------------------------------------------------------------
    def tile_sprite(
        self, out_format: str = "sprite.json", out_folder: Optional[str] = None
    ):
        """
        This resource returns sprite image and metadata
        """
        url = "{url}/resources/sprites/{f}".format(url=self._url, f=out_format)
        if out_folder is None:
            out_folder = tempfile.gettempdir()
        return self._con.get(path=url, params={}, out_folder=out_folder)

    # ----------------------------------------------------------------------
    @property
    def info(self):
        """This returns relative paths to a list of resource files"""
        url = "{url}/resources/info".format(url=self._url)
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @property
    def iteminfo(self):
        """returns the item information for the service"""
        url = "{url}/resources/info/iteminfo".format(url=self._url)
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @property
    def metadata(self):
        """returns the item's metadata for the service"""
        url = "{url}/resources/info/metedata".format(url=self._url)
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @property
    def thumbnail(self):
        """returns the item's thumbnail for the service"""
        url = "{url}/resources/info/thumbnail".format(url=self._url)
        params = {"f": "json"}
        return self._con.get(path=url, params=params)
