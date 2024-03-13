import os
import sys
import json
import uuid
from arcgis.gis import GIS
from arcgis.mapping import create_symbol
from arcgis._impl.common._isd import InsensitiveDict
from ._base import BaseOGC


###########################################################################
class GeoRSSLayer(BaseOGC):
    """
    The GeoRSSLayer class is used to create a layer based on GeoRSS. GeoRSS is a
    way to add geographic information to an RSS feed. The GeoRSSLayer supports
    both GeoRSS-Simple and GeoRSS GML encodings, and multiple geometry types.

    It exports custom RSS tags as additional attribute fields in the form of
    simple strings or an array of JSON objects.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required String. The URL of the GeoRSS sevice.
    ---------------     --------------------------------------------------------------------
    copyright           Optional String. Describes limitations and usage of the data.
    ---------------     --------------------------------------------------------------------
    line_symbol         Optional Dict. The symbol for the polyline data in the GeoRSS.
    ---------------     --------------------------------------------------------------------
    opacity             Optional Float.  This value can range between 1 and 0, where 0 is 100 percent transparent and 1 is completely opaque.
    ---------------     --------------------------------------------------------------------
    point_symbol        Optional Dict. The symbol for the point data in the GeoRSS.
    ---------------     --------------------------------------------------------------------
    polygon_symbol      Optional Dict. The symbol for the polygon data in the GeoRSS.
    ---------------     --------------------------------------------------------------------
    title               Optional String. The title of the layer used to identify it in places such as the Legend and LayerList widgets.
    ---------------     --------------------------------------------------------------------
    scale               Optional Tuple. The min/max scale of the layer where the positions are: (min, max) as float values.
    ===============     ====================================================================

    """

    _line_symbol = None
    _point_symbol = None
    _polygon_symbol = None
    _type = "GeoRSS"

    # ----------------------------------------------------------------------
    def __init__(self, url, **kwargs):
        super(GeoRSSLayer, self)
        self._id = kwargs.pop("id", uuid.uuid4().hex)
        self._url = url
        scale = kwargs.pop("scale", (0, 0))
        assert isinstance(scale, (list, tuple)) and len(scale) == 2
        self._min_scale = scale[0]
        self._max_scale = scale[1]
        self._opacity = kwargs.pop("opacity", 1)
        self._copyright = kwargs.pop("copyright", None)
        self._title = kwargs.pop("title", None)
        self.point_symbol = kwargs.pop("point_symbol", None)
        self.polygon_symbol = kwargs.pop("polygon_symbol", None)
        self.line_symbol = kwargs.pop("line_symbol", None)
        self.title = kwargs.pop("title", "GeoRSS Feed")

    # ----------------------------------------------------------------------
    @property
    def point_symbol(self) -> InsensitiveDict:
        """
        Gets/Sets the Point Symbol for Point Geometries

        :return:
            ``InsensitiveDict``: A case-insensitive ``dict`` like object used to update and alter JSON
            A variants of a case-less dictionary that allows for dot and bracket notation.
        """
        if self._point_symbol is None:
            self._point_symbol = InsensitiveDict(
                dict(create_symbol(geometry_type="point"))
            )
        return self._point_symbol

    # ----------------------------------------------------------------------
    @point_symbol.setter
    def point_symbol(self, value: dict):
        """
        Gets/Sets the Point Symbol for Point Geometries

        :return:
            ``InsensitiveDict``: A case-insensitive ``dict`` like object used to update and alter JSON
            A variants of a case-less dictionary that allows for dot and bracket notation.
        """
        if isinstance(value, dict):
            self._point_symbol = InsensitiveDict(value)
        elif value is None:
            self._point_symbol = InsensitiveDict(
                dict(create_symbol(geometry_type="point"))
            )
        else:
            self._point_symbol = InsensitiveDict(value)

    # ----------------------------------------------------------------------
    @property
    def line_symbol(self) -> InsensitiveDict:
        """
        Gets/Sets the Line Symbol for Polyline Geometries

        :return:
            ``InsensitiveDict``: A case-insensitive ``dict`` like object used to update and alter JSON
            A variants of a case-less dictionary that allows for dot and bracket notation.
        """
        if self._line_symbol is None:
            self._line_symbol = InsensitiveDict(
                dict(create_symbol(geometry_type="polyline"))
            )
        return self._line_symbol

    # ----------------------------------------------------------------------
    @line_symbol.setter
    def line_symbol(self, value: dict):
        """
        Gets/Sets the Line Symbol for Polyline Geometries

        :return:
            ``InsensitiveDict``: A case-insensitive ``dict`` like object used to update and alter JSON
            A variants of a case-less dictionary that allows for dot and bracket notation.
        """
        if isinstance(value, dict):
            self._line_symbol = InsensitiveDict(value)
        elif value is None:
            self._line_symbol = InsensitiveDict(
                dict(create_symbol(geometry_type="polyline"))
            )
        else:
            self._line_symbol = InsensitiveDict(value)

    # ----------------------------------------------------------------------
    @property
    def polygon_symbol(self) -> InsensitiveDict:
        """
        Gets/Sets the Polygon Symbol for Polygon Geometries

        :return:
            ``InsensitiveDict``: A case-insensitive ``dict`` like object used to update and alter JSON
            A variants of a case-less dictionary that allows for dot and bracket notation.
        """
        if self._polygon_symbol is None:
            self._polygon_symbol = InsensitiveDict(
                dict(create_symbol(geometry_type="polygon"))
            )
        return self._polygon_symbol

    # ----------------------------------------------------------------------
    @polygon_symbol.setter
    def polygon_symbol(self, value: dict):
        """
        Gets/Sets the Polygon Symbol for Polygon Geometries

        :return:
            ``InsensitiveDict``: A case-insensitive ``dict`` like object used to update and alter JSON
            A variants of a case-less dictionary that allows for dot and bracket notation.
        """
        if isinstance(value, dict):
            self._polygon_symbol = InsensitiveDict(value)
        elif value is None:
            self._polygon_symbol = InsensitiveDict(
                dict(create_symbol(geometry_type="polygon"))
            )
        else:
            self._polygon_symbol = InsensitiveDict(value)

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self) -> dict:
        """Represents the MapView's widget JSON format"""
        add_layer = {
            "type": self._type,
            "url": self._url,
            "opacity": self.opacity,
            "minScale": self.scale[0],
            "maxScale": self.scale[1],
            "pointSymbol": self.point_symbol._json(),
            "polygonSymbol": self.polygon_symbol._json(),
            "lineSymbol": self.line_symbol._json(),
            "id": self._id,
            "title": self.title,
        }
        return add_layer

    @property
    def _operational_layer_json(self) -> dict:
        """Represents the WebMap's JSON format"""
        return self._lyr_json
