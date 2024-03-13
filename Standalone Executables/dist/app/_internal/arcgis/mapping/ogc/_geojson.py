import json
import uuid
from arcgis.gis import GIS
from arcgis._impl.common._isd import InsensitiveDict
from arcgis.gis._impl._con._url_validator import validate_url
from pathlib import Path
from ._base import BaseOGC


def _is_file(path):
    """checks if the data is a file"""
    return Path(path).is_file()


###########################################################################
class GeoJSONLayer(BaseOGC):
    """
    The GeoJSONLayer class is used to create a layer based on GeoJSON.
    GeoJSON is a format for encoding a variety of geographic data
    structures. The GeoJSON data must comply with the RFC 7946
    specification which states that the coordinates are in
    spatial reference: WGS84 (wkid 4326).


    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Optional String. The web location of the GeoJSON file.
    ---------------     --------------------------------------------------------------------
    data                Optional String or Dict. A path to a GeoJSON file, the GeoJSON data as a string, or the GeoJSON data as a dictionary.
    ---------------     --------------------------------------------------------------------
    copyright           Optional String. Describes limitations and usage of the data.
    ---------------     --------------------------------------------------------------------
    opacity             Optional Float.  This value can range between 1 and 0, where 0 is 100 percent transparent and 1 is completely opaque.
    ---------------     --------------------------------------------------------------------
    renderer            Optional Dictionary. A custom set of symbology for the given geojson dataset.
    ---------------     --------------------------------------------------------------------
    scale               Optional Tuple. The min/max scale of the layer where the positions are: (min, max) as float values.
    ---------------     --------------------------------------------------------------------
    title               Optional String. The title of the layer used to identify it in places such as the Legend and Layer List widgets.
    ===============     ====================================================================


    """

    _type = "geojson"
    _url = ""
    _data = {}

    # ----------------------------------------------------------------------
    def __init__(self, url=None, data=None, **kwargs):
        """init"""
        super(GeoJSONLayer, self)
        if url is None and data is None:
            raise Exception("A `url` or `data` must be given to proceed")
        if isinstance(data, str) and _is_file(data):
            with open(data, "r") as r:
                self._data = json.loads(r.read())
        elif isinstance(data, str) and _is_file(data) == False:
            self._data = json.loads(data)
        elif isinstance(data, dict):
            self._data = dict(data)
        elif url is None and data and not isinstance(data, (str, dict)):
            raise ValueError("`data` must be of type string or dict.")
        if url and validate_url(url) == False:
            raise ValueError(f"Invalid `url` : {url}")
        self._url = url
        self._type = "GeoJSON"
        self._copyright = kwargs.pop("copyright", "")
        self._title = kwargs.pop("title", "GeoJSON Layer")
        self._id = kwargs.pop("id", uuid.uuid4().hex)  # hidden input, but accepted
        self._min_scale, self._max_scale = kwargs.pop("scale", (0, 0))
        self._opacity = kwargs.pop("opacity", 1)
        if "renderer" in kwargs:
            r = kwargs.pop("renderer", None)
            if isinstance(r, dict):
                self._renderer = InsensitiveDict(r)
            else:
                self._renderer = None
        else:
            self._renderer = None

    # ----------------------------------------------------------------------
    @property
    def renderer(self) -> InsensitiveDict:
        """Gets/Sets the renderer for the layer"""
        return self._renderer

    # ----------------------------------------------------------------------
    @renderer.setter
    def renderer(self, renderer: dict):
        """Gets/Sets the renderer for the layer"""
        if isinstance(renderer, dict) and renderer:
            self._renderer = InsensitiveDict(renderer)

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self) -> dict:
        """Represents the MapView widget's JSON format"""
        lyr = {
            "type": self._type,
            "url": self._url,
            "data": self._data,
            "copyright": self._copyright,
            "title": self._title,
            "id": self._id,
            "minScale": self.scale[0],
            "maxScale": self.scale[1],
            "opacity": self._opacity,
        }
        if self._renderer:
            lyr["renderer"] = self._renderer._json()
        return lyr

    @property
    def _operational_layer_json(self) -> dict:
        """Represents the WebMap's JSON format"""
        return self._lyr_json

    @property
    def url(self):
        """
        Get/Set the data associated with the GeoJSON Layer

        :return: String

        """
        return self._url or self._data

    @url.setter
    def url(self, data):
        """
        Get/Set the data associated with the GeoJSON Layer

        The data can be a string, file or URL.

        :return: String
        """

        if validate_url(data):
            self._url = data
        elif _is_file(data):
            with open(data, "r") as r:
                self._data = r.read()
        elif isinstance(data, str):
            self._data = data
        else:
            raise ValueError("The data must be a valid URL, file, or text blob.")
