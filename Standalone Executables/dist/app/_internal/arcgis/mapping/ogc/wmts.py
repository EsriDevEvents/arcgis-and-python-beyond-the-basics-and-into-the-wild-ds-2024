import json
import uuid
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs, ParseResult
import xml.etree.cElementTree as ET
from io import BytesIO, StringIO

from arcgis.gis import GIS
from arcgis import env as _env
from arcgis._impl.common._mixins import PropertyMap

from ._base import BaseOGC


###########################################################################
class WMTSLayer(BaseOGC):
    """
    Represents a Web Map Tile Service, which is an OGC web service endpoint.


    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required string. The web address of the endpoint.
    ---------------     --------------------------------------------------------------------
    version             Optional String. The version number of the WMTS service.  The default is `1.0.0`
    ---------------     --------------------------------------------------------------------
    gis                 Optional :class:`~arcgis.gis.GIS` . The GIS used to reference the service by. The arcgis.env.active_gis is used if not specified.
    ---------------     --------------------------------------------------------------------
    copyright           Optional String. Describes limitations and usage of the data.
    ---------------     --------------------------------------------------------------------
    opacity             Optional Float.  This value can range between 1 and 0, where 0 is 100 percent transparent and 1 is completely opaque.
    ---------------     --------------------------------------------------------------------
    scale               Optional Tuple. The min/max scale of the layer where the positions are: (min, max) as float values.
    ---------------     --------------------------------------------------------------------
    title               Optional String. The title of the layer used to identify it in places such as the Legend and Layer List widgets.
    ===============     ====================================================================



    """

    _gis = None
    _con = None
    _url = None
    _reader = None
    _cap_reader = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, version="1.0.0", gis=None, **kwargs):
        super(WMTSLayer, self)
        if gis:
            gis = gis
        elif gis is None and _env.active_gis:
            gis = _env.active_gis
        else:
            gis = GIS()
        assert isinstance(gis, GIS)
        self._id = kwargs.pop("id", uuid.uuid4().hex)
        self._version = version
        self._con = gis._con
        self._title = kwargs.pop("title", "WMTS Layer")
        self._gis = gis
        if url[-1] == "/":
            url = url[:-1]
        self._url = url
        self._add_token = str(self._con._auth).lower() == "builtin"
        self._min_scale, self._max_scale = kwargs.pop("scale", (0, 0))
        self._opacity = kwargs.pop("opacity", 1)
        self._type = "WebTiledLayer"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Returns the properties of the Layer.

        :return: PropertyMap
        """
        if self._properties is None:
            from arcgis._impl.common._mixins import PropertyMap

            if self._add_token:
                url = self._capabilities_url(
                    service_url=self._url, vendor_kwargs={"token": self._con.token}
                )
            else:
                url = self._capabilities_url(service_url=self._url)
            text = self._con.get(url, {}, try_json=False, add_token=False)
            if text.find("Invalid Token") > -1 or text.find("Get Token") > -1:
                url = self._capabilities_url(service_url=self._url)
                text = self._con.get(url, {}, try_json=False, add_token=False)
            elif text.lower().find("<html>") > -1:
                url = self._capabilities_url(service_url=self._url)
                text = self._con.get(url, {}, try_json=False, add_token=False)
            elif text.lower().find("<?xml version=") > -1:
                pass
            else:
                raise Exception("Could not connect to the WebMap Tile Service")
            sss = BytesIO()
            sss.write(text.encode())
            sss.seek(0)
            tree = ET.XML(text=sss.read())
            d = self._xml_to_dictionary(tree)
            self._properties = PropertyMap(d)
        return self._properties

    # ----------------------------------------------------------------------
    def _capabilities_url(self, service_url, vendor_kwargs=None):
        """Return a capabilities url"""
        pieces = urlparse(service_url)
        args = parse_qs(pieces.query)
        if "service" not in args:
            args["service"] = "WMTS"
        if "request" not in args:
            args["request"] = "GetCapabilities"
        if "version" not in args:
            args["version"] = self._version
        if vendor_kwargs:
            args.update(vendor_kwargs)
        query = urlencode(args, doseq=True)
        pieces = ParseResult(
            pieces.scheme,
            pieces.netloc,
            pieces.path,
            pieces.params,
            query,
            pieces.fragment,
        )
        return urlunparse(pieces)

    # ----------------------------------------------------------------------
    def _format_tags(self, tag):
        """attempts to format tags by stripping out the {text} from the keys"""
        import re

        regex = r".*\}(.*)"
        matches = re.search(regex, tag)
        if matches:
            return matches.groups()[0]
        return tag

    # ----------------------------------------------------------------------
    def _xml_to_dictionary(self, t):
        """converts the xml to a dictionary object (recursivly)"""
        import json
        from collections import defaultdict

        d = {self._format_tags(t.tag): {} if t.attrib else None}
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(self._xml_to_dictionary, children):
                for k, v in dc.items():
                    dd[self._format_tags(k)].append(v)
            d = {
                self._format_tags(t.tag): {
                    self._format_tags(k): v[0] if len(v) == 1 else v
                    for k, v in dd.items()
                }
            }
        if t.attrib:
            d[self._format_tags(t.tag)].update(
                [("@" + self._format_tags(k), v) for k, v in t.attrib.items()]
            )
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[self._format_tags(t.tag)]["#text"] = text
            else:
                d[self._format_tags(t.tag)] = text
        removals = [
            "{http://www.opengis.net/wmts/1.0}",
            "{http://www.opengis.net/ows/1.1}",
            "{http://www.w3.org/1999/xlink}",
        ]
        d = json.dumps(d)
        for remove in removals:
            d = d.replace(remove, "")
        return json.loads(d)

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        """Represents the MapView's widget JSON format"""
        return {
            "id": self._id,
            "title": self._title or "WMTS Layer",
            "url": self._url,
            "version": self._version,
            "minScale": self.scale[0],
            "maxScale": self.scale[1],
            "opacity": self.opacity,
            "type": self._type,
        }

    # ----------------------------------------------------------------------
    @property
    def __text__(self):
        """creates the item's text properties"""

        layer = None
        tile_matrix = None

        if isinstance(self.properties.Capabilities.Contents.Layer, (list, tuple)):
            layer = self.properties.Capabilities.Contents.Layer[0]
            tile_matrix = self.properties.Capabilities.Contents.TileMatrixSet[0]
        elif isinstance(
            self.properties.Capabilities.Contents.Layer, (dict, PropertyMap)
        ):
            layer = self.properties.Capabilities.Contents.Layer
            tile_matrix = self.properties.Capabilities.Contents.TileMatrixSet
        else:
            raise ValueError("Could not parse the results properly.")

        url_template = (
            layer.ResourceURL["@template"]
            .replace("{TileMatrix}", "{level}")
            .replace("{Style}", layer.Style.Identifier)
            .replace("{TileRow}", "{row}")
            .replace("{TileCol}", "{col}")
            .replace("{TileMatrixSet}", tile_matrix.Identifier)
        )
        fullExtent = [
            float(coord) for coord in layer.BoundingBox.LowerCorner.strip().split(" ")
        ] + [float(coord) for coord in layer.BoundingBox.UpperCorner.strip().split(" ")]
        lods = []
        WMTS_DPI = 90.71428571428571
        for l in tile_matrix.TileMatrix:
            lods.append(
                {
                    "level": int(l.Identifier),
                    "levelValue": l.Identifier,
                    "resolution": float(l.ScaleDenominator) * 0.00028,
                    "scale": float(l.ScaleDenominator) * WMTS_DPI / 96,
                }
            )
        return {
            "templateUrl": url_template,
            "copyright": "",
            "fullExtent": {
                "xmin": fullExtent[0],
                "ymin": fullExtent[1],
                "xmax": fullExtent[2],
                "ymax": fullExtent[3],
                "spatialReference": {
                    "wkid": int(layer.BoundingBox["@crs"].split(":")[-1])
                },
            },
            "tileInfo": {
                "rows": 256,
                "cols": 256,
                "dpi": 96,
                "origin": {
                    "x": (fullExtent[2] + fullExtent[0]) / 2,
                    "y": (fullExtent[3] + fullExtent[1]) / 2,
                    "spatialReference": {
                        "wkid": int(layer.BoundingBox["@crs"].split(":")[-1])
                    },
                },
                "spatialReference": {
                    "wkid": int(layer.BoundingBox["@crs"].split(":")[-1])
                },
                "lods": lods,
            },
            "wmtsInfo": {
                "url": self._url,
                "layerIdentifier": layer.Title,
                "tileMatrixSet": [tile_matrix.Identifier],
            },
        }

    @property
    def _operational_layer_json(self):
        """Represents the WebMap's JSON format"""
        return self.__text__
