import json
import uuid
import re
from io import BytesIO, StringIO
import xml.etree.cElementTree as ET
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs, ParseResult

from arcgis.gis import GIS
from arcgis import env as _env
from arcgis._impl.common._mixins import PropertyMap
from ._base import BaseOGC


###########################################################################
class WMSLayer(BaseOGC):
    """
    Represents a Web Map Service, which is an OGC web service endpoint.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required string. The administration URL for the ArcGIS Server.
    ---------------     --------------------------------------------------------------------
    version             Optional String. The version number of the WMS service.  The default is `1.3.0`.
    ---------------     --------------------------------------------------------------------
    gis                 Optional :class:`~arcgis.gis.GIS`. The GIS used to reference the service by. The arcgis.env.active_gis is used if not specified.
    ---------------     --------------------------------------------------------------------
    copyright           Optional String. Describes limitations and usage of the data.
    ---------------     --------------------------------------------------------------------
    scale               Optional Tuple. The min/max scale of the layer where the positions are: (min, max) as float values.
    ---------------     --------------------------------------------------------------------
    opacity             Optional Float.  This value can range between 1 and 0, where 0 is 100 percent transparent and 1 is completely opaque.
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
    _type = "WMS"

    # ----------------------------------------------------------------------
    def __init__(self, url, version="1.3.0", gis=None, **kwargs):
        super(WMSLayer, self)
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
        self._title = kwargs.pop("title", "WMS Layer")
        self._gis = gis
        if url[-1] == "/":
            url = url[:-1]
        self._url = url
        self._add_token = str(self._con._auth).lower() == "builtin"
        self._opacity = kwargs.pop("opacity", 1)
        self._min_scale, self._max_scale = kwargs.pop("scale", (0, 0))

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> PropertyMap:
        """
        Returns the properties of the Layer.

        :return: PropertyMap
        """
        if self._properties is None:
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
                raise Exception("Could not connect to the Web Map Service")
            sss = BytesIO()
            sss.write(text.encode())
            sss.seek(0)
            tree = ET.XML(text=sss.read())
            d = self._xml_to_dictionary(tree)
            self._properties = PropertyMap(d)
        return self._properties

    # ---------------------------------------------------------------------
    @property
    def _extents(self) -> list:
        """list of extents from the service in the form of
        [[minx, miny], [maxx, maxy]] for each entry in the list
        """
        try:
            bboxes = self.properties.WMS_Capabilities.Capability.Layer.BoundingBox
            output = []
            for bbox in bboxes:
                output.append(
                    [
                        [float(bbox["@minx"]), float(bbox["@miny"])],
                        [float(bbox["@maxx"]), float(bbox["@maxy"])],
                    ]
                )
            return output
        except Exception:
            return [
                [[0, 0], [0, 0]],
            ]

    @property
    def _spatial_references(self) -> list:
        try:
            crss = self.properties.WMS_Capabilities.Capability.Layer.CRS
            output = []
            for crs_str in crss:
                output += [int(crs_num) for crs_num in re.findall(r"[0-9]+", crs_str)]
            return output
        except Exception as e:
            return []

    # ----------------------------------------------------------------------
    @property
    def layers(self) -> list:
        """returns the layers of the WMS Layer"""
        try:
            return self.properties.WMS_Capabilities.Capability.Layer.Layer
        except:
            return self.properties.WMS_Capabilities.Capability.Layer

    # ----------------------------------------------------------------------
    def _capabilities_url(self, service_url: str, vendor_kwargs: dict = None) -> str:
        """Return a capabilities url"""
        pieces = urlparse(service_url)
        args = parse_qs(pieces.query)
        if "service" not in args:
            args["service"] = "WMS"
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
    def _format_tags(self, tag: str) -> str:
        """attempts to format tags by stripping out the {text} from the keys"""
        import re

        regex = r".*\}(.*)"
        matches = re.search(regex, tag)
        if matches:
            return matches.groups()[0]
        return tag

    # ----------------------------------------------------------------------
    def _xml_to_dictionary(self, t) -> dict:
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
    def _lyr_json(self) -> dict:
        """Represents the MapView's widget JSON format"""
        return {
            "type": self._type,
            "id": self._id,
            "title": self._title or "WMTS Layer",
            "url": self._url,
            "version": self._version,
            "sublayers": [{"name": lyr.Name} for lyr in self.layers],
            "minScale": self.scale[0],
            "maxScale": self.scale[1],
            "opacity": self.opacity,
        }

    @property
    def _operational_layer_json(self) -> dict:
        """Represents the WebMap's JSON format"""
        new_layer = self._lyr_json
        new_layer["layers"] = [
            {"name": subLyr.Name, "title": subLyr.Title} for subLyr in self.layers
        ]
        new_layer["visibleLayers"] = []
        if new_layer["layers"]:
            # Only have the first layer be the visible layer
            new_layer["visibleLayers"].append(new_layer["layers"][0]["name"])
        new_layer["extent"] = self._extents[0]
        new_layer["spatialReferences"] = self._spatial_references
        return new_layer
