"""
New Geometries Classes
"""
from __future__ import annotations
from arcgis.auth.tools import LazyLoader
import copy
import json
from typing import Any, Optional, Union
import ujson as _ujson

try:
    import numpy as np
except ImportError as e:
    pass
from functools import partial, lru_cache

_number_type = (int, float)
_empty_value = [None, "NaN"]

try:
    arcpy = LazyLoader("arcpy", strict=True)
    _HASARCPY = True
except:
    _HASARCPY = False
try:
    shapely = LazyLoader("shapely", strict=True)
    _HASSHAPELY = True
except:
    _HASSHAPELY = False


@lru_cache(maxsize=100)
def _check_geometry_engine():
    _HASARCPY = True
    try:
        arcpy = LazyLoader("arcpy", strict=True)
    except:
        _HASARCPY = False

    _HASSHAPELY = True
    try:
        shapely = LazyLoader("shapely", strict=True)
    except:
        _HASSHAPELY = False

    return _HASARCPY, _HASSHAPELY


def _is_valid(value):
    """checks if the value is valid"""

    if not isinstance(value.get("spatialReference", None), (dict, SpatialReference)):
        return False

    if isinstance(value, Point):
        if hasattr(value, "x") and hasattr(value, "y"):
            return True
        elif "x" in value and (value["x"] in _empty_value):
            return True
        return False
    elif isinstance(value, Envelope):
        if all(
            isinstance(getattr(value, extent, None), _number_type)
            for extent in ("xmin", "ymin", "xmax", "ymax")
        ):
            return True
        elif hasattr(value, "xmin") and (value.xmin in _empty_value):
            return True

        return False
    elif isinstance(value, (MultiPoint, Polygon, Polyline)):
        if "paths" in value:
            if len(value["paths"]) == 0:
                return True
            return _is_line(coords=value["paths"])
        elif "rings" in value:
            if len(value["rings"]) == 0:
                return True
            return _is_polygon(coords=value["rings"])
        elif "points" in value:
            if len(value["points"]) == 0:
                return True
            return _is_point(coords=value["points"])

    return False


def _is_polygon(coords):
    for coord in coords:
        if len(coord) < 4:
            return False
        if not _is_line(coord):
            return False
        if coord[0] != coord[-1]:
            return False

    return True


def _is_line(coords):
    """
    checks to see if the line has at
    least 2 points in the list
    """
    list_types = (list, tuple, set)
    if isinstance(coords, list_types) and len(coords) > 0:
        return all(_is_point(elem) for elem in coords)

    return True


def _is_point(coords):
    """
    checks to see if the point has at
    least 2 coordinates in the list
    """
    valid = False
    if isinstance(coords, (list, tuple)) and len(coords) > 1:
        for coord in coords:
            if not isinstance(coord, _number_type):
                if not _is_point(coord):
                    return False
            valid = True

    return valid


def _geojson_type_to_esri_type(type_):
    mapping = {
        "LineString": Polyline,
        "MultiLineString": Polyline,
        "Polygon": Polygon,
        "MultiPolygon": Polygon,
        "Point": Point,
        "MultiPoint": MultiPoint,
    }
    if mapping.get(type_):
        return mapping[type_]
    else:
        raise ValueError("Unknown GeoJSON Geometry type: {}".format(type_))


class BaseGeometry(dict):
    _ao = None
    _type = None
    _typ = None
    _HASARCPY = None
    _HASSHAPELY = None
    _class_attributes = {
        "_ao",
        "_type",
        "_typ",
        "_HASARCPY",
        "_HASSHAPELY",
        "_ipython_canary_method_should_not_exist_",
    }

    def __init__(self, iterable=None):
        if iterable is None:
            iterable = {}
        self.update(iterable)

    def is_valid(self):
        return _is_valid(self)

    @lru_cache(maxsize=10)
    def _check_geometry_engine(self):
        self._HASARCPY, self._HASSHAPELY = _check_geometry_engine()
        return self._HASARCPY, self._HASSHAPELY

    def __setattr__(self, key, value):
        """sets the attribute"""
        if key in self._class_attributes:
            super(BaseGeometry, self).__setattr__(key, value)
        else:
            self[key] = value
            self._ao = None

    def __setattribute__(self, key, value):
        if key in self._class_attributes:
            super(BaseGeometry, self).__setattr__(key, value)
        else:
            self[key] = value
            self._ao = None

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self._ao = None

    def __getattribute__(self, name):
        return super(BaseGeometry, self).__getattribute__(name)

    def __getattr__(self, name):
        try:
            if name in self._class_attributes:
                return super(BaseGeometry, self).__getattr__(name)
            return self.__getitem__(name)
        except:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    def __getitem__(self, k):
        return dict.__getitem__(self, k)


class GeometryFactory(type):
    """
    Creates the Geometry Objects Based on JSON
    """

    @staticmethod
    def _from_wkb(iterable):
        if _HASARCPY:
            return _ujson.loads(arcpy.FromWKB(iterable).JSON)
        else:
            from geomet import wkb

            return wkb.loads(iterable)
        return {}

    @staticmethod
    def _from_wkt(iterable):
        if _HASARCPY:
            if "SRID=" in iterable:
                wkid, iterable = iterable.split(";")
                geom = _ujson.loads(arcpy.FromWKT(iterable).JSON)
                geom["spatialReference"] = {"wkid": int(wkid.replace("SRID=", ""))}
                return geom
            return _ujson.loads(arcpy.FromWKT(iterable).JSON)
        else:
            from geomet.wkt import loads as _wkt_loads
            from geomet.esri import dumps as _esri_dumps

            if "SRID=" in iterable:
                wkid, iterable = iterable.split(";")
                geom = _esri_dumps(_wkt_loads(iterable))
                geom["spatialReference"] = {"wkid": int(wkid.replace("SRID=", ""))}
                return geom
        return {}

    @staticmethod
    def _from_gj(iterable):
        global _HASARCPY
        if _HASARCPY:
            gj = _ujson.loads(arcpy.AsShape(iterable, False).JSON)
            gj["spatialReference"]["wkid"] = 4326
            return gj
        else:
            sr = iterable.pop("sr", None)
            cls = _geojson_type_to_esri_type(iterable["type"])
            return cls._from_geojson(iterable, sr=sr)

    def __call__(cls, iterable=None, **kwargs):
        if iterable is None:
            iterable = {}

        if iterable:
            # WKB
            if isinstance(iterable, (bytearray, bytes)):
                iterable = GeometryFactory._from_wkb(iterable)
            elif hasattr(iterable, "JSON"):
                iterable = _ujson.loads(getattr(iterable, "JSON"))
            elif "coordinates" in iterable:
                iterable["sr"] = kwargs.pop("sr", None)
                iterable = GeometryFactory._from_gj(iterable)
            elif hasattr(iterable, "exportToString"):
                iterable = {"wkt": iterable.exportToString()}
            elif isinstance(iterable, str) and "{" in iterable:
                iterable = _ujson.loads(iterable)
            elif isinstance(iterable, str):  # WKT
                iterable = GeometryFactory._from_wkt(iterable)

            if "x" in iterable:
                cls = Point
            elif "rings" in iterable or "curveRings" in iterable:
                cls = Polygon
            elif "curvePaths" in iterable or "paths" in iterable:
                cls = Polyline
            elif "points" in iterable:
                cls = MultiPoint
            elif "xmin" in iterable:
                cls = Envelope
            elif "wkid" in iterable or "wkt" in iterable:
                return SpatialReference(iterable=iterable)
            elif isinstance(iterable, list):
                return Point(
                    {
                        "x": iterable[0],
                        "y": iterable[1],
                        "spatialReference": {"wkid": kwargs.pop("wkid", 4326)},
                    }
                )
            else:
                cls = Geometry
        return type.__call__(cls, iterable, **kwargs)


class Geometry(BaseGeometry, metaclass=GeometryFactory):
    """
    The base class for all geometries.

    You can create a Geometry even when you don't know the exact type. The Geometry constructor is able
    to figure out the geometry type and returns the correct type as the example below demonstrates:

    .. code-block:: python

        #Usage Example: Unknown Geometry

        >>> geom = Geometry({
        >>>     "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
        >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
        >>>               [-97.06326,32.759]]],
        >>>     "spatialReference" : {"wkid" : 4326}
        >>>                 })
        >>> print (geom.type) # POLYGON
        >>> print (isinstance(geom, Polygon) # True

    """

    def __init__(self, iterable=None, **kwargs):
        if iterable is None:
            iterable = ()
        super(Geometry, self).__init__(iterable, **kwargs)

    @property
    def __geo_interface__(self):
        """
        Converts an ESRI JSON to GeoJSON

        :return: string
        """
        _HASARCPY, _HASSHAPELY = _check_geometry_engine()
        if _HASARCPY:
            if isinstance(self.as_arcpy, arcpy.Point):
                return arcpy.PointGeometry(self.as_arcpy).__geo_interface__
            else:
                return self.as_arcpy.__geo_interface__
        elif _HASSHAPELY:
            if isinstance(self, Point):
                return {"type": "Point", "coordinates": (self.x, self.y)}
            elif isinstance(self, Polygon):
                col = []
                for part in self["rings"]:
                    col.append([tuple(pt) for pt in part])
                return {"coordinates": [col], "type": "MultiPolygon"}
            elif isinstance(self, Polyline):
                return {
                    "type": "MultiLineString",
                    "coordinates": [
                        [((pt[0], pt[1]) if pt else None) for pt in part]
                        for part in self["paths"]
                    ],
                }
            elif isinstance(self, MultiPoint):
                return {
                    "type": "Multipoint",
                    "coordinates": [(pt[0], pt[1]) for pt in self["points"]],
                }
            from arcgis._impl.common._arcgis2geojson import arcgis2geojson

            return arcgis2geojson(arcgis=self)
        else:
            from geomet import esri

            str_item = json.dumps(dict(self))
            return esri.loads(str_item)

    def __sub__(self, other):
        """
        Constructs the :class:`~arcgis.geometry.Geometry` that is composed only of the region unique to
        the base geometry but not part of the other geometry.
        """
        return self.difference(other)

    def __xor__(self, other):
        """
        Constructs the :class:`~arcgis.geometry.Geometry` that is the union of two geometries minus
        the instersection of those geometries.
        """
        return self.symmetric_difference(other)

    def __or__(self, other):
        """
        Constructs the :class:`~arcgis.geometry.Geometry` object that is the set-theoretic union
        of the input geometries.
        """
        return self.union(other)

    def __add__(self, other):
        """
        Constructs a geometry that is the geometric intersection of the two
        input geometries. Different dimension values can be used to create
        different shape types.
        """
        dimension = 1
        if isinstance(other, Polyline):
            dimension = 2
        elif isinstance(other, Polygon):
            dimension = 4
        return self.intersect(second_geometry=other, dimension=dimension)

    def __hash__(self):
        return hash(frozenset(self.items()))

    def __iter__(self):
        """
        Iterator for the Geometry
        """
        if isinstance(self, Polygon):
            avgs = []
            shape = 2
            for ring in self["rings"]:
                np_array_ring = np.array(ring)
                shape = np_array_ring.shape[1]
                avgs.append([np_array_ring[:, 0].mean(), np_array_ring[:, 1].mean()])

            avgs = np.array(avgs)
            res = []
            if shape == 2:
                res = [avgs[:, 0].mean(), avgs[:, 1].mean()]
            elif shape > 2:
                res = [
                    avgs[:, 0].mean(),
                    avgs[:, 1].mean(),
                    avgs[:, 2].mean(),
                ]
            for a in res:
                yield a
                del a
        elif isinstance(self, Polyline):
            avgs = []
            shape = 2
            for ring in self["paths"]:
                np_array_ring = np.array(ring)
                shape = np_array_ring.shape[1]
                avgs.append([np_array_ring[:, 0].mean(), np_array_ring[:, 1].mean()])
            avgs = np.array(avgs)
            res = []
            if shape == 2:
                res = [avgs[:, 0].mean(), avgs[:, 1].mean()]
            elif shape > 2:
                res = [
                    avgs[:, 0].mean(),
                    avgs[:, 1].mean(),
                    avgs[:, 2].mean(),
                ]
            for a in res:
                yield a
                del a
        elif isinstance(self, MultiPoint):
            a = np.array(self["points"])
            if a.shape[1] == 2:
                for i in [a[:, 0].mean(), a[:, 1].mean()]:
                    yield i
            elif a.shape[1] >= 3:  # has z
                for i in [a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()]:
                    yield i
        elif isinstance(self, Point):
            keys = ["x", "y", "z"]
            for k in keys:
                if k in self:
                    yield self[k]
                del k
        elif isinstance(self, Envelope):
            for i in [
                (self["xmin"] + self["xmax"]) / 2,
                (self["ymin"] + self["ymax"]) / 2,
            ]:
                yield i

    def _repr_svg_(self):
        """SVG representation for iPython notebook"""
        svg_top = (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
        )
        if self.is_empty:
            return svg_top + "/>"
        else:
            # Establish SVG canvas that will fit all the data + small space
            xmin, ymin, xmax, ymax = self.extent
            # Expand bounds by a fraction of the data ranges
            expand = 0.04  # or 4%, same as R plots
            widest_part = max([xmax - xmin, ymax - ymin])
            expand_amount = widest_part * expand

            if xmin == xmax and ymin == ymax:
                # This is a point; buffer using an arbitrary size
                try:
                    xmin, ymin, xmax, ymax = self.buffer(1).extent
                except:
                    xmin -= expand_amount
                    ymin -= expand_amount
                    xmax += expand_amount
                    ymax += expand_amount
            else:
                xmin -= expand_amount
                ymin -= expand_amount
                xmax += expand_amount
                ymax += expand_amount

            dx = xmax - xmin
            dy = ymax - ymin
            width = min([max([100.0, dx]), 300])
            height = min([max([100.0, dy]), 300])

            try:
                scale_factor = max([dx, dy]) / max([width, height])
            except ZeroDivisionError:
                scale_factor = 1.0

            view_box = "{0} {1} {2} {3}".format(xmin, ymin, dx, dy)
            transform = "matrix(1,0,0,-1,0,{0})".format(ymax + ymin)
            return svg_top + (
                'width="{1}" height="{2}" viewBox="{0}" '
                'preserveAspectRatio="xMinYMin meet">'
                '<g transform="{3}">{4}</g></svg>'
            ).format(view_box, width, height, transform, self.svg(scale_factor))

    @property
    def as_arcpy(self):
        """
        The ``as_arcpy`` method retrieves the Geometry as an ArcPy Geometry.

        If `ArcPy` is not installed, none is returned.

        .. note::
            The ``as_arcpy`` method requires ArcPy

        :return:
            An :class:`~arcgis.geometry.Geometry` object

        """
        _HASARCPY, _HASSHAPELY = _check_geometry_engine()
        if self._ao is not None or not _HASARCPY:
            return self._ao
        if isinstance(self, (Point, MultiPoint, Polygon, Polyline)):
            self._ao = arcpy.AsShape(json.dumps(dict(self)), True)
        elif isinstance(self, SpatialReference):
            if "wkid" in self:
                self._ao = arcpy.SpatialReference(self["wkid"])
            elif "wkt" in self:
                self._ao = arcpy.SpatialReference(self["wkt"])
            else:
                raise ValueError("Invalid SpatialReference")
        elif isinstance(self, Envelope):
            return arcpy.Extent(
                XMin=self["xmin"],
                YMin=self["ymin"],
                XMax=self["xmax"],
                YMax=self["ymax"],
            )
        return self._ao

    def _wkt(obj, fmt="%.16f"):
        """converts an arcgis.Geometry to WKT"""
        if isinstance(obj, Point):
            coords = [obj["x"], obj["y"]]
            if "z" in obj:
                coords.append(obj["z"])
            return "POINT (%s)" % " ".join(fmt % c for c in coords)
        elif isinstance(obj, Polygon):
            coords = obj["rings"]
            pt2 = []
            b = "MULTIPOLYGON (%s)"
            for part in coords:
                c2 = []
                for c in part:
                    c2.append("(%s,  %s)" % (fmt % c[0], fmt % c[1]))
                j = "(%s)" % ", ".join(c2)
                pt2.append(j)
            b = b % ", ".join(pt2)
            return b
        elif isinstance(obj, Polyline):
            coords = obj["paths"]
            pt2 = []
            b = "MULTILINESTRING (%s)"
            for part in coords:
                c2 = []
                for c in part:
                    c2.append("(%s,  %s)" % (fmt % c[0], fmt % c[1]))
                j = "(%s)" % ", ".join(c2)
                pt2.append(j)
            b = b % ", ".join(pt2)
            return b
        elif isinstance(obj, MultiPoint):
            coords = obj["points"]
            b = "MULTIPOINT (%s)"
            c2 = []
            for c in coords:
                c2.append("(%s,  %s)" % (fmt % c[0], fmt % c[1]))
            return b % ", ".join(c2)
        return ""

    @property
    def geoextent(self):
        """
        The ``geoextent`` property retrieves the current feature's extent

        .. code-block:: python

            #Usage Example
            >>> g = Geometry({...})
            >>> g.geoextent
            (1,2,3,4)

        :return: tuple
        """
        _HASARCPY, _HASSHAPELY = _check_geometry_engine()

        if not hasattr(self, "type"):
            return None

        a = None
        if str(self.type).upper() == "POLYGON":
            if "rings" in self:
                a = self["rings"]
            elif "curveRings" in self:
                if not _HASARCPY:
                    raise Exception(
                        "Cannot calculate the geoextent with curves without ArcPy."
                    )
                return (
                    self.as_arcpy.extent.XMin,
                    self.as_arcpy.extent.YMin,
                    self.as_arcpy.extent.XMax,
                    self.as_arcpy.extent.YMax,
                )
        elif str(self.type).upper() == "POLYLINE":
            if "paths" in self:
                a = self["paths"]
            elif "curvePaths" in self:
                if not _HASARCPY:
                    raise Exception(
                        "Cannot calculate the geoextent with curves without ArcPy."
                    )
                return (
                    self.as_arcpy.extent.XMin,
                    self.as_arcpy.extent.YMin,
                    self.as_arcpy.extent.XMax,
                    self.as_arcpy.extent.YMax,
                )
        elif str(self.type).upper() == "MULTIPOINT":
            a = np.array(self["points"])
            x_max = max(a[:, 0])
            x_min = min(a[:, 0])
            y_min = min(a[:, 1])
            y_max = max(a[:, 1])
            return x_min, y_min, x_max, y_max
        elif str(self.type).upper() == "POINT":
            return self["x"], self["y"], self["x"], self["y"]
        elif str(self.type).upper() == "ENVELOPE":
            return tuple(self.coordinates().tolist())
        else:
            return None

        if a is None or len(a) == 0:
            return None

        if len(a) == 1:  # single part
            x_max = max(a[0], key=lambda x: x[0])[0]
            x_min = min(a[0], key=lambda x: x[0])[0]
            y_max = max(a[0], key=lambda x: x[1])[1]
            y_min = min(a[0], key=lambda x: x[1])[1]
            return x_min, y_min, x_max, y_max
        else:
            if "points" in a:
                a = a["points"]
            elif "coordinates" in a:
                a = a["coordinates"]
            xs = []
            ys = []
            for pt in a:  # multiple part geometry
                x_max = max(pt, key=lambda x: x[0])[0]
                x_min = min(pt, key=lambda x: x[0])[0]
                y_max = max(pt, key=lambda x: x[1])[1]
                y_min = min(pt, key=lambda x: x[1])[1]
                xs.append(x_max)
                xs.append(x_min)
                ys.append(y_max)
                ys.append(y_min)
                del pt
            return min(xs), min(ys), max(xs), max(ys)

    @property
    def envelope(self):
        """
        The ``envelope`` method retrieves the geoextent as an :class:`~arcgis.geometry.Envelope` object

        :return:
            :class:`~arcgis.geometry.Envelope`
        """
        env_dict = {
            "xmin": self.geoextent[0],
            "ymin": self.geoextent[1],
            "xmax": self.geoextent[2],
            "ymax": self.geoextent[3],
            "spatialReference": self.spatial_reference,
        }

        return Envelope(env_dict)

    def skew(self, x_angle: float = 0, y_angle: float = 0, inplace: bool = False):
        """
        Creates a skew transform along one or both axes.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        x_angle             optional Float. Angle to skew in the x coordinate
        ---------------     --------------------------------------------------------------------
        y_angle             Optional Float. Angle to skew in the y coordinate
        ---------------     --------------------------------------------------------------------
        inplace             Optional Boolean. If True, the value is updated in the object, False
                            creates a new object
        ===============     ====================================================================


        :return:
            A :class:`~arcgis.geometry.Geometry` object

        """
        from .affine import skew

        s = skew(geom=copy.deepcopy(self), x_angle=x_angle, y_angle=y_angle)
        if inplace:
            self.update(s)
        return s

    def rotate(self, theta: float, inplace: bool = False):
        """
        Rotates a :class:`~arcgis.geometry.Geometry` object counter-clockwise by a given angle.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        theta               Required Float. The rotation angle.
        ---------------     --------------------------------------------------------------------
        inplace             Optional Boolean. If True, the value is updated in the object, False
                            creates a new object
        ===============     ====================================================================


        :return:
            A :class:`~arcgis.geometry.Geometry` object

        """
        from .affine import rotate

        r = rotate(copy.deepcopy(self), theta)
        if inplace:
            self.update(r)
        return r

    def scale(self, x_scale: float = 1, y_scale: float = 1, inplace: bool = False):
        """
        Scales a :class:`~arcgis.geometry.Geometry` object in either the x,y or both directions.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        x_scale             Optional Float. The x-scale factor.
        ---------------     --------------------------------------------------------------------
        y_scale             Optional Float. The y-scale factor.
        ---------------     --------------------------------------------------------------------
        inplace             Optional Boolean. If True, the value is updated in the object, False
                            creates a new object
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom2 = geom.sacle(x_scale = 3,
                                   y_scale = 0.5,
                                   inplace = False)

        """
        from .affine import scale
        import copy

        g = copy.copy(self)
        s = scale(g, *(x_scale, y_scale))
        if inplace:
            self.update(s)
        return s

    def translate(
        self, x_offset: float = 0, y_offset: float = 0, inplace: bool = False
    ):
        """
        Moves a :class:`~arcgis.geometry.Geometry` object in the x and y direction by a given
        distance.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        x_offset            Optional Float. Translation x offset
        ---------------     --------------------------------------------------------------------
        y_offset            Optional Float. Translation y offset
        ---------------     --------------------------------------------------------------------
        inplace             Optional Boolean. If True, updates the existing Geometry,else it
                            creates a new Geometry object
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.translate(x_offset = 40,
                               y_offset = 50,
                               inplace = True)

        """
        from .affine import translate

        t = translate(copy.deepcopy(self), x_offset, y_offset)
        if inplace:
            self.update(t)
        return t

    @property
    def is_empty(self):
        """
        Determines if the geometry is empty.

        :return:
           A boolean indicating empty (True), or filled (False)
        """
        if isinstance(self, Point) and self.get("x", "NaN") != "NaN":
            return False
        elif isinstance(self, Polygon):
            if "rings" in self:
                return len(self["rings"]) == 0
            elif "curveRings" in self:
                return len(self["curveRings"]) == 0
        elif isinstance(self, Polyline):
            return len(self["paths"]) == 0
        elif isinstance(self, MultiPoint):
            return len(self["points"]) == 0
        return True

    @property
    def as_shapely(self):
        """
        The ``as_shapely`` method retrieves a shapely :class:`~arcgis.geometry.Geometry` object

        :return:
            A shapely :class:`~arcgis.geometry.Geometry` object.
            If shapely is not installed, None is returned
        """
        _, _HASSHAPELY = _check_geometry_engine()
        if _HASSHAPELY:
            if isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
                from shapely.geometry import shape

                if "curvePaths" in self or "curveRings" in self:
                    return {}
                return shape(self.__geo_interface__)
        return None

    @property
    def JSON(self):
        """
        The ``JSON`` method retrieves an Esri JSON representation of the :class:`~arcgis.geometry.Geometry` object as a
        string.

        :return:
            A string representing a :class:`~arcgis.geometry.Geometry` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self.as_arcpy, arcpy.Geometry):
            return getattr(self.as_arcpy, "JSON", None)
        elif "coordinates" in self:
            from geomet import esri

            return json.dumps(esri.dumps(dict(self)))
        return json.dumps(self)

    # ----------------------------------------------------------------------
    @classmethod
    def from_shapely(
        cls,
        shapely_geometry: Geometry,
        spatial_reference: Optional[dict[str, Any]] = None,
    ):
        """
        Creates a Python API Geometry object from a Shapely geometry object.

        .. note::
            Must have shapely installed

        =================   ====================================================================
        **Parameter**        **Description**
        -----------------   --------------------------------------------------------------------
        shapely_geometry    Required Shapely Geometry
                            Single instance of Shapely Geometry to be converted to ArcGIS
                            Python API geometry instance.
        -----------------   --------------------------------------------------------------------
        spatial_reference   Optional SpatialReference
                            Defines the spatial reference for the output geometry.
        =================   ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        .. code-block:: python

            # Usage Example: importing shapely geometry object and setting spatial reference to WGS84

            Geometry.from_shapely(
                shapely_geometry=shapely_geometry_object,
                spatial_reference={'wkid': 4326}
            )

        """
        if _HASSHAPELY:
            gj = shapely_geometry.__geo_interface__
            geom_cls = _geojson_type_to_esri_type(gj["type"])

            if spatial_reference:
                geometry = geom_cls._from_geojson(gj, sr=spatial_reference)
            else:
                geometry = geom_cls._from_geojson(gj)

            return geometry
        else:
            raise ValueError("Shapely is required to execute from_shapely.")

    # ----------------------------------------------------------------------
    @property
    def EWKT(self):
        """
        Gets the ``extended well-known text`` (`EWKT`) representation for OGC geometry.
        It provides a portable representation of a geometry value as a text
        string.

        .. note::
            Any true curves in the geometry will be densified into approximate
            curves in the WKT string.

        :return:
            A String
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, "polygon", None)
                sr = self.spatial_reference.get("wkid", 4326)
                return f"SRID={sr};{p.WKT}"
            except:
                return None
        if HASARCPY:
            sr = self.spatial_reference.get("wkid", 4326)
            return f"SRID={sr};{getattr(self.as_arcpy, 'WKT', None)}"
        elif HASSHAPELY:
            try:
                sr = self.spatial_reference.get("wkid", 4326)
                return f"SRID={sr};{self.as_shapely.wkt}"
            except:
                sr = self.spatial_reference.get("wkid", 4326)
                return f"SRID={sr};{self._wkt(fmt='%.16f')}"
        else:
            sr = self.spatial_reference.get("wkid", 4326)
            return f"SRID={sr};{self._wkt(fmt='%.16f')}"

    # ----------------------------------------------------------------------
    @property
    def WKT(self):
        """
        Gets the ``well-known text`` (``WKT``) representation for OGC geometry.
        It provides a portable representation of a geometry value as a text
        string.

        .. note::
            Any true curves in the geometry will be densified into approximate
            curves in the WKT string.

        :return:
            A string
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, "polygon", None)
                return p.WKT
            except:
                return None
        if HASARCPY:
            return getattr(self.as_arcpy, "WKT", None)
        elif HASSHAPELY:
            try:
                return self.as_shapely.wkt
            except:
                return self._wkt(fmt="%.16f")
        else:
            from geomet import wkt

            geojson_item = self.__geo_interface__
            return wkt.dumps(geojson_item)

    # ----------------------------------------------------------------------
    @property
    def WKB(self):
        """
        Gets the ``well-known binary`` (`WKB`) representation for OGC geometry.
        It provides a portable representation of a geometry value as a
        contiguous stream of bytes.

        :return:
            bytes
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, "polygon", None)
                return p.WKB
            except:
                return None
        if HASARCPY:
            try:
                return getattr(self.as_arcpy, "WKB", None)
            except:
                return None
        elif HASSHAPELY:
            try:
                return self.as_shapely.wkb
            except:
                return None
        else:
            # geomet conversion
            from geomet import wkb

            geojson_item = self.__geo_interface__
            return wkb.dumps(geojson_item, big_endian=False)

    # ----------------------------------------------------------------------
    @property
    def area(self):
        """
        The ``area`` method retrieves the area of a :class:`~arcgis.geometry.Polygon` feature. The units of the returned
        area are based off the :class:`~arcgis.geometry.SpatialReference` field.

        .. note::
            None for all other feature types.

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.area
            -1.869999999973911e-06


        :return:
            A float
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, "polygon", None)
                return p.area
            except:
                return None
        if HASARCPY:
            return getattr(self.as_arcpy, "area", None)
        elif HASSHAPELY:
            return self.as_shapely.area
        elif isinstance(self, Polygon):
            return self._shoelace_area(parts=self["rings"])
        return None

    # ----------------------------------------------------------------------
    def _shoelace_area(self, parts: Union[list[float], list[int]]):
        """calculates the shoelace area"""
        area = 0.0
        area_parts = []
        for part in parts:
            n = len(part)
            for i in range(n):
                j = (i + 1) % n

                area += part[i][0] * part[j][1]
                area -= part[j][0] * part[i][1]

            area_parts.append(area / 2.0)
            area = 0.0
        return abs(sum(area_parts))

    # ----------------------------------------------------------------------
    @property
    def centroid(self):
        """
        The ``centroid`` method retrieves the center of the :class:`~arcgis.geometry.Geometry` object

        .. note::
            The ``centroid`` method requires ArcPy or Shapely

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.centroid
            (-97.06258999999994, 32.754333333000034)


        :return:
            A tuple(x,y) indicating the center
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, "polygon", None)
                return p.centroid
            except:
                return None
        if HASARCPY:
            if isinstance(self, Point):
                return tuple(self)
            else:
                g = getattr(self.as_arcpy, "centroid", None)
                if g is None:
                    return g
                return tuple(Geometry(arcpy.PointGeometry(g, self.spatial_reference)))
        elif HASSHAPELY:
            c = tuple(list(self.as_shapely.centroid.coords)[0])
            return c
        return

    # ----------------------------------------------------------------------
    @property
    def extent(self):
        """
        Get the extent of the :class:`~arcgis.geometry.Geometry` object as a tuple
        containing xmin, ymin, xmax, ymax

        .. note::
            The ``extent`` method requires ArcPy or Shapely

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.extent
            (-97.06326, 32.749, -97.06124, 32.837)

        :return:
            A tuple
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        ptX = []
        ptY = []
        if isinstance(self, Envelope):
            try:
                return tuple(self.coordinates.tolist())
            except:
                return None
        if HASARCPY:
            ext = getattr(self.as_arcpy, "extent", None)
            return ext.XMin, ext.YMin, ext.XMax, ext.YMax
        elif HASSHAPELY:
            return self.as_shapely.bounds
        elif isinstance(self, Polygon):
            for pts in self["rings"]:
                for part in pts:
                    ptX.append(part[0])
                    ptY.append(part[1])
            return min(ptX), min(ptY), max(ptX), max(ptY)

        elif isinstance(self, Polyline):
            for pts in self["paths"]:
                for part in pts:
                    ptX.append(part[0])
                    ptY.append(part[1])
            return min(ptX), min(ptY), max(ptX), max(ptY)
        elif isinstance(self, MultiPoint):
            ptX = [pt["x"] for pt in self["points"]]
            ptY = [pt["y"] for pt in self["points"]]
            return min(ptX), min(ptY), max(ptX), max(ptY)
        elif isinstance(self, Point):
            return self["x"], self["y"], self["x"], self["y"]
        return

    # ----------------------------------------------------------------------
    @property
    def first_point(self):
        """
        The ``first`` method retrieves first coordinate point of the geometry.

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.first_point
            {'x': -97.06138, 'y': 32.837, 'spatialReference': {'wkid': 4326, 'latestWkid': 4326}}


        :return: A :class:`~arcgis.geometry.Geometry` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            try:
                return Geometry(
                    {
                        "x": self["XMin"],
                        "y": self["YMin"],
                        "spatialReference": self["spatialReference"],
                    }
                )
            except:
                return None
        elif HASARCPY:
            return Geometry(
                _ujson.loads(
                    arcpy.PointGeometry(
                        getattr(self.as_arcpy, "firstPoint", None),
                        self.spatial_reference,
                    ).JSON
                )
            )
        elif isinstance(self, Point):
            return self
        elif isinstance(self, MultiPoint):
            if len(self["points"]) == 0:
                return
            geom = self["points"][0]
            return Geometry(
                {
                    "x": geom[0],
                    "y": geom[1],
                    "spatialReference": {"wkid": 4326},
                }
            )
        elif isinstance(self, Polygon):
            if len(self["rings"]) == 0:
                return
            geom = self["rings"][0][0]
            return Geometry(
                {
                    "x": geom[0],
                    "y": geom[1],
                    "spatialReference": {"wkid": 4326},
                }
            )
        elif isinstance(self, Polyline):
            if len(self["paths"]) == 0:
                return
            geom = self["paths"][0][0]
            return Geometry(
                {
                    "x": geom[0],
                    "y": geom[1],
                    "spatialReference": {"wkid": 4326},
                }
            )
        return

    # ----------------------------------------------------------------------
    @property
    def has_z(self):
        """
        The ``has_z`` method determines if the geometry has a `Z` value.

        :return:
            A boolean indicating yes (True), or no (False)

        """
        return self.get("hasZ", False) | self.get("z", False)

    # ----------------------------------------------------------------------
    @property
    def has_m(self):
        """
        The ``has_m`` method determines if the geometry has a `M` value.

        :return:
            A boolean indicating yes (True), or no (False)

        """
        return self.get("hasM", False) | self.get("m", False)

    # ----------------------------------------------------------------------
    @property
    def hull_rectangle(self):
        """
        The ``hull_rectangle`` method retrieves the space-delimited string of the coordinate pairs of the convex hull
        rectangle.

        .. note::
            The ``hull-rectangle`` method requires ArcPy or Shapely

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.hull_rectangle
            '-97.06153 32.749 -97.0632940971127 32.7490060186843 -97.0629938635673 32.8370055061228 -97.0612297664546 32.8369994874385'

        :return:
            A space-delimited string
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            return getattr(self.polygon.as_arcpy, "hullRectangle", None)
        if HASARCPY:
            return getattr(self.as_arcpy, "hullRectangle", None)
        elif HASSHAPELY:
            return self.as_shapely.convex_hull
        return

    # ----------------------------------------------------------------------
    @property
    def is_multipart(self):
        """
        The ``is_multipart`` method determines if the number of parts for this geometry is more than one.

        .. note::
            The ``is_multipart`` method requires ArcPy or Shapely

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>              [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.is_multipart
            True


        :return:
            A boolean indicating yes (True), or no (False)
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            return False
        elif HASARCPY:
            return getattr(self.as_arcpy, "isMultipart", None)
        elif HASSHAPELY:
            if self.type.lower().find("multi") > -1:
                return True
            else:
                return False
        return

    # ----------------------------------------------------------------------
    @property
    def label_point(self):
        """
        Gets the :class:`~arcgis.geometry.Point` at which the label is located.
        The ``label_point`` is always located within or on a feature.

        .. note::
            The ``label_point`` method requires ArcPy or Shapely

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>> })
            >>> geom.label_point
            {'x': -97.06258999999994, 'y': 32.754333333000034, 'spatialReference': {'wkid': 4326, 'latestWkid': 4326}}

        :return:
            A :class:`~arcgis.geometry.Point` object

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            return getattr(self.polygon.as_arcpy, "labelPoint", None)
        elif HASARCPY:
            return Geometry(
                arcpy.PointGeometry(
                    getattr(self.as_arcpy, "labelPoint", None),
                    self.spatial_reference,
                )
            )

        return self.centroid

    # ----------------------------------------------------------------------
    @property
    def last_point(self):
        """
        The ``last_point`` method retrieves the last coordinate :class:`~arcgis.geometry.Point` of the feature.

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.last_point
            {'x': -97.06326, 'y': 32.759, 'spatialReference': {'wkid': 4326, 'latestWkid': 4326}}


        :return: A :class:`~arcgis.geometry.Point` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            return Geometry(
                {
                    "x": self["XMax"],
                    "y": self["YMax"],
                    "spatialReference": self["spatialReference"],
                }
            )
        elif HASARCPY:
            return Geometry(
                arcpy.PointGeometry(
                    getattr(self.as_arcpy, "lastPoint", None),
                    self.spatial_reference,
                )
            )
        elif isinstance(self, Point):
            return self
        elif isinstance(self, Polygon):
            if self["rings"] == 0:
                return
            geom = self["rings"][-1][-1]
            return Geometry(
                {
                    "x": geom[0],
                    "y": geom[1],
                    "spatialReference": {"wkid": 4326},
                }
            )
        elif isinstance(self, Polyline):
            if self["paths"] == 0:
                return
            geom = self["paths"][-1][-1]
            return Geometry(
                {
                    "x": geom[0],
                    "y": geom[1],
                    "spatialReference": {"wkid": 4326},
                }
            )
        return

    # ----------------------------------------------------------------------
    @property
    def length(self):
        """
        Gets length of the linear feature.
        The length units is the same as the :class:`~arcgis.geometry.SpatialReference` field.

        .. note::
            The ``length`` method returns zero for :class:`~arcgis.geometry.Point` and
            :class:`~arcgis.geometry.MultiPoint` feature types.

        .. note::
            The ``length`` method requires ArcPy or Shapely

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.length
            0.03033576008004027

        :return: A float
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            return getattr(self.polygon.as_arcpy, "length", None)
        elif HASARCPY:
            return getattr(self.as_arcpy, "length", None)
        elif HASSHAPELY:
            return self.as_shapely.length

        return None

    # ----------------------------------------------------------------------
    @property
    def length3D(self):
        """
        The ``length3D`` method retrieves the 3D length of the linear feature. Zero for point and multipoint
        The length units is the same as the :class:`~arcgis.geometry.SpatialReference` field.

        .. note::
            The ``length3D`` method returns zero for :class:`~arcgis.geometry.Point` and
            :class:`~arcgis.geometry.MultiPoint` feature types.

        .. note::
            The ``length3D`` method requires ArcPy or Shapely

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.length3D
            0.03033576008004027

        :return: A float
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            return getattr(self.polygon.as_arcpy, "length3D", None)
        elif HASARCPY:
            return getattr(self.as_arcpy, "length3D", None)
        elif HASSHAPELY:
            return self.as_shapely.length

        return self.length

    # ----------------------------------------------------------------------
    @property
    def part_count(self):
        """
        The ``part_count`` method retrieves the number of :class:`~arcgis.geometry.Geometry` parts for the feature.


        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.part_count
            1

        :return: An Integer representing the amount of :class:`~arcgis.geometry.Geometry` parts
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            return 1
        elif HASARCPY:
            return getattr(self.as_arcpy, "partCount", None)
        elif isinstance(self, Polygon):
            return len(self["rings"])
        elif isinstance(self, Polyline):
            return len(self["paths"])
        elif isinstance(self, MultiPoint):
            return len(self["points"])
        elif isinstance(self, Point):
            return 1
        return

    # ----------------------------------------------------------------------
    @property
    def point_count(self):
        """
        The ``point_count`` method retrieves total number of :class:`~arcgis.geometry.Point` objects for the feature.


        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.point_count
            9

        :return: An Integer representing the amount of :class:`~arcgis.geometry.Point` objects
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if isinstance(self, Envelope):
            return 4
        elif HASARCPY:
            return getattr(self.as_arcpy, "pointCount", None)
        elif isinstance(self, Polygon):
            return sum([len(part) for part in self["rings"]])
        elif isinstance(self, Polyline):
            return sum([len(part) for part in self["paths"]])
        elif isinstance(self, MultiPoint):
            return sum([len(part) for part in self["points"]])
        elif isinstance(self, Point):
            return 1
        return

    # ----------------------------------------------------------------------
    @property
    def spatial_reference(self):
        """
        Gets the :class:`~arcgis.geometry.SpatialReference` of the geometry.

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.spatial_reference
            <SpatialReference Class>

        :return: A :class:`~arcgis.geometry.SpatialReference` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            v = getattr(self.polygon.as_arcpy, "spatialReference", None)
            if v:
                return SpatialReference(v)
        elif HASARCPY:
            return SpatialReference(self["spatialReference"])
        if "spatialReference" in self:
            return SpatialReference(self["spatialReference"])
        return None

    # ----------------------------------------------------------------------
    @property
    def true_centroid(self):
        """
        Gets the :class:`~arcgis.geometry.Point` representing the center of gravity
        for a feature.

        .. note::
            The ``true_centroid`` method requires ArcPy or Shapely

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.true_centroid
            {'x': -97.06272135472369, 'y': 32.746201426025, 'spatialReference': {'wkid': 4326, 'latestWkid': 4326}}

        :return: A :class:`~arcgis.geometry.Point` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, Envelope):
            return Geometry(
                arcpy.PointGeometry(
                    getattr(self.polygon.as_arcpy, "trueCentroid", None),
                    self.spatial_reference.as_arcpy,
                )
            )
        elif HASARCPY:
            return Geometry(
                arcpy.PointGeometry(
                    getattr(self.as_arcpy, "trueCentroid", None),
                    self.spatial_reference.as_arcpy,
                )
            )
        elif HASSHAPELY:
            centroid_tuple = self.centroid
            return Point(
                {
                    "x": centroid_tuple[0],
                    "y": centroid_tuple[1],
                    "spatialReference": self.spatial_reference,
                }
            )
        elif isinstance(self, Point):
            return self
        return

    # ----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """
        Gets the geometry type:
            1. A :class:`~arcgis.geometry.Polygon`
            2. A :class:`~arcgis.geometry.Polyline`
            3. A :class:`~arcgis.geometry.Point`
            4. A :class:`~arcgis.geometry.MultiPoint`

        .. code-block:: python

            >>> geom = Geometry({
            >>>     "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.geometry_type
            'polygon'


        :return: A string indicating the geometry type
        """
        if isinstance(self, Envelope):
            return "envelope"
        elif isinstance(self, Point):
            return "point"
        elif isinstance(self, MultiPoint):
            return "multipoint"
        elif isinstance(self, Polyline):
            return "polyline"
        elif isinstance(self, Polygon):
            return "polygon"
        return

    # Functions#############################################################
    # ----------------------------------------------------------------------
    def angle_distance_to(self, second_geometry: Geometry, method: str = "GEODESIC"):
        """
        The ``angle_distance_to`` method retrieves a tuple of angle and distance to another
        :class:`~arcgis.geometry.Point` using a measurement type.

        .. note::
            The ``angle_distance_to`` method requires `ArcPy`. If `ArcPy` is not installed, none is returned.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required Geometry. An :class:`~arcgis.geometry.Geometry` object.
        ---------------     --------------------------------------------------------------------
        method              Optional String. PLANAR measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC, and
                            LOXODROME measurement types may be chosen as an alternative, if desired.
        ===============     ====================================================================

        :return: A tuple of angle and distance to another :class:`~arcgis.geometry.Point` using a measurement type.

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.angle_distance_to(second_geometry = geom2,
            >>>                        method="PLANAR")
                {54.5530, 1000.1111}

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()

        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.angleAndDistanceTo(
                other_geometry=second_geometry, method=method
            )
        return None

    # ----------------------------------------------------------------------
    def boundary(self):
        """
        The ``boundary`` method constructs the boundary of the :class:`~arcgis.geometry.Geometry` object.

        :return:
            A :class:`~arcgis.geometry.Geometry` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()

        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.boundary())
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_shapely.boundary.buffer(1).__geo_interface__)
        return None

    # ----------------------------------------------------------------------
    def buffer(self, distance: float):
        """
        The buffer method constructs a :class:`~arcgis.geometry.Polygon` at a specified distance from the
        :class:`~arcgis.geometry.Geometry` object.

        .. note::
            The ``buffer`` method requires ArcPy

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        distance            Required float. The buffer distance. The buffer distance is in the
                            same units as the geometry that is being buffered.
                            A negative distance can only be specified against a polygon geometry.
        ===============     ====================================================================

        :return: A :class:`~arcgis.geometry.Polygon` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.buffer(distance))
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(
                self.as_shapely.buffer(distance).__geo_interface__,
                sr=self.spatial_reference,
            )
        return None

    # ----------------------------------------------------------------------
    def clip(self, envelope: tuple(float)):
        """
        The ``clip`` method constructs the intersection of the :class:`~arcgis.geometry.Geometry` object and the
        specified extent.

        .. note::
            The ``clip`` method requires `ArcPy`. If `ArcPy` is not installed, none is returned.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        envelope            Required tuple. The tuple must have (XMin, YMin, XMax, YMax) each value
                            represents the lower left bound and upper right bound of the extent.
        ===============     ====================================================================

        :return:
            The :class:`~arcgis.geometry.Geometry` object clipped to the extent
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(envelope, (list, tuple)) and len(envelope) == 4:
            envelope = arcpy.Extent(
                XMin=envelope[0],
                YMin=envelope[1],
                XMax=envelope[2],
                YMax=envelope[3],
            )
            return Geometry(self.as_arcpy.clip(envelope))
        elif (
            HASARCPY
            and isinstance(self, (Point, Polygon, Polyline, MultiPoint))
            and isinstance(envelope, arcpy.Extent)
        ):
            return Geometry(self.as_arcpy.clip(envelope))
        elif (
            HASARCPY
            and isinstance(self, (Point, Polygon, Polyline, MultiPoint))
            and isinstance(envelope, Envelope)
        ):
            return Geometry(self.as_arcpy.clip(envelope.as_arcpy))
        return None

    # ----------------------------------------------------------------------
    def contains(self, second_geometry: Geometry, relation: Optional[str] = None):
        """
        Indicates if the base :class:`~arcgis.geometry.Geometry` object contains the comparison
        :class:`~arcgis.geometry.Geometry` object.

        .. note::
            The ``contain`` method requires ArcPy/Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional string. The spatial relationship type.

                            + BOUNDARY - Relationship has no restrictions for interiors or boundaries.
                            + CLEMENTINI - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            + PROPER - Boundaries of geometries must not intersect.
        ===============     ====================================================================

        :return:
            A boolean indicating containment (True), or no containment (False)

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.contains(second_geometry = geom2,
                              relation="CLEMENTINI")
                True
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()

        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.contains(
                second_geometry=second_geometry, relation=relation
            )
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.contains(second_geometry)
        return None

    # ----------------------------------------------------------------------
    def convex_hull(self):
        """
        Constructs the :class:`~arcgis.geometry.Geometry` object that is the minimal bounding
        :class:`~arcgis.geometry.Polygon` such that all outer angles are convex.

        :return:
            A :class:`~arcgis.geometry.Geometry` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.convexHull())
        elif self.type.lower() == "polygon":
            from ._convexhull import convex_hull

            combine_pts = [pt for part in self["rings"] for pt in part]
            try:
                return Geometry(
                    {
                        "rings": [convex_hull(combine_pts)],
                        "spatialReference": self["spatialReference"],
                    }
                )
            except:
                from ._convexhull import convex_hull_GS

                return Geometry(
                    {
                        "rings": [convex_hull_GS(combine_pts)],
                        "spatialReference": self["spatialReference"],
                    }
                )
        elif self.type.lower() == "polyline":
            from ._convexhull import convex_hull

            combine_pts = [pt for part in self["paths"] for pt in part]
            try:
                return Geometry(
                    {
                        "rings": [convex_hull(combine_pts)],
                        "spatialReference": self["spatialReference"],
                    }
                )
            except:
                from ._convexhull import convex_hull_GS

                return Geometry(
                    {
                        "rings": [convex_hull_GS(combine_pts)],
                        "spatialReference": self["spatialReference"],
                    }
                )
        elif self.type.lower() == "multipoint":
            from ._convexhull import convex_hull

            combine_pts = self["points"]
            try:
                return Geometry(
                    {
                        "rings": [convex_hull(combine_pts)],
                        "spatialReference": self["spatialReference"],
                    }
                )
            except:
                from ._convexhull import convex_hull_GS

                return Geometry(
                    {
                        "rings": [convex_hull_GS(combine_pts)],
                        "spatialReference": self["spatialReference"],
                    }
                )
        return None

    # ----------------------------------------------------------------------
    def crosses(self, second_geometry: Geometry):
        """
        Indicates if the two :class:`~arcgis.geometry.Geometry` objects intersect in a
        geometry of a lesser shape type.

        .. note::
            The ``crosses`` method requires ArcPy/Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ===============     ====================================================================

        :return:
            A boolean indicating yes (True), or no (False)

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.crosses(second_geometry=second_geometry)
        elif HASSHAPELY:
            return self.as_shapely.crosses(other=second_geometry.as_shapely)
        return None

    # ----------------------------------------------------------------------
    def cut(self, cutter: Polyline):
        """
        Splits this :class:`~arcgis.geometry.Geometry` object into a part left of the cutting
        :class:`~arcgis.geometry.Polyline` and a part right of it.

        .. note::
            The ``cut`` method requires ArcPy

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        cutter              Required :class:`~arcgis.geometry.Polyline`. The cutting polyline geometry
        ===============     ====================================================================

        :return: a list of two :class:`~arcgis.geometry.Geometry` objects

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if isinstance(cutter, Polyline) and HASARCPY:
            if isinstance(cutter, Geometry):
                cutter = cutter.as_arcpy
            return Geometry(self.as_arcpy.cut(other=cutter))
        return None

    # ----------------------------------------------------------------------
    def densify(self, method: str, distance: float, deviation: float):
        """
        Creates a new :class:`~arcgis.geometry.Geometry` object with added vertices

        .. note::
            The ``densify`` method requires ArcPy

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. The type of densification: ``DISTANCE``, ``ANGLE``, or ``GEODESIC``
        ---------------     --------------------------------------------------------------------
        distance            Required float. The maximum distance between vertices. The actual
                            distance between vertices will usually be less than the maximum
                            distance as new vertices will be evenly distributed along the
                            original segment. If using a type of DISTANCE or ANGLE, the
                            distance is measured in the units of the geometry's spatial
                            reference. If using a type of GEODESIC, the distance is measured
                            in meters.
        ---------------     --------------------------------------------------------------------
        deviation           Required float. ``Densify`` uses straight lines to approximate curves.
                            You use deviation to control the accuracy of this approximation.
                            The deviation is the maximum distance between the new segment and
                            the original curve. The smaller its value, the more segments will
                            be required to approximate the curve.
        ===============     ====================================================================

        :return:
            A new :class:`~arcgis.geometry.Geometry` object

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom2 = geom.densify(method = "GEODESIC",
                                     distance = 1244.0,
                                     deviation = 100.0)

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(
                self.as_arcpy.densify(
                    method=method, distance=distance, deviation=deviation
                )
            )
        return None

    # ----------------------------------------------------------------------
    def difference(self, second_geometry: Geometry):
        """
        Constructs the :class:`~arcgis.geometry.Geometry` object that is composed only of the
        region unique to the base geometry but not part of the other geometry.

        .. note::
            The ``difference`` method requires ArcPy/Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            g = self.as_arcpy.difference(other=second_geometry)
            return Geometry(g)
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return Geometry(
                self.as_shapely.difference(second_geometry).__geo_interface__
            )
        return None

    # ----------------------------------------------------------------------
    def disjoint(self, second_geometry: Geometry):
        """
        Indicates if the base and comparison :class:`~arcgis.geometry.Geometry` objects share no
        :class:`~arcgis.geometry.Point` objects in common.

        .. note::
            The ``disjoint`` method requires ArcPy/Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ===============     ====================================================================

        :return:
            A boolean indicating no :class:`~arcgis.geometry.Point` objects in common (True), or some in common
            (False)

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.disjoint(second_geometry=second_geometry)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.disjoint(second_geometry)
        return None

    # ----------------------------------------------------------------------
    def distance_to(self, second_geometry: Geometry):
        """
        Retrieves the minimum distance between two :class:`~arcgis.geometry.Geometry` objects. If the
        geometries intersect, the minimum distance is 0.

        .. note::
            Both geometries must have the same projection.

        .. note::
            The ``distance_to`` method requires ArcPy/Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ===============     ====================================================================

        :return: A float

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.distanceTo(other=second_geometry)
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.distance(other=second_geometry)
        return None

    # ----------------------------------------------------------------------
    def equals(self, second_geometry: Geometry):
        """
        Indicates if the base and comparison :class:`~arcgis.geometry.Geometry` objects are of the
        same shape type and define the same set of points in the plane. This is
        a 2D comparison only; M and Z values are ignored.

        .. note::
            The ``equals`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return: Boolean indicating True if geometries are equal else False


        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.equals(second_geometry=second_geometry)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.equals(other=second_geometry)
        return None

    # ----------------------------------------------------------------------
    def generalize(self, max_offset: float):
        """
        Creates a new simplified :class:`~arcgis.geometry.Geometry` object using a specified
        maximum offset tolerance.

        .. note::
            The ``generalize`` method requires ArcPy or Shapely**

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        max_offset          Required float. The maximum offset tolerance.
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.generalize(distance=max_offset))
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_shapely.simplify(max_offset).__geo_interface__)
        return None

    # ----------------------------------------------------------------------
    def get_area(self, method: str, units: Optional[str] = None):
        """
        Retrieves the area of the :class:`~arcgis.geometry.Geometry` using a measurement type.

        .. note::
            The ``get_area`` method requires ArcPy or Shapely**

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. `PLANAR` measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). `GEODESIC`,
                            `GREAT_ELLIPTIC`, `LOXODROME`, and `PRESERVE_SHAPE` measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Optional String. Areal unit of measure keywords: `ACRES | ARES | HECTARES
                            | SQUARECENTIMETERS | SQUAREDECIMETERS | SQUAREINCHES | SQUAREFEET
                            | SQUAREKILOMETERS | SQUAREMETERS | SQUAREMILES |
                            SQUAREMILLIMETERS | SQUAREYARDS`
        ===============     ====================================================================

        :return: A float representing the area of the :class:`~arcgis.geometry.Geometry` object

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return self.as_arcpy.getArea(method=method, units=units)
        elif HASARCPY and isinstance(self, Envelope):
            return self.polygon.as_arcpy.getArea(method=method, units=units)
        return None

    # ----------------------------------------------------------------------
    def get_length(self, method: str, units: str):
        """
        Retrieves the length of the :class:`~arcgis.geometry.Geometry` using a measurement type.

        .. note::
            The ``get_length`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. `PLANAR` measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). `GEODESIC`,
                            `GREAT_ELLIPTIC`, `LOXODROME`, and `PRESERVE_SHAPE` measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Required String. Linear unit of measure keywords: `CENTIMETERS |
                            DECIMETERS | FEET | INCHES | KILOMETERS | METERS | MILES |
                            MILLIMETERS | NAUTICALMILES | YARDS`
        ===============     ====================================================================

        :return:
            A float representing the length of the :class:`~arcgis.geometry.Geometry` object

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return self.as_arcpy.getLength(method=method, units=units)
        elif HASARCPY and isinstance(self, Envelope):
            return self.polygon.as_arcpy.getLength(method=method, units=units)
        return None

    # ----------------------------------------------------------------------
    def get_part(self, index: Optional[int] = None):
        """
        Retrieves an array of :class:`~arcgis.geometry.Point` objects for a particular part of
        a :class:`~arcgis.geometry.Geometry` object or an array containing a number of arrays, one for each part.

        .. note::
            The ``get_part`` method requires ArcPy

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        index               Required Integer. The index position of the :class:`~arcgis.geometry.Geometry` object.
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return self.as_arcpy.getPart(index)
        return None

    # ----------------------------------------------------------------------
    def intersect(self, second_geometry: Geometry, dimension: int = 1):
        """
        Constructs a :class:`~arcgis.geometry.Geometry` object that is the geometric
        intersection of the two input geometries. Different dimension values can be used to create
        different shape types. The intersection of two geometries of the
        same shape type is a geometry containing only the regions of overlap
        between the original geometries.

        .. note::
            The ``intersect`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ---------------     --------------------------------------------------------------------
        dimension           Required Integer. The topological dimension (shape type) of the
                            resulting geometry.

                            + 1  -A zero-dimensional geometry (:class:`~arcgis.geometry.Point` or :class:`~arcgis.geometry.MultiPoint`).
                            + 2  -A one-dimensional geometry (:class:`~arcgis.geometry.Polyline`).
                            + 4  -A two-dimensional geometry (:class:`~arcgis.geometry.Polygon`).

        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object indicating an intersection, or None for no intersection

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> type(geom.intersect(second_geometry = geom2, dimension = 4))
                arcgis.geometry._types.Polygon

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
                dimension = 4
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            r = Geometry(
                self.as_arcpy.intersect(other=second_geometry, dimension=dimension)
            )
            if r.is_empty == True:
                return None
            else:
                return r

        elif HASARCPY and isinstance(self, Envelope):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
                dimension = 4
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            r = Geometry(
                self.polygon.as_arcpy.intersect(
                    other=second_geometry, dimension=dimension
                )
            )
            if r.is_empty == True:
                return None
            else:
                return r
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return Geometry(
                self.as_shapely.intersection(other=second_geometry).__geo_interface__
            )
        return None

    # ----------------------------------------------------------------------
    def measure_on_line(self, second_geometry: Geometry, as_percentage: bool = False):
        """
        Retrieves a measure from the start :class:`~arcgis.geometry.Point` of this line to
        the ``in_point``.

        .. note::
            The ``measure_on_line`` method requires ArcPy

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional Boolean. If False, the measure will be returned as a
                            distance; if True, the measure will be returned as a percentage.
        ===============     ====================================================================

        :return:
            A float

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.measure_on_line(second_geometry = geom2,
                                     as_percentage = True)
                0.33

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.measureOnLine(
                in_point=second_geometry, use_percentage=as_percentage
            )
        return None

    # ----------------------------------------------------------------------
    def overlaps(self, second_geometry: Geometry):
        """
        Indicates if the intersection of the two :class:`~arcgis.geometry.Geometry` objects has
        the same shape type as one of the input geometries and is **not** equivalent to
        either of the input geometries.

        .. note::
            The ``overlaps`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ===============     ====================================================================

        :return:
            A boolean indicating an intersection of same shape type (True), or different type (False)

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.overlaps(second_geometry=second_geometry)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.overlaps(other=second_geometry)
        return None

    # ----------------------------------------------------------------------
    def point_from_angle_and_distance(
        self, angle: float, distance: float, method: str = "GEODESCIC"
    ):
        """
        Retrieves a :class:`~arcgis.geometry.Point` at a given angle and distance,
        in degrees and meters, using the specified measurement type.

        .. note::
            The ``point_from_angle_and_distance`` method requires ArcPy

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        angle               Required Float. The angle in degrees to the returned point.
        ---------------     --------------------------------------------------------------------
        distance            Required Float. The distance in meters to the returned point.
        ---------------     --------------------------------------------------------------------
        method              Optional String. `PLANAR` measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). `GEODESIC`, `GREAT_ELLIPTIC`,
                            `LOXODROME`, and `PRESERVE_SHAPE` measurement types may be chosen as
                            an alternative, if desired.
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Point` object

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> point = geom.point_from_angle_and_distance(angle=60,
                                                           distance = 100000,
                                                           method = "PLANAR")
            >>> point.type
                "POINT"
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(
                self.as_arcpy.pointFromAngleAndDistance(
                    angle=angle, distance=distance, method=method
                )
            )
        return None

    # ----------------------------------------------------------------------
    def position_along_line(self, value: float, use_percentage: bool = False):
        """
        Retrieves a :class:`~arcgis.geometry.Point` on a line at a specified distance
        from the beginning of the line.

        .. note::
            The ``position_along_line`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required Float. The distance along the line.
        ---------------     --------------------------------------------------------------------
        use_percentage      Optional Boolean. The distance may be specified as a fixed unit
                            of measure or a ratio of the length of the line. If True, value
                            is used as a percentage; if False, value is used as a distance.

                            .. note::
                                For percentages, the value should be expressed as a double from
                                0.0 (0%) to 1.0 (100%).
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(
                self.as_arcpy.positionAlongLine(
                    value=value, use_percentage=use_percentage
                )
            )
        elif HASSHAPELY:
            return Geometry(
                self.as_shapely.interpolate(
                    value, normalized=use_percentage
                ).__geo_interface__
            )

        return None

    # ----------------------------------------------------------------------
    def project_as(
        self,
        spatial_reference: Union[dict[str, Any], SpatialReference],
        transformation_name: str = None,
    ):
        """
        Projects a :class:`~arcgis.geometry.Geometry` object and optionally applies a
        ``geotransformation``.

        .. note::
            The ``project_as`` method requires ArcPy or pyproj>=1.9 and PROJ.4

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required SpatialReference. The new spatial reference. This can be a
                                 :class:`~arcgis.geometry.SpatialReference` object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Required String. The ``geotransformation`` name.
        ====================     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom2 = geom.project_as(spatial_reference="GCS",
                                        transformation_name = "transformation")
            >>> geom2.type
                arcgis.geometry.Geometry
        """

        HASARCPY, HASSHAPELY = _check_geometry_engine()

        if HASARCPY:
            if isinstance(spatial_reference, SpatialReference):
                spatial_reference = spatial_reference.as_arcpy
            elif isinstance(spatial_reference, dict):
                spatial_reference = SpatialReference(spatial_reference).as_arcpy
            elif isinstance(spatial_reference, arcpy.SpatialReference):
                spatial_reference = spatial_reference
            elif isinstance(spatial_reference, int):
                spatial_reference = arcpy.SpatialReference(spatial_reference)
            elif isinstance(spatial_reference, str):
                spatial_reference = arcpy.SpatialReference(text=spatial_reference)
            else:
                raise ValueError("Invalid spatial reference object.")
            return Geometry(
                self.as_arcpy.projectAs(
                    spatial_reference=spatial_reference,
                    transformation_name=transformation_name,
                )
            )

        try:
            import pyproj
            from shapely.ops import transform

            HASPROJ = True
        except:
            HASPROJ = False

        # Project using Proj4 (pyproj)
        if HASPROJ:
            esri_projections = {102100: 3857, 102113: 3857}

            # Get the input spatial reference
            in_srid = self.spatial_reference.get("wkid", None)
            in_srid = self.spatial_reference.get("latestWkid", in_srid)
            # Convert web mercator from esri SRID
            in_srid = esri_projections.get(int(in_srid), in_srid)
            in_srid = "epsg:{}".format(in_srid)

            if isinstance(spatial_reference, dict) or isinstance(
                spatial_reference, SpatialReference
            ):
                out_srid = spatial_reference.get("wkid", None)
                out_srid = spatial_reference.get("latestWkid", out_srid)
            elif isinstance(spatial_reference, int):
                out_srid = spatial_reference
            elif isinstance(spatial_reference, str):
                out_srid = spatial_reference
            else:
                raise ValueError("Invalid spatial reference object.")

            out_srid = esri_projections.get(int(out_srid), out_srid)
            out_srid = "epsg:{}".format(out_srid)

            try:
                if [int(i) for i in pyproj.__version__.split(".") if i.isdigit()][
                    0
                ] == 2:
                    from pyproj import Transformer

                    project = Transformer.from_crs(
                        in_srid, out_srid, always_xy=True
                    ).transform
                else:
                    project = partial(
                        pyproj.transform,
                        pyproj.Proj(init=in_srid),
                        pyproj.Proj(init=out_srid),
                    )
            except RuntimeError as e:
                raise ValueError(
                    "pyproj projection from {0} to {1} not currently supported".format(
                        in_srid, out_srid
                    )
                )
            # in geomet there is transform in tools and then call from_geomet to make it back to what it was
            g = transform(project, self.as_shapely)
            return Geometry.from_shapely(g, spatial_reference=spatial_reference)

        return None

    # ----------------------------------------------------------------------
    def query_point_and_distance(
        self, second_geometry: Geometry, use_percentage: bool = False
    ):
        """
        Finds the :class:`~arcgis.geometry.Point` on the
        :class:`~arcgis.geometry.Polyline` nearest to the `in_point` and the
        distance between those points. ``query_point_and_distance`` retrieves information about the
        side of the line the `in_point` is on as well as the distance along
        the line where the nearest point occurs.

        .. note::
            The ``query_point_and_distance`` method requires ArcPy

        .. note::
            The ``query_point_and_distance`` method only is valid for Polyline geometries.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Point` object. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional boolean - if False, the measure will be returned as
                            distance, True, measure will be a percentage
        ===============     ====================================================================

        :return:
            A tuple of the point and the distance

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if (
            HASARCPY
            and isinstance(self, Polyline)
            and isinstance(second_geometry, Point)
        ):
            if isinstance(second_geometry, Point):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.queryPointAndDistance(
                in_point=second_geometry, use_percentage=use_percentage
            )
        return None

    # ----------------------------------------------------------------------
    def segment_along_line(
        self,
        start_measure: float,
        end_measure: float,
        use_percentage: bool = False,
    ):
        """
        Retrieves a :class:`~arcgis.geometry.Polyline` between ``start`` and ``end``
        measures. ``segment_along_line`` is similar to the :attr:`~arcgis.geometry.Polyline.positionAlongLine` method
        but will return a polyline segment between two points on the polyline instead of a single
        :class:`~arcgis.geometry.Point`.

        .. note::
            The ``segment_along_line`` method requires ArcPy

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        start_measure       Required Float. The starting distance from the beginning of the line.
        ---------------     --------------------------------------------------------------------
        end_measure         Required Float. The ending distance from the beginning of the line.
        ---------------     --------------------------------------------------------------------
        use_percentage      Optional Boolean. The start and end measures may be specified as
                            fixed units or as a ratio.
                            If True, start_measure and end_measure are used as a percentage; if
                            False, start_measure and end_measure are used as a distance.

                            .. note::
                                For
                                percentages, the measures should be expressed as a double from 0.0
                                (0 percent) to 1.0 (100 percent).
        ===============     ====================================================================

        :return: A float

        .. code-block:: python

            >>> geom = Geometry({
            >>>   "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
            >>>               [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
            >>>               [-97.06326,32.759]]],
            >>>   "spatialReference" : {"wkid" : 4326}
            >>>                 })
            >>> geom.segment_along_line(start_measure =0,
                                        end_measure= 1000,
                                        use_percentage = True)
                0.56
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(
                self.as_arcpy.segmentAlongLine(
                    start_measure=start_measure,
                    end_measure=end_measure,
                    use_percentage=use_percentage,
                )
            )
        return None

    # ----------------------------------------------------------------------
    def snap_to_line(self, second_geometry: Geometry):
        """
        The ``snap_to_line`` method retrieves a new :class:`~arcgis.geometry.Point` based on `in_point` snapped to this
        :class:`~arcgis.geometry.Geometry` object.

        .. note::
            The ``snap_to_line`` method requires ArcPy

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` - A second geometry
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Point` object

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return Geometry(self.as_arcpy.snapToLine(in_point=second_geometry))
        return None

    # ----------------------------------------------------------------------
    def symmetric_difference(self, second_geometry: Geometry):
        """
        The ``symmetric_difference`` method constructs a new :class:`~arcgis.geometry.Geometry` object that is the union
        of two geometries minus the intersection of those geometries.

        .. note::
            The two input geometries must be the same shape type.

        .. note::
            The ``symmetric_difference`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()

        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return Geometry(self.as_arcpy.symmetricDifference(other=second_geometry))
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return Geometry(
                self.as_shapely.symmetric_difference(
                    other=second_geometry
                ).__geo_interface__
            )
        return None

    # ----------------------------------------------------------------------
    def touches(self, second_geometry: Geometry):
        """
        Indicates if the boundaries of the two :class:`~arcgis.geometry.Geometry` objects
        intersect.

        .. note::
            The ``touches`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ===============     ====================================================================

        :return:
            A boolean indicating whether the :class:`~arcgis.geometry.Geometry` objects touch (True), or if they do not
            touch (False)

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.touches(second_geometry=second_geometry)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.touches(second_geometry)
        return None

    # ----------------------------------------------------------------------
    def union(self, second_geometry: Geometry):
        """
        Constructs the :class:`~arcgis.geometry.Geometry` object that is the set-theoretic union
        of the input geometries.

        .. note::
            The ``union`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.geometry.Geometry` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return Geometry(self.as_arcpy.union(other=second_geometry))
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return Geometry(self.as_shapely.union(second_geometry).__geo_interface__)
        return None

    # ----------------------------------------------------------------------
    def within(self, second_geometry: Geometry, relation: Optional[str] = None):
        """
        Indicates if the base :class:`~arcgis.geometry.Geometry` object is within the comparison
        :class:`~arcgis.geometry.Geometry` object.

        .. note::
            The ``within`` method requires ArcPy or Shapely

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry` object. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional String. The spatial relationship type.

                            - `BOUNDARY`  - Relationship has no restrictions for interiors or boundaries.
                            - `CLEMENTINI`  - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            - `PROPER`  - Boundaries of geometries must not intersect.

        ===============     ====================================================================

        :return:
            A boolean indicating the :class:`~arcgis.geometry.Geometry` object is within (True), or not within (False)

        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.within(
                second_geometry=second_geometry, relation=relation
            )
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.within(second_geometry)
        return None


###########################################################################
class MultiPoint(Geometry):
    """
    A ``multipoint`` contains an array of :class:`~arcgis.geometry.Point`, along with a
    :class:`~arcgis.geometry.SpatialReference` field. A ``multipoint`` can also have
    boolean-valued `hasZ` and `hasM` fields. These fields control the interpretation of elements of the points
    array.

    .. note::
        Omitting an `hasZ` or `hasM` field is equivalent to setting it to
        false.

    Each element of the points array is itself an array of two, three, or
    four numbers. It will have two elements for 2D points, two or three
    elements for 2D points with Ms, three elements for 3D points, and three
    or four elements for 3D points with Ms. In all cases, the x coordinate
    is at index 0 of a point's array, and the y coordinate is at index 1.
    For 2D points with Ms, the m coordinate, if present, is at index 2. For
    3D points, the Z coordinate is required and is at index 2. For 3D
    points with Ms, the Z coordinate is at index 2, and the M coordinate,
    if present, is at index 3.

    .. note::
        An empty multipoint has a points field with no elements. Empty points
        are ignored.
    """

    _typ = "Multipoint"
    _type = "Multipoint"

    def __init__(self, iterable=None, **kwargs):
        if iterable is None:
            iterable = ()
        super(MultiPoint, self).__init__(iterable)
        self.update(kwargs)

    @property
    def __geo_interface__(self):
        return {
            "type": "Multipoint",
            "coordinates": [(pt[0], pt[1]) for pt in self["points"]],
        }

    # ----------------------------------------------------------------------
    @property
    def type(self):
        """Gets the type of the current ``MultiPoint`` object."""
        return self._type

    # ----------------------------------------------------------------------
    def svg(self, scale_factor: float = 1.0, fill_color: Optional[str] = None):
        """
        Returns a group of SVG (Scalable Vector Graphic) circle element for the ``MultiPoint`` geometry.


        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        scale_factor      An optional float. Multiplication factor for the SVG circle diameter.  Default is 1.
        ----------------  -------------------------------------------------------------------------------
        fill_color        An optional string. Hex string for fill color. Default is to use "#66cc99" if geometry is
                          valid, and "#ff3333" if invalid.
        ================  ===============================================================================

        :return:
            A group of SVG circle elements
        """
        if self.is_empty:
            return "<g />"
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return (
            "<g>"
            + "".join(
                (
                    '<circle cx="{0.x}" cy="{0.y}" r="{1}" '
                    'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="0.6" />'
                ).format(
                    Point({"x": p[0], "y": p[1]}),
                    3 * scale_factor,
                    1 * scale_factor,
                    fill_color,
                )
                for p in self["points"]
            )
            + "</g>"
        )

    # ----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))

    # ----------------------------------------------------------------------
    def coordinates(self):
        """
        Retrieves the coordinates of the ``MultiPoint`` as an np.array

        .. code-block:: python

            #Usage Example

            >>> coords = multiPoint.coordinates()
            >>> coords
                [ [x1,y1,m1,z1], [x2,y2,m2,z2],...]

        :return:
            An np.array containing coordinate values for each point
        """
        import numpy as np

        if "points" in self:
            return np.array(self["points"], dtype=object)
        else:
            return np.array([])

    # ----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support"""
        self.__dict__.update(d)
        self = MultiPoint(iterable=d)

    # ----------------------------------------------------------------------
    def __getstate__(self):
        """pickle support"""
        return dict(self)

    # ----------------------------------------------------------------------
    @classmethod
    def _from_geojson(cls, data, sr=None):
        if sr is None:
            sr = {"wkid": 4326}

        coordkey = "coordinates"
        for d in data:
            if d.lower() == "coordinates":
                coordkey = d

        coordinates = data[coordkey]

        return cls({"points": [p for p in coordinates], "spatialReference": sr})


########################################################################
class Point(Geometry):
    """
    The ``Point`` class contains x and y fields along with a :class:`~arcgis.geometry.SpatialReference` field. A
    ``Point`` can also contain m and z fields. A ``Point`` is empty when its x
    field is present and has the value `null` or the string `NaN`. An empty
    ``point`` has **no** location in space.
    """

    _typ = "Point"
    _type = "Point"

    # ----------------------------------------------------------------------
    def __init__(self, iterable=None):
        """Constructor"""
        super(Point, self)
        if iterable is None:
            iterable = {}
        self.update(iterable)

    # ----------------------------------------------------------------------
    @property
    def type(self):
        """Gets the type of the current ``Point`` object."""
        return self._type

    # ----------------------------------------------------------------------
    def svg(self, scale_factor: float = 1, fill_color: Optional[str] = None):
        """
        Returns a SVG (Scalable Vector Graphic) circle element for the ``Point`` geometry.
        SVG defines vector-based graphics in XML format.

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        scale_factor      An optional float. Multiplication factor for the SVG circle diameter.  Default is 1.
        ----------------  -------------------------------------------------------------------------------
        fill_color        An optional string. Hex string for fill color. Default is to use "#66cc99" if geometry is
                          valid, and "#ff3333" if invalid.
        ================  ===============================================================================

        :return:
            An SVG circle element
        """
        if self.is_empty:
            return "<g />"
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return (
            '<circle cx="{0.x}" cy="{0.y}" r="{1}" '
            'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="0.6" />'
        ).format(self, 3 * scale_factor, 1 * scale_factor, fill_color)

    # ----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support"""
        self.__dict__.update(d)
        self = Point(iterable=d)

    # ----------------------------------------------------------------------
    def __getstate__(self):
        """pickle support"""
        return dict(self)

    # ----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))

    # ----------------------------------------------------------------------
    def coordinates(self):
        """
        Retrieves the coordinates of the ``Point`` as an np.array

        .. code-block:: python

            #Usage Example

            >>> coords = point.coordinates()
            >>> coords
                [x1,y1,m1,z1]

        :return:
            An np.array containing coordinate values
        """
        import numpy as np

        if "x" in self and "y" in self and "z" in self:
            return np.array([self["x"], self["y"], self["z"]], dtype=float)
        elif "x" in self and "y" in self:
            return np.array([self["x"], self["y"]], dtype=float)
        else:
            return np.array([])

    @classmethod
    def _from_geojson(cls, data, sr=None):
        if sr == None:
            sr = {"wkid": 4326}

        coordkey = "coordinates"
        for d in data:
            if d.lower() == "coordinates":
                coordkey = d
        coordinates = data[coordkey]

        return cls(
            {
                "x": coordinates[0],
                "y": coordinates[1],
                "spatialReference": sr,
            }
        )


########################################################################
class Polygon(Geometry):
    """
    The ``Polygon`` contains an array of `rings` or `curveRings` and a
    :class:`~arcgis.geometry.SpatialReference`. For ``Polygons`` with curveRings, see the sections on
    JSON curve object and ``Polygon`` with curve. Each ring is represented as
    an array of :class:`~arcgis.geometry.Point`. The first point of each ring is always the same as
    the last point. Each point in the ring is represented as an array of
    numbers. A ``Polygon`` can also have boolean-valued hasM and hasZ fields.

    An empty ``Polygon`` is represented with an empty array for the rings
    field. `Null` and/or `NaNs` embedded in an otherwise defined coordinate
    stream for :class:`~arcgis.geometry.Polyline` and ``Polygons`` is a syntax error.
    Polygons should be topologically simple. Exterior rings are oriented
    clockwise, while holes are oriented counter-clockwise. Rings can touch
    at a vertex or self-touch at a vertex, but there should be no other
    intersections. Polygons returned by services are topologically simple.
    When drawing a polygon, use the even-odd fill rule. The even-odd fill
    rule will guarantee that the polygon will draw correctly even if the
    ring orientation is not as described above.
    """

    _typ = "Polygon"
    _type = "Polygon"

    def __init__(self, iterable=None, **kwargs):
        if iterable is None:
            iterable = ()
        super(Polygon, self).__init__(iterable)
        self.update(kwargs)

    # ----------------------------------------------------------------------
    def svg(self, scale_factor: float = 1, fill_color: Optional[str] = None):
        """
        The ``svg`` method retrieves SVG (Scalable Vecotr Graphic) polygon element.
        SVG defines vector-based graphics in XML format.

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        scale_factor      An optional float. Multiplication factor for the SVG stroke-width.  Default is 1.
        ----------------  -------------------------------------------------------------------------------
        fill_color        An optional string. Hex string for fill color. Default is to use "#66cc99" if geometry is
                          valid, and "#ff3333" if invalid.
        ================  ===============================================================================

        :return:
            The SVG polygon element
        """

        if self.is_empty:
            return "<g />"
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        rings = []
        s = ""

        if "rings" not in self:
            densify_geom = self.densify("ANGLE", -1, 0.1)
            geom_json = json.loads(densify_geom.JSON)["rings"]
        else:
            geom_json = self["rings"]
        for ring in geom_json:
            rings = ring
            exterior_coords = [["{},{}".format(*c) for c in rings]]
            path = " ".join(
                [
                    "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
                    for coords in exterior_coords
                ]
            )
            s += (
                '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
                'stroke-width="{0}" opacity="0.6" d="{1}" />'
            ).format(2.0 * scale_factor, path, fill_color)
        return s

    # ----------------------------------------------------------------------
    @property
    def type(self):
        """Gets the type of the current ``Polyline`` object."""
        return self._type

    # ----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))

    # ----------------------------------------------------------------------
    def coordinates(self):
        """
        Retrieves the coordinates of the ``Polygon`` as an np.array

        .. code-block:: python

            #Usage Example

            >>> coords = polygon.coordinates()
            >>> coords
                [ [x1,y1,m1,z1], [x2,y2,m2,z2],...,[x1,y1,m1,z1] ]

        :return:
            An np.array containing coordinate values
        """
        import numpy as np

        if "rings" in self:
            return np.array(self["rings"], dtype=object)
        else:
            return np.array([])

    # ----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support"""
        self.__dict__.update(d)
        self = Polygon(iterable=d)

    # ----------------------------------------------------------------------
    def __getstate__(self):
        """pickle support"""
        return dict(self)

    @classmethod
    def _from_geojson(cls, data, sr=None):
        if sr is None:
            sr = {"wkid": 4326}

        coordinates = data["coordinates"]
        if data["type"].lower() == "polygon":
            coordinates = [coordinates]

        part_list = []
        for part in coordinates:
            for ring in part:
                part_item = []
                for coord in reversed(ring):
                    part_item.append(coord)
                part_list.append(part_item)
        return cls({"rings": part_list, "spatialReference": sr})


########################################################################
class Polyline(Geometry):
    """
    The ``Polyline`` contains an array of paths or curvePaths and a
    :class:`~arcgis.geometry.SpatialReference`. For ``Polylines`` with curvePaths, see the sections on
    JSON curve object and ``Polyline`` with curve. Each path is represented as
    an array of :class:`~arcgis.geometry.Point`, and each point in the path is represented as an
    array of numbers. A ``Polyline`` can also have boolean-valued hasM and hasZ
    fields.

    .. note::
        See the description of :class:`~arcgis.geometry.MultiPoint` for details on how the point arrays are interpreted.

    An empty ``PolyLine`` is represented with an empty array for the paths
    field. Nulls and/or NaNs embedded in an otherwise defined coordinate
    stream for ``Polylines`` and  :class:`~arcgis.geometry.Polygon` objects is a syntax error.
    """

    _typ = "Polyline"
    _type = "Polyline"

    def __init__(self, iterable=None, **kwargs):
        if iterable is None:
            iterable = {}
        super(Polyline, self).__init__(iterable)
        self.update(kwargs)

    # ----------------------------------------------------------------------
    def svg(self, scale_factor: float = 1, stroke_color: Optional[str] = None):
        """
        Retrieves SVG (Scalable Vector Graphic) polyline element for the LineString geometry.

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        scale_factor      An optional float. Multiplication factor for the SVG stroke-width.  Default is 1.
        ----------------  -------------------------------------------------------------------------------
        stroke_color      An optional string. Hex string for fill color. Default is to use "#66cc99" if geometry is
                          valid, and "#ff3333" if invalid.
        ================  ===============================================================================

        :return:
            The SVG polyline element for the LineString Geometry
        """
        if self.is_empty:
            return "<g />"
        if stroke_color is None:
            stroke_color = "#66cc99" if self.is_valid else "#ff3333"
        paths = []

        if "paths" not in self:
            densify_geom = self.densify("DISTANCE", 1.0, 0.1)
            geom_json = json.loads(densify_geom.JSON)["paths"]
        else:
            geom_json = self["paths"]
        for path in geom_json:
            pnt_format = " ".join(["{0},{1}".format(*c) for c in path])
            s = (
                '<polyline fill="none" stroke="{2}" stroke-width="{1}" '
                'points="{0}" opacity="0.8" />'
            ).format(pnt_format, 2.0 * scale_factor, stroke_color)
            paths.append(s)
        return "<g>" + "".join(paths) + "</g>"

    # ----------------------------------------------------------------------
    @property
    def type(self):
        """Gets the type of the current ``Polyline`` object."""
        return self._type

    # ----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))

    # ----------------------------------------------------------------------
    def coordinates(self):
        """
        Retrieves the coordinates of the ``Polyline`` as a np.array

        .. code-block:: python

            #Usage Example

            >>> coords = polyLine.coordinates()
            >>> coords
                [ [x1,y1,m1,z1], [x2,y2,m2,z2],...]

        :return:
            An np.array containing coordinate values
        """
        import numpy as np

        if "paths" in self:
            return np.array(self["paths"], dtype=object)
        else:
            return np.array([])

    # ----------------------------------------------------------------------
    @property
    def __geo_interface__(self):
        return {
            "type": "MultiLineString",
            "coordinates": [
                [((pt[0], pt[1]) if pt else None) for pt in part]
                for part in self["paths"]
            ],
        }

    # ----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support"""
        self.__dict__.update(d)
        self = Polyline(iterable=d)

    # ----------------------------------------------------------------------
    def __getstate__(self):
        """pickle support"""
        return dict(self)

    @classmethod
    def _from_geojson(cls, data, sr=None):
        if sr is None:
            sr = {"wkid": 4326}
        if data["type"].lower() == "linestring":
            coordinates = [data["coordinates"]]
        else:
            coordinates = data["coordinates"]

        return cls(
            {
                "paths": [[p for p in part] for part in coordinates],
                "spatialReference": sr,
            }
        )


########################################################################
class Envelope(Geometry):
    """
    The ``Envelope`` class represents a rectangle defined by a range of values for each
    coordinate and attribute. It also has a :class:`~arcgis.geometry.SpatialReference` field. The
    fields for the `z` and `m` ranges are optional.

    .. note::
        An empty ``Envelope`` has no points in space and is defined by the presence of an `xmin` field a null value
        or a `NaN` string.
    """

    _typ = "Envelope"
    _type = "Envelope"

    def __init__(self, iterable=None, **kwargs):
        if iterable is None:
            iterable = ()
        super(Envelope, self).__init__(iterable)
        self.update(kwargs)

    # ----------------------------------------------------------------------
    @property
    def type(self):
        """Gets the type of the current ``Polyline`` object."""
        return self._type

    # ----------------------------------------------------------------------
    # def __hash__(self):
    #    return hash(json.dumps(dict(self)))
    # ----------------------------------------------------------------------
    def svg(self, scale_factor: float = 1, fill_color: Optional[str] = None):
        """
        Returns a SVG (Scalable Vector Graphic) envelope element for the ``Envelope`` geometry.


        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        scale_factor      An optional float. Multiplication factor for the SVG circle diameter.  Default is 1.
        ----------------  -------------------------------------------------------------------------------
        fill_color        An optional string. Hex string for fill color. Default is to use "#66cc99" if geometry is
                          valid, and "#ff3333" if invalid.
        ================  ===============================================================================

        :return:
            A SVG envelope element
        """
        return self.polygon.svg(scale_factor, fill_color)

    # ----------------------------------------------------------------------
    def _repr_svg_(self):
        """SVG representation for iPython notebook"""
        return self.polygon._repr_svg_()

    # ----------------------------------------------------------------------
    def coordinates(self):
        """
        The ``coordinates`` method retrieves the coordinates of the ``Envelope`` as a np.array

        .. code-block:: python

            #Usage Example

            >>> coords = envelope.coordinates()
            >>> coords
                [ [x1,y1,m1,z1], [x2,y2,m2,z2],...]

        :return:
            An np.array containing coordinate values
        """
        import numpy as np

        if "xmin" in self and "xmax" in self and "ymin" in self and "ymax" in self:
            if "zmin" in self and "zmax" in self:
                return np.array(
                    [
                        self["xmin"],
                        self["ymin"],
                        self["zmin"],
                        self["xmax"],
                        self["ymax"],
                        self["zmax"],
                    ],
                    dtype=float,
                )
            return np.array(
                [self["xmin"], self["ymin"], self["xmax"], self["ymax"]], dtype=float
            )
        else:
            return np.array([])

    # ----------------------------------------------------------------------
    @property
    def geohash(self):
        """
        The ``geohash`` method retrieves a geohash string of the extent of the ``Envelope.

        :return:
            A geohash String
        """
        return getattr(self.as_arcpy, "geohash", None)

    # ----------------------------------------------------------------------
    @property
    def geohash_covers(self):
        """
        The ``geohash_covers`` method retrieves a list of up to the four longest geohash strings that
        fit within the extent of the ``Envelope``.

        :return:
           A list of geohash Strings
        """
        return getattr(self.as_arcpy, "geohashCovers", None)

    # ----------------------------------------------------------------------
    @property
    def geohash_neighbors(self):
        """
        Gets a list of the geohash neighbor strings for the extent of the
        ``Envelope``.

        :return:
           A list of geohash neighbor Strings
        """
        return getattr(self.as_arcpy, "geohashNeighbors", None)

    # ----------------------------------------------------------------------
    @property
    def height(self):
        """
        Gets the extent height value.

        :return:
            The extent height value
        """
        return getattr(self.as_arcpy, "height", None)

    # ----------------------------------------------------------------------
    @property
    def width(self):
        """
        Gets the extent width value.

        :return:
            The extent width value
        """
        return getattr(self.as_arcpy, "width", None)

    # ----------------------------------------------------------------------
    @property
    def polygon(self):
        """
        Gets the ``Envelope`` as a :class:`~arcgis.geometry.Polygon` object.

        :return:
            A :class:`~arcgis.geometry.Polygon` object
        """
        fe = self.coordinates().tolist()
        if "spatialReference" in self:
            sr = SpatialReference(self["spatialReference"])
        else:
            sr = SpatialReference({"wkid": 4326})
        return Geometry(
            {
                "rings": [
                    [
                        [fe[0], fe[1]],
                        [fe[0], fe[3]],
                        [fe[2], fe[3]],
                        [fe[2], fe[1]],
                        [fe[0], fe[1]],
                    ]
                ],
                "spatialReference": sr,
            }
        )

    # ----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support"""
        self.__dict__.update(d)
        self = Envelope(iterable=d)

    # ----------------------------------------------------------------------
    def __getstate__(self):
        """pickle support"""
        return dict(self)


########################################################################
class SpatialReference(BaseGeometry):
    """
    A ``SpatialReference`` object can be defined using a `well-known ID` (`wkid`) or
    `well-known text` (`wkt`). The default tolerance and resolution values for
    the associated coordinate system are used.

    .. note::
        The x, y and z tolerance
        values are 1 mm or the equivalent in the unit of the coordinate system.
        If the coordinate system uses feet, the tolerance is 0.00328083333 ft.
        The resolution values are 10x smaller or 1/10 the tolerance values.
        Thus, 0.0001 m or 0.0003280833333 ft. For geographic coordinate systems
        using degrees, the equivalent of a mm at the equator is used.

    The `well-known ID` (`WKID`) for a given spatial reference can occasionally
    change. For example, the WGS 1984 Web Mercator (Auxiliary Sphere)
    projection was originally assigned `WKID` 102100, but was later changed
    to 3857. To ensure backward compatibility with older spatial data
    servers, the JSON `wkid` property will always be the value that was
    originally assigned to an SR when it was created.
    An additional property, latestWkid, identifies the current `WKID` value
    (as of a given software release) associated with the same spatial
    reference.

    A ``SpatialReference`` object can optionally include a definition for a `vertical`
    `coordinate system` (`VCS`), which is used to interpret the z-values of a
    geometry. A `VCS` defines units of measure, the location of z = 0, and
    whether the positive vertical direction is up or down. When a vertical
    coordinate system is specified with a `WKID`, the same caveat as
    mentioned above applies.

    .. note::
        There are two `VCS WKID` properties: `vcsWkid` and
        `latestVcsWkid`. A VCS WKT can also be embedded in the string value of
        the wkt property. In other words, the WKT syntax can be used to define
        an SR with both horizontal and vertical components in one string. If
        either part of an SR is custom, the entire SR will be serialized with
        only the wkt property.

    .. note::
        Starting at 10.3, Image Service supports image coordinate systems.
    """

    _typ = "SpatialReference"
    _type = "SpatialReference"

    def __init__(self, iterable=None, **kwargs):
        super(SpatialReference, self)
        if iterable is None:
            iterable = {}
        if isinstance(iterable, int):
            iterable = {"wkid": iterable}
        if isinstance(iterable, str):
            iterable = {"wkt": iterable}
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY and isinstance(iterable, arcpy.SpatialReference):
            if iterable.factoryCode:
                iterable = {"wkid": iterable.factoryCode}
            else:
                iterable = {"wkt": iterable.exportToString()}
        if len(iterable) > 0:
            self.update(iterable)
        if len(kwargs) > 0:
            self.update(kwargs)

    # ----------------------------------------------------------------------
    @property
    def type(self):
        """Gets the type of the current ``Point`` object."""
        return self._type

    # ----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))

    # ----------------------------------------------------------------------
    _repr_svg_ = None

    # ----------------------------------------------------------------------
    def svg(self, scale_factor: float = 1, fill_color: Optional[str] = None):
        """
        Retrieves SVG (Scalable Vector Graphic) polygon element for a ``SpatialReference`` field.

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        scale_factor      An optional float. Multiplication factor for the SVG stroke-width.  Default is 1.
        ----------------  -------------------------------------------------------------------------------
        fill_color        An optional string. Hex string for fill color. Default is to use "#66cc99" if geometry is
                          valid, and "#ff3333" if invalid.
        ================  ===============================================================================

        :return:
            The SVG element
        """
        return "<g/>"

    # ----------------------------------------------------------------------
    def __eq__(self, other):
        """checks if the spatial reference is not equal"""
        if "wkt" in self and "wkt" in other and self["wkt"] == other["wkt"]:
            return True
        elif "wkid" in self and "wkid" in other and self["wkid"] == other["wkid"]:
            return True
        return False

    # ----------------------------------------------------------------------
    def __ne__(self, other):
        """checks if the two values are unequal"""
        return self.__eq__(other) == False

    # ----------------------------------------------------------------------
    @property
    def as_arcpy(self):
        """
        The ``as_arcpy`` property retrieves the class as an ``arcpy SpatialReference`` object.

        :return:
            An ``arcpy SpatialReference`` object
        """
        HASARCPY, HASSHAPELY = _check_geometry_engine()
        if HASARCPY:
            if "wkid" in self:
                return arcpy.SpatialReference(self["wkid"])
            elif "wkt" in self:
                sr = arcpy.SpatialReference()
                sr.loadFromString(self["wkt"])
                return sr
        return None

    # ----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support"""
        self.__dict__.update(d)
        self = SpatialReference(iterable=d)

    # ----------------------------------------------------------------------
    def __getstate__(self):
        """pickle support"""
        return dict(self)
