"""
Holds Delegate and Accessor Logic
"""
from __future__ import annotations
import logging
import pandas as pd
from collections.abc import Iterable

from ._internals import register_dataframe_accessor, register_series_accessor
from pandas.core.dtypes.common import infer_dtype_from_object
from ._array import GeoType
from ._io.fileops import (
    to_featureclass,
    from_featureclass,
    _sanitize_column_names,
    read_feather,
)

from arcgis.auth.tools import LazyLoader

os = LazyLoader("os")
copy = LazyLoader("copy")
uuid = LazyLoader("uuid")
shutil = LazyLoader("shutil")
datetime = LazyLoader("datetime")
np = LazyLoader("numpy")
tempfile = LazyLoader("tempfile")
warnings = LazyLoader("warnings")
features = LazyLoader("arcgis.features")
_gis = LazyLoader("arcgis.gis")
_geometry = LazyLoader("arcgis.geometry")
_mixins = LazyLoader("arcgis._impl.common._mixins")
_isd = LazyLoader("arcgis._impl.common._isd")
_pa = LazyLoader("pyarrow")

_LOGGER = logging.getLogger(__name__)
############################################################################


def _is_geoenabled(df):
    """
    Checks if a Panda's DataFrame is 'geo-enabled'.

    This means that a spatial column is defined and is a GeoArray

    :return: boolean
    """
    try:
        if (
            isinstance(df, pd.DataFrame)
            and hasattr(df, "spatial")
            and df.spatial.name
            and df[df.spatial.name].dtype.name.lower() == "geometry"
        ):
            return True
        else:
            return False
    except:
        return False


###########################################################################
@pd.api.extensions.register_series_accessor("geom")
class GeoSeriesAccessor:
    """ """

    _data = None
    _index = None
    _name = None
    # ----------------------------------------------------------------------

    def __init__(self, obj):
        """initializer"""
        self._validate(obj)
        self._data = obj.values
        self._index = obj.index
        self._name = obj.name

    # ----------------------------------------------------------------------
    @staticmethod
    def _validate(obj):
        if not is_geometry_type(obj):
            raise AttributeError(
                "Cannot use 'geom' accessor on objects of "
                "dtype '{}'.".format(obj.dtype)
            )

    ##---------------------------------------------------------------------
    ##   Accessor Properties
    ##---------------------------------------------------------------------
    @property
    def area(self):
        """
        The ``area`` method retrieves the :class:`~arcgis.features.Feature` object's area.

         :return:
             A float in a series
        """
        return pd.Series(self._data.area, name="area", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def as_arcpy(self):
        """
        The ``as_arcpy`` method retrieves the features as an ArcPy `geometry <https://pro.arcgis.com/en/pro-app/latest/arcpy/classes/geometry.htm>`_
        object.

        :return:
            An arcpy.geometry as a series
        """
        return pd.Series(self._data.as_arcpy, name="as_arcpy", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def as_shapely(self):
        """
        The ``as_shapely`` method retrieves the features as Shapely`Geometry <https://shapely.readthedocs.io/en/stable/manual.html#geometric-objects>`_

        :return:
            shapely.Geometry objects in a series
        """
        return pd.Series(self._data.as_shapely, name="as_shapely", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def centroid(self):
        """
        Returns the feature's centroid

        :return: tuple (x,y) in series
        """
        return pd.Series(self._data.centroid, name="centroid", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def extent(self):
        """
        The ``extent`` method retrieves the feature's extent

        :return:
            A tuple (xmin,ymin,xmax,ymax) in series
        """
        return pd.Series(self._data.extent, name="extent", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def first_point(self):
        """
        The ``first_point`` property retrieves the feature's first :class:`~arcgis.geometry.Point` object

        :return:
            A :class:`~arcgis.geometry.Point` object
        """
        return pd.Series(self._data.first_point, name="first_point", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def geoextent(self):
        """
        The ``geoextent`` method retrieves the :class:`~arcgis.geometry.Geometry` object's extents

        :return:
            A Series of Floats
        """
        # res = self._data.geoextent
        # res.index = self._index
        return pd.Series(self._data.geoextent, name="geoextent", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """
        The ``geometry_type`` property retrieves the :class:`~arcgis.geometry.Geometry` object's type.

        :return:
            A Series of strings
        """
        return pd.Series(
            self._data.geometry_type, name="geometry_type", index=self._index
        )

    # ----------------------------------------------------------------------
    @property
    def hull_rectangle(self):
        """
        The ``hull_rectangle`` retrieves a space-delimited string of the coordinate pairs of the convex hull

        :return:
            A Series of strings
        """
        return pd.Series(
            self._data.hull_rectangle,
            name="hull_rectangle",
            index=self._index,
        )

    # ----------------------------------------------------------------------
    @property
    def has_z(self):
        """
        The ``has_z`` method determines if the :class:`~arcgis.geometry.Geometry` object has a `Z` value

        :return:
            A Series of Booleans
        """
        return pd.Series(self._data.has_z, name="has_z", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def has_m(self):
        """
        The ``has_m`` method determines if the :class:`~arcgis.geometry.Geometry` objects has an `M` value

        :return:
            A Series of Booleans
        """
        return pd.Series(self._data.has_m, name="has_m", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def is_empty(self):
        """
        The ``is_empty`` method determines if the :class:`~arcgis.geometry.Geometry` object is empty.

        :return:
            A Series of Booleans
        """
        return pd.Series(self._data.is_empty, name="is_empty", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def is_multipart(self):
        """
        The ``is_multipart`` method determines if features has multiple parts.

        :return:
            A Series of Booleans
        """
        return pd.Series(
            self._data.is_multipart, name="is_multipart", index=self._index
        )

    # ----------------------------------------------------------------------
    @property
    def is_valid(self):
        """
        The ``is_valid`` method determines if the features :class:`~arcgis.geometry.Geometry` is valid

        :return:
            A Series of Booleans
        """
        return pd.Series(self._data.is_valid, name="is_valid", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def JSON(self):
        """
        The ``JSON`` method creates a JSON string out of the :class:`~arcgis.geometry.Geometry` object.

        :return: Series of strings
        """
        return pd.Series(self._data.JSON, name="JSON", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def label_point(self):
        """
        The ``label_point`` method determines the :class:`~arcgis.geometry.Point` for the optimal label location.

        :return:
            A Series of :class:`~arcgis.geometry.Geometry` object
        """
        return pd.Series(self._data.label_point, name="label_point", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def last_point(self):
        """
        The ``last_point`` method retrieves the :class:`~arcgis.geometry.Geometry` of the last point in a feature.

        :return:
            A Series of :class:`~arcgis.geometry.Geometry` objects
        """
        return pd.Series(self._data.last_point, name="last_point", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def length(self):
        """
        The ``length`` method retrieves the length of the features.

        :return:
            A Series of floats
        """
        return pd.Series(self._data.length, name="length", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def length3D(self):
        """
        The ``length3D`` method retrieves the length of the features

        :return:
            A Series of floats
        """
        return pd.Series(self._data.length3D, name="length3D", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def part_count(self):
        """
        The ``part_count`` method retrieves the number of parts in a feature's :class:`~arcgis.geometry.Geometry`

        :return:
            A Series of Integers
        """
        return pd.Series(self._data.part_count, name="part_count", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def point_count(self):
        """
        The ``point_count`` method retrieves the number of :class:`~arcgis.geometry.Point` objects in a feature's
        :class:`~arcgis.geometry.Geometry`.

        :return:
            A Series of Integers
        """
        return pd.Series(self._data.part_count, name="point_count", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def spatial_reference(self):
        """
        The ``spatial_reference`` method retrieves the  :class:`~arcgis.geometry.SpatialReference` of the
        :class:`~arcgis.geometry.Geometry`

        :return:
            A Series of :class:`~arcgis.geometry.SpatialReference` objects.
        """
        return pd.Series(
            self._data.spatial_reference,
            name="spatial_reference",
            index=self._index,
        )

    # ----------------------------------------------------------------------
    @property
    def true_centroid(self):
        """
        The ``true_centroid`` method retrieves the true centroid of the :class:`~arcgis.geometry.Geometry` object.

        :return:
            A Series of :class:`~arcgis.geometry.Point` objects
        """
        return pd.Series(
            self._data.true_centroid, name="true_centroid", index=self._index
        )

    # ----------------------------------------------------------------------
    @property
    def WKB(self):
        """
        The ``WKB`` method retrieves the :class:`~arcgis.geometry.Geometry` object as a ``WKB``

        :return:
            A Series of Bytes
        """
        return pd.Series(self._data.WKB, name="WKB", index=self._index)

    # ----------------------------------------------------------------------
    @property
    def WKT(self):
        """
        The ``WKT`` method retrieves the :class:`~arcgis.geometry.Geometry` object's `WKT <http://wiki.gis.com/wiki/index.php/Well-known_text>`_

        :return: Series of String
        """
        return pd.Series(self._data.WKT, name="WKT", index=self._index)

    ##---------------------------------------------------------------------
    ##  Accessor Geometry Method
    ##---------------------------------------------------------------------
    def angle_distance_to(self, second_geometry, method="GEODESIC"):
        """
        The ``angle_distance_to`` method retrieves a tuple of angle and distance to another point using a
        measurement method.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required Geometry.  A :class:`~arcgis.geometry.Geometry` object.
        ---------------     --------------------------------------------------------------------
        method              Optional String. `PLANAR` measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). `GEODESIC`, `GREAT_ELLIPTIC`, and
                            `LOXODROME` measurement types may be chosen as an alternative, if desired.
        ===============     ====================================================================

        :return:
            A Series where each element is a tuple of angle and distance to another point using a measurement type.
        """
        res = self._data.angle_distance_to(
            **{"second_geometry": second_geometry, "method": method}
        )
        return pd.Series(res, index=self._index, name="angle_distance_to")

    # ----------------------------------------------------------------------
    def boundary(self):
        """
        The ``boundary`` method constructs the boundary of the :class:`~arcgis.geometry.Geometry` object.

        :return:
           A Pandas Series of :class:`~arcgis.geometry.Polyline` objects
        """
        return pd.Series(self._data.boundary(), index=self._index, name="boundary")

    # ----------------------------------------------------------------------
    def buffer(self, distance):
        """
        The ``buffer`` method constructs a :class:`~arcgis.geometry.Polygon` at a specified distance from the
        :class:`~arcgis.geometry.Geometry` object.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        distance            Required float. The buffer distance. The buffer distance is in the
                            same units as the geometry that is being buffered.
                            A negative distance can only be specified against a polygon geometry.
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Polygon` objects
        """
        return pd.Series(
            self._data.buffer(**{"distance": distance}),
            index=self._index,
            name="buffer",
        )

    # ----------------------------------------------------------------------
    def clip(self, envelope):
        """
        The ``clip`` method constructs the intersection of the :class:`~arcgis.geometry.Geometry` object and the
        specified extent.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        envelope            required tuple. The tuple must have (XMin, YMin, XMax, YMax) each value
                            represents the lower left bound and upper right bound of the extent.
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects

        """
        return pd.Series(
            self._data.clip(**{"envelope": envelope}),
            index=self._index,
            name="clip",
        )

    # ----------------------------------------------------------------------
    def contains(self, second_geometry, relation=None):
        """
        The ``contains`` method indicates if the base :class:`~arcgis.geometry.Geometry` contains the
        comparison :class:`~arcgis.geometry.Geometry`.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional string. The spatial relationship type.

                            + BOUNDARY - Relationship has no restrictions for interiors or boundaries.
                            + CLEMENTINI - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            + PROPER - Boundaries of geometries must not intersect.
        ===============     ====================================================================

        :return:
            A Pandas Series of booleans indicating success (True), or failure (False)
        """
        return pd.Series(
            self._data.contains(
                **{"second_geometry": second_geometry, "relation": relation}
            ),
            name="contains",
            index=self._index,
        )

    # ----------------------------------------------------------------------
    def convex_hull(self):
        """
        The ``convex_hull`` method constructs the :class:`~arcgis.geometry.Geometry` that is the minimal bounding
        :class:`~arcgis.geometry.Polygon` such that all outer angles are convex.

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects
        """
        return pd.Series(
            self._data.convex_hull(), index=self._index, name="convex_hull"
        )

    # ----------------------------------------------------------------------
    def crosses(self, second_geometry):
        """
        The ``crosses`` method indicates if the two :class:`~arcgis.geometry.Geometry` objects intersect in a geometry
        of a lesser shape type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of booleans indicating success (True), or failure (False)

        """
        return pd.Series(
            self._data.crosses(**{"second_geometry": second_geometry}),
            name="crosses",
            index=self._index,
        )

    # ----------------------------------------------------------------------
    def cut(self, cutter):
        """
        The ``cut`` method splits this :class:`~arcgis.geometry.Geometry` into a part to the left of the cutting
        :class:`~arcgis.geometry.Polyline` and a part to the right of it.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        cutter              Required :class:`~arcgis.geometry.Polyline`. The cutting polyline geometry
        ===============     ====================================================================

        :return:
            A Pandas Series where each element is a list of two :class:`~arcgis.geometry.Geometry` objects

        """
        return pd.Series(
            self._data.cut(**{"cutter": cutter}),
            index=self._index,
            name="cut",
        )

    # ----------------------------------------------------------------------
    def densify(self, method, distance, deviation):
        """
        The ``densify`` method creates a new :class:`~arcgis.geometry.Geometry` with added vertices

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. The type of densification, DISTANCE, ANGLE, or GEODESIC
        ---------------     --------------------------------------------------------------------
        distance            Required float. The maximum distance between vertices. The actual
                            distance between vertices will usually be less than the maximum
                            distance as new vertices will be evenly distributed along the
                            original segment. If using a type of DISTANCE or ANGLE, the
                            distance is measured in the units of the geometry's spatial
                            reference. If using a type of GEODESIC, the distance is measured
                            in meters.
        ---------------     --------------------------------------------------------------------
        deviation           Required float. Densify uses straight lines to approximate curves.
                            You use deviation to control the accuracy of this approximation.
                            The deviation is the maximum distance between the new segment and
                            the original curve. The smaller its value, the more segments will
                            be required to approximate the curve.
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects

        """
        return pd.Series(
            self._data.densify(
                **{
                    "method": method,
                    "distance": distance,
                    "deviation": deviation,
                }
            ),
            index=self._index,
            name="densify",
        )

    # ----------------------------------------------------------------------
    def difference(self, second_geometry):
        """
        The ``difference`` method constructs the :class:`~arcgis.geometry.Geometry` that is composed only of the
        region unique to the base geometry but not part of the other geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects
        """
        return pd.Series(
            self._data.difference(**{"second_geometry": second_geometry}),
            index=self._index,
            name="difference",
        )

    # ----------------------------------------------------------------------
    def disjoint(self, second_geometry):
        """
        The ``disjoint`` method indicates if the base and comparison :class:`~arcgis.geometry.Geometry` objects share
        no :class:`~arcgis.geometry.Point` objects in common.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of booleans indicating success (True), or failure (False)

        """
        res = self._data.disjoint(**{"second_geometry": second_geometry})
        return pd.Series(res, index=self._index, name="disjoint")

    # ----------------------------------------------------------------------
    def distance_to(self, second_geometry):
        """
        The ``distance_to`` method retrieves the minimum distance between two :class:`~arcgis.geometry.Geometry`.
        If the geometries intersect, the minimum distance is 0.

        .. note::
            Both geometries must have the same projection.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of floats

        """
        res = self._data.distance_to(**{"second_geometry": second_geometry})
        return pd.Series(res, index=self._index, name="distance_to")

    # ----------------------------------------------------------------------
    def equals(self, second_geometry):
        """
        The ``equals`` method indicates if the base and comparison :class:`~arcgis.geometry.Geometry` objects are of
        the same shape type and define the same set of :class:`~arcgis.geometry.Point` objects in the plane.

        .. note::
            This is a 2D comparison only; M and Z values are ignored.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of booleans indicating success (True), or failure (False)


        """
        if isinstance(second_geometry, _geometry.Geometry):
            return pd.Series(
                self._data.equals(**{"second_geometry": second_geometry}),
                name="equals",
                index=self._index,
            )
        elif isinstance(second_geometry, GeoSeriesAccessor):
            # Do a GeoArray eq
            return self._data == second_geometry._data

    # ----------------------------------------------------------------------
    def generalize(self, max_offset):
        """
        The ``generalize`` method creates a new simplified :class:`~arcgis.geometry.Geometry` using a specified maximum
        offset tolerance.

        .. note::
            This only works on :class:`~arcgis.geometry.Polyline` and :class:`~arcgis.geometry.Polygon` objects.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        max_offset          Required float. The maximum offset tolerance.
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects

        """
        res = self._data.generalize(**{"max_offset": max_offset})
        return pd.Series(res, index=self._index, name="generalize")

    # ----------------------------------------------------------------------
    def get_area(self, method, units=None):
        """
        The ``get_area`` method retreives the area of the feature using a measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. `PLANAR` measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). `GEODESIC`,
                            `GREAT_ELLIPTIC`, `LOXODROME`, and `PRESERVE_SHAPE` measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Optional String. Areal unit of measure keywords:` ACRES | ARES | HECTARES
                            | SQUARECENTIMETERS | SQUAREDECIMETERS | SQUAREINCHES | SQUAREFEET
                            | SQUAREKILOMETERS | SQUAREMETERS | SQUAREMILES |
                            SQUAREMILLIMETERS | SQUAREYARDS`
        ===============     ====================================================================

        :return:
            A Pandas Series of floats

        """
        res = self._data.get_area(**{"method": method, "units": units})
        return pd.Series(res, index=self._index, name="get_area")

    # ----------------------------------------------------------------------
    def get_length(self, method, units):
        """
        The ``get_length`` method retrieves the length of the feature using a measurement type.

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
            A A Pandas Series of floats

        """
        res = self._data.get_length(**{"method": method, "units": units})
        return pd.Series(res, index=self._index, name="get_length")

    # ----------------------------------------------------------------------
    def get_part(self, index=None):
        """
        The ``get_part`` method retrieves an array of :class:`~arcgis.geometry.Point` objects for a particular part of
        :class:`~arcgis.geometry.Geometry` or an array containing a number of arrays, one for each part.

        **requires arcpy**

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        index               Required Integer. The index position of the geometry.
        ===============     ====================================================================

        :return:
            AnA Pandas Series of  arcpy.Arrays

        """
        res = self._data.get_part(**{"index": index})
        return pd.Series(res, index=self._index, name="get_part")

    # ----------------------------------------------------------------------
    def intersect(self, second_geometry, dimension=1):
        """
        The ``intersect`` method constructs a :class:`~arcgis.geometry.Geometry` that is the geometric intersection of
        the two input geometries. Different dimension values can be used to create
        different shape types.

        .. note::
            The intersection of two :class:`~arcgis.geometry.Geometry` objects of the
            same shape type is a geometry containing only the regions of overlap
            between the original geometries.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ---------------     --------------------------------------------------------------------
        dimension           Required Integer. The topological dimension (shape type) of the
                            resulting geometry.

                            + 1  -A zero-dimensional geometry (point or multipoint).
                            + 2  -A one-dimensional geometry (polyline).
                            + 4  -A two-dimensional geometry (polygon).

        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects

        """
        return pd.Series(
            self._data.intersect(
                **{
                    "second_geometry": second_geometry,
                    "dimension": dimension,
                }
            ),
            name="intersect",
            index=self._index,
        )

    # ----------------------------------------------------------------------
    def measure_on_line(self, second_geometry, as_percentage=False):
        """
        The ``measure_on_line`` method retrieves the measure from the start :class:`~arcgis.geometry.Point` of this line
        to the ``in_point``.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional Boolean. If False, the measure will be returned as a
                            distance; if True, the measure will be returned as a percentage.
        ===============     ====================================================================

        :return:
            A Pandas Series of floats

        """
        res = self._data.measure_on_line(
            **{
                "second_geometry": second_geometry,
                "as_percentage": as_percentage,
            }
        )
        return pd.Series(res, index=self._index, name="measure_on_line")

    # ----------------------------------------------------------------------
    def overlaps(self, second_geometry):
        """
        The ``overlaps`` method indicates if the intersection of the two :class:`~arcgis.geometry.Geometry` objects has
        the same shape type as one of the input geometries and is not equivalent to
        either of the input geometries.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of booleans indicating success (True), or failure (False)

        """
        return pd.Series(
            self._data.overlaps(**{"second_geometry": second_geometry}),
            name="overlaps",
            index=self._index,
        )

    # ----------------------------------------------------------------------
    def point_from_angle_and_distance(self, angle, distance, method="GEODESCIC"):
        """
        The ``point_from_angle_and_distance`` retrieves a :class:`~arcgis.geometry.Point` at a given angle and distance
        in degrees and meters using the specified measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        angle               Required Float. The angle in degrees to the returned point.
        ---------------     --------------------------------------------------------------------
        distance            Required Float. The distance in meters to the returned point.
        ---------------     --------------------------------------------------------------------
        method              Optional String. PLANAR measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC,
                            LOXODROME, and PRESERVE_SHAPE measurement types may be chosen as
                            an alternative, if desired.
        ===============     ====================================================================

        :return:
         A Pandas Series of :class:`~arcgis.geometry.Geometry` objects


        """
        res = self._data.point_from_angle_and_distance(
            **{"angle": angle, "distance": distance, "method": method}
        )
        return pd.Series(res, index=self._index, name="point_from_angle_and_distance")

    # ----------------------------------------------------------------------
    def position_along_line(self, value, use_percentage=False):
        """
        The ``position_along_line`` method retrieves a :class:`~arcgis.geometry.Point` on a line at a specified
        distance from the beginning of the line.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required Float. The distance along the line.
        ---------------     --------------------------------------------------------------------
        use_percentage      Optional Boolean. The distance may be specified as a fixed unit
                            of measure or a ratio of the length of the line. If True, value
                            is used as a percentage; if False, value is used as a distance.
                            For percentages, the value should be expressed as a double from
                            0.0 (0%) to 1.0 (100%).
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects.

        """
        res = self._data.position_along_line(
            **{"value": value, "use_percentage": use_percentage}
        )
        return pd.Series(res, index=self._index, name="position_along_line")

    # ----------------------------------------------------------------------
    def project_as(self, spatial_reference, transformation_name=None):
        """
        The ``project_as`` method projects a :class:`~arcgis.geometry.Geometry`and optionally applies a
        ``geotransformation``.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required :class:`~arcgis.geometry.SpatialReference`.
                                 The new spatial reference. This can be a
                                 :class:`~arcgis.geometry.SpatialReference` object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Required String. The `geotransformation` name.
        ====================     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects
        """
        res = self._data.project_as(
            **{
                "spatial_reference": spatial_reference,
                "transformation_name": transformation_name,
            }
        )
        return pd.Series(res, index=self._index, name="project_as")

    # ----------------------------------------------------------------------
    def query_point_and_distance(self, second_geometry, use_percentage=False):
        """
        The ``query_point_and_distance`` finds the :class:`~arcgis.geometry.Point` on the
        :class:`~arcgis.geometry.Polyline` nearest to the ``in_point`` and the
        distance between those points.

        .. note::
            ``query_point_and_distance`` also returns information about the
            side of the line the ``in_point`` is on as well as the distance along
            the line where the nearest point occurs.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional boolean - if False, the measure will be returned as
                            distance, True, measure will be a percentage
        ===============     ====================================================================

        :return:
            A Pandas Series of tuples

        """
        res = self._data.query_point_and_distance(
            **{
                "second_geometry": second_geometry,
                "use_percentage": use_percentage,
            }
        )
        return pd.Series(res, index=self._index, name="query_point_and_distance")

    # ----------------------------------------------------------------------
    def segment_along_line(self, start_measure, end_measure, use_percentage=False):
        """
        The ``segment_along_line`` method retrieves a :class:`~arcgis.geometry.Polyline` between start and end measures.
        Similar to :attr:`~arcgis.geometry.Polyline.positionAlongLine` but will return a polyline segment between
        two points on the polyline instead of a single :class:`~arcgis.geometry.Point`.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        start_measure       Required Float. The starting distance from the beginning of the line.
        ---------------     --------------------------------------------------------------------
        end_measure         Required Float. The ending distance from the beginning of the line.
        ---------------     --------------------------------------------------------------------
        use_percentage      Optional Boolean. The start and end measures may be specified as
                            fixed units or as a ratio.
                            If ``True``, ``start_measure`` and ``end_measure`` are used as a percentage; if
                            ``False``, ``start_measure`` and ``end_measure`` are used as a distance. For
                            percentages, the measures should be expressed as a double from 0.0
                            (0 percent) to 1.0 (100 percent).
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects

        """
        res = self._data.segment_along_line(
            **{
                "start_measure": start_measure,
                "end_measure": end_measure,
                "use_percentage": use_percentage,
            }
        )
        return pd.Series(res, index=self._index, name="segment_along_line")

    # ----------------------------------------------------------------------
    def snap_to_line(self, second_geometry):
        """
        The ``snap_to_line`` method creates a new :class:`~arcgis.geometry.Point` based on ``in_point`` snapped to this
        :class:`~arcgis.geometry.Geometry` object.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects

        """
        res = self._data.snap_to_line(**{"second_geometry": second_geometry})
        return pd.Series(res, index=self._index, name="snap_to_line")

    # ----------------------------------------------------------------------
    def symmetric_difference(self, second_geometry):
        """
        The ``symmetric_difference`` method constructs the :class:`~arcgis.geometry.Geometry` that is the union of two
        geometries minus the intersection of those geometries.

        .. note::
            The two input :class:`~arcgis.geometry.Geometry` must be the same shape type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects
        """
        res = self._data.symmetric_difference(**{"second_geometry": second_geometry})
        return pd.Series(res, index=self._index, name="symmetric_difference")

    # ----------------------------------------------------------------------
    def touches(self, second_geometry):
        """
        The ``touches`` method indicates if the boundaries of the :class:`~arcgis.geometry.Geometry` intersect.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of booleans indicating touching (True), or not touching (False)
        """
        return pd.Series(
            self._data.touches(**{"second_geometry": second_geometry}),
            name="touches",
            index=self._index,
        )

    # ----------------------------------------------------------------------
    def union(self, second_geometry):
        """
        The ``union`` method constructs the :class:`~arcgis.geometry.Geometry` object that is the set-theoretic union
        of the input geometries.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ===============     ====================================================================

        :return:
            A Pandas Series of :class:`~arcgis.geometry.Geometry` objects
        """
        res = self._data.union(**{"second_geometry": second_geometry})
        return pd.Series(res, index=self._index, name="union")

    # ----------------------------------------------------------------------
    def within(self, second_geometry, relation=None):
        """
        The ``within`` method indicates if the base :class:`~arcgis.geometry.Geometry` is within the comparison
        :class:`~arcgis.geometry.Geometry`.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required :class:`~arcgis.geometry.Geometry`. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional String. The spatial relationship type.

                            - BOUNDARY  - Relationship has no restrictions for interiors or boundaries.
                            - CLEMENTINI  - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            - PROPER  - Boundaries of geometries must not intersect.

        ===============     ====================================================================

        :return:
            A Pandas Series of booleans indicating within (True), or not within (False)

        """
        return pd.Series(
            self._data.within(
                **{"second_geometry": second_geometry, "relation": relation}
            ),
            name="within",
            index=self._index,
        )


# --------------------------------------------------------------------------
def is_geometry_type(obj):
    t = getattr(obj, "dtype", obj)
    try:
        return isinstance(t, GeoType) or issubclass(t, GeoType)
    except Exception:
        return False


###########################################################################
@register_dataframe_accessor("spatial")
class GeoAccessor(object):
    """
    The ``GeoAccessor`` class adds a spatial namespace that performs spatial operations on the given Pandas
    `DataFrame. <https://pandas.pydata.org/docs/reference/frame.html#dataframe>`_
    The ``GeoAccessor`` class includes visualization, spatial indexing, IO and dataset level properties.
    """

    _sr = None
    _viz = None
    _data = None
    _name = None
    _index = None
    _stype = None
    _kdtree = None
    _sindex = None
    _sfname = None
    _renderer = None
    _HASARCPY = None
    _HASSHAPELY = None
    # ----------------------------------------------------------------------

    def __init__(self, obj):
        self._data = obj
        self._index = obj.index
        self._name = None

    # ----------------------------------------------------------------------
    @property
    def _meta(self):
        """
        Users have the ability to store the source reference back to the
        dataframe.  This will allow the user to compare SeDF with source
        data such as FeatureLayers and Feature Classes.

        ===============   =======================================================
        **Parameter**     **Description**
        ---------------   -------------------------------------------------------
        source            String/Object Reference to the source of the dataframe.
        ===============   =======================================================

        :return: object/string

        """
        from arcgis.features.geo._tools import _metadata

        if (
            "metadata" in self._data.attrs
            and self._data.attrs["metadata"]
            and isinstance(self._data.attrs["metadata"], _metadata._Metadata)
        ):
            return self._data.attrs["metadata"]
        else:
            self._meta = _metadata._Metadata()
            return self._meta

    # ----------------------------------------------------------------------
    @_meta.setter
    def _meta(self, source):
        """
        See main ``_meta`` property docstring
        """
        from ._tools import _metadata

        if not "metadata" in self._data.attrs and isinstance(
            source, _metadata._Metadata
        ):  # creates the attrs entry
            self._data.attrs["metadata"] = source
        elif (
            "metadata" in self._data.attrs
            and isinstance(source, _metadata._Metadata)
            and source != self._data.attrs["metadata"]
        ):  # sets the new metadata value
            self._data.attrs["metadata"] = source
        elif source is None:  # resets/drops the source
            self._data.attrs["metadata"] = _metadata._Metadata()

    # ----------------------------------------------------------------------
    @property
    def renderer(self):
        """
        The ``renderer`` property defines the renderer for the Spatially-enabled DataFrame.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        value                   Required dict. If none is given, then the value is reset
        ==================      ====================================================================

        :return:
            ```InsensitiveDict```: A case-insensitive ``dict`` like object used to update and alter JSON
            A varients of a case-less dictionary that allows for dot and bracket notation.

        """
        if self._meta.renderer is None:
            self._meta.renderer = self._build_renderer()
        return self._meta.renderer

    # ----------------------------------------------------------------------
    @renderer.setter
    def renderer(self, renderer):
        """
        See main ``renderer`` property docstring
        """
        if renderer is None:
            renderer = self._build_renderer()
        if isinstance(renderer, dict):
            renderer = _isd.InsensitiveDict.from_dict(renderer)
        elif isinstance(renderer, _mixins.PropertyMap):
            renderer = _isd.InsensitiveDict.from_dict(dict(renderer))
        elif isinstance(renderer, _isd.InsensitiveDict):
            pass
        else:
            raise ValueError("renderer must be a dictionary type.")
        self._meta.renderer = renderer

    # ----------------------------------------------------------------------
    def _build_renderer(self):
        """sets the default symbology"""
        if self._meta.source and hasattr(self._meta.source, "properties"):
            return self._meta.renderer
        elif self.name is None:
            self._meta.renderer = _isd.InsensitiveDict({})
            return self._meta.renderer
        gt = self.geometry_type[0]
        base_renderer = {
            "labelingInfo": None,
            "label": "",
            "description": "",
            "type": "simple",
            "symbol": None,
        }
        if gt.lower() in ["point", "multipoint"]:
            base_renderer["symbol"] = {
                "color": [0, 128, 0, 128],
                "size": 18,
                "angle": 0,
                "xoffset": 0,
                "yoffset": 0,
                "type": "esriSMS",
                "style": "esriSMSCircle",
                "outline": {
                    "color": [0, 128, 0, 255],
                    "width": 1,
                    "type": "esriSLS",
                    "style": "esriSLSSolid",
                },
            }

        elif gt.lower() == "polyline":
            base_renderer["symbol"] = {
                "type": "esriSLS",
                "style": "esriSLSSolid",
                "color": [0, 128, 0, 128],
                "width": 1,
            }
        elif gt.lower() == "polygon":
            base_renderer["symbol"] = {
                "type": "esriSFS",
                "style": "esriSFSSolid",
                "color": [0, 128, 0, 128],
                "outline": {
                    "type": "esriSLS",
                    "style": "esriSLSSolid",
                    "color": [110, 110, 110, 255],
                    "width": 1,
                },
            }
        self._meta.renderer = _isd.InsensitiveDict(base_renderer)
        return self._meta.renderer

    # ----------------------------------------------------------------------
    def _repr_svg_(self):
        """draws the dataframe as SVG features"""

        if self.name:

            def fn(g, n):
                return getattr(g, n, None)() if g is not None else None

            vals = np.vectorize(fn, otypes="O")(self._data[self.name], "svg")
            svg = "\n".join(vals.tolist())
            svg_top = (
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'xmlns:xlink="http://www.w3.org/1999/xlink" '
            )
            if len(self._data) == 0:
                return svg_top + "/>"
            else:
                # Establish SVG canvas that will fit all the data + small space
                xmin, ymin, xmax, ymax = self.full_extent
                if xmin == xmax and ymin == ymax:
                    # This is a point; buffer using an arbitrary size
                    xmin, ymin, xmax, ymax = (
                        xmin - 0.001,
                        ymin - 0.001,
                        xmax + 0.001,
                        ymax + 0.001,
                    )
                else:
                    # Expand bounds by a fraction of the data ranges
                    expand = 0.04  # or 4%, same as R plots
                    widest_part = max([xmax - xmin, ymax - ymin])
                    expand_amount = widest_part * expand
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
                    scale_factor = 1
                view_box = "{0} {1} {2} {3}".format(xmin, ymin, dx, dy)
                transform = "matrix(1,0,0,-1,0,{0})".format(ymax + ymin)
                return svg_top + (
                    'width="{1}" height="{2}" viewBox="{0}" '
                    'preserveAspectRatio="xMinYMin meet">'
                    '<g transform="{3}">{4}</g></svg>'
                ).format(view_box, width, height, transform, svg)
        return

    # ----------------------------------------------------------------------
    @staticmethod
    def from_parquet(path: str, columns: list = None, **kwargs) -> pd.DataFrame:
        """
        Load a Parquet object from the file path, returning a Spatially Enabled DataFrame.

        You can read a subset of columns in the file using the ``columns`` parameter.
        However, the structure of the returned Spatially Enabled DataFrame will depend on which
        columns you read:

        * if no geometry columns are read, this will raise a ``ValueError`` - you
          should use the pandas `read_parquet` method instead.
        * if the primary geometry column saved to this file is not included in
          columns, the first available geometry column will be set as the geometry
          column of the returned Spatially Enabled DataFrame.

        Requires 'pyarrow'.

        .. versionadded:: arcgis 1.9

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        path                   Required String. path object
        ------------------     --------------------------------------------------------------------
        columns                Optional List[str]. The defaulti s `None`. If not None, only these
                               columns will be read from the file.  If the primary geometry column
                               is not included, the first secondary geometry read from the file will
                               be set as the geometry column of the returned Spatially Enabled
                               DataFrame.  If no geometry columns are present, a ``ValueError``
                               will be raised.
        ------------------     --------------------------------------------------------------------
        **kwargs**             Optional dict. Any additional kwargs that can be given to the
                               `pyarrow.parquet.read_table <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html#pyarrow-parquet-read-table>`_ method.
        ==================     ====================================================================



        :returns: Spatially Enabled DataFrame

        Examples
        --------
        >>> df = pd.DataFrame.spatial.from_parquet("data.parquet")  # doctest: +SKIP

        Specifying columns to read:

        >>> df = pd.DataFrame.spatial.from_parquet(
        ...     "data.parquet",
        ...     columns=["SHAPE", "pop_est"]
        ... )  # doctest: +SKIP
        """
        from ._io._arrow import _read_parquet

        return _read_parquet(path=path, columns=columns, **kwargs)

    @staticmethod
    def from_feather(path, spatial_column="SHAPE", columns=None, use_threads=True):
        """
        The ``from-feather`` method loads a feather-format object from the file path.

        ======================    =========================================================
        **Parameter**              **Description**
        ----------------------    ---------------------------------------------------------
        path                      String. Path object or file-like object. Any valid string
                                  path is acceptable. The string could be a URL. Valid
                                  URL schemes include http, ftp, s3, and file. For file URLs, a host is
                                  expected. A local file could be:

                                  ``file://localhost/path/to/table.feather``.

                                  If you want to pass in a path object, pandas accepts any
                                  ``os.PathLike``.

                                  By file-like object, we refer to objects with a ``read()`` method,
                                  such as a file handler (e.g. via builtin ``open`` function)
                                  or ``StringIO``.
        ----------------------    ---------------------------------------------------------
        spatial_column            Optional String. The default is `SHAPE`. Specifies the column
                                  containing the geo-spatial information.
        ----------------------    ---------------------------------------------------------
        columns                   Sequence/List/Array. The default is `None`.  If not
                                  provided, all columns are read.
        ----------------------    ---------------------------------------------------------
        use_threads               Boolean. The default is `True`. Whether to parallelize
                                  reading using multiple threads.
        ======================    =========================================================

        :return:
            A Pandas DataFrame (pd.DataFrame)

        """
        return read_feather(
            path=path,
            spatial_column=spatial_column,
            columns=columns,
            use_threads=use_threads,
        )

    # ----------------------------------------------------------------------
    def set_geometry(self, col, sr=None, inplace=True):
        """
        The ``set_geometry`` method assigns the geometry column by name or by list.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        col                    Required string, Pandas Series, GeoArray, list or tuple. If a string, this
                               is the name of the column containing the geometry. If a Pandas Series
                               GeoArray, list or tuple, it is an iterable of Geometry objects.
        ------------------     --------------------------------------------------------------------
        sr                     Optional integer or spatial reference of the geometries described in
                               the first parameter. If the geometry objects already have the spatial
                               reference defined, this is not necessary. If the spatial reference for
                               the geometry objects is NOT define, it will default to WGS84 (wkid 4326).
        ------------------     --------------------------------------------------------------------
        inplace                Optional bool. Whether or not to modify the dataframe in place, or return
                               a new dataframe. If True, nothing is returned and the dataframe is modified
                               in place. If False, a new dataframe is returned with the geometry set.
                               Defaults to True.
        ==================     ====================================================================

        :return:
            Spatially Enabled DataFrame or None
        """
        from ._array import GeoArray

        if (
            isinstance(col, str)
            and col in self._data.columns
            and self._data[col].dtype.name.lower() != "geometry"
        ):
            idx = self._data[col].first_valid_index()
            if sr is None:
                try:
                    g = self._data.iloc[idx][col]
                    if isinstance(g, dict):
                        self._sr = _geometry.SpatialReference(
                            _geometry.Geometry(g["spatialReference"])
                        )
                    else:
                        self._sr = _geometry.SpatialReference(g["spatialReference"])
                except:
                    self._sr = _geometry.SpatialReference({"wkid": 4326})
            self._name = col
            # q = self._data[col].isna()
            # self._data.loc[q, "SHAPE"] = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._data[col] = GeoArray(self._data[col])
        elif (
            isinstance(col, str)
            and col in self._data.columns
            and self._data[col].dtype.name.lower() == "geometry"
        ):
            self._name = col
            # self._data[col] = self._data[col]
        elif isinstance(col, str) and col not in self._data.columns:
            raise ValueError("Column {name} does not exist".format(name=col))
        elif isinstance(col, pd.Series):
            self._data["SHAPE"] = GeoArray(col.values)
            self._name = "SHAPE"
        elif isinstance(col, GeoArray):
            self._data["SHAPE"] = col
            self._name = "SHAPE"
        elif isinstance(col, (list, tuple)):
            self._data["SHAPE"] = GeoArray(values=col)
            self._name = "SHAPE"
        else:
            raise ValueError(
                "Column {name} is not valid. Please ensure it is of type Geometry".format(
                    name=col
                )
            )

        if not inplace:
            return self._data.copy()

    # ----------------------------------------------------------------------
    @property
    def name(self):
        """
        The ``name`` method retrieves the name of the geometry column.

        :return:
            A string
        """
        if self._name is None:
            try:
                if any(self._data.dtypes == "geometry"):
                    name = self._data.dtypes[self._data.dtypes == "geometry"].index[0]
                    self.set_geometry(name)
                elif "shape" in [str(c).lower() for c in self._data.columns.tolist()]:
                    cols = [str(c).lower() for c in self._data.columns.tolist()]
                    idx = cols.index("shape")
                    self.set_geometry(self._data.columns[idx])
            except:
                raise Exception("Spatial column not defined, please use `set_geometry`")
        return self._name

    # ----------------------------------------------------------------------
    def validate(self, strict=False):
        """
        The ``validate`` method determines if the `GeoAccessor` is Valid with
        :class:`~arcgis.geometry.Geometry` objects in all values

        :return:
            A boolean indicating Success (True), or Failure (False)
        """
        if self._name is None:
            return False
        if strict:
            q = self._data[self.name].notna()
            gt = pd.unique(self._data[q][self.name].geom.geometry_type)
            if len(gt) == 1:
                return True
            else:
                return False
        else:
            q = self._data[self.name].notna()
            return all(pd.unique(self._data[q][self.name].geom.is_valid))
        return True

    # ----------------------------------------------------------------------
    def join(
        self,
        right_df,
        how="inner",
        op="intersects",
        left_tag="left",
        right_tag="right",
    ):
        """
        The ``join`` method joins the current DataFrame to another Spatially-Enabled DataFrame based
        on spatial location based.

        .. note::
            The ``join`` method requires the Spatially-Enabled DataFrame to be in the same coordinate system


        ======================    =========================================================
        **Parameter**              **Description**
        ----------------------    ---------------------------------------------------------
        right_df                  Required pd.DataFrame. Spatially enabled dataframe to join.
        ----------------------    ---------------------------------------------------------
        how                       Required string. The type of join:

                                    + `left` - use keys from current dataframe and retains only current geometry column
                                    + `right` - use keys from right_df; retain only right_df geometry column
                                    + `inner` - use intersection of keys from both dfs and retain only current geometry column

        ----------------------    ---------------------------------------------------------
        op                        Required string. The operation to use to perform the join.
                                  The default is `intersects`.

                                  supported perations: `intersects`, `within`, and `contains`
        ----------------------    ---------------------------------------------------------
        left_tag                  Optional String. If the same column is in the left and
                                  right dataframe, this will append that string value to
                                  the field.
        ----------------------    ---------------------------------------------------------
        right_tag                 Optional String. If the same column is in the left and
                                  right dataframe, this will append that string value to
                                  the field.
        ======================    =========================================================

        :return:
          Spatially enabled Pandas' DataFrame
        """
        allowed_hows = ["left", "right", "inner"]
        allowed_ops = ["contains", "within", "intersects"]
        if how not in allowed_hows:
            raise ValueError(
                "`how` is an invalid inputs of %s, but should be %s"
                % (how, allowed_hows)
            )
        if op not in allowed_ops:
            raise ValueError(
                "`how` is an invalid inputs of %s, but should be %s" % (op, allowed_ops)
            )
        same_sr = False
        if self.sr == right_df.spatial.sr:
            same_sr = True
        else:
            # check for cases where there is latestWkid by iterating through values of sr
            for value in self.sr.values():
                if value in right_df.spatial.sr.values():
                    same_sr = True
        if same_sr is False:
            raise Exception("Difference Spatial References, aborting operation")
        index_left = "index_{}".format(left_tag)
        index_right = "index_{}".format(right_tag)
        if any(self._data.columns.isin([index_left, index_right])) or any(
            right_df.columns.isin([index_left, index_right])
        ):
            raise ValueError(
                "'{0}' and '{1}' cannot be names in the frames being"
                " joined".format(index_left, index_right)
            )
        # Setup the Indexes in temporary coumns
        #
        left_df = self._data.copy(deep=True)
        left_df.spatial.set_geometry(self.name)
        left_df.reset_index(inplace=True)
        left_df.spatial.set_geometry(self.name)
        # process the right df
        shape_right = right_df.spatial._name
        right_df = right_df.copy(deep=True)
        right_df.reset_index(inplace=True)
        right_df.spatial.set_geometry(shape_right)
        # rename the indexes
        right_df.index = right_df.index.rename(index_right)
        left_df.index = left_df.index.rename(index_left)

        if op == "within":
            # within implemented as the inverse of contains; swap names
            left_df, right_df = right_df, left_df

        tree_idx = right_df.spatial.sindex("quadtree")

        idxmatch = (
            left_df[self.name]
            .apply(lambda x: x.extent)
            .apply(lambda x: list(tree_idx.intersect(x)))
        )
        idxmatch = idxmatch[idxmatch.apply(len) > 0]
        if idxmatch.shape[0] > 0:
            # if output from join has overlapping geometries
            r_idx = np.concatenate(idxmatch.values)
            l_idx = np.concatenate([[i] * len(v) for i, v in idxmatch.items()])

            # Vectorize predicate operations
            def find_intersects(a1, a2):
                return a1.disjoint(a2) == False

            def find_contains(a1, a2):
                return a1.contains(a2)

            predicate_d = {
                "intersects": find_intersects,
                "contains": find_contains,
                "within": find_contains,
            }

            check_predicates = np.vectorize(predicate_d[op])

            result = pd.DataFrame(
                np.column_stack(
                    [
                        l_idx,
                        r_idx,
                        check_predicates(
                            left_df[self.name].apply(lambda x: x)[l_idx],
                            right_df[right_df.spatial._name][r_idx],
                        ),
                    ]
                )
            )

            result.columns = ["_key_left", "_key_right", "match_bool"]
            result = pd.DataFrame(result[result["match_bool"] == 1]).drop(
                "match_bool", axis=1
            )
        else:
            # when output from the join has no overlapping geometries
            result = pd.DataFrame(columns=["_key_left", "_key_right"], dtype=float)
        if op == "within":
            # within implemented as the inverse of contains; swap names
            left_df, right_df = right_df, left_df
            result = result.rename(
                columns={
                    "_key_left": "_key_right",
                    "_key_right": "_key_left",
                }
            )

        if how == "inner":
            result = result.set_index("_key_left")
            joined = left_df.merge(result, left_index=True, right_index=True).merge(
                right_df.drop(right_df.spatial.name, axis=1),
                left_on="_key_right",
                right_index=True,
                suffixes=("_%s" % left_tag, "_%s" % right_tag),
            )
            joined = joined.set_index(index_left).drop(["_key_right"], axis=1)
            joined.index.name = None
        elif how == "left":
            result = result.set_index("_key_left")
            joined = left_df.merge(
                result, left_index=True, right_index=True, how="left"
            ).merge(
                right_df.drop(right_df.spatial.name, axis=1),
                how="left",
                left_on="_key_right",
                right_index=True,
                suffixes=("_%s" % left_tag, "_%s" % right_tag),
            )
            joined = joined.set_index(index_left).drop(["_key_right"], axis=1)
            joined.index.name = None
        else:  # 'right join'
            joined = (
                left_df.drop(left_df.spatial._name, axis=1)
                .merge(
                    result.merge(
                        right_df,
                        left_on="_key_right",
                        right_index=True,
                        how="right",
                    ),
                    left_index=True,
                    right_on="_key_left",
                    how="right",
                )
                .set_index("index_y")
            )
            joined = joined.drop(["_key_left", "_key_right"], axis=1)
        try:
            joined.spatial.set_geometry(self.name)
        except:
            raise Exception("Could not create spatially enabled dataframe.")
        joined.reset_index(drop=True, inplace=True)
        return joined

    # ----------------------------------------------------------------------
    def plot(self, map_widget=None, **kwargs):
        """

        The ``plot`` draws the data on a web map. The user can describe in simple terms how to
        renderer spatial data using symbol.

        .. note::
            To make the process simpler, a palette
            for which colors are drawn from can be used instead of explicit colors.


        ======================  =========================================================
        **Explicit Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        map_widget              optional ``WebMap`` object. This is the map to display
                                the data on.
        ----------------------  ---------------------------------------------------------
        palette                 optional string/dict. Color mapping. Can also be listed
                                as 'colors' or 'cmap'. For a simple renderer, just
                                provide the string name of a colormap or a RGB + alpha
                                int array. For a unique renderer, a list of colormaps can
                                be provided. For heatmaps, a list of 3+ specific
                                colorstops can be provided in the form of an array of RGB
                                + alpha values or a list of colormaps, or the name of a
                                single colormap can be provided.

                                Accepts palettes exported from colorbrewer or imported
                                from palettable as well. To get a list of built-in
                                palettes, use the **display_colormaps** method.
        ----------------------  ---------------------------------------------------------
        renderer_type           optional string.  Determines the type of renderer to use
                                for the provided dataset. The default is 's' which is for
                                simple renderers.

                                Allowed values:

                                + 's' - is a simple renderer that uses one symbol only.
                                + 'u' - unique renderer symbolizes features based on one
                                        or more matching string attributes.
                                + 'c' - A class breaks renderer symbolizes based on the
                                        value of some numeric attribute.
                                + 'h' - heatmap renders point data into a raster
                                        visualization that emphasizes areas of higher
                                        density or weighted values.
        ----------------------  ---------------------------------------------------------
        symbol_type             optional string. This is the type of symbol the user
                                needs to create.  Valid inputs are: simple, picture,
                                text, or carto.  The default is simple.
        ----------------------  ---------------------------------------------------------
        symbol_style            optional string. This is the symbology used by the
                                geometry.  For example 's' for a Line geometry is a solid
                                line. And '-' is a dash line.

                                Allowed symbol types based on geometries:

                                **Point Symbols**

                                 + 'o' - Circle (default)
                                 + '+' - Cross
                                 + 'D' - Diamond
                                 + 's' - Square
                                 + 'x' - X

                                 **Polyline Symbols**

                                 + 's' - Solid (default)
                                 + '-' - Dash
                                 + '-.' - Dash Dot
                                 + '-..' - Dash Dot Dot
                                 + '.' - Dot
                                 + '--' - Long Dash
                                 + '--.' - Long Dash Dot
                                 + 'n' - Null
                                 + 's-' - Short Dash
                                 + 's-.' - Short Dash Dot
                                 + 's-..' - Short Dash Dot Dot
                                 + 's.' - Short Dot

                                 **Polygon Symbols**

                                 + 's' - Solid Fill (default)
                                 + '\' - Backward Diagonal
                                 + '/' - Forward Diagonal
                                 + '|' - Vertical Bar
                                 + '-' - Horizontal Bar
                                 + 'x' - Diagonal Cross
                                 + '+' - Cross

        ----------------------  ---------------------------------------------------------
        col                     optional string/list. Field or fields used for heatmap,
                                class breaks, or unique renderers.
        ----------------------  ---------------------------------------------------------
        alpha                   optional float.  This is a value between 0 and 1 with 1
                                being the default value.  The alpha sets the transparency
                                of the renderer when applicable.
        ======================  =========================================================

        **Render Syntax**

        The render syntax allows for users to fully customize symbolizing the data.

        **Simple Renderer**

        A simple renderer is a renderer that uses one symbol only.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        symbol_type             optional string. This is the type of symbol the user
                                needs to create.  Valid inputs are: simple, picture, text,
                                or carto.  The default is simple.
        ----------------------  ---------------------------------------------------------
        symbol_style            optional string. This is the symbology used by the
                                geometry.  For example 's' for a Line geometry is a solid
                                line. And '-' is a dash line.

                                **Point Symbols**

                                + 'o' - Circle (default)
                                + '+' - Cross
                                + 'D' - Diamond
                                + 's' - Square
                                + 'x' - X

                                **Polyline Symbols**

                                + 's' - Solid (default)
                                + '-' - Dash
                                + '-.' - Dash Dot
                                + '-..' - Dash Dot Dot
                                + '.' - Dot
                                + '--' - Long Dash
                                + '--.' - Long Dash Dot
                                + 'n' - Null
                                + 's-' - Short Dash
                                + 's-.' - Short Dash Dot
                                + 's-..' - Short Dash Dot Dot
                                + 's.' - Short Dot

                                **Polygon Symbols**

                                + 's' - Solid Fill (default)
                                + '\' - Backward Diagonal
                                + '/' - Forward Diagonal
                                + '|' - Vertical Bar
                                + '-' - Horizontal Bar
                                + 'x' - Diagonal Cross
                                + '+' - Cross
        ----------------------  ---------------------------------------------------------
        description             Description of the renderer.
        ----------------------  ---------------------------------------------------------
        rotation_expression     A constant value or an expression that derives the angle
                                of rotation based on a feature attribute value. When an
                                attribute name is specified, it's enclosed in square
                                brackets.
        ----------------------  ---------------------------------------------------------
        rotation_type           String value which controls the origin and direction of
                                rotation on point features. If the rotationType is
                                defined as arithmetic, the symbol is rotated from East in
                                a counter-clockwise direction where East is the 0 degree
                                axis. If the rotationType is defined as geographic, the
                                symbol is rotated from North in a clockwise direction
                                where North is the 0 degree axis.

                                Must be one of the following values:

                                + arithmetic
                                + geographic

        ----------------------  ---------------------------------------------------------
        visual_variables        An array of objects used to set rendering properties.
        ======================  =========================================================

        **Heatmap Renderer**

        The HeatmapRenderer renders point data into a raster visualization that emphasizes
        areas of higher density or weighted values.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        blur_radius             The radius (in pixels) of the circle over which the
                                majority of each point's value is spread.
        ----------------------  ---------------------------------------------------------
        field                   This is optional as this renderer can be created if no
                                field is specified. Each feature gets the same
                                value/importance/weight or with a field where each
                                feature is weighted by the field's value.
        ----------------------  ---------------------------------------------------------
        max_intensity           The pixel intensity value which is assigned the final
                                color in the color ramp.
        ----------------------  ---------------------------------------------------------
        min_intensity           The pixel intensity value which is assigned the initial
                                color in the color ramp.
        ----------------------  ---------------------------------------------------------
        ratio                   A number between 0-1. Describes what portion along the
                                gradient the colorStop is added.
        ----------------------  ---------------------------------------------------------
        show_none               Boolean. Determines the alpha value of the base color for
                                the heatmap. Setting this to ``True`` covers an entire
                                map with the base color of the heatmap. Default is
                                ``False``.
        ======================  =========================================================

        **Unique Renderer**

        This renderer symbolizes features based on one or more matching string attributes.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        background_fill_symbol  A symbol used for polygon features as a background if the
                                renderer uses point symbols, e.g. for bivariate types &
                                size rendering. Only applicable to polygon layers.
                                PictureFillSymbols can also be used outside of the Map
                                Viewer for Size and Predominance and Size renderers.
        ----------------------  ---------------------------------------------------------
        default_label           Default label for the default symbol used to draw
                                unspecified values.
        ----------------------  ---------------------------------------------------------
        default_symbol          Symbol used when a value cannot be matched.
        ----------------------  ---------------------------------------------------------
        field1, field2, field3  Attribute field renderer uses to match values.
        ----------------------  ---------------------------------------------------------
        field_delimiter         String inserted between the values if multiple attribute
                                fields are specified.
        ----------------------  ---------------------------------------------------------
        rotation_expression     A constant value or an expression that derives the angle
                                of rotation based on a feature attribute value. When an
                                attribute name is specified, it's enclosed in square
                                brackets. Rotation is set using a visual variable of type
                                rotation info with a specified field or value expression
                                property.
        ----------------------  ---------------------------------------------------------
        rotation_type           String property which controls the origin and direction
                                of rotation. If the rotation type is defined as
                                arithmetic the symbol is rotated from East in a
                                counter-clockwise direction where East is the 0 degree
                                axis. If the rotation type is defined as geographic, the
                                symbol is rotated from North in a clockwise direction
                                where North is the 0 degree axis.
                                Must be one of the following values:

                                + arithmetic
                                + geographic

        ----------------------  ---------------------------------------------------------
        arcade_expression       An Arcade expression evaluating to either a string or a
                                number.
        ----------------------  ---------------------------------------------------------
        arcade_title            The title identifying and describing the associated
                                Arcade expression as defined in the valueExpression
                                property.
        ----------------------  ---------------------------------------------------------
        visual_variables        An array of objects used to set rendering properties.
        ======================  =========================================================

        **Class Breaks Renderer**

        A class breaks renderer symbolizes based on the value of some numeric attribute.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        background_fill_symbol  A symbol used for polygon features as a background if the
                                renderer uses point symbols, e.g. for bivariate types &
                                size rendering. Only applicable to polygon layers.
                                PictureFillSymbols can also be used outside of the Map
                                Viewer for Size and Predominance and Size renderers.
        ----------------------  ---------------------------------------------------------
        default_label           Default label for the default symbol used to draw
                                unspecified values.
        ----------------------  ---------------------------------------------------------
        default_symbol          Symbol used when a value cannot be matched.
        ----------------------  ---------------------------------------------------------
        method                  Determines the classification method that was used to
                                generate class breaks.

                                Must be one of the following values:

                                + esriClassifyDefinedInterval
                                + esriClassifyEqualInterval
                                + esriClassifyGeometricalInterval
                                + esriClassifyNaturalBreaks
                                + esriClassifyQuantile
                                + esriClassifyStandardDeviation
                                + esriClassifyManual

        ----------------------  ---------------------------------------------------------
        field                   Attribute field used for renderer.
        ----------------------  ---------------------------------------------------------
        class_count             Number of classes that will be considered in the
                                selected classification method for the class breaks.
        ----------------------  ---------------------------------------------------------
        min_value               The minimum numeric data value needed to begin class
                                breaks.
        ----------------------  ---------------------------------------------------------
        normalization_field     Used when normalizationType is field. The string value
                                indicating the attribute field by which the data value is
                                normalized.
        ----------------------  ---------------------------------------------------------
        normalization_total     Used when normalizationType is percent-of-total, this
                                number property contains the total of all data values.
        ----------------------  ---------------------------------------------------------
        normalization_type      Determine how the data was normalized.

                                Must be one of the following values:

                                + esriNormalizeByField
                                + esriNormalizeByLog
                                + esriNormalizeByPercentOfTotal
        ----------------------  ---------------------------------------------------------
        rotation_expression     A constant value or an expression that derives the angle
                                of rotation based on a feature attribute value. When an
                                attribute name is specified, it's enclosed in square
                                brackets.
        ----------------------  ---------------------------------------------------------
        rotation_type           A string property which controls the origin and direction
                                of rotation. If the rotation_type is defined as
                                arithmetic, the symbol is rotated from East in a
                                couter-clockwise direction where East is the 0 degree
                                axis. If the rotationType is defined as geographic, the
                                symbol is rotated from North in a clockwise direction
                                where North is the 0 degree axis.

                                Must be one of the following values:

                                + arithmetic
                                + geographic

        ----------------------  ---------------------------------------------------------
        arcade_expression       An Arcade expression evaluating to a number.
        ----------------------  ---------------------------------------------------------
        arcade_title            The title identifying and describing the associated
                                Arcade expression as defined in the arcade_expression
                                property.
        ----------------------  ---------------------------------------------------------
        visual_variables        An object used to set rendering options.
        ======================  =========================================================



        ** Symbol Syntax **

        =======================  =========================================================
        **Optional Argument**    **Description**
        -----------------------  ---------------------------------------------------------
        symbol_type              optional string. This is the type of symbol the user
                                 needs to create.  Valid inputs are: simple, picture, text,
                                 or carto.  The default is simple.
        -----------------------  ---------------------------------------------------------
        symbol_style             optional string. This is the symbology used by the
                                 geometry.  For example 's' for a Line geometry is a solid
                                 line. And '-' is a dash line.

                                 **Point Symbols**

                                 + 'o' - Circle (default)
                                 + '+' - Cross
                                 + 'D' - Diamond
                                 + 's' - Square
                                 + 'x' - X

                                 **Polyline Symbols**

                                 + 's' - Solid (default)
                                 + '-' - Dash
                                 + '-.' - Dash Dot
                                 + '-..' - Dash Dot Dot
                                 + '.' - Dot
                                 + '--' - Long Dash
                                 + '--.' - Long Dash Dot
                                 + 'n' - Null
                                 + 's-' - Short Dash
                                 + 's-.' - Short Dash Dot
                                 + 's-..' - Short Dash Dot Dot
                                 + 's.' - Short Dot

                                 **Polygon Symbols**

                                 + 's' - Solid Fill (default)
                                 + '\' - Backward Diagonal
                                 + '/' - Forward Diagonal
                                 + '|' - Vertical Bar
                                 + '-' - Horizontal Bar
                                 + 'x' - Diagonal Cross
                                 + '+' - Cross
        -----------------------  ---------------------------------------------------------
        cmap                     optional string or list.  This is the color scheme a user
                                 can provide if the exact color is not needed, or a user
                                 can provide a list with the color defined as:
                                 [red, green blue, alpha]. The values red, green, blue are
                                 from 0-255 and alpha is a float value from 0 - 1.
                                 The default value is 'jet' color scheme.
        -----------------------  ---------------------------------------------------------
        cstep                    optional integer.  If provided, its the color location on
                                 the color scheme.
        =======================  =========================================================

        **Simple Symbols**

        This is a list of optional parameters that can be given for point, line or
        polygon geometries.

        ====================  =========================================================
        **Parameter**          **Description**
        --------------------  ---------------------------------------------------------
        marker_size           optional float.  Numeric size of the symbol given in
                              points.
        --------------------  ---------------------------------------------------------
        marker_angle          optional float. Numeric value used to rotate the symbol.
                              The symbol is rotated counter-clockwise. For example,
                              The following, angle=-30, in will create a symbol rotated
                              -30 degrees counter-clockwise; that is, 30 degrees
                              clockwise.
        --------------------  ---------------------------------------------------------
        marker_xoffset        Numeric value indicating the offset on the x-axis in points.
        --------------------  ---------------------------------------------------------
        marker_yoffset        Numeric value indicating the offset on the y-axis in points.
        --------------------  ---------------------------------------------------------
        line_width            optional float. Numeric value indicating the width of the line in points
        --------------------  ---------------------------------------------------------
        outline_style         Optional string. For polygon point, and line geometries , a
                              customized outline type can be provided.

                              Allowed Styles:

                              + 's' - Solid (default)
                              + '-' - Dash
                              + '-.' - Dash Dot
                              + '-..' - Dash Dot Dot
                              + '.' - Dot
                              + '--' - Long Dash
                              + '--.' - Long Dash Dot
                              + 'n' - Null
                              + 's-' - Short Dash
                              + 's-.' - Short Dash Dot
                              + 's-..' - Short Dash Dot Dot
                              + 's.' - Short Dot
        --------------------  ---------------------------------------------------------
        outline_color         optional string or list.  This is the same color as the
                              cmap property, but specifically applies to the outline_color.
        ====================  =========================================================

        **Picture Symbol**

        This type of symbol only applies to Points, MultiPoints and Polygons.

        ====================  =========================================================
        **Parameter**          **Description**
        --------------------  ---------------------------------------------------------
        marker_angle          Numeric value that defines the number of degrees ranging
                              from 0-360, that a marker symbol is rotated. The rotation
                              is from East in a counter-clockwise direction where East
                              is the 0 axis.
        --------------------  ---------------------------------------------------------
        marker_xoffset        Numeric value indicating the offset on the x-axis in points.
        --------------------  ---------------------------------------------------------
        marker_yoffset        Numeric value indicating the offset on the y-axis in points.
        --------------------  ---------------------------------------------------------
        height                Numeric value used if needing to resize the symbol. Specify a value in points. If images are to be displayed in their original size, leave this blank.
        --------------------  ---------------------------------------------------------
        width                 Numeric value used if needing to resize the symbol. Specify a value in points. If images are to be displayed in their original size, leave this blank.
        --------------------  ---------------------------------------------------------
        url                   String value indicating the URL of the image. The URL should be relative if working with static layers. A full URL should be used for map service dynamic layers. A relative URL can be dereferenced by accessing the map layer image resource or the feature layer image resource.
        --------------------  ---------------------------------------------------------
        image_data            String value indicating the base64 encoded data.
        --------------------  ---------------------------------------------------------
        xscale                Numeric value indicating the scale factor in x direction.
        --------------------  ---------------------------------------------------------
        yscale                Numeric value indicating the scale factor in y direction.
        --------------------  ---------------------------------------------------------
        outline_color         optional string or list.  This is the same color as the
                              cmap property, but specifically applies to the outline_color.
        --------------------  ---------------------------------------------------------
        outline_style         Optional string. For polygon point, and line geometries , a
                              customized outline type can be provided.

                              Allowed Styles:

                              + 's' - Solid (default)
                              + '-' - Dash
                              + '-.' - Dash Dot
                              + '-..' - Dash Dot Dot
                              + '.' - Dot
                              + '--' - Long Dash
                              + '--.' - Long Dash Dot
                              + 'n' - Null
                              + 's-' - Short Dash
                              + 's-.' - Short Dash Dot
                              + 's-..' - Short Dash Dot Dot
                              + 's.' - Short Dot
        --------------------  ---------------------------------------------------------
        outline_color         optional string or list.  This is the same color as the
                              cmap property, but specifically applies to the outline_color.
        --------------------  ---------------------------------------------------------
        line_width            optional float. Numeric value indicating the width of the line in points
        ====================  =========================================================

        **Text Symbol**

        This type of symbol only applies to Points, MultiPoints and Polygons.

        ====================  =========================================================
        **Parameter**          **Description**
        --------------------  ---------------------------------------------------------
        font_decoration       The text decoration. Must be one of the following values:
                              - line-through
                              - underline
                              - none
        --------------------  ---------------------------------------------------------
        font_family           Optional string. The font family.
        --------------------  ---------------------------------------------------------
        font_size             Optional float. The font size in points.
        --------------------  ---------------------------------------------------------
        font_style            Optional string. The text style.
                              - italic
                              - normal
                              - oblique
        --------------------  ---------------------------------------------------------
        font_weight           Optional string. The text weight.
                              Must be one of the following values:
                              - bold
                              - bolder
                              - lighter
                              - normal
        --------------------  ---------------------------------------------------------
        background_color      optional string/list. Background color is represented as
                              a four-element array or string of a color map.
        --------------------  ---------------------------------------------------------
        halo_color            Optional string/list. Color of the halo around the text.
                              The default is None.
        --------------------  ---------------------------------------------------------
        halo_size             Optional integer/float. The point size of a halo around
                              the text symbol.
        --------------------  ---------------------------------------------------------
        horizontal_alignment  optional string. One of the following string values
                              representing the horizontal alignment of the text.
                              Must be one of the following values:
                              - left
                              - right
                              - center
                              - justify
        --------------------  ---------------------------------------------------------
        kerning               optional boolean. Boolean value indicating whether to
                              adjust the spacing between characters in the text string.
        --------------------  ---------------------------------------------------------
        line_color            optional string/list. Outline color is represented as
                              a four-element array or string of a color map.
        --------------------  ---------------------------------------------------------
        line_width            optional integer/float. Outline size.
        --------------------  ---------------------------------------------------------
        marker_angle          optional int. A numeric value that defines the number of
                              degrees (0 to 360) that a text symbol is rotated. The
                              rotation is from East in a counter-clockwise direction
                              where East is the 0 axis.
        --------------------  ---------------------------------------------------------
        marker_xoffset        optional int/float.Numeric value indicating the offset
                              on the x-axis in points.
        --------------------  ---------------------------------------------------------
        marker_yoffset        optional int/float.Numeric value indicating the offset
                              on the x-axis in points.
        --------------------  ---------------------------------------------------------
        right_to_left         optional boolean. Set to true if using Hebrew or Arabic
                              fonts.
        --------------------  ---------------------------------------------------------
        rotated               optional boolean. Boolean value indicating whether every
                              character in the text string is rotated.
        --------------------  ---------------------------------------------------------
        text                  Required string.  Text Value to display next to geometry.
        --------------------  ---------------------------------------------------------
        vertical_alignment    Optional string. One of the following string values
                              representing the vertical alignment of the text.
                              Must be one of the following values:
                              - top
                              - bottom
                              - middle
                              - baseline
        ====================  =========================================================

        **Cartographic Symbol**

        This type of symbol only applies to line geometries.

        ====================  =========================================================
        **Parameter**          **Description**
        --------------------  ---------------------------------------------------------
        line_width            optional float. Numeric value indicating the width of the line in points
        --------------------  ---------------------------------------------------------
        cap                   Optional string.  The cap style.
        --------------------  ---------------------------------------------------------
        join                  Optional string. The join style.
        --------------------  ---------------------------------------------------------
        miter_limit           Optional string. Size threshold for showing mitered line joins.
        ====================  =========================================================

        The kwargs parameter accepts all parameters of the create_symbol method and the
        create_renderer method.

        :return:
            A ``MapView`` object with new drawings

        """
        from ._viz.mapping import plot

        # small helper to consolidate the plotting function
        def _plot_map_widget(mp_wdgt):
            plot(
                df=self._data,
                map_widget=mp_wdgt,
                name=kwargs.pop("name", "Feature Collection Layer"),
                renderer_type=kwargs.pop("renderer_type", None),
                symbol_type=kwargs.pop("symbol_type", None),
                symbol_style=kwargs.pop("symbol_style", None),
                col=kwargs.pop("col", None),
                colors=kwargs.pop("cmap", None)
                or kwargs.pop("colors", None)
                or kwargs.pop("pallette", None)
                or kwargs.pop("palette", "jet"),
                alpha=kwargs.pop("alpha", 1),
                **kwargs,
            )

        # small helper to address zoom level
        def _adjust_zoom(mp_wdgt):
            # if a single point, the extent will zoom to a scale so large it is almost irrelevant, so back out slightly
            if mp_wdgt.zoom > 16:
                mp_wdgt.zoom = 16

            # if zooming to an extent, it will zoom one level too far, so back out one to make all data visible
            else:
                mp_wdgt.zoom = mp_wdgt.zoom - 1

        # if the map widget is explicitly defined
        if map_widget:
            orig_col = copy.deepcopy(self._data.columns)
            self._data.columns = [c.replace(" ", "_") for c in self._data.columns]
            # plot and be merry
            _plot_map_widget(map_widget)
            self._data.columns = orig_col
            return True

        # otherwise, if a map widget is NOT explicitly defined
        else:
            from arcgis.gis import GIS
            from arcgis.env import active_gis

            # if a gis is not already created in the session, create an anonymous one
            gis = active_gis
            if gis is None:
                gis = GIS()

            # use the GIS to create a map widget
            map_widget = gis.map()

            # plot the data in the map widget
            orig_col = copy.deepcopy(self._data.columns)
            self._data.columns = [c.replace(" ", "_") for c in self._data.columns]
            _plot_map_widget(map_widget)
            self._data.columns = orig_col
            # zoom the map widget to the extent of the data
            map_widget.extent = {
                "spatialReference": self._data.spatial.sr,
                "xmin": self._data.spatial.full_extent[0],
                "ymin": self._data.spatial.full_extent[1],
                "xmax": self._data.spatial.full_extent[2],
                "ymax": self._data.spatial.full_extent[3],
            }

            # adjust the zoom level so the map displays the data as expected
            map_widget.on_draw_end(_adjust_zoom, True)

            # return the map widget so it will be displayed below the cell in Jupyter Notebook
            return map_widget

    # ----------------------------------------------------------------------
    def insert_layer(
        self,
        feature_service: _gis.Item | str,
        gis=None,
        sanitize_columns: bool = False,
        service_name: str = None,
    ):
        """
        This method creates a feature layer from the spatially enabled dataframe and adds (inserts)
        it to an existing feature service.

        .. note::
            Inserting table data in Enterprise is not currently supported.

        ============================    ====================================================================
        **Parameter**                   **Description**
        ----------------------------    --------------------------------------------------------------------
        feature_service                 Required :class:`~arcgis.gis.Item` or Feature Service Id. Depicts
                                        the feature service to which the layer will be added.
        ----------------------------    --------------------------------------------------------------------
        gis                             Optional :class:`~arcgis.gis.GIS`. The GIS object.
        ----------------------------    --------------------------------------------------------------------
        sanitize_columns                Optional Boolean. If ``True``, column names will be converted to
                                        string, invalid characters removed and other performed. The
                                        default is ``False``.
        ----------------------------    --------------------------------------------------------------------
        service_name                    Optional String. The name for the service that will be added to the
                                        :class:`~arcgis.gis.Item` The name cannot be used already or contain
                                        special characters, spaces, or a number as the first character.
        ============================    ====================================================================

        :return: The feature service item that was appended to.
        """
        from arcgis import env
        import copy

        if gis is None:
            gis = env.active_gis
            if gis is None:
                raise ValueError("GIS object must be provided")
        content = gis.content

        # Check that the user is the owner of both the source and the published item
        user = gis._username
        if isinstance(feature_service, str):
            service = content.get(feature_service)
            fs_id = feature_service
        else:
            service = feature_service
            fs_id = feature_service.id

        if (
            gis.users.me.username != service.owner
            and "portal:admin:updateItems" not in self._gis.users.me.privileges
        ):
            raise AssertionError(
                "You must own the service to insert data to it or have administrative privileges."
            )
        # Get the data related
        related_items = service.related_items(rel_type="Service2Data")
        for item in related_items:
            if (
                item.owner != user
                and "portal:admin:updateItems" not in self._gis.users.me.privileges
            ):
                raise AssertionError(
                    "You must own the service data to insert data to it or have administrative privileges."
                )

        origin_columns = self._data.columns.tolist()
        origin_index = copy.deepcopy(self._data.index)

        if service_name:
            # sanitize name
            service_name = service_name.replace(" ", "")
            if service_name[0].isnumeric():
                raise ValueError(
                    "First character of service_name cannot be an integer."
                )
            if (
                content.is_service_name_available(service_name, "featureService")
                is False
            ):
                raise ValueError(
                    "This service name is unavailable for Feature Service."
                )
        result = content.import_data(
            self._data,
            sanitize_columns=sanitize_columns,
            service_name=service_name,
            append=True,
            service={"featureServiceId": fs_id, "layer": None},
        )
        self._data.columns = origin_columns
        self._data.index = origin_index
        return result

    # ----------------------------------------------------------------------
    def to_arrow(self, index: bool = None) -> "pyarrow.Table":
        """
        Converts a Pandas DatFrame to an Arrow Table

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        index                  Optional Bool. If ``True``, always include the dataframe's
                               index(es) as columns in the file output.
                               If ``False``, the index(es) will not be written to the file.
                               If ``None``, the index(ex) will be included as columns in the file
                               output except `RangeIndex` which is stored as metadata only.
        ==================     ====================================================================

        :returns: pyarrow.Table

        """
        from ._io import _arrow

        _arrow._validate_dataframe(self._data)
        df = self._data
        # create geo metadata before altering incoming data frame
        geo_metadata = _arrow._create_metadata(df)
        table = _pa.Table.from_pandas(df, preserve_index=index)

        # Store geopandas specific file-level metadata
        # This must be done AFTER creating the table or
        # it is not persisted
        metadata = table.schema.metadata
        metadata.update({b"geo": _arrow._encode_metadata(geo_metadata)})
        fin = table.replace_schema_metadata(metadata)

        return fin

    # ----------------------------------------------------------------------
    @staticmethod
    def from_arrow(table: "pyarrow.Table") -> pd.DataFrame:
        """
        Converts a Pandas DatFrame to an Arrow Table

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        table                  Required pyarrow.Table. The Arrow Table to convert back into a
                               spatially enabled dataframe.
        ==================     ====================================================================

        :returns: pandas.DataFrame

        """
        from ._io import _arrow

        return _arrow._arrow_to_sedf(table)

    # ----------------------------------------------------------------------
    def to_featureclass(
        self,
        location,
        overwrite=True,
        has_z=None,
        has_m=None,
        sanitize_columns=True,
    ):
        """
        The ``to_featureclass`` exports a spatially enabled dataframe to a feature class.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string. The output of the table.
        ---------------------------     --------------------------------------------------------------------
        overwrite                       Optional Boolean.  If True and if the feature class exists, it will be
                                        deleted and overwritten.  This is default.  If False, the feature class
                                        and the feature class exists, and exception will be raised.
        ---------------------------     --------------------------------------------------------------------
        has_z                           Optional Boolean.  If True, the dataset will be forced to have Z
                                        based geometries.  If a geometry is missing a Z value when true, a
                                        RuntimeError will be raised.  When False, the API will not use the
                                        Z value.
        ---------------------------     --------------------------------------------------------------------
        has_m                           Optional Boolean.  If True, the dataset will be forced to have M
                                        based geometries.  If a geometry is missing a M value when true, a
                                        RuntimeError will be raised. When False, the API will not use the
                                        M value.
        ---------------------------     --------------------------------------------------------------------
        sanitize_columns                Optional Boolean. If True, column names will be converted to string,
                                        invalid characters removed and other checks will be performed. The
                                        default is True.
        ===========================     ====================================================================

        :return:
            A String

        """
        if location and not str(os.path.dirname(location)).lower() in [
            "memory",
            "in_memory",
        ]:
            location = os.path.abspath(path=location)
        origin_columns = self._data.columns.tolist()
        origin_index = copy.deepcopy(self._data.index)
        result = to_featureclass(
            self,
            location=location,
            overwrite=overwrite,
            has_z=has_z,
            sanitize_columns=sanitize_columns,
            has_m=has_m,
        )
        self._data.columns = origin_columns
        self._data.index = origin_index
        return result

    # ----------------------------------------------------------------------
    def to_table(self, location, overwrite=True, **kwargs):
        """
        The ``to_table`` method exports a geo enabled dataframe to a :class:`~arcgis.features.Table` object.

        .. note::
            Null integer values will be changed to 0 when using shapely instead
            of ArcPy due to shapely conventions.
            With ArcPy null integer values will remain null.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string. The output of the table.
        ---------------------------     --------------------------------------------------------------------
        overwrite                       Optional Boolean.  If True and if the table exists, it will be
                                        deleted and overwritten.  This is default.  If False, the table and
                                        the table exists, and exception will be raised.
        ---------------------------     --------------------------------------------------------------------
        sanitize_columns                Optional Boolean. If True, column names will be converted to
                                        string, invalid characters removed and other checks will be
                                        performed. The default is True.
        ===========================     ====================================================================

        :return: String

        """
        from arcgis.features.geo._io.fileops import to_table
        from ._tools._utils import run_and_hide

        sanitize_columns = kwargs.pop("sanitize_columns", True)
        origin_columns = self._data.columns.tolist()
        origin_index = copy.deepcopy(self._data.index)
        if location and not str(os.path.dirname(location)).lower() in [
            "memory",
            "in_memory",
        ]:
            location = os.path.abspath(path=location)
        table = run_and_hide(
            to_table,
            **{
                "geo": self,
                "location": location,
                "overwrite": overwrite,
                "sanitize_columns": sanitize_columns,
            },
        )
        self._data.columns = origin_columns
        self._data.index = origin_index
        return table

    # ----------------------------------------------------------------------
    def to_parquet(
        self,
        path: str,
        index: bool = None,
        compression: str = "gzip",
        **kwargs,
    ) -> str:
        """
        Write a Spatially Enabled DataFrame to the Parquet format.

        Any geometry columns present are serialized to WKB format in the file.

        Requires 'pyarrow'.

        WARNING: this is an initial implementation of Parquet file support and
        associated metadata.  This is tracking version 0.4.0 of the metadata
        specification at:
        https://github.com/geopandas/geo-arrow-spec

        .. versionadded:: 2.1.0

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        path                   Required String. The save file path
        ------------------     --------------------------------------------------------------------
        index                  Optional Bool. If ``True``, always include the dataframe's
                               index(es) as columns in the file output.
                               If ``False``, the index(es) will not be written to the file.
                               If ``None``, the index(ex) will be included as columns in the file
                               output except `RangeIndex` which is stored as metadata only.
        ------------------     --------------------------------------------------------------------
        compression            Optional string. {'snappy', 'gzip', 'brotli', None}, default 'gzip'
                               Name of the compression to use. Use ``None`` for no compression.
        ------------------     --------------------------------------------------------------------
        **kwargs               Optional dict. Any additional kwargs that can be given to the
                               `pyarrow.parquet.write_table` method.
        ==================     ====================================================================

        :returns: string

        """
        from ._io._arrow import _to_parquet

        return _to_parquet(
            df=self._data,
            path=path,
            index=index,
            compression=compression,
            **kwargs,
        )

    # ----------------------------------------------------------------------
    def to_featurelayer(
        self,
        title=None,
        gis=None,
        tags=None,
        folder=None,
        sanitize_columns=False,
        service_name=None,
        **kwargs,
    ):
        """
        The ``to_featurelayer`` method publishes a spatial dataframe to a new
        :class:`~arcgis.features.FeatureLayer` object.

        .. note::
            Null integer values will be changed to 0 when using shapely instead
            of ArcPy due to shapely conventions.
            With ArcPy null integer values will remain null.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        title                           Optional string. The name of the service. If not provided, a random
                                        string is generated.
        ---------------------------     --------------------------------------------------------------------
        gis                             Optional GIS. The GIS connection object
        ---------------------------     --------------------------------------------------------------------
        tags                            Optional list of strings. A comma seperated list of descriptive
                                        words for the service.
        ---------------------------     --------------------------------------------------------------------
        folder                          Optional string. Name of the folder where the featurelayer item
                                        and imported data would be stored.
        ---------------------------     --------------------------------------------------------------------
        sanitize_columns                Optional Boolean. If True, column names will be converted to string,
                                        invalid characters removed and other checks will be performed. The
                                        default is False.
        ---------------------------     --------------------------------------------------------------------
        service_name                    Optional String. The name for the service that will be added to the Item.
                                        Name cannot be used already and cannot contain special characters, spaces,
                                        or a numerical value as the first letter.
        ===========================     ====================================================================

        When publishing a Spatial Dataframe, additional options can be given:

        ===========================     ====================================================================
        **Optional Arguments**          **Description**
        ---------------------------     --------------------------------------------------------------------
        overwrite                       Optional boolean. If True, the specified layer in the `service` parameter
                                        will be overwritten.
                                        .. note::
                                            Overwriting table data in Enterprise is not currently supported.
        ---------------------------     --------------------------------------------------------------------
        service                         Dictionary that is required if `overwrite = True`. Dictionary with two
                                        keys: "FeatureServiceId" and "layers".
                                        "featureServiceId" value is a string of the feature service id that the layer
                                        belongs to.
                                        "layer" value is an integer depicting the index value of the layer to
                                        overwrite.

                                        Example:
                                        {"featureServiceId" : "9311d21a9a2047d19c0faaebd6f2cca6", "layer": 0}
        ===========================     ====================================================================

        :return:
            A  :class:`~arcgis.features.FeatureLayer` object.

        """
        from arcgis import env
        import copy

        if gis is None:
            gis = env.active_gis
            if gis is None:
                raise ValueError("GIS object must be provided")
        content = gis.content
        origin_columns = self._data.columns.tolist()
        origin_index = copy.deepcopy(self._data.index)
        if title is None:
            title = uuid.uuid4().hex
        if service_name:
            # sanitize name
            service_name = service_name.replace(" ", "")
            if service_name[0].isnumeric():
                raise ValueError(
                    "First character of service_name cannot be an integer."
                )
            if (
                content.is_service_name_available(service_name, "featureService")
                is False
            ):
                raise ValueError(
                    "This service name is unavailable for Feature Service."
                )

        result = content.import_data(
            self._data,
            folder=folder,
            title=title,
            tags=tags,
            sanitize_columns=sanitize_columns,
            service_name=service_name,
            **kwargs,
        )
        self._data.columns = origin_columns
        self._data.index = origin_index
        return result

    # ----------------------------------------------------------------------
    @staticmethod
    def from_df(
        df,
        address_column="address",
        geocoder=None,
        sr=None,
        geometry_column=None,
    ):
        """
        The ``from_df`` creates a Spatially Enabled DataFrame from a dataframe with an address column.

        ====================    =========================================================
        **Parameter**            **Description**
        --------------------    ---------------------------------------------------------
        df                      Required Pandas DataFrame. Source dataset
        --------------------    ---------------------------------------------------------
        address_column          Optional String. The default is "address". This is the
                                name of a column in the specified dataframe that contains
                                addresses (as strings). The addresses are batch geocoded
                                using the GIS's first configured geocoder and their
                                locations used as the geometry of the spatial dataframe.
                                Ignored if the 'geometry_column' is specified.
        --------------------    ---------------------------------------------------------
        geocoder                Optional Geocoder. The geocoder to be used. If not
                                specified, the active GIS's first geocoder is used.
        --------------------    ---------------------------------------------------------
        sr                      Optional integer. The WKID of the spatial reference.
        --------------------    ---------------------------------------------------------
        geometry_column         Optional String.  The name of the geometry column to
                                convert to the arcgis.Geometry Objects (new at version 1.8.1)
        ====================    =========================================================

        :return:
            Spatially Enabled DataFrame




        NOTE: Credits will be consumed for batch_geocoding, from
        the GIS to which the geocoder belongs.

        """
        orig_df = df.copy()
        import arcgis
        from arcgis.geocoding import get_geocoders, geocode, batch_geocode
        from arcgis.geometry import Geometry

        if geometry_column:
            from arcgis.features import GeoAccessor, GeoSeriesAccessor

            if sr is None:
                try:
                    valid_index = df[geometry_column].first_valid_index()
                except:
                    raise ValueError(
                        "Column provided is all NULL, please provide a valid column"
                    )
                g = _geometry.Geometry(df[geometry_column].iloc[valid_index])
                sr = g.spatial_reference
                if isinstance(sr, Iterable) and "wkid" in sr:
                    sr = sr["wkid"] or 4326
                elif isinstance(sr, Iterable) and "wkt" in sr:
                    sr = sr["wkt"] or 4326
                else:
                    sr = 4326
            from ._array import GeoArray

            def _set_default_sr(geom):
                if sr:
                    geom["spatialReference"] = {"wkid": sr}
                elif "spatialReference" not in geom:
                    geom["spatialReference"] = {"wkid": 4326}
                elif geom["spatialReference"] is None:
                    geom["spatialReference"] = {"wkid": 4326}
                elif (
                    geom["spatialReference"].get("wkid", None) is None
                    and geom["spatialReference"].get("wkt", None) is None
                ):
                    geom["spatialReference"] = {"wkid": 4326}
                return geom

            series = df[geometry_column].apply(Geometry).apply(_set_default_sr)
            df[geometry_column] = GeoArray(series)
            df.spatial.set_geometry(geometry_column)
            df.spatial.project(sr)
            return df
        else:
            if geocoder is None:
                geocoder = arcgis.env.active_gis._tools.geocoders[0]
            sr = dict(geocoder.properties.spatialReference)
            if address_column in df.columns:
                batch_size = geocoder.properties.locatorProperties.MaxBatchSize
                pieces = [
                    df.iloc[i : i + batch_size] for i in range(0, len(df), batch_size)
                ]
                data = []
                for df in pieces:
                    piece_df = batch_geocode(
                        list(df[address_column]),
                        geocoder=geocoder,
                        as_featureset=True,
                    ).sdf.sort_values(by="ResultID")
                    piece_df.index = df.index
                    piece_df["ResultID"] = df.index.tolist()
                    data.append(piece_df)
                if len(data) == 1:
                    merged = orig_df.merge(
                        data[0], left_index=True, right_on="ResultID"
                    )
                else:
                    merged = orig_df.merge(
                        pd.concat(data), left_index=True, right_on="ResultID"
                    )
            else:
                raise ValueError("Address column not found in dataframe")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if "SHAPE" in merged.columns:
                    merged.spatial.set_geometry("SHAPE")
            return merged

    # ----------------------------------------------------------------------
    @staticmethod
    def from_xy(
        df,
        x_column,
        y_column,
        sr=4326,
        z_column=None,
        m_column=None,
        **kwargs,
    ):
        """
        The ``from_xy`` method converts a Pandas DataFrame into a Spatially Enabled DataFrame
        by providing the X/Y columns.

        ====================    =========================================================
        **Parameter**            **Description**
        --------------------    ---------------------------------------------------------
        df                      Required Pandas DataFrame. Source dataset
        --------------------    ---------------------------------------------------------
        x_column                Required string.  The name of the X-coordinate series
        --------------------    ---------------------------------------------------------
        y_column                Required string.  The name of the Y-coordinate series
        --------------------    ---------------------------------------------------------
        sr                      Optional int.  The wkid number of the spatial reference.
                                4326 is the default value.
        --------------------    ---------------------------------------------------------
        z_column                Optional string.  The name of the Z-coordinate series
        --------------------    ---------------------------------------------------------
        m_column                Optional string.  The name of the M-value series
        ====================    =========================================================


        ====================    =========================================================
        **kwargs**              **Description**
        --------------------    ---------------------------------------------------------
        oid_field               Optional string. If the value is provided the OID field
                                will not be converted from int64 to int32.
        ====================    =========================================================

        :return: DataFrame

        """
        from ._io.fileops import _from_xy

        return _from_xy(
            df=df,
            x_column=x_column,
            y_column=y_column,
            sr=sr,
            z_column=z_column,
            m_column=m_column,
            oid_field=kwargs.pop("oid_field", None),
        )

    # ----------------------------------------------------------------------
    @staticmethod
    def from_layer(layer):
        """
        The ``from_layer`` method imports a :class:`~arcgis.features.FeatureLayer` to a Spatially Enabled DataFrame

        .. note::
            This operation converts a :class:`~arcgis.features.FeatureLayer` or
            :class:`~arcgis.features.Table` to a Pandas' DataFrame

        ====================    =========================================================
        **Parameter**            **Description**
        --------------------    ---------------------------------------------------------
        layer                   Required FeatureLayer or TableLayer. The service to convert
                                to a Spatially enabled DataFrame.
        ====================    =========================================================

        Usage:

        >>> from arcgis.features import FeatureLayer
        >>> mylayer = FeatureLayer(("https://sampleserver6.arcgisonline.com/arcgis/rest"
                            "/services/CommercialDamageAssessment/FeatureServer/0"))
        >>> df = from_layer(mylayer)
        >>> print(df.head())

        :return:
            A Pandas' `DataFrame`

        """
        import json

        try:
            from arcgis.features.geo._io.serviceops import from_layer

            return from_layer(layer=layer)
        except ImportError:
            raise ImportError("Could not load `from_layer`.")
        except json.JSONDecodeError as je:
            raise Exception(
                "Malformed response from server, could not load the dataset: %s"
                % str(je)
            )
        except Exception as e:
            raise Exception("Could not load the dataset: %s" % str(e))

    # ----------------------------------------------------------------------
    @staticmethod
    def from_featureclass(location, **kwargs):
        """
        The ``from_featureclass`` creates a Spatially enabled `pandas.DataFrame` from a
        :class:`~arcgis.features.Features` class.

        .. note::
            Null integer values will be changed to 0 when using shapely instead
            of ArcPy due to shapely conventions.
            With ArcPy null integer values will remain null.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string or pathlib.Path. Full path to the feature class or URL (shapefile only).
        ===========================     ====================================================================

        *Optional parameters when ArcPy library is available in the current environment*:

        ===========================     ====================================================================
        **Optional Argument**           **Description**
        ---------------------------     --------------------------------------------------------------------
        sql_clause                      sql clause to parse data down. To learn more see
                                        `ArcPy Search Cursor <https://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm>`_
        ---------------------------     --------------------------------------------------------------------
        where_clause                    where statement. To learn more see `ArcPy SQL reference <https://pro.arcgis.com/en/pro-app/help/mapping/navigation/sql-reference-for-elements-used-in-query-expressions.htm>`_
        ---------------------------     --------------------------------------------------------------------
        fields                          list of strings specifying the field names.
        ---------------------------     --------------------------------------------------------------------
        spatial_filter                  A `Geometry` object that will filter the results.  This requires
                                        `arcpy` to work.
        ---------------------------     --------------------------------------------------------------------
        sr                              A Spatial reference to project (or transform) output GeoDataFrame
                                        to. This requires `arcpy` to work.
        ---------------------------     --------------------------------------------------------------------
        datum_transformation            Used in combination with 'sr' parameter. if the spatial reference of
                                        output GeoDataFrame and input data do not share the same datum,
                                        an appropriate datum transformation should be specified.
                                        To Learn more see [Geographic datum transformations](https://pro.arcgis.com/en/pro-app/help/mapping/properties/geographic-coordinate-system-transformation.htm)
                                        This requires `arcpy` to work.
        ===========================     ====================================================================

        **Optional Parameters are not supported for URL based resources**

        :return:
            A pandas.core.frame.DataFrame object
        """
        return from_featureclass(filename=location, **kwargs)

    # ----------------------------------------------------------------------
    @staticmethod
    def from_table(filename, **kwargs):
        """
        The ``from_table`` method allows a :class:`~arcgis.gis.User` to read from a non-spatial table

        .. note::
            The ``from_table`` method requires ArcPy

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        filename            Required string or pathlib.Path. The path to the
                            table.
        ===============     ====================================================

        **Keyword Arguments**

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        fields              Optional List/Tuple. A list (or tuple) of field
                            names. For a single field, you can use a string
                            instead of a list of strings.

                            Use an asterisk (*) instead of a list of fields if
                            you want to access all fields from the input table
                            (raster and BLOB fields are excluded). However, for
                            faster performance and reliable field order, it is
                            recommended that the list of fields be narrowed to
                            only those that are actually needed.

                            Geometry, raster, and BLOB fields are not supported.

        ---------------     ----------------------------------------------------
        where               Optional String. An optional expression that limits
                            the records returned.
        ---------------     ----------------------------------------------------
        skip_nulls          Optional Boolean. This controls whether records
                            using nulls are skipped. Default is True.
        ---------------     ----------------------------------------------------
        null_value          Optional String/Integer/Float. Replaces null values
                            from the input with a new value.
        ===============     ====================================================

        :return:
            A Pandas DataFrame (pd.DataFrame)
        """
        from arcgis.features.geo._io.fileops import from_table

        return from_table(filename, **kwargs)

    # ----------------------------------------------------------------------
    def sindex(self, stype="quadtree", reset=False, **kwargs):
        """
        The ``sindex`` creates a spatial index for the given dataset.

        .. note::
            By default, the spatial index is a QuadTree spatial index.

        r-tree indexes should be used for large datasets.  This will allow
        users to create very large out of memory indexes.  To use r-tree indexes,
        the r-tree library must be installed.  To do so, install via conda using
        the following command: `conda install -c conda-forge rtree`

        :return:
            A spatial index for the given dataset.
        """
        from arcgis.features.geo._index._impl import SpatialIndex

        c = 0
        filename = kwargs.pop("filename", None)
        if reset:
            self._sindex = None
            self._sfname = None
            self._stype = None
        if self._sindex:
            return self._sindex
        # bbox = self.full_extent
        if (
            self.name
            and filename
            and os.path.isfile(filename + ".dat")
            and os.path.isfile(filename + ".idx")
        ):
            l = len(self._data[self.name])
            self._sindex = SpatialIndex(
                stype=stype, filename=filename, bbox=self.full_extent
            )
            for idx, g in zip(self._index, self._data[self.name]):
                if g:
                    if g.type.lower() == "point":
                        ge = g.geoextent
                        gext = (
                            ge[0] - 0.001,
                            ge[1] - 0.001,
                            ge[2] + 0.001,
                            ge[3] - 0.001,
                        )
                        self._sindex.insert(oid=idx, bbox=gext)
                    else:
                        self._sindex.insert(oid=idx, bbox=g.geoextent)
                    if c >= int(l / 4) + 1:
                        self._sindex.flush()
                        c = 0
                    c += 1
            self._sindex.flush()
            return self._sindex
        elif self.name:
            c = 0
            l = len(self._data[self.name])
            self._sindex = SpatialIndex(
                stype=stype, filename=filename, bbox=self.full_extent
            )
            for idx, g in zip(self._index, self._data[self.name]):
                if g:
                    if g.type.lower() == "point":
                        ge = g.geoextent
                        gext = (
                            ge[0] - 0.001,
                            ge[1] - 0.001,
                            ge[2] + 0.001,
                            ge[3] - 0.001,
                        )
                        self._sindex.insert(oid=idx, bbox=gext)
                    else:
                        self._sindex.insert(oid=idx, bbox=g.geoextent)
                    if c >= int(l / 4) + 1:
                        self._sindex.flush()
                        c = 0
                    c += 1
            self._sindex.flush()
            return self._sindex
        else:
            raise ValueError(
                ("The Spatial Column must " "be set, call df.spatial.set_geometry.")
            )

    # ----------------------------------------------------------------------
    @property
    def __geo_interface__(self):
        """returns the object as an Feature Collection JSON string"""
        template = {"type": "FeatureCollection", "features": []}
        for index, row in self._data.iterrows():
            geom = row[self.name]
            del row[self.name]
            gj = copy.copy(geom.__geo_interface__)
            gj["attributes"] = pd.io.json.loads(
                pd.io.json.dumps(row)
            )  # ensures the values are converted correctly
            template["features"].append(gj)
        return pd.io.json.dumps(template)

    # ----------------------------------------------------------------------
    @property
    def __feature_set__(self):
        """returns a dictionary representation of an Esri FeatureSet"""
        import arcgis

        fields = []
        features = []
        date_fields = []
        _geom_types = {
            arcgis.geometry._types.Point: "esriGeometryPoint",
            arcgis.geometry._types.Polyline: "esriGeometryPolyline",
            arcgis.geometry._types.MultiPoint: "esriGeometryMultipoint",
            arcgis.geometry._types.Polygon: "esriGeometryPolygon",
        }
        if self.sr is None:
            sr = {"wkid": 4326}
        else:
            sr = self.sr
        if self.name is None:
            geom_type = "esriGeometryPoint"
        else:
            geom_type = _geom_types[
                type(self._data[self.name][self._data[self.name].first_valid_index()])
            ]

        fs = {
            "objectIdFieldName": "",
            "globalIdFieldName": "",
            "displayFieldName": "",
            "geometryType": geom_type,
            "spatialReference": sr,
            "fields": [],
            "features": [],
        }
        # Ensure all number values are 0 so errors do not occur.
        replace_mappings = {
            pd.NA: None,
            np.nan: None,
            np.NaN: None,
            np.NAN: None,
            pd.NaT: None,
        }
        df = self._data.copy()
        date_fields = [
            col for col in df.columns if df[col].dtype.name.find("datetime") > -1
        ]
        time_delta_fields = [
            col
            for col in df.columns
            if df[col].dtype.name.find("timedelta") > -1
            or df[col].dtype.name.find("<m8[ns]") > -1
        ]
        cols_norm = [col for col in df.columns]
        cols_lower = [col.lower() for col in df.columns]

        if "objectid" in cols_lower:
            fs["objectIdFieldName"] = cols_norm[cols_lower.index("objectid")]
            fs["displayFieldName"] = cols_norm[cols_lower.index("objectid")]
            if df[fs["objectIdFieldName"]].is_unique == False:
                old_series = df[fs["objectIdFieldName"]].copy()
                df[fs["objectIdFieldName"]] = list(range(1, df.shape[0] + 1))

        elif "fid" in cols_lower:
            fs["objectIdFieldName"] = cols_norm[cols_lower.index("fid")]
            fs["displayFieldName"] = cols_norm[cols_lower.index("fid")]
            if df[fs["objectIdFieldName"]].is_unique == False:
                old_series = df[fs["objectIdFieldName"]].copy()
                df[fs["objectIdFieldName"]] = list(range(1, df.shape[0] + 1))

        elif "oid" in cols_lower:
            fs["objectIdFieldName"] = cols_norm[cols_lower.index("oid")]
            fs["displayFieldName"] = cols_norm[cols_lower.index("oid")]
            if df[fs["objectIdFieldName"]].is_unique == False:
                old_series = df[fs["objectIdFieldName"]].copy()
                df[fs["objectIdFieldName"]] = list(range(1, df.shape[0] + 1))

        else:
            fs["objectIdFieldName"] = "OBJECTID"
            fs["displayFieldName"] = "OBJECTID"

            df["OBJECTID"] = list(range(1, df.shape[0] + 1))
            cols_norm = [col for col in df.columns]
            cols_lower = [col.lower() for col in df.columns]
            # res = self.__feature_set__
            # del df['OBJECTID']
            # return res
        if "objectIdFieldName" in fs:
            fields.append(
                {
                    "name": fs["objectIdFieldName"],
                    "type": "esriFieldTypeOID",
                    "alias": fs["objectIdFieldName"],
                }
            )
            cols_norm.pop(cols_norm.index(fs["objectIdFieldName"]))
        if "globalIdFieldName" in fs and len(fs["globalIdFieldName"]) > 0:
            fields.append(
                {
                    "name": fs["globalIdFieldName"],
                    "type": "esriFieldTypeGlobalID",
                    "alias": fs["globalIdFieldName"],
                }
            )
            cols_norm.pop(cols_norm.index(fs["globalIdFieldName"]))
        elif "globalIdFieldName" in fs and len(fs["globalIdFieldName"]) == 0:
            del fs["globalIdFieldName"]
        if self.name in cols_norm:
            cols_norm.pop(cols_norm.index(self.name))
        from numpy import dtype as _dtype

        _look_up = {
            np.int8: "esriFieldTypeInteger",
            _dtype(bool): "esriFieldTypeInteger",
            bool: "esriFieldTypeInteger",
            _dtype(np.int8): "esriFieldTypeInteger",
            np.int16: "esriFieldTypeInteger",
            _dtype(np.int16): "esriFieldTypeInteger",
            np.int32: "esriFieldTypeInteger",
            _dtype(np.int32): "esriFieldTypeInteger",
            np.int64: "esriFieldTypeBigInteger",
            _dtype(np.int64): "esriFieldTypeBigInteger",
            pd.Int64Dtype(): "esriFieldTypeBigInteger",
            pd.Int32Dtype(): "esriFieldTypeInteger",
            int: "esriFieldTypeInteger",
            float: "esriFieldTypeDouble",
            np.float16: "esriFieldTypeSingle",
            _dtype(np.float16): "esriFieldTypeSingle",
            np.float32: "esriFieldTypeDouble",
            _dtype(np.float32): "esriFieldTypeDouble",
            np.float64: "esriFieldTypeDouble",
            _dtype(np.float64): "esriFieldTypeDouble",
            pd.Float32Dtype(): "esriFieldTypeDouble",
            pd.Float64Dtype(): "esriFieldTypeDouble",
            "geometry": "esriFieldTypeGeometry",
            str: "esriFieldTypeString",
            _dtype("O"): "esriFieldTypeString",
            object: "esriFieldTypeString",
            _dtype(str): "esriFieldTypeString",
            pd.StringDtype(): "esriFieldTypeString",
            "<m8[ns]": "esriFieldTypeDouble",
            _dtype("<m8[ns]"): "esriFieldTypeDouble",
            _dtype("<M8[s]"): "esriFieldTypeDateOnly",
            _dtype("<m8[us]"): "esriFieldTypeTimeOnly",
            _dtype("<M8[us]"): "esriFieldTypeTimestampOffset",
            "<M8[us]": "esriFieldTypeDate",
            np.dtype("<M8[ns]"): "esriFieldTypeDate",
            datetime: "esriFieldTypeDate",
            np.datetime64: "esriFieldTypeDate",
            _dtype(np.datetime64): "esriFieldTypeDate",
            arcgis.features.geo._array.GeoType(): "esriFieldTypeGeometry",
            arcgis.features.geo._array.GeoType: "esriFieldTypeGeometry",
            arcgis.geometry._types.Geometry: "esriFieldTypeGeometry",
            pd.CategoricalDtype: "category",
            pd.Timedelta: "esriFieldTypeDouble",
            pd.Timestamp: "esriFieldTypeDate",
            pd.BooleanDtype: "esriFieldTypeInteger",
            pd.BooleanDtype(): "esriFieldTypeInteger",
            pd.UInt8Dtype: "esriFieldTypeInteger",
            pd.UInt8Dtype(): "esriFieldTypeInteger",
            pd.UInt16Dtype: "esriFieldTypeInteger",
            pd.UInt16Dtype(): "esriFieldTypeInteger",
            pd.UInt32Dtype: "esriFieldTypeInteger",
            pd.UInt32Dtype(): "esriFieldTypeInteger",
            pd.UInt64Dtype: "esriFieldTypeBigInteger",
            pd.UInt64Dtype(): "esriFieldTypeBigInteger",
        }
        fields = []
        for idx, dtype in enumerate(self._data.dtypes):
            column = None
            col = self._data.dtypes.index[idx]
            if fs["objectIdFieldName"] == col:
                column = {
                    "name": col,
                    "type": "esriFieldTypeOID",
                    "alias": col,
                }
            elif dtype.name.find("datetime") > -1:
                lu = _look_up[np.datetime64]
                column = {
                    "name": col,
                    "type": lu,
                    "alias": col,
                }
            elif isinstance(dtype, pd.CategoricalDtype):
                length = None
                if dtype.categories.dtype.name == "object":
                    lu = "esriFieldTypeString"
                    try:
                        length = max(dtype.categories.str.len())
                    except:
                        length = 254
                elif dtype.categories.dtype.name.find("datetime") > -1:
                    lu = _look_up[dtype.categories.dtype]
                elif dtype.categories.dtype.name.find("timedelta") > -1:
                    lu = _look_up[dtype.categories.dtype]
                else:
                    lu = _look_up[dtype.categories.dtype]
                column = {
                    "name": col,
                    "type": lu,
                    "alias": col,
                }
                if length:
                    column["length"] = length
            else:
                column = {
                    "name": col,
                    "type": _look_up[dtype],
                    "alias": col,
                }
            if column["type"] == "esriFieldTypeString":
                try:
                    max_length = int(self._data[col].str.len().max())
                    if max_length == 0:
                        max_length = 256
                    column["length"] = max_length
                except:
                    column["length"] = 256
            if column and isinstance(dtype, pd.CategoricalDtype):
                fields.append(column)
            elif column and (
                (dtype in _look_up and _look_up[dtype] != "esriFieldTypeGeometry")
                or _look_up[infer_dtype_from_object(dtype)]
            ):
                fields.append(column)

        fs["fields"] = fields
        number_columns = df.select_dtypes(np.number).columns.tolist()
        df[number_columns] = df[number_columns].replace(pd.NA, 0)
        string_column = df.select_dtypes(pd.StringDtype()).columns.tolist()
        df[string_column] = df[string_column].replace(pd.NA, "")
        df = df.replace(replace_mappings).convert_dtypes().copy()  #

        for td in time_delta_fields:
            df[td] = df[td].dt.total_seconds() * 1000
        for f in date_fields:
            fn = (
                lambda x: int(x.timestamp() * 1000)
                if isinstance(x, pd.Timestamp)
                else 0
            )

            df[f] = pd.to_datetime(df[date_fields[-1]]).apply(fn)
        for row in df.to_dict("records"):
            geom = {}
            if self.name in row:
                geom = row[self.name]
                del row[self.name]
            if geom and pd.notna(geom):
                features.append({"geometry": dict(geom), "attributes": row})
            elif pd.notna(geom) == False:
                features.append({"geometry": None, "attributes": row})
            else:
                features.append({"geometry": geom, "attributes": row})
            del row
            del geom
        fs["features"] = features

        return fs

    # ----------------------------------------------------------------------
    def _check_geometry_engine(self):
        if self._HASARCPY is None:
            try:
                import arcpy

                self._HASARCPY = True
            except:
                self._HASARCPY = False
        if self._HASSHAPELY is None:
            try:
                import shapely

                self._HASSHAPELY = True
            except:
                self._HASSHAPELY = False
        return self._HASARCPY, self._HASSHAPELY

    # ----------------------------------------------------------------------
    @property
    def sr(self):
        """
        The ``sr`` property gets and sets the :class:`~arcgis.geometry.SpatialReference` of the dataframe

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        value                   Spatial Reference
        ==================      ====================================================================
        """
        if self.name:
            data = [
                getattr(g, "spatialReference", None) or g["spatialReference"]
                for g in self._data[self.name]
                if g not in [None, np.NaN, np.nan, "", {}] and isinstance(g, dict)
            ]
            srs = [
                _geometry.SpatialReference(sr)
                for sr in pd.DataFrame(data).drop_duplicates().to_dict("records")
            ]
            if len(srs) == 1:
                return srs[0]
            return srs

    # ----------------------------------------------------------------------
    @sr.setter
    def sr(self, ref):
        """
        See main ``sr`` property docstring
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            try:
                sr = self.sr
            except:
                sr = None
            if sr and "wkid" in sr:
                wkid = sr["wkid"]
            elif sr and "latestWkid" in sr:
                wkid = sr["latestWkid"]
            if sr and "wkt" in sr:
                wkt = sr["wkt"]

            if isinstance(ref, (dict, _geometry.SpatialReference)) and sr is None:
                self._data[self.name] = self._data[self.name].geom.project_as(ref)
            elif isinstance(ref, _geometry.SpatialReference):
                if ref != sr:
                    self._data[self.name] = self._data[self.name].geom.project_as(ref)
            elif isinstance(ref, int):
                if ref != wkid:
                    self._data[self.name] = self._data[self.name].geom.project_as(ref)
            elif isinstance(ref, str):
                if ref != wkt:
                    self._data[self.name] = self._data[self.name].geom.project_as(ref)
            elif isinstance(ref, dict):
                nsr = _geometry.SpatialReference(ref)
                if sr != nsr:
                    self._data[self.name] = self._data[self.name].geom.project_as(ref)
        else:
            if ref:
                if isinstance(ref, str):
                    ref = {"wkt": ref}
                elif isinstance(ref, int):
                    ref = {"wkid": ref}
                if len(self._data[self.name]) > 0:
                    self._data[self.name].apply(
                        lambda x: x.update({"spatialReference": ref})
                        if pd.notnull(x)
                        else None
                    )

    # ----------------------------------------------------------------------
    def to_featureset(self):
        """
        The ``to_featureset`` method converts a Spatially Enabled DataFrame object. to a
        :class:`~arcgis.features.FeatureSet` object.

        :return:
            A :class:`~arcgis.features.FeatureSet` object
        """
        from arcgis.features import FeatureSet

        d = self.__feature_set__
        return FeatureSet.from_dict(d)

    # ----------------------------------------------------------------------
    def to_feature_collection(
        self,
        name=None,
        drawing_info=None,
        extent=None,
        global_id_field=None,
        sanitize_columns=False,
    ):
        """
        The ``to_feature_collection`` converts a spatially enabled a Pandas DataFrame to a
        :class:`~arcgis.features.FeatureCollection` .

        =====================  ===============================================================
        **optional argument**  **Description**
        ---------------------  ---------------------------------------------------------------
        name                   optional string. Name of the :class:`~arcgis.features.FeatureCollection`
        ---------------------  ---------------------------------------------------------------
        drawing_info           Optional dictionary. This is the rendering information for a
                               Feature Collection.  Rendering information is a dictionary with
                               the symbology, labelling and other properties defined.  See the
                               `Renderer Objects <https://developers.arcgis.com/documentation/common-data-types/renderer-objects.htm>`_
                               page in the ArcGIS REST API for more information.
        ---------------------  ---------------------------------------------------------------
        extent                 Optional dictionary.  If desired, a custom extent can be
                               provided to set where the map starts up when showing the data.
                               The default is the full extent of the dataset in the Spatial
                               DataFrame.
        ---------------------  ---------------------------------------------------------------
        global_id_field        Optional string. The Global ID field of the dataset.
        ---------------------  ---------------------------------------------------------------
        sanitize_columns       Optional Boolean. If True, column names will be converted to string,
                               invalid characters removed and other checks will be performed. The
                               default is False.
        =====================  ===============================================================

        :return:
            A :class:`~arcgis.features.FeatureCollection` object
        """
        from arcgis.features import FeatureCollection
        import string
        import copy
        import random

        old_columns, old_index = None, None
        if sanitize_columns:
            old_columns = self._data.columns.tolist()
            old_index = copy.deepcopy(self._data.index)
            pd.DataFrame.reset_index(self._data)
            self._data.reset_index(drop=True)
            self.sanitize_column_names(inplace=True)
        if name is None:
            name = random.choice(string.ascii_letters) + uuid.uuid4().hex[:5]
        template = {"showLegend": True, "layers": []}
        if extent is None:
            ext = self.full_extent
            extent = {
                "xmin": ext[0],
                "ymin": ext[1],
                "xmax": ext[2],
                "ymax": ext[3],
                "spatialReference": self.sr,
            }
        fs = self.__feature_set__
        fields = []
        for fld in fs["fields"]:
            if fld["name"].lower() == fs["objectIdFieldName"].lower():
                fld["editable"] = False
                fld["sqlType"] = "sqlTypeOther"
                fld["domain"] = None
                fld["defaultValue"] = None
                fld["nullable"] = False
            else:
                fld["editable"] = True
                fld["sqlType"] = "sqlTypeOther"
                fld["domain"] = None
                fld["defaultValue"] = None
                fld["nullable"] = True
        if drawing_info is None:
            import json

            di = {"renderer": json.loads(self._data.spatial.renderer.json)}
        else:
            di = drawing_info
        layer = {
            "layerDefinition": {
                "currentVersion": 10.7,
                "id": 0,
                "name": name,
                "type": "Feature Layer",
                "displayField": "",
                "description": "",
                "copyrightText": "",
                "defaultVisibility": True,
                "relationships": [],
                "isDataVersioned": False,
                "supportsAppend": True,
                "supportsCalculate": True,
                "supportsASyncCalculate": True,
                "supportsTruncate": False,
                "supportsAttachmentsByUploadId": True,
                "supportsAttachmentsResizing": True,
                "supportsRollbackOnFailureParameter": True,
                "supportsStatistics": True,
                "supportsExceedsLimitStatistics": True,
                "supportsAdvancedQueries": True,
                "supportsValidateSql": True,
                "supportsCoordinatesQuantization": True,
                "supportsFieldDescriptionProperty": True,
                "supportsQuantizationEditMode": True,
                "supportsApplyEditsWithGlobalIds": False,
                "supportsMultiScaleGeometry": True,
                "supportsReturningQueryGeometry": True,
                "hasGeometryProperties": True,
                "advancedQueryCapabilities": {
                    "supportsPagination": True,
                    "supportsPaginationOnAggregatedQueries": True,
                    "supportsQueryRelatedPagination": True,
                    "supportsQueryWithDistance": True,
                    "supportsReturningQueryExtent": True,
                    "supportsStatistics": True,
                    "supportsOrderBy": True,
                    "supportsDistinct": True,
                    "supportsQueryWithResultType": True,
                    "supportsSqlExpression": True,
                    "supportsAdvancedQueryRelated": True,
                    "supportsCountDistinct": True,
                    "supportsReturningGeometryCentroid": True,
                    "supportsReturningGeometryProperties": True,
                    "supportsQueryWithDatumTransformation": True,
                    "supportsHavingClause": True,
                    "supportsOutFieldSQLExpression": True,
                    "supportsMaxRecordCountFactor": True,
                    "supportsTopFeaturesQuery": True,
                    "supportsDisjointSpatialRel": True,
                    "supportsQueryWithCacheHint": True,
                },
                "useStandardizedQueries": False,
                "geometryType": fs["geometryType"],
                "minScale": 0,
                "maxScale": 0,
                "extent": extent,
                "drawingInfo": di,
                "allowGeometryUpdates": True,
                "hasAttachments": False,
                "htmlPopupType": "esriServerHTMLPopupTypeNone",
                "hasM": False,
                "hasZ": False,
                "objectIdField": fs["objectIdFieldName"] or "OBJECTID",
                "globalIdField": "",
                "typeIdField": "",
                "fields": fs["fields"],
                "types": [],
                "supportedQueryFormats": "JSON, geoJSON",
                "hasStaticData": True,
                "maxRecordCount": 32000,
                "standardMaxRecordCount": 4000,
                "tileMaxRecordCount": 4000,
                "maxRecordCountFactor": 1,
                "capabilities": "Query",
            },
            "featureSet": {
                "features": fs["features"],
                "geometryType": fs["geometryType"],
            },
        }
        if global_id_field is not None:
            layer["layerDefinition"]["globalIdField"] = global_id_field
        if not old_columns is None and not old_index is None:
            self._data.columns = old_columns
            self._data.index = old_index
        return FeatureCollection(layer)

    # ---------------------------------------------------------------------

    @staticmethod
    def from_geodataframe(geo_df, inplace=False, column_name="SHAPE"):
        """
        The ``from_geodataframe`` loads a Geopandas GeoDataFrame into an ArcGIS Spatially Enabled DataFrame.

        .. note::
            The ``from_geodataframe`` method requires geopandas library be installed in current environment.

        =====================  ===============================================================
        **Parameter**           **Description**
        ---------------------  ---------------------------------------------------------------
        geo_df                 GeoDataFrame object, created using GeoPandas library
        ---------------------  ---------------------------------------------------------------
        inplace                Optional Bool. When True, the existing GeoDataFrame is spatially
                                enabled and returned. When False, a new Spatially Enabled
                                DataFrame object is returned. Default is False.
        ---------------------  ---------------------------------------------------------------
        column_name            Optional String. Sets the name of the geometry column. Default
                                is `SHAPE`.
        =====================  ===============================================================

        :return:
            A Spatially Enabled DataFrame.
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "Requires Geopandas library installed for this functionality"
            )

        # import geometry libraries
        from arcgis.geometry import Geometry as ags_geometry
        from arcgis.features.geo._array import GeoArray

        # import pandas
        import pandas as pd
        import numpy as np

        # get wkid
        try:
            if geo_df.crs is not None and hasattr(geo_df.crs, "to_epsg"):
                # check for pyproj
                epsg_code = geo_df.crs.to_epsg()
            elif geo_df.crs is not None and "init" in geo_df.crs:
                epsg_code = geo_df.crs["init"].split(":")[-1]
                epsg_code = int(epsg_code)  # convert string to number
            elif geo_df.crs is not None:
                # crs is present, but no epsg code. Try to reproject to 4326
                geo_df.to_crs(epsg=4326, inplace=True)
                epsg_code = 4326
            else:
                _LOGGER.info(
                    "Cannot acquire spatial reference from GeoDataFrame. Setting it a default of WKID 4326"
                )
                epsg_code = 4326  # set a safe default value

        except Exception as proj_ex:
            _LOGGER.warning(
                "Error acquiring spatial reference from GeoDataFrame"
                " Spatial reference will not be set." + str(proj_ex)
            )
            epsg_code = None

        if epsg_code:
            spatial_reference = {"wkid": epsg_code}
        else:
            spatial_reference = None

        # convert geometry
        def _converter(g):
            if g is not None:
                # return ags_geometry(shp_mapping(g))
                return ags_geometry.from_shapely(g, spatial_reference=spatial_reference)
            else:
                return None

        # vectorize converter so it will run efficiently on pd.Series - avoids loops
        v_func = np.vectorize(_converter, otypes="O")

        # initialize empty array
        ags_geom = np.empty(geo_df.shape[0], dtype="O")

        ags_geom[:] = v_func(geo_df[geo_df.geometry.name].values)

        if inplace:
            geo_df[column_name] = GeoArray(ags_geom)
        else:
            geo_df = pd.DataFrame(geo_df.drop(columns=geo_df.geometry.name))
            geo_df[column_name] = GeoArray(ags_geom)

        geo_df.spatial.set_geometry(column_name)
        geo_df.spatial.sr = spatial_reference

        return geo_df

    # ----------------------------------------------------------------------
    def eq(self, other: GeoAccessor | pd.DataFrame):
        """
        Check if two DataFrames are equal to each other. Equal means
        same shape and corresponding elements
        """
        return self.__eq__(other)

    # ----------------------------------------------------------------------
    def __eq__(self, other: GeoAccessor):
        """
        Check if two DataFrames are equal to each other. Equal means
        same shape and corresponding elements
        """
        # Convert DataFrame
        if isinstance(other, pd.DataFrame):
            if _is_geoenabled(other):
                other = other.spatial
            else:
                raise ValueError(
                    "The comparative item must be a DataFrame with spatial capabilities or be an instance of GeoAccessor."
                )

        if not isinstance(other, GeoAccessor):
            raise ValueError("Input must be features.geo.GeoAccessor")

        # Check the shape
        if self._data.shape != other._data.shape:
            return False

        # Check columns are the same
        if set(self._data.columns) != set(other._data.columns):
            return False

        # Check rows are the same
        compared = self.compare(other)
        if (
            compared["added_rows"].empty
            and compared["deleted_rows"].empty
            and compared["modified_rows"].empty
        ):
            return True
        else:
            return False

    # ----------------------------------------------------------------------
    def compare(self, other: GeoAccessor | pd.DataFrame, match_field: str = None):
        """
        Compare the current spatially enabled DataFrame with another spatially enabled DataFrame and identify the differences
        in terms of added, deleted, and modified rows based on a specified match field.

        ===============     ===========================================================
        **Parameter**       **Description**
        ---------------     -----------------------------------------------------------
        other               Required spatially enabled DataFrame (GeoAccessor object).
        ---------------     -----------------------------------------------------------
        match_field         Optional string. The field to use for matching rows between
                            the DataFrames. The default will be the spatial column's name.
        ===============     ===========================================================

        :return: A dictionary containing the differences between the two DataFrames:
            - 'added_rows': DataFrame representing the rows added in the other DataFrame.
            - 'deleted_rows': DataFrame representing the rows deleted from the current DataFrame.
            - 'modified_rows': DataFrame representing the rows modified between the DataFrames.

        """
        if isinstance(other, pd.DataFrame):
            if _is_geoenabled(other):
                other = other.spatial
            else:
                raise ValueError(
                    "The comparative item must be a DataFrame with spatial capabilities or be an instance of GeoAccessor."
                )
        if match_field is None:
            match_field = self.name
        old_df = self._data
        new_df = other._data
        diff = {
            "added_rows": {},
            "deleted_rows": {},
            "modified_rows": {},
        }

        if old_df.empty and new_df.empty:
            _LOGGER.error(
                "Both dataframes are empty, cannot compate two empty dataframes"
            )
            return diff

        if old_df.empty and not new_df.empty:
            old_df = pd.DataFrame(data=None, columns=new_df.columns, index=new_df.index)

        if new_df.empty and not old_df.empty:
            new_df = pd.DataFrame(data=None, columns=old_df.columns, index=old_df.index)

        # Finding changes in rows
        merged_rows = new_df.merge(
            old_df,
            on=match_field,
            how="outer",
            indicator=True,
            suffixes=("_new", "_old"),
        )

        # Finding added rows
        added_rows = merged_rows[merged_rows["_merge"] == "left_only"].drop(
            columns=["_merge"]
        )
        # Removing the old
        for column in added_rows.columns:
            if column.endswith("_old"):
                added_rows = added_rows.drop(columns=[column])
            # Renaming the new
            if column.endswith("_new"):
                added_rows = added_rows.rename(columns={column: column.rstrip("_new")})
        diff["added_rows"] = added_rows

        # Finding deleted rows
        deleted_rows = merged_rows[merged_rows["_merge"] == "right_only"].drop(
            columns=["_merge"]
        )
        # Removing the new
        for column in deleted_rows.columns:
            if column.endswith("_new"):
                deleted_rows = deleted_rows.drop(columns=[column])
            # Renaming the old
            deleted_rows = deleted_rows.rename(columns={column: column.rstrip("_old")})
        diff["deleted_rows"] = deleted_rows

        # Finding modified rows
        common_rows_match_field_list = merged_rows[merged_rows["_merge"] == "both"][
            match_field
        ].to_list()

        # Looking at the rows that are existing in both the old and new layers so that we can compare them
        common_rows_new = new_df[new_df[match_field].isin(common_rows_match_field_list)]
        common_rows_old = old_df[old_df[match_field].isin(common_rows_match_field_list)]

        # Compare common columns attributes
        merged_common_rows = common_rows_new.merge(
            common_rows_old,
            on=None,
            how="outer",
            indicator=True,
        )

        modified_rows = merged_common_rows[
            merged_common_rows["_merge"] == "left_only"
        ].drop(columns=["_merge"])
        diff["modified_rows"] = modified_rows

        return diff

    # ---------------------------------------------------------------------

    @property
    def full_extent(self):
        """
        The ``full_extent`` method retrieves the extent of the DataFrame.

        :return:
            A tuple

        >>> df.spatial.full_extent
        (-118, 32, -97, 33)

        """
        ge = self._data[self.name].geom.extent
        q = ge.notnull()
        data = ge[q].tolist()
        array = np.array(data)
        return (
            float(array[:, 0][array[:, 0] != None].min()),
            float(array[:, 1][array[:, 1] != None].min()),
            float(array[:, 2][array[:, 2] != None].max()),
            float(array[:, 3][array[:, 3] != None].max()),
        )

    # ----------------------------------------------------------------------
    @property
    def area(self):
        """
        The ``area`` method retrieves the total area of the  ``GeoAccessor`` dataframe.

        :return:
            A float

        >>> df.spatial.area
        143.23427

        """
        return self._data[self.name].values.area.sum()

    # ----------------------------------------------------------------------
    @property
    def length(self):
        """
        The ``length`` method retrieves the total length of the DataFrame

        :return:
            A float

        >>> df.spatial.length
        1.23427

        """
        return self._data[self.name].values.length.sum()

    # ----------------------------------------------------------------------
    @property
    def centroid(self):
        """
        The ``centroid`` method retrieves the centroid of the dataframe

        :return:
            :class:`~arcgis.geometry.Geometry`

        >>> df.spatial.centroid
        (-14.23427, 39)

        """
        q = self._data[self.name].geom.centroid.isnull()
        df = pd.DataFrame(
            self._data[~q][self.name].geom.centroid.tolist(),
            columns=["x", "y"],
        )
        return df["x"].mean(), df["y"].mean()

    # ----------------------------------------------------------------------
    @property
    def true_centroid(self):
        """
        The ``true_centroid`` property retrieves the true centroid of the DataFrame

        :return:
            A :class:`~arcgis.geometry.Geometry` object

        >>> df.spatial.true_centroid
        (1.23427, 34)

        """
        q = self._data[self.name].notnull()
        df = pd.DataFrame(
            data=list(row.true_centroid for row in self._data[self.name][q]),
            columns=["x", "y"],
        ).mean()
        return df["x"], df["y"]

    # ----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """
        The ``geometry_type`` property retrieves a list of Geometry Types for the DataFrame.

        :return:
            A List
        """
        gt = self._data[self.name].geom.geometry_type
        return pd.unique(gt).tolist()

    # ----------------------------------------------------------------------
    @property
    def has_z(self):
        """
        The ``has_z`` property determines if the datasets have `Z` values

        :return:
            A boolean indicating `Z` values (True), or not (False)
        """
        return self._data[self.name].geom.has_z.any()

    # ----------------------------------------------------------------------
    @property
    def has_m(self):
        """
        The ``has_m`` property determines if the datasets have `M` values

        :return:
            A boolean indicating `M` values (True), or not (False)
        """
        return self._data[self.name].geom.has_m.any()

    # ----------------------------------------------------------------------
    @property
    def bbox(self):
        """
        The ``bbox`` property retrieves the total length of the dataframe

        :return:
            :class:`~arcgis.geometry.Polygon`

        >>> df.spatial.bbox
        {'rings' : [[[1,2], [2,3], [3,3],....]], 'spatialReference' {'wkid': 4326}}
        """
        xmin, ymin, xmax, ymax = self.full_extent
        sr = self.sr
        if isinstance(sr, list) and len(sr) > 0:
            sr = sr[0]
        if xmin == xmax:
            xmin -= 0.001
            xmax += 0.001
        if ymin == ymax:
            ymin -= 0.001
            ymax += 0.001
        return _geometry.Geometry(
            {
                "rings": [
                    [
                        [xmin, ymin],
                        [xmin, ymax],
                        [xmax, ymax],
                        [xmax, ymin],
                        [xmin, ymin],
                    ]
                ],
                "spatialReference": dict(sr),
            }
        )

    # ----------------------------------------------------------------------
    def distance_matrix(self, leaf_size=16, rebuild=False):
        """
        The ``distance_matrix`` creates a k-d tree to calculate the nearest-neighbor problem.

        .. note::
            The ``distance_matrix`` method requires SciPy

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        leafsize                 Optional Integer. The number of points at which the algorithm
                                 switches over to brute-force. Default: 16.
        --------------------     --------------------------------------------------------------------
        rebuild                  Optional Boolean. If True, the current KDTree is erased. If false,
                                 any KD-Tree that exists will be returned.
        ====================     ====================================================================


        :return: scipy's KDTree class

        """
        _HASARCPY, _HASSHAPELY = self._check_geometry_engine()
        if _HASARCPY == False and _HASSHAPELY == False:
            return None
        if rebuild:
            self._kdtree = None
        if self._kdtree is None:
            try:
                from scipy.spatial import cKDTree as KDTree
            except ImportError:
                from scipy.spatial import KDTree
            xy = self._data[self.name].geom.centroid.tolist()
            self._kdtree = KDTree(data=xy, leafsize=leaf_size)
            return self._kdtree
        else:
            return self._kdtree

    # ----------------------------------------------------------------------
    def select(self, other):
        """
        The ``select`` operation performs a dataset wide **selection** by geometric
        intersection. A geometry or another Spatially enabled DataFrame
        can be given and ``select`` will return all rows that intersect that
        input geometry.  The ``select`` operation uses a spatial index to
        complete the task, so if it is not built before the first run, the
        function will build a quadtree index on the fly.

        .. note::
            The ``select`` method requires ArcPy or Shapely

        :return:
            A Pandas DataFrame (pd.DataFrame, spatially enabled)

        """
        from arcgis.features.geo._tools import select

        return select(sdf=self._data, other=other)

    # ----------------------------------------------------------------------
    def overlay(self, sdf, op="union"):
        """
        The ``overlay`` performs spatial operation operations on two spatially enabled dataframes.

        .. note::
            The ``overlay`` method requires ArcPy or Shapely

        =========================    =========================================================
        **Parameter**                 **Description**
        -------------------------    ---------------------------------------------------------
        sdf                          Required Spatially Enabled DataFrame. The geometry to
                                     perform the operation from.
        -------------------------    ---------------------------------------------------------
        op                           Optional String. The spatial operation to perform.  The
                                     allowed value are: union, erase, identity, intersection.
                                     `union` is the default operation.
        =========================    =========================================================

        :return:
            A Spatially enabled DataFrame (pd.DataFrame)

        """
        from arcgis.features.geo._tools import overlay

        return overlay(sdf1=self._data, sdf2=sdf, op=op.lower())

    # ----------------------------------------------------------------------
    def relationship(self, other, op, relation=None):
        """
        The ``relationship`` method allows for dataframe to dataframe comparison using
        spatial relationships.

        .. note::
            The return is a Pandas DataFrame (pd.DataFrame) that meet the operations' requirements.

        =========================    =========================================================
        **Parameter**                 **Description**
        -------------------------    ---------------------------------------------------------
        other                        Required Spatially Enabled DataFrame. The geometry to
                                     perform the operation from.

        -------------------------    ---------------------------------------------------------
        op                           Optional String. The spatial operation to perform.  The
                                     allowed value are: contains,crosses,disjoint,equals,
                                     overlaps,touches, or within.

                                     - contains - Indicates if the base geometry contains the comparison geometry.
                                     - crosses -  Indicates if the two geometries intersect in a geometry of a lesser shape type.
                                     - disjoint - Indicates if the base and comparison geometries share no points in common.
                                     - equals - Indicates if the base and comparison geometries are of the same shape type and define the same set of points in the plane. This is a 2D comparison only; M and Z values are ignored.
                                     - overlaps - Indicates if the intersection of the two geometries has the same shape type as one of the input geometries and is not equivalent to either of the input geometries.
                                     - touches - Indicates if the boundaries of the geometries intersect.
                                     - within - Indicates if the base geometry contains the comparison geometry.
                                     - intersect - Indicates if the base geometry has an intersection of the other geometry.

                                     Note - contains and within will lead to same results when performing spatial operations.
        -------------------------    ---------------------------------------------------------
        relation                     Optional String.  The spatial relationship type.  The
                                     allowed values are: BOUNDARY, CLEMENTINI, and PROPER.

                                     + BOUNDARY - Relationship has no restrictions for interiors or boundaries.
                                     + CLEMENTINI - Interiors of geometries must intersect. This is the default.
                                     + PROPER - Boundaries of geometries must not intersect.

                                     This only applies to contains,
        =========================    =========================================================

        :return:
            Spatially enabled DataFrame (pd.DataFrame)


        """
        from ._tools import contains, crosses, disjoint
        from ._tools import equals, overlaps, touches
        from ._tools import within

        _ops_allowed = {
            "contains": contains,
            "crosses": crosses,
            "disjoint": disjoint,
            "intersect": disjoint,
            "equals": equals,
            "overlaps": overlaps,
            "touches": touches,
            "within": contains,
        }

        if not op.lower() in _ops_allowed.keys():
            raise ValueError("Invalid `op`. Please use a proper operation.")

        if op.lower() in ["contains", "within"]:
            fn = _ops_allowed[op.lower()]
            return fn(sdf=self._data, other=other, relation=relation)
        elif op.lower() in ["intersect"]:
            fn = _ops_allowed[op.lower()]
            return fn(sdf=self._data, other=other) == False
        else:
            fn = _ops_allowed[op.lower()]
            return fn(sdf=self._data, other=other)

    # ----------------------------------------------------------------------
    def voronoi(self):
        """
        The ``voronoi`` method generates a voronoi diagram on the whole dataset.

        .. note::
            If the :class:`~arcgis.geometry.Geometry` object is not a `:class:`~arcgis.geometry.Point` then the
            centroid is used for the geometry.  The result is a :class:`~arcgis.geometry.Polygon` `GeoArray` Series
            that matches 1:1 to the original dataset.

        .. note::
            The ``voronoi`` method requires SciPy

        :return:
            A Pandas Series (pd.Series)
        """
        _HASARCPY, _HASSHAPELY = self._check_geometry_engine()
        if _HASARCPY == False and _HASSHAPELY == False:
            return None
        radius = max(
            abs(self.full_extent[0] - self.full_extent[2]),
            abs(self.full_extent[1] - self.full_extent[3]),
        )
        from ._array import GeoArray
        from scipy.spatial import Voronoi

        xy = self._data[self.name].geom.centroid
        vor = Voronoi(xy.tolist())
        if vor.points.shape[1] != 2:
            raise ValueError("Supports 2-D only.")
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        # Construct a map containing all ridges for a
        # given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))
        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue
            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue
                # Compute the missing endpoint of an
                # infinite ridge
                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            # Sort region counterclockwise.
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())
        sr = self.sr
        return pd.Series(
            GeoArray(
                [
                    _geometry.Geometry(
                        {
                            "rings": [[new_vertices[l] for l in r]],
                            "spatialReference": sr,
                        }
                    ).buffer(0)
                    for r in new_regions
                ]
            )
        )

    # ----------------------------------------------------------------------
    def project(self, spatial_reference, transformation_name=None):
        """
        The ``project`` method reprojects the who dataset into a new :class:`~arcgis.geometry.SpatialReference`.
        This is an inplace operation meaning that it will update the defined geometry column from the ``set_geometry``.

        .. note::
            The ``project`` method requires ArcPy or pyproj v4

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required :class:`~arcgis.geometry.SpatialReference`. The new spatial reference.
                                 This can be a SpatialReference object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Optional String. The ``geotransformation`` name.
        ====================     ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        HASPYPROJ = True
        try:
            import importlib

            i = importlib.util.find_spec("pyproj")
            if i is None:
                raise ImportError("Cannot find pyproj.")
        except ImportError:
            HASPYPROJ = False
        try:
            if isinstance(spatial_reference, (int, str)) and HASARCPY:
                import arcpy

                spatial_reference = arcpy.SpatialReference(spatial_reference)
                vals = self._data[self.name].values.project_as(
                    **{
                        "spatial_reference": spatial_reference,
                        "transformation_name": transformation_name,
                    }
                )
                self._data[self.name] = vals
                return True
            elif isinstance(spatial_reference, _geometry.SpatialReference) and HASARCPY:
                vals = self._data[self.name].values.project_as(
                    **{
                        "spatial_reference": spatial_reference.as_arcpy,
                        "transformation_name": transformation_name,
                    }
                )
                self._data[self.name] = vals
                return True
            elif isinstance(spatial_reference, dict) and HASARCPY:
                spatial_reference = _geometry.SpatialReference(
                    spatial_reference
                ).as_arcpy
                vals = self._data[self.name].values.project_as(
                    **{
                        "spatial_reference": spatial_reference,
                        "transformation_name": transformation_name,
                    }
                )
                self._data[self.name] = vals
                return True

            elif isinstance(spatial_reference, (int, str)) and HASPYPROJ:
                vals = self._data[self.name].values.project_as(
                    **{
                        "spatial_reference": spatial_reference,
                        "transformation_name": transformation_name,
                    }
                )
                self._data[self.name] = vals
                return True
            else:
                return False
        except Exception as e:
            raise Exception(e)

    def sanitize_column_names(
        self,
        convert_to_string=True,
        remove_special_char=True,
        inplace=False,
        use_snake_case=True,
    ):
        """
        The ``sanitize_column_names`` cleans column names by converting them to string, removing special characters,
        renaming columns without column names to ``noname``, renaming duplicates with integer suffixes and switching
        spaces or Pascal or camel cases to Python's favored snake_case style.

        Snake_casing gives you consistent column names, no matter what the flavor of your backend database is
        when you publish the DataFrame as a Feature Layer in your web GIS.

        ==============================     ====================================================================
        **Parameter**                       **Description**
        ------------------------------     --------------------------------------------------------------------
        convert_to_string                  Optional Boolean. Default is True. Converts column names to string
        ------------------------------     --------------------------------------------------------------------
        remove_special_char                Optional Boolean. Default is True. Removes any characters in column
                                           names that are not numeric or underscores. This also ensures column
                                           names begin with alphabets by removing numeral prefixes.
        ------------------------------     --------------------------------------------------------------------
        inplace                            Optional Boolean. Default is False. If True, edits the DataFrame
                                           in place and returns Nothing. If False, returns a new DataFrame object.
        ------------------------------     --------------------------------------------------------------------
        use_snake_case                     Optional Boolean. Default is True. Makes column names lower case,
                                           and replaces spaces between words with underscores. If column names
                                           are in PascalCase or camelCase, it replaces them to snake_case.
        ==============================     ====================================================================

        :return:
            pd.DataFrame object if inplace= ``False`` . Else ``None`` .
        """

        return _sanitize_column_names(
            self,
            convert_to_string,
            remove_special_char,
            inplace,
            use_snake_case,
        )
