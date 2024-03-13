from distutils.version import LooseVersion
import numbers
import operator
import warnings
import operator
import json

import numpy as np
import pandas as pd


from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.api.extensions import ExtensionArray

from collections.abc import Iterable
from arcgis.geometry import Geometry, Point, Polygon, Polyline

# -----------------------------------------------------------------------------
# pandas version checker
# -----------------------------------------------------------------------------
PANDAS_GE_024 = str(pd.__version__) >= LooseVersion("0.24.0")
PANDAS_GE_025 = str(pd.__version__) >= LooseVersion("0.25.0")
PANDAS_GE_10 = str(pd.__version__) >= LooseVersion("1")


# --------------------------------------------------------------------------
def _isna(value):
    """
    Check if scalar value is NA-like (None or np.nan).

    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    if value is None:
        return True
    elif isinstance(value, float) and np.isnan(value):
        return True
    else:
        return False


# --------------------------------------------------------------------------
def _unary_geo(op, left, *args, **kwargs):
    """
    Unary operation that returns new geometries

    **used for accessing properties on objects**

    :return: GeoArray
    """
    data = np.empty(len(left), dtype=object)
    data[:] = [getattr(geom, op, None) for geom in left]
    return GeoArray(data)


# --------------------------------------------------------------------------
def _unary_op(op, left, null_value=False):
    """
    Unary operation that returns a Series

    **used for accessing properties on objects**

    :return: pd.Series

    """
    data = np.empty(len(left), dtype=object)
    data[:] = [getattr(geom, op, null_value) for geom in left]
    return data


# --------------------------------------------------------------------------
def _binary_predicate(name, left, right, *args, **kwargs):
    """

    Binary operation performed on the GeoArray that returns only boolean ndarray.

    Supports:
    -  contains
    -  disjoint
    -  intersect
    -  touches
    -  crosses
    -  within
    -  overlaps
    -  equals

    Parameters
    ----------

         name: string
         left : GeoArray
         right: GeoArray or Geometry

    :return: np.array (should be dtype bool)

    """
    if isinstance(right, pd.Series):
        right = right.values
    if isinstance(right, Geometry):
        data = np.empty(len(left), dtype=bool)
        data[:] = [
            getattr(s, name)(right, *args, **kwargs) if s is not None else left.na_value
            for s in left
        ]
        return data
    elif isinstance(right, GeoArray):
        if len(left) != len(right):
            msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                len(left), len(right)
            )
            raise ValueError(msg)
        data = np.empty(len(left), dtype=bool)
        data[:] = [
            getattr(this_elem, name)(other_elem, *args, **kwargs)
            if not (this_elem is None or other_elem is None)
            else False
            for this_elem, other_elem in zip(left, right)
        ]
        return data
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


# --------------------------------------------------------------------------
def _binary_op(name, left, right=None, *args, **kwargs):
    """Binary operation on GeoArray that returns a ndarray of dtype object"""

    if isinstance(right, pd.Series):
        right = right.values
    null_value = None
    if right is None:
        data = np.empty(len(left), dtype=object)
        data[:] = [
            getattr(s, name)(*args, **kwargs) if s is not None else null_value
            for s in left
        ]
        return data
    elif isinstance(right, Geometry):
        data = np.empty(len(left), dtype=object)
        data[:] = [
            getattr(s, name)(right, *args, **kwargs) if s is not None else null_value
            for s in left
        ]
        return data
    elif isinstance(right, GeoArray):
        if len(left) != len(right):
            msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                len(left), len(right)
            )
            raise ValueError(msg)
        data = np.empty(len(left), dtype=object)
        data[:] = [
            getattr(this_elem, name)(other_elem, *args, **kwargs)
            if not (this_elem is None or other_elem is None)
            else null_value
            for this_elem, other_elem in zip(left, right)
        ]
        return data
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


# --------------------------------------------------------------------------
def _binary_op_geo(name, left, right=None, *args, **kwargs):
    """Binary operation on GeoArray that returns a GeoArray"""

    if isinstance(right, pd.Series):
        right = right.values
    null_value = None
    if right is None:
        data = np.empty(len(left), dtype=object)
        data[:] = [
            getattr(s, name)(*args, **kwargs) if s is not None else null_value
            for s in left
        ]
        return GeoArray(data)
    elif isinstance(right, Geometry):
        data = np.empty(len(left), dtype=object)
        data[:] = [
            getattr(s, name)(right, *args, **kwargs) if s is not None else null_value
            for s in left
        ]
        return GeoArray(data)
    elif isinstance(right, GeoArray):
        if len(left) != len(right):
            msg = "Lengths of inputs do not match. Left: {0}, Right: {1}".format(
                len(left), len(right)
            )
            raise ValueError(msg)
        data = np.empty(len(left), dtype=object)
        data[:] = [
            getattr(this_elem, name)(other_elem, *args, **kwargs)
            if not (this_elem is None or other_elem is None)
            else null_value
            for this_elem, other_elem in zip(left, right)
        ]
        return GeoArray(data)
    else:
        raise TypeError("Type not known: {0} vs {1}".format(type(left), type(right)))


# --------------------------------------------------------------------------


class GeoType(ExtensionDtype):
    type = Geometry
    name = "geometry"
    na_value = None
    # np.nan

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
            )

    @classmethod
    def construct_array_type(cls):
        return GeoArray


if PANDAS_GE_024:
    from pandas.api.extensions import register_extension_dtype

    register_extension_dtype(GeoType)


class GeoArray(ExtensionArray):
    """
    Class wrapping a numpy array of Shapely objects and
    holding the array-based implementations.
    """

    _dtype = GeoType()

    def __init__(self, values):
        if isinstance(values, self.__class__):
            data = values.data
        elif isinstance(values, pd.Series):
            data = values.values
        elif isinstance(values, (list, tuple)):
            data = np.array(values)
        elif isinstance(values, np.ndarray):
            data = values
        elif not isinstance(values, np.ndarray):
            raise TypeError("'data' should be array of geometry objects.")
        elif not values.ndim == 1:
            raise ValueError(
                "'data' should be a 1-dimensional array of geometry objects."
            )
        self.data = data
        self._validate_data()

    def _validate_data(self):
        data = self.data
        check = np.where(self.data != None)[0]
        if len(check) > 0:
            vindx = check[0]
            if isinstance(data[vindx], Geometry) == False:
                self.data[:] = [Geometry(d) if d else None for d in data]

                # Extra step for shapely, need to transform to correct Geometry type instance if not already
                geom = self.data[0]
                if (
                    not isinstance(geom, Point)
                    or not isinstance(geom, Polyline)
                    or not isinstance(geom, Polygon)
                ):
                    if "type" in geom and (
                        geom["type"] == "Point" or geom["type"] == "MultiPoint"
                    ):
                        self.data[:] = [Point(d) if d else None for d in data]
                    elif "type" in geom and geom["type"] == "Polyline":
                        self.data[:] = [Polyline(d) if d else None for d in data]
                    elif "type" in geom and (
                        geom["type"] == "Polygon" or geom["type"] == "MultiPolygon"
                    ):
                        self.data[:] = [Polygon(d) if d else None for d in data]

    def __arrow_array__(self, type=None):
        """converts the data to a pyarrow array"""
        import pyarrow

        return pyarrow.array([d.WKB for d in self.data if d], type=type)

    def __eq__(self, other: Geometry):
        """Checks if the Geometries are Equal"""
        if isinstance(other, Geometry):
            return self.equals(other)
        elif isinstance(other, GeoArray):
            return np.array_equal(self, other)
        else:
            raise ValueError(
                "Input must be a arcgis.geometry.Geometry or arcgis.features.geo.GeoArray"
            )

    def __ne__(self, other: Geometry):
        """Checks if the Geometries are Equal"""
        if isinstance(other, Geometry):
            return self.equals(other) == False
        else:
            raise ValueError("Input must be a arcgis.geometry.Geometry")

    def _formatting_values_backport(self):
        return np.array(self._format_values(), dtype="object")

    def _format_values(self):
        if self.data.ndim == 0:
            return ""
        return [_format(x) if x else None for x in self.data]

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self.shape[0]

    @classmethod
    def from_geometry(cls, data, copy=False):
        """ """
        if copy:
            data = data.copy()
        new = GeoArray([])
        new.data = np.array(data)
        return new

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self.data[idx]
        # array-like, slice
        if PANDAS_GE_10 and pd.api.types.is_list_like(idx):
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # to numpy array
            if not pd.api.types.is_array_like(idx):
                idx = pd.array(idx)
            dtype = idx.dtype
            if pd.api.types.is_bool_dtype(dtype):
                idx = pd.api.indexers.check_array_indexer(self, idx)
            elif pd.api.types.is_integer_dtype(dtype):
                idx = np.asarray(idx, dtype="int")
        if isinstance(idx, (Iterable, slice)):
            return GeoArray(self.data[idx])
        else:
            raise TypeError("Index type not supported", idx)

    def __setitem__(self, key, value):
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, GeoArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("cannot set a single element with an array")
            self.data[key] = value.data
        elif isinstance(value, Geometry) or _isna(value):
            if _isna(value):
                # internally only use None as missing value indicator
                # but accept others
                value = None
            if isinstance(key, (list, np.ndarray)):
                value_array = np.empty(1, dtype=object)
                value_array[:] = [value]
                self.data[key] = value_array
            else:
                self.data[key] = value
        elif isinstance(value, str) and value != "":
            value = Geometry(value)
            self.data[key] = value
        else:
            raise TypeError(
                "Value should be either a Geometry or None, got %s" % str(value)
            )

    # -------------------------------------------------------------------------
    # general array like compat
    # -------------------------------------------------------------------------

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return (self.size,)

    @property
    def ndim(self):
        return len(self.shape)

    def copy(self, *args, **kwargs):
        # still taking args/kwargs for compat with pandas 0.24
        return GeoArray(self.data.copy())

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = 0

        result = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        if fill_value == 0:
            result[result == 0] = None
        return GeoArray(result)

    def _fill(self, idx, value):
        """Fill index locations with value

        Value should be a Geometry
        """
        if not (isinstance(value, Geometry) or value is None):
            raise TypeError(
                "Value should be either a Geometry or None, got %s" % str(value)
            )
        # self.data[idx] = value
        self.data[idx] = np.array([value], dtype=object)
        return self

    def fillna(self, value=None, method=None, limit=None):
        """Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        Returns
        -------
        filled : ExtensionArray with NA/NaN filled
        """
        if method is not None:
            raise NotImplementedError("fillna with a method is not yet supported")

        if _isna(value):
            value = None
        elif not isinstance(value, Geometry):
            raise NotImplementedError(
                "fillna currently only supports filling with a scalar geometry"
            )

        mask = self.isna()
        new_values = self.copy()

        if mask.any():
            # fill with value
            new_values = new_values._fill(mask, value)

        return new_values

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, GeoType):
            if copy:
                return self.copy()
            else:
                return self
        elif pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(
            dtype
        ):
            return np.array([g.JSON for g in self.data])
        else:
            return np.array(self, dtype=dtype, copy=copy)

    @property
    def na_value(self):
        return self.dtype.na_value

    def isna(self):
        """
        Boolean NumPy array indicating if each value is missing
        """
        return np.array([g is self.na_value for g in self.data], dtype="bool")

    def unique(self):
        """Compute the ExtensionArray of unique values.

        Returns
        -------
        uniques : ExtensionArray
        """
        from pandas import factorize

        _, uniques = factorize(self)
        return uniques

    @property
    def nbytes(self):
        return self.data.nbytes

    # -------------------------------------------------------------------------
    # ExtensionArray specific
    # -------------------------------------------------------------------------

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : boolean, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        data = np.empty(len(scalars), dtype=object)
        data[:] = [Geometry(s) for s in scalars]
        return cls(data)

    def _values_for_factorize(self):
        """Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factoraization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.
        """
        return self, 0

    @classmethod
    def _from_factorized(cls, values):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.

        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        return cls(values)

    def _values_for_argsort(self):
        # type: () -> np.ndarray
        """Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort
        """
        # Note: this is used in `ExtensionArray.argsort`.
        raise TypeError("geometries are not orderable")

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """
        if boxed:
            return json.dumps
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple array

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray
        """
        data = np.concatenate([ga.data for ga in to_concat])
        return GeoArray(data)

    def _reduce(self, name, skipna=True, **kwargs):
        # including the base class version here (that raises by default)
        # because this was not yet defined in pandas 0.23
        if name == "any" or name == "all":
            return getattr(self.data, name)()
        raise TypeError(
            "cannot perform {name} with type {dtype}".format(
                name=name, dtype=self.dtype
            )
        )

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
        """
        return self.data

    # ----------------------------------------------------------------------
    @property
    def area(self):
        """returns the geometry area"""
        return _unary_op("area", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def as_arcpy(self):
        """returns the geometry area"""
        return _unary_op("as_arcpy", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def as_shapely(self):
        """returns the geometry area"""
        return _unary_op("as_shapely", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def centroid(self):
        """returns Geometry centroid"""
        return _unary_op("centroid", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def extent(self):
        """returns the extent of the geometry"""
        return _unary_op("extent", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def first_point(self):
        """
        The first coordinate point of the geometry for each entry.
        """
        return _unary_geo("first_point", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def geoextent(self):
        return _unary_op("geoextent", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def geometry_type(self):
        return _unary_op("geometry_type", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def has_z(self):
        return _unary_op("has_z", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def has_m(self):
        return _unary_op("has_m", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def hull_rectangle(self):
        return _unary_op("hull_rectangle", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def is_empty(self):
        return _unary_op("is_empty", self.data, False)

    # ----------------------------------------------------------------------
    @property
    def is_multipart(self):
        return _unary_op("is_multipart", self.data, False)

    # ----------------------------------------------------------------------
    @property
    def is_valid(self):
        return _binary_op(name="is_valid", left=self.data, right=None)

    # ----------------------------------------------------------------------
    @property
    def JSON(self):
        return _unary_op("JSON", self.data, "")

    # ----------------------------------------------------------------------
    @property
    def label_point(self):
        return _unary_geo("label_point", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def last_point(self):
        return _unary_geo("last_point", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def length(self):
        return _unary_op("length", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def length3D(self):
        return _unary_op("length3D", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def part_count(self):
        return _unary_op("part_count", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def point_count(self):
        return _unary_op("point_count", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def spatial_reference(self):
        return _unary_op("spatial_reference", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def true_centroid(self):
        return _unary_geo("true_centroid", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def WKB(self):
        return _unary_op("WKB", self.data, None)

    # ----------------------------------------------------------------------
    @property
    def WKT(self):
        return _unary_op("WKT", self.data, None)

    # ----------------------------------------------------------------------
    def angle_distance_to(self, second_geometry, method="GEODESIC"):
        """
        Returns a tuple of angle and distance to another point using a
        measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required Geometry.  A arcgis.Geometry object.
        ---------------     --------------------------------------------------------------------
        method              Optional String. PLANAR measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC, and
                            LOXODROME measurement types may be chosen as an alternative, if desired.
        ===============     ====================================================================

        :return: a tuple of angle and distance to another point using a measurement type.
        """
        return _binary_op(
            name="angle_distance_to",
            left=self.data,
            right=second_geometry,
            **{"method": method},
        )

    # ----------------------------------------------------------------------
    def boundary(self):
        """
        Constructs the boundary of the geometry.

        :return: arcgis.geometry.Polyline
        """

        return _binary_op_geo(name="boundary", left=self.data, right=None)

    # ----------------------------------------------------------------------
    def buffer(self, distance):
        """
        Constructs a polygon at a specified distance from the geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        distance            Required float. The buffer distance. The buffer distance is in the
                            same units as the geometry that is being buffered.
                            A negative distance can only be specified against a polygon geometry.
        ===============     ====================================================================

        :return: arcgis.geometry.Polygon
        """
        return _binary_op_geo(name="buffer", left=self.data, **{"distance": distance})

    # ----------------------------------------------------------------------
    def clip(self, envelope):
        """
        Constructs the intersection of the geometry and the specified extent.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        envelope            required tuple. The tuple must have (XMin, YMin, XMax, YMax) each value
                            represents the lower left bound and upper right bound of the extent.
        ===============     ====================================================================

        :return: output geometry clipped to extent

        """
        return _binary_op_geo(name="clip", left=self.data, **{"envelope": envelope})

    # ----------------------------------------------------------------------
    def contains(self, second_geometry, relation=None):
        """
        Indicates if the base geometry contains the comparison geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional string. The spatial relationship type.

                            + BOUNDARY - Relationship has no restrictions for interiors or boundaries.
                            + CLEMENTINI - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            + PROPER - Boundaries of geometries must not intersect.
        ===============     ====================================================================

        :return: boolean
        """
        return _binary_predicate(
            name="contains",
            left=self.data,
            right=second_geometry,
            **{"relation": relation},
        )

    # ----------------------------------------------------------------------
    def convex_hull(self):
        """
        Constructs the geometry that is the minimal bounding polygon such
        that all outer angles are convex.
        """
        return _binary_op_geo(name="convex_hull", left=self.data)

    # ----------------------------------------------------------------------
    def crosses(self, second_geometry):
        """
        Indicates if the two geometries intersect in a geometry of a lesser
        shape type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean

        """
        return _binary_predicate(name="crosses", left=self.data, right=second_geometry)

    # ----------------------------------------------------------------------
    def cut(self, cutter):
        """
        Splits this geometry into a part left of the cutting polyline, and
        a part right of it.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        cutter              Required Polyline. The cuttin polyline geometry
        ===============     ====================================================================

        :return: a list of two geometries

        """
        return _binary_op_geo(name="cut", left=self.data, right=cutter)

    # ----------------------------------------------------------------------
    def densify(self, method, distance, deviation):
        """
        Creates a new geometry with added vertices

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

        :return: arcgis.geometry.Geometry

        """
        return _binary_op_geo(
            name="densify",
            left=self.data,
            **{
                "method": method,
                "distance": distance,
                "deviation": deviation,
            },
        )

    # ----------------------------------------------------------------------
    def difference(self, second_geometry):
        """
        Constructs the geometry that is composed only of the region unique
        to the base geometry but not part of the other geometry. The
        following illustration shows the results when the red polygon is the
        source geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.geometry.Geometry

        """
        return _binary_op_geo(name="difference", left=self.data, right=second_geometry)

    # ----------------------------------------------------------------------
    def disjoint(self, second_geometry):
        """
        Indicates if the base and comparison geometries share no points in
        common.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean

        """
        return _binary_predicate(
            name="disjoint", left=self.data, right=second_geometry, **{}
        )

    # ----------------------------------------------------------------------
    def distance_to(self, second_geometry):
        """
        Returns the minimum distance between two geometries. If the
        geometries intersect, the minimum distance is 0.
        Both geometries must have the same projection.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: float

        """
        return _binary_op(
            name="distance_to", left=self.data, right=second_geometry, **{}
        )

    # ----------------------------------------------------------------------
    def equals(self, second_geometry):
        """
        Indicates if the base and comparison geometries are of the same
        shape type and define the same set of points in the plane. This is
        a 2D comparison only; M and Z values are ignored.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean


        """
        return _binary_predicate(name="equals", left=self.data, right=second_geometry)

    # ----------------------------------------------------------------------
    def generalize(self, max_offset):
        """
        Creates a new simplified geometry using a specified maximum offset
        tolerance.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        max_offset          Required float. The maximum offset tolerance.
        ===============     ====================================================================

        :return: arcgis.geometry.Geometry

        """
        return _binary_op_geo(
            name="generalize", left=self.data, **{"max_offset": max_offset}
        )

    # ----------------------------------------------------------------------
    def get_area(self, method, units=None):
        """
        Returns the area of the feature using a measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. PLANAR measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). GEODESIC,
                            GREAT_ELLIPTIC, LOXODROME, and PRESERVE_SHAPE measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Optional String. Areal unit of measure keywords: ACRES | ARES | HECTARES
                            | SQUARECENTIMETERS | SQUAREDECIMETERS | SQUAREINCHES | SQUAREFEET
                            | SQUAREKILOMETERS | SQUAREMETERS | SQUAREMILES |
                            SQUAREMILLIMETERS | SQUAREYARDS
        ===============     ====================================================================

        :return: float

        """
        return _binary_op(
            name="get_area",
            left=self.data,
            **{"method": method, "units": units},
        )

    # ----------------------------------------------------------------------
    def get_length(self, method, units):
        """
        Returns the length of the feature using a measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. PLANAR measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). GEODESIC,
                            GREAT_ELLIPTIC, LOXODROME, and PRESERVE_SHAPE measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Required String. Linear unit of measure keywords: CENTIMETERS |
                            DECIMETERS | FEET | INCHES | KILOMETERS | METERS | MILES |
                            MILLIMETERS | NAUTICALMILES | YARDS
        ===============     ====================================================================

        :return: float

        """
        return _binary_op(
            name="get_length",
            left=self.data,
            **{"method": method, "units": units},
        )

    # ----------------------------------------------------------------------
    def get_part(self, index=None):
        """
        Returns an array of point objects for a particular part of geometry
        or an array containing a number of arrays, one for each part.

        **requires arcpy**

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        index               Required Integer. The index position of the geometry.
        ===============     ====================================================================

        :return: arcpy.Array

        """
        return _binary_op(name="get_part", left=self.data, **{"index": index})

    # ----------------------------------------------------------------------
    def intersect(self, second_geometry, dimension=1):
        """
        Constructs a geometry that is the geometric intersection of the two
        input geometries. Different dimension values can be used to create
        different shape types. The intersection of two geometries of the
        same shape type is a geometry containing only the regions of overlap
        between the original geometries.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        dimension           Required Integer. The topological dimension (shape type) of the
                            resulting geometry.

                            + 1  -A zero-dimensional geometry (point or multipoint).
                            + 2  -A one-dimensional geometry (polyline).
                            + 4  -A two-dimensional geometry (polygon).

        ===============     ====================================================================

        :return: boolean array

        """
        return _binary_predicate(
            name="intersect",
            left=self.data,
            right=second_geometry,
            **{"dimension": dimension},
        )

    # ----------------------------------------------------------------------
    def measure_on_line(self, second_geometry, as_percentage=False):
        """
        Returns a measure from the start point of this line to the in_point.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional Boolean. If False, the measure will be returned as a
                            distance; if True, the measure will be returned as a percentage.
        ===============     ====================================================================

        :return: float

        """
        return _binary_op(
            name="measure_on_line",
            left=self.data,
            right=second_geometry,
            **{"as_percentage": as_percentage},
        )

    # ----------------------------------------------------------------------
    def overlaps(self, second_geometry):
        """
        Indicates if the intersection of the two geometries has the same
        shape type as one of the input geometries and is not equivalent to
        either of the input geometries.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean

        """
        return _binary_predicate(name="overlaps", left=self.data, right=second_geometry)

    # ----------------------------------------------------------------------
    def point_from_angle_and_distance(self, angle, distance, method="GEODESCIC"):
        """
        Returns a point at a given angle and distance in degrees and meters
        using the specified measurement type.

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

        :return: arcgis.geometry.Geometry


        """
        return _binary_op_geo(
            name="point_from_angle_and_distance",
            left=self.data,
            **{"angle": angle, "distance": distance, "method": method},
        )

    # ----------------------------------------------------------------------
    def position_along_line(self, value, use_percentage=False):
        """
        Returns a point on a line at a specified distance from the beginning
        of the line.

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

        :return: arcgis.gis.Geometry

        """
        return _binary_op_geo(
            name="position_along_line",
            left=self.data,
            **{"value": value, "use_percentage": use_percentage},
        )

    # ----------------------------------------------------------------------
    def project_as(self, spatial_reference, transformation_name=None):
        """
        Projects a geometry and optionally applies a geotransformation.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required SpatialReference. The new spatial reference. This can be a
                                 SpatialReference object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Required String. The geotransformation name.
        ====================     ====================================================================

        :return: arcgis.geometry.Geometry
        """
        return _binary_op_geo(
            name="project_as",
            left=self.data,
            **{
                "spatial_reference": spatial_reference,
                "transformation_name": transformation_name,
            },
        )

    # ----------------------------------------------------------------------
    def query_point_and_distance(self, second_geometry, use_percentage=False):
        """
        Finds the point on the polyline nearest to the in_point and the
        distance between those points. Also returns information about the
        side of the line the in_point is on as well as the distance along
        the line where the nearest point occurs.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional boolean - if False, the measure will be returned as
                            distance, True, measure will be a percentage
        ===============     ====================================================================

        :return: tuple

        """
        return _binary_op(
            name="query_point_and_distance",
            left=self.data,
            right=second_geometry,
            **{"use_percentage": use_percentage},
        )

    # ----------------------------------------------------------------------
    def segment_along_line(self, start_measure, end_measure, use_percentage=False):
        """
        Returns a Polyline between start and end measures. Similar to
        Polyline.positionAlongLine but will return a polyline segment between
        two points on the polyline instead of a single point.

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
                            False, start_measure and end_measure are used as a distance. For
                            percentages, the measures should be expressed as a double from 0.0
                            (0 percent) to 1.0 (100 percent).
        ===============     ====================================================================

        :return: Geometry

        """
        return _binary_op_geo(
            name="segment_along_line",
            left=self.data,
            **{
                "start_measure": start_measure,
                "end_measure": end_measure,
                "use_percentage": use_percentage,
            },
        )

    # ----------------------------------------------------------------------
    def snap_to_line(self, second_geometry):
        """
        Returns a new point based on in_point snapped to this geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry

        """
        return _binary_op_geo(
            name="snap_to_line", left=self.data, right=second_geometry
        )

    # ----------------------------------------------------------------------
    def symmetric_difference(self, second_geometry):
        """
        Constructs the geometry that is the union of two geometries minus the
        instersection of those geometries.

        The two input geometries must be the same shape type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry
        """
        return _binary_op_geo(
            name="symmetric_difference",
            left=self.data,
            right=second_geometry,
        )

    # ----------------------------------------------------------------------
    def touches(self, second_geometry):
        """
        Indicates if the boundaries of the geometries intersect.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean
        """
        return _binary_predicate(name="touches", left=self.data, right=second_geometry)

    # ----------------------------------------------------------------------
    def union(self, second_geometry):
        """
        Constructs the geometry that is the set-theoretic union of the input
        geometries.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry
        """
        return _binary_op_geo(name="union", left=self.data, right=second_geometry)

    # ----------------------------------------------------------------------
    def within(self, second_geometry, relation=None):
        """
        Indicates if the base geometry is within the comparison geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional String. The spatial relationship type.

                            - BOUNDARY  - Relationship has no restrictions for interiors or boundaries.
                            - CLEMENTINI  - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            - PROPER  - Boundaries of geometries must not intersect.

        ===============     ====================================================================

        :return: boolean

        """
        return _binary_predicate(
            name="within",
            left=self.data,
            right=second_geometry,
            **{"relation": relation},
        )


def _format(g):
    if g in {None, np.nan}:
        return ""
    return json.dumps(g)
