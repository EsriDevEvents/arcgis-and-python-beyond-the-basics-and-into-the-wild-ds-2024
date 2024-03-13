"""
Allows for dataset to dataset comparisons by passing in DataFrames or
Geometries.
"""
import math
from functools import reduce

import pandas as pd
import numpy as np

from arcgis.geometry._types import Geometry, Point, Polygon, Polyline, MultiPoint
from arcgis.features.geo._accessor import GeoAccessor
from arcgis.features.geo._accessor import GeoSeriesAccessor
from arcgis.features.geo._accessor import _is_geoenabled
from arcgis.features.geo._array import GeoArray

_HASARCPY, _HASSHAPELY = None, None


# ----------------------------------------------------------------------
def _check_geometry_engine():
    """checks if the geometry engine exists"""
    global _HASARCPY
    global _HASSHAPELY
    if _HASARCPY is None:
        try:
            import arcpy

            _HASARCPY = True
        except:
            _HASARCPY = False
    if _HASSHAPELY is None:
        try:
            import shapely

            _HASSHAPELY = True
        except:
            _HASSHAPELY = False
    return _HASARCPY, _HASSHAPELY


# --------------------------------------------------------------------------
def contains(sdf, other, relation="CLEMENTINI"):
    """
    Indicates if the base geometry contains the comparison geometry.

    `contains` is the opposite of `within`.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf                          Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    other                        Required Spatially Enabled DataFrame or arcgis.Geometry.  This is the selecting data.
    -------------------------    ---------------------------------------------------------
    relation                     Optional String.  The spatial relationship type.  The
                                 allowed values are: BOUNDARY, CLEMENTINI, and PROPER.

                                 BOUNDARY - Relationship has no restrictions for interiors
                                            or boundaries.
                                 CLEMENTINI - Interiors of geometries must intersect. This
                                              is the default.
                                 PROPER - Boundaries of geometries must not intersect.
    =========================    =========================================================

    :return: pd.DataFrame (Spatially enabled DataFrame)

    """
    global _HASARCPY, _HASSHAPELY

    if _HASARCPY == False and _HASSHAPELY == False:
        return None

    ud = pd.Series([False] * len(sdf))
    if isinstance(other, (Point, Polygon, Polyline, MultiPoint)):
        sindex = sdf.spatial.sindex()
        q1 = sindex.intersect(bbox=other.extent)
        sub = sdf.iloc[q1]
        dj = sub[sdf.spatial.name].geom.contains(other, relation)
        dj.index = sub.index
        ud = ud | dj
        return sdf[ud]
    elif _is_geoenabled(other):
        sindex = sdf.spatial.sindex()
        name = other.spatial.name
        for index, seg in other.iterrows():
            g = seg[name]
            q1 = sindex.intersect(bbox=g.extent)
            sub = sdf.iloc[q1]
            if len(sub) > 0:
                dj = sub[sdf.spatial.name].geom.contains(g, relation)
                dj.index = sub.index
                ud = ud | dj
        return sdf[ud]
    else:
        raise ValueError(
            (
                "Invalid input, please verify that `other` "
                "is a Point, Polygon, Polyline, MultiPoint, "
                "or Spatially enabled DataFrame"
            )
        )
    return None


# --------------------------------------------------------------------------
def crosses(sdf, other):
    """

    Indicates if the two geometries intersect in a geometry of a lesser
    shape type.

    Two polylines cross if they share only points in common, at least one
    of which is not an endpoint. A polyline and an polygon cross if they
    share a polyline or a point (for vertical line) in common on the
    interior of the polygon which is not equivalent to the entire polyline.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf                          Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    other                        Required Spatially Enabled DataFrame or arcgis.Geometry.  This is the selecting data.
    =========================    =========================================================

    :return: pd.DataFrame (Spatially enabled DataFrame)

    """
    global _HASARCPY, _HASSHAPELY

    if _HASARCPY == False and _HASSHAPELY == False:
        return None

    ud = pd.Series([False] * len(sdf))
    if isinstance(other, (Point, Polygon, Polyline, MultiPoint)):
        sindex = sdf.spatial.sindex()
        q1 = sindex.intersect(bbox=other.extent)
        sub = sdf.iloc[q1]
        dj = sub[sdf.spatial.name].geom.crosses(other)
        dj.index = sub.index
        ud = ud | dj
        return sdf[ud]
    elif _is_geoenabled(other):
        sindex = sdf.spatial.sindex()
        name = other.spatial.name
        for index, seg in other.iterrows():
            g = seg[name]
            q1 = sindex.intersect(bbox=g.extent)
            sub = sdf.iloc[q1]
            if len(sub) > 0:
                dj = sub[sdf.spatial.name].geom.crosses(g)
                dj.index = sub.index
                ud = ud | dj
        return sdf[ud]
    else:
        raise ValueError(
            (
                "Invalid input, please verify that `other` "
                "is a Point, Polygon, Polyline, MultiPoint, "
                "or Spatially enabled DataFrame"
            )
        )
    return None


# --------------------------------------------------------------------------
def disjoint(sdf, other):
    """
    Indicates if the base and comparison geometries share no points in common.

    **Two geometries intersect if `disjoint` returns False.**

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf                          Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    other                        Required Spatially Enabled DataFrame or arcgis.Geometry.  This is the selecting data.
    =========================    =========================================================

    :return: pd.DataFrame (Spatially enabled DataFrame)

    """
    global _HASARCPY, _HASSHAPELY

    if _HASARCPY == False and _HASSHAPELY == False:
        return None

    ud = pd.Series([True] * len(sdf))
    if isinstance(other, (Point, Polygon, Polyline, MultiPoint)):
        sindex = sdf.spatial.sindex()
        q1 = sindex.intersect(bbox=other.extent)
        sub = sdf.iloc[q1]
        dj = sub[sdf.spatial.name].geom.disjoint(other)
        dj.index = sub.index
        ud = ~(~ud | ~dj)
        return sdf[ud]
    elif _is_geoenabled(other):
        sindex = sdf.spatial.sindex()
        name = other.spatial.name
        for index, seg in other.iterrows():
            g = seg[name]
            q1 = sindex.intersect(bbox=g.extent)
            sub = sdf.iloc[q1]
            if len(sub) > 0:
                dj = sub[sdf.spatial.name].geom.disjoint(g)
                dj.index = sub.index
                ud = ~(~ud | ~dj)
        return sdf[ud]
    else:
        raise ValueError(
            (
                "Invalid input, please verify that `other` "
                "is a Point, Polygon, Polyline, MultiPoint, "
                "or Spatially enabled DataFrame"
            )
        )
    return None


# --------------------------------------------------------------------------
def equals(sdf, other):
    """

    Indicates if the base and comparison geometries are of the same shape
    type and define the same set of points in the plane. This is a 2D
    comparison only; M and Z values are ignored.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf                          Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    other                        Required Spatially Enabled DataFrame or arcgis.Geometry.  This is the selecting data.
    =========================    =========================================================

    :return: pd.DataFrame (Spatially enabled DataFrame)

    """
    global _HASARCPY, _HASSHAPELY

    if _HASARCPY == False and _HASSHAPELY == False:
        return None

    ud = pd.Series([False] * len(sdf))
    if isinstance(other, (Point, Polygon, Polyline, MultiPoint)):
        sindex = sdf.spatial.sindex()
        q1 = sindex.intersect(bbox=other.extent)
        sub = sdf.iloc[q1]
        dj = sub[sdf.spatial.name].geom.equals(other)
        dj.index = sub.index
        ud = ud | dj
        return sdf[ud]
    elif _is_geoenabled(other):
        sindex = sdf.spatial.sindex()
        name = other.spatial.name
        for index, seg in other.iterrows():
            g = seg[name]
            q1 = sindex.intersect(bbox=g.extent)
            sub = sdf.iloc[q1]
            if len(sub) > 0:
                dj = sub[sdf.spatial.name].geom.equals(g)
                dj.index = sub.index
                ud = ud | dj
        return sdf[ud]
    else:
        raise ValueError(
            (
                "Invalid input, please verify that `other` "
                "is a Point, Polygon, Polyline, MultiPoint, "
                "or Spatially enabled DataFrame"
            )
        )
    return None


# --------------------------------------------------------------------------
def overlaps(sdf, other):
    """

    Indicates if the intersection of the two geometries has the same shape
    type as one of the input geometries and is not equivalent to either of
    the input geometries.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf                          Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    other                        Required Spatially Enabled DataFrame or arcgis.Geometry.  This is the selecting data.
    =========================    =========================================================

    :return: pd.DataFrame (Spatially enabled DataFrame)

    """
    global _HASARCPY, _HASSHAPELY

    if _HASARCPY == False and _HASSHAPELY == False:
        return None

    ud = pd.Series([False] * len(sdf))
    if isinstance(other, (Point, Polygon, Polyline, MultiPoint)):
        sindex = sdf.spatial.sindex()
        q1 = sindex.intersect(bbox=other.extent)
        sub = sdf.iloc[q1]
        dj = sub[sdf.spatial.name].geom.overlaps(other)
        dj.index = sub.index
        ud = ud | dj
        return sdf[ud]
    elif _is_geoenabled(other):
        sindex = sdf.spatial.sindex()
        name = other.spatial.name
        for index, seg in other.iterrows():
            g = seg[name]
            q1 = sindex.intersect(bbox=g.extent)
            sub = sdf.iloc[q1]
            if len(sub) > 0:
                dj = sub[sdf.spatial.name].geom.overlaps(g)
                dj.index = sub.index
                ud = ud | dj
        return sdf[ud]
    else:
        raise ValueError(
            (
                "Invalid input, please verify that `other` "
                "is a Point, Polygon, Polyline, MultiPoint, "
                "or Spatially enabled DataFrame"
            )
        )
    return None


# --------------------------------------------------------------------------
def touches(sdf, other):
    """

    Indicates if the boundaries of the geometries intersect.

    Two geometries touch when the intersection of the geometries is not
    empty, but the intersection of their interiors is empty. For example,
    a point touches a polyline only if the point is coincident with one of
    the polyline end points.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf                          Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    other                        Required Spatially Enabled DataFrame or arcgis.Geometry.  This is the selecting data.
    =========================    =========================================================

    :return: pd.DataFrame (Spatially enabled DataFrame)

    """
    global _HASARCPY, _HASSHAPELY

    if _HASARCPY == False and _HASSHAPELY == False:
        return None

    ud = pd.Series([False] * len(sdf))
    if isinstance(other, (Point, Polygon, Polyline, MultiPoint)):
        sindex = sdf.spatial.sindex()
        q1 = sindex.intersect(bbox=other.extent)
        sub = sdf.iloc[q1]
        dj = sub[sdf.spatial.name].geom.touches(other)
        dj.index = sub.index
        ud = ud | dj
        return sdf[ud]
    elif _is_geoenabled(other):
        sindex = sdf.spatial.sindex()
        name = other.spatial.name
        for index, seg in other.iterrows():
            g = seg[name]
            q1 = sindex.intersect(bbox=g.extent)
            sub = sdf.iloc[q1]
            if len(sub) > 0:
                dj = sub[sdf.spatial.name].geom.touches(g)
                dj.index = sub.index
                ud = ud | dj
        return sdf[ud]
    else:
        raise ValueError(
            (
                "Invalid input, please verify that `other` "
                "is a Point, Polygon, Polyline, MultiPoint, "
                "or Spatially enabled DataFrame"
            )
        )
    return None


# --------------------------------------------------------------------------
def within(sdf, other, relation="CLEMENTINI"):
    """

    Indicates if the base geometry is within the comparison geometry.

    `within` is the opposite operator of `contains`.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf                          Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    other                        Required Spatially Enabled DataFrame or arcgis.Geometry.  This is the selecting data.
    -------------------------    ---------------------------------------------------------
    relation                     Optional String.  The spatial relationship type.  The
                                 allowed values are: BOUNDARY, CLEMENTINI, and PROPER.

                                 BOUNDARY - Relationship has no restrictions for interiors
                                            or boundaries.
                                 CLEMENTINI - Interiors of geometries must intersect. This
                                              is the default.
                                 PROPER - Boundaries of geometries must not intersect.
    =========================    =========================================================

    :return: pd.DataFrame (Spatially enabled DataFrame)

    """
    global _HASARCPY, _HASSHAPELY

    if _HASARCPY == False and _HASSHAPELY == False:
        return None

    ud = pd.Series([False] * len(sdf))
    if isinstance(other, (Point, Polygon, Polyline, MultiPoint)):
        sindex = sdf.spatial.sindex()
        q1 = sindex.intersect(bbox=other.extent)
        sub = sdf.iloc[q1]
        dj = sub[sdf.spatial.name].geom.within(other, relation)
        dj.index = sub.index
        ud = ud | dj
        return sdf[ud]
    elif _is_geoenabled(other):
        sindex = sdf.spatial.sindex()
        name = other.spatial.name
        for index, seg in other.iterrows():
            g = seg[name]
            q1 = sindex.intersect(bbox=g.extent)
            sub = sdf.iloc[q1]
            if len(sub) > 0:
                dj = sub[sdf.spatial.name].geom.within(other, relation)
                dj.index = sub.index
                ud = ud | dj
        return sdf[ud]
    else:
        raise ValueError(
            (
                "Invalid input, please verify that `other` "
                "is a Point, Polygon, Polyline, MultiPoint, "
                "or Spatially enabled DataFrame"
            )
        )
    return None
