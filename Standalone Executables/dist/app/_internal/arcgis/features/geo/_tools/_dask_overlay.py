import math
from functools import reduce, lru_cache

import pandas as pd
import numpy as np

from arcgis.geometry._types import Geometry
from arcgis.features.geo._accessor import GeoAccessor
from arcgis.features.geo._accessor import GeoSeriesAccessor
from arcgis.features.geo._accessor import _is_geoenabled
from arcgis.features.geo._array import GeoArray

_HASARCPY, _HASSHAPELY = None, None


# ----------------------------------------------------------------------
@lru_cache(maxsize=10)
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
def _overlay_difference(df1, df2):
    """
    Overlay Difference operation used in overlay function
    """
    # Spatial Index to create intersections
    spatial_index = df2.spatial.sindex()
    bbox = df1[df1.spatial.name].geom.extent
    fn = lambda x: spatial_index.intersect(x)
    sidx = bbox.apply(fn)
    # Create differences
    new_g = []
    for geom, neighbours in zip(df1[df1.spatial.name], sidx):
        new = reduce(
            lambda x, y: x.difference(y).buffer(0),
            [geom] + list(df2[df2.spatial.name].iloc[neighbours]),
        )
        new_g.append(new)
    differences = pd.Series(GeoArray(new_g), index=df1.index)
    q = differences.isnull()
    geom_diff = differences[~q].copy()
    dfdiff = df1[~q].copy()
    dfdiff[dfdiff.spatial.name] = geom_diff
    return dfdiff


# --------------------------------------------------------------------------
def _symmetric_difference(df1, df2):
    """
    The symmetric difference, also known as the disjunctive union, of two
    sets is the set of elements which are in either of the sets and not in
    their intersection.

    :returns: pd.DataFrame (Spatially enabled DataFrame)

    """
    dfdiff1 = _overlay_difference(df1, df2)
    dfdiff2 = _overlay_difference(df2, df1)
    dfdiff1["__idx1"] = range(len(dfdiff1))
    dfdiff2["__idx2"] = range(len(dfdiff2))
    dfdiff1["__idx2"] = np.nan
    dfdiff2["__idx1"] = np.nan
    dfsym = dfdiff1.merge(
        dfdiff2, on=["__idx1", "__idx2"], how="outer", suffixes=["_1", "_2"]
    )
    geometry = dfsym.SHAPE_1.copy()
    geometry[dfsym.SHAPE_1.isnull()] = dfsym.loc[dfsym.SHAPE_1.isnull(), "SHAPE_2"]
    dfsym.drop(["SHAPE_1", "SHAPE_2"], axis=1, inplace=True)
    dfsym.reset_index(drop=True, inplace=True)
    dfsym = pd.DataFrame(data=dfsym)
    dfsym["SHAPE"] = geometry
    dfsym.spatial.name
    return dfsym


# --------------------------------------------------------------------------
def _overlay_intersection(df1, df2):
    """
    Overlay Intersection operation used in overlay function
    """

    def _ops_intersection(this, other):
        """performs the series vs series intersection"""
        lu = {"polygon": 4, "point": 1, "multipoint": 1, "line": 2, "polyline": 2}
        null_value = None
        gtype = lu[this.geom.geometry_type[0]]
        this, other = this.align(other)
        data = np.array(
            [
                getattr(this_elem, "intersect")(other_elem, gtype)
                if not this_elem.is_empty | other_elem.is_empty
                else null_value
                for this_elem, other_elem in zip(this, other)
            ]
        )

        return pd.Series(GeoArray(data), index=this.index)

    # Spatial Index to create intersections
    spatial_index = df2.spatial.sindex()
    bbox = df1[df1.spatial.name].geom.extent
    fn = lambda x: spatial_index.intersect(x)
    sidx = bbox.apply(fn)
    # Create pairs of geometries in both dataframes to be intersected
    nei = []
    for i, j in enumerate(sidx):
        for k in j:
            nei.append([i, k])
    if nei != []:
        pairs = pd.DataFrame(nei, columns=["__idx1", "__idx2"])
        left = df1[df1.spatial.name].take(pairs["__idx1"].values)
        left.reset_index(drop=True, inplace=True)
        right = df2[df2.spatial.name].take(pairs["__idx2"].values)
        right.reset_index(drop=True, inplace=True)
        intersections = _ops_intersection(left, right).geom.buffer(0)
        intersections.name = "intersections"

        # only keep actual intersecting geometries
        pairs_intersect = pairs[~intersections.isnull()]
        geom_intersect = intersections[~intersections.isnull()]

        # merge data for intersecting geometries
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        dfinter = pairs_intersect.merge(
            df1.drop(df1.spatial.name, axis=1), left_on="__idx1", right_index=True
        )
        dfinter = dfinter.merge(
            df2.drop(df2.spatial.name, axis=1),
            left_on="__idx2",
            right_index=True,
            suffixes=["_1", "_2"],
        )
        dfinter["SHAPE"] = geom_intersect
        return dfinter
    else:
        return pd.DataFrame(
            data=[],
            columns=list(set(df1.columns).union(df2.columns)) + ["__idx1", "__idx2"],
        )


def _overlay_intersection2(df1, df2):
    """
    Overlay Intersection operation used in overlay function
    """

    def _ops_intersection(this, other):
        """performs the series vs series intersection"""
        lu = {"polygon": 4, "point": 1, "multipoint": 1, "line": 2, "polyline": 2}
        null_value = None
        gtype = lu[this.geom.geometry_type[0]]
        this, other = this.align(other)
        data = np.array(
            [
                getattr(this_elem, "intersect")(other_elem, gtype)
                if not this_elem.is_empty | other_elem.is_empty
                else null_value
                for this_elem, other_elem in zip(this, other)
            ]
        )
        return pd.Series(GeoArray(data), index=this.index)

    # Spatial Index to create intersections
    spatial_index = df2.spatial.sindex()
    bbox = df1[df1.spatial.name].geom.extent
    fn = lambda x: spatial_index.intersect(x)
    sidx = bbox.apply(fn)
    # Create pairs of geometries in both dataframes to be intersected
    nei = []
    for i, j in enumerate(sidx):
        for k in j:
            nei.append([i, k])
    if nei != []:
        pairs = pd.DataFrame(nei, columns=["__idx1", "__idx2"])
        left = df1[df1.spatial.name].values.take(pairs["__idx1"].values)
        left = pd.Series(left)
        left.reset_index(drop=True, inplace=True)
        right = df2[df2.spatial.name].values.take(pairs["__idx2"].values)
        right.reset_index(drop=True, inplace=True)
        intersections = _ops_intersection(left, right).geom.buffer(0)
        intersections.name = "intersections"
        # only keep actual intersecting geometries
        pairs_intersect = pairs[~intersections.isnull()]
        geom_intersect = intersections[~intersections.isnull()]
        # merge data for intersecting geometries
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        dfinter = pairs_intersect.merge(
            df1.drop(df1.spatial.name, axis=1), left_on="__idx1", right_index=True
        )
        dfinter = dfinter.merge(
            df2.drop(df2.spatial.name, axis=1),
            left_on="__idx2",
            right_index=True,
            suffixes=["_1", "_2"],
        )
        dfinter["SHAPE"] = geom_intersect
        return dfinter
    else:
        return pd.DataFrame(
            data=[],
            columns=list(set(df1.columns).union(df2.columns)) + ["__idx1", "__idx2"],
        )


# --------------------------------------------------------------------------
def _erase(df1, df2):
    """
    This overlay operation erases geometries by the second SeDF object.

    """
    # Spatial Index to create intersections
    spatial_index = df2.sindex
    bbox = df1.geometry.apply(lambda x: x.bounds)
    sidx = bbox.apply(lambda x: list(spatial_index.intersection(x)))
    # Create differences
    new_g = []
    for geom, neighbours in zip(df1.geometry, sidx):
        new = reduce(
            lambda x, y: x.difference(y).buffer(0),
            [geom] + list(df2.geometry.iloc[neighbours]),
        )
        new_g.append(new)
    differences = pd.Series(new_g, index=df1.index)
    geom_diff = differences[~differences.is_empty].copy()
    dfdiff = df1[~differences.is_empty].copy()
    dfdiff[dfdiff._geometry_column_name] = geom_diff
    return dfdiff


def _overlay_union(df1, df2):
    """
    Overlay Union operation used in overlay function
    """
    dfinter = _overlay_intersection(df1, df2)
    dfsym = _symmetric_difference(df1, df2)
    dfunion = pd.concat([dfinter, dfsym], ignore_index=True, sort=False)
    columns = list(dfunion.columns)
    return dfunion.reindex(columns=columns)


def overlay_dask(sdf1, sdf2, op="union"):
    """
    Perform spatial overlay operations between two polygons Spatially
    Enabled DataFrames.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf1                         Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    sdf2                         Required Spatially Enabled DataFrame. The geometry to perform the operation from.
    -------------------------    ---------------------------------------------------------
    op                           Optional String. The spatial operation to perform.  The
                                 allowed value are: union, erase, identity, intersection
    =========================    =========================================================

    :returns: Spatially enabled DataFrame (pd.DataFrame)

    """
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",
        "erase",
    ]

    op = str(op).lower()

    _hasao, _hasshp = _check_geometry_engine()

    if (
        _hasao == False
        and _hasshp
        and sdf1.spatial.geometry_type != ["polygon"]
        and sdf2.spatial.geometry_type != ["polygon"]
    ):
        raise ValueError(
            ("Using shapely's geometry engine only " "support Polygon geometries.")
        )

    if (_hasao or _hasshp) and op in allowed_hows:
        if op in ["union", "identity"]:
            return _overlay_union(sdf1, sdf2)
        elif op in ["difference", "erase"]:
            return _overlay_difference(sdf1, sdf2)
        elif op == "intersection":
            return _overlay_intersection(sdf1, sdf2)
        elif op == "symmetric_difference":
            return _symmetric_difference(sdf1, sdf2)
    return None
