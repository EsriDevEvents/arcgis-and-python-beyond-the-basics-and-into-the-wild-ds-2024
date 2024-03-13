import pandas as pd
from arcgis.geometry import Geometry, Point, Polygon, Polyline, MultiPoint
from arcgis.features.geo._accessor import _is_geoenabled


def select(sdf, other):
    """
    Performs a select by location operation

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    sdf                          Required Spatially Enabled DataFrame. The dataframe to have the operation performed on.
    -------------------------    ---------------------------------------------------------
    other                        Required Spatially Enabled DataFrame or arcgis.Geometry.  This is the selecting data.
    =========================    =========================================================

    :return: pd.DataFrame (Spatially enabled DataFrame)

    """
    ud = pd.Series([False] * len(sdf))
    ud.index = sdf.index
    if isinstance(other, (Point, Polygon, Polyline, MultiPoint)):
        sindex = sdf.spatial.sindex()
        q1 = sindex.intersect(bbox=other.extent)
        sub = sdf.iloc[q1]
        dj = sub[sdf.spatial.name].geom.disjoint(other) == False
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
                dj = sub[sdf.spatial.name].geom.disjoint(g) == False
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
