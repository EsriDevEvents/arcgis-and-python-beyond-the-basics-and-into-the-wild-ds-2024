from arcgis.geometry import Geometry

import numpy as np
from pandas.api.types import is_list_like


def to_geo(values):
    """Convert values to GeoArray

    Parameters
    ----------
    values : WKT, GeoJSON or Esri JSON in a list

    Returns
    -------
    addresses : GeoArray

    Examples
    --------
    Parse strings
    >>> to_geo(['{"x" : -118.15, "y" : 33.80, "spatialReference" : {"wkid" : 4326}}'])
    <GeoArray(['{"x" : -118.15, "y" : 33.80, "spatialReference" : {"wkid" : 4326}}'])>

    Or dictionaries
    >>> to_geo([{"x" : -118.15, "y" : 33.80, "spatialReference" : {"wkid" : 4326}}])
    <GeoArray(['{"x" : -118.15, "y" : 33.80, "spatialReference" : {"wkid" : 4326}}'])>

    Or Geometry Objects
    >>> to_geo([Geometry({"x" : -118.15, "y" : 33.80, "spatialReference" : {"wkid" : 4326}})])
    <GeoArray(['{"x" : -118.15, "y" : 33.80, "spatialReference" : {"wkid" : 4326}}'])>
    """
    from ._array import GeoArray

    if not is_list_like(values):
        values = [values]

    return GeoArray(_to_geo_array(values))


def _to_geo_array(values):
    from ._array import GeoArray, GeoType

    if isinstance(values, GeoArray):
        return values.data
    if not (isinstance(values, np.ndarray) and values.dtype == GeoType._record_type):
        values = [Geometry(v) for v in values]
    return np.asarray(values, dtype=GeoType._record_type)
