"""
The ``Filters`` modules contain functions to filter query results by a spatial relationship with another
:class:`~arcgis.geometry.Geometry` object. The ``Filters`` module is used when querying feature layers and imagery
layers.
"""
from typing import Union
from arcgis.geometry._types import Geometry, SpatialReference


# esriSpatialRelIntersects | esriSpatialRelContains | esriSpatialRelCrosses | esriSpatialRelEnvelopeIntersects | \
# esriSpatialRelIndexIntersects | esriSpatialRelOverlaps | esriSpatialRelTouches | esriSpatialRelWithin


def _filter(geometry, sr, rel):
    if not isinstance(geometry, Geometry):
        geometry = Geometry(geometry)

    gt = {
        "point": "Point",
        "multipoint": "Multipoint",
        "polygon": "Polygon",
        "polyline": "Polyline",
        "envelope": "Envelope",
    }

    filter = {
        "geometry": geometry,
        "geometryType": "esriGeometry" + gt[str(geometry.type).lower()],
        "spatialRel": rel,
    }

    if sr is None:
        if "spatialReference" in geometry:
            sr = geometry["spatialReference"]

    if sr is not None:
        filter["inSR"] = sr

    return filter


def intersects(geometry: Geometry, sr: Union[SpatialReference, dict, None] = None):
    """
    The ``intersects`` method filters results whose geometry intersects with the specified
    :class:`~arcgis.geometry.Geometry` object.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ================  ===============================================================================

    :return: A dictionary

    .. code-block:: python

        USAGE EXAMPLE: Select the gas lines that intersect a specific
        freeway feature, United States Interstate 15.

        from arcgis.geometry import Geometry
        from arcgis.geometry.filters import intersects

        # get a feature layer item from the gis
        >>> flyr_item = gis.content.search("Freeways*", "Feature Layer")[0]
        >>> freeway_lyr = flyr_item.layers[0]

        # assign a variable to the spatial reference
        >>> freeway_sr = freeway_lyr.properties.extent["spatialReference"]

        # select a filter feature to construct its geometry
        >>> rte15_fset = freeway_lyr.query(where="ROUTE_NUM = 'I15'")
        >>> rte15_geom_dict = rte15_fset.features[0].geometry
        >>> rte15_geom_dict['spatialReference'] = freeway_sr
        >>> rte15_geom = Geometry(rte15_geom_dict)

        # construct a geometry filter using the filter geometry
        >>> flyr_filter = intersects(rte15_geom, sr=freeway_sr)

        # query a feature layer for features that meet filter criteria
        >>> gas_lines_I15 = gas_line_lyr.query(geometry_filter=flyr_filter)
    """

    return _filter(geometry, sr, "esriSpatialRelIntersects")


def contains(geometry: Geometry, sr: Union[SpatialReference, dict, None] = None):
    """
    The ``contains`` method returns a feature if its shape is wholly contained within the search
    :class:`~arcgis.geometry.Geometry` object.

    .. note::
        Valid for all shape type combinations.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ================  ===============================================================================

    :return: A dictionary
    """
    return _filter(geometry, sr, "esriSpatialRelContains")


def crosses(geometry: Geometry, sr: Union[SpatialReference, dict, None] = None):
    """
    The ``crosses`` method retrieves a feature if the intersection of the interiors of the two shapes is not empty
    **and** has a lower dimension than the maximum dimension of the two shapes. Two lines that share an endpoint in
    common do not cross.

    .. note::
        Valid for Line/Line, Line/Area, :class:`~arcgis.geometry.MultiPoint` /Area, and
        :class:`~arcgis.geometry.MultiPoint` /Line shape type combinations.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ================  ===============================================================================

    :return: A dictionary
    """
    return _filter(geometry, sr, "esriSpatialRelCrosses")


def envelope_intersects(
    geometry: Geometry, sr: Union[SpatialReference, dict, None] = None
):
    """
    The ``envelope_intersects`` retrieves features if the :class:`~arcgis.geometry.Envelope` of the two shapes
    intersects.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ================  ===============================================================================

    :return: A dictionary
    """
    return _filter(geometry, sr, "esriSpatialRelEnvelopeIntersects")


def index_intersects(
    geometry: Geometry, sr: Union[SpatialReference, dict, None] = None
):
    """
    The ``index_intersects`` method retrieves a feature if the :class:`~arcgis.geometry.Envelope` of the query
    :class:`~arcgis.geometry.Geometry` intersects the index entry for the target geometry.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ================  ===============================================================================

    :return: A dictionary
    """
    return _filter(geometry, sr, "esriSpatialRelIndexIntersects")


def overlaps(geometry: Geometry, sr: Union[SpatialReference, dict, None] = None):
    """
    The ``overlaps`` method retrieves a feature if the intersection of the two shapes results in an object of the same
    dimension, but different from both of the shapes.

    .. note::
        This applies to Area/Area, Line/Line, and Multi-point/Multi-point shape type combinations.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ================  ===============================================================================

    :return: A dictionary
    """
    return _filter(geometry, sr, "esriSpatialRelOverlaps")


def touches(geometry: Geometry, sr: Union[SpatialReference, dict, None] = None):
    """
    The ``touches`` method retrieves a feature if the two shapes share a common boundary. However, the intersection of
    the interiors of the two shapes must be empty.

    .. note::
        In the Point/Line case, the point may touch an endpoint only of the line. Applies to all
        combinations except Point/Point.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ================  ===============================================================================

    :return: A dictionary
    """
    return _filter(geometry, sr, "esriSpatialRelTouches")


def within(geometry: Geometry, sr: Union[SpatialReference, dict, None] = None):
    """
     The ``within`` method retrieves a feature if its shape wholly contains the search
     :class:`~arcgis.geometry.Geometry`.

     .. note::
        Valid for all shape type combinations.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ================  ===============================================================================

    :return: A dictionary

    """
    return _filter(geometry, sr, "esriSpatialRelWithin")
