"""
The ``Functions`` module is used to take :class:`~arcgis.geometry.Geometry` types as parameters and return
:class:`~arcgis.geometry.Geometry` type results.
"""
from __future__ import annotations
from enum import Enum
import json
from typing import Any, Optional, Union
from arcgis.geometry import (
    Geometry,
    Point,
    MultiPoint,
    Polyline,
    Polygon,
    SpatialReference,
)
import arcgis.env
from arcgis.gis import GIS


class AreaUnits(Enum):
    """
    Represents the Supported Geometry Service Area Units Enumerations.
    Example: areas_and_lengths(polygons=[geom],area_unit=AreaUnits.ACRES)
    """

    UNKNOWNAREAUNITS = {"areaUnit": "esriUnknownAreaUnits"}
    SQUAREINCHES = {"areaUnit": "esriSquareInches"}
    SQUAREFEET = {"areaUnit": "esriSquareFeet"}
    SQUAREYARDS = {"areaUnit": "esriSquareYards"}
    ACRES = {"areaUnit": "esriAcres"}
    SQUAREMILES = {"areaUnit": "esriSquareMiles"}
    SQUAREMILLIMETERS = {"areaUnit": "esriSquareMillimeters"}
    SQUARECENTIMETERS = {"areaUnit": "esriSquareCentimeters"}
    SQUAREDECIMETERS = {"areaUnit": "esriSquareDecimeters"}
    SQUAREMETERS = {"areaUnit": "esriSquareMeters"}
    ARES = {"areaUnit": "esriAres"}
    HECTARES = {"areaUnit": "esriHectares"}
    SQUAREKILOMETERS = {"areaUnit": "esriSquareKilometers"}


class LengthUnits(Enum):
    """
    Represents the Geometry Service Length Units Enumerations
    Example: areas_and_lengths(polygons=[geom],length_unit=LengthUnits.FOOT)
    """

    BRITISH1936FOOT = 9095
    GOLDCOASTFOOT = 9094
    INTERNATIONALCHAIN = 9097
    INTERNATIONALLINK = 9098
    INTERNATIONALYARD = 9096
    STATUTEMILE = 9093
    SURVEYYARD = 109002
    FIFTYKMLENGTH = 109030
    ONEFIFTYKMLENGTH = 109031
    DECIMETER = 109005
    CENTIMETER = 1033
    MILLIMETER = 1025
    INTERNATIONALINCH = 109008
    USSURVEYINCH = 109009
    INTERNATIONALROD = 109010
    USSURVEYROD = 109011
    USNAUTICALMILE = 109012
    UKNAUTICALMILE = 109013
    METER = 9001
    GERMANMETER = 9031
    FOOT = 9002
    SURVEYFOOT = 9003
    CLARKEFOOT = 9005
    FATHOM = 9014
    NAUTICALMILE = 9030
    SURVEYCHAIN = 9033
    SURVEYLINK = 9034
    SURVEYMILE = 9035
    KILOMETER = 9036
    CLARKEYARD = 9037
    CLARKECHAIN = 9038
    CLARKELINK = 9039
    SEARSYARD = 9040
    SEARSFOOT = 9041
    SEARSCHAIN = 9042
    SEARSLINK = 9043
    BENOIT1895A_YARD = 9050
    BENOIT1895A_FOOT = 9051
    BENOIT1895A_CHAIN = 9052
    BENOIT1895A_LINK = 9053
    BENOIT1895B_YARD = 9060
    BENOIT1895B_FOOT = 9061
    BENOIT1895B_CHAIN = 9062
    BENOIT1895B_LINK = 9063
    INDIANFOOT = 9080
    INDIAN1937FOOT = 9081
    INDIAN1962FOOT = 9082
    INDIAN1975FOOT = 9083
    INDIANYARD = 9084
    INDIAN1937YARD = 9085
    INDIAN1962YARD = 9086
    INDIAN1975YARD = 9087
    FOOT1865 = 9070
    RADIAN = 9101
    DEGREE = 9102
    ARCMINUTE = 9103
    ARCSECOND = 9104
    GRAD = 9105
    GON = 9106
    MICRORADIAN = 9109
    ARCMINUTECENTESIMAL = 9112
    ARCSECONDCENTESIMAL = 9113
    MIL6400 = 9114


# -------------------------------------------------------------------------
def areas_and_lengths(
    polygons: Polygon,
    length_unit: str | LengthUnits,
    area_unit: str | AreaUnits,
    calculation_type: str,
    spatial_ref: int = 4326,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``areas_and_lengths`` function calculates areas and perimeter lengths
    for each :class:`~arcgis.geometry.Polygon` specified in the input array.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    polygons          The array of :class:`~arcgis.geometry.Polygon` whose areas and lengths are to be computed.
    ----------------  -------------------------------------------------------------------------------
    length_unit       The length unit in which the perimeters of
                      polygons will be calculated. If ``calculation_type``
                      is planar, then ``length_unit`` can be any esriUnits
                      constant (string or integer). If ``calculationType`` is
                      not planar, then ``length_unit`` must be a linear
                      esriUnits constant, such as `esriSRUnit_Meter`(i.e. `9001`|`LengthUnits.METER`) or
                      `esriSRUnit_SurveyMile`(i.e. `9035`|`LengthUnits.SURVEYMILE`). If ``length_unit`` is not
                      specified, the units are derived from ``spatial_ref``. If ``spatial_ref`` is not
                      specified as well, the units are in meters. For a list of
                      valid units, see `esriSRUnitType Constants` and
                      `esriSRUnit2Type Constants`.
    ----------------  -------------------------------------------------------------------------------
    area_unit         The area unit in which areas of polygons will be
                      calculated. If calculation_type is planar, then
                      area_unit can be any `esriAreaUnits` constant (dict or enum). If ``calculation_type`` is
                      not planar, then ``area_unit`` must be a `esriAreaUnits` constant such
                      as `AreaUnits.SQUAREMETERS` (i.e. `{"areaUnit": "esriSquareMeters"}`) or
                      `AreaUnits.SQUAREMILES` (i.e. `{"areaUnit": "esriSquareMiles"}`). If
                      ``area_unit`` is not specified, the units are derived
                      from ``spatial_ref``. If ``spatial_ref`` is not specified, then the units are in square
                      meters. For a list of valid units, see
                      `esriAreaUnits Constants`.
                      The list of valid esriAreaUnits constants include,
                      `esriSquareInches | esriSquareFeet |
                      esriSquareYards | esriAcres | esriSquareMiles |
                      esriSquareMillimeters | esriSquareCentimeters |
                      esriSquareDecimeters | esriSquareMeters | esriAres
                      | esriHectares | esriSquareKilometers.`
    ----------------  -------------------------------------------------------------------------------
    calculation_type  The type defined for the area and length calculation of the input geometries. The type can be one
                      of the following values:

                          1. planar - Planar measurements use 2D Euclidean distance to calculate area and length. This
                          should only be used if the area or length needs to be calculated in the given
                          :class:`~arcgis.geometry.SpatialReference`. Otherwise, use ``preserveShape``.

                          2. geodesic - Use this type if you want to calculate an area or length using only the vertices
                          of the :class:`~arcgis.geometry.Polygon` and define the lines between the points as geodesic
                          segments independent of the actual shape of the :class:`~arcgis.geometry.Polygon`. A geodesic
                          segment is the shortest path between two points on an ellipsoid.

                          3. preserveShape - This type calculates the area or length of the geometry on the surface of
                          the Earth ellipsoid. The shape of the geometry in its coordinate system is preserved.
    ----------------  -------------------------------------------------------------------------------
     future           Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        A JSON as dictionary, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

            >>> # Use case 1
            >>> areas_and_lengths(polygons =[polygon1, polygon2,...],
                                  length_unit = 9001,
                                  area_unit = {"areaUnit": "esriSquareMeters"},
                                  calculation_type = "planar")
            >>> # Use case 2
            >>> from arcgis.geometry import LengthUnits, AreaUnits
            >>> areas_and_lengths(polygons =[polygon1, polygon2,...],
                                  length_unit = LengthUnits.METER,
                                  area_unit = AreaUnits.SQUAREMETERS,
                                  calculation_type = "planar",
                                  future = True)
    """
    if gis is None:
        gis = arcgis.env.active_gis
    if isinstance(length_unit, LengthUnits):
        length_unit = length_unit.value
    if isinstance(area_unit, AreaUnits):
        area_unit = area_unit.value
    return gis._tools.geometry.areas_and_lengths(
        polygons,
        length_unit,
        area_unit,
        calculation_type,
        spatial_ref,
        future=future,
    )


# -------------------------------------------------------------------------
def auto_complete(
    polygons: Optional[list[Polygon]] = None,
    polylines: Optional[list[Polyline]] = None,
    spatial_ref: Optional[SpatialReference] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``auto_complete`` function simplifies the process of
    constructing new :class:`~arcgis.geometry.Polygon` objects that are adjacent to other polygons.
    It constructs polygons that fill in the gaps between existing
    polygons and a set of :class:`~arcgis.geometry.Polyline` objects.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    polygons          A List of :class:`~arcgis.geometry.Polygon` objects
    ----------------  -------------------------------------------------------------------------------
    polylines         A List of :class:`~arcgis.geometry.Polyline` objects
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries WKID
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        A :class:`~arcgis.geometry.Polygon` object, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.auto_complete(
        polygons, polylines, spatial_ref, future=future
    )


def buffer(
    geometries: list,
    in_sr: Union[int, dict[str, Any]],
    distances: float | list[float],
    unit: str | LengthUnits,
    out_sr: Optional[Union[int, dict[str, Any]]] = None,
    buffer_sr: Optional[float] = None,
    union_results: Optional[bool] = None,
    geodesic: Optional[bool] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``buffer`` function is performed on a geometry service resource
    The result of this function is a buffered :class:`~arcgis.geometry.Polygon` at the
    specified distances for the input :class:`~arcgis.geometry.Geometry` array.

    .. note::
        The options are available to union buffers and to use geodesic distance.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        The array of geometries to be buffered
    ----------------  -------------------------------------------------------------------------------
    in_sr             The well-known ID of the :class:`~arcgis.geometry.SpatialReference` or a spatial
                      reference JSON object for the input geometries.
    ----------------  -------------------------------------------------------------------------------
    distances         The distances that each of the input geometries is
                      buffered.
    ----------------  -------------------------------------------------------------------------------
    unit              The units for calculating each buffer distance. If unit
                      is not specified, the units are derived from ``bufferSR``. If
                      ``bufferSR`` is not specified, the units are derived from ``in_sr``.
    ----------------  -------------------------------------------------------------------------------
    out_sr            The well-known ID of the :class:`~arcgis.geometry.SpatialReference` or a
                      spatial reference JSON object for the output geometries.
    ----------------  -------------------------------------------------------------------------------
    buffer_sr         The well-known ID of the :class:`~arcgis.geometry.SpatialReference` or a
                      spatial reference JSON object for the buffer geometries.
    ----------------  -------------------------------------------------------------------------------
    union_results     A boolean. If True, all geometries buffered at a given
                      distance are unioned into a single (gis,possibly multipart)
                      :class:`~arcgis.geometry.Polygon`, and the unioned geometry is placed in the output
                      array. The default is False.
    ----------------  -------------------------------------------------------------------------------
    geodesic          Set geodesic to true to buffer the input geometries
                      using geodesic distance. Geodesic distance is the shortest
                      path between two points along the ellipsoid of the earth. If
                      geodesic is set to False, the 2D Euclidean distance is used
                      to buffer the input geometries.

                      .. note::
                        The default value depends on the `geometry type`, `unit` and `bufferSR`.
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        A list of :class:`~arcgis.geometry.Polygon` object, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

            >>> buffer(geometries =[geom1, geom2,...],
                       in_sr = "wkid_in",
                       unit = LengthUnits.METER,
                       out_sr = "wkid_out",
                       buffer_sr = "wkid_buffer",
                       union_results =True,
                       geodesic = True,
                       future = True)

    """
    if gis is None:
        gis = arcgis.env.active_gis
    if isinstance(unit, LengthUnits):
        unit = unit.value
    if isinstance(distances, list):
        distances = ",".join([str(d) for d in distances])
    return gis._tools.geometry.buffer(
        geometries,
        in_sr,
        distances,
        unit,
        out_sr,
        buffer_sr,
        union_results,
        geodesic,
        future=future,
    )


def convex_hull(
    geometries: Union[list[Polygon], list[Polyline], list[MultiPoint], list[Point]],
    spatial_ref: Optional[Union[int, dict[str, Any]]] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The `convex_hull` function is performed on a `Geometry Service
    resource <https://developers.arcgis.com/rest/services-reference/enterprise/geometry-service.htm>`_.
    It returns the minimum bounding shape that contains the input geometry. The
    input geometry can be a :class:`~arcgis.geometry.Point`, :class:`~arcgis.geometry.MultiPoint`,
    :class:`~arcgis.geometry.Polyline` , or :class:`~arcgis.geometry.Polygon` object.

    .. note::
        The convex hull is typically a polygon but can also be a polyline
        or point in degenerate cases.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        A list of :class:`~arcgis.geometry.Point`, :class:`~arcgis.geometry.MultiPoint`,
                      :class:`~arcgis.geometry.Polyline`, or :class:`~arcgis.geometry.Polygon` objects.
                      The structure of each geometry in the array is defined the same as the
                      `JSON geometry objects <https://developers.arcgis.com/documentation/common-data-types/geometry-objects.htm>`_
                      returned by the ArcGIS REST API.

                      .. note::
                          :class:`~arcgis.geometry.Geometry` objects can be obtained by querying a
                          :class:`~arcgis.features.FeatureLayer`, returning it as a Pandas
                          data frame, and then assigning variables to a geometry based on the row index.

                          .. code-block:: python

                              >>> flyr_item = gis.content.search("*", "Feature Layer")[0]

                              >>> flyr_df = flyr_item.query(where="1=1", as_df=True)
                              >>> geom0 = flyr_df.loc[0].SHAPE

    ----------------  -------------------------------------------------------------------------------
    spatial_ref       An integer value, or a :class:`~arcgis.geometry.SpatialReference` object
                      defined using the the Well-Known ID (`wkid`) of the Spatial Reference.

                      .. note:: See `Spatial Reference <https://developers.arcgis.com/documentation/common-data-types/geometry-objects.htm#GUID-DFF0E738-5A42-40BC-A811-ACCB5814BABC>`_
                          in the `Geometry objects` help, or `Using Spatial References <https://developers.arcgis.com/rest/services-reference/enterprise/using-spatial-references.htm>`_
                          for details on concepts and resources for finding specific `wkid` values.

                      .. code-block:: python

                          >>> geom_result = convex_hull(geometries=[geometry_object]
                                                        spatial_ref=<wkid>)

                      or

                      .. code-block:: python

                          >>> geom_result = convex_hull(geometries=[geometry_object],
                                                        spatial_ref={"wkid": <wkid>})

                      or

                      .. code-block:: python

                          >>> from arcgis.geometry import SpatialReference
                          >>> sr_obj_wkid = SpatialReference(<wkid>)

                          >>> geom_result = convex_hull(geometries=[geometry_object],
                                                        spatial_ref=sr_obj_wkid)

    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        A list containing the :class:`~arcgis.geometry.Geometry` object of the result, or  if ``future=True``,
        a :class:`~concurrent.futures.Future` object. Call ``result()`` on the `future` to get
        the response details.


    .. code-block:: python

        # Usage Example:

        >>> from arcgis.gis import GIS
        >>> from arcgis.geometry import convex_hull

        >>> gis = GIS(profile="your_organization_profile")

        >>> flyr_item = gis.content.get("<item_id for feature layer>")
        >>> flyr = flyr_item.layers[0]

        >>> df = flyr.query(where="OBJECTID=1", as_df=True)

        >>> geom1 = df.loc[0].SHAPE
        >>> hull_geom1 = convex_hull(geometries=[geom1],
                                     spatial_ref={"wkid": 2056})

        >>> hull_geom1[0]

        {'rings': [[[2664507.7925999984, 1212609.7138999999],
        .,
        .,
        [2664678.264199998, 1212618.6860999987],
        [2664507.7925999984, 1212609.7138999999]]],
        'spatialReference': {'wkid': {'wkid': 2056}}}
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.convex_hull(geometries, spatial_ref, future=future)


def cut(
    cutter: Polyline,
    target: Union[list[Polyline], list[Polygon]],
    spatial_ref: Optional[Union[int, dict[str, Any]]] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The `cut` function is performed on a :class:`~arcgis.geometry.Geometry` service resource. This
    function splits the target :class:`~arcgis.geometry.Polyline` or :class:`~arcgis.geometry.Polygon` where it is
    crossed by the cutter polyline.

    .. note::
        At 10.1 and later, this function calls simplify on the input
        cutter and target geometries.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    cutter            The :class:`~arcgis.geometry.Polyline` that will be used to divide the target
                      into pieces where it crosses the target.The spatial reference
                      of the polylines is specified by ``spatial_ref``.

                      .. note::
                        The structure of the
                        polyline is the same as the structure of the JSON polyline
                        objects returned by the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    target            The array of :class:`~arcgis.geometry.Polyline` or :class:`~arcgis.geometry.Polygon` to be cut.
                      The structure of the geometry is the same as the structure of the
                      JSON geometry objects returned by the ArcGIS REST API. The
                      spatial reference of the target geometry array is specified by
                      spatial_ref.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or a JSON
                      object for the output geometry
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        A List of :class:`~arcgis.geometry.Geometry` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.cut(cutter, target, spatial_ref, future=future)


def densify(
    geometries: Union[list[Polygon], list[Polyline], list[MultiPoint], list[Point]],
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    max_segment_length: Optional[float],
    length_unit: Optional[str] | Optional[LengthUnits],
    geodesic: bool = False,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``densify`` function is performed using the :class:`~arcgis.gis.GIS` geometry engine.
    This function densifies :class:`~arcgis.geometry.Geometry` objects by plotting :class:`~arcgis.geometry.Point`
    objects between existing vertices.


    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        An array of :class:`~arcgis.geometry.Point`, :class:`~arcgis.geometry.MultiPoint`,
                      :class:`~arcgis.geometry.Polyline`, or :class:`~arcgis.geometry.Polygon` objects.
                      The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       The ``well-known ID`` or a spatial reference JSON object for
                      the input :class:`~arcgis.geometry.Polyline` object.

                      .. note::
                        For a list of valid WKID values, see
                        Projected coordinate systems and Geographic coordinate systems.
    ----------------  -------------------------------------------------------------------------------
    max_segment_len   All segments longer than ``maxSegmentLength`` are
                      replaced with sequences of lines no longer than ``max_segment_length``.
    ----------------  -------------------------------------------------------------------------------
    length_unit       The length unit of ``max_segment_length``. If ``geodesic`` is
                      set to `false`, then the units are derived from ``spatial_ref``, and
                      ``length_unit`` is ignored. If ``geodesic`` is set to `true`, then
                      ``length_unit`` must be a linear unit. In a case where ``length_unit`` is
                      not specified and ``spatial_ref`` is a PCS, the units are derived from ``spatial_ref``.
                      In a case where ``length_unit`` is not specified and ``spatial_ref`` is a GCS,
                      then the units are meters.
    ----------------  -------------------------------------------------------------------------------
    geodesic          If geodesic is set to true, then geodesic distance is
                      used to calculate max_segment_length. Geodesic distance is the
                      shortest path between two points along the ellipsoid of the
                      earth. If geodesic is set to false, then 2D Euclidean distance
                      is used to calculate max_segment_length. The default is false.
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        A list of :class:`~arcgis.geometry.Geometry` object, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

            >>> densify(geometries =[geom1, geom2,...],
                        spatial_ref = "wkid",
                        max_segment_length = 100.0,
                        length_unit = LengthUnits.METER,
                        geodesic = True,
                        future = False)

    """
    if gis is None:
        gis = arcgis.env.active_gis
    if isinstance(length_unit, LengthUnits):
        length_unit = length_unit.value

    return gis._tools.geometry.densify(
        geometries,
        spatial_ref,
        max_segment_length,
        length_unit,
        geodesic,
        future=future,
    )


def difference(
    geometries: Union[list[Polygon], list[Polyline], list[MultiPoint], list[Point]],
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    geometry: Geometry,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``difference`` function is performed on a geometry service
    resource. This function constructs the set-theoretic difference
    between each element of an array of geometries and another geometry
    the so-called difference geometry. In other words, let B be the
    difference geometry. For each geometry, A, in the input geometry
    array, it constructs A-B.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        An array of :class:`~arcgis.geometry.Point`, :class:`~arcgis.geometry.MultiPoint`,
                      :class:`~arcgis.geometry.Polyline`, or :class:`~arcgis.geometry.Polygon` objects.
                      The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    geometry          A single geometry of any type and of a dimension equal
                      to or greater than the elements of geometries. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        A list of :class:`~arcgis.geometry.Geometry` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis

    return gis._tools.geometry.difference(
        geometries, spatial_ref, geometry, future=future
    )


def distance(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    geometry1: Geometry,
    geometry2: Geometry,
    distance_unit: str | LengthUnits | None = "",
    geodesic: bool = False,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``distance`` function is performed on a geometry service resource.
    It reports the `2D Euclidean` or `geodesic` distance between the two
    :class:`~arcgis.geometry.Geometry` objects.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry1         The :class:`~arcgis.geometry.Geometry` object from which the distance is
                      measured. The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    geometry2         The :class:`~arcgis.geometry.Geometry` object to which the distance is
                      measured. The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    distance_unit     Optional. One of :class:`~arcgis.geometry.functions.LengthUnits` enumeration
                      members. See Geometry Service
                      `distance <https://developers.arcgis.com/rest/services-reference/enterprise/distance.htm>`_
                      for full details.
    ----------------  -------------------------------------------------------------------------------
    geodesic          If ``geodesic`` is set to true, then the geodesic distance
                      between the ``geometry1`` and ``geometry2`` geometries is returned.
                      Geodesic distance is the shortest path between two points along
                      the ellipsoid of the earth. If ``geodesic`` is set to false or not
                      specified, the planar distance is returned. The default value is false.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known
                      ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        The 2D or geodesic distance between the two :class:`~arcgis.geometry.Geometry` objects, or
        a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    if isinstance(distance_unit, LengthUnits):
        distance_unit = distance_unit.value
    elif distance_unit in [None, ""]:
        distance_unit = ""
    return gis._tools.geometry.distance(
        spatial_ref,
        geometry1,
        geometry2,
        distance_unit,
        geodesic,
        future=future,
    )


def find_transformation(
    in_sr: Optional[Union[int, dict[str, Any]]],
    out_sr: Optional[Union[int, dict[str, Any]]],
    extent_of_interest: Optional[dict[str, Any]] = None,
    num_of_results: int = 1,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``find_transformations`` function is performed on a :class:`~arcgis.geometry.Geometry`
    service resource. This function returns a list of applicable
    geographic transformations you should use when projecting
    geometries from the input :class:`~arcgis.geometry.SpatialReference` to the output
    :class:`~arcgis.geometry.SpatialReference`. The transformations are in JSON format and are returned
    in order of most applicable to least applicable. Recall that a
    geographic transformation is not needed when the input and output
    spatial references have the same underlying geographic coordinate
    systems. In this case, findTransformations returns an empty list.

    .. note::
        Every returned geographic transformation is a forward
        transformation meaning that it can be used as-is to project from
        the input spatial reference to the output spatial reference. In the
        case where a predefined transformation needs to be applied in the
        reverse direction, it is returned as a forward composite
        transformation containing one transformation and a transformForward
        element with a value of false.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    in_sr             The well-known ID of the :class:`~arcgis.geometry.SpatialReference` or a spatial
                      reference JSON object for the input geometries.
    ----------------  -------------------------------------------------------------------------------
    out_sr            The well-known ID of the :class:`~arcgis.geometry.SpatialReference` or a
                      spatial reference JSON object for the output geometries.
    ----------------  -------------------------------------------------------------------------------
    ext_of_interest   The bounding box of the area of interest specified as a JSON envelope.If provided, the extent of
                      interest is used to return the most applicable geographic
                      transformations for the area.

                      .. note::
                        If a :class:`~arcgis.geometry.SpatialReference` is not
                        included in the JSON envelope, the ``in_sr`` is used for the
                        envelope.

    ----------------  -------------------------------------------------------------------------------
    num_of_results    The number of geographic transformations to
                      return. The default value is 1.

                      .. note::
                        If ``num_of_results`` has a value of -1, all applicable transformations are returned.
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        A List of geographic transformations, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.find_transformation(
        in_sr, out_sr, extent_of_interest, num_of_results, future=future
    )


def from_geo_coordinate_string(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    strings: list[str],
    conversion_type: Optional[str],
    conversion_mode: Optional[str] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``from_geo_coordinate_string`` function is performed on a :class:`~arcgis.geometry.Geometry`
    service resource. The function converts an array of well-known
    strings into xy-coordinates based on the conversion type and
    :class:`~arcgis.geometry.SpatialReference` supplied by the user. An optional conversion mode
    parameter is available for some conversion types. See :attr:`~arcgis.geometry.functions.to_geo_coordinate_strings`
    for more information on the opposite conversion.


    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    strings           An array of strings formatted as specified by conversion_type.
                      Syntax: [<string1>,...,<stringN>]
    ----------------  -------------------------------------------------------------------------------
    conversion-type   The conversion type of the input strings.

                      .. note::
                        Valid conversion types are:
                        `MGRS` - Military Grid Reference System
                        `USNG` - United States National Grid
                        `UTM` - Universal Transverse Mercator
                        `GeoRef` - World Geographic Reference System
                        `GARS` - Global Area Reference System
                        `DMS` - Degree Minute Second
                        `DDM` - Degree Decimal Minute
                        `DD` - Decimal Degree
    ----------------  -------------------------------------------------------------------------------
    conversion_mode   Conversion options for MGRS, UTM and GARS conversion types.

                      .. note::
                        Valid conversion modes for MGRS are:
                        `mgrsDefault` - Default. Uses the spheroid from the given spatial reference.

                        `mgrsNewStyle` - Treats all spheroids as new, like WGS 1984. The 80 degree longitude falls into Zone 60.

                        `mgrsOldStyle` - Treats all spheroids as old, like Bessel 1841. The 180 degree longitude falls into Zone 60.

                        `mgrsNewWith180InZone01` - Same as mgrsNewStyle except the 180 degree longitude falls into Zone 01

                        `mgrsOldWith180InZone01` - Same as mgrsOldStyle except the 180 degree longitude falls into Zone 01

                      .. note::
                        Valid conversion modes for UTM are:
                        `utmDefault` - Default. No options.
                        `utmNorthSouth` - Uses north/south latitude indicators instead of
                        `zone numbers` - Non-standard. Default is recommended

    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        An array of (x,y) coordinates, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

            >>> coords = from_geo_coordinate_string(spatial_ref = "wkid",
                                            strings = ["01N AA 66021 00000","11S NT 00000 62155", "31U BT 94071 65288"]
                                            conversion_type = "MGRS",
                                            conversion_mode = "mgrs_default",
                                            future = False)
            >>> coords
                [[x1,y1], [x2,y2], [x3,y3]]
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.from_geo_coordinate_string(
        spatial_ref, strings, conversion_type, conversion_mode, future=future
    )


def generalize(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    geometries: list[Geometry],
    max_deviation: int,
    deviation_unit: str | LengthUnits | None = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``generalize`` function is performed on a :class:`~arcgis.geometry.Geometry` service
    resource. The `generalize` function simplifies the input geometries
    using the `Douglas-Peucker` algorithm with a specified maximum
    deviation distance.

    .. note::
        The output geometries will contain a subset of
        the original input vertices.


    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        The array :class:`~arcgis.geometry.Geometry` objects to be generalized.
    ----------------  -------------------------------------------------------------------------------
    max_deviation     ``max_deviation`` sets the maximum allowable offset,
                      which will determine the degree of simplification. This value
                      limits the distance the output geometry can differ from the input
                      geometry.
    ----------------  -------------------------------------------------------------------------------
    deviation_unit    If ``geodesic`` is set to true, then the geodesic distance
                      between the ``geometry1`` and ``geometry2`` geometries is returned.
                      Geodesic distance is the shortest path between two points along
                      the ellipsoid of the earth. If ``geodesic`` is set to false or not
                      specified, the planar distance is returned. The default value is false.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        An array of the simplified :class:`~arcgis.geometry.Geometry` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    if isinstance(deviation_unit, LengthUnits):
        deviation_unit = deviation_unit.value
    elif deviation_unit is None:
        deviation_unit = ""
    return gis._tools.geometry.generalize(
        spatial_ref, geometries, max_deviation, deviation_unit, future=future
    )


def intersect(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    geometries: list[Geometry],
    geometry: Geometry,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``intersect`` function is performed on a :class:`~arcgis.geometry.Geometry` service
    resource. This function constructs the set-theoretic intersection
    between an array of geometries and another geometry.

    .. note::
        The dimension of each resultant geometry is the minimum dimension of the input
        geometry in the geometries array and the other geometry specified by the geometry parameter.


    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        An array of :class:`~arcgis.geometry.Point`, :class:`~arcgis.geometry.MultiPoint`,
                      :class:`~arcgis.geometry.Polyline`, or :class:`~arcgis.geometry.Polygon` objects.
                      The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    geometry          A single :class:`~arcgis.geometry.Geometry` of any type and of a dimension equal
                      to or greater than the elements of geometries. The structure of
                      geometry is the same as the structure of the JSON geometry
                      objects returned by the ArcGIS REST API. The use of simple
                      syntax is not supported.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        The set-theoretic dimension between :class:`~arcgis.geometry.Geometry` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.intersect(
        spatial_ref, geometries, geometry, future=future
    )


def label_points(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    polygons: list[Polygon],
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``label_points`` function is performed on a :class:`~arcgis.geometry.Geometry` service
    resource. The ``labelPoints`` function calculates an interior :class:`~arcgis.geometry.Point`
    for each :class:`~arcgis.geometry.Polygon` specified in the input array. These interior
    points can be used by clients for labeling the polygons.


    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    polygons          An array of :class:`~arcgis.geometry.Polygon` objects whose label :class:`~arcgis.geometry.Point`
                      objects are to be computed. The spatial reference of the polygons is specified by
                      ``spatial_ref``.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        An array of :class:`~arcgis.geometry.Point` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.label_points(spatial_ref, polygons, future=future)


def lengths(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    polylines: Polyline,
    length_unit: str | LengthUnits,
    calculation_type: str,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``lengths`` function is performed on a :class:`~arcgis.geometry.Geometry` service resource.
    This function calculates the` 2D Euclidean` or `geodesic` lengths of
    each :class:`~arcgis.geometry.Polyline` specified in the input array.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    polylines         The array of :class:`~arcgis.geometry.Polyline` whose lengths are to be computed.
    ----------------  -------------------------------------------------------------------------------
    length_unit       The length unit in which the length of
                      :class:`~arcgis.geometry.Polyline` will be calculated. If ``calculation_type``
                      is planar, then ``length_unit`` can be any `esriUnits`
                      constant. If ``lengthUnit`` is not specified, the
                      units are derived from ``spatial_ref``. If ``calculationType`` is
                      not planar, then `lengthUnit` must be a linear
                      esriUnits constant, such as `esriSRUnit_Meter` or
                      `esriSRUnit_SurveyMile`. If ``length_unit`` is not
                      specified, the units are meters. For a list of
                      valid units, see `esriSRUnitType Constants` and
                      `esriSRUnit2Type Constant`.
    ----------------  -------------------------------------------------------------------------------
    calculation_type  The type defined for the length calculation of the input geometries. The type can be one
                      of the following values:

                          1. planar - Planar measurements use 2D Euclidean distance to calculate area and length. This
                          should only be used if the area or length needs to be calculated in the given
                          :class:`~arcgis.geometry.SpatialReference`. Otherwise, use ``preserveShape``.

                          2. geodesic - Use this type if you want to calculate an area or length using only the vertices
                          of the :class:`~arcgis.geometry.Polygon` and define the lines between the points as geodesic
                          segments independent of the actual shape of the :class:`~arcgis.geometry.Polygon`. A geodesic
                          segment is the shortest path between two points on an ellipsoid.

                          3. preserveShape - This type calculates the area or length of the geometry on the surface of
                          the Earth ellipsoid. The shape of the geometry in its coordinate system is preserved.
    ----------------  -------------------------------------------------------------------------------
     future           Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        A list of floats of 2D-Euclidean or Geodesic lengths, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    if isinstance(length_unit, LengthUnits):
        length_unit = length_unit.value
    service = gis._tools.geometry
    return service.lengths(
        spatial_ref, polylines, length_unit, calculation_type, future=future
    )


def offset(
    geometries: Union[list[Polygon], list[Polyline], list[MultiPoint], list[Point]],
    offset_distance: float,
    offset_unit: str | LengthUnits,
    offset_how: str = "esriGeometryOffsetRounded",
    bevel_ratio: int = 10,
    simplify_result: bool = False,
    spatial_ref: Optional[Union[int, dict[str, Any]]] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``offset`` function is performed on a :class:`~arcgis.geometry.Geometry` service resource.
    This function constructs geometries that are offset from the
    given input geometries. If the offset parameter is positive, the
    constructed offset will be on the right side of the geometry. Left
    side offsets are constructed with negative parameters.

    .. note::
        Tracing the geometry from its first vertex to the last will give you a
        direction along the geometry. It is to the right and left
        perspective of this direction that the positive and negative
        parameters will dictate where the offset is constructed. In these
        terms, it is simple to infer where the offset of even horizontal geometries will be constructed.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        An array of :class:`~arcgis.geometry.Point`, :class:`~arcgis.geometry.MultiPoint`,
                      :class:`~arcgis.geometry.Polyline`, or :class:`~arcgis.geometry.Polygon` objects.
                      The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    offset_distance   Specifies the distance for constructing an offset
                      based on the input geometries.

                      .. note::
                        If the ``offset_distance`` parameter is
                        positive, the constructed offset will be on the right side of the
                        curve. Left-side offsets are constructed with negative values.
    ----------------  -------------------------------------------------------------------------------
    offset_unit       A unit for offset distance. If a unit is not specified, the units are derived from
                      ``spatial_ref``.
    ----------------  -------------------------------------------------------------------------------
    offset_how        The ``offset_how`` parameter determines how outer corners between segments are handled.
                      The three options are as follows:
                        1. ``esriGeometryOffsetRounded`` - Rounds the corner between extended offsets.
                        2. ``esriGeometryOffsetBevelled`` - Squares off the corner after a given ratio distance.
                        3. ``esriGeometryOffsetMitered`` - Attempts to allow extended offsets to naturally intersect,
                        but if that intersection occurs too far from the corner, the corner is eventually bevelled off
                        at a fixed distance.
    ----------------  -------------------------------------------------------------------------------
    bevel_ratio       ``bevel_ratio`` is multiplied by the ``offset_distance``, and
                      the result determines how far a mitered offset intersection can
                      be located before it is bevelled. When mitered is specified,
                      bevel_ratio is ignored and 10 is used internally. When bevelled is
                      specified, 1.1 will be used if bevel_ratio is not specified.
                      ``bevel_ratio`` is ignored for rounded offset.
    ----------------  -------------------------------------------------------------------------------
    simplify_result   if ``simplify_result`` is set to true, then self
                      intersecting loops will be removed from the result offset geometries. The default is false.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        A list of :class:`~arcgis.geometry.Geometry` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

            >>> new_job = offset( geometries = [geom1,geom2,...],
                                  offset_distance = 100,
                                  offset_unit = "esriMeters",
                                  offset_how = "esriGeometryOffsetRounded",
                                  bevel_ratio = 0,
                                  simplify_result = True
                                  spatial_ref = "wkid",
                                  future = True)

    """
    if gis is None:
        gis = arcgis.env.active_gis
    if isinstance(offset_unit, LengthUnits):
        offset_unit = offset_unit.value
    return gis._tools.geometry.offset(
        geometries,
        offset_distance,
        offset_unit,
        offset_how,
        bevel_ratio,
        simplify_result,
        spatial_ref,
        future=future,
    )


def project(
    geometries: Union[list[Polygon], list[Polyline], list[MultiPoint], list[Point]],
    in_sr: Optional[Union[int, dict[str, Any]]],
    out_sr: Optional[Union[int, dict[str, Any]]],
    transformation: str = "",
    transform_forward: bool = False,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``project`` function is performed on a :class:`~arcgis.geometry.Geometry` service resource.
    This function projects an array of input geometries from the input
    :class:`~arcgis.geometry.SpatialReference` to the output :class:`~arcgis.geometry.SpatialReference`

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        An array of :class:`~arcgis.geometry.Point`, :class:`~arcgis.geometry.MultiPoint`,
                      :class:`~arcgis.geometry.Polyline`, or :class:`~arcgis.geometry.Polygon` objects.
                      The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    in_sr             The well-known ID of the :class:`~arcgis.geometry.SpatialReference` or a spatial
                      reference JSON object for the input geometries.
    ----------------  -------------------------------------------------------------------------------
    out_sr            The well-known ID of the :class:`~arcgis.geometry.SpatialReference` or a
                      spatial reference JSON object for the output geometries.
    ----------------  -------------------------------------------------------------------------------
    transformations   The WKID or a JSON object specifying the
                      geographic transformation (gis,also known as datum transformation) to be applied to the projected
                      geometries.

                      .. note::
                        A transformation is needed only if the output :class:`~arcgis.geometry.SpatialReference`
                        contains a different geographic coordinate system than the input
                        spatial reference.

    ----------------  -------------------------------------------------------------------------------
    transformforward  A Boolean value indicating whether or not to transform forward. The forward or reverse direction
                      of transformation is implied in the name of the transformation. If
                      transformation is specified, a value for the ``transform_Forward``
                      parameter must also be specified. The default value is false.
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        A list of :class:`~arcgis.geometry.Geometry` objects in the ``out_sr`` coordinate system, or
        a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

        #Usage Example

        >>> result = project(geometries = [{"x": -17568824.55, "y": 2428377.35}, {"x": -17568456.88, "y": 2428431.352}],
                             in_sr = 3857,
                             out_sr = 4326)
            [{"x": -157.82343617279275, "y": 21.305781607280093}, {"x": -157.8201333369876, "y": 21.306233559873714}]
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.project(
        geometries,
        in_sr,
        out_sr,
        transformation,
        transform_forward,
        future=future,
    )


def relation(
    geometries1: list[Geometry],
    geometries2: list[Geometry],
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    spatial_relation: str = "esriGeometryRelationIntersection",
    relation_param: str = "",
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``relation`` function is performed on a :class:`~arcgis.geometry.Geometry` service resource.
    This function determines the pairs of geometries from the input
    geometry arrays that participate in the specified spatial relation.
    Both arrays are assumed to be in the spatial reference specified by
    ``spatial_ref``, which is a required parameter. Geometry types cannot be mixed
    within an array.

    .. note::
        The relations are evaluated in 2D. In other words, `z` coordinates are not used.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometry1         The first array of :class:`~arcgis.geometry.Geometry` objects to compute relations.
    ----------------  -------------------------------------------------------------------------------
    geometry2         The second array of :class:`~arcgis.geometry.Geometry` objects to compute relations.
    ----------------  -------------------------------------------------------------------------------
    relation_param    The Shape Comparison Language string to be evaluated.
    ----------------  -------------------------------------------------------------------------------
    spatial_relation  The spatial relationship to be tested between the two input geometry arrays.
                      Values: `esriGeometryRelationCross | esriGeometryRelationDisjoint |
                      esriGeometryRelationIn | esriGeometryRelationInteriorIntersection |
                      esriGeometryRelationIntersection | esriGeometryRelationLineCoincidence |
                      esriGeometryRelationLineTouch | esriGeometryRelationOverlap |
                      esriGeometryRelationPointTouch | esriGeometryRelationTouch |
                      esriGeometryRelationWithin | esriGeometryRelationRelation`
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================


    :returns:
        A JSON dict of geometryNIndex between two lists of geometries, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

            >>> new_res = relation(geometry1 = [geom1,geom2,...],
                                   geometry2 = [geom21,geom22,..],
                                   relation_param = "relationParameter",
                                   spatial_relation = "esriGeometryRelationPointTouch"
                                   spatial_ref = "wkid",
                                   future = False)
            >>> new_res
                {'relations': [{'geometry1Index': 0, 'geometry2Index': 0}]}
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.relation(
        geometries1,
        geometries2,
        spatial_ref,
        spatial_relation,
        relation_param,
        future=future,
    )


def reshape(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    target: Union[Polyline, Polygon],
    reshaper: Polyline,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``reshape`` function is performed on a :class:`~arcgis.geometry.Geometry` service resource.
    It reshapes a :class:`~arcgis.geometry.Polyline` or :class:`~arcgis.geometry.Polygon` feature by constructing a
    polyline over the feature. The feature takes the shape of the
    `reshaper` polyline from the first place the `reshaper` intersects the
    feature to the last.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    target            The :class:`~arcgis.geometry.Polyline` or :class:`~arcgis.geometry.Polygon` to be reshaped.
    ----------------  -------------------------------------------------------------------------------
    reshaper          The single-part :class:`~arcgis.geometry.Polyline` that does the reshaping.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or a JSON
                      object for the input geometry
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        A reshaped :class:`~arcgis.geometry.Polyline` or :class:`~arcgis.geometry.Polygon` object, or
        a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.reshape(spatial_ref, target, reshaper, future=future)


def simplify(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    geometries: Union[list[Polygon], list[Polyline], list[MultiPoint], list[Point]],
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``simplify`` function is performed on a :class:`~arcgis.geometry.Geometry` service resource.
    ``simplify`` permanently alters the input geometry so that the geometry
    becomes topologically consistent. This resource applies the ArcGIS
    ``simplify`` function to each geometry in the input array.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        An array of :class:`~arcgis.geometry.Point`, :class:`~arcgis.geometry.MultiPoint`,
                      :class:`~arcgis.geometry.Polyline`, or :class:`~arcgis.geometry.Polygon` objects.
                      The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        An array of :class:`~arcgis.geometry.Geometry` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.simplify(spatial_ref, geometries, future=future)


def to_geo_coordinate_string(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    coordinates: json,
    conversion_type: str,
    conversion_mode: str = "mgrsDefault",
    num_of_digits: Optional[int] = None,
    rounding: bool = True,
    add_spaces: bool = True,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``to_geo_coordinate_string`` function is performed on a :class:`~arcgis.geometry.Geometry`
    service resource. The function converts an array of
    xy-coordinates into well-known strings based on the conversion type
    and :class:`~arcgis.geometry.SpatialReference` supplied by the :class:`~arcgis.gis.User`. Optional parameters are
    available for some conversion types. See :attr:`~arcgis.geometry.functions.from_geo_coordinate_strings` for more
    information on the opposite conversion.

    .. note::
        If an optional parameter is not applicable for a particular conversion type, but a
        value is supplied for that parameter, the value will be ignored.


    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    coordinates       An array of xy-coordinates in JSON format to be converted. Syntax: [[x1,y2],...[xN,yN]]
    ----------------  -------------------------------------------------------------------------------
    conversion-type   The conversion type of the input strings.

                      .. note::
                        Valid conversion types are:
                        `MGRS` - Military Grid Reference System
                        `USNG` - United States National Grid
                        `UTM` - Universal Transverse Mercator
                        `GeoRef` - World Geographic Reference System
                        `GARS` - Global Area Reference System
                        `DMS` - Degree Minute Second
                        `DDM` - Degree Decimal Minute
                        `DD` - Decimal Degree
    ----------------  -------------------------------------------------------------------------------
    conversion_mode   Conversion options for MGRS, UTM and GARS conversion types.

                      .. note::
                        Valid conversion modes for MGRS are:
                        `mgrsDefault` - Default. Uses the spheroid from the given spatial reference.

                        `mgrsNewStyle` - Treats all spheroids as new, like WGS 1984. The 80 degree longitude falls into Zone 60.

                        `mgrsOldStyle` - Treats all spheroids as old, like Bessel 1841. The 180 degree longitude falls into Zone 60.

                        `mgrsNewWith180InZone01` - Same as mgrsNewStyle except the 180 degree longitude falls into Zone 01

                        `mgrsOldWith180InZone01` - Same as mgrsOldStyle except the 180 degree longitude falls into Zone 01

                      .. note::
                        Valid conversion modes for UTM are:
                        `utmDefault` - Default. No options.
                        `utmNorthSouth` - Uses north/south latitude indicators instead of
                        `zone numbers` - Non-standard. Default is recommended
    ----------------  -------------------------------------------------------------------------------
    num_of_digits     The number of digits to output for each of the numerical portions in the string. The default
                      value for ``num_of_digits`` varies depending on ``conversion_type``.
    ----------------  -------------------------------------------------------------------------------
    rounding          If ``True``, then numeric portions of the string are rounded to the nearest whole magnitude as
                      specified by
                      num_of_digits. Otherwise, numeric portions of the string are
                      truncated. The rounding parameter applies only to conversion
                      types `MGRS`, `USNG` and `GeoRef`. The default value is ``True``.
    ----------------  -------------------------------------------------------------------------------
    addSpaces         If ``True``, then spaces are added between components of the string. The ``addSpaces`` parameter
                      applies only to conversion types `MGRS`, `USNG` and `UTM`. The default value for `MGRS` is
                      ``False``, while the default value for both `USNG` and `UTM` is ``True``.
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        An array of Strings, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

        .. code-block:: python

            >>> strings = from_geo_coordinate_string(spatial_ref = "wkid",
                                                     coordinates = [[x1,y1], [x2,y2], [x3,y3]]
                                                     conversion_type = "MGRS",
                                                     conversion_mode = "mgrs_default",
                                                     future = False)
            >>> strings
                ["01N AA 66021 00000","11S NT 00000 62155", "31U BT 94071 65288"]
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.to_geo_coordinate_string(
        spatial_ref,
        coordinates,
        conversion_type,
        conversion_mode,
        num_of_digits,
        rounding,
        add_spaces,
        future=future,
    )


def trim_extend(
    spatial_ref: Optional[Union[int, dict[str, Any]]],
    polylines: list[Polyline],
    trim_extend_to: Polyline,
    extend_how: int = 0,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``trim_extend`` function is performed on a :class:`~arcgis.geometry.Geometry` service
    resource. This function trims or extends each :class:`~arcgis.geometry.Polyline` specified
    in the input array, using the user-specified guide polylines.

    .. note::
        When trimming features, the part to the left of the oriented cutting
        line is preserved in the output, and the other part is discarded.
        An empty :class:`~arcgis.geometry.Polyline` is added to the output array if the corresponding
        input polyline is neither cut nor extended.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    polylines         An array of :class:`~arcgis.geometry.Polyline` objects to trim or extend
    ----------------  -------------------------------------------------------------------------------
    trim_extend_to    A :class:`~arcgis.geometry.Polyline` that is used as a guide for trimming or
                      extending input polylines.
    ----------------  -------------------------------------------------------------------------------
    extend_how        A flag that is used along with the trimExtend function.

                      ``0`` - By default, an extension considers both ends of a path. The
                      old ends remain, and new points are added to the extended ends.
                      The new points have attributes that are extrapolated from adjacent existing segments.

                      ``1`` - If an extension is performed at an end, relocate the end
                      point to the new position instead of leaving the old point and
                      adding a new point at the new position.

                      ``2`` - If an extension is performed at an end, do not extrapolate
                      the end-segment's attributes for the new point. Instead, make
                      its attributes the same as the current end. Incompatible with `esriNoAttributes`.

                      ``4`` - If an extension is performed at an end, do not extrapolate
                      the end-segment's attributes for the new point. Instead, make
                      its attributes empty. Incompatible with esriKeepAttributes.

                      ``8`` - Do not extend the 'from' end of any path.

                      ``16`` - Do not extend the 'to' end of any path.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       A :class:`~arcgis.geometry.SpatialReference` of the input geometries Well-Known ID or JSON object
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
    ================  ===============================================================================

    :returns:
        An array of :class:`~arcgis.geometry.Polyline` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

            >>> polylines_arr = trim_extends(polylines = [polyline1,polyline2, ...],
                                             trim_extend_to = polyline_trimmer
                                             extend_how = 2,
                                             spatial_ref = "wkid",
                                             future = False)
            >>> polyline_arr
                [polyline1, polyline2,...]
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.trim_extend(
        spatial_ref, polylines, trim_extend_to, extend_how, future=future
    )


def union(
    geometries: Union[list[Polygon], list[Polyline], list[MultiPoint], list[Point]],
    spatial_ref: Optional[Union[str, dict[str:str]]] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    The ``union`` function is performed on a :class:`~arcgis.geometry.Geometry` service resource.
    This function constructs the set-theoretic union of the geometries
    in the input array.

    .. note::
        All inputs must be of the same type.

    ================  ===============================================================================
    **Keys**          **Description**
    ----------------  -------------------------------------------------------------------------------
    geometries        Required. An array of :class:`~arcgis.geometry.Point`,
                      :class:`~arcgis.geometry.MultiPoint`, :class:`~arcgis.geometry.Polyline`,
                      or :class:`~arcgis.geometry.Polygon` objects.
                      The structure of each geometry in the array is the
                      same as the structure of the JSON geometry objects returned by
                      the ArcGIS REST API.
    ----------------  -------------------------------------------------------------------------------
    spatial_ref       An optional String or JSON Dict representing the wkid to be used. The default is the
                      spatial reference found in the geometry or, if None found, then "4326".

                      Example: "4326" or {"wkid":"4326"}
    ----------------  -------------------------------------------------------------------------------
    future            Optional boolean. If True, a future object will be returned and the process
                      will not wait for the task to complete. The default is False, which means wait for results.
                      If setting future to True there is a limitation of 6500 geometries that can be processed in one call.
    ================  ===============================================================================

    :returns:
        The set-theoretic union of the :class:`~arcgis.geometry.Geometry` objects, or a `GeometryJob` object. If ``future = True``,
        then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    if spatial_ref is None:
        spatial_ref = [
            geom.spatialReference
            for geom in geometries
            if "spatialReference" in geom and geom.spatialReference is not None
        ]
        spatial_ref = spatial_ref[0] if len(spatial_ref) > 0 else "4326"
    if isinstance(spatial_ref, dict):
        spatial_ref = spatial_ref["wkid"]
    return gis._tools.geometry.union(spatial_ref, geometries, future=future)
