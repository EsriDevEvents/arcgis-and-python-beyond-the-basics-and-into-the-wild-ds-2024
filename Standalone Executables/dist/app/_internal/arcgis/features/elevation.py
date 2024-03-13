"""
These functions help you use elevation analysis
"""
from arcgis.auth.tools import LazyLoader

_util = LazyLoader("arcgis._impl.common._utils")
_logging = LazyLoader("logging")
arcgis = LazyLoader("arcgis")
_geoprocessing = LazyLoader("arcgis.geoprocessing")
from arcgis.features import FeatureSet


_log = _logging.getLogger(__name__)

_use_async = True


def profile(
    input_line_features: FeatureSet = {
        "exceededTransferLimit": False,
        "spatialReference": {"latestWkid": 3857, "wkid": 102100},
        "geometryType": "esriGeometryPolyline",
        "fields": [
            {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
            {
                "name": "Shape_Length",
                "type": "esriFieldTypeDouble",
                "alias": "Shape_Length",
            },
        ],
        "displayFieldName": "",
        "features": [],
    },
    profile_id_field: str = None,
    dem_resolution: str = None,
    maximum_sample_distance: float = None,
    maximum_sample_distance_units: str = """Meters""",
    gis=None,
    future=False,
) -> FeatureSet:
    """
    .. image:: _static/images/elevation_profile/elevation_profile.png

    The profile method is used to create profiles along input lines from which a profile graph can be created.

    In asynchronous mode, the maximum number of input line features that can be accepted by the task for each request is 1000.

    =====================================    ===========================================================================
    **Parameter**                             **Description**
    -------------------------------------    ---------------------------------------------------------------------------
    input_line_features                      Required featureset. The line features that will be profiled over the surface.
    -------------------------------------    ---------------------------------------------------------------------------
    profile_id_field                         Optional string. A unique identifier to tie profiles to their corresponding input line features.
    -------------------------------------    ---------------------------------------------------------------------------
    dem_resolution                           Optional string. The approximate spatial resolution (cell size) of the source elevation data used for the calculation.
                                             The resolution values are an approximation of the spatial resolution of the digital elevation model. While many elevation sources are distributed in units of arc seconds, the keyword is an approximation of those resolutions in meters for easier understanding.
    -------------------------------------    ---------------------------------------------------------------------------
    maximum_sample_distance                  Optional float. The maximum sampling distance along the line to sample elevation values.
    -------------------------------------    ---------------------------------------------------------------------------
    maximum_sample_distance_units            Optional string. The units for the MaximumSampleDistance.

                                             Choice list:['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles']
    -------------------------------------    ---------------------------------------------------------------------------
    future                                   Optional boolean. If True, the result will be a `GPJob` and results will be returned asynchronously.
    =====================================    ===========================================================================

    :return: Output Profile as a FeatureSet

    .. code-block:: python

        USAGE EXAMPLE: To create profile of mountains feature.
        elevation = profile(input_line_features=mountain_fs,
                            dem_resolution='FINEST',
                            maximum_sample_distance=500,
                            maximum_sample_distance_units='Meters')
    """

    param_db = {
        "input_line_features": input_line_features,
        "profile_id_field": profile_id_field,
        "dem_resolution": dem_resolution,
        "maximum_sample_distance": maximum_sample_distance,
        "maximum_sample_distance_units": maximum_sample_distance_units,
        "future": True,
        "gis": gis,
    }

    if gis is None:
        gis = arcgis.env.active_gis

    url = gis.properties.helperServices.elevation.url
    tbx = _geoprocessing.import_toolbox(url, gis=gis)
    param_db = _util.inspect_function_inputs(tbx.profile, **param_db)
    param_db["future"] = True
    gpjob = tbx.profile(**param_db)
    if future:
        return gpjob
    return gpjob.result()


def viewshed(
    input_points: FeatureSet = {
        "exceededTransferLimit": False,
        "spatialReference": {"latestWkid": 3857, "wkid": 102100},
        "geometryType": "esriGeometryPoint",
        "fields": [
            {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
            {"name": "offseta", "type": "esriFieldTypeDouble", "alias": "offseta"},
            {"name": "offsetb", "type": "esriFieldTypeDouble", "alias": "offsetb"},
        ],
        "displayFieldName": "",
        "features": [],
    },
    maximum_distance: float = None,
    maximum_distance_units: str = """Meters""",
    dem_resolution: str = None,
    observer_height: float = None,
    observer_height_units: str = """Meters""",
    surface_offset: float = None,
    surface_offset_units: str = """Meters""",
    generalize_viewshed_polygons: bool = True,
    gis=None,
    future=False,
) -> FeatureSet:
    """
    .. image:: _static/images/elevation_viewshed/elevation_viewshed.png

    The ``viewshed`` method is used to identify visible areas based on observer locations you provide as well as ArcGIS Online Elevation data.

    ===============================    =========================================================
    **Parameter**                       **Description**
    -------------------------------    ---------------------------------------------------------
    input_points                       Required FeatureSet. The point features to use as the observer locations. See :ref:`Feature Input<FeatureInput>`.
    -------------------------------    ---------------------------------------------------------
    maximum_distance                   Optional float. This is a cutoff distance where the computation of visible areas stops.
                                       Beyond this distance, it is unknown whether the analysis points and the other objects can see each other.

                                       It is useful for modeling current weather conditions or a given time of day, such as dusk. Large values increase computation time.

                                       Unless specified, a default maximum distance will be computed based on the resolution and extent of the source DEM.
                                       The allowed maximum value is 50 kilometers.

                                       Use ``maximum_distance_units`` to set the units for ``maximum_distance``.
    -------------------------------    ---------------------------------------------------------
    maximum_distance_units             Optional string. The units for the ``maximum_distance`` parameter.

                                       Choice list:['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles'].

                                       The default is 'Meters'.
    -------------------------------    ---------------------------------------------------------
    dem_resolution                     Optional string. The approximate spatial resolution (cell size) of the source elevation data used for the calculation.
                                       The resolution values are an approximation of the spatial resolution of the digital elevation model.
                                       While many elevation sources are distributed in units of arc seconds, the keyword is an approximation of those resolutions in meters for easier understanding.

                                       Choice list:[' ', 'FINEST', '10m', '30m', '90m'].

                                       The default is 90m.
    -------------------------------    ---------------------------------------------------------
    observer_height                    Optional float. This is the height above the ground of the observer locations.

                                       The default is 1.75 meters, which is approximately the average height of a person.
                                       If you are looking from an elevated location, such as an observation tower or a tall building, use that height instead.

                                       Use ``observer_height_units`` to set the units for ``observer_height``.
    -------------------------------    ---------------------------------------------------------
    observer_height_units              Optional string. The units for the ``observer_height`` parameter.

                                       Choice list:['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles']
    -------------------------------    ---------------------------------------------------------
    surface_offset                     Optional float. The height above the surface of the object you are trying to see.

                                       The default value is 0.0. If you are trying to see buildings or wind turbines, use their height here.
    -------------------------------    ---------------------------------------------------------
    surface_offset_units               Optional string. The units for the ``surface_offset`` parameter.

                                       Choice list:['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles']
    -------------------------------    ---------------------------------------------------------
    generalize_viewshed_polygons       Optional boolean. Determines whether or not the viewshed polygons are to be generalized.

                                       The viewshed calculation is based on a raster elevation model that creates a result with
                                       stair-stepped edges. To create a more pleasing appearance and improve performance, the
                                       default behavior is to generalize the polygons. The generalization process smooths the
                                       boundary of the visible areas and may remove some single-cell visible areas.
    -------------------------------    ---------------------------------------------------------
    future                             Optional boolean. If True, the result will be a `GPJob` and results will be returned asynchronously.
    ===============================    =========================================================

    :return: output_viewshed - Output Viewshed as a FeatureSet (polygons of visible areas for a given set of input observation points.)

    .. code-block:: python

        # USAGE EXAMPLE: To identify visible areas from esri headquarter office.
        visible_windfarms = viewshed(input_points=hq_fs,
                                     maximum_distance=200,
                                     maximum_distance_units='Meters',
                                     observer_height=6,
                                     observer_height_units='Feet',
                                     surface_offset=100,
                                     surface_offset_units='Meters',
                                     generalize_viewshed_polygons=True)
    """

    if gis is None:
        gis = arcgis.env.active_gis

    url = gis.properties.helperServices.elevation.url
    tbx = _geoprocessing.import_toolbox(url, gis=gis)
    param_db = {
        "dem_resolution": dem_resolution,
        "generalize_viewshed_polygons": generalize_viewshed_polygons,
        "input_points": input_points,
        "maximum_distance": maximum_distance,
        "maximum_distance_units": maximum_distance_units,
        "observer_height": observer_height,
        "observer_height_units": observer_height_units,
        "surface_offset": surface_offset,
        "surface_offset_units": surface_offset_units,
        "future": future,
        "gis": gis,
    }
    param_db = _util.inspect_function_inputs(tbx.viewshed, **param_db)
    param_db["future"] = True
    gpjob = tbx.viewshed(**param_db)
    if future:
        return gpjob
    return gpjob.result()


def summarize_elevation(
    input_features: FeatureSet = {},
    feature_id_field: str = None,
    dem_resolution: str = None,
    include_slope_aspect: bool = False,
    gis=None,
    future=False,
) -> FeatureSet:
    """
    .. image:: _static/images/summarize_elevation/summarize_elevation.png

    The ``summarize_elevation`` method calculates summary statistics for features you provide based
    on ArcGIS Online Elevation data. It accepts point, line, or polygon input and returns statistics
    for the elevation, slope, and aspect of the features.

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    input_features               Reqauired FeatureSet. Input features to summarize the elevation for. The features can be point, line, or area. See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    feature_id_field             Optional string. The unique ID field to use for the input features.
    -------------------------    ---------------------------------------------------------
    dem_resolution               Optional string. The approximate spatial resolution (cell size) of the source elevation data used for the calculation.

                                 Choice list:[' ', 'FINEST', '10m', '30m', '90m']

                                 The default value is None.
    -------------------------    ---------------------------------------------------------
    include_slope_aspect         Optional boolean. Determines if slope and aspect for the input feature(s) will be included in the output. The slope and aspect values in the output are in degrees.

                                 The default value is False.
    -------------------------    ---------------------------------------------------------
    future                       Optional boolean. If True, the result will be a `GPJob` and results will be returned asynchronously.
    =========================    =========================================================

    :return: result_layer : Output Summary as a FeatureSet

    .. code-block:: python

        # USAGE EXAMPLE: To calculate summary statistics for mountain polyline features.
        summarize = summarize_elevation(input_features=mountain_fs,
                           dem_resolution='FINEST',
                           include_slope_aspect=True)
    """

    if gis is None:
        gis = arcgis.env.active_gis
    url = gis.properties.helperServices.elevation.url
    tbx = _geoprocessing.import_toolbox(url, gis=gis)
    param_db = {
        "dem_resolution": dem_resolution,
        "feature_id_field": feature_id_field,
        "include_slope_aspect": include_slope_aspect,
        "input_features": input_features,
        "gis": gis,
        "future": future,
    }
    param_db = _util.inspect_function_inputs(tbx.summarize_elevation, **param_db)
    param_db["future"] = True
    gpjob = tbx.summarize_elevation(**param_db)
    if future:
        return gpjob
    return gpjob.result()
