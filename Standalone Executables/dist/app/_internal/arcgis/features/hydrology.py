# coding: utf-8
"""
These functions help you use hydrology analysis.
"""

import logging as _logging
from typing import Optional, Union

import arcgis
from arcgis.features.feature import FeatureCollection
from arcgis.geoprocessing._support import _execute_gp_tool
from arcgis.features import FeatureSet
from arcgis.gis import GIS
from .._impl.common._utils import inspect_function_inputs
from arcgis.geoprocessing import import_toolbox as _import_toolbox

_log = _logging.getLogger(__name__)


def _evaluate_spatial_input(input_points):
    """
    Helper function to determine if the input is either a FeatureSet or Spatially Enabled DataFrame, and
    output to FeatureSet for subsequent processing.
    :param input_points: FeatureSet or Spatially Enabled DataFrame

    :return: FeatureSet
    """
    try:
        from arcgis.features.geo._accessor import _is_geoenabled
        from pandas import DataFrame
    except ImportError as ie:
        _log.warning(
            "One or more of the libraries needed for this feature is not available. "
            "Please resolve the following error: " + str(ie)
        )
        raise ie

    if isinstance(input_points, FeatureSet):
        return input_points

    elif isinstance(input_points, DataFrame) and _is_geoenabled(input_points):
        return input_points.spatial.to_featureset()

    elif isinstance(input_points, DataFrame) and not _is_geoenabled(input_points):
        raise Exception(
            "input_points is a DataFrame, but does not appear to be spatially enabled. Using the <df>.spatial.set_geometry(col, sr=None) may help. (https://esri.github.io/arcgis-python-api/apidoc/html/arcgis.features.toc.html#arcgis.features.GeoAccessor.set_geometry)"
        )

    else:
        raise Exception(
            "input_points must be either a FeatureSet or Spatially Enabled DataFrame instead of {}".format(
                type(input_points)
            )
        )


def trace_downstream(
    input_points,
    point_id_field: Optional[str] = None,
    source_database: str = "Finest",
    generalize: bool = False,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """

    .. image:: _static/images/trace_downstream/trace_downstream.png

    The ``trace_downstream`` method delineates the downstream path from a specified location.
    Esri-curated elevation data is used to create an output polyline delineating the flow path
    downstream from the specified input location. This method accesses a service using multiple
    source databases which are available for different geographic areas and at different
    spatial scales.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    input_points           Required FeatureSet or Spatially Enabled DataFrame
                           Points delineating the starting location to calculate the downstream
                           location from. See :ref:`Feature Input<FeatureInput>`.
    ------------------     --------------------------------------------------------------------
    point_id_field         Optional string. Field used to identify the feature from the source data. This is
                           useful for relating the results back to the original source data.
    ------------------     --------------------------------------------------------------------
    source_database        Optional string. Keyword indicating the source data that will be used in the
                           analysis. This keyword is an approximation of the spatial resolution
                           of the digital elevation model used to build the foundation
                           hydrologic  database. Since many elevation sources are distributed
                           with units of  arc seconds, this keyword is an approximation in
                           meters for easier understanding.


                           - Finest: Finest resolution available at each location from all
                             possible data sources.
                           - 10m: The hydrologic source was built from 1/3 arc second -
                             approximately 10 meter resolution, elevation data.
                           - 30m: The hydrologic source was built from 1 arc second -
                             approximately 30 meter resolution, elevation data.
                           - 90m: The hydrologic source was built from 3 arc second -
                             approximately 90 meter resolution, elevation data.

                           The default value is 'Finest'.
    ------------------     --------------------------------------------------------------------
    generalize             Optional boolean. Determines if the output downstream trace lines will be smoothed
                           into simpler lines.

                           The default value is False.
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS` Object instance. If not provided as input, a GIS object instance logged into an
                           active portal with elevation helper services defined must already
                           be created in the active Python session. A GIS object instance can
                           also be optionally explicitly passed in through this parameter.
    ------------------     --------------------------------------------------------------------
    future                 Optional boolean. If True, a future object will be returned and the process
                           will not wait for the task to complete. The default is False, which means wait for results.
    ==================     ====================================================================

    :return: A new :class:`~arcgis.features.FeatureSet`

    .. code-block:: python

        # USAGE EXAMPLE: To trace downstream path from from the outlet points.
        path = trace_downstream(input_points=fs,
                                source_database='Finest',
                                generalize=False)
    """

    # use helper function to evaluate the input points and convert them, if necessary, to a FeatureSet
    input_fs = _evaluate_spatial_input(input_points)

    if input_fs.geometry_type != "esriGeometryPoint":
        raise Exception(
            "input_points FeatureSet must be point esriGeometryPoint, not {}".format(
                input_fs.geometry_type
            )
        )

    input_fields = input_fs.fields
    if (
        point_id_field
        and point_id_field not in [f["name"] for f in input_fields]
        and len(input_fields)
    ):
        input_fields_str = ",".join(input_fields)
        raise Exception(
            "The provided point_id_field {} does not appear to be in the input_points FeatureSet fields - {}".format(
                point_id_field, input_fields_str
            )
        )

    if source_database not in ["Finest", "10m", "30m", "90m"]:
        raise Exception(
            'source_database must be either "Finest", "10m", "30m", or "90m". {} does not appear to be one of these.'.format(
                source_database
            )
        )

    if gis is None and arcgis.env.active_gis is None:
        raise Exception(
            "GIS must be defined either by directly passing in a GIS object created using credentials, or one must already be created in the active Python session."
        )
    elif gis is None:
        gis = arcgis.env.active_gis

    url = gis.properties.helperServices.hydrology.url
    tbx = _import_toolbox(url, gis=gis)
    kwargs = {
        "input_points": input_fs,
        "point_id_field": point_id_field,
        "source_database": source_database,
        "generalize": generalize,
        "gis": gis,
        "future": future,
    }
    kwargs = inspect_function_inputs(tbx.trace_downstream, **kwargs)
    kwargs["future"] = True
    gpjob = tbx.trace_downstream(**kwargs)
    if future:
        return gpjob
    return gpjob.result()


def watershed(
    input_points,
    point_id_field: Optional[str] = None,
    snap_distance: float = 10,
    snap_distance_units: str = "Meters",
    source_database: str = "Finest",
    generalize: bool = False,
    gis: Optional[GIS] = None,
    return_snapped_points: bool = True,
    future: bool = False,
):
    """
    .. image:: _static/images/create_watersheds/create_watersheds.png

    The ``watershed`` is used to identify catchment areas based on a particular location you
    provide and ArcGIS Online Elevation data.


    ========================     ====================================================================
    **Parameter**                 **Description**
    ------------------------     --------------------------------------------------------------------
    input_points                 Required FeatureSet or Spatially Enabled DataFrame. Points delineating the starting location to calculate the downstream
                                 location from. See :ref:`Feature Input<FeatureInput>`.
    ------------------------     --------------------------------------------------------------------
    point_id_field               Optional String. Field used to identify the feature from the source data. This is
                                 useful for relating the results back to the original source data.
    ------------------------     --------------------------------------------------------------------
    snap_distance                Optional float. The maximum distance to move the location of an input point.

                                 Interactive input points and documented gage locations may not exactly align with the stream location in the DEM.
                                 This parameter allows the task to move the point to a nearby location with the largest contributing area.

                                 The snap distance should always be larger than the source data resolution. By default, the snapping distance
                                 is calculated as the resolution of the source data multiplied by 5.

                                 The default value is 10.
    ------------------------     --------------------------------------------------------------------
    snap_distance_units          Optional String. The linear units specified for the snap distance.

                                 Choice list: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles'].

                                 The default value is 'Meters'.
    ------------------------     --------------------------------------------------------------------
    source_database              Optional String. Keyword indicating the source data that will be used in the analysis.
                                 This keyword is an approximation of the spatial resolution of the
                                 digital elevation model used to build the foundation hydrologic
                                 database. Since many elevation sources are distributed with units of
                                 arc seconds, this keyword is an approximation in meters for easier
                                 understanding.

                                 * ``Finest``: Finest resolution available at each location from all possible data sources.
                                 * ``10m``: The hydrologic source was built from 1/3 arc second - approximately 10 meter resolution, elevation data.
                                 * ``30m``: The hydrologic source was built from 1 arc second - approximately 30 meter resolution, elevation data.
                                 * ``90m``: The hydrologic source was built from 3 arc second - approximately 90 meter resolution, elevation data.

                                 The default value is 'Finest'.
    ------------------------     --------------------------------------------------------------------
    generalize                   Optional boolean. Determines if the output downstream trace lines will be smoothed
                                 into simpler lines.

                                 The default value is False.
    ------------------------     --------------------------------------------------------------------
    gis                          Optional :class:`~arcgis.gis.GIS` Object instance. If not provided as input, a GIS object instance logged into an
                                 active portal with elevation helper services defined must already
                                 be created in the active Python session. A GIS object instance can
                                 also be optionally explicitly passed in through this parameter.
    ------------------------     --------------------------------------------------------------------
    return_snapped_points        Optional boolean. Determines if a point feature at the watershed’s pour point will be returned.
                                 If snapping is enabled, this might not be the same as the input point.

                                 The default value is True.
    ------------------------     --------------------------------------------------------------------
    future                       Optional boolean. If True, a future object will be returned and the process
                                 will not wait for the task to complete. The default is False, which means wait for results.
    ========================     ====================================================================

    :return:
        Result object comprised of two FeatureSets - one for watershed_area, and another for snapped_points

    .. code-block:: python

            # USAGE EXAMPLE: To identify catchment areas around Chennai lakes.
            lakes_watershed = watershed(input_points=lakes_fs,
                                        snap_distance=10,
                                        snap_distance_units='Meters',
                                        source_database='Finest',
                                        generalize=False)
    """

    # use helper function to evaluate the input points and convert them, if necessary, to a FeatureSet
    input_fs = _evaluate_spatial_input(input_points)

    if input_fs.geometry_type != "esriGeometryPoint":
        raise Exception(
            "input_points FeatureSet must be point esriGeometryPoint, not {}.".format(
                input_fs.geometry_type
            )
        )

    input_fields = input_fs.fields
    if (
        point_id_field
        and point_id_field not in [f["name"] for f in input_fields]
        and len(input_fields)
    ):
        input_fields_str = ",".join(input_fields)
        raise Exception(
            "The provided point_id_field {} does not appear to be in the input_points FeatureSet fields - {}".format(
                point_id_field, input_fields_str
            )
        )

    if gis is None and arcgis.env.active_gis is None:
        raise Exception(
            "GIS must be defined either by directly passing in a GIS object created using credentials, or one must already be created in the active Python session."
        )
    elif gis is None:
        gis = arcgis.env.active_gis

    url = gis.properties.helperServices.hydrology.url
    tbx = _import_toolbox(url, gis=gis)
    kwargs = {
        "input_points": input_fs,
        "point_id_field": point_id_field,
        "snap_distance": snap_distance,
        "snap_distance_units": snap_distance_units,
        "source_database": source_database,
        "generalize": generalize,
        "gis": gis,
        "return_snapped_points": return_snapped_points,
        "future": future,
    }
    kwargs["future"] = True
    kwargs = inspect_function_inputs(tbx.watershed, **kwargs)
    gpjob = tbx.watershed(**kwargs)
    if future:
        return gpjob
    return gpjob.result()
