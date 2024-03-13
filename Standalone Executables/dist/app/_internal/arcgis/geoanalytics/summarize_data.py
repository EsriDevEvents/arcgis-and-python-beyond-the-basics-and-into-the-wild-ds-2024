"""
The Summarize Data module contains functions that calculate total counts, lengths, areas, and basic descriptive statistics of features and their attributes within areas or near other features.

aggregate_points calculates statistics about points that fall within specified areas or bins.
join_features calculates statistics about features that share a spatial, temporal, or attribute relationship with other features.
reconstruct_tracks calculates statistics about points or polygons that belong to the same track and reconstructs inputs into tracks.
summarize_attributes calculates statistics about feature or tabular data that share attributes.
summarize_within calculates statistics for area features and attributes that overlap each other.
"""
from __future__ import annotations
import json as _json
from datetime import datetime
import logging as _logging
from typing import Any, Optional, Union
import arcgis as _arcgis
from arcgis.features.feature import FeatureCollection
from arcgis.features.layer import FeatureLayer, FeatureLayerCollection
from arcgis.geoprocessing import import_toolbox as _import_toolbox
from arcgis._impl.common._utils import inspect_function_inputs
from arcgis.geoprocessing import DataFile
from arcgis.gis import GIS, Item
from ._util import (
    _id_generator,
    _feature_input,
    _set_context,
    _create_output_service,
    GAJob,
    _prevent_bds_item,
)

_log = _logging.getLogger(__name__)

_use_async = True


# --------------------------------------------------------------------------
def summarize_center_and_dispersion(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    summary_type: str,
    ellipse_size: Optional[int] = None,
    weight_field: Optional[str] = None,
    group_fields: Optional[str] = None,
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    context: Optional[dict[str, Any]] = None,
    future: bool = False,
):
    """
    The `summarize_center_and_dispersion` task finds central features and directional
    distributions. It can be used to answer questions such as the following:

         + Where is the center?
         + Which feature is the most accessible from all other features?
         + How dispersed, compact, or integrated are the features?
         + Are there directional trends?

    For an example, suppose you have used the GeoAnalytics tool Find Point
    Clusters to identify groups of power outages across an entire year. The
    result will be time enabled point representing cluster locations of power
    outages. However, you are interested in identifying the center of the
    power outages for visualization. To do this, you use Summarize Center And
    Dispersion a group by field of the outage cluster ids.

    ===================================================================    =============================================================================
    **Parameter**                                                                                    **Description**
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    input_layer                                                            Required Layer. A layer that will be used in analysis.
                                                                           See :ref:`Feature Input<gaxFeatureInput>`.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    summary_type                                                           Required String. The method with which to summarize the `input_layer`.
                                                                           Values: CentralFeature, MeanCenter, MedianCenter, or Ellipse.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    ellipse_size                                                           Optional Integer. The number representing the number of standard deviations
                                                                           represented in the output ellipse layer. The default ellipse size is 1. Valid
                                                                           choices are 1, 2, or 3 standard deviations. This option is only used if
                                                                           Ellipse is chosen from the `summary_type` parameter.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    weight_field                                                           Optional String. A numeric field in the input_layer to be used to weight
                                                                           locations according to their relative importance.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    group_fields                                                           Optional String. One or more fields used to group features for summarization.
                                                                           The `group_fields` can be of integer, date, or string type.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    output_name                                                            Optional string. The task will create a feature service of the results. You define the name of the service.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    gis                                                                    Optional :class:`~arcgis.gis.GIS`. The GIS object where the analysis will take place.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    context                                                                Optional string. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                                                                            * ``extent`` - a bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                                            * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                                                            * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                                            * ``dataStore`` Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    future                                                                 Optional boolean. If ``True``, a future object will be returned and the process
                                                                           will not wait for the task to complete. The default is ``False``, which means wait for results.
    ===================================================================    =============================================================================

    """

    input_layer = _prevent_bds_item(input_layer)
    tool_name = "SummarizeCenterAndDispersion"
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url

    params = {
        "input_layer": input_layer,
        "summary_type": summary_type,
        "ellipse_size": ellipse_size,
        "weight_field": weight_field,
        "group_fields": group_fields,
        "output_name": group_fields,
        "gis": gis,
        "context": context,
        "future": future,
    }

    if output_name is None:
        output_service_name = _id_generator(prefix="Sum_Cntr_and_Disp_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Summarize Center And Dispersion",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_service_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_service_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    try:
        tbx = _import_toolbox(url_or_item=url, gis=gis)
        params = inspect_function_inputs(tbx.summarize_center_and_dispersion, **params)
        params["future"] = True
        gpjob = tbx.summarize_center_and_dispersion(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise


# --------------------------------------------------------------------------
def build_multivariable_grid(
    input_layers: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    variable_calculations: list[dict[str, Any]],
    bin_size: float,
    bin_unit: str = "Meters",
    bin_type: str = "Square",
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
    context: Optional[dict[str, Any]] = None,
):
    """

    .. image:: _static/images/Grid/Grid.png

    The ``build_multivariable_grid`` task works with one or more layers of point, line, or polygon features.
    The task generates a grid of square or hexagonal bins and compiles information about each input layer into each bin.
    For each input layer, this information can include the following variables:

        * ``Distance to Nearest`` - The distance from each bin to the nearest feature.
        * ``Attribute of Nearest`` - An attribute value of the feature nearest to each bin.
        * ``Attribute Summary of Related`` - A statistical summary of all features within ``search_distance`` of each bin.

    Only variables you specify in ``variable_calculations`` will be included in the result layer. These variables can help
    you understand the proximity of your data throughout the extent of your analysis. The results can help you answer
    questions such as the following:

        * Given multiple layers of public transportation infrastructure, what part of the city is least accessible by public transportation?
        * Given layers of lakes and rivers, what is the name of the water body closest to each location in the U.S.?
        * Given a layer of household income, where in the U.S. is the variation of income in the surrounding 50 miles the greatest?

    The result of ``build_multivariable_grid`` can also be used in prediction and classification workflows. The task allows you
    to calculate and compile information from many different data sources into a single, spatially continuous layer in one step.
    This layer can then be used with the Enrich From Multi-Variable Grid task to quickly enrich point features with the variables
    you have calculated, reducing the amount of effort required to build prediction and classification models from point data.

    ===================================================================    =============================================================================
    **Parameter**                                                                                    **Description**
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    input_layers                                                           Required list of layers. A list of input layers that will be used in analysis.
                                                                           See :ref:`Feature Input<gaxFeatureInput>`.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    variable_calculations                                                  Required list of dicts. A dict containing objects that describe
                                                                           the variables that will be calculated for each layer in ``input_layers``.

                                                                           .. code-block:: python

                                                                               variable_calculations =
                                                                                [
                                                                                 {
                                                                                     "layer":<index>,
                                                                                     "variables":[
                                                                                         {
                                                                                             "type":"DistanceToNearest",
                                                                                             "outFieldName":"<output field name>",
                                                                                             "searchDistance":<number>,
                                                                                             "searchDistanceUnit":"<unit>",
                                                                                             "filter":"<filter>"
                                                                                         },
                                                                                         {
                                                                                             "type":"AttributeOfNearest",
                                                                                             "outFieldName":"<output field name>",
                                                                                             "attributeField":"<field name>",
                                                                                             "searchDistance":<number>,
                                                                                             "searchDistanceUnit":"<unit>",
                                                                                             "filter":"<filter>"
                                                                                         },
                                                                                         {
                                                                                             "type":"AttributeSummaryOfRelated",
                                                                                             "outFieldName":"<output field name>",
                                                                                             "statisticType":"<statistic type>",
                                                                                             "statisticField":"<field name>",
                                                                                             "searchDistance":<number>,
                                                                                             "searchDistanceUnit":"<unit>",
                                                                                             "filter":"<filter>"
                                                                                         },
                                                                                       ]
                                                                                     },
                                                                                   ]

                                                                           Description of the above snippet:

                                                                                * ``layer`` is the index of the layer in ``input_layers`` that will be
                                                                                  used to calculate the specified variables.

                                                                                * ``variables`` is an array of dict objects that describe the variables
                                                                                  you want to include in the result layer. The array must contain at least
                                                                                  one variable for each layer.

                                                                                * ``type`` can be one of the following variable types:

                                                                                   * DistanceToNearest
                                                                                   * AttributeOfNearest
                                                                                   * AttributeSummaryOfRelated

                                                                                * Each type must be configured with a unique set of parameters:

                                                                                  * ``outFieldName`` is the name of the field that will be created in the result
                                                                                    layer to store a variable. This is required.
                                                                                  * ``searchDistance`` is a number and ``searchDistanceUnit`` is a linear unit.

                                                                                    For:
                                                                                      * ``DistanceToNearest`` and ``AttributeOfNearest`` - both are required
                                                                                        to define the maximum distance that the tool will search from the
                                                                                        center of each bin to find a feature in the layer. If no feature is within the
                                                                                        distance, null is returned.
                                                                                      * ``AttributeSummaryOfRelated`` - , they are optional to define the radius of
                                                                                        a circular neighborhood surrounding each bin. All features that intersect this
                                                                                        neighborhood will be used to calculate ``statisticType``. If a distance is not defined,
                                                                                        only features that intersect a bin will be used to calculate ``statisticType``.
                                                                                  * ``attributeField`` is required by ``AttributeOfNearest`` and is the name of a field `
                                                                                    in the input layer. The value of this field in the closest feature to each bin will
                                                                                    be included in the result layer.
                                                                                  * ``statisticField`` is required by ``AttributeSummaryOfRelated`` and is the name of a
                                                                                    field in the input layer. This field's values will be used to calculate ``statisticType``.
                                                                                  * ``statisticType`` is required by ``AttributeSummaryOfRelated`` and is one of the following

                                                                                    * when ``statisticField`` is a numeric field:

                                                                                       * ``Count`` - Totals the number of features near or intersecting each bin.
                                                                                       * ``Sum`` - Adds the total value of all features near or intersecting each bin.
                                                                                       * ``Mean`` - Calculates the average of all features near or intersecting each bin.
                                                                                       * ``Min`` - Finds the smallest value of all features near or intersecting each bin.
                                                                                       * ``Max`` - Finds the largest value of all features near or intersecting each bin.
                                                                                       * ``Range`` - Finds the difference between Min and Max.
                                                                                       * ``Stddev`` - Finds the standard deviation of all features near or intersecting each bin.
                                                                                       * ``Var`` - Finds the variance of all features near or intersecting each bin.

                                                                                    * when ``statisticField`` is a string field:

                                                                                       * ``Count`` - Totals the number of strings for all features near or intersecting each bin.
                                                                                       * ``Any`` - Returns a sample string of all features near or intersecting each bin.

                                                                                  * ``filter`` is optional for all variable types and is formatted as described in the Feature Input topic.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    bin_size                                                               Required float. The distance for the bins of type ``bin_type`` in the output polygon layer.
                                                                           ``variable_calculations`` will be calculated at the center of each bin.
                                                                           When generating bins,
                                                                             * if ``bin_type`` is ``Square`` - the number and units specified determine the height and length of the square.
                                                                             * if ``bin_type`` is ``Hexagon`` 0 the number and units specified determine the distance between parallel sides.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    bin_unit                                                               Optional string. The distance unit for the bins that will be used to calculate ``variable_calculations``.

                                                                           Choice list:

                                                                             * ``Feet``
                                                                             * ``Yard``
                                                                             * ``Miles``
                                                                             * ``Meters``
                                                                             * ``Kilometers``
                                                                             * ``NauticalMiles``
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    bin_type                                                               Optional string. The type of bin that will be used to generate the result grid. Bin options are the following:

                                                                           Choice list:

                                                                             * ``Hexagon``
                                                                             * ``Square``

                                                                           .. note::
                                                                               Analysis using ``Square`` or ``Hexagon`` bins requires a projected coordinate system.
                                                                               When aggregating layers into bins, the input layers or processing extent (``processSR``)
                                                                               must have a projected coordinate system. If a projected coordinate system is not
                                                                               specified when running analysis, the World Cylindrical Equal Area (WKID 54034) projection
                                                                               will be used.

                                                                               * At 10.7 or later, if a projected coordinate system is not specified when
                                                                                 running analysis, a projection will be picked based on the extent of the data.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    output_name                                                            Optional string. The task will create a feature service of the results. You define the name of the service.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    context                                                                Optional string. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                                                                             * ``extent`` - a bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                                             * ``processSR`` The features will be projected into this coordinate system for analysis.
                                                                             * ``outSR`` - the features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                                             * ``dataStore`` - results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    -------------------------------------------------------------------    -----------------------------------------------------------------------------
    future                                                                 Optional boolean. If ``True``, a future object will be returned and the process
                                                                           will not wait for the task to complete. The default is ``False``, which means wait for results.
    ===================================================================    =============================================================================

    :return: boolean

    .. code-block:: python

            # Usage Example: To create multivariable grid by summarizing information such as distance to nearest

            variables = [ { "layer":0,
                            "variables":[
                                { "type":"DistanceToNearest",
                                  "outFieldName":"road",
                                  "searchDistance":10,
                                  "searchDistanceUnit":"Kilometers"
                                }
                            ]
                          },
                          { "layer":1,
                          "variables":[
                              { "type":"AttributeSummaryOfRelated",
                                "outFieldName":"MeanPopAge",
                                "statisticType":"Mean",
                                "statisticField":"Age",
                                "searchDistance":50,
                                "searchDistanceUnit":"Kilometers"
                              }
                          ]
                          }
                        ]
            grid = build_multivariable_grid(input_layers=[lyr0, lyr1],
                                            variable_calculations=variables,
                                            bin_size=100,
                                            bin_unit='Meters',
                                            bin_type='Square',
                                            output_name="multi_variable_grid")
    """

    input_layers = [_prevent_bds_item(input_layer) for input_layer in input_layers]
    flayers = []
    for il in input_layers:
        if hasattr(il, "_lyr_dict"):
            flayers.append(il._lyr_dict)
        elif hasattr(il, "_lyr_json"):
            flayers.append(il._lyr_json)
        else:
            flayers.append(il)

    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)
    params = {
        "bin_type": bin_type,
        "bin_size": bin_size,
        "bin_size_unit": bin_unit,
        "input_layers": flayers,
        "variable_calculations": variable_calculations,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": future,
    }
    for key in list(params.keys()):
        value = params[key]
        if key == "variable_calculations":
            params[key] = _json.dumps(value)
        elif value is None:
            del params[key]

    if output_name is None:
        output_service_name = _id_generator(prefix="Build Multi Variable Grid_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Build Multi Variable Grid ",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_service_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_service_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    params = inspect_function_inputs(tbx.build_multi_variable_grid, **params)
    params["future"] = True

    try:
        gpjob = tbx.build_multi_variable_grid(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise


# --------------------------------------------------------------------------
def aggregate_points(
    point_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    bin_type: Optional[str] = None,
    bin_size: Optional[float] = None,
    bin_size_unit: Optional[str] = None,
    polygon_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    time_step_interval: Optional[int] = None,
    time_step_interval_unit: Optional[str] = None,
    time_step_repeat_interval: Optional[int] = None,
    time_step_repeat_interval_unit: Optional[str] = None,
    time_step_reference: Optional[datetime] = None,
    summary_fields: Optional[list[dict[str, Any]]] = None,
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    future: bool = False,
    context: Optional[dict[str, Any]] = None,
):
    """
    .. image:: _static/images/aggregate_points/aggregate_points.png

    This ``aggregate_points`` tool works with a layer of point features and a layer of areas.
    The layer of areas can be an input polygon layer or it can be square or hexagonal bins calculated
    when the task is run. The tool first determines which points fall within each specified area.
    After determining this point-in-area spatial relationship, statistics about all points in the
    area are calculated and assigned to the area. The most basic statistic is the count of the
    number of points within the area, but you can get other statistics as well.

    For example, suppose you have point features of coffee shop locations and area features of counties,
    and you want to summarize coffee sales by county. Assuming the coffee shops have a TOTAL_SALES attribute,
    you can get the sum of all TOTAL_SALES within each county, the minimum or maximum TOTAL_SALES within each
    county, or other statistics like the count, range, standard deviation, and variance.

    This tool can also work on data that is time-enabled. If time is enabled on the input points, then
    the time slicing options are available. Time slicing allows you to calculate the point-in area relationship
    while looking at a specific slice in time. For example, you could look at hourly intervals, which would
    result in outputs for each hour.

    For an example with time, suppose you had point features of every transaction made at a coffee shop location and no area layer.
    The data has been recorded over a year, and each transaction has a location and a time stamp. Assuming each transaction has a
    TOTAL_SALES attribute, you can get the sum of all TOTAL SALES within the space and time of interest. If these transactions are
    for a single city, we could generate areas that are one kilometer grids, and look at weekly time slices to summarize the
    transactions in both time and space.

    .. note::
        Either ``bin_type`` or ``polygon_layer`` must be specified. If ``bin_type`` is used, both ``bin_size`` and
        ``bin_size_unit`` must be included.

    =================================================     ========================================================================
    **Parameter**                                          **Description**
    -------------------------------------------------     ------------------------------------------------------------------------
    point_layer                                           Required point :class:`~arcgis.features.FeatureLayer`. The point features that will be aggregated
                                                          into the polygons in the ``polygon_layer`` or bins specified by ``bin_type``.
                                                          See :ref:`Feature Input<gaxFeatureInput>`.
    -------------------------------------------------     ------------------------------------------------------------------------
    bin_type                                              Optional string. If ``polygon_layer`` is not provided, it is required.

                                                          The type of bin that will be generated and into which points will be aggregated.

                                                          Choice list:

                                                            * ``Square``
                                                            * ``Hexagon``

                                                          The default value is ``Square``.

                                                          .. note::
                                                              Required if ``polygon_layer`` not provided.
    -------------------------------------------------     ------------------------------------------------------------------------
    bin_size (Required if ``bin_type`` is used)           Optional float. The dimension for each bin of ``bin_type`` that
                                                          the ``point_layer`` will be aggregated into.

                                                            * if ``Square`` - the height and length of the sides
                                                            * if ``Hexagon`` - the distance between parallel sides
    -------------------------------------------------     ------------------------------------------------------------------------
    bin_size_unit (Required if ``bin_size`` is used)      Optional string. The unit for the bins specified by ``bin_type`` that
                                                          the ``point_layer`` will be aggregated into.

                                                          Choice list:

                                                            * ``Feet``
                                                            * ``Yards``
                                                            * ``Miles``
                                                            * ``Meters``
                                                            * ``Kilometers``
                                                            * ``NauticalMiles``

                                                          For ``bin_type``:
                                                              * if ``Square`` - the units of ``bin_size`` the height and
                                                                length of the sides
                                                              * if ``Hexagon`` -  the units of ``bin_size`` between
                                                                parallel sides
    -------------------------------------------------     ------------------------------------------------------------------------
    polygon_layer                                         Optional polygon :class:`feature layer <~arcgis.features.FeatureLayer>`.
                                                          The polygon features (areas) into which the input points will be aggregated.
                                                          See :ref:`Feature Input<gaxFeatureInput>`.

                                                          .. note::
                                                              Required if ``bin_type`` not provided.
    -------------------------------------------------     ------------------------------------------------------------------------
    time_step_interval                                    Optional integer. A numeric value that specifies duration of the time step interval. This option is only
                                                          available if the input points are time-enabled and represent an instant in time.

                                                          The default value is ``None``.
    -------------------------------------------------     ------------------------------------------------------------------------
    time_step_interval_unit                               Optional string. A string that specifies units of the time step interval.

                                                          .. note::
                                                              Only available if input points are time-enabled and represent an
                                                              instant in time.

                                                          Choice list:

                                                            * ``Years``
                                                            * ``Months``
                                                            * ``Weeks``
                                                            * ``Days``
                                                            * ``Hours``
                                                            * ``Minutes``
                                                            * ``Seconds``
                                                            * ``Milliseconds``

                                                          The default value is ``None``.
    -------------------------------------------------     ------------------------------------------------------------------------
    time_step_repeat_interval                             Optional integer. A numeric value that specifies how often the time step repeat occurs.
                                                          This option is only available if the input points are time-enabled and of time type instant.
    -------------------------------------------------     ------------------------------------------------------------------------
    time_step_repeat_interval_unit                        Optional string. A string that specifies the temporal unit of the step repeat.

                                                          .. note::
                                                              This option is only available if the input points are time-enabled and
                                                              of time type `instant`.

                                                          Choice list:
                                                            * ``Years``
                                                            * ``Months``
                                                            * ``Weeks``
                                                            * ``Days``
                                                            * ``Hours``
                                                            * ``Minutes``
                                                            * ``Seconds``
                                                            * ``Milliseconds``

                                                          The default value is ``None``.
    -------------------------------------------------     ------------------------------------------------------------------------
    time_step_reference                                   Optional datetime. A date that specifies the reference time to align the time slices to, represented in milliseconds from epoch.
                                                          The default is January 1, 1970, at 12:00 a.m. (epoch time stamp 0). This option is only available if the
                                                          input points are time-enabled and of time type instant.
    -------------------------------------------------     ------------------------------------------------------------------------
    summary_fields                                        Optional list of dicts. A list of field names and statistical summary types that you want to calculate
                                                          for all points within each polygon or bin. Note that the count of points within each polygon is always
                                                          returned. By default, all statistics are returned.

                                                          Example:
                                                          ``[{"statisticType": "Count", "onStatisticField": "fieldName1"}, {"statisticType": "Any", "onStatisticField": "fieldName2"}]``

                                                            * ``onStatisticField`` is the name of the field in the input point layer.
                                                            * ``statisticType`` is one of the following:

                                                              * for numeric fields:

                                                                * ``Count`` - Totals the number of values of all the points in each polygon.
                                                                * ``Sum`` - Adds the total value of all the points in each polygon.
                                                                * ``Mean`` - Calculates the average of all the points in each polygon.
                                                                * ``Min`` - Finds the smallest value of all the points in each polygon.
                                                                * ``Max`` - Finds the largest value of all the points in each polygon.
                                                                * ``Range`` - Finds the difference between the Min and Max values.
                                                                * ``Stddev`` - Finds the standard deviation of all the points in each polygon.
                                                                * ``Var`` - Finds the variance of all the points in each polygon.

                                                              * for string fields:

                                                                * ``Count`` - Totals the number of strings for all the points in each polygon.
                                                                * ``Any`` - Returns a sample string of a point in each polygon.
    -------------------------------------------------     ------------------------------------------------------------------------
    output_name                                           Optional string. The method will create a feature service of the results. You define the name of the service.
    -------------------------------------------------     ------------------------------------------------------------------------
    gis                                                   Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    -------------------------------------------------     ------------------------------------------------------------------------
    context                                               Optional dict. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                                                              * ``extent`` - a bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                              * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                                              * ``outSR`` - the features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                              * ``dataStore`` - results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    -------------------------------------------------     ------------------------------------------------------------------------
    future                                                Optional boolean. If ``True``, a future object will be returned and the process
                                                          will not wait for the task to complete. The default is ``False``, which means wait for results.
    =================================================     ========================================================================

    :return: result_layer : Output Features as :class:`~arcgis.features.FeatureLayer`.

    .. code-block:: python

            # Usage Example: To aggregate number of 911 calls within 1 km summarized by Day count.

            agg_result = aggregate_points(calls,
                                          bin_size=1,
                                          bin_size_unit='Kilometers',
                                          time_step_interval=1,
                                          time_step_interval_unit="Years",
                                          summary_fields=[{"statisticType": "Count", "onStatisticField": "Day"}],
                                          output_name='testaggregatepoints01')
    """
    point_layer = _prevent_bds_item(point_layer)
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)
    params = {
        "point_layer": point_layer,
        "bin_type": bin_type,
        "bin_size": bin_size,
        "bin_size_unit": bin_size_unit,
        "polygon_layer": polygon_layer or "",
        "time_step_interval": time_step_interval,
        "time_step_interval_unit": time_step_interval_unit,
        "time_step_repeat_interval": time_step_repeat_interval,
        "time_step_repeat_interval_unit": time_step_repeat_interval_unit,
        "time_step_reference": time_step_reference,
        "summary_fields": summary_fields,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": future,
    }
    for key in list(params.keys()):
        value = params[key]
        if value is None:
            del params[key]

    if context is not None:
        params["context"] = context
    if output_name is None:
        output_service_name = _id_generator(prefix="Aggregate Points_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Aggregate Points",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"

    if isinstance(summary_fields, list):
        summary_fields = _json.dumps(summary_fields)

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    params = inspect_function_inputs(tbx.aggregate_points, **params)
    params["future"] = True
    try:
        gpjob = tbx.aggregate_points(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise


# --------------------------------------------------------------------------
def describe_dataset(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    extent_output: bool = False,
    sample_size: Optional[int] = None,
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    context: Optional[dict[str, Any]] = None,
    future: bool = False,
    return_tuple: bool = False,
):
    """
    .. image:: _static/images/describe_dataset/describe_dataset.png

    The ``describe_dataset`` task provides an overview of your big data.
    The tool outputs a feature layer representing a sample of your
    input features or a single polygon feature layer that represents the extent of your
    input features. You can choose to output one, both, or none.

    For example, imagine you are tasked with completing an analysis workflow on a large
    volume of data. You want to try the workflow, but it could take hours or days with
    your full dataset. Instead of using time and resources running the full analysis,
    first create a sample layer to efficiently test your workflow before running it
    on the full dataset.

    .. note::
        Only available at ArcGIS Enterprise 10.7 and later.

    See `Describe Dataset <https://developers.arcgis.com/rest/services-reference/enterprise/describe-dataset.htm>`_
    for additional information.

    ================   ===============================================================
    **Parameter**       **Description**
    ----------------   ---------------------------------------------------------------
    input_layer        Required feature layer. The table, point, line, or polygon feature
                       layer that will be described, summarized, and sampled.
                       See :ref:`Feature Input<gaxFeatureInput>`.
    ----------------   ---------------------------------------------------------------
    extent_output      Optional boolean. The task will output a single rectangle
                       feature representing the extent of the ``input_layer`` if this value
                       is set to 'True'.

                       The default value is ``False.``
    ----------------   ---------------------------------------------------------------
    sample_size        Optional integer. The task will output a feature layer
                       representing a sample of features from the ``input_layer``. Specify
                       the number of sample features to return. If the input value is
                       0 or empty then no sample layer will be created. The output
                       will have the same schema, geometry, and time type as the input
                       layer.
    ----------------   ---------------------------------------------------------------
    output_name        Optional string. The task will create a feature layer of the results.
                       You define the name of the service.
    ----------------   ---------------------------------------------------------------
    gis                Optional :class:`~arcgis.gis.GIS`. The GIS object where the analysis will take place.
    ----------------   ---------------------------------------------------------------
    context            Optional dict. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                        * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                        * ``processSR`` - The features will be projected into this coordinate system for analysis.
                        * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                        * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    ----------------   ---------------------------------------------------------------
    future             Optional boolean. If ``True``, a future object will be returned and the process
                       will not wait for the task to complete. The default is ``False``, which means wait for results.
    ----------------   ---------------------------------------------------------------
    return_tuple       Optional boolean. If ``True``, a named tuple with multiple output keys is returned.
                       The default value is ``False``.
    ================   ===============================================================

    :return:

      * if ``return_tuple`` is ``True``, an tuple with the following keys:

         * "output_json" : dict
         * "output" : :class:`~arcgis.features.Layer`
         * "extent_layer" : :class:`~arcgis.features.FeatureLayer`
         * "sample_layer" : :class:`~arcgis.features.FeatureLayer`
         * "process_info" : list
      * if return_tuple`` is ``False``:

         * :class:`~arcgis.features.FeatureLayer` of the results.
      * if ``future`` is ``True``:

         * a job object with a ``result()`` method to access results

    .. code-block:: python

            # Usage Example: To get an overview of your big data item

            data = describe_dataset(input_layer=big_data_layer,
                                    extent_output=True,
                                    sample_size=2000,
                                    output_name="describe dataset")
    """
    input_layer = _prevent_bds_item(input_layer)
    tool_name = "DescribeDataset"
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)
    params = {
        "input_layer": input_layer,
        "sample_size": sample_size,
        "extent_output": extent_output,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": future,
    }
    for key in list(params.keys()):
        value = params[key]
        if value is None:
            del params[key]

    if output_name is None:
        output_service_name = _id_generator(prefix="Describe_Dataset_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Describe Dataset",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    params = inspect_function_inputs(tbx.describe_dataset, **params)
    params["future"] = True

    try:
        gpjob = tbx.describe_dataset(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        if return_tuple:
            return gpjob.result()
        else:
            gpjob.result()
            return output_service
    except:
        output_service.delete()
        raise


# --------------------------------------------------------------------------
def join_features(
    target_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    join_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    join_operation: str = "JoinOneToOne",
    join_fields: Optional[list[dict[str, str]]] = None,
    summary_fields: Optional[list[dict[str, Any]]] = None,
    spatial_relationship: Optional[str] = None,
    spatial_near_distance: Optional[float] = None,
    spatial_near_distance_unit: Optional[str] = None,
    temporal_relationship: Optional[str] = None,
    temporal_near_distance: Optional[int] = None,
    temporal_near_distance_unit: Optional[str] = None,
    attribute_relationship: Optional[list[dict[str, str]]] = None,
    join_condition: Optional[str] = None,
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    context: Optional[dict[str, Any]] = None,
    future: bool = False,
    keep_target: Optional[bool] = None,
):
    """
     .. image:: _static/images/join_features_geo/join_features_geo.png

     Using either feature layers or tabular data, you can join features and records based on
     specific relationships between the input layers or tables. Joins will be determined by
     spatial, temporal, and attribute relationships, and summary statistics can be optionally
     calculated.

     For example

         * Given point locations of crime incidents with a time, join the crime data to itself
           specifying a spatial relationship of crimes within 1 kilometer of each other and that
           occurred within 1 hour of each other to determine if there are a sequence of crimes
           close to each other in space and time.

         * Given a table of ZIP Codes with demographic information and area features representing
           residential buildings, join the demographic information to the residences so each
           residence now has the information.

     The ``join_features`` task works with two layers. ``join_features`` joins attributes from one
     feature to another based on spatial, temporal, and attribute relationships or some
     combination of the three. The tool determines all input features that meet the specified
     join conditions and joins the second input layer to the first. You can optionally join
     all features to the matching features or summarize the matching features.

     ``join_features`` can be applied to points, lines, areas, and tables. A temporal join
     requires that your input data is time-enabled, and a spatial join requires that your
     data has a geometry.

    See `Join Features <https://developers.arcgis.com/rest/services-reference/enterprise/join-features.htm>`_
    for additional information.

     ==========================================================================================================  =============================================================================================
     **Parameter**                                                                                                **Description**
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     target_layer                                                                                                Required layer. The table, point, line, or polygon features to be joined to. See :ref:`Feature Input<gaxFeatureInput>`.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     join_layer                                                                                                  Required layer. The point, line, or polygon features that will be joined to the ``target_layer``.
                                                                                                                 See :ref:`Feature Input<gaxFeatureInput>`.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     join_operation                                                                                              Optional string. A string representing the type of join that will be applied.

                                                                                                                 Choice list:

                                                                                                                   * ``JoinOneToOne`` - If multiple join features are found that have the same relationships
                                                                                                                     with a single target feature, the attributes from the multiple join features will be
                                                                                                                     aggregated using the specified summary statistics. For example, if a point target
                                                                                                                     feature is found within two separate polygon join features, the attributes from
                                                                                                                     the two polygons will be aggregated before being transferred to the output point
                                                                                                                     feature class. If one polygon has an attribute value of 3 and the other has a value
                                                                                                                     of 7, and a SummaryField of sum is selected, the aggregated value in the output
                                                                                                                     feature class will be 10. There will always be a Count field calculated, with a
                                                                                                                     value of 2, for the number of features specified. This is the default.
                                                                                                                   * ``JoinOneToMany`` - If multiple join features are found that have the same relationship
                                                                                                                     with a single target feature, the output feature class will contain multiple copies (records)
                                                                                                                     of the target feature. For example, if a single point target feature is found within two
                                                                                                                     separate polygon join features, the output feature class will contain two copies of the
                                                                                                                     target feature: one record with the attributes of the first polygon, and another record
                                                                                                                     with the attributes of the second polygon. There are no summary statistics calculated
                                                                                                                     with this method.

                                                                                                                 The default value is ``JoinOneToOne``.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     join_fields                                                                                                 Optional list of dicts. A list of modifications to field names in the joinLayer to be
                                                                                                                 made before completing analysis. Any field that is removed will not have
                                                                                                                 statistics calculated on it.

                                                                                                                 Syntax: ``[{ "action" : "action", "field" : "fieldname1"}, { "action" : "action", "field" : "initial_fieldname", "to" : "new_fieldname"}]``

                                                                                                                   * ``action`` can be the following:

                                                                                                                      * remove - Remove a field from analysis and output .
                                                                                                                      * rename - Rename a field before running analysis.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     summary_fields                                                                                              Optional list of dicts. A list of field names and statistical summary types you want to calculate.
                                                                                                                 Note that the count is always returned. By default, all statistics are returned.

                                                                                                                 Syntax: ``[{"statisticType" : "<statistic type>", "onStatisticField" : "<field name>" }, ...]``

                                                                                                                   * ``onStatisticField`` is the name of the field in the target layer.
                                                                                                                   * ``statisticType`` is one of the following:

                                                                                                                     * for numeric fields:

                                                                                                                       * ``Count`` - Totals the number of values of all the points in each polygon.
                                                                                                                       * ``Sum`` - Adds the total value of all the points in each polygon.
                                                                                                                       * ``Mean`` - Calculates the average of all the points in each polygon.
                                                                                                                       * ``Min`` - Finds the smallest value of all the points in each polygon.
                                                                                                                       * ``Max`` - Finds the largest value of all the points in each polygon.
                                                                                                                       * ``Range`` - Finds the difference between the Min and Max values.
                                                                                                                       * ``Stddev`` - Finds the standard deviation of all the points in each polygon.
                                                                                                                       * ``Var`` - Finds the variance of all the points in each polygon.

                                                                                                                     * for string fields:

                                                                                                                       * ``Count`` - Totals the number of strings for all the points in each polygon.
                                                                                                                       * ``Any`` - Returns a sample string of a point in each polygon.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     spatial_relationship                                                                                        Optional string. Defines the spatial relationship used to spatially join features.

                                                                                                                 Choice list:

                                                                                                                   * ``Equals``
                                                                                                                   * ``Intersects``
                                                                                                                   * ``Contains``
                                                                                                                   * ``Within``
                                                                                                                   * ``Crosses``
                                                                                                                   * ``Touches``
                                                                                                                   * ``Overlaps``
                                                                                                                   * ``Near``
                                                                                                                   * ``NearGeodesic``
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     spatial_near_distance (Required if ``spatial_relationship`` is Near or NearGeodesic)                        Optional float.  A float value used for the search distance to determine if
                                                                                                                 the target features are near the join features.

                                                                                                                 .. note::
                                                                                                                      This is only applied if ``Near`` or ``NearGeodesic`` is the selected ``spatial_relationship``.
                                                                                                                      You can only enter a single distance value. The unit value is supplied by the
                                                                                                                      ``spatial_near_distance`` parameter.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     spatial_near_distance_unit (Required if ``spatial_relationship`` is Near or NearGeodesic)                   Optional string. The linear unit to be used with the distance value specified in ``spatial_near_distance``.

                                                                                                                 Choice list:

                                                                                                                   * ``Feet``
                                                                                                                   * ``Yards``
                                                                                                                   * ``Miles``
                                                                                                                   * ``Meters``
                                                                                                                   * ``Kilometers``
                                                                                                                   * ``NauticalMiles``
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     temporal_relationship                                                                                       Optional string. Defines the temporal relationship used to temporally join features.

                                                                                                                 Choice list :

                                                                                                                   * ``Equals``
                                                                                                                   * ``Intersects``
                                                                                                                   * ``During``
                                                                                                                   * ``Contains``
                                                                                                                   * ``Finishes``
                                                                                                                   * ``FinishedBy``
                                                                                                                   * ``Meets``
                                                                                                                   * ``MetBy``
                                                                                                                   * ``Overlaps``
                                                                                                                   * ``OverlappedBy``
                                                                                                                   * ``Starts``
                                                                                                                   * ``StartedBy``
                                                                                                                   * ``Near``
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     temporal_near_distance (Required if ``temporal_relationship`` is Near, NearBefore, or NearAfter)            Optional integer. An integer value used for the temporal search distance to determine
                                                                                                                 if the target features are temporally near the join features.

                                                                                                                 .. note::
                                                                                                                     This is only applied if ``Near``, ``NearBefore``, or ``NearAfter`` is the selected ``temporal_relationship``.
                                                                                                                     You can only enter a single value. distance value. The units of the distance values are supplied by the
                                                                                                                     ``temporal_near_distance_unit`` parameter.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     temporal_near_distance_unit (Required if ``temporal_relationship`` is Near, NearBefore, or NearAfter)       Optional string.The temporal unit to be used with the distance value specified in ``temporal_near_distance``.

                                                                                                                 Choice list:

                                                                                                                   * ``Years``
                                                                                                                   * ``Months``
                                                                                                                   * ``Weeks``
                                                                                                                   * ``Days``
                                                                                                                   * ``Hours``
                                                                                                                   * ``Minutes``
                                                                                                                   * ``Seconds``
                                                                                                                   * ``Milliseconds``
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     attribute_relationship                                                                                      Optional list of dicts. A target field, relationship, and join field used to join equal attributes.

                                                                                                                 Syntax: ``[{ "targetField" : "fieldname1", "joinField" : "fieldname2", "operator" : "operator" }]``
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     join_condition                                                                                              Optional string. Applies a condition to specified fields. Only features with fields that meet
                                                                                                                 these conditions will be joined. For example, to apply a join to a dataset for only those features
                                                                                                                 where health_spending is greater than 20 percent of income, apply a join condition of target['health_spending'] > (join['income'] * .20)
                                                                                                                 using the field health_spending from the first dataset (``target_layer``) and the income field from the second dataset (``join_layer``).
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     output_name                                                                                                 Optional string. The task will create a feature service of the
                                                                                                                 results. You define the name of the service.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     gis                                                                                                         Optional :class:`~arcgis.gis.GIS`. The GIS object where the analysis will take place.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     context                                                                                                     Optional dict. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                                                                                                                   * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                                                                                   * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                                                                                                   * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                                                                                   * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     future                                                                                                      Optional boolean. If ``True``, a GPJob is returned instead of
                                                                                                                 results. The GPJob can be queried on the status of the execution.

                                                                                                                 The default value is ``False``.
     ----------------------------------------------------------------------------------------------------------  ---------------------------------------------------------------------------------------------
     keep_target                                                                                                 Optional boolean. Specifies whether all target features will be maintained in the output
                                                                                                                 feature class (known as a left outer join) or only those that have the specified
                                                                                                                 relationships with the join features (inner join). This option is only available when the
                                                                                                                 ``join_operation`` parameter is `JoinOneToOne``. ``False`` (inner join) is the default.

                                                                                                                 .. note::
                                                                                                                     This parameter is available at ArcGIS Enterprise 10.9 and later.
     ==========================================================================================================  =============================================================================================

     :return: Output Features as :class:`~arcgis.features.FeatureLayerCollection`

     .. code-block:: python

             # Usage Example: To find power outages in your state that may have been caused by a lightning strike.

             output = join_features(target_layer=outages_layer,
                                    join_layer=lightning,
                                    join_operation="JoinOneToMany",
                                    spatial_relationship="Near",
                                    spatial_near_distance=20,
                                    spatial_near_distance_unit="Miles",
                                    temporal_relationship="NearAfter",
                                    temporal_near_distance=30,
                                    temporal_near_distance_unit="Minutes",
                                    output_name="LightningOutages")
    """

    target_layer = _prevent_bds_item(target_layer)
    join_layer = _prevent_bds_item(join_layer)
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)
    params = {
        "target_layer": target_layer,
        "join_layer": join_layer,
        "join_operation": join_operation,
        "join_fields": join_fields,
        "summary_fields": summary_fields,
        "spatial_relationship": spatial_relationship,
        "spatial_near_distance": spatial_near_distance,
        "spatial_near_distance_unit": spatial_near_distance_unit,
        "temporal_relationship": temporal_relationship,
        "temporal_near_distance": temporal_near_distance,
        "temporal_near_distance_unit": temporal_near_distance_unit,
        "attribute_relationship": attribute_relationship,
        "join_condition": join_condition,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": future,
        "keep_all_target_features": keep_target,
    }
    for key in list(params.keys()):
        value = params[key]
        if value is None:
            del params[key]

    if output_name is None:
        output_service_name = _id_generator(prefix="Join_Features_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Join Features",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    params = inspect_function_inputs(tbx.join_features, **params)
    params["future"] = True

    try:
        gpjob = tbx.join_features(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise


# --------------------------------------------------------------------------
def reconstruct_tracks(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    track_fields: str,
    method: str = "Planar",
    buffer_field: Optional[str] = None,
    summary_fields: Optional[list[dict[str, Any]]] = None,
    distance_split: Optional[int] = None,
    distance_split_unit: Optional[str] = None,
    time_boundary_split: Optional[int] = None,
    time_boundary_split_unit: Optional[str] = None,
    time_boundary_reference: Optional[datetime] = None,
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    time_split: Optional[int] = None,
    time_split_unit: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    future: bool = False,
    arcade_split: Optional[str] = None,
    split_boundary: Optional[str] = None,
):
    """
    .. image:: _static/images/reconstruct_tracks/reconstruct_tracks.png

    The ``reconstruct_tracks`` task works with a time-enabled layer of either point or polygon
    features that represents an instant in time. It first determines which features belong to a
    track using an identifier. Using the time at each location, the tracks are ordered sequentially
    and transformed into a line or polygon representing the path of movement over time. Optionally,
    the input can be buffered by a field, which will create a polygon at each location. These buffered
    points, or polygons if the inputs are polygons, are then joined sequentially to create a track as a
    polygon where the width is representative of the attribute of interest. Resulting tracks have start
    and end times that represent the time at the first and last feature in a given track. When the tracks
    are created, statistics about the input features are calculated and assigned to the output track. The
    most basic statistic is the count of points within the area, but other statistics can be calculated as
    well. Features in time-enabled layers can be represented in one of two ways:

        * Instant - A single moment in time
        * Interval - A start and end time

    For example, suppose you have GPS measurements of hurricanes every 10 minutes. Each GPS measurement records
    the hurricane name, location, time of recording, and the wind speed. You could create tracks of the hurricanes
    using the name of the hurricane as the track identification, and all hurricanes' tracks would be generated.
    You could calculate statistics such as the mean, maximum, and minimum wind speed of each hurricane, as well
    as the count of measurements in each track.

    ======================================================================================  ===============================================================
    **Parameter**                                                                            **Description**
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    input_layer                                                                             Required layer. The point or polygon features from which tracks
                                                                                            will be constructed. See :ref:`Feature Input<gaxFeatureInput>`.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    track_fields                                                                            Required string. The fields used to identify distinct tracks. There can
                                                                                            be multiple ``track_fields``.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    method                                                                                  Optional string. The method used to apply reconstruct tracks and, optionally,
                                                                                            to apply the buffer. There are two methods to choose from:

                                                                                            * ``Planar`` - This method joins points using a plane method and will not
                                                                                              cross the international date line. For buffers, this method applies a Euclidean
                                                                                              buffer and is appropriate for local analysis on projected data. This is the default.
                                                                                            * ``Geodesic`` - This method joins points geodesically and will allow tracks to cross
                                                                                              the international date line. For buffers, this method is appropriate for large areas
                                                                                              and any geographic coordinate system.

                                                                                            The default value is 'Planar'.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    buffer_field                                                                            Optional string. A field in the ``input_layer`` that contains a buffer distance or a buffer expression.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    summary_fields                                                                          Optional list of dicts. A list of field names and statistical summary types you want to calculate.

                                                                                            .. note::
                                                                                                ``count`` is always returned. By default, all statistics are returned.

                                                                                            Syntax: ``[{"statisticType" : "<statistic type>", "onStatisticField" : "<field name>" }, ...]``

                                                                                              * ``onStatisticField`` is the name of the field in the target layer.
                                                                                              * ``statisticType`` is one of the following:

                                                                                                * for numeric fields:

                                                                                                  * ``Count`` - Totals the number of values of all the points in each polygon.
                                                                                                  * ``Sum`` - Adds the total value of all the points in each polygon.
                                                                                                  * ``Mean`` - Calculates the average of all the points in each polygon.
                                                                                                  * ``Min`` - Finds the smallest value of all the points in each polygon.
                                                                                                  * ``Max`` - Finds the largest value of all the points in each polygon.
                                                                                                  * ``Range`` - Finds the difference between the Min and Max values.
                                                                                                  * ``Stddev`` - Finds the standard deviation of all the points in each polygon.
                                                                                                  * ``Var`` - Finds the variance of all the points in each polygon.
                                                                                                  * ``First`` - Returns the first value of a specified field in the summarized track.
                                                                                                    This parameters was introduced at ArcGIS Enterprise 10.8.1.
                                                                                                  * ``Last`` - Returns the last value of a specified field in the summarized track.
                                                                                                    This parameters was introduced at ArcGIS Enterprise 10.8.1.

                                                                                                * for string fields:

                                                                                                  * ``Count`` - Totals the number of strings for all the points in each polygon.
                                                                                                  * ``Any`` - Returns a sample string of a point in each polygon.e of all the points in each polygon.
                                                                                                  * ``First`` - Returns the first value of a specified field in the summarized track.
                                                                                                    This parameters was introduced at ArcGIS Enterprise 10.8.1.
                                                                                                  * ``Last`` - Returns the last value of a specified field in the summarized track.
                                                                                                    This parameters was introduced at ArcGIS Enterprise 10.8.1.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    time_boundary_split                                                                     Optional integer. A time boundary allows your to analyze values within a defined
                                                                                            time span. For example, if you use a time boundary of 1 day, starting on January 1st,
                                                                                            1980 tracks will be analyzed 1 day at a time. The time boundary parameter was introduced
                                                                                            in ArcGIS Enterprise 10.7.

                                                                                            The ``time_boundary_split`` parameter defines the scale of the time boundary. In the
                                                                                            case above, this would be 1.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    time_boundary_split_unit (Required if ``time_boundary_split`` is specified)             Optional string. The unit applied to the time boundary. Required
                                                                                            if a ``time_boundary_split`` is provided.

                                                                                            Choice list:

                                                                                              * ``Years``
                                                                                              * ``Months``
                                                                                              * ``Weeks``
                                                                                              * ``Days``
                                                                                              * ``Hours``
                                                                                              * ``Minutes``
                                                                                              * ``Seconds``
                                                                                              * ``Milliseconds``
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    time_boundary_reference                                                                 Optional datetime.datetime. A date that specifies the reference time to align the time boundary to,
                                                                                            represented in milliseconds from epoch.
                                                                                            This option is only available if the ``time_boundary_split`` and ``time_boundary_split_unit`` are set.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    distance_split                                                                          Optional float. A distance used to split tracks. Any features in the ``input_layer`` that are in the same
                                                                                            track and are greater than this distance apart will be split into a new track. The units of the distance
                                                                                            values are supplied by the ``distance_split_unit`` parameter.

                                                                                            If both ``distance_split`` and ``time_split`` are used, the track is split when at least one condition is met.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    distance_split_unit (Required if ``distance_split`` is specified)                       Optional string. The distance unit to be used with the distance value specified in ``distance_split``.

                                                                                            Choice list:

                                                                                              * ``Meters``
                                                                                              * ``Kilometers``
                                                                                              * ``Feet``
                                                                                              * ``Miles``
                                                                                              * ``NauticalMiles``
                                                                                              * ``Yards``
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    output_name                                                                             Optional string. The task will create a feature service of the results. You define the name of the service.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    gis                                                                                     Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    time_split                                                                              Optional integer. A time duration used to split tracks. Any features in the ``input_layer`` that are in
                                                                                            the same track and are greater than this time apart will be split into a new track. The units of the distance
                                                                                            values are supplied by the ``time_split`` parameter.

                                                                                            If both ``distance_split`` and ``time_split`` are used, a track is split when at least one condition is met.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    time_split_unit  (Required if ``time_split`` is specified)                              Optional string. The temporal unit to be used with the temporal distance value specified in ``time_split``.

                                                                                            Choice list:

                                                                                              * ``Years``
                                                                                              * ``Months``
                                                                                              * ``Weeks``
                                                                                              * ``Days``
                                                                                              * ``Hours``
                                                                                              * ``Minutes``
                                                                                              * ``Seconds``
                                                                                              * ``Milliseconds``
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    context                                                                                 Optional dict. The context parameter contains additional settings that affect task execution.
                                                                                            For this task, there are four settings:

                                                                                              * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                                                              * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                                                                              * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                                                              * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    future                                                                                  Optional boolean. If ``True``, a GPJob is returned instead of results. The GPJob can be queried on the status of the execution.

                                                                                            The default value is ``False``.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    arcade_split                                                                            Optional String.  An expression that splits tracks based on values,
                                                                                            geometry, or time values. Expressions that validate to true will be
                                                                                            split. This parameter is only available with ArcGIS Enterprise 10.9
                                                                                            and later.  The default is `None`.
    --------------------------------------------------------------------------------------  ---------------------------------------------------------------
    split_boundary                                                                          Optional String.

                                                                                            Specifies how the track segment between two features is created
                                                                                            when a track is split. The split type is applied to split
                                                                                            expressions, distance splits, and time splits. This parameter
                                                                                            is only available with ArcGIS Enterprise 10.9 and later.

                                                                                            - `Gap` - No segment is created between the two features. This is the default when `None` is specified.
                                                                                            - `FinishLast` - A segment is created between the two features that ends after the split.
                                                                                            - `StartNext` - A segment is created between the two features that ends before the split.

                                                                                            The default is `None`.
    ======================================================================================  ===============================================================

    :return: :class:`~arcgis.features.FeatureLayerCollection`

    .. code-block:: python

            # Usage Example: To reconstruct hurricane tracks.

            tracks = reconstruct_tracks(input_layer=hurricane_lyr,
                                        track_fields='season, trackID',
                                        method='Geodesic',
                                        buffer_field='size',
                                        summary_fields=[{"statisticType" : "Range", "onStatisticField" : "Wind" }],
                                        distance_split=1,
                                        distance_split_unit='Kilometers',
                                        time_boundary_split=1,
                                        time_boundary_split_unit='Days',
                                        output_name='reconstruct hurricane tracks')
    """

    input_layer = _prevent_bds_item(input_layer)
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)
    params = {
        "input_layer": input_layer,
        "track_fields": track_fields,
        "method": method,
        "buffer_field": buffer_field,
        "summary_fields": summary_fields,
        "time_split": time_split,
        "time_split_unit": time_split_unit,
        "distance_split": distance_split,
        "distance_split_unit": distance_split_unit,
        "time_boundary_split": time_boundary_split,
        "time_boundary_split_unit": time_boundary_split_unit,
        "time_boundary_reference": time_boundary_reference,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": future,
        "arcade_split": arcade_split,
        "split_boundary_option": split_boundary,
    }
    for key in list(params.keys()):
        value = params[key]
        if value is None:
            del params[key]

    if output_name is None:
        output_service_name = _id_generator(prefix="Reconstruct_Tracks_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Reconstruct Tracks",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    if isinstance(summary_fields, list):
        summary_fields = _json.dumps(summary_fields)
        params["summary_fields"] = summary_fields

    params = inspect_function_inputs(tbx.reconstruct_tracks, **params)
    params["future"] = True
    try:
        gpjob = tbx.reconstruct_tracks(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise


# --------------------------------------------------------------------------
def summarize_attributes(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    fields: Optional[str] = None,
    summary_fields: Optional[list[dict[str, Any]]] = None,
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    context: Optional[dict[str, Any]] = None,
    future: bool = False,
    time_step_interval: Optional[int] = None,
    time_step_interval_unit: Optional[str] = None,
    time_step_repeat_interval: Optional[int] = None,
    time_step_repeat_interval_unit: Optional[str] = None,
    time_step_reference: Optional[datetime] = None,
):
    """
    .. image:: _static/images/summarize_attributes/summarize_attributes.png

    ===========================================================================  ===============================================================
    **Parameter**                                                                 **Description**
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    input_layer                                                                  Required layer. The features that will be summarized. See :ref:`Feature Input<gaxFeatureInput>`.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    fields                                                                       Optional string. The fields that will be used to summarize like features. For example,
                                                                                 if you chose a field called property type with the values of commercial and residential,
                                                                                 all of the features with property type residential would be summarized together with
                                                                                 summary statistics calculated and all of the commercial features would be summarized together.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    summary_fields                                                               Optional list of dicts. A list of field names and statistical summary types you want
                                                                                 to calculate for features that are summarized together. Note that the count of features
                                                                                 with the same fields values is always returned. By default, all statistics are returned.

                                                                                 Syntax: ``[{"statisticType" : "<statistic type>", "onStatisticField" : "<field name>" }]``

                                                                                 * ``onStatisticField`` is the name of the field in the input point layer
                                                                                 * ``statisticType`` is one of the following:

                                                                                   * for numeric fields:

                                                                                      * ``Count`` - Totals the number of values of all the points in each polygon.
                                                                                      * ``Sum`` - Adds the total value of all the points in each polygon.
                                                                                      * ``Mean`` - Calculates the average of all the points in each polygon.
                                                                                      * ``Min`` - Finds the smallest value of all the points in each polygon.
                                                                                      * ``Max`` - Finds the largest value of all the points in each polygon.
                                                                                      * ``Range`` - Finds the difference between the Min and Max values.
                                                                                      * ``Stddev`` - Finds the standard deviation of all the points in each polygon.
                                                                                      * ``Var`` - Finds the variance of all the points in each polygon.

                                                                                   * for string fields:

                                                                                      * Count - Totals the number of strings for all the points in each polygon.
                                                                                      * Any - Returns a sample string of a point in each polygon.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    output_name                                                                  Optional string. The task will create a feature service of the results. You define the name of the service.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    gis                                                                          Optional :class:`~arcgis.gis.GIS`. The GIS on which this tool runs. If not specified, the active GIS is used.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    context                                                                      Optional dict. Context contains additional settings that affect task execution. For this task,
                                                                                 there is one setting:

                                                                                   * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                                                   * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    future                                                                       Optional boolean. If ``True``, a GPJob is returned instead of results. The GPJob can be queried on the status of the execution.

                                                                                 The default value is ``False``.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    time_step_interval                                                           Optional integer. A numeric value that specifies duration of the time step interval. This option is only
                                                                                 available if the input points are time-enabled and represent an instant in time.

                                                                                 The default value is ``None``.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    time_step_interval_unit                                                      Optional string. A string that specifies units of the time step interval. This option is only available if the
                                                                                 input points are time-enabled and represent an instant in time.

                                                                                 Choice list:

                                                                                   * ``Years``
                                                                                   * ``Months``
                                                                                   * ``Weeks``
                                                                                   * ``Days``
                                                                                   * ``Hours``
                                                                                   * ``Minutes``
                                                                                   * ``Seconds``
                                                                                   * ``Milliseconds``

                                                                                 The default value is ``None``.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    time_step_repeat_interval                                                    Optional integer. A numeric value that specifies how often the time step repeat occurs.
                                                                                 This option is only available if the input points are time-enabled and of time type instant.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    time_step_repeat_interval_unit                                               Optional string. A string that specifies the temporal unit of the step repeat.
                                                                                 This option is only available if the input points are time-enabled and of time type instant.

                                                                                 Choice list:

                                                                                   * ``Years``
                                                                                   * ``Months``
                                                                                   * ``Weeks``
                                                                                   * ``Days``
                                                                                   * ``Hours``
                                                                                   * ``Minutes``
                                                                                   * ``Seconds``
                                                                                   * ``Milliseconds`

                                                                                 The default value is ``None``.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    time_step_reference                                                          Optional datetime. A date that specifies the reference time to align the time slices to, represented in milliseconds from epoch.
                                                                                 The default is January 1, 1970, at 12:00 a.m. (epoch time stamp 0). This option is only available if the
                                                                                 input points are time-enabled and of time type instant.
    ===========================================================================  ===============================================================

    :return:
        :class:`~arcgis.features.FeatureLayerCollection`

    .. code-block:: python

            # Usage Example: To summarize similar types of storms to find the sum of property damage.

            summarized_result = summarize_attributes(input_layer=storms,
                                                     fields="Storm_type",
                                                     summary_fields=[{"statisticType" : "Sum", "onStatisticField" : "PropertyDamage"}],
                                                     output_name="summarized_storms")
    """

    input_layer = _prevent_bds_item(input_layer)
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)
    params = {
        "input_layer": input_layer,
        "fields": fields,
        "summary_fields": summary_fields,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": future,
        "time_step_interval": time_step_interval,
        "time_step_interval_unit": time_step_interval_unit,
        "time_step_repeat_interval": time_step_repeat_interval,
        "time_step_repeat_interval_unit": time_step_repeat_interval_unit,
        "time_step_reference": time_step_reference,
    }
    for key in list(params.keys()):
        value = params[key]
        if value is None:
            del params[key]

    if output_name is None:
        output_service_name = _id_generator(prefix="Summarize Attributes_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Summarize Attributes",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"

    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    if isinstance(summary_fields, list):
        summary_fields = _json.dumps(summary_fields)
        params["summary_fields"] = summary_fields

    params = inspect_function_inputs(tbx.summarize_attributes, **params)
    params["future"] = True
    try:
        gpjob = tbx.summarize_attributes(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise


# --------------------------------------------------------------------------
def summarize_within(
    summarized_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    summary_polygons: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    bin_type: Optional[str] = None,
    bin_size: Optional[float] = None,
    bin_size_unit: Optional[str] = None,
    standard_summary_fields: Optional[list[dict[str, Any]]] = None,
    weighted_summary_fields: Optional[list[dict[str, Any]]] = None,
    sum_shape: bool = True,
    shape_units: Optional[str] = None,
    group_by_field: Optional[str] = None,
    minority_majority: bool = False,
    percent_shape: bool = False,
    output_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    context: Optional[dict[str, Any]] = None,
    future: bool = False,
):
    """
    .. image:: _static/images/summarize_within_geo/summarize_within_geo.png

    The ``summarize_within`` task finds features (and portions of features) that are within the
    boundaries of areas in the first input layer. The following are examples:

        * Given a layer of watershed boundaries and a layer of land-use boundaries, calculate the total acreage of land-use type for each watershed.
        * Given a layer of parcels in a county and a layer of city boundaries, summarize the average value of vacant parcels within each city boundary.
        * Given a layer of counties and a layer of roads, summarize the total mileage of roads by road type within each county.

    You can think of ``summarize_within`` as taking two layers and stacking them on top of each other. One of the layers,
    ``summary_polygons``, must be a polygon layer, and imagine that these polygon boundaries are all colored red. The other layer,
    ``summarized_layer``, can be any feature type—point, line, or polygon. After stacking these layers on top of each other, you
    peer down through the stack and count the number of features in ``summarized_layer`` that fall within the polygons with the
    red boundaries (``summary_polygons``). Not only can you count the number of features, you can calculate simple statistics about
    the attributes of the features in ``summarized_layer``, such as sum, mean, minimum, maximum, and so on.

    .. note::
        Either ``summary_polygons`` or ``bin_type`` must be specified.

    ===========================================================================  ===============================================================
    **Parameter**                                                                 **Description**
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    summarized_layer                                                             Required layer. Point, line, or polygon features that will be summarized for each
                                                                                 polygon in ``summary_polygons`` or bins. See :ref:`Feature Input<gaxFeatureInput>`.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    summary_polygons                                                             Optional layer. The polygon features. Features, or portions of features,
                                                                                 in ``summarized_layer`` that fall within the boundaries of these polygons
                                                                                 will be summarized. You can choose to summarize within a polygon layer that you
                                                                                 provide or within square or hexagon bins that are generated when the tool runs.
                                                                                 See :ref:`Feature Input<gaxFeatureInput>`.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    bin_type (Required if ``summary_polygons`` is not specified)                 Optional string. The type of bin that will be generated and ``summarized_layer`` will be summarized into.

                                                                                 Choice list:

                                                                                   * ``Hexagon``
                                                                                   * ``Square``

                                                                                 .. note::
                                                                                     If ``bin_type`` is chosen, ``bin_size`` and ``bin_size_unit`` are required.

                                                                                 .. note::
                                                                                       Analysis using ``Square`` or ``Hexagon`` bins requires a projected coordinate system.
                                                                                       When aggregating layers into bins, the input layer or processing extent (``processSR``)
                                                                                       must have a projected coordinate system. If a projected coordinate system is not specified:

                                                                                           * At 10.5.1, 10.6, and 10.6.1, the World Cylindrical Equal Area (WKID 54034) projection will be used.
                                                                                           * At 10.7 or later, a projection will be picked based on   the extent of the data.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    bin_size  (Required if ``bin_type`` is specified)                            Optional float. The distance for the bins of type ``bin_type``.
                                                                                 When generating bins, for Square, the number and units specified determine the
                                                                                 height and length of the square, and for Hexagon, the number and units specified
                                                                                 determine the distance between parallel sides.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    bin_size_unit (Required if ``bin_size`` is specified)                        Optional string. The linear distance unit for the bins that ``summarized_layer`` will be summarized into.

                                                                                 Choice list:

                                                                                   * ``Feet``
                                                                                   * ``Yards``
                                                                                   * ``Miles``
                                                                                   * ``Meters``
                                                                                   * ``Kilometers``
                                                                                   * ``NauticalMiles``
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    standard_summary_fields                                                      Optional list of dicts. A list of field names and statistical summary type that you want to calculate
                                                                                 for all features in ``summarized_layer`` that are within each polygon in ``summary_polygons`` or ``bin_type`` bins.
                                                                                 The standard statistics are calculated using the whole attribute values from any feature
                                                                                 that is within ``summary_polygons``.

                                                                                 Syntax: ``[{"statisticType" : "<statistic type>", "onStatisticField" : "<field name>" }]``

                                                                                   * ``onStatisticField`` is the name of the field in the input point layer.
                                                                                   * ``statisticType`` is one of the following

                                                                                     * for numeric fields:

                                                                                       * ``Count`` - Totals the number of features in each polygon.
                                                                                       * ``Sum`` - Adds the total value of all the features in each polygon.
                                                                                       * ``Mean`` - Calculates the average of all the features in each polygon.
                                                                                       * ``Min`` - Finds the smallest value of all the features in each polygon.
                                                                                       * ``Max`` - Finds the largest value of all the features in each polygon.
                                                                                       * ``Range`` - Finds the difference between Min and Max.
                                                                                       * ``Stddev`` - Finds the standard deviation of all the features in each polygon.
                                                                                       * ``Var`` - Finds the variance of all the features in each polygon.

                                                                                     * for string fields:

                                                                                       * ``Count`` - Totals the number of strings for all the features in each polygon.
                                                                                       * ``Any`` - Returns a sample string of a feature in each polygon.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    weighted_summary_fields                                                      Optional list of dicts. A list of field names and statistical summary type that you want to calculate
                                                                                 for all features in ``summarized_layer`` that are within each polygon in ``summary_polygons`` or ``bin_type`` bins.
                                                                                 The weighted statistics are calculated using the geographically weighted attribute values
                                                                                 from features that are within ``summary_polygons``. Resulting fields from proportional statistics
                                                                                 will be denoted with a ``p``. Weighted statistics can only be applied to a ``summarized_layer`` with
                                                                                 either line or polygon geometry.

                                                                                 Syntax: ``[{"statisticType" : "<statistic type>", "onStatisticField" : "<field name>" }]``

                                                                                   * ``onStatisticField`` is the name of the field in the input point layer.
                                                                                   * ``statisticType`` is one of the following:

                                                                                     * for numeric fields:

                                                                                       * ``Count`` - The count of each field multiplied by the proportion of the summarized layer within the polygons.
                                                                                       * ``Sum`` - The sum of weighted of values in each field. Where the weight applied is the proportion of the summarized layer within the polygons.
                                                                                       * ``Mean`` - The weighted mean of values in each field. Where the weight applied is the proportion of the summarized layer within the polygons.
                                                                                       * ``Min`` - The minimum of weighted values in each field. Where the weight applied is the proportion of the summarized layer within the polygons.
                                                                                       * ``Max`` - The maximum of weighted values in each field. Where the weight applied is the proportion of the summarized layer within the polygons.
                                                                                       * ``Range`` - Finds the difference between Min and Max.
                                                                                       * ``Stddev`` - The standard deviation of weighted values in each field. Where the weight applied is the proportion of the summarized layer within the polygons. (Added 10.9.1)
                                                                                       * ``Var`` - The variance of weighted values in each field. Where the weight applied is the proportion of the summarized layer within the polygons. (Added 10.9.1)

    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    sum_shape                                                                    Optional boolean. A boolean value that instructs the task to calculate statistics based on the
                                                                                 shape type of ``summarized_layer``, such as the length of lines or areas of polygons
                                                                                 of ``summarized_layer`` within each polygon in ``summary_polygons``.

                                                                                 The default value is ``True``.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    shape_units                                                                  Optional string. The units used to calculate ``sum_shape``.

                                                                                 Values:

                                                                                    * When ``summarized_layer`` contains polygons, Choice list: ['Acres', 'Hectares', 'SquareMeters', 'SquareKilometers', 'SquareMiles', 'SquareYards', 'SquareFeet'].
                                                                                    * When ``summarized_layer`` contains lines, Choice list: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles']
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    group_by_field                                                               Optional string. This is a field of the ``summarized_layer`` features that you can use to calculate
                                                                                 statistics separately for each unique attribute value. For example, suppose the ``summarized_layer``
                                                                                 depicts city boundaries and the ``summary_polgyons`` features are parcels. `
                                                                                 The parcels layer has a `Status` attribute whose value is either `VACANT` or `OCCUPIED``.
                                                                                 To calculate the total area of vacant and occupied parcels within the boundaries of cities,
                                                                                 use `Status` as the ``group_by_field`` field argument.

                                                                                 .. note::
                                                                                     This parameter is available at ArcGIS Enterprise 10.6.1 and later.

                                                                                 When a ``group_by_field`` field is provided, the service returns a table containing the
                                                                                 statistics in the ``groupBySummaryoutput`` parameter.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    minority_majority                                                            Optioal boolean. This boolean parameter is applicable only when a ``group_by_field`` is specified.
                                                                                 If true, the minority (least dominant) or the majority (most dominant) attribute values
                                                                                 for each group field are calculated. Two new fields are added to the ``result_layer`` prefixed with
                                                                                 `Majority_` and `Minority_`.

                                                                                 .. note::
                                                                                     This parameter is available at ArcGIS Enterprise 10.6.1 and later.

                                                                                 The default value is 'False'.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    percent_shape                                                                Optioal boolean. This boolean parameter is applicable only when a ``group_by_field`` is specified.
                                                                                 If set to true, the percentage of each unique ``group_by_field`` value is calculated for
                                                                                 each sum within layer polygon. The default is false.

                                                                                 .. note::
                                                                                     This parameter is available at ArcGIS Enterprise 10.6.1 and later.

                                                                                 The default value is ``False``.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    output_name                                                                  Optional string. The task will create a feature service of the results. You define the name of the service.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    context                                                                      Optional dict. The context parameter contains additional settings that affect task execution. For this task, there are four settings:

                                                                                  * ``extent`` - A bounding box that defines the analysis area. Only those features that intersect the bounding box will be analyzed.
                                                                                  * ``processSR`` - The features will be projected into this coordinate system for analysis.
                                                                                  * ``outSR`` - The features will be projected into this coordinate system after the analysis to be saved. The output spatial reference for the spatiotemporal big data store is always WGS84.
                                                                                  * ``dataStore`` - Results will be saved to the specified data store. For ArcGIS Enterprise, the default is the spatiotemporal big data store.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    gis                                                                          Optional :class:`~arcgis.gis.GIS`. The GIS on which this tool runs. If not specified, the active GIS is used.
    ---------------------------------------------------------------------------  ---------------------------------------------------------------
    future                                                                       Optional boolean. If ``True``, a future object will be returned and the process
                                                                                 will not wait for the task to complete. The default is ``False``, which means wait for results.
    ===========================================================================  ===============================================================

    :return:
        :class:`~arcgis.features.FeatureLayer`.

    .. code-block:: python

            # Usage Example: To calculate the distance and average slope of bike lanes within each city district.

            summarize_within_result = summarize_within(summary_polygons=districts,
                                                       summarized_layer=bike_lanes,
                                                       weighted_summary_fields=[{"statisticType" : "Average","onStatisticField" : "Slope"}],
                                                       output_name="summary_of_bike_lanes")
    """
    summarized_layer = _prevent_bds_item(summarized_layer)
    gis = _arcgis.env.active_gis if gis is None else gis
    url = gis.properties.helperServices.geoanalytics.url
    tbx = _import_toolbox(url, gis=gis)

    params = {
        "summary_polygons": summary_polygons or "",
        "bin_type": bin_type,
        "bin_size": bin_size,
        "bin_size_unit": bin_size_unit,
        "summarized_layer": summarized_layer,
        "standard_summary_fields": standard_summary_fields,
        "weighted_summary_fields": weighted_summary_fields,
        "sum_shape": sum_shape,
        "shape_units": shape_units,
        "group_by_field": group_by_field,
        "minority_majority": minority_majority,
        "percent_shape": percent_shape,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": future,
    }
    for key in list(params.keys()):
        value = params[key]
        if value is None:
            del params[key]

    if output_name is None:
        output_service_name = _id_generator(prefix="Summarize Within_")
        output_name = output_service_name.replace(" ", "_")
    else:
        output_service_name = output_name.replace(" ", "_")
    if context is not None:
        output_datastore = context.get("dataStore", None)
    else:
        output_datastore = None
    output_service = _create_output_service(
        gis,
        output_name,
        output_service_name,
        "Summarize Within",
        output_datastore=output_datastore,
    )

    if output_service:
        params["output_name"] = _json.dumps(
            {
                "serviceProperties": {
                    "name": output_name,
                    "serviceUrl": output_service.url,
                },
                "itemProperties": {"itemId": output_service.itemid},
            }
        )
    else:
        params["output_name"] = output_name
        output_service = f"Results were written to: '{params['context']['dataStore']}' with the name: '{output_name}'"
    if context is not None:
        params["context"] = context
    else:
        _set_context(params)

    params = inspect_function_inputs(tbx.summarize_within, **params)
    params["future"] = True
    try:
        gpjob = tbx.summarize_within(**params)
        if future:
            return GAJob(gpjob=gpjob, return_service=output_service)
        gpjob.result()
        return output_service
    except:
        output_service.delete()
        raise
