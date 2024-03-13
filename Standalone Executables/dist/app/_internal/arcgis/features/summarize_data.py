"""
These functions calculate total counts, lengths, areas, and basic descriptive statistics of features and their attributes
within areas or near other features.

aggregate_points calculates statistics about points that fall within specified areas.
summarize_nearby calculates statistics for features and their attributes that are within a specified distance.
summarize_within calculates statistics for area features and attributes that overlap each other.
"""
from __future__ import annotations
from datetime import datetime
from re import U
from typing import Any, Optional, Union

import arcgis as _arcgis
from arcgis.features.feature import FeatureCollection
from arcgis.features.layer import FeatureLayer, FeatureLayerCollection
from arcgis.gis import GIS, Item
from .._impl.common._utils import _date_handler
from .._impl.common._utils import inspect_function_inputs
import arcgis.network as network


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
    polygon_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ] = None,
    keep_boundaries_with_no_points: bool = True,
    summary_fields: list[str] = [],
    group_by_field: Optional[str] = None,
    minority_majority: bool = False,
    percent_points: bool = False,
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
    bin_type: Optional[str] = None,
    bin_size: Optional[float] = None,
    bin_size_unit: Optional[str] = None,
):
    """
    .. image:: _static/images/agg_points_standard/aggregate_points_standard.png

    The Aggregate Points task works with a layer of point features and a layer of polygon features. It first figures out which points fall within each polygon's area.
    After determining this point-in-polygon spatial relationship, statistics about all points in the polygon are calculated and assigned to the area. The most basic statistic is the count of the number of points within the polygon, but you can get other statistics as well.

    For example, if your points represented coffee shops and each point has a TOTAL_SALES attribute, you can get statistics like the sum of all TOTAL_SALES within the polygon, or the minimum or maximum TOTAL_SALES value, or the standard deviation of all sales within the polygon.

    ====================================    ====================================================================
    **Parameter**                            **Description**
    ------------------------------------    --------------------------------------------------------------------
    point_layer                             Required point layer. The point features that will be aggregated
                                            into the polygons in the polygon_layer. See :ref:`Feature Input<FeatureInput>`.
    ------------------------------------    --------------------------------------------------------------------
    polygon_layer                           Optional polygon layer. The polygon features (areas) into which the input points will be aggregated. See :ref:`Feature Input<FeatureInput>`. The `polygon_layer` is **required** if the `bin_type`, `bin_size` and `bin_size_unit` are not specified.
    ------------------------------------    --------------------------------------------------------------------
    keep_boundaries_with_no_points          Optional boolean. A Boolean value that specifies whether the polygons that have no points within them should be returned in the output. The default is true.
    ------------------------------------    --------------------------------------------------------------------
    summary_fields                          Optional list of strings. A list of field names and statistical summary type that you wish to calculate for all points within each polygon.
                                            Note that the count of points within each polygon is always returned.
                                            summary type is one of the following:

                                            * Sum - Adds the total value of all the points in each polygon
                                            * Mean - Calculates the average of all the points in each polygon.
                                            * Min - Finds the smallest value of all the points in each polygon.
                                            * Max - Finds the largest value of all the points in each polygon.
                                            * Stddev - Finds the standard deviation of all the points in each polygon.

                                            Example [fieldName1 summaryType1,fieldName2 summaryType2].
    ------------------------------------    --------------------------------------------------------------------
    group_by_field                          Optional string. A field name in the point_layer. Points that have
                                            the same value for the group by field will have their own counts and
                                            summary field statistics. You can create statistical groups using an
                                            attribute in the analysis layer.
                                            For example, if you are aggregating crimes to neighborhood boundaries,
                                            you may have an attribute Crime_type with five different crime types.
                                            Each unique crime type forms a group, and the statistics you choose will
                                            be calculated for each unique value of Crime_type. When you choose
                                            a grouping attribute, two results are created: the result layer and a
                                            related table containing the statistics.
    ------------------------------------    --------------------------------------------------------------------
    minority_majority                       Optional boolean. This boolean parameter is applicable only when a
                                            group_by_field is specified. If true, the minority (least dominant) or
                                            the majority (most dominant) attribute values for each group field
                                            within each boundary are calculated. Two new fields are added to the
                                            aggregated_layer prefixed with `Majority_` and `Minority_`.
                                            The default is false.
    ------------------------------------    --------------------------------------------------------------------
    percent_points                          Optional boolean. This boolean parameter is applicable only when a
                                            group_by_field is specified. If set to true, the percentage count of
                                            points for each unique group_by_field value is calculated.
                                            A new field is added to the group summary output table containing the
                                            percentages of each attribute value within each group.

                                            If minority_majority is true, two additional fields are added to the
                                            aggregated_layer containing the percentages of the minority and majority
                                            attribute values within each group.
    ------------------------------------    --------------------------------------------------------------------
    output_name                             Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                            feature layer will cause the new layer to be appended to the Feature Service.
                                            If overwrite is True in context, new layer will overwrite existing layer.
                                            If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ------------------------------------    --------------------------------------------------------------------
    context                                 Optional dict. Additional settings such as processing extent and output spatial reference.
                                            For aggregate_points, there are three settings (`overwrite` is required).

                                            - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                            - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                            - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                                .. code-block:: python

                                                    # Example Usage
                                                    context = {"extent": {"xmin": 3164569.408035,
                                                                        "ymin": -9187921.892449,
                                                                        "xmax": 3174104.927313,
                                                                        "ymax": -9175500.875353,
                                                                        "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                                "outSR": {"wkid": 3857},
                                                                "overwrite": True}
    ------------------------------------    --------------------------------------------------------------------
    gis                                     Optional, the :class:`~arcgis.gis.GIS` on which this tool runs.
                                            If not specified, the active GIS is used.
    ------------------------------------    --------------------------------------------------------------------
    estimate                                Optional Boolean. If True, the number of credits to run the operation
                                            will be returned.
    ------------------------------------    --------------------------------------------------------------------
    future                                  Optional boolean. If True, a future object will be returned and the process
                                            will not wait for the task to complete. The default is False, which means wait for results.
    ------------------------------------    --------------------------------------------------------------------
    bin_type                                Optional String. The type of bin that will be generated and points
                                            will be aggregated into. Bin options are as follows: Hexagon and Square.
                                            Square is the Default. When generating bins, for Square, the number and
                                            units specified determine the height and length of the square.
                                            For Hexagon, the number and units specified determine the distance
                                            between parallel sides. Either `bin_type` or `polygon_layer` must be
                                            specified. If `bin_type` is chosen, then `bin_size` and `bin_size_unit`
                                            specifying the size of the bins must be included.
    ------------------------------------    --------------------------------------------------------------------
    bin_size                                Optional Float. The distance for the bins of type
                                            `bin_type` that the `point_layer` will be aggregated into.
                                            When generating bins for `Square` the number and units specified determine
                                            the height and length of the square. For `Hexagon`, the number and units
                                            specified determine the distance between parallel sides.
    ------------------------------------    --------------------------------------------------------------------
    bin_size_unit                           Optional String. The linear unit to be used with the distance value
                                            specified in `bin_size`.
                                            Values: `Meters, Kilometers, Feet, Miles, NauticalMiles, or Yards`
    ====================================    ====================================================================

    :return: result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection`.
    If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.


    .. code-block:: python

        USAGE EXAMPLE: To find number of permits issued in each zip code of US.

        agg_result = aggregate_points(point_layer=permits,
                                polygon_layer=zip_codes,
                                keep_boundaries_with_no_points=False,
                                summary_fields=["DeclValNu mean","DeclValNu2 mean"],
                                group_by_field='Declared_V',
                                minority_majority=True,
                                percent_points=True,
                                output_name="aggregated_permits",
                                context={"extent":{"xmin":-8609738.077325115,"ymin":4743483.445485223,"xmax":-8594030.268012533,"ymax":4752206.821338257,"spatialReference":{"wkid":102100,"latestWkid":3857}}})

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "point_layer": point_layer,
        "polygon_layer": polygon_layer,
        "keep_boundaries_with_no_points": keep_boundaries_with_no_points,
        "summary_fields": summary_fields,
        "group_by_field": group_by_field,
        "minority_majority": minority_majority,
        "percent_points": percent_points,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
        "bin_type": bin_type,
        "bin_size": bin_size,
        "bin_size_unit": bin_size_unit,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.aggregate_points, **kwargs
    )
    return gis._tools.featureanalysis.aggregate_points(**params)


# --------------------------------------------------------------------------
def summarize_nearby(
    sum_nearby_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    summary_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    near_type: str = "StraightLine",
    distances: Optional[list[str]] = [],
    units: str = "Meters",
    time_of_day: Optional[datetime] = None,
    time_zone_for_time_of_day: str = "GeoLocal",
    return_boundaries: bool = True,
    sum_shape: bool = True,
    shape_units: Optional[str] = None,
    summary_fields: Optional[list[str]] = [],
    group_by_field: Optional[str] = None,
    minority_majority: bool = False,
    percent_shape: bool = False,
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/summarize_nearby/summarize_nearby.png

    The ``summarize_nearby`` method finds features that are within a specified distance of features in the input layer.
    Distance can be measured as a straight-line distance, a drive-time distance (for example, within 10 minutes), or a
    drive distance (within 5 kilometers). Statistics are then calculated for the nearby features. For example:

    * Calculate the total population within five minutes of driving time of a proposed new store location.
    * Calculate the number of freeway access ramps within a one-mile driving distance of a proposed new store location to use as a measure of
      store accessibility.

    =========================   ====================================================================================================================
    **Parameter**                **Description**
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    sum_nearby_layer            Required :class:`~arcgis.features.FeatureLayer` . Point, line, or polygon features from which distances will be measured to features in the ``summary_layer``. See :ref:`Feature Input<FeatureInput>`.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    summary_layer               Required layer. Point, line, or polygon features. Features in this layer that are within the specified distance to features in the ``sum_nearby_layer`` will be summarized. See :ref:`Feature Input<FeatureInput>`.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    near_type                   Optional string.
                                Defines what kind of distance measurement you want to use, either straight-line distance, travel
                                time or travel distance along a street network using various modes of transportation known as travel modes.
                                The default is ``StraightLine``.

                                Choice list:

                                * ``StraightLine``,
                                * ``Driving Distance``,
                                * ``Driving Time``,
                                * ``Rural Driving Distance``,
                                * ``Rural Driving Time``,
                                * ``Trucking Distance``,
                                * ``Trucking Time``,
                                * ``Walking Distance``,
                                * ``Walking Time``
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    distances                   Optional list of float values. Defines the search distance for 'StraightLine' and distance-based travel modes, or time
                                duration for time-based travel modes. You can enter single or multiple values, separating each value with a space.
                                Features that are within (or equal to) the distances you enter will be summarized. The unit for `distances` is
                                supplied by the units parameter.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    units                       Optional string. If :attr:`near_type` is `StraightLine` or a distance-based travel mode, this is the linear unit to be
                                used with the distance value(s) specified in distances.

                                Choice list:
                                | [``Meters``, ``Kilometers``, ``Feet``, ``Yards``, ``Miles``]

                                If ``near_type`` is a time-based travel mode, the following values can be used as units:

                                Choice list:

                                | [``Seconds``, ``Minutes``, ``Hours``]

                                The default is 'Meters'.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    time_of_day                 Optional datetime.datetime. Specify whether travel times should consider traffic conditions. To use traffic in the analysis,
                                set ``near_type`` to a travel mode object whose impedance_attribute_name property is set to travel_time and assign a value
                                to ``time_of_day``. (A travel mode with other impedance_attribute_name values don't support traffic.) The ``time_of_day`` value represents
                                the time at which travel begins, or departs, from the origin points. The time is specified as datetime.datetime.

                                The service supports two kinds of traffic: typical and live. Typical traffic references travel speeds that are made up of historical
                                averages for each five-minute interval spanning a week. Live traffic retrieves speeds from a traffic feed that processes phone probe
                                records, sensors, and other data sources to record actual travel speeds and predict speeds for the near future.

                                The `data coverage <http://www.arcgis.com/home/webmap/viewer.html?webmap=b7a893e8e1e04311bd925ea25cb8d7c7>`_ page shows the countries
                                Esri currently provides traffic data for.

                                Typical Traffic:

                                To ensure the task uses typical traffic in locations where it is available, choose a time and day of the week, and then convert the day
                                of the week to one of the following dates from 1990:

                                * Monday - 1/1/1990
                                * Tuesday - 1/2/1990
                                * Wednesday - 1/3/1990
                                * Thursday - 1/4/1990
                                * Friday - 1/5/1990
                                * Saturday - 1/6/1990
                                * Sunday - 1/7/1990
                                Set the time and date as datetime.datetime.

                                For example, to solve for 1:03 p.m. on Thursdays, set the time and date to 1:03 p.m., 4 January 1990; and convert to
                                datetime eg. datetime.datetime(1990, 1, 4, 1, 3).

                                Live Traffic:

                                To use live traffic when and where it is available, choose a time and date and convert to datetime.

                                Esri saves live traffic data for 4 hours and references predictive data extending 4 hours into the future. If the time and date you
                                specify for this parameter is outside the 24-hour time window, or the travel time in the analysis continues past the predictive data window,
                                the task falls back to typical traffic speeds.

                                Examples:
                                from datetime import datetime

                                * "time_of_day"- datetime(1990, 1, 4, 1, 3) # 13:03, 4 January 1990. Typical traffic on Thursdays at 1:03 p.m.
                                * "time_of_day"- datetime(1990, 1, 7, 17, 0) # 17:00, 7 January 1990. Typical traffic on Sundays at 5:00 p.m.
                                * "time_of_day"- datetime(2014, 10, 22, 8, 0) # 8:00, 22 October 2014. If the current time is between 8:00 p.m., 21 Oct. 2014 and 8:00 p.m., 22 Oct. 2014,
                                live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are referenced.
                                * "time_of_day"- datetime(2015, 3, 18, 10, 20) # 10:20, 18 March 2015. If the current time is between 10:20 p.m., 17 Mar. 2015 and 10:20 p.m., 18 Mar. 2015,
                                live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are referenced.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    time_zone_for_time_of_day   Optional string. Specify the time zone or zones of the ``time_of_day`` parameter.

                                Choice list: ['GeoLocal', 'UTC']

                                GeoLocal-refers to the time zone in which the originsLayer points are located.

                                UTC-refers to Coordinated Universal Time.

                                The default is 'GeoLocal'.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    return_boundaries           Optional boolean. If true, the ``result_layer`` will contain areas defined by the specified ``near_type``. For example, if using 'StraightLine' of 5 miles,
                                the ``result_layer`` will contain areas with a 5 mile radius around the input ``sum_nearby_layer`` features.

                                If False, the ``result_ayer`` will contain the same features as the ``sum_nearby_layer``.

                                The default is True.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    sum_shape                   Optional boolean. A boolean value that instructs the task to calculate statistics based on shape type of the ``summary_layer``,
                                such as the length of lines or areas of polygons of the ``summary_layer`` within each polygon in ``sum_within_layer``.

                                The default is True.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    shape_units                 Optional string. If ``sum_shape`` is true, you must specify the units of the shape summary.
                                Values:

                                * When ``summary_layer`` contains polygons: Values: ['Acres', 'Hectares', 'SquareMeters', 'SquareKilometers', 'SquareFeet', 'SquareYards', 'SquareMiles']
                                * When ``summary_layer`` contains lines: Values: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles']
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    summary_fields              Optional list of strings.A list of field names and statistical summary types that you want to calculate.
                                Note that the count is always returned by default.

                                fieldName is the name of one of the numeric fields found in the input join layer.

                                statisticType is one of the following:

                                * ``SUM``-Adds the total value of all the points in each polygon
                                * ``MEAN``-Calculates the average of all the points in each polygon
                                * ``MIN``-Finds the smallest value of all the points in each polygon
                                * ``MAX``-Finds the largest value of all the points in each polygon
                                * ``STDDEV``-Finds the standard deviation of all the points in each polygon

                                Example: ["fieldName summaryType","fieldName summaryType", ...]
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    group_by_field              Optional string. This is a field of the ``summary_layer`` features that you can use to calculate statistics separately for each unique attribute value.
                                For example, suppose the ``summary_layer`` contains point locations of businesses that store hazardous materials, and one of the fields is HazardClass
                                containing codes that describe the type of hazardous material stored. To calculate summaries by each unique value of HazardClass, use HazardClass as
                                the ``group_by_field`` field.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    minority_majority           Optional boolean. This boolean parameter is applicable only when a ``group_by_field`` is specified. If true, the minority (least dominant) or the
                                majority (most dominant) attribute values for each group field within each nearby area are calculated. Two new fields are added to
                                the ``result_layer`` prefixed with `Majority_` and `Minority_`.

                                The default is False.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    percent_shape               Optional boolean. This Boolean parameter is applicable only when a ``group_by_field`` is specified. If set to true,
                                the percentage of each unique ``group_by_field`` value is calculated for each ``sum_nearby_layer`` feature.

                                The default is False.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    output_name                 Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                feature layer will cause the new layer to be appended to the Feature Service.
                                If overwrite is True in context, new layer will overwrite existing layer.
                                If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    context                     Optional dict. Additional settings such as processing extent and output spatial reference.
                                For summarize_nearby, there are three settings.

                                - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                    .. code-block:: python

                                        # Example Usage
                                        context = {"extent": {"xmin": 3164569.408035,
                                                            "ymin": -9187921.892449,
                                                            "xmax": 3174104.927313,
                                                            "ymax": -9175500.875353,
                                                            "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                    "outSR": {"wkid": 3857},
                                                    "overwrite": True}
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    estimate                    Optional boolean. Returns the number of credit for the operation.
    -------------------------   --------------------------------------------------------------------------------------------------------------------
    future                      Optional boolean. If True, a future object will be returned and the process
                                will not wait for the task to complete. The default is False, which means wait for results.
    =========================   ====================================================================================================================

    :return:
        result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection` dictionary.

        dict with the following keys:

        "result_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

        "group_by_summary" : layer (:class:`~arcgis.features.FeatureCollection`)

    .. code-block:: python

         # USAGE EXAMPLE: To find hospital facilities that are within 5 miles of a school.
         summarize_nearby(sum_nearby_layer=item2.layers[0],
                          summary_layer=item1.layers[0],
                          near_type='StraightLine',
                          distances=[5],
                          units='Miles',
                          time_zone_for_time_of_day='GeoLocal',
                          return_boundaries=False,
                          sum_shape=True,
                          shape_units=None,
                          output_name='nearest hospitals to schools')
    """
    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "sum_nearby_layer": sum_nearby_layer,
        "summary_layer": summary_layer,
        "near_type": near_type,
        "distances": distances,
        "units": units,
        "time_of_day": time_of_day,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "return_boundaries": return_boundaries,
        "sum_shape": sum_shape,
        "shape_units": shape_units,
        "summary_fields": summary_fields,
        "group_by_field": group_by_field,
        "percent_shape": percent_shape,
        "minority_majority": minority_majority,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis.summarize_nearby, **kwargs
    )
    if isinstance(near_type, str):
        if near_type != "StraightLine":
            route_service = network.RouteLayer(
                gis.properties.helperServices.route.url, gis=gis
            )
            near_type = [
                i
                for i in route_service.retrieve_travel_modes()["supportedTravelModes"]
                if i["name"] == near_type
            ][0]
            params["near_type"] = near_type
    return gis._tools.featureanalysis.summarize_nearby(**params)


# --------------------------------------------------------------------------
def summarize_center_and_dispersion(
    analysis_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    summarize_type: list[str] = ["CentralFeature"],
    ellipse_size: Optional[str] = None,
    weight_field: Optional[str] = None,
    group_field: Optional[str] = None,
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/summarize_center_and_dispersion/summarize_center_and_dispersion.png

    The ``summarize_center_and_dispersion`` method finds central features and directional distributions. It can be used to answer questions such as:

    * Where is the center?
    * Which feature is the most accessible from all other features?
    * How dispersed, compact, or integrated are the features?
    * Are there directional trends?

    ====================    ============================================================================================
    **Parameter**            **Description**
    --------------------    --------------------------------------------------------------------------------------------
    analysis_layer          Required :class:`~arcgis.features.FeatureLayer` . The point, line, or polygon features to be analyzed. See :ref:`Feature Input<FeatureInput>`.
    --------------------    --------------------------------------------------------------------------------------------
    summarize_type          Required list of strings. The method with which to summarize the ``analysis_layer``.

                            Choice list: ["CentralFeature", "MeanCenter", "MedianCenter", "Ellipse"]
    --------------------    --------------------------------------------------------------------------------------------
    ellipse_size            Optional string. The size of the output ellipse in standard deviations.

                            Choice list: ['1 standard deviations', '2 standard deviations', '3 standard deviations']

                            The default ellipse size is '1 standard deviations'.
    --------------------    --------------------------------------------------------------------------------------------
    weight_field            Optional field. A numeric field in the ``analysis_layer`` to be used to
                            weight locations according to their relative importance.
    --------------------    --------------------------------------------------------------------------------------------
    group_field             Optional field. The field used to group features for separate directional
                            distribution calculations. The ``group_field`` can be of
                            integer, date, or string type.
    --------------------    --------------------------------------------------------------------------------------------
    output_name             Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                            feature layer will cause the new layer to be appended to the Feature Service.
                            If overwrite is True in context, new layer will overwrite existing layer.
                            If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    --------------------    --------------------------------------------------------------------------------------------
    context                 Optional dict. Additional settings such as processing extent and output spatial reference.
                            For summarize_center_and_dispersion, there are three settings.

                            - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                            - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                            - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 11+

                                .. code-block:: python

                                    # Example Usage
                                    context = {"extent": {"xmin": 3164569.408035,
                                                        "ymin": -9187921.892449,
                                                        "xmax": 3174104.927313,
                                                        "ymax": -9175500.875353,
                                                        "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                "outSR": {"wkid": 3857},
                                                "overwrite": True}
    --------------------    --------------------------------------------------------------------------------------------
    estimate                Optional boolean. If True, the number of credits to run the operation will be returned.
    --------------------    --------------------------------------------------------------------------------------------
    future                  Optional boolean. If True, a future object will be returned and the process
                            will not wait for the task to complete. The default is False, which means wait for results.
    ====================    ============================================================================================

    :return: list of items if ``output_name`` is supplied else, a Python dictionary with the following keys:

        | "central_feature_result_layer" : layer (:class:`~arcgis.features.FeatureCollection`)
        | "mean_feature_result_layer" : layer (:class:`~arcgis.features.FeatureCollection`)
        | "median_feature_result_layer" : layer (:class:`~arcgis.features.FeatureCollection`)
        | "ellipse_feature_result_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

    .. code-block:: python

        # USAGE EXAMPLE: To find central features and mean center of earthquake over past months.
        central_features = summarize_center_and_dispersion(analysis_layer=earthquakes,
                                                           summarize_type=["CentralFeature","MeanCenter"],
                                                           ellipse_size='2 standard deviations',
                                                           weight_field='mag',
                                                           group_field='magType',
                                                           output_name='find central features and mean center of earthquake over past months')

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "analysis_layer": analysis_layer,
        "summarize_type": summarize_type,
        "ellipse_size": ellipse_size,
        "weight_field": weight_field,
        "group_field": group_field,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.summarize_center_and_dispersion,
        **kwargs,
    )
    return gis._tools.featureanalysis.summarize_center_and_dispersion(**params)


# --------------------------------------------------------------------------
def summarize_within(
    sum_within_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    summary_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    sum_shape: bool = True,
    shape_units: Optional[str] = None,
    summary_fields: Optional[list[str]] = [],
    group_by_field: Optional[str] = None,
    minority_majority: bool = False,
    percent_shape: bool = False,
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
    bin_type: str = "Square",
    bin_size: Optional[float] = None,
    bin_size_unit: Optional[str] = None,
):
    """
    .. image:: _static/images/summarize_within/summarize_within.png

    The ``summarize_within`` method finds the point, line, or polygon features (or portions of these features)
    that are within the boundaries of polygons in another layer. For example:

    * Given a layer of watershed boundaries and a layer of land-use boundaries by land-use type, calculate total acreage of land-use type for each watershed.
    * Given a layer of parcels in a county and a layer of city boundaries, summarize the average value of vacant parcels within each city boundary.
    * Given a layer of counties and a layer of roads, summarize the total mileage of roads by road type within each county.

    You can think of ``summarize_within`` as taking two layers and stacking them on top of each other.
    One of the layers, the ``sum_within_layer`` must be a polygon layer, and imagine that these polygon
    boundaries are all colored red. The other layer, the ``summary_layer``, can be any feature type point,
    line, or polygon. After stacking these layers on top of each other, you peer down through the stack
    and count the number of features in the ``summary_layer`` that fall within the polygons with the red
    boundaries (the ``sum_within_layer``). Not only can you count the number of features, you can calculate
    simple statistics about the attributes of the features in the ``summary_layer``, such as sum, mean, minimum, maximum, and so on.

    =====================================   =========================================================
    **Parameter**                            **Description**
    -------------------------------------   ---------------------------------------------------------
    sum_within_layer                        Required :class:`~arcgis.features.FeatureLayer` . The polygon features. Features, or
                                            portions of features, in the ``summary_layer`` (below) that fall within
                                            the boundaries of these polygons will be summarized. See :ref:`Feature Input<FeatureInput>`.
    -------------------------------------   ---------------------------------------------------------
    summary_layer                           Required :class:`~arcgis.features.FeatureLayer` . Point, line, or polygon features that will be summarized for each polygon in the ``sum_within_layer``.
                                            See :ref:`Feature Input<FeatureInput>`.
    -------------------------------------   ---------------------------------------------------------
    sum_shape                               Optional boolean. A boolean value that instructs the task to calculate statistics
                                            based on shape type of the ``summary_layer``, such as the length of lines or areas of
                                            polygons of the ``summary_layer`` within each polygon in ``sum_within_layer``.

                                            The default is True.
    -------------------------------------   ---------------------------------------------------------
    shape_units                             Optional string. Specify units to summarize the length or areas when ``sum_shape`` is set to true. Units is not required to summarize
                                            points.

                                            * When ``summary_layer`` contains polygons: ['Acres', 'Hectares', 'SquareMeters', 'SquareKilometers', 'SquareMiles', 'SquareYards', 'SquareFeet']

                                            * When ``summary_layer`` contains lines: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles']
    -------------------------------------   ---------------------------------------------------------
    summary_fields                          Optional list of strings. A list of field names and statistical summary type that you wish
                                            to calculate for all features in the ``summary_layer`` that are within each polygon in the ``sum_within_layer`` .

                                            Example: ["fieldname1 summary", "fieldname2 summary"]
    -------------------------------------   ---------------------------------------------------------
    group_by_field                          Optional string. This is a field of the ``summary_layer`` features that you can use to calculate statistics separately
                                            for each unique attribute value. For example, suppose the ``sum_within_layer`` contains city boundaries and
                                            the ``summary_layer`` features are parcels. One of the fields of the parcels is Status which contains
                                            two values: VACANT and OCCUPIED. To calculate the total area of vacant and occupied parcels within the
                                            boundaries of cities, use Status as the ``group_by_field`` field.
    -------------------------------------   ---------------------------------------------------------
    minority_majority                       Optional boolean. This boolean parameter is applicable only when a ``group_by_field`` is specified.
                                            If true, the minority (least dominant) or the majority (most dominant) attribute values for each group
                                            field are calculated. Two new fields are added to the ``result_layer`` prefixed with `Majority_` and `Minority_`.

                                            The default is False.
    -------------------------------------   ---------------------------------------------------------
    percent_shape                           Optional boolean. This Boolean parameter is applicable only when a ``group_by_field`` is specified.
                                            If set to true, the percentage of each unique ``group_by_field`` value is calculated for
                                            each ``sum_within_layer`` polygon.

                                            The default is False.
    -------------------------------------   ---------------------------------------------------------
    output_name                             Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                            feature layer will cause the new layer to be appended to the Feature Service.
                                            If overwrite is True in context, new layer will overwrite existing layer.
                                            If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------------------   ---------------------------------------------------------
    context                                 Optional dict. Additional settings such as processing extent and output spatial reference.
                                            For summarize_within, there are three settings.

                                            - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                            - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                            - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                                .. code-block:: python

                                                    # Example Usage
                                                    context = {"extent": {"xmin": 3164569.408035,
                                                                        "ymin": -9187921.892449,
                                                                        "xmax": 3174104.927313,
                                                                        "ymax": -9175500.875353,
                                                                        "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                                "outSR": {"wkid": 3857},
                                                                "overwrite": True}
    -------------------------------------   ---------------------------------------------------------
    estimate                                Optional boolean. If True, the number of credits to run the operation will be returned.
    -------------------------------------   ---------------------------------------------------------
    future                                  Optional boolean. If True, a future object will be returned and the process
                                            will not wait for the task to complete. The default is False, which means wait for results.
    -------------------------------------   ---------------------------------------------------------
    bin_type                                Required string. The type of bin used to calculate density.

                                            Choice list: ['Hexagon', 'Square'].
    -------------------------------------   ---------------------------------------------------------
    bin_size                                Required float. The distance for the bins that the ``input_layer`` will be analyzed using.
                                            When generating bins, for Square, the number and units specified determine the
                                            height and length of the square. For ``Hexagon``, the number and units specified
                                            determine the distance between parallel sides.
    -------------------------------------   ---------------------------------------------------------
    bin_size_unit                           Required string. The distance unit for the bins for which the density will be calculated.
                                            The linear unit to be used with the value specified in ``bin_size``.

                                            The default is 'Meters'.
    =====================================   =========================================================

    :return:
        result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection` dictionary.

        dict with the following keys:

        "result_layer" : layer (FeatureCollection)

        "group_by_summary" : layer (FeatureCollection)

    .. code-block:: python

        # USAGE EXAMPLE: To summarize traffic accidents within each county and group them by the day of accident.
        acc_within_county = summarize_within(sum_within_layer=boundaries,
                                             summary_layer=collision_lyr,
                                             sum_shape=True,
                                             group_by_field='Day',
                                             minority_majority=True,
                                             percent_shape=True,
                                             output_name='summarize accidents within each county',
                                             context={"extent":{"xmin":-13160690.837046918,"ymin":4041586.5461609075,"xmax":-13132466.464352652,"ymax":4058001.397985127,"spatialReference":{"wkid":102100,"latestWkid":3857}}})
    """
    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "sum_within_layer": sum_within_layer,
        "summary_layer": summary_layer,
        "sum_shape": sum_shape,
        "shape_units": shape_units,
        "summary_fields": summary_fields,
        "group_by_field": group_by_field,
        "minority_majority": minority_majority,
        "percent_shape": percent_shape,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
        "bin_type": bin_type,
        "bin_size": bin_size,
        "bin_size_unit": bin_size_unit,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.summarize_within, **kwargs
    )
    if not estimate is None:
        params["estimate"] = estimate
    return gis._tools.featureanalysis.summarize_within(**params)


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
    spatial_relationship: Optional[str] = None,
    spatial_relationship_distance: Optional[float] = None,
    spatial_relationship_distance_units: Optional[str] = None,
    attribute_relationship: Optional[list[dict[str, Any]]] = None,
    join_operation: str = """JoinOneToOne""",
    summary_fields: Optional[list[dict[str[Any]]]] = None,
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
    join_type: str = "INNER",
    records_to_match: Optional[dict[str, Any]] = None,
):
    """
    .. image:: _static/images/join_features/join_features.png

    The ``join_features`` method works with two layers and joins the attributes
    from one feature to another based on spatial and attribute relationships.

    ============================================================================================    =================================================================================================================================
    **Parameter**                                                                                    **Description**
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    target_layer                                                                                    Required layer. The point, line, polygon or table layer that will have attributes from
                                                                                                    the ``join_layer`` appended to its table. See :ref:`Feature Input<FeatureInput>`.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    join_layer                                                                                      Required layer. The point, line, polygon or table layer that will be joined to the ``target_layer``. See :ref:`Feature Input<FeatureInput>`.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    spatial_relationship                                                                            Required string if not table layers. Defines the spatial relationship used to spatially join features.

                                                                                                    Choice list: ['identicalto', 'intersects', 'completelycontains', 'completelywithin', 'withindistance']
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    spatial_relationship_distance                                                                   Optional float. A float value used for the search distance to determine if the target features are near or within a
    (Required if ``spatial_relationship`` is withindistance)                                        specified distance of the join features.
                                                                                                    This is only applied if Within a distance of is the selected ``spatial_relationship``.
                                                                                                    You can only enter a single distance value. The units of the distance values are supplied by the
                                                                                                    ``spatial_relationship_distance_units`` parameter.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    spatial_relationship_distance_units                                                             Optional string. The linear unit to be used with the distance value specified in ``spatial_relationship_distance``.
    (Required if ``spatial_relationship`` is withindistance)                                        Choice list: ['Miles', 'Yards', 'Feet', 'NauticalMiles', 'Meters', 'Kilometers']

                                                                                                    The default is 'Miles'.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    attribute_relationship                                                                          Optional list of dicts. Defines an attribute relationship used to join features. Features are matched when the field
                                                                                                    values in the join layer are equal to field values in the target layer.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    join_operation                                                                                  Optional string. A string representing the type of join that will be applied

                                                                                                    Choice list: ['JoinOneToOne', 'JoinOneToMany']

                                                                                                    * ``JoinOneToOne`` - If multiple join features are found that have the same relationships with asingle target feature, the attributes from the multiple join features will be aggregated using the specified summary statistics.
                                                                                                      For example, if a point target feature is found within two separate polygon join features, the attributes from the two polygons will be aggregated before being transferred to the output point feature class.
                                                                                                      If one polygon has an attribute value of 3 and the other has a value of 7, and a SummaryField of sum is selected, the aggregated value
                                                                                                      in the output feature class will be 10. There will always be a Count field calculated, with a value of 2, for the number of features specified. This is the default.

                                                                                                    * ``JoinOneToMany`` - If multiple join features are found that have the same relationship with a single target feature, the output feature class will contain multiple copies (records) of the target feature.
                                                                                                      For example, if a single point target feature is found within two separate polygon join features, the output feature class will contain two copies of the target feature:
                                                                                                      one record with the attributes of the first polygon, and another record with the attributes of the second polygon. There are no summary statistics calculated with this method.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    summary_fields                                                                                  Optional list of dicts. A list of field names and statistical summary types that you want to calculate. Note that the count is always returned by default.

                                                                                                    fieldName is the name of one of the numeric fields found in the input join layer.

                                                                                                    statisticType is one of the following:

                                                                                                    * ``SUM`` - Adds the total value of all the points in each polygon
                                                                                                    * ``MEAN`` - Calculates the average of all the points in each polygon
                                                                                                    * ``MIN`` - Finds the smallest value of all the points in each polygon
                                                                                                    * ``MAX`` - Finds the largest value of all the points in each polygon
                                                                                                    * ``STDDEV`` - Finds the standard deviation of all the points in each polygon
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    output_name                                                                                     Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                                                                                    feature layer will cause the new layer to be appended to the Feature Service.
                                                                                                    If overwrite is True in context, new layer will overwrite existing layer.
                                                                                                    If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    context                                                                                         Optional dict. Additional settings such as processing extent and output spatial reference.
                                                                                                    For join_features, there are three settings.

                                                                                                    - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                                                                                    - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                                                                                    - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online or Enterprise 10.9.1+

                                                                                                        .. code-block:: python

                                                                                                            # Example Usage
                                                                                                            context = {"extent": {"xmin": 3164569.408035,
                                                                                                                                "ymin": -9187921.892449,
                                                                                                                                "xmax": 3174104.927313,
                                                                                                                                "ymax": -9175500.875353,
                                                                                                                                "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                                                                                        "outSR": {"wkid": 3857},
                                                                                                                        "overwrite": True}
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    estimate                                                                                        Optional boolean. If True, the number of credits to run the operation will be returned.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    future                                                                                          Optional boolean. If True, a future object will be returned and the process
                                                                                                    will not wait for the task to complete. The default is False, which means wait for results.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    join_type                                                                                       Optional String.  Determines the type of join performed on the datasets.  The allowed values are INNER or LEFT.
    --------------------------------------------------------------------------------------------    ---------------------------------------------------------------------------------------------------------------------------------
    records_to_match                                                                                Optional Dict. Defines how two features are joined.
                                                                                                    Example:
                                                                                                        | {"groupByFields":"",
                                                                                                        | "orderByFields":"objectid ASC",
                                                                                                        | "topCount":1}
    ============================================================================================    =================================================================================================================================

    :return: result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection`.

    .. code-block:: python

        USAGE EXAMPLE: To summarize traffic accidents within each parcel using spatial relationship.
        accident_count_in_each_parcel = join_features(target_layer=parcel_lyr,
                                                      join_layer=traffic_accidents_lyr,
                                                      spatial_relationship='intersects',
                                                      summary_fields=[{"statisticType": "Mean", "onStatisticField": "Population"},
                                                      output_name='join features',
                                                      context={"extent":{"xmin":-9375809.87305117,"ymin":4031882.3806860778,"xmax":-9370182.196843527,"ymax":4034872.9794178144,"spatialReference":{"wkid":102100,"latestWkid":3857}}}, )
    """
    kwargs = {
        "target_layer": target_layer,
        "join_layer": join_layer,
        "spatial_relationship": spatial_relationship,
        "spatial_relationship_distance": spatial_relationship_distance,
        "spatial_relationship_distance_units": spatial_relationship_distance_units,
        "attribute_relationship": attribute_relationship,
        "join_operation": join_operation,
        "summary_fields": summary_fields,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "future": future,
        "join_type": join_type,
        "records_to_match": records_to_match,
    }
    gis = _arcgis.env.active_gis if gis is None else gis
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.join_features, **kwargs
    )
    if not estimate is None:
        params["estimate"] = estimate
    if "attribute_relationship" not in params:
        params["attribute_relationship"] = attribute_relationship
    if "summary_fields" not in params:
        params["summary_fields"] = summary_fields
    return gis._tools.featureanalysis.join_features(**params)
