"""
These functions are used for both the day-to-day management of geographic data and for combining data prior to analysis.

dissolve_boundaries merges together areas that share a common boundary and a common attribute value.
extract_data creates new datasets by extracting features from your existing data.
merge_layers copies all the features from two or more existing layers into a new layer.
overlay_layers combines two or more layers into one single layer. You can think of overlay as peering through a stack of
maps and creating a single map containing all the information found in the stack.
"""
from __future__ import annotations
from typing import Any, Optional, Union
import arcgis as _arcgis
from arcgis.features.feature import FeatureCollection
from arcgis.features.layer import FeatureLayer, FeatureLayerCollection
from arcgis.gis import GIS, Item
from .._impl.common._utils import inspect_function_inputs


# ----------------------------------------------------------------------
def generate_tessellation(
    extent_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ] = None,
    bin_size: float = 1,
    bin_size_unit: str = "SquareKilometers",
    bin_type: str = "SQUARE",
    intersect_study_area: bool = False,
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
    bin_resolution: Optional[int] = None,
):
    """
    Generates a tessellated grid of regular polygons.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    extent_layer                             Optional layer. A layer defining the processing extent.
    ------------------------------------     --------------------------------------------------------------------
    bin_size                                 Optional Float. The size of each individual shape that makes up the tessellation.
    ------------------------------------     --------------------------------------------------------------------
    bin_size_unit                            Optional String. Size unit of each individual shape. The allowed
                                             values are: 'SquareKilometers', 'Hectares', 'SquareMeters',
                                             'SquareMiles', 'Acres', 'SquareYards', 'SquareFeet', 'SquareInches',
                                             'Miles', 'Yards', 'Feet', 'Kilometers', 'Meters', and
                                             'NauticalMiles'.
    ------------------------------------     --------------------------------------------------------------------
    bin_type                                 Optional String. The type of shape to tessellate.
                                             Allowed values are: 'SQUARE', 'HEXAGON', 'TRIANGLE', 'DIAMOND',
                                             'TRANSVERSEHEXAGON', or `H3_HEXAGON`.
    ------------------------------------     --------------------------------------------------------------------
    intersect_study_area                     Optional Boolean. A boolean defines whether to keep only tessellations intersect with the study area.

    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                             feature layer will cause the new layer to be appended to the Feature Service.
                                             If overwrite is True in context, new layer will overwrite existing layer.
                                             If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dict. Additional settings such as processing extent and output spatial reference.
                                             For calculate_density, there are three settings.

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
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    estimate                                 Optional Boolean. If True, the number of credits to run the operation will be returned.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Optional boolean. If True, a future object will be returned and the process
                                             will not wait for the task to complete. The default is False, which means wait for results.
    ------------------------------------     --------------------------------------------------------------------
    bin_resolution                           Optional Integer. This becomes required when H3_HEXAGON is used.
                                             The H3 resolution of the hexagons. Resolution ranges from 0 to 15.
                                             With each increasing resolution size, the area of the polygons will
                                             be one seventh the size.
    ====================================     ====================================================================

    .. note::
            The tool requires either an 'extent' given in the `context` or an `extent_layer`.

    :return:
        :class:`~arcgis.features.FeatureLayer` if out_put name specified or
        a :class:`~arcgis.features.FeatureLayerCollection`.
        If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    """
    if not bin_resolution is None and (bin_resolution > 15 or bin_resolution < 0):
        raise ValueError("bin_resolution must be between 0 to 15")
    gis = _arcgis.env.active_gis if gis is None else gis
    if not ((context and "extent" in context) or extent_layer):
        raise ValueError("Tool requires an extent_layer or defined extent.")
    kwargs = {
        "extent_layer": extent_layer,
        "bin_size": bin_size,
        "bin_size_unit": bin_size_unit,
        "bin_type": bin_type,
        "intersect_study_area": intersect_study_area,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
        "bin_resolution": bin_resolution,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.generate_tessellations, **kwargs
    )
    if extent_layer is None:
        params["extent_layer"] = None
    return gis._tools.featureanalysis.generate_tesselation(**params)


# ----------------------------------------------------------------------
def dissolve_boundaries(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    dissolve_fields: Union[list[str], list] = [],
    summary_fields: Union[list[str], list] = [],
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    multi_part_features: bool = True,
    future: bool = False,
):
    """
    .. image:: _static/images/dissolve_boundaries/dissolve_boundaries.png

    The dissolve_boundaries method finds polygons that overlap or share a common boundary and merges them together to form a single polygon.

    You can control which boundaries are merged by specifying a field. For example, if you have a layer of counties, and each county
    has a State_Name attribute, you can dissolve boundaries using the State_Name attribute. Adjacent counties will be merged together
    if they have the same value for State_Name. The end result is a layer of state boundaries.

    ====================================     =====================================================================================
    **Parameter**                             **Description**
    ------------------------------------     -------------------------------------------------------------------------------------
    input_layer                              Required layer. The layer containing polygon features that will be dissolved. See :ref:`Feature Input<FeatureInput>`.
    ------------------------------------     -------------------------------------------------------------------------------------
    dissolve_fields                          Optional list of strings. One or more fields on the input_layer that control which polygons
                                             are merged. If you don't supply dissolve_fields , or you supply an empty list of fields, polygons
                                             that share a common border (that is, they are adjacent) or polygon areas that overlap will be dissolved into one polygon.

                                             If you do supply values for the dissolve_fields parameter, polygons that share a common border
                                             and contain the same value in one or more fields will be dissolved. For example, if you have a layer of counties,
                                             and each county has a State_Name attribute, you can dissolve boundaries using the State_Name attribute.
                                             Adjacent counties will be merged together if they have the same value for State_Name. The end result is a layer of
                                             state boundaries.If two or more fields are specified, the values in these fields must be the same for the boundary to be dissolved.
    ------------------------------------     -------------------------------------------------------------------------------------
    summary_fields                           Optional list of strings.
                                             A list of field names and statistical summary types that you
                                             wish to calculate from the polygons that are dissolved together:

                                             *["fieldName summary type", "fieldName2 summaryType"]*

                                             `fieldName` is the name of one of the numeric fields found in the
                                             input_layer.
                                             `summary type` is one of the following:

                                             * ``Sum`` - Adds the total value of all the points in each polygon
                                             * ``Mean`` - Calculates the average of all the points in each polygon.
                                             * ``Min`` - Finds the smallest value of all the points in each polygon.
                                             * ``Max`` - Finds the largest value of all the points in each polygon.
                                             * ``Stddev`` - Finds the standard deviation of all the points in each polygon.

                                             For example, if you are dissolving counties based on `State_Name`, and each
                                             county has a `Population` field, you can sum the `Population` for all the
                                             counties sharing the same `State_Name` attribute. The result would be a
                                             layer of state boundaries with total population.

                                             .. code-block:: python
                                                :emphasize-lines: 5

                                                # Usage Example

                                                >>> dissolve_boundaries(input_layer="US_Counties",
                                                                        dissolve_fields="State_Name",
                                                                        summary_fields=["Population Sum"],
                                                                        output_name="US_States")
    ------------------------------------     -------------------------------------------------------------------------------------
    output_name                              Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                             feature layer will cause the new layer to be appended to the Feature Service.
                                             If overwrite is True in context, new layer will overwrite existing layer.
                                             If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ------------------------------------     -------------------------------------------------------------------------------------
    context                                  Optional dict. Additional settings such as processing extent and output spatial reference.
                                             For calculate_density, there are three settings.

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
    ------------------------------------     -------------------------------------------------------------------------------------
    gis                                      Optional, the :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     -------------------------------------------------------------------------------------
    estimate                                 Optional Boolean. If True, the number of credits to run the operation will be returned.
    ------------------------------------     -------------------------------------------------------------------------------------
    multi_part_features                      Optional boolean. Specifies whether multipart features (i.e. features which
                                             share a common attribute table but are not visibly connected) are allowed in
                                             the output feature class.

                                             Choice list: [``True``, ``False``]

                                             * ``True``: Specifies multipart features are allowed.
                                             * ``False``: Specifies multipart features are not allowed. Instead of creating multipart features, individual features will be created for each part.

                                             The default value is ``True``.
    ------------------------------------     -------------------------------------------------------------------------------------
    future                                   Optional boolean. If True, a future object will be returned and the process
                                             will not wait for the task to complete. The default is False, which means wait for results.
    ====================================     =====================================================================================

    :return:
        result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`Feature Collection <arcgis.features.FeatureCollection>`.
        If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.


    .. code-block:: python

        USAGE EXAMPLE: To dissolve boundaries of polygons with same state name. The dissolved polygons are summarized using population as summary field and standard deviation as summary type.
        diss_counties = dissolve_boundaries(input_layer=usa_counties,
                                            dissolve_fields=["STATE_NAME"],
                                            summary_fields=["POPULATION Stddev"],
                                            output_name="DissolveBoundaries")
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "input_layer": input_layer,
        "dissolve_fields": dissolve_fields,
        "summary_fields": summary_fields,
        "multi_part_features": multi_part_features,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.dissolve_boundaries, **kwargs
    )
    return gis._tools.featureanalysis.dissolve_boundaries(**params)


# ----------------------------------------------------------------------
def extract_data(
    input_layers: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    extent: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    clip: bool = False,
    data_format: Optional[str] = None,
    output_name: Optional[Union[str, dict]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/extract_data/extract_data.png

    The ``extract_data`` method is used to extract data from one or more layers within a given extent.
    The extracted data format can be a file geodatabase, shapefiles, csv, or kml.
    File geodatabases and shapefiles are added to a zip file that can be downloaded.

    ===================================    =========================================================
    **Parameter**                           **Description**
    -----------------------------------    ---------------------------------------------------------
    input_layers                           Required list of strings. A list of input layers to be extracted. See :ref:`Feature Input<FeatureInput>`.
    -----------------------------------    ---------------------------------------------------------
    extent                                 Optional layer. The extent is the area of interest used to extract the input features. If not specified, all features from each input layer are extracted. See :ref:`Feature Input<FeatureInput>`.
    -----------------------------------    ---------------------------------------------------------
    clip                                   Optional boolean. A Boolean value that specifies whether the features within the input layer are clipped
                                           within the extent. By default, features are not clipped and all features intersecting the extent are returned.

                                           The default is false.
    -----------------------------------    ---------------------------------------------------------
    data_format                            Optional string. A keyword defining the output data format for your extracted data.

                                           Choice list: ['FileGeodatabase', 'ShapeFile', 'KML', 'CSV']

                                           The default is 'CSV'.

                                           If *FileGeodatase* is specified *and* the input layer has `attachments: <https://enterprise.arcgis.com/en/portal/latest/use/manage-hosted-layers.htm#ESRI_SECTION2_EF4F7A72F7B74E47B5CBCC1F343445E2>`_

                                            * if *clip=False*, the attachments will be extracted to the output file
                                            * if *clip=True*, the attachments will not be extracted
    -----------------------------------    ---------------------------------------------------------
    output_name                            Optional string or dict.

                                           When ``output_name`` is a string, the output item in your My contents page
                                           will be named by the value. Other item properties will receive default values.

                                           .. code-block:: python

                                               output_name = "my_extracted_item"

                                           To explicitly provide other item properties, use a dict with the following Syntax.

                                           .. code-block:: python

                                               output_name = {"title": "<title>",
                                                              "tag": "<tags>",
                                                              "snippet": "<snippet>",
                                                              "description": "<description>"}

                                           For more information on these and other item properties, see the Item resource page in the `ArcGIS REST API. <https://developers.arcgis.com/rest/users-groups-and-items/item.htm>`_
    -----------------------------------    ---------------------------------------------------------
    gis                                    Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    -----------------------------------    ---------------------------------------------------------
    estimate                               Optional boolean. If True, the number of credits to run the operation will be returned.
    -----------------------------------    ---------------------------------------------------------
    future                                 Optional boolean. If True, a future object will be returned and the process
                                           will not wait for the task to complete. The default is False, which means wait for results.
    ===================================    =========================================================

    :return: result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`Feature Collection <arcgis.features.FeatureCollection>`.
    If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

        # USAGE EXAMPLE: To extract data from highways layer with the extent of a state boundary.

        ext_state_highway = extract_data(input_layers=[highways.layers[0]],
                                 extent=state_area_boundary.layers[0],
                                 clip=True,
                                 data_format='ShapeFile',
                                 output_name='state highway extracted')
    """
    if data_format is None:
        data_format = "CSV"
    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "input_layers": input_layers,
        "extent": extent,
        "clip": clip,
        "data_format": data_format,
        "output_name": output_name,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }

    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.extract_data, **kwargs
    )
    return gis._tools.featureanalysis.extract_data(**params)


# ----------------------------------------------------------------------
def merge_layers(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    merge_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    merging_attributes: list[str] = [],
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/merge_layers/merge_layers.png

    The ``merge_layers`` method copies features from two layers into a new layer.
    The layers to be merged must all contain the same feature types (points, lines, or polygons).
    You can control how the fields from the input layers are joined and copied. For example:

    * I have three layers for England, Wales, and Scotland and I want a single layer of Great Britain.
    * I have two layers containing parcel information for contiguous townships. I want to join them together into a single layer, keeping only the fields that have the same name and type on the two layers.

    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    input_layer         Required feature layer. The point, line or polygon features with the ``merge_layer``. See :ref:`Feature Input<FeatureInput>`.
    ----------------    ---------------------------------------------------------------
    merge_layer         Required feature layer. The point, line, or polygon features to merge with the ``input_layer``.
                        The ``merge_layer`` must contain the same feature type (point, line, or polygon) as the ``input_layer``. See :ref:`Feature Input<FeatureInput>`.
    ----------------    ---------------------------------------------------------------
    merge_attributes    Optional list. Defines how the fields in ``merge_layer`` will be modified. By default, all fields from both inputs will be included in the output layer.

                        If a field exists in one layer but not the other, the output
                        layer will still contain the field. The output field will
                        contain null values for the input features that did not have the
                        field. For example, if the ``input_layer`` contains a field named
                        TYPE but the ``merge_layer`` does not contain TYPE, the output will
                        contain TYPE, but its values will be null for all the features
                        copied from the ``merge_layer``.

                        You can control how fields in the ``merge_layer`` are written to the
                        output layer using the following merge types that operate on a
                        specified ``merge_layer`` field:

                        + ``Remove`` - The field in the ``merge_layer`` will be removed from the output layer.

                        + ``Rename`` - The field in the ``merge_layer`` will be renamed in the output layer. You cannot rename a field in the ``merge_layer`` to a field in the ``input_layer``. If you want to make field names equivalent, use Match.

                        + ``Match`` - A field in the ``merge_layer`` is made equivalent to a field in the ``input_layer`` specified by merge value. For example, the ``input_layer`` has a field named CODE and the ``merge_layer`` has a field named STATUS. You can match STATUS to CODE, and the output will contain the CODE field with values of the STATUS field used for features copied from the ``merge_layer``. Type casting is supported (for example, float to integer, integer to string) except for string to numeric.
    ----------------    ---------------------------------------------------------------
    output_name         Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                        feature layer will cause the new layer to be appended to the Feature Service.
                        If overwrite is True in context, new layer will overwrite existing layer.
                        If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ----------------    ---------------------------------------------------------------
    context             Optional dict. Additional settings such as processing extent and output spatial reference.
                        For calculate_density, there are three settings.

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
    ----------------    ---------------------------------------------------------------
    gis                 Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    ----------------    ---------------------------------------------------------------
    estimate            Optional boolean. If True, the estimated number of credits required to run the operation will be returned.
    ----------------    ---------------------------------------------------------------
    future              Optional boolean. If True, a future object will be returned and the process
                        will not wait for the task to complete. The default is False, which means wait for results.
    ================    ===============================================================

    :return: result_layer : :class:`~arcgis.features.FeatureLayer` if ``output_name`` is specified, else Feature Collection.
    If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

        #USAGE EXAMPLE: To merge two layers into a new layer using merge attributes.
        merged = merge_layers(input_layer=esri_offices,
                              merge_layer=satellite_soffice_lyr,
                              merging_attributes=["State Match Place_Name"],
                              output_name="merge layers")
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "input_layer": input_layer,
        "merge_layer": merge_layer,
        "merging_attributes": merging_attributes,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.merge_layers, **kwargs
    )
    return gis._tools.featureanalysis.merge_layers(**params)


# ----------------------------------------------------------------------
def overlay_layers(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    overlay_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    overlay_type: str = "Intersect",
    snap_to_input: bool = False,
    output_type: str = "Input",
    tolerance: Optional[float] = None,
    output_name: Optional[Union[str, FeatureLayer]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images//overlay_layers/overlay_layers.png

    .. |Intersect| image:: _static/images/overlay_layers/overlay_intersect.png
    .. |Union| image:: _static/images/overlay_layers/overlay_union.png
    .. |Erase| image:: _static/images/overlay_layers/overlay_erase.png


    The ``overlay_layers`` method combines two or more layers into one single layer.
    You can think of overlay as peering through a stack of maps and creating a single map containing
    all the information found in the stack. In fact, before the advent of GIS, cartographers would
    literally copy maps onto clear acetate sheets, overlay these sheets on a light table, and hand
    draw a new map from the overlaid data. Overlay is much more than a merging of line work; all the
    attributes of the features taking part in the overlay are carried through to the final product.
    Overlay is used to answer one of the most basic questions of geography, "what is on top of what?" For example:

    + What parcels are within the 100-year floodplain? (Within is just another way of saying on top of.)
    + What roads are within what counties?
    + What land use is on top of what soil type?
    + What wells are within abandoned military bases?

    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    input_layer         Required layer. The point, line, or polygon features that will be
                        overlayed with the ``overlay_layer``. See :ref:`Feature Input<FeatureInput>`.
    ----------------    ---------------------------------------------------------------
    overlay_layer       Required layer. The features that will be overlaid with the ``input_layer`` features. See :ref:`Feature Input<FeatureInput>`.
    ----------------    ---------------------------------------------------------------
    overlay_type        Optional string. The type of overlay to be performed.

                        Choice list: ['Intersect', 'Union', 'Erase']

                        +--------------+--------------------------------------------------------------------------------------------------------+
                        | |Intersect|  | ``Intersect``-Computes a geometric intersection of the input layers. Features or portions of           |
                        |              | features which overlap in both the ``input_layer`` and ``overlay_layer`` layer will be written         |
                        |              | to the output layer. This is the default.                                                              |
                        +--------------+--------------------------------------------------------------------------------------------------------+
                        | |Union|      | ``Union``-Computes a geometric union of the input layers. All features and their attributes will       |
                        |              | be written to the output layer. This option is only valid if both the ``input_layer`` and              |
                        |              | the ``overlay_layer`` contain polygon features.                                                        |
                        +--------------+--------------------------------------------------------------------------------------------------------+
                        | |Erase|      | ``Erase``-Only those features or portions of features in the ``overlay_layer`` that are not within the |
                        |              | features in the ``input_layer`` layer are written to the output.                                       |
                        +--------------+--------------------------------------------------------------------------------------------------------+

                        The default value is 'Intersect'.

    ----------------    ---------------------------------------------------------------
    snap_to_input       Optional boolean. A Boolean value indicating if feature vertices in the ``input_layer`` are allowed to move.
                        The default is false and means if the distance between features is less than the ``tolerance`` value, all features from both
                        layers can move to allow snapping to each other. When set to true, only features in ``overlay_layer`` can move to snap to the ``input_layer`` features.
    ----------------    ---------------------------------------------------------------
    output_type         Optional string. The type of intersection you want to find.
                        This parameter is only valid when the ``overlay_type`` is Intersect.

                        Choice list: ['Input', 'Line', 'Point']

                        *  ``Input`` - The features returned will be the same geometry type as the ``input_layer`` or ``overlay_layer`` with the lowest dimension geometry.
                           If all inputs are polygons, the output will contain polygons. If one or more of the inputs are lines and none of the inputs are points, the output will be line. If one or more of the inputs are points, the output will contain points. This is the default.

                        *  ``Line`` - Line intersections will be returned. This is only valid if none of the inputs are points.

                        *  ``Point`` - Point intersections will be returned. If the inputs are line or polygon, the output will be a multipoint layer.
    ----------------    ---------------------------------------------------------------
    tolerance           Optional float. A float value of the minimum distance separating all feature coordinates
                        as well as the distance a coordinate can move in X or Y (or both). The units of tolerance are the same as the units of the ``input_layer``.
    ----------------    ---------------------------------------------------------------
    output_name         Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                        feature layer will cause the new layer to be appended to the Feature Service.
                        If overwrite is True in context, new layer will overwrite existing layer.
                        If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ----------------    ---------------------------------------------------------------
    context             Optional dict. Additional settings such as processing extent and output spatial reference.
                        For calculate_density, there are three settings.

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
    ----------------    ---------------------------------------------------------------
    gis                 Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    ----------------    ---------------------------------------------------------------
    future              Optional boolean. If True, a future object will be returned and the process
                        will not wait for the task to complete. The default is False, which means wait for results.
    ================    ===============================================================

    :return:
        result_layer : :class:`~arcgis.features.FeatureLayer` if ``output_name`` is specified, else Feature Collection.
    If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.


    .. code-block:: python

        #USAGE EXAMPLE: To clip a buffer in the shape of Capitol hill neighborhood.
        cliped_buffer = overlay_layers(buffer,
                                       neighbourhood,
                                       output_name="Cliped buffer")


    """
    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "input_layer": input_layer,
        "overlay_layer": overlay_layer,
        "overlay_type": overlay_type,
        "snap_to_input": snap_to_input,
        "output_type": output_type,
        "tolerance": tolerance,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.overlay_layers, **kwargs
    )
    return gis._tools.featureanalysis.overlay_layers(**params)


# ----------------------------------------------------------------------
def create_route_layers(
    route_data_item: Item,
    delete_route_data_item: bool = False,
    tags: Optional[str] = None,
    summary: Optional[str] = None,
    route_name_prefix: Optional[str] = None,
    folder_name: Optional[str] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    The ``create_route_layers`` method creates route layer items on the portal from the input route data.

    A route layer includes all the information for a particular route such as the stops assigned to
    the route as well as the travel directions. Creating route layers is useful if you want to share
    individual routes with other members in your organization.


    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    route_data                   Required item. The item id for the route data item that is used to create route layer items.
                                 Before running this task, the route data must be added to your portal as an item.
    -------------------------    ---------------------------------------------------------
    delete_route_data_item       Required boolean. Indicates if the input route data item should be deleted. You may want to
                                 delete the route data in case it is no longer required after the route layers have been created from it.

                                 When ``delete_route_data_item`` is set to true and the task fails to delete the route data item,
                                 it will return a warning message but still continue execution.

                                 The default value is False.
    -------------------------    ---------------------------------------------------------
    tags                         Optional string. Tags used to describe and identify the route layer items.
                                 Individual tags are separated using a comma. The route name is always
                                 added as a tag even when a value for this argument is not specified.
    -------------------------    ---------------------------------------------------------
    summary                      Optional string. The summary displayed as part of the item information for the route layer item.
                                 If a value for this argument is not specified, a default summary text "Route and directions for <Route Name>" is used.
    -------------------------    ---------------------------------------------------------
    route_name_prefix            Optional string. A qualifier added to the title of every route layer item. This can be used to designate all routes that are shared for a
                                 specific purpose to have the same prefix in the title. The name of the route is always appended after this qualifier.
                                 If a value for the route_name_prefix is not specified, the title for the route layer item is created using only the route name.
    -------------------------    ---------------------------------------------------------
    folder_name                  Optional string. The folder within your personal online workspace (My Content in your ArcGIS Online or Portal for ArcGIS organization) where the
                                 route layer items will be created. If a folder with the specified name does not exist, a new folder will be created.
                                 If a folder with the specified name exists, the items will be created in the existing folder.
                                 If a value for folder_name is not specified, the route layer items are created in the root folder of your online workspace.
    -------------------------    ---------------------------------------------------------
    gis                          Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    -------------------------    ---------------------------------------------------------
    estimate                     Optional boolean. If True, the estimated number of credits required to run the operation will be returned.
    -------------------------    ---------------------------------------------------------
    future                       Optional boolean. If True, a future object will be returned and the process
                                 will not wait for the task to complete. The default is False, which means wait for results.
    =========================    =========================================================

    :return: result_layer : A list (items) or an :class:`~arcgis.gis.Item`.
    If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

    .. code-block:: python

        USAGE EXAMPLE: To create route layers from geodatabase item.
        route = create_route_layers(route_data_item=route_item,
                            delete_route_data_item=False,
                            tags="datascience",
                            summary="example of create route layers method",
                            route_name_prefix="santa_ana",
                            folder_name="create route layers")
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    kwargs = {
        "route_data_item": route_data_item,
        "delete_route_data_item": delete_route_data_item,
        "tags": tags,
        "summary": summary,
        "route_name_prefix": route_name_prefix,
        "folder_name": folder_name,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params_tool = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.create_route_layers, **kwargs
    )
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis.create_route_layers, **kwargs
    )
    if "context" not in params_tool and "context" in params:
        params.pop("context", None)

    output_name = {}
    output_item_properties = {}
    if route_name_prefix:
        output_item_properties["title"] = route_name_prefix
    if tags:
        output_item_properties["tags"] = tags
    if summary:
        output_item_properties["snippet"] = summary
    if folder_name:
        folder_id = ""
        # Get a dict of folder names for the current user
        folders = {fld["title"]: fld for fld in gis.users.me.folders}
        # if the folder already exists, just get its folder id
        if folder_name in folders:
            folder_id = folders[folder_name].get("id", "")
        else:
            # Create a new folder and get its folder id
            new_folder = gis.content.create_folder(folder_name)
            folder_id = new_folder.get("id", "")
        if folder_id:
            output_item_properties["folderId"] = folder_id
    if output_item_properties:
        output_name["itemProperties"] = output_item_properties
    if output_name:
        params["output_name"] = output_name
    return gis._tools.featureanalysis.create_route_layers(**params)
