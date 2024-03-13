"""
These functions are used to identify areas that meet a number of different criteria you specify. These criteria can be based
upon attribute queries (for example, parcels that are vacant) and spatial queries (for example, within 1 kilometer of a
river). The areas that are found can be selected from existing features (such as existing land parcels) or new features
can be created where all the requirements are met.

find_existing_locations searches for existing areas in a layer that meet a series of criteria.
derive_new_locations creates new areas from locations in your study area that meet a series of criteria.
find_similar_locations finds locations most similar to one or more reference locations based on criteria you specify.
find_centroids finds and generates points from the representative center (centroid) of each input multipoint, line, or area feature.
choose_best_facilities choose the best locations for facilities by allocating locations that have demand for these
facilities in a way that satisfies a given goal.
create_viewshed creates areas that are visible based on locations you specify.
create_watersheds creates catchment areas based on locations you specify.
trace_downstream determines the flow paths in a downstream direction from the locations you specify
"""
from __future__ import annotations
from datetime import datetime
import logging
from re import U
from typing import Any, Optional, Union
import arcgis as _arcgis
from arcgis.features.feature import FeatureCollection
from arcgis.features.layer import FeatureLayer, FeatureLayerCollection
from arcgis.gis import GIS, Item
import arcgis.network as network
from .._impl.common._utils import inspect_function_inputs

_logger = logging.getLogger()


# --------------------------------------------------------------------------
def find_existing_locations(
    input_layers: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ] = None,
    expressions: Optional[dict[str, Any]] = None,
    output_name: Optional[Union[FeatureLayer, str]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/find_existing_locations/find_existing_locations.png

    .. |intersect| image:: _static/images/find_existing_locations/existing_intersect.png
    .. |distance| image:: _static/images/find_existing_locations/existing_distance.png
    .. |within| image:: _static/images/find_existing_locations/existing_within.png
    .. |nearest| image:: _static/images/find_existing_locations/existing_nearest.png
    .. |contains| image:: _static/images/find_existing_locations/existing_contains.png

    The ``find_existing_locations`` method selects features in the input layer that meet a query you specify.
    A query is made up of one or more expressions. There are two types of expressions: attribute and spatial.
    An example of an attribute expression is that a parcel must be vacant, which is an attribute of the Parcels layer (where STATUS = 'VACANT').
    An example of a spatial expression is that the parcel must also be within a certain distance of a river (Parcels within a distance of 0.75 Miles from Rivers).

    ====================================    ======================================================================================================
    **Parameter**                            **Description**
    ------------------------------------    ------------------------------------------------------------------------------------------------------
    input_layers                            Required list of feature layers. A list of layers that will be used in the expressions parameter. Each layer in the list can be:

                                            * a feature service layer with an optional filter to select specific features, or
                                            * a feature collection

                                            See :ref:`Feature Input<FeatureInput>`.
    ------------------------------------    ------------------------------------------------------------------------------------------------------
    expressions                             Required dict. There are two types of expressions, attribute and spatial.

                                            Example attribute expression:

                                                | {
                                                | "operator": "and",
                                                | "layer": 0,
                                                | "where": "STATUS = 'VACANT'"
                                                | }

                                            .. note::
                                                * operator can be either ``and`` or ``or``
                                                * layer is the index of the layer in the ``input_layers`` parameter.
                                                * The where clause must be surrounded by double quotes.
                                                * When dealing with text fields, values must be single-quoted ('VACANT').
                                                * Date fields support all queries except LIKE. Dates are strings in YYYY:MM:DD hh:mm:ss format.

                                            Here's an example using the date field ObsDate:

                                                "where": "ObsDate >= '1998-04-30 13:30:00' "

                                            +----------+------------------------------------------------------------------+
                                            | =        | Equal                                                            |
                                            +----------+------------------------------------------------------------------+
                                            | >        | Greater than                                                     |
                                            +----------+------------------------------------------------------------------+
                                            | <        | Less than                                                        |
                                            +----------+------------------------------------------------------------------+
                                            | >=       | Greater than or equal to                                         |
                                            +----------+------------------------------------------------------------------+
                                            | <=       | Less than or equal to                                            |
                                            +----------+------------------------------------------------------------------+
                                            | <>       | Not equal                                                        |
                                            +----------+------------------------------------------------------------------+
                                            | LIKE '%  | A percent symbol (%) signifies a wildcard, meaning that          |
                                            | <string>'| anything is acceptable in its place-one character, a             |
                                            |          | hundred characters, or no character. This expression             |
                                            |          | would select Mississippi and Missouri among USA                  |
                                            |          | state names: STATE_NAME LIKE 'Miss%'                             |
                                            +----------+------------------------------------------------------------------+
                                            | BETWEEN  | Selects a record if it has a value greater than or equal         |
                                            | <value1> | to <value1> and less than or equal to <value2>.                  |
                                            | AND      | For example, this expression selects all records with            |
                                            | <value2> | an HHSIZE value greater than or equal to 3 and less              |
                                            |          | than or equal to 10:                                             |
                                            |          |                                                                  |
                                            |          | HHSIZE BETWEEN 3 AND 10                                          |
                                            |          |                                                                  |
                                            |          | The above is equivalent to:                                      |
                                            |          |                                                                  |
                                            |          | HHSIZE >= 3 AND HHSIZE <= 10                                     |
                                            |          | This operator applies to numeric or date fields.                 |
                                            |          | Here is an example of a date query on the field ObsDate:         |
                                            |          |                                                                  |
                                            |          | ObsDate BETWEEN '1998-04-30 00:00:00' AND '1998-04-30 23:59:59'  |
                                            |          |                                                                  |
                                            |          | Time is optional.                                                |
                                            +----------+------------------------------------------------------------------+
                                            | NOT      | Selects a record if it has a value outside the range between     |
                                            | BETWEEN  | <value1> and less than or equal to <value2>.                     |
                                            | <value1> | For example, this expression selects all records whose           |
                                            | AND      | HHSIZE value is less than 5 and greater than 7.                  |
                                            | <value2> |                                                                  |
                                            |          | HHSIZE NOT BETWEEN 5 AND 7                                       |
                                            |          |                                                                  |
                                            |          | The above is equivalent to:                                      |
                                            |          |                                                                  |
                                            |          | HHSIZE < 5 OR HHSIZE > 7                                         |
                                            |          | This operator applies to numeric or date fields.                 |
                                            |          |                                                                  |
                                            |          | **Note**                                                         |
                                            |          |                                                                  |
                                            |          | You can use the contains relationship with points and lines.     |
                                            |          | For example, you have a layer of street centerlines (lines) and  |
                                            |          | a layer of manhole covers (points), and you want to find streets |
                                            |          | that contain a manhole cover. You could use contains to find     |
                                            |          | streets that contain manhole covers, but in order for a line to  |
                                            |          | contain a point, the point must be exactly on the line (that is, |
                                            |          | in GIS terms, they are snapped to each other). If there is any   |
                                            |          | doubt about this, use the withinDistance relationship with a     |
                                            |          | suitable distance value.                                         |
                                            +----------+------------------------------------------------------------------+

                                            Example spatial expression:

                                                | {
                                                | "operator": "and",
                                                | "layer": 0,
                                                | "spatialRel": "withinDistance",
                                                | "selectingLayer": 1,
                                                | "distance": 10,
                                                | "units": "miles"
                                                | }

                                            .. note::
                                                * operator can be either ``and`` or ``or``
                                                * layer is the index of the layer in ``the input_layers`` parameter. The result of the expression is features in this layer.
                                                * spatialRel is the spatial relationship. There are nine spatial relationships.
                                                * distance is the distance to use for the withinDistance and notWithinDistance spatial relationship.
                                                * units is the units for distance.

                                            +-------------------+----------------------------------------------------------------------------------------+
                                            | spatialRel        | Description                                                                            |
                                            +-------------------+----------------------------------------------------------------------------------------+
                                            | intersects        | |intersect|                                                                            |
                                            |                   |                                                                                        |
                                            |                   | A feature in layer passes the intersect test if it overlaps                            |
                                            | notIntersects     | any part of a feature in selectingLayer, including touches                             |
                                            |                   | (where features share a common point).                                                 |
                                            |                   |                                                                                        |
                                            |                   | * intersects-If a feature in layer intersects a feature in                             |
                                            |                   |   selectingLayer, the portion of the feature in layer that                             |
                                            |                   |   intersects the feature in selectingLayer is included in                              |
                                            |                   |   the output.                                                                          |
                                            |                   | * notintersects-If a feature in layer intersects a feature in                          |
                                            |                   |   selectingLayer, the portion of the feature in layer that                             |
                                            |                   |   intersects the feature in selectingLayer is excluded from                            |
                                            |                   |   the output.                                                                          |
                                            +-------------------+----------------------------------------------------------------------------------------+
                                            | withinDistance    | |distance|                                                                             |
                                            |                   |                                                                                        |
                                            |                   | The within a distance relationship uses the straight-line                              |
                                            | notWithinDistance | distance between features in layer to those in selectingLayer.                         |
                                            |                   | withinDistance-The portion of the feature in layer that is                             |
                                            |                   | within the specified distance of a feature in selectingLayer                           |
                                            |                   | is included in the output.                                                             |
                                            |                   | notwithinDistance-The portion of the feature in layer that is                          |
                                            |                   | within the specified distance of a feature in selectingLayer is                        |
                                            |                   | excluded from output. You can think of this relationship as                            |
                                            |                   | "is farther away than".                                                                |
                                            +-------------------+----------------------------------------------------------------------------------------+
                                            | contains          | |intersect|                                                                            |
                                            |                   |                                                                                        |
                                            |                   | A feature in layer passes this test if it completely                                   |
                                            | notContains       | surrounds a feature in selectingLayer. No portion of the                               |
                                            |                   | containing feature; however, the contained feature is allowed                          |
                                            |                   | to touch the containing feature (that is, share a common                               |
                                            |                   | point along its boundary).                                                             |
                                            |                   |                                                                                        |
                                            |                   | contains-If a feature in layer contains a feature in                                   |
                                            |                   | selectingLayer, the feature in layer is included in the output.                        |
                                            |                   | notcontains-If a feature in layer contains a feature in                                |
                                            |                   | selectingLayer, the feature in the first layer is excluded                             |
                                            +-------------------+----------------------------------------------------------------------------------------+
                                            | within            | |within|                                                                               |
                                            |                   |                                                                                        |
                                            |                   | A feature in layer passes this test if it is completely                                |
                                            | notWithin         | surrounded by a feature in selectingLayer. The entire feature                          |
                                            |                   | layer must be within the containing feature; however, the two                          |
                                            |                   | features are allowed to touch (that is, share a common point                           |
                                            |                   | along its boundary).                                                                   |
                                            |                   |                                                                                        |
                                            |                   | * within-If a feature in layer is completely within a feature in                       |
                                            |                   |   selectingLayer, the feature in layer is included in the output.                      |
                                            |                   | * notwithin-If a feature in layer is completely within a feature                       |
                                            |                   |   in selectingLayer, the feature in layer is excluded from the                         |
                                            |                   |   output.                                                                              |
                                            |                   |                                                                                        |
                                            |                   | **Note:**                                                                              |
                                            |                   |                                                                                        |
                                            |                   | can use the within relationship for points and lines, just as                          |
                                            |                   | you can with the contains relationship. For example, your first                        |
                                            |                   | layer contains points representing manhole covers and you want                         |
                                            |                   | to find the manholes that are on street centerlines (as opposed                        |
                                            |                   | to parking lots or other non-street features). You could use                           |
                                            |                   | within to find manhole points within street centerlines, but                           |
                                            |                   | in order for a point to contain a line, the point must be exactly                      |
                                            |                   | on the line (that is, in GIS terms, they are snapped to each                           |
                                            |                   | other). If there is any doubt about this, use the withinDistance                       |
                                            |                   | relationship with a suitable distance value.                                           |
                                            +-------------------+----------------------------------------------------------------------------------------+
                                            | nearest           | |nearest|                                                                              |
                                            |                   |                                                                                        |
                                            |                   | feature in the first layer passes this test if it is nearest                           |
                                            |                   | to a feature in the second layer.                                                      |
                                            |                   |                                                                                        |
                                            |                   | * nearest-If a feature in the first layer is nearest to a                              |
                                            |                   |   feature in the second layer, the feature in the first layer                          |
                                            |                   |   is included in the output.                                                           |
                                            +-------------------+----------------------------------------------------------------------------------------+

                                            * ``distance`` is the distance to use for the withinDistance and notWithinDistance spatial relationship.
                                            * ``units`` is the units for distance.

                                            Choice list: ['Meters', 'Kilometers', 'Feet', 'Yards', 'Miles']

                                            An expression may be a list, which denotes a group. The first operator in the group indicates how the group expression
                                            is added to the previous expression. Grouping expressions is only necessary when you need to create two or more distinct
                                            sets of features from the same layer. One way to think of grouping is that without grouping, you would have to execute
                                            ``find_existing_locations`` multiple times and merge the results.
    ------------------------------------    ------------------------------------------------------------------------------------------------------
    output_name                             Optional string or :class:`~arcgis.features.FeatureLayer` . Existing feature layer will cause the new layer to be appended to the Feature Service.
                                            If overwrite is True in context, new layer will overwrite existing layer. If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ------------------------------------    ------------------------------------------------------------------------------------------------------
    context                                 Optional dict. Additional settings such as processing extent and output spatial reference. For find_existing_locations, there are three settings.

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
    ------------------------------------    ------------------------------------------------------------------------------------------------------
    gis                                     Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------    ------------------------------------------------------------------------------------------------------
    estimate                                Optional, If True, a future object will be returned and the process
                                            will not wait for the task to complete. The default is False, which means wait for results.
    ====================================    ======================================================================================================

    :return: :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection`.

    .. code-block:: python

        #USAGE EXAMPLE: To find busy (where SEGMENT_TY is 1 and where ARTERIAL_C is 1) streets from the existing seattle streets layer.

        arterial_streets = find_existing_locations(input_layers=[bike_route_streets],
                                                   expressions=[{"operator":"","layer":0,"where":"SEGMENT_TY = 1"},
                                                   {"operator":"and","layer":0,"where":"ARTERIAL_C = 1"}],
                                                   output_name='ArterialStreets')



    """

    if input_layers is None:
        raise TypeError(
            "find_existing_locations missing 1 required positional argument: 'input_layers'"
        )
    if expressions is None:
        raise TypeError(
            "find_existing_locations missing 1 required positional argument: 'expressions'"
        )
    gis = _arcgis.env.active_gis if gis is None else gis
    if gis is None:
        raise TypeError(
            "Please make sure you are logged into an instance of ArcGIS Online or ArcGIS Enterprise"
        )
    kwargs = {
        "input_layers": input_layers,
        "expressions": expressions,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.find_existing_locations, **kwargs
    )
    return gis._tools.featureanalysis.find_existing_locations(**params)


# --------------------------------------------------------------------------
def derive_new_locations(
    input_layers: Union[
        list[FeatureLayer],
        list[FeatureCollection],
    ] = [],
    expressions: Optional[dict[str, Any]] = [],
    output_name: Optional[Union[FeatureLayer, str]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/derive_new_locations/derive_new_locations.png

    .. |intersect| image:: _static/images/derive_new_locations/derive_intersect.png
    .. |distance| image:: _static/images/derive_new_locations/derive_distance.png
    .. |within| image:: _static/images/derive_new_locations/derive_within.png
    .. |nearest| image:: _static/images/derive_new_locations/derive_nearest.png
    .. |contains| image:: _static/images/derive_new_locations/derive_contains.png

    The ``derive_new_locations`` method derives new features from the input layers that meet a query you specify. A query is
    made up of one or more expressions. There are two types of expressions: attribute and spatial. An example of an
    attribute expression is that a parcel must be vacant, which is an attribute of the Parcels layer
    (STATUS = 'VACANT'). An example of a spatial expression is that the parcel must also be within a certain
    distance of a river (Parcels within a distance of 0.75 Miles from Rivers).

    The ``derive_new_locations`` method is very similar to the ``find_existing_locations`` method, the main difference is that
    the result of ``derive_new_locations`` can contain partial features.

    * In both methods, the attribute expression  ``where`` and the spatial relationships within and contains return the same result.
      This is because these relationships return entire features.
    * When ``intersects`` or ``within_distance`` is used, ``derive_new_locations`` creates new features
      in the result. For example, when intersecting a parcel feature and a flood zone area that partially overlap each other,
      ``find_existing_locations`` will return the entire parcel whereas ``derive_new_locations`` will return just the portion of
      the parcel that is within the flood zone.

    =====================================    ======================================================================================================
    **Parameter**                             **Description**
    -------------------------------------    ------------------------------------------------------------------------------------------------------
    input_layers                             Required list of feature layers. A list of layers that will be used in the expressions parameter.
                                             Each layer in the list can be:

                                             * a feature service layer with an optional filter to select specific features, or
                                             * a feature collection
    -------------------------------------    ------------------------------------------------------------------------------------------------------
    expressions                              Required dict. There are two types of expressions, attribute and spatial.

                                             Example attribute expression:

                                                 | {
                                                 |   "operator": "and",
                                                 |   "layer": 0,
                                                 |   "where": "STATUS = 'VACANT'"
                                                 | }

                                             .. note::
                                                 * operator can be either ``and`` or ``or``
                                                 * layer is the index of the layer in the ``input_layers`` parameter.
                                                 * The where clause must be surrounded by double quotes.
                                                 * When dealing with text fields, values must be single-quoted ('VACANT').
                                                 * Date fields support all queries except LIKE. Dates are strings in YYYY:MM:DD hh:mm:ss format. Here's an example using the date field ObsDate:
                                                   "where": "ObsDate >= '1998-04-30 13:30:00' "

                                             +----------+------------------------------------------------------------------+
                                             | =        | Equal                                                            |
                                             +----------+------------------------------------------------------------------+
                                             | >        | Greater than                                                     |
                                             +----------+------------------------------------------------------------------+
                                             | <        | Less than                                                        |
                                             +----------+------------------------------------------------------------------+
                                             | >=       | Greater than or equal to                                         |
                                             +----------+------------------------------------------------------------------+
                                             | <=       | Less than or equal to                                            |
                                             +----------+------------------------------------------------------------------+
                                             | <>       | Not equal                                                        |
                                             +----------+------------------------------------------------------------------+
                                             | LIKE '%  | A percent symbol (%) signifies a wildcard, meaning that          |
                                             | <string>'| anything is acceptable in its place-one character, a             |
                                             |          | hundred characters, or no character. This expression             |
                                             |          | would select Mississippi and Missouri among USA                  |
                                             |          | state names: STATE_NAME LIKE 'Miss%'                             |
                                             +----------+------------------------------------------------------------------+
                                             | BETWEEN  | Selects a record if it has a value greater than or equal         |
                                             | <value1> | to <value1> and less than or equal to <value2>.                  |
                                             | AND      | For example, this expression selects all records with            |
                                             | <value2> | an HHSIZE value greater than or equal to 3 and less              |
                                             |          | than or equal to 10:                                             |
                                             |          |                                                                  |
                                             |          | HHSIZE BETWEEN 3 AND 10                                          |
                                             |          |                                                                  |
                                             |          | The above is equivalent to:                                      |
                                             |          |                                                                  |
                                             |          | HHSIZE >= 3 AND HHSIZE <= 10                                     |
                                             |          | This operator applies to numeric or date fields.                 |
                                             |          | Here is an example of a date query on the field ObsDate:         |
                                             |          |                                                                  |
                                             |          | ObsDate BETWEEN '1998-04-30 00:00:00' AND '1998-04-30 23:59:59'  |
                                             |          |                                                                  |
                                             |          | Time is optional.                                                |
                                             +----------+------------------------------------------------------------------+
                                             | NOT      | Selects a record if it has a value outside the range between     |
                                             | BETWEEN  | <value1> and less than or equal to <value2>.                     |
                                             | <value1> | For example, this expression selects all records whose           |
                                             | AND      | HHSIZE value is less than 5 and greater than 7.                  |
                                             | <value2> |                                                                  |
                                             |          | HHSIZE NOT BETWEEN 5 AND 7                                       |
                                             |          |                                                                  |
                                             |          | The above is equivalent to:                                      |
                                             |          |                                                                  |
                                             |          | HHSIZE < 5 OR HHSIZE > 7                                         |
                                             |          | This operator applies to numeric or date fields.                 |
                                             |          |                                                                  |
                                             |          | .. note::                                                        |
                                             |          |                                                                  |
                                             |          | You can use the contains relationship with points and lines.     |
                                             |          | For example, you have a layer of street centerlines (lines) and  |
                                             |          | a layer of manhole covers (points), and you want to find streets |
                                             |          | that contain a manhole cover. You could use contains to find     |
                                             |          | streets that contain manhole covers, but in order for a line to  |
                                             |          | contain a point, the point must be exactly on the line (that is, |
                                             |          | in GIS terms, they are snapped to each other). If there is any   |
                                             |          | doubt about this, use the withinDistance relationship with a     |
                                             |          | suitable distance value.                                         |
                                             +----------+------------------------------------------------------------------+

                                             Example spatial expression:

                                                 | {
                                                 |   "operator": "and",
                                                 |   "layer": 0,
                                                 |   "spatialRel": "withinDistance",
                                                 |   "selectingLayer": 1,
                                                 |   "distance": 10,
                                                 |   "units": "miles"
                                                 | }

                                             * operator can be either ``and`` or ``or``
                                             * layer is the index of the layer in ``the input_layers`` parameter. The result of the expression is features in this layer.
                                             * spatialRel is the spatial relationship. There are nine spatial relationships.
                                             * distance is the distance to use for the withinDistance and notWithinDistance spatial relationship.
                                             * units is the units for distance.

                                             +-------------------+----------------------------------------------------------------------------------------+
                                             | spatialRel        | Description                                                                            |
                                             +-------------------+----------------------------------------------------------------------------------------+
                                             | intersects        | |intersect|                                                                            |
                                             |                   |                                                                                        |
                                             |                   | A feature in layer passes the intersect test if it overlaps                            |
                                             | notIntersects     | any part of a feature in selectingLayer, including touches                             |
                                             |                   | (where features share a common point).                                                 |
                                             |                   |                                                                                        |
                                             |                   | * intersects-If a feature in layer intersects a feature in                             |
                                             |                   |   selectingLayer, the portion of the feature in layer that                             |
                                             |                   |   intersects the feature in selectingLayer is included in                              |
                                             |                   |   the output.                                                                          |
                                             |                   | * notintersects-If a feature in layer intersects a feature in                          |
                                             |                   |   selectingLayer, the portion of the feature in layer that                             |
                                             |                   |   intersects the feature in selectingLayer is excluded from                            |
                                             |                   |   the output.                                                                          |
                                             +-------------------+----------------------------------------------------------------------------------------+
                                             | withinDistance    | |distance|                                                                             |
                                             |                   |                                                                                        |
                                             |                   | The within a distance relationship uses the straight-line                              |
                                             | notWithinDistance | distance between features in layer to those in selectingLayer.                         |
                                             |                   | withinDistance-The portion of the feature in layer that is                             |
                                             |                   | within the specified distance of a feature in selectingLayer                           |
                                             |                   | is included in the output.                                                             |
                                             |                   | notwithinDistance-The portion of the feature in layer that is                          |
                                             |                   | within the specified distance of a feature in selectingLayer is                        |
                                             |                   | excluded from output. You can think of this relationship as                            |
                                             |                   | "is farther away than".                                                                |
                                             +-------------------+----------------------------------------------------------------------------------------+
                                             | contains          | |intersect|                                                                            |
                                             |                   |                                                                                        |
                                             |                   | A feature in layer passes this test if it completely                                   |
                                             | notContains       | surrounds a feature in selectingLayer. No portion of the                               |
                                             |                   | containing feature; however, the contained feature is allowed                          |
                                             |                   | to touch the containing feature (that is, share a common                               |
                                             |                   | point along its boundary).                                                             |
                                             |                   |                                                                                        |
                                             |                   | contains-If a feature in layer contains a feature in                                   |
                                             |                   | selectingLayer, the feature in layer is included in the output.                        |
                                             |                   | notcontains-If a feature in layer contains a feature in                                |
                                             |                   | selectingLayer, the feature in the first layer is excluded                             |
                                             +-------------------+----------------------------------------------------------------------------------------+
                                             | within            | |within|                                                                               |
                                             |                   |                                                                                        |
                                             |                   | A feature in layer passes this test if it is completely                                |
                                             | notWithin         | surrounded by a feature in selectingLayer. The entire feature                          |
                                             |                   | layer must be within the containing feature; however, the two                          |
                                             |                   | features are allowed to touch (that is, share a common point                           |
                                             |                   | along its boundary).                                                                   |
                                             |                   |                                                                                        |
                                             |                   | * within-If a feature in layer is completely within a feature in                       |
                                             |                   |   selectingLayer, the feature in layer is included in the output.                      |
                                             |                   | * notwithin-If a feature in layer is completely within a feature                       |
                                             |                   |   in selectingLayer, the feature in layer is excluded from the                         |
                                             |                   |   output.                                                                              |
                                             |                   |                                                                                        |
                                             |                   | **Note:**                                                                              |
                                             |                   |                                                                                        |
                                             |                   | can use the within relationship for points and lines, just as                          |
                                             |                   | you can with the contains relationship. For example, your first                        |
                                             |                   | layer contains points representing manhole covers and you want                         |
                                             |                   | to find the manholes that are on street centerlines (as opposed                        |
                                             |                   | to parking lots or other non-street features). You could use                           |
                                             |                   | within to find manhole points within street centerlines, but                           |
                                             |                   | in order for a point to contain a line, the point must be exactly                      |
                                             |                   | on the line (that is, in GIS terms, they are snapped to each                           |
                                             |                   | other). If there is any doubt about this, use the withinDistance                       |
                                             |                   | relationship with a suitable distance value.                                           |
                                             +-------------------+----------------------------------------------------------------------------------------+
                                             | nearest           | |nearest|                                                                              |
                                             |                   |                                                                                        |
                                             |                   | feature in the first layer passes this test if it is nearest                           |
                                             |                   | to a feature in the second layer.                                                      |
                                             |                   |                                                                                        |
                                             |                   | * nearest-If a feature in the first layer is nearest to a                              |
                                             |                   |   feature in the second layer, the feature in the first layer                          |
                                             |                   |   is included in the output.                                                           |
                                             +-------------------+----------------------------------------------------------------------------------------+

    -------------------------------------    ------------------------------------------------------------------------------------------------------
    output_name                              Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                             feature layer will cause the new layer to be appended to the Feature Service.
                                             If overwrite is True in context, new layer will overwrite existing layer.
                                             If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------------------    ------------------------------------------------------------------------------------------------------
    context                                  Optional dict. Additional settings such as processing extent and output spatial reference.
                                             For derive_new_locations, there are three settings.

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
    -------------------------------------    ------------------------------------------------------------------------------------------------------
    gis                                      Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    -------------------------------------    ------------------------------------------------------------------------------------------------------
    estimate                                 Optional boolean. Is true, the number of credits needed to run the operation will be returned as a float.
    -------------------------------------    ------------------------------------------------------------------------------------------------------
    future                                   Optional, If True, a future object will be returned and the process
                                             will not wait for the task to complete. The default is False, which means wait for results.
    =====================================    ======================================================================================================

    :return: :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection`.

    .. code-block:: python

        USAGE EXAMPLE: To Identify areas that are suitable cougar habitat using the criteria defined by experts.

        new_location = derive_new_locations(input_layers=[slope, vegetation, streams, highways],
                                    expressions=[{"operator":"","layer":0,"selectingLayer":1,"spatialRel":"intersects"},
                                                 {"operator":"and","layer":0,"selectingLayer":2,"spatialRel":"withinDistance","distance":500,"units":"Feet"},
                                                 {"operator":"and","layer":0,"selectingLayer":3,"spatialRel":"notWithinDistance","distance":1500,"units":"Feet"},
                                                 {"operator":"and","layer":0,"where":"GRIDCODE = 1"}],
                                    output_name='derive_new_loactions')



    """
    if input_layers is None:
        raise TypeError(
            "derive_new_locations missing 1 required positional argument: 'input_layers'"
        )
    if expressions is None:
        raise TypeError(
            "derive_new_locations missing 1 required positional argument: 'input_layer'"
        )

    gis = _arcgis.env.active_gis if gis is None else gis
    if gis is None:
        raise TypeError(
            "Please make sure you are logged into an instance of ArcGIS Online or ArcGIS Enterprise"
        )
    kwargs = {
        "input_layers": input_layers,
        "expressions": expressions,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.derive_new_locations, **kwargs
    )

    return gis._tools.featureanalysis.derive_new_locations(**params)


# --------------------------------------------------------------------------
def find_similar_locations(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    search_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    analysis_fields: Optional[list[str]] = [],
    input_query: Optional[str] = None,
    number_of_results: int = 0,
    output_name: Optional[Union[FeatureLayer, str]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/find_similar_locations/find_similar_locations.png

    The ``find_similar_locations`` method measures the similarity of candidate locations to one or more reference locations.

    Based on criteria you specify, ``find_similar_locations`` can answer questions such as the following:

    - Which of your stores are most similar to your top performers with regard to customer profiles?
    - Based on characteristics of villages hardest hit by the disease, which other villages are high risk?

    To answer questions such as these, you provide the reference locations (the ``input_layer`` parameter), the candidate
    locations (the ``search_layer`` parameter), and the fields representing the criteria you want to match. For example,
    the ``input_layer`` might be a layer containing your top performing stores or the villages hardest hit by the disease.
    The ``search_layer`` contains your candidate locations to search. This might be all of your stores or all other villages.
    Finally, you supply a list of fields to use for measuring similarity. ``find_similar_locations`` will rank all of the
    candidate locations by how closely they match your reference locations across all of the fields you have selected.

    =======================     ===========================================================================================
    **Parameter**                **Description**
    -----------------------     -------------------------------------------------------------------------------------------
    input_layer                 Required feature layer. The ``input_layer`` contains one or more
                                reference locations against which features in the ``search_layer``
                                will be evaluated for similarity. For example, the ``input_layer``
                                might contain your top performing stores or the villages hardest
                                hit by a disease.
                                It is not uncommon that the ``input_layer`` and ``search_layer`` are the
                                same feature service. For example, the feature service contains
                                locations of all stores, one of which is your top performing store.
                                If you want to rank the remaining stores from most to least similar
                                to your top performing store, you can provide a filter for both the
                                inputLayer and the ``search_layer``. The filter on the ``input_layer`` would
                                select the top performing store while the filter on the ``search_layer``
                                would select all stores except for the top performing store. You can
                                also use the optional ``input_query`` parameter to specify reference locations.

                                If there is more than one reference location, similarity will be based
                                on averages for the fields you specify in the ``analysis_fields`` parameter.
                                So, for example, if there are two reference locations and you are
                                interested in matching population, the task will look for candidate
                                locations in the ``search_layer`` with populations that are most like the
                                average population for both reference locations. If the values for the
                                reference locations are 100 and 102, for example, the method will look
                                for candidate locations with populations near 101. Consequently, you
                                will want to use fields for the reference locations fields that have
                                similar values. If, for example, the population values for one reference
                                location is 100 and the other is 100,000, the tool will look for candidate
                                locations with population values near the average of those two values: 50,050.
                                Notice that this averaged value is nothing like the population for either
                                of the reference locations. See :ref:`Feature Input<FeatureInput>`.
    -----------------------     -------------------------------------------------------------------------------------------
    search_layer                Required feature layer. The layer containing candidate locations that
                                will be evaluated against the reference locations. See :ref:`Feature Input<FeatureInput>`.
    -----------------------     -------------------------------------------------------------------------------------------
    analysis_fields             Required list of strings. A list of fields whose values are used to determine similarity.
                                They must be numeric fields and the fields must exist on both the ``input_layer`` and
                                the ``search_layer``. The method will find features in the ``search_layer`` that have field
                                values closest to those of the features in your ``input_layer``.
    -----------------------     -------------------------------------------------------------------------------------------
    input_query                 Optional string. In the situation where the ``input_layer`` and the ``search_layer`` are the same feature service,
                                this parameter allows you to input a query on the ``input_layer`` to specify which features are the reference locations.
                                The reference locations specified by this query will not be analyzed as candidates.
                                The syntax of ``input_query`` is the same as a filter.
    -----------------------     -------------------------------------------------------------------------------------------
    number_of_results           Optional int. The number of ranked candidate locations output to the ``similar_result_layer``.
                                If ``number_of_results`` is not specified, or set to zero, all candidate locations will be ranked and output.
    -----------------------     -------------------------------------------------------------------------------------------
    output_name                 Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                feature layer will cause the new layer to be appended to the Feature Service.
                                If overwrite is True in context, new layer will overwrite existing layer.
                                If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -----------------------     -------------------------------------------------------------------------------------------
    context                     Optional dict. Additional settings such as processing extent and output spatial reference.
                                For find_similar_locations, there are three settings.

                                - ``extent`` - a bounding box that defines the analysis area. Only those features in the input_layer that intersect the bounding box will be analyzed.
                                - ``outSR`` - the output features will be projected into the output spatial reference referred to by the `wkid`.
                                - ``overwrite`` - if True, then the feature layer in output_name will be overwritten with new feature layer. Available for ArcGIS Online and ArcGIS Enterprise 11.1+.

                                    .. code-block:: python

                                        # Example Usage
                                        context = {"extent": {"xmin": 3164569.408035,
                                                            "ymin": -9187921.892449,
                                                            "xmax": 3174104.927313,
                                                            "ymax": -9175500.875353,
                                                            "spatialReference":{"wkid":102100,"latestWkid":3857}},
                                                    "outSR": {"wkid": 3857},
                                                    "overwrite": True}
    -----------------------     -------------------------------------------------------------------------------------------
    estimate                    Optional boolean. If True, the number of credits to run the operation will be returned.
    -----------------------     -------------------------------------------------------------------------------------------
    future                      Optional, If True, a future object will be returned and the process
                                will not wait for the task to complete. The default is False, which means wait for results.
    =======================     ===========================================================================================

    :return: :class:`~arcgis.features.FeatureLayer` if ``output_name`` is specified, else Python dictionary with the following keys:

        "similar_result_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

        "process_info" : list of message

    .. code-block:: python

        #USAGE EXAMPLE: To find top 4 most locations from the candidates layer that are similar to the target location.
        top_4_most_similar_locations = find_similar_locations(target_lyr, candidates_lyr,
                                                    analysis_fields=['THH17','THH35','THH02','THH05','POPDENS14','FAMGRW10_14','UNEMPRT_CY'],
                                                    output_name = "top 4 similar locations",
                                                    number_of_results=4)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    if gis is None:
        raise TypeError(
            "Please make sure you are logged into an instance of ArcGIS Online or ArcGIS Enterprise"
        )
    kwargs = {
        "input_layer": input_layer,
        "search_layer": search_layer,
        "analysis_fields": analysis_fields,
        "input_query": input_query,
        "number_of_results": number_of_results,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.find_similar_locations, **kwargs
    )

    return gis._tools.featureanalysis.find_similar_locations(**params)


# --------------------------------------------------------------------------
def find_centroids(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    point_location: bool = False,
    output_name: Optional[Union[FeatureLayer, str]] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/find_centroids/find_centroids.png

    The ``find_centroids`` method that finds and generates points from the representative center (centroid) of
    each input multipoint, line, or area feature. Finding the centroid of a feature is very common for many analytical
    workflows where the resulting points can then be used in other analytic workflows.

    For example, polygon features that contain demographic data can be converted to centroids that can be used in network analysis.

    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    input_layer         Required feature layer. The multipoint, line, or polygon features that will be used to generate centroid point features. See :ref:`Feature Input<FeatureInput>`.
    ----------------    ---------------------------------------------------------------
    point_location      Optional boolean. A Boolean value that determines the output location of the points.

                        + True - Output points will be the nearest point to the actual centroid, but located inside or contained by the bounds of the input feature.
                        + False - Output point locations will be determined by the calculated geometric center of each input feature. This is the default.
    ----------------    ---------------------------------------------------------------
    output_name         Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                        feature layer will cause the new layer to be appended to the Feature Service.
                        If overwrite is True in context, new layer will overwrite existing layer.
                        If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    ----------------    ---------------------------------------------------------------
    context             Optional dict. Additional settings such as processing extent and output spatial reference.
                        For find_centroids, there are three settings.

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
    estimate            Optional boolean. If True, the number of credits to run the operation will be returned.
    ----------------    ---------------------------------------------------------------
    future              Optional, If True, a future object will be returned and the process
                        will not wait for the task to complete. The default is False, which means wait for results.
    ================    ===============================================================

    :return: result_layer : :class:`~arcgis.features.FeatureLayer` if ``output_name`` is specified, else :class:`~arcgis.features.FeatureCollection`.

    .. code-block:: python

        # USAGE EXAMPLE: To find centroids of madison fields nearest to the actual centroids.

        centroid = find_centroids(madison_fields,
                                  point_location=True,
                                  output_name='find centroids')
    """
    gis = _arcgis.env.active_gis if gis is None else gis
    if gis is None:
        raise TypeError(
            "Please make sure you are logged into an instance of ArcGIS Online or ArcGIS Enterprise"
        )
    if gis._portal.is_arcgisonline == False and gis.version < [7, 3]:
        raise Exception(
            "find_centroids is only available on ArcGIS Online and ArcGIS Enterprise 10.8.0+"
        )
    kwargs = {
        "input_layer": input_layer,
        "point_location": point_location,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params_tool = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.find_centroids, **kwargs
    )
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis.find_centroids, **kwargs
    )
    if "context" not in params_tool and "context" in params:
        del params["context"]
    return gis._tools.featureanalysis.find_centroids(**params)


# --------------------------------------------------------------------------
def choose_best_facilities(
    goal: str = "Allocate",
    demand_locations_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    demand: float = 1,
    demand_field: Optional[str] = None,
    max_travel_range: float = 2147483647,
    max_travel_range_field: Optional[str] = None,
    max_travel_range_units: str = "Minutes",
    travel_mode: Optional[str] = None,
    time_of_day: Optional[datetime] = None,
    time_zone_for_time_of_day: str = "GeoLocal",
    travel_direction: str = "FacilityToDemand",
    required_facilities_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    required_facilities_capacity: float = 2147483647,
    required_facilities_capacity_field: Optional[str] = None,
    candidate_facilities_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    candidate_count: float = 1,
    candidate_facilities_capacity: float = 2147483647,
    candidate_facilities_capacity_field: Optional[str] = None,
    percent_demand_coverage: float = 100,
    output_name: Optional[Union[FeatureLayer, str]] = None,
    context: Optional[dict] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    point_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    line_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    polygon_barrier_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    future: bool = False,
):
    """
    .. image:: _static/images/choose_best_facilities/choose_best_facilities.png

    The ``choose_best_facilities`` method finds the set of facilities that will best serve demand from surrounding areas.

    Facilities might be public institutions that offer a service, such as fire stations, schools, or libraries,
    or they might be commercial ones such as drug stores or distribution centers for a parcel delivery service.
    Demand represents the need for a service that the facilities can meet. Demand is associated with point locations,
    with each location representing a given amount of demand.

    =====================================    =========================================================
    **Parameter**                             **Description**
    -------------------------------------    ---------------------------------------------------------
    goal                                     Optional string. Specify the goal that must be satisfied when allocating
                                             demand locations to facilities.

                                             Choice list: ['Allocate', 'MinimizeImpedance', 'MaximizeCoverage', 'MaximizeCapacitatedCoverage', 'PercentCoverage']

                                             Default value is 'Allocate'.

    -------------------------------------    ---------------------------------------------------------
    demand_locations_layer                   Required point feature layer. A point layer specifying the locations
                                             that have demand for facilities. See :ref:`Feature Input<FeatureInput>`.
    -------------------------------------    ---------------------------------------------------------
    demand                                   Optional float. The amount of demand available at every demand locations.

                                             The default value is 1.0.
    -------------------------------------    ---------------------------------------------------------
    demand_field                             Optional string. A numeric field on the ``demand_locations_layer``
                                             representing the amount of demand available at each demand location.
                                             If specified, the ``demand`` parameter is ignored.
    -------------------------------------    ---------------------------------------------------------
    max_travel_range                         Optional float. Specify the maximum travel time or distance allowed
                                             between a demand location and the facility it is allocated to.

                                             The default is unlimited (2,147,483,647.0).
    -------------------------------------    ---------------------------------------------------------
    max_travel_range_field                   Optional string. A numeric field on the ``demand_locations_layer`` specifying the maximum
                                             travel time or distance allowed between a demand location
                                             and the facility it is allocated to. If specified, the ``max_travel_range`` parameter is ignored.
    -------------------------------------    ---------------------------------------------------------
    max_travel_range_units                   Optional string. The units for the maximum travel time or distance allowed
                                             between a demand location and the facility it is allocated to.

                                             Choice list: ['Seconds', 'Minutes', 'Hours', 'Days', 'Meters', 'Kilometers', 'Feet', 'Yards', 'Miles'].

                                             The default is 'Minutes'.
    -------------------------------------    ---------------------------------------------------------
    travel_mode                              Specify the mode of transportation for the analysis.

                                             Choice list: ['Driving Distance', 'Driving Time', 'Rural Driving Distance', 'Rural Driving Time', 'Trucking Distance', 'Trucking Time', 'Walking Distance', 'Walking Time']

    -------------------------------------    ---------------------------------------------------------
    time_of_day                              Optional datetime.datetime. Specify whether travel times
                                             should consider traffic conditions. To use traffic in the
                                             analysis, set travel_mode to a travel mode object whose
                                             impedance_attribute_name property is set to travel_time and
                                             assign a value to time_of_day. (A travel mode with other
                                             impedance_attribute_name values don't support traffic.)
                                             The ``time_of_day`` value represents the time at which travel
                                             begins, or departs, from the origin points. The time is
                                             specified as datetime.datetime.

                                             The service supports two kinds of traffic: ty
                                             pical and live.
                                             Typical traffic references travel speeds that are made up of
                                             historical averages for each five-minute interval spanning a week.
                                             Live traffic retrieves speeds from a traffic feed that processes
                                             phone probe records, sensors, and other data sources to record
                                             actual travel speeds and predict speeds for the near future.

                                             The `data coverage <http://www.arcgis.com/home/webmap/viewer.html?webmap=b7a893e8e1e04311bd925ea25cb8d7c7>`_ page shows the countries Esri currently provides traffic data for.

                                             Typical Traffic:

                                             To ensure the task uses typical traffic in locations where it
                                             is available, choose a time and day of the week, and then convert
                                             the day of the week to one of the following dates from 1990:

                                             * Monday - 1/1/1990
                                             * Tuesday - 1/2/1990
                                             * Wednesday - 1/3/1990
                                             * Thursday - 1/4/1990
                                             * Friday - 1/5/1990
                                             * Saturday - 1/6/1990
                                             * Sunday - 1/7/1990

                                             Set the time and date as datetime.datetime.

                                             For example, to solve for 1:03 p.m. on Thursdays, set the
                                             time and date to 1:03 p.m., 4 January 1990; and convert to
                                             datetime eg. datetime.datetime(1990, 1, 4, 1, 3).

                                             Live Traffic:

                                             To use live traffic when and where it is available,
                                             choose a time and date and convert to datetime.

                                             Esri saves live traffic data for 4 hours and references
                                             predictive data extending 4 hours into the future. If the
                                             time and date you specify for this parameter is outside the
                                             24-hour time window, or the travel time in the analysis
                                             continues past the predictive data window, the task falls
                                             back to typical traffic speeds.

                                             Examples:
                                             from datetime import datetime

                                             * "time_of_day": datetime(1990, 1, 4, 1, 3) # 13:03, 4 January 1990. Typical traffic on Thursdays at 1:03 p.m.
                                             * "time_of_day": datetime(1990, 1, 7, 17, 0) # 17:00, 7 January 1990. Typical traffic on Sundays at 5:00 p.m.
                                             * "time_of_day": datetime(2014, 10, 22, 8, 0) # 8:00, 22 October 2014. If the current time is between 8:00 p.m., 21 Oct. 2014 and 8:00 p.m., 22 Oct. 2014, live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are referenced.
                                             * "time_of_day": datetime(2015, 3, 18, 10, 20) # 10:20, 18 March 2015. If the current time is between 10:20 p.m., 17 Mar. 2015 and 10:20 p.m., 18 Mar. 2015, live traffic speeds are referenced in the analysis; otherwise, typical traffic speeds are referenced.
    -------------------------------------    ---------------------------------------------------------
    time_zone_for_time_of_day                Optional string. Specify the time zone or zones of the time_of_day parameter.

                                             Choice list: ['GeoLocal', 'UTC']

                                             GeoLocal-refers to the time zone in which the origins_layer points are located.

                                             UTC-refers to Coordinated Universal Time.
    -------------------------------------    ---------------------------------------------------------
    travel_direction                         Optional string. Specify whether to measure travel times or distances
                                             from facilities to demand locations or from demand locations to facilities.

                                             Choice list: ['FacilityToDemand', 'DemandToFacility']

    -------------------------------------    ---------------------------------------------------------
    required_facilities_layer                Optional point feature layer. A point layer specifying one or more locations that act as facilities
                                             by providing some kind of service. Facilities specified by this parameter
                                             are required to be part of the output solution and will be used before any
                                             facilities from the ``candidate_facilities_layer`` when allocating demand locations.
    -------------------------------------    ---------------------------------------------------------
    required_facilities_capacity             Optional float. Specify how much demand every facility in the ``required_facilities_layer``
                                             is capable of supplying.

                                             The default value is unlimited (2,147,483,647).

    -------------------------------------    ---------------------------------------------------------
    required_facilities_capacity_field       Optional string. A field on the required_facilities_layer
                                             representing how much demand each facility in the
                                             ``required_facilities_layer`` is capable of supplying. This
                                             parameter takes precedence when ``required_facilities_capacity``
                                             parameter is also specified.
    -------------------------------------    ---------------------------------------------------------
    candidate_facilities_layer               Optional point layer. A point layer specifying one or more
                                             locations that act as facilities by providing some kind of
                                             service. Facilities specified by this parameter are not
                                             required to be part of the output solution and will be used
                                             only after all the facilities from the
                                             ``candidate_facilities_layer`` have been used when
                                             allocating demand locations.

    -------------------------------------    ---------------------------------------------------------
    candidate_count                          Optional integer. The number of candidate facilities to
                                             choose when allocating demand locations. Note that the sum
                                             of the features in the ``required_facilities_capacity``
                                             and the value specified for ``candidate_count`` cannot
                                             exceed 100.

                                             The default value is 1.

    -------------------------------------    ---------------------------------------------------------
    candidate_facilities_capacity            Optional float. Specify how much demand every facility in
                                             the ``candidate_facilities_layer`` is capable of supplying.

                                             The default value is unlimited (2,147,483,647.0).

    -------------------------------------    ---------------------------------------------------------
    candidate_facilities_capacity_field      Optional string. A field on the ``candidate_facilities_layer``
                                             representing how much demand each facility in the
                                             ``candidate_facilities_layer`` is capable of supplying. This
                                             parameter takes precedence when ``candidate_facilities_capacity``
                                             parameter is also specified.
    -------------------------------------    ---------------------------------------------------------
    percent_demand_coverage                  Optional float. Specify the percentage of the total demand
                                             that you want the chosen and required facilities to capture.

                                             The default value is 100.
    -------------------------------------    ---------------------------------------------------------
    output_name                              Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                             feature layer will cause the new layer to be appended to the Feature Service.
                                             If overwrite is True in context, new layer will overwrite existing layer.
                                             If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------------------    ---------------------------------------------------------
    context                                  Optional dict. Additional settings such as processing extent and output spatial reference.
                                             For choose_best_facilities, there are three settings.

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
    -------------------------------------    ---------------------------------------------------------
    gis                                      Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not
                                             specified, the active GIS is used.
    -------------------------------------    ---------------------------------------------------------
    estimate                                 Optional boolean. Is true, the number of credits needed
                                             to run the operation will be returned as a float.
    -------------------------------------    ---------------------------------------------------------
    point_barrier_layer                      Optional layer. Specify one or more point features that
                                             act as temporary restrictions (in other words, barriers)
                                             when traveling on the underlying streets.

                                             A point barrier can model a fallen tree, an accident, a
                                             downed electrical line, or anything that completely blocks
                                             traffic at a specific position along the street. Travel is
                                             permitted on the street but not through the barrier.
                                             See :ref:`Feature Input<FeatureInput>`.
    -------------------------------------    ---------------------------------------------------------
    line_barrier_layer                       Optional layer. Specify one or more line features that prohibit travel anywhere the lines intersect the streets.

                                             A line barrier prohibits travel anywhere the barrier intersects the streets. For example, a parade or protest that blocks traffic across several street
                                             segments can be modeled with a line barrier. See :ref:`Feature Input<FeatureInput>`.
    -------------------------------------    ---------------------------------------------------------
    polygon_barrier_layer                    Optional layer. Specify one or more polygon features that completely restrict travel on the streets intersected by the polygons.

                                             One use of this type of barrier is to model floods covering areas of the street network and making road travel there impossible. See :ref:`Feature Input<FeatureInput>`.
    -------------------------------------    ---------------------------------------------------------
    future                                   Optional boolean. If True, a future object will be returned and the process
                                             will not wait for the task to complete. The default is False, which means wait for results.
    =====================================    =========================================================


    :return: When an output_name is specified, a :class:`~arcgis.features.FeatureCollection` with 3 layers is returned (see dictionary below for details), else a dict with the following keys:

       "allocated_demand_locations_layer" : layer (:class:`~arcgis.features.FeatureCollection`)

       "allocation_lines_layer"  : layer (:class:`~arcgis.features.FeatureCollection`)

       "assigned_facilities_layer"   : layer (:class:`~arcgis.features.FeatureCollection`)

    .. code-block:: python

        USAGE EXAMPLE: To minimize overall distance travelled for travelling from esri offices to glider airports.

        best_facility = choose_best_facilities(goal="MinimizeImpedance",
                                    demand_locations_layer=esri_offices,
                                    travel_mode='Driving Distance',
                                    travel_direction="DemandToFacility",
                                    required_facilities_layer=gliderport_lyr,
                                    candidate_facilities_layer=balloonport_lyr,
                                    candidate_count=1,
                                    output_name="choose best facilities")
    """
    if demand_locations_layer is None:
        raise TypeError(
            "choose_best_facilities missing 1 required positional argument: 'demand_locations_layer'"
        )

    gis = _arcgis.env.active_gis if gis is None else gis
    if gis is None:
        raise TypeError(
            "Please make sure you are logged into an instance of ArcGIS Online or ArcGIS Enterprise"
        )
    kwargs = {
        "goal": goal,
        "demand_locations_layer": demand_locations_layer,
        "demand": demand,
        "demand_field": demand_field,
        "max_travel_range": max_travel_range,
        "max_travel_range_field": max_travel_range_field,
        "max_travel_range_units": max_travel_range_units,
        "travel_mode": travel_mode,
        "time_of_day": time_of_day,
        "time_zone_for_time_of_day": time_zone_for_time_of_day,
        "travel_direction": travel_direction,
        "required_facilities_layer": required_facilities_layer,
        "required_facilities_capacity": required_facilities_capacity,
        "required_facilities_capacity_field": required_facilities_capacity_field,
        "candidate_facilities_layer": candidate_facilities_layer,
        "candidate_count": candidate_count,
        "candidate_facilities_capacity": candidate_facilities_capacity,
        "candidate_facilities_capacity_field": candidate_facilities_capacity_field,
        "percent_demand_coverage": percent_demand_coverage,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "point_barrier_layer": point_barrier_layer,
        "line_barrier_layer": line_barrier_layer,
        "polygon_barrier_layer": polygon_barrier_layer,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.choose_best_facilities, **kwargs
    )
    try:
        if isinstance(travel_mode, str):
            travel_mode = network._utils.find_travel_mode(
                gis=gis, travel_mode=travel_mode
            )
            params["travel_mode"] = travel_mode
        elif isinstance(travel_mode, dict):
            params["travel_mode"] = travel_mode
        else:
            params["travel_mode"] = network._utils.find_travel_mode(gis=gis)
    except Exception as e:
        msg = f"Using the given travel_mode without validation due to the following error: {str(e)}"
        _logger.warn(msg)

    return gis._tools.featureanalysis.choose_best_facilities(**params)


# --------------------------------------------------------------------------
def create_viewshed(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    dem_resolution: str = "Finest",
    maximum_distance: Optional[float] = None,
    max_distance_units: str = "Meters",
    observer_height: Optional[float] = None,
    observer_height_units: str = "Meters",
    target_height: Optional[float] = None,
    target_height_units: str = "Meters",
    generalize: bool = True,
    output_name: Optional[Union[FeatureLayer, str]] = None,
    context: Optional[dict] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/create_viewshed/create_viewshed.png

    The create_viewshed method identifies visible areas based on the observer locations you provide.
    The results are areas where the observers can see the observed objects (and the observed objects can see the observers).

    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    input_layer                  Required point feature layer. The features to use as the observer locations. See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    dem_resolution               Optional string. The approximate spatial resolution (cell size) of the source elevation data used for the calculation.

                                 The resolution values are an approximation of the spatial resolution of the digital elevation model.
                                 While many elevation sources are distributed in units of arc seconds, the keyword is an approximation
                                 of those resolutions in meters for easier understanding.

                                 Choice list: ['FINEST', '10m', '24m', '30m', '90m']

                                 The default is the finest resolution available.
    -------------------------    ---------------------------------------------------------
    maximum_distance             Optional float. This is a cutoff distance where the computation of visible areas stops. Beyond this distance, it is unknown whether the analysis points and the other objects can see each other.

                                 It is useful for modeling current weather conditions or a given time of day, such as dusk. Large values increase computation time.

                                 Unless specified, a default maximum distance will be computed based on the resolution and extent of the source DEM. The allowed maximum value is 50 kilometers.
                                 Use max_distance_units to set the units for maximum_distance.
    -------------------------    ---------------------------------------------------------
    max_distance_units           Optional string. The units for the maximum_distance parameter.

                                 Choice list: ['Meters', 'Kilometers', 'Feet', 'Miles', 'Yards']

                                 The default is 'Meters'.
    -------------------------    ---------------------------------------------------------
    observer_height              Optional float. This is the height above the ground of the observer locations.

                                 The default is 1.75 meters, which is approximately the average height of a person. If you are looking from an elevated location, such as an observation tower or a tall building, use that height instead.

                                 Use observer_height_units to set the units for observer_height.

    -------------------------    ---------------------------------------------------------
    observer_height_units        Optional string. The units for the observer_height parameter.

                                 Choice list: ['Meters', 'Kilometers', 'Feet', 'Miles', 'Yards']

                                 The default is 'Meters'.
    -------------------------    ---------------------------------------------------------
    target_height                Optional float. This is the height of structures or people on the ground used to
                                 establish visibility. The result viewshed are those areas where an input point can see these other objects.
                                 The converse is also true; the other objects can see an input point.

                                 * If your input points represent wind turbines and you want to determine where people standing on the
                                   ground can see the turbines, enter the average height of a person (approximately 6 feet).
                                   The result is those areas where a person standing on the ground can see the wind turbines.
                                 * If your input points represent fire lookout towers and you want to determine which lookout
                                   towers can see a smoke plume 20 feet high or higher, enter 20 feet for the height. The result
                                   is those areas where a fire lookout tower can see a smoke plume at least 20 feet high.
                                 * If your input points represent scenic overlooks along roads and trails and you want to determine
                                   where wind turbines 400 feet high or higher can be seen, enter 400 feet for the height. The result
                                   is those areas where a person standing at a scenic overlook can see a wind turbine at least 400 feet high.
                                 * If your input points represent scenic overlooks and you want to determine how much area on the ground
                                   people standing at the overlook can see, enter zero. The result is those areas that can be seen from the scenic overlook.

                                 Use target_height_units to set the units for target_height.

    -------------------------    ---------------------------------------------------------
    target_height_units          Optional string. The units for the target_height parameter.

                                 Choice list: ['Meters', 'Kilometers', 'Feet', 'Miles', 'Yards']

                                 The default is 'Meters'.
    -------------------------    ---------------------------------------------------------
    generalize                   Optional boolean. Determines whether or not the viewshed polygons are to be generalized.

                                 The viewshed calculation is based on a raster elevation model that creates a result with stair-stepped edges.
                                 To create a more pleasing appearance and improve performance, the default behavior is to generalize the polygons.
                                 The generalization process smooths the boundary of the visible areas and may remove some single-cell visible areas.

                                 The default value is True.
    -------------------------    ---------------------------------------------------------
    output_name                  Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                 feature layer will cause the new layer to be appended to the Feature Service.
                                 If overwrite is True in context, new layer will overwrite existing layer.
                                 If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------    ---------------------------------------------------------
    context                      Optional dict. Additional settings such as processing extent and output spatial reference.
                                 For create_viewshed, there are three settings.

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
    -------------------------    ---------------------------------------------------------
    gis                          Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    -------------------------    ---------------------------------------------------------
    estimate                     Optional boolean. If True, the estimated number of credits required to run the operation will be returned.
    -------------------------    ---------------------------------------------------------
    future                       Optional boolean. If True, a future object will be returned and the process
                                 will not wait for the task to complete. The default is False, which means wait for results.
    =========================    =========================================================

    :return:
        result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection`.

    .. code-block:: python

        USAGE EXAMPLE: To create viewshed around esri headquarter office.

        viewshed3 = create_viewshed(hq_lyr,
                            maximum_distance=9,
                            max_distance_units='Miles',
                            target_height=6,
                            target_height_units='Feet',
                            output_name="create Viewshed")

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    if gis is None:
        raise TypeError(
            "Please make sure you are logged into an instance of ArcGIS Online or ArcGIS Enterprise"
        )
    kwargs = {
        "input_layer": input_layer,
        "dem_resolution": dem_resolution,
        "maximum_distance": maximum_distance,
        "max_distance_units": max_distance_units,
        "observer_height": observer_height,
        "observer_height_units": observer_height_units,
        "target_height": target_height,
        "target_height_units": target_height_units,
        "generalize": generalize,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis.create_viewshed, **kwargs
    )
    return gis._tools.featureanalysis.create_viewshed(**params)


# --------------------------------------------------------------------------
def create_watersheds(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    search_distance: Optional[float] = None,
    search_units: str = "Meters",
    source_database: str = "FINEST",
    generalize: bool = True,
    output_name: Optional[Union[FeatureLayer, str]] = None,
    context: Optional[dict] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/create_watersheds/create_watersheds.png

    The ``create_watersheds`` method determines the watershed, or upstream contributing area, for each point
    in your analysis layer. For example, suppose you have point features representing locations
    of waterborne contamination, and you want to find the likely sources of the contamination.
    Since the source of the contamination must be somewhere within the watershed upstream of the
    point, you would use this tool to define the watersheds containing the sources of the contaminant.


    =========================    =========================================================
    **Parameter**                 **Description**
    -------------------------    ---------------------------------------------------------
    input_layer                  Required point feature layer. The point features used for calculating watersheds.
                                 These are referred to as pour points, because it is the location at which water pours out of the watershed.
                                 See :ref:`Feature Input<FeatureInput>`.
    -------------------------    ---------------------------------------------------------
    search_distance              Optional float. The maximum distance to move the location of an input point.
                                 Use search_units to set the units for search_distance.

                                 If your input points are located away from a drainage line, the resulting watersheds
                                 are likely to be very small and not of much use in determining the upstream source of
                                 contamination. In most cases, you want your input points to snap to the nearest drainage
                                 line in order to find the watersheds that flows to a point located on the drainage line.
                                 To find the closest drainage line, specify a search distance. If you do not specify a
                                 search distance, the tool will compute and use a conservative search distance.

                                 To use the exact location of your input point, specify a search distance of zero.

                                 For analysis purposes, drainage lines have been precomputed by Esri using standard
                                 hydrologic models. If there is no drainage line within the search distance, the location
                                 containing the highest flow accumulation within the search distance is used.
    -------------------------    ---------------------------------------------------------
    search_units                 Optional string. The linear units specified for the search distance.

                                 Choice list: ['Meters', 'Kilometers', 'Feet', 'Miles', 'Yards']
    -------------------------    ---------------------------------------------------------
    source_database              Optional string. Keyword indicating the data source resolution that will be used in the analysis.

                                 Choice list: ['Finest', '30m', '90m']

                                 * Finest (Default): Finest resolution available at each location from all possible data sources.
                                 * 30m: The hydrologic source was built from 1 arc second - approximately 30 meter resolution, elevation data.
                                 * 90m: The hydrologic source was built from 3 arc second - approximately 90 meter resolution, elevation data.
    -------------------------    ---------------------------------------------------------
    generalize                   Optional boolean. Determines if the output watersheds will be smoothed into simpler shapes or conform
                                 to the cell edges of the original DEM.

                                 * True: The polygons will be smoothed into simpler shapes. This is the default.
                                 * False: The edge of the polygons will conform to the edges of the original DEM.

                                 The default value is True.
    -------------------------    ---------------------------------------------------------
    output_name                  Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                 feature layer will cause the new layer to be appended to the Feature Service.
                                 If overwrite is True in context, new layer will overwrite existing layer.
                                 If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------    ---------------------------------------------------------
    context                      Optional dict. Additional settings such as processing extent and output spatial reference.
                                 For create_watersheds, there are three settings.

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
    -------------------------    ---------------------------------------------------------
    gis                          Optional, the :class:`~arcgis.gis.GIS`  on which this tool runs. If not specified, the active GIS is used.
    -------------------------    ---------------------------------------------------------
    estimate                     Optional boolean. If True, the estimated number of credits required to run the operation will be returned.
    -------------------------    ---------------------------------------------------------
    future                       Optional boolean. If True, a future object will be returned and the process
                                 will not wait for the task to complete. The default is False, which means wait for results.
    =========================    =========================================================

    :returns result_layer : :class:`~arcgis.features.FeatureLayer` if output_name is specified, else :class:`~arcgis.features.FeatureCollection`.

    .. code-block:: python

        USAGE EXAMPLE: To create watersheds for Chennai lakes.

        lakes_watershed = create_watersheds(lakes_lyr,
                                            search_distance=3,
                                            search_units='Kilometers',
                                            source_database='90m',
                                            output_name='create watersheds')

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    if gis is None:
        raise TypeError(
            "Please make sure you are logged into an instance of ArcGIS Online or ArcGIS Enterprise"
        )
    kwargs = {
        "input_layer": input_layer,
        "search_distance": search_distance,
        "search_units": search_units,
        "source_database": source_database,
        "generalize": generalize,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.create_watersheds, **kwargs
    )

    return gis._tools.featureanalysis.create_watersheds(**params)


# --------------------------------------------------------------------------
def trace_downstream(
    input_layer: Union[
        Item,
        FeatureCollection,
        FeatureLayer,
        FeatureLayerCollection,
        str,
        dict[str, Any],
    ],
    split_distance: Optional[float] = None,
    split_units: str = "Kilometers",
    max_distance: Optional[float] = None,
    max_distance_units: str = "Kilometers",
    bounding_polygon_layer: Optional[
        Union[
            Item,
            FeatureCollection,
            FeatureLayer,
            FeatureLayerCollection,
            str,
            dict[str, Any],
        ]
    ] = None,
    source_database: Optional[str] = None,
    generalize: bool = True,
    output_name: Optional[Union[FeatureLayer, str]] = None,
    context: Optional[dict] = None,
    gis: Optional[GIS] = None,
    estimate: bool = False,
    future: bool = False,
):
    """
    .. image:: _static/images/trace_downstream/trace_downstream.png

    The ``trace_downstream`` method determines the trace, or flow path, in a downstream direction from the points in your analysis layer.

    For example, suppose you have point features representing sources of contamination and you want to determine where in your study
    area the contamination will flow. You can use ``trace_downstream`` to identify the path the contamination will take. This trace
    can also be divided into individual line segments by specifying a distance value and units. The line being returned can be the
    total length of the flow path, a specified maximum trace length, or clipped to area features such as your study area. In many
    cases, if the total length of the trace path is returned, it will be from the source all the way to the ocean.

    =====================================   =========================================================
    **Parameter**                            **Description**
    -------------------------------------   ---------------------------------------------------------
    input_layer                             Required feature layer. The point features used for the starting location of a downstream trace.
                                            See :ref:`Feature Input<FeatureInput>`.
    -------------------------------------   ---------------------------------------------------------
    split_distance                          Optional float. The trace line will be split into multiple lines where each line is of the specified length.
                                            The resulting trace will have multiple line segments, each with fields FromDistance and ToDistance.
    -------------------------------------   ---------------------------------------------------------
    split_units                             Optional string. The units used to specify split distance.

                                            Choice list: ['Meters', 'Kilometers', 'Feet' 'Yards', 'Miles'].

                                            The default is 'Kilometers'.
    -------------------------------------   ---------------------------------------------------------
    max_distance                            Optional float. Determines the total length of the line that will be returned. If you provide a
                                            ``bounding_polygon_layer`` to clip the trace, the result will be clipped to the features in ``bounding_polygon_layer``,
                                            regardless of the distance you enter here.
    -------------------------------------   ---------------------------------------------------------
    max_distance_units                      Optional string. The units used to specify maximum distance.

                                            Choice list: ['Meters', 'Kilometers', 'Feet' 'Yards', 'Miles'].

                                            The default is 'Kilometers'.
    -------------------------------------   ---------------------------------------------------------
    bounding_polygon_layer                  Optional feature layer. A polygon layer specifying the area(s) where you want the trace
                                            downstreams to be calculated in. For example, if you only want to calculate the trace downstream
                                            with in a county polygon, provide a layer containing the county polygon and the resulting trace
                                            lines will be clipped to the county boundary. See :ref:`Feature Input<FeatureInput>`.
    -------------------------------------   ---------------------------------------------------------
    source_database                         Optional string. Keyword indicating the data source resolution that will be used in the analysis.

                                            Choice list: ['Finest', '30m', '90m'].

                                            * Finest: Finest resolution available at each location from all possible data sources.

                                            * 30m: The hydrologic source was built from 1 arc second - approximately 30 meter resolution, elevation data.

                                            * 90m: The hydrologic source was built from 3 arc second - approximately 90 meter resolution, elevation data.

                                            The default is 'Finest'.
    -------------------------------------   ---------------------------------------------------------
    generalize                              Optional boolean. Determines if the output trace downstream lines will be smoothed
                                            into simpler lines or conform to the cell edges of the original DEM.
    -------------------------------------   ---------------------------------------------------------
    output_name                             Optional string or :class:`~arcgis.features.FeatureLayer`. Existing
                                            feature layer will cause the new layer to be appended to the Feature Service.
                                            If overwrite is True in context, new layer will overwrite existing layer.
                                            If output_name not indicated then new :class:`~arcgis.features.FeatureCollection` created.
    -------------------------------------   ---------------------------------------------------------
    context                                 Optional dict. Additional settings such as processing extent and output spatial reference.
                                            For trace_downstream, there are three settings.

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
    =====================================   =========================================================

    :return: :class:`~arcgis.features.FeatureLayer` if ``output_name`` is set, else :class:`~arcgis.features.FeatureCollection`.

    .. code-block:: python

        # USAGE EXAMPLE: To identify the path the water contamination  will take.
        path = trace_downstream(input_layer=water_source_lyr,
                                split_distance=2,
                                split_units='Miles',
                                max_distance=2,
                                max_distance_units='Miles',
                                source_database='Finest',
                                generalize=True,
                                output_name='trace downstream')
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    if gis is None:
        raise TypeError(
            "Please make sure you are logged into an instance of ArcGIS Online or ArcGIS Enterprise"
        )
    kwargs = {
        "input_layer": input_layer,
        "split_distance": split_distance,
        "split_units": split_units,
        "max_distance": max_distance,
        "max_distance_units": max_distance_units,
        "bounding_polygon_layer": bounding_polygon_layer,
        "source_database": source_database,
        "generalize": generalize,
        "output_name": output_name,
        "context": context,
        "gis": gis,
        "estimate": estimate,
        "future": future,
    }
    params = inspect_function_inputs(
        fn=gis._tools.featureanalysis._tbx.trace_downstream, **kwargs
    )

    return gis._tools.featureanalysis.trace_downstream(**params)
