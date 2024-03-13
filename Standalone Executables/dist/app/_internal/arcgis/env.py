"""
The **env** module provides a shared environment used by the different modules.
It stores globals such as the currently active :class:`~arcgis.gis.GIS`, the default geocoder and so on.
It also stores environment settings that are common among all geoprocessing tools,
such as the output spatial reference.

active_gis
==========

.. py:data:: active_gis
The currently active :class:`~arcgis.gis.GIS`, that is used for analysis functions unless explicitly specified
when calling the functions.
Creating a new :class:`~arcgis.gis.GIS` object makes it active unless set_active=False is passed in the :class:`~arcgis.gis.GIS` constructor.


analysis_extent
===============

.. py:data:: analysis_extent
The processing extent used by analysis tools, specified as an :class:`~arcgis.geometry.Envelope` .

out_spatial_reference
=====================

.. py:data:: out_spatial_reference
The spatial reference of the output geometries. If not specified, the output geometries are in the
spatial reference of the input geometries. If process_spatial_reference is specified and out_spatial_reference
is not specified, the output geometries are in the spatial reference of the process spatial reference.

process_spatial_reference
=========================

.. py:data:: process_spatial_reference
The spatial reference that the geoprocessor will use to perform geometry operations. If specified and
out_spatial_reference is not specified, the output geometries are in the spatial reference of the
process spatial reference.

output_datastore
================

.. py:data:: output_datastore
The data store where GeoAnalytics results should be stored. The supported values of this parameter are ``relational`` and
``spatiotemporal``. By default, results are stored in the spatiotemporal data store. It is recommended that results are
stored in the spatiotemporal data store due to the scalability of the spatiotemporal big data store.

return_z
========

.. py:data:: return_z
If ``True``, Z values will be included in the geoprocessing results if the features have Z values.
Otherwise Z values are not returned. The default is ``False``.


return_m
========

.. py:data:: return_m
If ``True``, M values will be included in the results if the features have M values.
Otherwise M values are not returned. The default is ``False``.

verbose
========

.. py:data:: verbose

If ``True``, messages from geoprocessing tools will be printed to stdout.
In any case, all geoprocessing messages are available through Python logging module. The default is ``False``.

default_aggregation_styles
==========================

.. py:data:: default_aggregation_styles

Tasks that have the default_aggregation_styles property set to ``True`` will set the default aggregations for the
resulting layer. Default aggregations can be square, pointy triangle, flat triangle, pointy hexagon, flat
hexagon, and geohash. All aggregation styles are supported using WKID 4326 (WGS_1984).
The default_aggregation_styles is ``False``. (supported at 10.6.1+)

snap_raster
===========

.. py:data:: snap_raster

Tasks that honor the snap_raster environment will adjust the extent of output rasters
so that they match the cell alignment of the specified snap raster.
(For more information about this environment setting,
please refer to `Snap Raster (Environment setting) <http://pro.arcgis.com/en/pro-app/tool-reference/environment-settings/snap-raster.htm>`_)

cell_size
=========

.. py:data:: cell_size

Tasks that honor the cell_size environment setting set the output raster cell size, or resolution,
for the operation.  The default output resolution is determined by the largest cell size of
all the input rasters.
(For more information about this environment setting,
please refer to `Cell Size (Environment setting) <http://pro.arcgis.com/en/pro-app/tool-reference/environment-settings/cell-size.htm>`_)

mask
====

.. py:data:: mask

Tasks that honor the mask environment will only consider those cells that fall within the analysis
mask in the operation
(For more information about this environment setting,
please refer to `Mask (Environment setting) <http://pro.arcgis.com/en/pro-app/tool-reference/environment-settings/mask.htm>`_)

parallel_processing_factor
==========================

.. py:data:: parallel_processing_factor

Tasks that honor the parallel_processing_factor environment will divide and perform operations across
multiple processes.
(For more information about this environment setting,
please refer to `Parallel Processing Factor (Environment setting) <http://pro.arcgis.com/en/pro-app/tool-reference/environment-settings/parallel-processing-factor.htm>`_)

union_dimension
===============

.. py:data:: union_dimension

Tasks (raster functions present in :mod:`~arcgis.raster.functions` module) that honor the union_dimension environment
will generate a multidimensional raster that includes all the dimensions from the input multidimensional rasters.
(For more information about this environment setting,
please refer to `Union Dimension (Environment setting) <https://pro.arcgis.com/en/pro-app/latest/tool-reference/environment-settings/union-dimension.htm>`_)

match_variables
===============

.. py:data:: match_variables

Tasks (raster functions present in :mod:`~arcgis.raster.functions` module) that honor the match_variables environment
will generate a multidimensional raster only if the input multidimensional rasters share at least one variable
with the same name.
(For more information about this environment setting,
please refer to `Match Multidimensional Variable (Environment setting) <https://pro.arcgis.com/en/pro-app/latest/tool-reference/environment-settings/match-multidimensional-variable.htm>`_)
"""

#: The currently active GIS, that is used for analysis functions unless explicitly specified.
#: Creating a new GIS object makes it active by default unless set_active=False is passed in the GIS constructor.
active_gis = None

#: The spatial reference of the output geometries. If not specified, the output geometries are in the
#: spatial reference of the input geometries. If process_spatial_reference is specified and out_spatial_reference
#: is not specified, the output geometries are in the spatial reference of the process spatial reference.
out_spatial_reference = None

#: The spatial reference that analysis and geoprocessing tools will use to perform geometry operations. If specified and
#: out_spatial_reference is not specified, the output geometries are in the spatial reference of the
#: process spatial reference.
process_spatial_reference = None

#: The data store where GeoAnalytics results should be stored. The supported values of this parameter are "relational" and
#: "spatiotemporal". By default, results are stored in the spatiotemporal data store. It is recommended that results be
#: stored in the spatiotemporal data store due to the scalability of the spatiotemporal big data store.
output_datastore = None

#: The processing extent used by analysis tools
analysis_extent = None

#: If True, Z values will be included in the geoprocessing results if the features have Z values.
#: Otherwise Z values are not returned. The default is False.
return_z = False

#: If True, M values will be included in the results if the features have M values.
#: Otherwise M values are not returned. The default is False.
return_m = False

#: If True, messages from geoprocessing tools will be printed to stdout
verbose = False

#: Tasks that have the default_aggregation_styles property set to true will set the default aggregations for the
#: resulting layer. Default aggregations can be square, pointy triangle, flat triangle, pointy hexagon, flat
#: hexagon, and geohash. All aggregation styles are supported using WKID 4326 (WGS_1984).
#: The default_aggregation_styles is False. (supported at 10.6.1+)
default_aggregation_styles = False

#: Tasks that honor the snap_raster environment will adjust the extent of output rasters
#: so that they match the cell alignment of the specified snap raster.
snap_raster = None

#: Tasks that honor the cell_size environment setting set the output raster cell size, or resolution, for the operation.
#: The default output resolution is determined by the largest cell size of all the input rasters
cell_size = None

#: Tasks that honor the mask environment will only consider those cells that fall within the analysis mask in the operation
mask = None

# Tasks that honor the parallel_processing_factor environment will divide and perform operations across multiple processes.
parallel_processing_factor = None

# Raster functions in arcgis.raster.functions module that honor the Union Dimension environment will generate a multidimensional raster that includes all the
# dimensions from the input multidimensional rasters.
union_dimension = None

# Raster functions in arcgis.raster.functions module that honor the Match Multidimensional Variable environment will generate a multidimensional raster only
# if the input multidimensional rasters share at least one variable with the same name.
match_variables = None
