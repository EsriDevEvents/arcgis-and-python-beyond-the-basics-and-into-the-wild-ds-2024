"""
The ``arcgis.raster`` module contains classes and analysis functions for working with raster data and
imagery layers.

Raster data is made up of a grid of cells, where each cell or pixel can have a value. It is useful
for storing data that varies continuously, as in a satellite image, a surface of chemical concentrations, or
an elevation surface.

Use :meth:`~arcgis.raster.analytics.is_supported` to check if raster analysis is supported in your GIS.
"""

from ._layer import (
    ImageryLayer,
    ImageryLayerCacheManager,
    RasterManager,
    ImageryTileManager,
    RasterCatalogItem,
    Raster,
    _ImageServerRaster,
    _ArcpyRaster,
    RasterCollection,
)
from .analytics import *
from .functions import *
from .orthomapping import *
from .utils import *

# from .charts import *
from ._RasterInfo import RasterInfo
