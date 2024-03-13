import os
from uuid import uuid4
import json
import traceback
import tempfile
import logging

log = logging.getLogger()

from arcgis.widgets._mapview._raster.engine import RasterData, MatplotlibRasterEngine


class ArcPyRasterEngine(MatplotlibRasterEngine):
    def try_get_raster_data(self, image, bands) -> RasterData:
        try:
            return self._try_get_raster_data(image, bands)
        except Exception as e:
            return super().try_get_raster_data(image, bands)

    def _try_get_raster_data(self, image, bands) -> RasterData:
        from arcpy import Raster, RasterToNumPyArray

        raster = Raster(image)
        img_data = RasterToNumPyArray(raster)
        if len(img_data.shape) == 3:
            img_data = img_data.transpose((1, 2, 0)).squeeze()
            if bands:
                img_data = img_data[:, :, bands].squeeze()
        extent = self._get_extent(raster)
        return RasterData(
            img_data=img_data,
            extent=extent,
            id_=str(hash(raster.extent.JSON + raster.name)),
        )

    def _get_extent(self, raster):
        import arcpy
        from arcpy import Raster, RasterToNumPyArray, SpatialReference

        try:
            transformations = arcpy.ListTransformations(
                raster.spatialReference, arcpy.SpatialReference(4326)
            )
            if len(transformations) > 0:
                transformation = transformations[0]
            else:
                transformation = None
            extent = json.loads(
                raster.extent.projectAs(SpatialReference(4326), transformation).JSON
            )
            if extent["spatialReference"]["wkid"] is not None:
                return extent
            else:
                raise Exception(f"{extent} contains invalid spatial reference")
        except Exception as e:
            log.debug("Handled exception on inferring extent w/ arcpy engine")
            log.debug(e)
            return None
