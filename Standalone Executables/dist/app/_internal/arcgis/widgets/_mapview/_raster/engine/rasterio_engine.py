import logging

log = logging.getLogger()

from arcgis.widgets._mapview._raster.engine import RasterData, MatplotlibRasterEngine


class RasterIOEngine(MatplotlibRasterEngine):
    def try_get_raster_data(self, image, bands) -> RasterData:
        try:
            return self._try_get_raster_data(image, bands)
        except Exception as e:
            return super().try_get_raster_data(image, bands)

    def _try_get_raster_data(self, image, bands) -> RasterData:
        import rasterio

        with rasterio.open(image) as dataset:
            extent = {
                "xmin": dataset.bounds.left,
                "ymin": dataset.bounds.bottom,
                "xmax": dataset.bounds.right,
                "ymax": dataset.bounds.top,
                "spatialReference": {},
            }
            if dataset.crs and dataset.crs.to_epsg():
                extent["spatialReference"]["wkid"] = dataset.crs.to_epsg()
            elif dataset.crs and dataset.crs.to_wkt():
                extent["spatialReference"]["wkt"] = dataset.crs.to_wkt()
            else:
                extent = {}
            img_data = dataset.read()
            img_data = img_data.transpose((1, 2, 0)).squeeze()
            if bands:
                img_data = img_data[:, :, bands].squeeze()
            return RasterData(img_data=img_data, extent=extent, id_=str(hash(image)))
