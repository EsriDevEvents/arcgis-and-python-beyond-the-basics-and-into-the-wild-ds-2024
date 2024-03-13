import os

from arcgis.widgets._mapview._raster._numpy_utils import *

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from arcgis.widgets._mapview._raster.engine import RasterData


class MatplotlibRasterEngine:
    def _is_numpy_array(self, image):
        """numpy arrays behave unpredictably in `isinstance()` func calls:
        do a hacky string comparison on the type() of image arg
        """
        return "numpy" in str(type(image)) and "array" in str(type(image))

    def _get_uuid_from_numpy_array(self, image):
        return str(hash(str(image.tostring())))

    _SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".tif"]

    def _is_supported_image_file(self, image):
        if not isinstance(image, str):
            return False
        _, ext = os.path.splitext(image)
        return os.path.exists(image) and ext in self._SUPPORTED_IMAGE_FORMATS

    def _is_arcgis_raster(self, image):
        try:
            from arcgis.raster import Raster

            return isinstance(image, Raster)
        except ImportError:
            return False

    def try_get_raster_data(self, image, bands) -> RasterData:
        if is_numpy_array(image):
            return RasterData(
                img_data=image, extent=None, id_=get_hash_numpy_array(image)
            )
        elif self._is_arcgis_raster(image):
            img_data = image.read().squeeze()
            return RasterData(
                img_data=img_data,
                extent=image.extent,
                id_=get_hash_numpy_array(img_data),
            )
        elif self._is_supported_image_file(image):
            return RasterData(
                img_data=plt.imread(image), extent=None, id_=str(hash(image))
            )
        elif isinstance(image, str) and not os.path.exists(image):
            raise Exception(f"File {image} does not exist")
        else:
            raise Exception(f"Argument {image} not supported")
