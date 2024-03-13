import os
import time
import sys

from arcgis.widgets._mapview._raster import *
from arcgis.widgets._mapview._raster._numpy_utils import *
from arcgis.widgets._mapview._raster._jupyter_utils import *


class RasterOverlay:
    def __init__(self, id_: str, img_url: str, extent: dict, opacity: float):
        self.id = id_
        self.img_url = img_url
        self.extent = extent
        self.opacity = opacity

    def as_dict(self) -> dict:
        return {"href": self.img_url, "extent": self.extent, "opacity": self.opacity}


class RasterData:
    def __init__(self, img_data, extent: dict, id_: str):
        self.img_data = img_data
        self.extent = extent
        self.id = id_


class LocalRasterOverlayManager:
    """
    .. warning::
        Overlying local rasters on a ``MapView`` instance have the following
        limitations:

        - Local raster overlays do not persist beyond the notebook session on
          published web maps/web scenes -- you would need to seperately publish
          these local rasters.

        - The entire raster image data is placed on the MapView's canvas with
          no performance optimizations. This means no pyramids, no dynamic
          downsampling, etc. Please be mindful of the size of the local raster
          and your computer's hardware limitations.

        - Pixel values and projections are not guaranteed to be accurate,
          especially when the local raster's Spatial Reference doesn't
          reproject accurately to Web Mercator (what the ``MapView``
          widget uses).
    """

    def __init__(self, mapview):
        self._mapview = mapview
        self._attempt_infer_file_format()

    _overlays = []

    _file_format = "jpg"

    @property
    def file_format(self) -> str:
        return self._file_format

    @file_format.setter
    def file_format(self, value):
        self._file_format = value

    def overlay(self, raster):
        self._clear_old_image_overlays()
        raster_data = self.try_get_raster_data(raster)
        img_path = self._save_img_to_jupyter_accessible_dir(
            raster_data.img_data,
            raster.cmap,
            vmin=raster.vmin,
            vmax=raster.vmax,
            filename=str(raster_data.id) + "." + self.file_format,
        )

        extent = self._make_extent_valid(raster.extent)

        img_url = self._get_jupyter_accessible_url(img_path)

        raster_overlay = RasterOverlay(raster_data.id, img_url, extent, raster.opacity)
        self._mapview._add_overlay(raster_overlay)
        self._overlays.append(
            {
                "id": raster_data.id,
                "image": raster,
                "img_path": img_path,
                "img_url": img_url,
                "extent": extent,
            }
        )

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

    def try_get_raster_data(self, image) -> RasterData:
        img_data = image.read().squeeze()
        return RasterData(
            img_data=img_data, extent=image.extent, id_=get_hash_numpy_array(img_data)
        )

    def remove(self, raster=None):
        image = raster
        any_removed = False
        for i in reversed(range(0, len(self._overlays))):
            overlay = self._overlays[i]
            if image is None or self._are_equal(image, overlay["image"]):
                del self._overlays[i]
                self._mapview._remove_overlay(overlay["id"])
                any_removed = True
        if any_removed:
            return True
        else:
            raise Exception(f"Could not find {image} to remove")

    def _are_equal(self, image1, image2):
        if is_numpy_array(image1):
            return get_hash_numpy_array(image1) == get_hash_numpy_array(image2)
        else:
            return image1 == image2

    def list(self):
        """
        Returns a list of rasters overlayed. Will also visually display these
        rasters in a Jupyter Notebook environment.
        """
        from IPython.display import display

        scroll_list = JupyterHorizScrollImageList()
        for overlay in self._overlays:
            scroll_list.append(src=overlay["img_url"], title=str(overlay["extent"]))
        display(scroll_list)

        return list(overlay["image"] for overlay in self._overlays)

    def _make_extent_valid(self, extent):
        if not extent:
            raise RuntimeError(
                f"Could not infer extent of raster! Please manually "
                f"specify an extent"
            )
        return extent

    def _attempt_infer_file_format(self):
        try:
            from PIL import Image

            self.file_format = "jpg"
        except ImportError:
            self.file_format = "png"

    _jupyter_notebook_dir_override = ""

    def set_current_executing_nb_dir(self, path: str):
        """To display in a notebook, rasters must be placed in the
        ``_image_overlays`` folder in the same directory as the current
        executing notebook. This value is normally inferred. To override,
        call this function with the path to the current executing notebook.
        """
        self._jupyter_notebook_dir_override = path

    _DEFAULT_CMAP = "Greys_r"  # Mirrors the style of default ArcGIS Pro

    def _save_img_to_jupyter_accessible_dir(self, img_data, cmap, vmin, vmax, filename):
        import matplotlib.pyplot as plt

        img_path = os.path.join(self._get_image_overlays_dir(), filename)
        num_bands = self._get_num_bands(img_data)
        kwargs = {}
        if vmin:
            kwargs["vmin"] = vmin
        if vmax:
            kwargs["vmax"] = vmax
        if not cmap:
            if num_bands != 1 and num_bands != 3 and num_bands != 4:
                raise Exception(
                    f"This raster has {num_bands} bands -- Number "
                    f"of bands must be 1 (greyscale), 3 (RGB), or 4 (RGBA). "
                    f"Consider choosing a subset of bands, or make sure the "
                    f"shape of the data is valid. (Shape: {img_data.shape})."
                )
            if num_bands == 1:
                kwargs["cmap"] = self._DEFAULT_CMAP
        else:
            if num_bands != 1:
                raise Exception(
                    f"To use a cmap, the input raster "
                    f"must have only 1 band, not {num_bands}"
                )
            else:
                kwargs["cmap"] = cmap

        plt.imsave(img_path, img_data, **kwargs)

        return img_path

    def _get_num_bands(self, img_data):
        num_bands = -1
        if len(img_data.shape) == 2:
            num_bands = 1
        elif len(img_data.shape) == 3:
            num_bands = img_data.shape[2]
        return num_bands

    _image_overlays_dir_name = "_image_overlays"

    def _get_image_overlays_dir(self):
        if self._jupyter_notebook_dir_override:
            curr_jupyter_notebook_dir = self._jupyter_notebook_dir_override
        else:
            from arcgis.widgets._mapview._raster._jupyter_utils import (
                get_dir_of_curr_exec_notebook,
            )

            curr_jupyter_notebook_dir = get_dir_of_curr_exec_notebook()

        image_overlays_dir = os.path.join(
            curr_jupyter_notebook_dir, self._image_overlays_dir_name
        )
        if not os.path.isdir(image_overlays_dir):
            os.mkdir(image_overlays_dir)
        return image_overlays_dir

    def _clear_old_image_overlays(self):
        """Clears all files older than 2 days in `_image_overlays` dir"""
        dir_ = self._get_image_overlays_dir()
        now = time.time()
        for file_name in os.listdir(dir_):
            file_path = os.path.join(dir_, file_name)
            if os.stat(file_path).st_mtime < now - 2 * 86400:
                # If older than 2 days
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    def _get_jupyter_accessible_url(self, image_path):
        filename = os.path.basename(image_path)
        image_url = os.path.join(self._image_overlays_dir_name, filename)
        return image_url
