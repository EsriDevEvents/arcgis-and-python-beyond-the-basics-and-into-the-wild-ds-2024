"""
The arcgis.widgets module provides components for visualizing GIS data and analysis.
This module includes the MapView Jupyter notebook widget for visualizing maps and layers
"""
import json
import time
import logging
import base64
from typing import Union
import urllib.request
from uuid import uuid4
from collections import OrderedDict
import os
import datetime as dt
import dateutil.parser
import tempfile

from arcgis.geometry import Point, Polygon, Polyline, MultiPoint, Geometry
from arcgis.features import (
    FeatureSet,
    Feature,
    FeatureCollection,
    FeatureLayer,
)
from arcgis.raster import (
    ImageryLayer,
    Raster,
    _ImageServerRaster,
    _ArcpyRaster,
)
from arcgis.gis import Layer
from arcgis.gis import Item

import ipywidgets
from ipywidgets import widgets
from ipywidgets.embed import embed_minimal_html
from traitlets import Unicode, List, Bool, Dict, Tuple, Float, observe

Datetime = ipywidgets.trait_types.Datetime
from IPython.display import display, HTML

from arcgis.widgets._mapview._webscene_utils import (
    DEFAULT_WEBSCENE_TEXT_PROPERTY,
)
from arcgis.widgets._mapview._loading_icon_str import _loading_icon_str
from arcgis.widgets._mapview._raster import LocalRasterOverlayManager
from arcgis.widgets._mapview._raster._numpy_utils import *
from arcgis import __version__ as py_api_version
from arcgis.auth._auth._schain import _MultiAuth, SupportMultiAuth
import arcgis.mapping
import arcgis

try:
    import pandas as pd
    from arcgis.features.geo import _is_geoenabled
except ImportError:

    def _is_geoenabled(**kwargs):
        return False

    pd = None


log = logging.getLogger(__name__)

DEFAULT_ELEMENT_HEIGHT = "400px"

_DEFAULT_JS_CDN = "https://js.arcgis.com/4.15/"
_js_cdn_override_global = ""


def _is_iterable(obj):
    """Returns true for all iterables but strings"""
    try:
        _ = iter(obj)
        return not isinstance(obj, str)
    except TypeError:
        return False


def _make_jsonable_dict(obj):
    """Attempts to make any object a jsonable dict. Will exclude any
    value that is non-serializable"""
    flag = "K9SS8NRqLyiXF6pXDELETEME"

    def default_func(x):
        """If you come across anything non-jsonable, return a flag that
        will later be used to delete all values with this value"""
        return flag

    dict_ = json.loads(json.dumps(obj, default=default_func))
    return {k: v for k, v in dict_.items() if v != flag}


def _flatten_list(*unpacked_list):
    return_list = []
    for x in unpacked_list:
        if isinstance(x, (list, tuple)):
            return_list.extend(_flatten_list(*x))
        else:
            return_list.append(x)
    return return_list


def _get_extent(item):
    if isinstance(item, Raster):
        if isinstance(item._engine_obj, _ImageServerRaster):
            item = item._engine_obj
        elif isinstance(item._engine_obj, _ArcpyRaster):
            return dict(item.extent)
    if isinstance(item, Item):
        return list(map(_get_extent, item.layers))
    elif isinstance(item, list):
        return list(map(_get_extent, item))
    elif isinstance(item, pd.DataFrame):
        return _get_extent_of_dataframe(item)
    elif isinstance(item, FeatureSet):
        return _get_extent(item.sdf)
    elif isinstance(item, FeatureCollection):
        return dict(item.properties.layerDefinition.extent)
    elif isinstance(item, Layer):
        try:
            if "extent" in item.properties:
                return dict(item.properties.extent)
            elif "fullExtent" in item.properties:
                return dict(item.properties["fullExtent"])
            elif "initialExtent" in item.properties:
                return dict(item.properties["initialExtent"])
        except:
            ext = item.extent
            return {
                "spatialReference": {"wkid": 4326, "latestWkid": 4326},
                "xmin": ext[0][1],
                "ymin": ext[0][0],
                "xmax": ext[1][1],
                "ymax": ext[1][0],
            }
    else:
        raise Exception("could not infer layer type")


def _get_extent_of_layers(list_of_layers):
    extents = []
    for layer in list_of_layers:
        extents.append(layer.properties["extent"])
    if len(extents) == 1:
        return extents[0]
    return _get_master_extent(extents)


def _get_extent_of_dataframe(sdf):
    if hasattr(sdf, "spatial"):
        sdf_ext = sdf.spatial.full_extent
        return {
            "spatialReference": sdf.spatial.sr,
            "xmin": sdf_ext[0],
            "ymin": sdf_ext[1],
            "xmax": sdf_ext[2],
            "ymax": sdf_ext[3],
        }
    else:
        raise Exception(
            "Could not add get extent of DataFrame it is not a spatially enabled DataFrame."
        )


default_sr = {"wkid": 102100, "latestWkid": 3857}


def _get_master_extent(list_of_extents, target_sr=None):
    if target_sr is None:
        target_sr = default_sr
    # Check if any extent is different from one another
    varying_spatial_reference = False
    for extent in list_of_extents:
        if not target_sr == extent["spatialReference"]:
            varying_spatial_reference = True
    if varying_spatial_reference:
        list_of_extents = _reproject_extent(list_of_extents, target_sr)

    # Calculate master_extent
    master_extent = list_of_extents[0]
    for extent in list_of_extents:
        master_extent["xmin"] = min(master_extent["xmin"], extent["xmin"])
        master_extent["ymin"] = min(master_extent["ymin"], extent["ymin"])
        master_extent["xmax"] = max(master_extent["xmax"], extent["xmax"])
        master_extent["ymax"] = max(master_extent["ymax"], extent["ymax"])
    return master_extent


def _reproject_extent(extents, target_sr={"wkid": 102100, "latestWkid": 3857}):
    """Reproject Extent

    ==================     ====================================================================
    **Parameter**              **Description**
    ------------------     --------------------------------------------------------------------
    extents                   extent or list of extents you want to project.
    ------------------     --------------------------------------------------------------------
    target_sr                 The target Spatial Reference you want to get your extent in.
                              default is {'wkid': 102100, 'latestWkid': 3857}
    ==================     ====================================================================

    """
    if not type(extents) == list:
        extents = [extents]

    extents_to_reproject = {}
    for i, extent in enumerate(extents):
        if not extent["spatialReference"] == target_sr:
            in_sr_str = str(extent["spatialReference"])
            if not in_sr_str in extents_to_reproject:
                extents_to_reproject[in_sr_str] = {}
                extents_to_reproject[in_sr_str]["spatialReference"] = extent[
                    "spatialReference"
                ]
                extents_to_reproject[in_sr_str]["extents"] = []
                extents_to_reproject[in_sr_str]["indexes"] = []
            extents_to_reproject[in_sr_str]["extents"].extend(
                [
                    {"x": extent["xmin"], "y": extent["ymin"]},
                    {"x": extent["xmax"], "y": extent["ymax"]},
                ]
            )
            extents_to_reproject[in_sr_str]["indexes"].append(i)

    for in_sr_str in extents_to_reproject:  # Reproject now
        reprojected_extents = arcgis.geometry.project(
            extents_to_reproject[in_sr_str]["extents"],
            in_sr=extents_to_reproject[in_sr_str]["spatialReference"],
            out_sr=target_sr,
        )
        for i in range(0, len(reprojected_extents), 2):
            source_idx = extents_to_reproject[in_sr_str]["indexes"][int(i / 2)]
            extents[source_idx] = {
                "xmin": reprojected_extents[i]["x"],
                "ymin": reprojected_extents[i]["y"],
                "xmax": reprojected_extents[i + 1]["x"],
                "ymax": reprojected_extents[i + 1]["y"],
                "spatialReference": target_sr,
            }

    if len(extents) == 1:
        return extents[0]
    return extents


@widgets.register
class MapView(widgets.DOMWidget):
    """
    The ``MapView`` class creates a mapping widget for Jupyter Notebook and JupyterLab.

    ==================     ====================================================================
    **Parameter**              **Description**
    ------------------     --------------------------------------------------------------------
    gis                    The active :class:`~arcgis.gis.GIS` instance you want this map widget to use.
    ------------------     --------------------------------------------------------------------
    item                   A ``WebMap`` or ``WebScene`` item instance that you want to visualize
    ------------------     --------------------------------------------------------------------
    mode                   Whether to construct a '2D' map or '3D' map. See the ``mode`` property
                           for more information.
    ==================     ====================================================================

    .. note::
        If the Jupyter Notebook server is running over http, you need to
        configure your portal/organization to allow your host and port; or else
        you will run into ``CORS`` issues.

        This can be accomplished programmatically by signing into your organization
        and running this code:

        .. code-block:: python

            >>> from arcgis.gis import GIS
            >>> gis = GIS(profile="your_admin_profile")

            >>> more_origins = {"allowedOrigins":"http://localhost:8888"} #replace your port

            >>> gis.update_properties(more_origins)

    """

    _view_name = Unicode("ArcGISMapIPyWidgetView").tag(sync=True)
    _model_name = Unicode("ArcGISMapIPyWidgetModel").tag(sync=True)
    _view_module = Unicode("arcgis-map-ipywidget").tag(sync=True)
    _model_module = Unicode("arcgis-map-ipywidget").tag(sync=True)
    _view_module_version = Unicode(str(py_api_version)).tag(sync=True)
    _model_module_version = Unicode(str(py_api_version)).tag(sync=True)

    # Start model specific state
    # Start map specific drawing state

    _zoom = Float(-1).tag(sync=True)
    _readonly_zoom = Float(-1).tag(sync=True)

    @property
    def zoom(self):
        """
        Get/Set the level of zoom applied to the Map Widget.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required int.
                            .. note::
                                The higher the number, the more zoomed in you are.
        ===============     ====================================================================

        :return:
            Int value that represent the zoom level

        .. code-block:: python

            # Usage example

            >>> from arcgis.gis import GIS, Item
            >>> from arcgis.widgets import MapView

            >>> map2 = gis.map("California")
            >>> map2.basemap = 'national-geographic'
            >>> map2
            >>> map2.zoom = 200
        """
        return self._readonly_zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = value

    _scale = Float(-1).tag(sync=True)
    _readonly_scale = Float(-1).tag(sync=True)

    @property
    def scale(self):
        """
        Get/Set the map scale at the center of the view. If set to X, the scale
        of the map would be 1:X.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required int.
        ===============     ====================================================================

        .. note::
            For continuous values to apply and not get "snapped" to the closest
            level of detail, set `mapview.snap_to_zoom = False`.

        .. code-block:: python

            # Usage example: Sets the scale to 1:24000

            map = gis.map()
            map.scale = 24000
        """
        return self._readonly_scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    _snap_to_zoom = Bool(True).tag(sync=True)

    @property
    def snap_to_zoom(self):
        """
        The ``snap_to_zoom`` property is used to determine how the zoom is enabled when the map widget is created.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required bool.
                            Values:
                                + True: snap to the next level of detail when zooming in or out.
                                + False: the zoom is continuous.
        ===============     ====================================================================

        .. note::
            The ``snap_to_zoom`` method only applies in 2D mode.
        """
        return self._snap_to_zoom

    @snap_to_zoom.setter
    def snap_to_zoom(self, value):
        self._snap_to_zoom = value

    _rotation = Float(0).tag(sync=True)
    _readonly_rotation = Float(0).tag(sync=True)
    _link_writeonly_rotation = Float(0).tag(sync=True)

    @property
    def rotation(self):
        """
        Get/Set the clockwise rotation of due north in relation to the top
        of the view in degrees in 2D mode.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required float.
        ===============     ====================================================================

        .. note::
            ``rotation`` cannot be set in 3D mode. Rather, 3D mode uses the :attr:`~arcgis.widgets.MapView.heading`
            property.

        .. code-block:: python

            #Usage Example in 2D mode

            >>> from arcgis.gis import GIS, Item
            >>> from arcgis.widgets import MapView
            >>> map2 = gis.map("California")
            >>> map2.rotation
            <134.17566310758235>
        """
        return self._readonly_rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value

    _heading = Float(0).tag(sync=True)
    _readonly_heading = Float(0).tag(sync=True)
    _link_writeonly_heading = Float(0).tag(sync=True)

    @property
    def heading(self):
        """
         Get/Set the compass heading of the camera in degrees when in 3D mode. ``heading`` is
         zero when north is the top of the screen. It increases as the view rotates
         clockwise. The angles are always normalized between 0 and 360 degrees.

        .. note::
             ``heading`` cannot be set in 2D mode. Rather, 2D mode uses the :attr:`~arcgis.widgets.MapView.rotation`
             property.

         .. code-block:: python

             #Usage Example in 3D mode

             >>> from arcgis.gis import GIS, Item
             >>> from arcgis.widgets import MapView
             >>> map3d = gis.map("California", mode ="3D")
             >>> map3d.heading
             <225.82433689241765>
        """
        return self._readonly_heading

    @heading.setter
    def heading(self, value):
        self._heading = value

    _tilt = Float(0).tag(sync=True)
    _readonly_tilt = Float(0).tag(sync=True)
    _link_writeonly_tilt = Float(0).tag(sync=True)

    @property
    def tilt(self):
        """
        Get/Set the tilt of the camera in degrees with respect to the
        surface as projected down from the camera position, when in 3D mode. ``tilt`` is zero when
        looking straight down at the surface and 90 degrees when the camera is
        looking parallel to the surface.

        .. note::
            The ``tilt`` method is not applicable, and cannot be set, in 2D mode.
        """
        return self._readonly_tilt

    @tilt.setter
    def tilt(self, value):
        self._tilt = value

    @property
    def basemap(self):
        """
        Get/Set the basemap you would like to apply to the widget.

        ===============     ====================================================================
        **Parameter**       **Description**
        ---------------     --------------------------------------------------------------------
        value               Required string. Ex: ('topo', 'national-geographic', etc.).

                            .. note::
                                See :attr:`~arcgis.widgets.MapView.basemaps` for a full list
                                of options.
        ===============     ====================================================================

        :return: basemap being used.

        .. code-block:: python

            # Usage example: Set the widget basemap equal to an item

            >>> from arcgis.mapping import WebMap
            >>> widget = gis.map()

            >>> # Use basemap from another item as your own
            >>> widget.basemap = webmap
            >>> widget.basemap = tiled_map_service_item
            >>> widget.basemap = image_layer_item
            >>> widget.basemap = webmap2.basemap
            >>> widget.basemap - 'national-geographic'

        """
        return self._basemap

    @basemap.setter
    def basemap(self, value):
        if value in self.basemaps:
            if value.startswith("arcgis-"):
                if self.gis is None or self.gis._portal.is_logged_in is False:
                    raise ValueError(
                        "This basemap requires you to be authenticated. Please login or try a different basemap."
                    )
            self._basemap = value
            self.webmap.basemap = value
            self._webmap = self.webmap._webmapdict
        elif value in self.gallery_basemaps:
            self._basemap = value
            self.webmap.basemap = value
            self._webmap = self.webmap._webmapdict
        else:
            try:
                self.webmap.basemap = value
                # takes dict object
                self._gallery_basemaps["base"] = self.webmap._basemap
                self._basemap = "base"
                # You need to re-write this dict to trigger the JS side change
                copy_gallery = dict(self._gallery_basemaps)
                self._gallery_basemaps = {}
                self._gallery_basemaps = copy_gallery
                self._webmap = self.webmap._webmapdict
            except Exception:
                raise RuntimeError("Basemap '{}' isn't valid".format(value))

    _basemap = Unicode("topo-vector").tag(sync=True)
    """What basemap you would like to apply to the widget ('topo',
    'national-geographic', etc.). See `basemaps` for a full list
    """
    mode = Unicode("2D").tag(sync=True)
    """The string that specifies whether the map displays in '2D' mode
    (MapView) or '3D' mode (SceneView). Possible values: '2D', '3D'.

    .. raw:: html

        <p>Note that you can also toggle between '2D' and
        '3D' mode by pressing the
        <img src=" data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAA
        Bzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH4
        gcFFxIGmlhcGAAAAslJREFUWMPt10+IV1UUB/DPNMMYCTqzqV4GLkpCkIwDLgYKbGG4iLIa
        KNNdRoGFUIFFUpYtLNBVhFIQJJQKEaK5CIpgWgxmZxVRWJsIHiSNM4WG5mCbG/x4zPzmzc8
        ZKvDs7r3nvvO958/3nMc1+Zelr3MRERdxZo47K3AOF7AqM5dcDYCBxvpMZq6ZTTkinsAWPJ
        SZUxHx7dV6YKCtYkSsxsPYkJnTCxWC6+ahey8+WkjjrT0QEbdiO+5c6CQcaGG8HxtxfLbXR
        8ThGbZHMnNlTwAiog+bcQrf4BeMzPaRzHysJajWHliBvfi6ABnHXxGxEY9gJyYXMwQbcCNO
        Z+bJ8qKdeAA/4DsMYvmiAMjM9yNi+T/nJSSPZ+baRph+bLj9JXyVmWMde0PYk5nPzrcMa2y
        JiEG8jM8bIK/gz8advVgXEfcU48PYjV298MAkJnAZv2F1KUddkvFKZu7H2lKyu/BKZk617Q
        XzptbZqDsiHsWJzDz//+mGXchoKe7PzCNz6A3iSWzFUAnfBI7gvcy81LzT38L4MryGbVVV9
        dV1faoLXR/D97ipcMWlkoT3YbSqqtN1Xf/RGkApxdexB+twvKqq0aqqxuu67tRbgpPYgZ8R
        uB4/dZT629hXVdWJuq6n23bDN7A7M8+VhBsr7PhiQ+8pTJdXj+KtzHwTR0sXHcrMs/gE21o
        3o5nIo4AYa2xvLS881ii5o/gS5wsnfIrP8E4v80A3uaGEaFlHWDbhjrK8BYexCit7mojmkH
        68Whjz+TI9HcJtmflrB6i7MdwrgJGZWmxpxb8XL/QXGr4ZZ9EkobtKI5s/gDmGiw/xXEnOf
        XgG6/FFRExgc2ZOloQ8tBghOFBmhhcwXhrV9uL2D3A5ItaU+L/b61DazTsX8WAZVtZHxO0d
        x8PlbD82NdmwbyF5vRDS0+XfYQhT5QfmYxwsQK/Jf0v+BvBD+VardrUKAAAAAElFTkSuQmC
        C"> icon in the widget UI.</p>

    """

    _readonly_extent = Dict({}).tag(sync=True)
    _extent = Dict({}).tag(sync=True)
    _link_writeonly_extent = Dict({}).tag(sync=True)

    @property
    def extent(self):
        """
        Get/Set the map widget's extent.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        value                  Required dict.
                                A `[[xmin, ymin], [xmax, ymax]]` list, Spatially Enabled Data Frame ``full_extent``,
                                or a dict that represents the JSON of the map widget's extent.

                                Examples for each:
                                web_map.extent = [[-124.35, 32.54], [-114.31, 41.95]]
                                web_map.extent = data_frame.spatial.full_extent
                                web_map.extent = {
                                        "xmin": -124.35,
                                        "ymin": 32.54,
                                        "xmax": -114.31,
                                        "ymax": 41.95
                                    }
        ==================     ====================================================================

        .. code-block:: python

                #Usage Example

                >>> from arcgis.gis import GIS, Item
                >>> from arcgis.widgets import MapView
                >>> map2 = gis.map("California")
                >>> map2.extent
                {
                'xmin': -124.20822999999997, 'ymin': 31.436105693000048,
                'xmax': -114.33222999999997, 'ymax': 41.31210569300005
                }

        """
        if self._readonly_extent:
            return self._readonly_extent
        else:
            return self._extent

    @extent.setter
    def extent(self, value):
        try:
            if isinstance(value, dict):
                if "spatialReference" in value and "spatialReference" in self._extent:
                    if (
                        not self._extent["spatialReference"]
                        == value["spatialReference"]
                    ):
                        value = _reproject_extent(
                            value, self._extent["spatialReference"]
                        )
                self._extent = value
            elif _is_iterable(value) and isinstance(value, tuple):
                self._extent = {
                    "xmin": value[0],
                    "ymin": value[1],
                    "xmax": value[2],
                    "ymax": value[3],
                }
            elif _is_iterable(value):
                self._extent = {
                    "xmin": value[0][0],
                    "ymin": value[0][1],
                    "xmax": value[1][0],
                    "ymax": value[1][1],
                }
            elif len(value) == 0:
                pass
            else:
                raise Exception
        except Exception:
            if _is_iterable(value) and len(value) == 0:
                pass
            else:
                log.warning(
                    "extent must be set to either a 2d list, spatially "
                    "enabled data frame full_extent, or dict. Values specified "
                    "must include xmin, ymin, xmax, ymax. Please see the API doc for "
                    "more information"
                )

    _readonly_center = Dict({}).tag(sync=True)
    _center = Dict({}).tag(sync=True)
    _center_long_lat = List([]).tag(sync=True)

    @property
    def center(self):
        """
        Get/Set the center of the ``Map Widget``.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        setter                 A `[lat, long]` list, or a dict that represents the JSON of the map
                               widget's center.
        ==================     ====================================================================

        :return: A dict that represents the JSON of the map widget's center.

        .. code-block:: python

            #Usage Example

            >>> from arcgis.gis import GIS, Item
            >>> from arcgis.widgets import MapView
            >>> map2 = gis.map("California")
            >>> map2
            >>> map2.center
            {
            'spatialReference': {'latestWkid': 3857, 'wkid': 102100},
            'x': -13277101.270396618,
            'y': 4374001.4894094905
            }
        """
        if self._readonly_center:
            return self._readonly_center
        else:
            return self._center

    @center.setter
    def center(self, value):
        if isinstance(value, dict):
            self._center = value
        elif len(value) != 2:
            log.warning(
                "If setting center to a list/tuple, the len() "
                "must be exactly 2 entries long"
            )
        else:
            self._center_long_lat = [value[1], value[0]]

    # Start layer specific state
    # There is no backbone event type for a foolist.append(), but writing a new
    # tuple to self.portal_items will trigger a model change
    _add_this_notype_layer = Dict({}).tag(sync=True)
    _draw_these_notype_layers_on_widget_load = Tuple(tuple()).tag(sync=True)

    _layers_to_remove = Tuple(()).tag(sync=True)
    _add_this_graphic = Dict({}).tag(sync=True)
    _draw_these_graphics_on_widget_load = Tuple(tuple()).tag(sync=True)

    # End layer specific state
    # Start webmap/webscene state
    _webmap = Dict({}).tag(sync=True)
    _webscene = Dict({}).tag(sync=True)
    _trigger_webscene_save_to_this_portal_id = Unicode("").tag(sync=True)
    _readonly_webmap_from_js = Dict({}).tag(sync=True)

    # end webmap/webscene state
    # start miscellanous model state
    _custom_msg = Unicode("").tag(sync=True)
    _portal_token = Unicode("").tag(sync=True)
    _auth_mode = Unicode("").tag(sync=True)
    _portal_url = Unicode("").tag(sync=True)
    _portal_sharing_rest_url = Unicode("").tag(sync=True)
    _username = Unicode("").tag(sync=True)
    _trigger_interactive_draw_mode_for = Dict({}).tag(sync=True)
    _trigger_new_jlab_window_with_args = Dict({}).tag(sync=True)
    hide_mode_switch = Bool(False).tag(sync=True)
    """When ``hide_mode_switch`` is set to ``True`` the 2D/3D switch button will be hidden from the widget.
   
    .. note::
        Once the button is hidden, it cannot be made visible again: a new MapView instance must be created to see the 
        button.
    """
    jupyter_target = Unicode("").tag(sync=True)
    """
    ``jupyter_target`` is a readonly string that is either ``lab`` or ``notebook``: ``jupyter_target`` represents if
    this widget is drawn in a Jupyter Notebook environment or a JupyterLab
    environment.
    """
    tab_mode = Unicode("auto").tag(sync=True)
    """
    .. raw:: html

        <p>The ``tab_mode`` property is a string property that specifies the 'default' behavior of toggling a
        new window in a JupyterLab environment, whether that is called by
        pressing the <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC
        AAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYA
        AAAB3RJTUUH4gcUAAIIxu7mQQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoI
        EdJTVBkLmUHAAABI0lEQVRYw+1Wy3GDMBTch3JLBanA+TQR0gDz4JB0Y1xISkBAA4Eq
        HOeYU6qAzYVkPAyWYSLii/Ymzb63q89qBAQEXBgyNamquYhs5zSw1v72SNOUp3gkd2V
        Z5uP5K1dzkrslqxnzRWQD4MVV4zQw5XguP8uyDckGwBeAm1M10Rrnqqq3fd+3JEVEYh
        c3WkMcQDMM46IoPv7NQJIkdyLS/oiXZXk4VxP5FDfGNEvEz17Cuciy7J7kGwB0XRfXd
        X0Yp+NoZ/waGMQbABzE35ekKfIkDhF5mhJf7Q6o6sOReFwUxd7BzdM0ffR5BALglaQA
        iK21eydZZEsSAFpfBkjy2Rhz7Vr5qimoqurTR4JWeYqDgWDAWwpUNb+ogbn/woCAgL/
        gG9knfWwKBGcvAAAAAElFTkSuQmCC"> icon in the widget UI, or by calling
        <i>toggle_window_view()</i> function without arguments.</p>

    .. note::
        After a widget is 'seperated' from the notebook, you can drag
        it, split it, put it in a new tab, etc. See the JupyterLab guide pages
        for more information.

    ==================     ====================================================================
    **Value**              **Description**
    ------------------     --------------------------------------------------------------------
    'auto'                 The default tab_mode: will move the widget to a `split-right` view
                           if there are no other open windows besides the current notebook,
                           otherwise will place the widget in a new tab.
    ------------------     --------------------------------------------------------------------
    'split-top'            Will move the widget above the current notebook instance.
    ------------------     --------------------------------------------------------------------
    'split-bottom'         Will move the widget below the current notebook instance.
    ------------------     --------------------------------------------------------------------
    'split-left'           Will move the widget to the left of the current notebook instance.
    ------------------     --------------------------------------------------------------------
    'split-right'          Will move the widget to the right of the current notebook instance.
    ------------------     --------------------------------------------------------------------
    'tab-before'           Will move the widget to a new tab 'before' the tab that represents
                           the current notebook instance.
    ------------------     --------------------------------------------------------------------
    'tab-after'            Will move the widget to a new tab 'after' the tab that represents
                           the current notebook instance.
    ==================     ====================================================================
    """

    ready = Bool(False).tag(sync=True)
    """
    ``ready`` is a readonly bool that represents if the map widget has been drawn
    in the notebook.
    """
    _js_cdn_override = Unicode().tag(sync=True)

    legend = Bool(False).tag(sync=True)
    """
    If ``legend`` is set to ``True``, a legend will display in the widget that
    describes all layers added to the map. If set to ``False``,the legend will be hidden.
    
    .. note::
        The default is ``False`` .
    """

    _gallery_basemaps = Dict({}).tag(sync=True)

    @property
    def gallery_basemaps(self):
        """
        The ``gallery_basemaps`` property allows for viewing of your portal's custom basemap group.
        """
        if len(self._gallery_basemaps) <= 1:
            # If the only loaded gallery_basemaps is 'default', load the rest
            bmquery = self.gis.properties["basemapGalleryGroupQuery"]
            basemapsgrp = self.gis.groups.search(bmquery, outside_org=True)
            if len(basemapsgrp) == 1:
                for bm in basemapsgrp[0].content():
                    if bm.type.lower() == "web map":  #  Only use WebMaps
                        item_data = bm.get_data()
                        bm_title = bm.title.lower().replace(" ", "_")
                        self._gallery_basemaps[bm_title] = item_data["baseMap"]
                # Appending to dict doesn't cause change in model:
                # Must overwrite with a blank dict, then put the new dict
                copy_gallery = dict(self._gallery_basemaps)
                self._gallery_basemaps = {}
                self._gallery_basemaps = copy_gallery
                return list(self._gallery_basemaps.keys())
            else:
                return list(self._gallery_basemaps.keys())
        else:
            return list(self._gallery_basemaps.keys())

    _uuid = Unicode("").tag(sync=True)

    _trigger_print_js_debug_info = Unicode("").tag(sync=True)

    # end miscellanous model state
    # End model specific state

    # Start Other properties that don't interact with the model
    _hashed_layers = OrderedDict()  # how we store layers
    _default_webscene_text_property = DEFAULT_WEBSCENE_TEXT_PROPERTY

    @property
    def layers(self):
        """
        The ``layers`` property is a list of the JSON representation of layers added to the map widget
        using the :attr:`~arcgis.widgets.MapView.add_layers` method.

        .. code-block:: python

            #Usage Example in 2D mode

            >>> from arcgis.gis import GIS, Item
            >>> from arcgis.widgets import MapView
            >>> gis = GIS("pro")
            >>> itms = gis.content.search("owner:"+ gis.users.me.username)
            >>> single_item =itms[1]
            >>> new_layer= FeatureLayer.fromitem(item = single_item)
            >>> map1 = gis.map("United States")
            >>> map1.add_layer(new_layer)
            >>> map1.layers
            [<FeatureLayer url:"https://services7.arcgis.com/JEwYeAy2cc8qOe3o/arcgis/rest/services/Power_Plants_Itm/FeatureServer/0">]
        """
        return [self._hashed_layers[key] for key in self._hashed_layers]

    # end how we store layers
    @property
    def basemaps(self):
        """
        The ``basemaps`` layers are a list of possible basemaps to set :attr:`~arcgis.widgets.MapView.basemap` with:

        1. Dark Grey Vector
        2. Gray Vector
        3. Hybrid
        4. Oceans
        5. OSM
        6. Satellite
        7. Streets Navigation Vector
        8. Streets Night Vector
        9. Streets Relief Vector
        10. Streets Vector
        11. Terrain
        12. Topographic Vector

        There are basemap layers available if you are authenticated or provide an api key.

        1. ArcGIS Imagery
        2. ArcGIS Imagery Standard
        3. ArcGIS Imagery Labels
        4. ArcGIS Light Gray
        5. ArcGIS Dark Gray
        6. ArcGIS Navigation
        7. ArcGIS Navigation Night
        8. ArcGIS Streets
        9. ArcGIS Streets Night
        10. ArcGIS Streets Relief
        11. ArcGIS Topographic
        12. ArcGIS Oceans
        13. ArcGIS Standard
        14. ArcGIS Standard Relief
        15. ArcGIS Streets
        16. ArcGIS Streets Relief
        17. ArcGIS Open Street Map Light Gray
        18. ArcGIS Open Street Map Dark Gray
        19. ArcGIS Terrain
        20. ArcGIS Community
        21. ArcGIS Charted Territory
        22. ArcGIS Colored Pencil
        23. ArcGIS Nova
        24. ArcGIS Modern Antique
        25. ArcGIS Midcentury
        26. ArcGIS Newspaper
        27. ArcGIS Hillshade Light
        28. ArcGIS Hillshade Dark
        29. ArcGIS Human Geography
        30. ArcGIS Human Geography Dark

        """
        if self._gis is not None and self._gis._is_authenticated:
            return [
                "dark-gray-vector",
                "gray-vector",
                "hybrid",
                "oceans",
                "osm",
                "satellite",
                "streets-navigation-vector",
                "streets-night-vector",
                "streets-relief-vector",
                "streets-vector",
                "terrain",
                "topo-vector",
                "arcgis-imagery",
                "arcgis-imagery-standard",
                "arcgis-imagery-labels",
                "arcgis-light-gray",
                "arcgis-dark-gray",
                "arcgis-navigation",
                "arcgis-navigation-night",
                "arcgis-streets",
                "arcgis-streets-night",
                "arcgis-streets-relief",
                "arcgis-topographic",
                "arcgis-oceans",
                "osm-standard",
                "osm-standard-relief",
                "osm-streets",
                "osm-streets-relief",
                "osm-light-gray",
                "osm-dark-gray",
                "arcgis-terrain",
                "arcgis-community",
                "arcgis-charted-territory",
                "arcgis-colored-pencil",
                "arcgis-nova",
                "arcgis-modern-antique",
                "arcgis-midcentury",
                "arcgis-newspaper",
                "arcgis-hillshade-light",
                "arcgis-hillshade-dark",
                "arcgis-human-geography",
                "arcgis-human-geography-dark",
            ]
        else:
            return [
                "dark-gray-vector",
                "gray-vector",
                "hybrid",
                "oceans",
                "osm",
                "satellite",
                "streets-navigation-vector",
                "streets-night-vector",
                "streets-relief-vector",
                "streets-vector",
                "terrain",
                "topo-vector",
            ]

    # End other properties that don't interact with the model

    def __init__(self, gis=None, item=None, mode="2D", **kwargs):
        """Constructor of Map widget.
        Accepts the following keyword arguments:
        gis     The gis instance with which the map widget works, used for authentication, and adding secure layers and
                private items from that GIS
        item    web map item from portal with which to initialize the map widget
        """
        super(MapView, self).__init__(**kwargs)
        self._uuid = str(uuid4())

        # Set up the visual display of the layout
        self.layout.height = DEFAULT_ELEMENT_HEIGHT
        self.layout.width = "100%"

        # Set up gis object
        self._setup_gis_properties(gis)

        # Set up miscellanous properties needed on startup
        self.mode = mode
        self._hashed_layers = OrderedDict()
        self._setup_js_cdn()
        self._setup_default_basemap()

        # Handle webmaps and webscenes
        self.webmap_item = None
        self.webscene_item = None
        if item != None:
            self._check_if_webmap(item)
            self._check_if_webscene(item)
        else:
            from arcgis.mapping import WebMap

            self.webmap = WebMap()

        # Handle callbacks and such
        self.on_msg(self._handle_map_msg)
        self._draw_end_handlers = widgets.CallbackDispatcher()
        self._click_handlers = widgets.CallbackDispatcher()

        # Set up LocalRasterOverlay instance
        self._raster = LocalRasterOverlayManager(mapview=self)

        self._synced_mapviews = []
        self._mapview_uuid_to_dlinks = {}
        self._dlinks = []
        if gis:
            self._gis = gis
        else:
            from arcgis import env

            self._gis = env.active_gis

    # Start screenshot specific section

    def _ipython_display_(self):
        """Override the parent ipython display function that is called
        whenever is displayed in the notebook. Display a blank area
        below the map widget that can be controlled via a display handler
        set to self._preview_image_display_handler.
        """
        self._setup_gis_properties(self.gis)
        super(MapView, self)._ipython_display_()
        self._preview_image_display_handler = display(
            HTML(self._assemble_img_preview_html_str("")),
            display_id="preview-" + str(self._uuid),
        )
        self._preview_html_embed_display_handler = display(
            HTML(self._assemble_html_embed_html_str("")),
            display_id="preview-html-" + str(self._uuid),
        )

    def _assemble_img_preview_html_str(self, img_src):
        """Helper function that creates an HTML string of the <img> tag
        to add to the notebook with the correct <div> class to be hidden
        """
        img_html = '<img src="' + img_src + '"></img>'
        class_id = "map-static-img-preview-" + self._uuid
        return '<div class="' + class_id + '">' + img_html + "</div>"

    _preview_screenshot_callback_resp = Unicode("").tag(sync=True)

    @observe("_preview_screenshot_callback_resp")
    def _preview_image_update_callback(self, change):
        """Called every time the front end takes a screenshot for a
        map preview
        """
        img_data_uri_str = self._parse_js_resp(change["new"])
        self._preview_image_display_handler.update(
            HTML(self._assemble_img_preview_html_str(img_data_uri_str))
        )

    _cell_output_screenshot_callback_resp = Unicode("").tag(sync=True)

    @observe("_cell_output_screenshot_callback_resp")
    def _cell_output_screenshot_update_callback(self, change):
        """Called every time the front end takes a screenshot for a
        cell output"""
        if self._cell_output_display_handler:
            img_data_uri_str = self._parse_js_resp(change["new"])
            self._cell_output_display_handler.update(
                HTML("<img src=" + img_data_uri_str + "></img>")
            )

    _file_output_screenshot_callback_resp = Unicode("").tag(sync=True)

    @observe("_file_output_screenshot_callback_resp")
    def _file_output_screenshot_update_callback(self, change):
        """Called every time the front end takes a screenshot for a file
        write"""
        img_data_uri_str = self._parse_js_resp(change["new"])
        img_data_raw_str = img_data_uri_str.split("base64,")[-1]
        if self._screenshot_file_output_path and img_data_raw_str:
            with open(self._screenshot_file_output_path, "wb") as f:
                img_data_raw_bytes = base64.b64decode(img_data_raw_str)
                f.write(img_data_raw_bytes)

    def _parse_js_resp(self, change_new):
        """In 3D mode, the Data URI is returned from JS. In 2D mode, a URL
        to a png is returned. This function will always return a base64
        data URI of the image represented
        """
        if not change_new.startswith("http"):
            # Already is a data URI
            return change_new
        else:
            # It's a URL: download and parse to base64 data URI, return it
            with urllib.request.urlopen(change_new) as resp:
                encoded_body = base64.b64encode(resp.read())
                return "data:image/png;base64,{}".format(encoded_body.decode())

    def _assemble_html_embed_html_str(
        self, iframe_srcdoc_html, class_id_root="map-html-embed-preview-"
    ):
        """Helper function that creates an HTML string of the <iframe> tag
        to add to the notebook with the correct <div> class to be hidden
        """
        iframe_html = (
            f"<iframe height='{self.layout.height}' width='100%' "
            f"srcdoc='{iframe_srcdoc_html}'>"
            f"Your browser doesn't support map widget previews"
            f"</iframe>"
        )
        iframe_html = iframe_html if iframe_srcdoc_html else ""
        other_html = "<br><h4></h4>"
        class_id = class_id_root + self._uuid
        return f'<div class="{class_id}">{iframe_html}</div>'

    print_service_url = Unicode("").tag(sync=True)
    """
    .. warning::
        This property is obselete as of >v1.6 of the Python API, since
        the underlying JavaScript code ran during a `take_screenshot()` Python
        call has been has been changed to `MapView.takeScreenshot()` instead
        of calling a Print Service URL. Any value you set to this property
        will be ignored (2D screenshots will still be taken successfully).
    """

    _trigger_screenshot_with_args = Dict({}).tag(sync=True)

    def take_screenshot(self, output_in_cell=True, set_as_preview=True, file_path=""):
        """
        The ``take_screenshot`` method takes a screenshot of the current widget view.

        .. note::
            Only works in a Jupyter Notebook environment.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        output_in_cell         Optional bool, default ``True``. Will display the screenshot in the
                               output area of the cell where this function is called.
        ------------------     --------------------------------------------------------------------
        set_as_preview         Optional bool, default ``True``. Will set the screenshot as the static
                               image preview in the cell where the map widget is being displayed.
                               Use this flag if you want the generated HTML previews of your
                               notebook to have a map image visible.
        ------------------     --------------------------------------------------------------------
        file_path              Optional String, default `""`. To output the screenshot to a `.png`
                               file on disk, set this String to a writeable location file path
                               (Ex. `file_path = "/path/to/documents/my_screenshot.png"`).
        ==================     ====================================================================

        In all notebook outputs, each image will be encoded to a base64
        data URI and wrapped in an HTML <img> tag, like
        `<img src="base64Str">`. This means that the data for the image lives
        inside the notebook file itself, allowing for easy sharing of
        notebooks and generated HTML previews of notebooks.

        .. note::
            This function acts asynchronously, meaning that the Python function
            will return right away, with the notebook outputs/files being
            written after an indeterminate amount of time. Avoid calling this
            function  multiple times in a row if the asynchronous portion of
            the function hasn't finished yet.

        .. note::
            When this function is called with ``set_as_preview = True``, the
            static image preview will overwrite the embedded HTML element
            preview from any previous ``MapView.embed_html(set_as_preview=True)``
            call

        """
        self._clear_embed_html_preview()
        if not self.ready:
            log.warning(
                "Cannot take screenshot if widget is not visible in "
                "notebook: Please try again when widget is visible."
            )
            return False

        if output_in_cell:
            self._cell_output_screenshot_callback_resp = ""  # Clear existing
            loading_html = HTML(
                '<img style="float: left; padding-right: '
                '5px" src="' + _loading_icon_str + '"></img>'
                "Taking Screenshot: Please wait... </div>"
            )
            self._cell_output_display_handler = display(
                loading_html, display_id="cell-output-" + str(uuid4())
            )
        if file_path:
            self._file_output_screenshot_callback_resp = ""  # Clear existing
            if not file_path.endswith(".png"):
                file_path = file_path + ".png"
            self._screenshot_file_output_path = file_path

        self._trigger_screenshot_with_args = {
            "_": str(uuid4()),
            "set_as_preview": set_as_preview,
            "output_in_cell": output_in_cell,
            "file_path": bool(file_path),
        }

    def _clear_embed_html_preview(self):
        self._preview_html_embed_display_handler.update(
            HTML(self._assemble_html_embed_html_str(""))
        )

    def _clear_static_image_preview(self):
        self._preview_image_display_handler.update(
            HTML(self._assemble_img_preview_html_str(""))
        )

    def embed(self, output_in_cell=True, set_as_preview=True):
        """
        The ``embed`` method embeds the current state of the map into the underlying notebook
        as an interactive HTML/JS/CSS element.

        .. note::
            This element will always display
            this 'snapshot' state of the map, regardless of any future Python code ran.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        output_in_cell         Optional bool, default ``True``. Will display the embedded HTML
                               interactive map in the output area of the cell where this function
                               is called.
        ------------------     --------------------------------------------------------------------
        set_as_preview         Optional bool, default `True`. Will display the embedded HTML
                               interactive map in the cell where the map widget is being displayed.
                               Use this flag if you want the generated HTML previews of your
                               notebook to have an interactive map displayed.
        ==================     ====================================================================

        In all notebook outputs, each embedded HTML element will contain
        the entire map state and all relevant HTML wrapped in an <iframe>
        element. This means that the data for the embedded HTML element lives
        inside the notebook file itself, allowing for easy sharing of
        notebooks and generated HTML previews of notebooks.

        .. note::
            When this function is called with ``set_as_preview = True``, the
            embedded HTML preview element will overwrite the static image
            preview from any previous `MapView.take_screenshot(set_as_preview=True)`
            call

        .. note::
            Any embedded maps must only reference publicly available data. The
            embedded map must also have access to the `Unpkg <https://unpkg.com>`_
            to load the necessary `JavaScript` components on the page

        """
        self._clear_static_image_preview()
        html_repr_path = os.path.join(tempfile.gettempdir(), f".{self._uuid}.html")
        self.export_to_html(html_repr_path)
        with open(html_repr_path, "r") as f:
            iframe_srcdoc_html = f.read()
            if output_in_cell:
                display(
                    HTML(
                        self._assemble_html_embed_html_str(
                            iframe_srcdoc_html,
                            class_id_root="map-html-embed-in-cell-",
                        )
                    )
                )
            if set_as_preview:
                self._preview_html_embed_display_handler.update(
                    HTML(self._assemble_html_embed_html_str(iframe_srcdoc_html))
                )

    # End screenshot specific section

    def _setup_gis_properties(self, gis):
        # This function is called during __init__, as well as during any
        # subsequent draw in a notebook. Priority of how the GIS properties
        # of the widget are set:
        # - Always use the gis object passed in as an arg in __init__
        # - Fallback to the active_gis if no arg passed in
        # - Fallback to anon AGOL non-active if neither of the above specified
        gis = arcgis.env.active_gis if gis is None else gis
        if gis is None:
            from arcgis.gis import GIS

            gis = GIS(set_active=False)
        self.gis = gis

        # Now that self.gis property is properly set, determine the auth mode
        if self.gis._portal.con.token:
            self._portal_token = str(self.gis._portal.con.token)
            self._auth_mode = "tokenBased"
        elif isinstance(
            self.gis._con._session.auth, (_MultiAuth, SupportMultiAuth)
        ) and hasattr(self.gis._con._session.auth, "authentication_modes"):
            tokens = [
                auth.token
                for auth in self.gis._con._session.auth.authentication_modes
                if hasattr(auth, "token")
            ]
            if len(tokens) > 0:
                self._portal_token = str(tokens[0])
                self._auth_mode = "tokenBased"
            else:
                self._auth_mode = "anonymous"
        else:
            self._auth_mode = "anonymous"

        # Set the properties that aren't dependent on auth mode
        self._portal_url = self._get_portal_url()
        self._portal_sharing_rest_url = self.gis._public_rest_url
        self._username = str(gis._username)

    def _get_portal_url(self):
        try:
            return self.gis._portal._public_rest_url.split("sharing")[0]
        except Exception as e:
            return self.gis.url

    def _setup_js_cdn(self):
        if _js_cdn_override_global != "":
            # If the user had previously set this global property, use it
            self._js_cdn_override = _js_cdn_override_global
        elif os.environ.get("JSAPI_CDN", ""):
            self._js_cdn_override = os.environ.get("JSAPI_CDN", "")
        else:
            # Else, test default CDNs and portal CDNs
            default_cdn_unreachable = not self._is_reachable(_DEFAULT_JS_CDN)
            if default_cdn_unreachable:
                _portal_cdn = "{}/jsapi/jsapi4/".format(getattr(self.gis, "_url", ""))
                # use gis._url to get private url (disconn IWA edge case)
                self._js_cdn_override = _portal_cdn
                log.debug(
                    "Disconnected environment detected: "
                    + "using JS API CDN from {}. ".format(_portal_cdn)
                    + "Make sure you have a JSAPI4 compatible basemap "
                    + "set as the default basemap in your portal."
                )

    def _setup_default_basemap(self, basemap=None):
        """This method gets called once on startup, it populates the 'default'
        basemap field and the corresponding JSON without loading the rest
        of the `gallery_basemaps` property (which has a long load time)
        """
        if basemap:
            # used instead of basemap setter to avoid the reset of the associated webmap's basemap on instantiation
            self._gallery_basemaps["base"] = basemap
            self._basemap = "base"
            # You need to re-write this dict to trigger the JS side change
            copy_gallery = dict(self._gallery_basemaps)
            self._gallery_basemaps = {}
            self._gallery_basemaps = copy_gallery
        elif (
            self.gis.org_settings is not None
            and "defaultBasemap" in self.gis.org_settings
            and self.gis.org_settings["defaultBasemap"]
        ):
            self._gallery_basemaps["default"] = self.gis.org_settings["defaultBasemap"]
            self._basemap = "default"
            # Add to text property so default is recorded
            self._default_webscene_text_property["baseMap"] = self._gallery_basemaps[
                "default"
            ]
            # You need to re-write this dict to trigger the JS side change
            copy_gallery = dict(self._gallery_basemaps)
            self._gallery_basemaps = {}
            self._gallery_basemaps = copy_gallery
        else:
            # Enterprise 10.7.1 workflow
            self._gallery_basemaps["default"] = self.gis.properties["defaultBasemap"]
            self._basemap = "default"
            # Add to text property so default is recorded
            self._default_webscene_text_property["baseMap"] = self._gallery_basemaps[
                "default"
            ]
            # You need to re-write this dict to trigger the JS side change
            copy_gallery = dict(self._gallery_basemaps)
            self._gallery_basemaps = {}
            self._gallery_basemaps = copy_gallery

    def _is_reachable(self, url):
        import urllib.request

        def is_good_range(code):
            return (code >= 200) and (code < 300)

        try:
            conn = urllib.request.urlopen(url)
        except urllib.error.HTTPError as e:
            #  Return code error (e.g. 404, 501, ...)
            return is_good_range(e.code)
        except Exception as e:
            #  Not an HTTP-specific error (e.g. connection refused)
            #  as well as any malformed url passed in
            return False
        else:
            #  200 or other 'good' code
            return is_good_range(conn.code)

    def _check_if_webmap(self, item):
        from arcgis.mapping import WebMap

        if isinstance(item, Item) and (item.type.lower() == "web map"):
            item = WebMap(item)
        if isinstance(item, WebMap):
            self.webmap = item
            if hasattr(item, "item") and hasattr(item.item, "extent"):
                self.extent = item.item.extent

    def _check_if_webscene(self, item):
        if isinstance(item, Item):
            if item.type.lower() == "web scene":
                self.webscene_item = item
                if hasattr(item, "id"):
                    self._webscene = {"portalItem": {"id": item.id}}

    @classmethod
    def set_js_cdn(cls, js_cdn):
        """
        The ``set_js_cdn`` function is called before the creation of any MapView object, and
        each instantiated object will use the specified ``js_cdn`` parameter as
        the  ArcGIS API for JavaScript CDN URL instead of the default
        http://js.arcgis.com/4.X/. This functionality is necessary in
        disconnected  environments if the portal you are connecting to doesn't
        ship with the minimum necessary JavaScript API version.

        .. note::
            You may not need to call this function to view the widget in
            disconnected environments: if your computer cannot reach js.arcgis.com,
            and you have a :class:`~arcgis.gis.GIS` connection to a portal, the widget will
            automatically attempt to use that Portal's JS API that it ships with.
        """
        global _js_cdn_override_global
        _js_cdn_override_global = js_cdn

    @property
    def local_raster_file_format(self):
        """
        The ``local_raster_file_format`` method is a string getter & setter. When calling
        ``map.add_layer(arcgis.raster.Raster())`` for a local raster file,
        an intermediate image file must be written to disk in order to
        successfully display on the map. This file format can be one of
        the following:

        ===================     ================================================
        **Possible Values**     **Description**
        -------------------     ------------------------------------------------
        ``"jpg"`` (Default)     Write raster to a ``.JPG`` file. This results
                                in a lossy image, but should draw quicker than
                                a ``.PNG`` file. Requires the ``PIL`` image
                                processing package (distributed under the name
                                of it's active fork, "Pillow")
        -------------------     ------------------------------------------------
        ``"png"``               Write raster to a ``.PNG`` file. This results
                                in a lossless image, but it might take a longer
                                time to draw than a ``.JPG`` file.
        ===================     ================================================
        """
        return self._raster.file_format

    @local_raster_file_format.setter
    def local_raster_file_format(self, value):
        self._raster.file_format = value

    def add_layer(self, item, options=None):
        """
        The ``add_layer`` method adds the specified ``Layer`` or :class:`~arcgis.gis.Item` to the
        map widget.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required object. You can specify :class:`~arcgis.gis.Item` objects, ``Layer`` objects
                               such as :class:`~arcgis.features.FeatureLayer` , ImageryLayer, MapImageLayer,
                               :class:`~arcgis.features.FeatureSet` ,
                               :class:`~arcgis.features.Collection`, :class:`~arcgis.raster.Raster` objects, etc.

                               .. note::
                               :class:`~arcgis.gis.Item` objects will have all of their layers individually
                               added to the map widget.
        ------------------     --------------------------------------------------------------------
        options                Optional dict. Specify visualization options such as renderer info,
                               opacity, definition expressions. See example below
        ==================     ====================================================================

        .. warning::
            Calling ``MapView.add_layer()`` on an :class:`~arcgis.raster.Raster` instance
            has the following limitations:

            - Local raster overlays do not persist beyond the notebook session on
              published web maps/web scenes -- you would need to separately publish
              these local rasters.

            - The entire raster image data is placed on the MapView's canvas with
              no performance optimizations. This means no pyramids, no dynamic
              downsampling, etc. Please be mindful of the size of the local raster
              and your computer's hardware limitations.

            - Pixel values and projections are not guaranteed to be accurate,
              especially when the local raster's Spatial Reference doesn't
              reproject accurately to Web Mercator (what the ``MapView``
              widget uses).

        .. code-block:: python

            # USAGE EXAMPLE: Add a feature layer with smart mapping renderer and
            # a definition expression to limit the features drawn.
            map1 = gis.map("Seattle, WA")
            map1.add_layer(wa_streets_feature_layer, {'renderer':'ClassedSizeRenderer',
                                                      'field_name':'DistMiles',
                                                      'opacity':0.75})

        """
        if options is None:
            options = {}
        if isinstance(item, arcgis.features.FeatureLayer) and "renderer" not in options:
            renderer_dict = json.loads(item.renderer.json)
            if "renderer" in renderer_dict.keys():
                options.update(renderer_dict)
            else:
                options["renderer"] = renderer_dict
        elif (
            isinstance(item, pd.DataFrame)
            and "renderer" not in options
            and _is_geoenabled(item)
        ):
            item = item.spatial.to_feature_collection()
        self._add_layer_to_widget(item, options)

        # Only for imagery layers
        if isinstance(item, ImageryLayer) and "opacity" in options:
            # Extra steps because the layer opacity will only update after renderering on
            # the widget. Weird behavior with no other solution found.
            wm_layer = dict(self.webmap.layers[-1])  # last layer added
            self.update_layer(wm_layer)

    def _add_layer_to_webmap(self, item, options):
        webmap_options = dict(options)
        webmap_options["layerId"] = self._get_hash(item)
        self.webmap.add_layer(item, webmap_options)

    def _add_layer_to_widget(self, item, options):
        from arcgis.mapping import MapImageLayer, VectorTileLayer
        from arcgis.mapping.ogc._base import BaseOGC

        self._update_time_extent_if_applicable(item)

        if isinstance(item, Raster):
            if isinstance(item._engine_obj, _ImageServerRaster):
                item = item._engine_obj
            elif isinstance(item._engine_obj, _ArcpyRaster):
                out = self._raster.overlay(item)
                self._add_to_hashed_layers(item)
                return out

        if isinstance(item, Item):
            try:
                if hasattr(item, "layers"):
                    if item.layers is None:
                        log.warning(
                            "Item.layers is a 'NoneType' object: nothing to be added to map"
                        )
                    else:
                        for layer in item.layers:
                            self.add_layer(layer, options)
                            # self._add_layer_to_widget(layer, options)
            except KeyError:
                log.warning("No 'layers' in Item: will not be added to map")
        elif isinstance(item, Layer):
            self._add_layer_to_webmap(item, options)
            _lyr = _make_jsonable_dict(item._lyr_json)
            if ("type" in _lyr and _lyr["type"] == "MapImageLayer") and (
                "TilesOnly" in item.properties.capabilities
            ):
                # If it's a TilesOnly MapImageLayer, switch to TiledService
                _lyr["type"] = "ArcGISTiledMapServiceLayer"
            if "options" in _lyr:
                lyr_options = json.loads(_lyr["options"])
                lyr_options.update(options)
                _lyr["options"] = lyr_options
            else:
                _lyr["options"] = options

            _lyr["_hashFromPython"] = self._get_hash(item)
            self._add_notype_layer(item, _lyr, True)
        elif isinstance(item, pd.DataFrame):
            if hasattr(item, "spatial"):
                self.add_layer(item.spatial.to_featureset())
            else:
                raise Exception(
                    "Could not add DataFrame to map it is not a Spatially Enabled DataFrame"
                )
        elif isinstance(item, FeatureSet):
            fset_symbol = options["symbol"] if options and "symbol" in options else None
            fc = FeatureCollection.from_featureset(item, symbol=fset_symbol)
            self._add_layer_to_widget(fc, options)

        elif isinstance(item, dict):
            """Must have 'type' and 'url' in keys. Adds to notype_layers"""
            if ("type" not in item) or ("url" not in item):
                raise Exception("dict layers must have 'type' and 'url'")
            if "renderer" in options:
                item["renderer"] = options["renderer"]
            added_successful = False
            try:
                if item["type"] == "FeatureLayer":
                    layer = FeatureLayer(item.pop("url"))
                    self._add_layer_to_widget(layer, item)
                elif item["type"] == "ImageryLayer":
                    layer = ImageryLayer(item.pop("url"))
                    self._add_layer_to_widget(layer, item)
                elif item["type"] in ["MapImageLayer", "TileLayer"]:
                    mil_item = MapImageLayer(item.pop("url"))
                    for layer in mil_item:
                        self._add_layer_to_widget(layer, item)
                elif item["type"] == "VectorTileLayer":
                    layer = VectorTileLayer(item.pop("url"))
                    self._add_layer_to_widget(layer, item)
            except:
                pass
            else:
                added_successful = True
            finally:
                if not added_successful:
                    item["_hashFromPython"] = self._get_hash(item)
                    self._add_notype_layer(item, item, True)
        elif isinstance(item, BaseOGC):
            self._add_layer_to_webmap(item, options)
            _lyr = _make_jsonable_dict(item._lyr_json)
            _lyr["_hashFromPython"] = self._get_hash(item)
            self._add_notype_layer(item, _lyr, True)
        elif _is_iterable(item):
            # If it's any iterable not previously checked, attempt to infer
            if "layers" in item:
                for layer in item.layers:
                    self._add_layer_to_widget(layer, options)
            else:
                for item_ in item:
                    self._add_layer_to_widget(item_, options)
        else:
            raise RuntimeError("Cannot infer layer: will not be added to map")

    def _add_notype_layer(self, item, lyr_json, new):
        # Add the original item to the hashed layers
        if new is True:
            self._add_to_hashed_layers(item)
        # but draw the json representation
        else:
            # remove old representation otherwise both will appear on map
            # applied for update layer
            layers = list(self._draw_these_notype_layers_on_widget_load)
            for layer in layers:
                if layer["_hashFromPython"] == lyr_json["_hashFromPython"]:
                    layers.remove(layer)
            self._draw_these_notype_layers_on_widget_load = tuple(layers)
        self._draw_these_notype_layers_on_widget_load += (lyr_json,)
        if self.ready:
            self._add_this_notype_layer = {}
            self._add_this_notype_layer = lyr_json

    def update_layer(self, layer: Union[dict, FeatureLayer]):
        """
        To update the layer dictionary for a layer in the map. For example, to update the renderer dictionary for a layer
        and have it by dynamically changed on the mapview.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        layer                   Required Feature Layer or Feature Layer dictionary.
                                The existing mapview layer with updated properties.
                                In order to get a layer on the mapview, use the ``layers`` method and
                                assign the output to a value. Make edits on the this value and pass
                                it in as a dict to update the rendering on the map.

                                .. warning::
                                    If the itemId of the feature layer is changed, this will not work.
        ==================      ====================================================================

        .. code-block:: python

            # Create a mapview and add layer
            map1 = gis.map("Oregon")
            map1.add_layer(<fl_to_add>)

            # Get a layer and edit the color
            fl = map1.layers[0]
            fl.renderer.symbol.color = [0, 204, 204, 255]

            # Update the layer to see it render on map
            map1.update_layer(fl)

            # Save with the updates
            wm_properties= {"title": "Test Update", "tags":["update_layer"], "snippet":"Updated a layer and now save"}
            map1.save(wm_properties)
        """
        if isinstance(layer, dict):
            layer = json.loads(json.dumps(layer))
        # Update the webmap part
        self.webmap.update_layer(layer)

        # Update the mapview part
        self._webmap = self.webmap._webmapdict

    def remove_layers(self, layers=None):
        """
        The ``remove_layers`` method removes the layers added to the map widget.

        .. note::
            A list of layers added to the widget can be retrieved by querying the
        :attr:`~arcgis.widgets.MapView.layers` property.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        layers                 Optional list. Specify the list of layers to be removed from the map widget. You can get
                               the list of layers on the map by querying the 'layers' property.
                               If not specified, it removes all layers added to the widget.
        ==================     ====================================================================

        :return:
            True if layer is successfully removed. Else, False.
        """
        # Developer notes: infer all the layer types the user passed in.
        # Remove everything if the user didn't specify. Then, look up the hash
        # for each layer, remove it from the python side, trigger the removal
        # from the JS side

        output_bool = True
        if layers is None:
            layers = self.layers
        else:
            layers = self._infer_layers(layers)
        layer_hashes_to_remove = []
        for layer in layers:
            hash_ = self._get_hash(layer)
            if hash_ in self._hashed_layers:
                # Remove layer from py
                popped_layer = self._hashed_layers.pop(hash_, None)
                if isinstance(popped_layer, Raster):
                    output_bool = self._raster.remove(popped_layer)
                else:
                    # stage remove from js
                    layer_hashes_to_remove.append(hash_)
                    # Remove layer from webmap
                    for wm_layer in self.webmap.layers:
                        if wm_layer["id"] == hash_:
                            self.webmap.remove_layer(wm_layer)
            else:
                log.warning("Could not find layer {} in layers".format(layer))
                output_bool = False

        # Layer is removed from python side: trigger removal from JS side
        self._layers_to_remove = tuple("nonexistant_layer_id")
        self._layers_to_remove = tuple(layer_hashes_to_remove)

        return output_bool

    def _infer_layers(self, arg):
        """For a generic list of Layers, Items, FeatureSets, or an individual
        'layer', or anything, attempt to return a list of of 'layer' types
        that would exist in self.layers
        """
        from arcgis.mapping.ogc._base import BaseOGC

        output_layers = []
        if isinstance(arg, Raster):
            if isinstance(arg._engine_obj, _ImageServerRaster):
                arg = arg._engine_obj
            elif isinstance(arg._engine_obj, _ArcpyRaster):
                output_layers.append(arg)
                return output_layers

        if isinstance(arg, Layer):
            output_layers.append(arg)
        elif isinstance(arg, ImageryLayer):
            output_layers.append(arg)
        elif isinstance(arg, BaseOGC):
            output_layers.append(arg)
        elif isinstance(arg, Item):
            for layer in arg.layers:
                output_layers.append(layer)
        elif isinstance(arg, FeatureSet):
            fc = FeatureCollection.from_featureset(arg)
            for layer in fc["layers"]:
                output_layers.append(layer)
        elif isinstance(arg, dict):
            output_layers.append(arg)
        elif _is_iterable(arg):
            # If it's any iterable not previously checked, attempt to infer
            if hasattr(arg, "layers"):
                for layer in arg.layers:
                    output_layers.append(layer)
            else:
                for item_ in arg:
                    inferred_item = self._infer_layers(item_)[0]
                    output_layers.append(inferred_item)
        else:
            raise RuntimeError("Cannot infer layer {}: will not be removed".format(arg))
        return output_layers

    def _add_to_hashed_layers(self, item):
        hash_ = self._get_hash(item)
        self._hashed_layers[hash_] = item

    def _remove_from_notype_layers(self, layer):
        layers_to_test = list(self.layers)
        for i in range(0, len(layers_to_test)):
            layer_to_test = layers_to_test[i]
            if layer_to_test["_hashFromPython"] == self._get_hash(
                layer
            ):  # ['_hashFromPython']:
                layers_to_test.pop(i)
        self._notype_layers = tuple(layers_to_test)

    def _get_hash(self, item):
        """Returns the str representation of the hash of any supported obj"""
        if isinstance(item, dict) and "_hashFromPython" in item:
            return item["_hashFromPython"]
        if self._is_hashable(item):
            return str(hash(item))
        else:
            if isinstance(item, dict):
                return str(hash(frozenset(item)))
            elif is_numpy_array(item):
                return get_hash_numpy_array(item)
            elif isinstance(item, Raster):
                return str(hash(item.catalog_path))
            elif isinstance(item, ImageryLayer):
                return str(hash(item.url))
            else:
                try:
                    return str(hash(item.url))
                except Exception:
                    raise Exception("Cannot hash item {}".format(item))

    def _is_hashable(self, item):
        try:
            hash(item)
            return True
        except Exception:
            return False

    def display_message(self, msg):
        """
        The ``display_message`` method displays a message on the upper-right corner of the ``map widget``.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        msg                    A required String. The message to be displayed.
        ==================     ====================================================================
        .. note::
            Only one message can be sent at a time, multiple messages do not show up.
        """
        self._custom_msg = ""
        self._custom_msg = msg

    def save(
        self,
        item_properties,
        mode=None,
        thumbnail=None,
        metadata=None,
        owner=None,
        folder=None,
    ):
        """
        The ``save`` method saves the map widget object into a new web map :class:`~arcgis.gis.Item` or a new web scene
        item in your GIS.

        .. note::
            If you started out with a fresh map widget object, use this method
            to save it as a the webmap/webscene item in your :class:`~arcgis.gis.GIS`.
            If you started with a map widget object from an existing
            webmap/webscene object, calling this method will create a new item
            with your changes. If you want to update the existing item with your
            changes, call the :attr:`~arcgis.widgets.MapView.update` method instead.

        .. note::
            Saving as a WebScene item only works in a Jupyter environment:
            the map must be visually displayed in the notebook before
            calling this method.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_properties     Required dictionary. See table below for the keys and values.
        ---------------     --------------------------------------------------------------------
        mode                Optional string. Whether to save this map instance as a 2D WebMap,
                            or a 3D WebScene. Possible strings: "2D", "webmap", "3D", or
                            "webscene".
        ---------------     --------------------------------------------------------------------
        thumbnail           Optional string. Either a path or URL to a thumbnail image.
        ---------------     --------------------------------------------------------------------
        metadata            Optional string. Either a path or URL to the metadata.
        ---------------     --------------------------------------------------------------------
        owner               Optional string. User object corresponding to the desired owner of
                            this item. Defaults to the logged in user.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Name of the folder where placing item.
        ===============     ====================================================================

        *Key:Value Dictionary Options for Parameter item_properties*

        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        typeKeywords       Optional string. Provide a lists all sub-types, see URL 1 below for
                           valid values.
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of
                           strings. Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250
                           characters) of the what the item is.
        -----------------  ---------------------------------------------------------------------
        accessInformation  Optional string. Information on the source of the content.
        -----------------  ---------------------------------------------------------------------
        licenseInfo        Optional string.  Any license information or restrictions regarding
                           the content.
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Locale, country and language information.
        -----------------  ---------------------------------------------------------------------
        access             Optional string. Valid values are private, shared, org, or public.
        -----------------  ---------------------------------------------------------------------
        commentsEnabled    Optional boolean. Default is true, controls whether comments are
                           allowed (true) or not allowed (false).
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Language and country information.
        =================  =====================================================================

        :return:
            Item object corresponding to the new web map Item created.

        .. code-block:: python

           USAGE EXAMPLE: Save map widget as a new web map item in GIS
           map1 = gis.map("Italy")
           map1.add_layer(Italy_streets_item)
           map1.basemap = 'dark-gray-vector'
           italy_streets_map = map1.save({'title':'Italy streets',
                                        'snippet':'Arterial road network of Italy',
                                        'tags':'streets, network, roads'})
        """
        if mode == None:
            mode = self.mode
        if mode == "2D" or "webmap" in mode.lower():
            self.mode = "2D"
            return self._save_as_webmap(
                item_properties=item_properties,
                thumbnail=thumbnail,
                metadata=metadata,
                owner=owner,
                folder=folder,
            )
        elif mode == "3D" or "webscene" in mode.lower():
            return self._save_as_webscene(
                item_properties,
                thumbnail=thumbnail,
                metadata=metadata,
                owner=owner,
                folder=folder,
            )
        else:
            raise RuntimeError(
                "'mode' arg must be one of '2D', 'webmap'," " '3D', or 'webscene'"
            )

    def _save_as_webmap(
        self,
        item_properties,
        thumbnail=None,
        metadata=None,
        owner=None,
        folder=None,
    ):
        from arcgis.mapping import WebMap

        self.mode = "2D"
        self._check_item_properties(item_properties)
        if ("basemap" in self._readonly_webmap_from_js) and (
            "baseMapLayers" in self._readonly_webmap_from_js["basemap"]
        ):
            self.webmap._basemap["baseMapLayers"] = self._readonly_webmap_from_js[
                "basemap"
            ]["baseMapLayers"]
        if self.extent:
            self.webmap._extent = self.extent
        self._update_webmap_layers_from_js()
        item = self.webmap.save(item_properties, thumbnail, metadata, owner, folder)
        self.webmap.item = item
        return item

    def _update_webmap_layers_from_js(self):
        if self.ready:
            self._check_for_renderer_updates()
            self._check_for_drawn_layers()

    def _check_for_renderer_updates(self):
        js_layers = {
            layer["id"]: layer for layer in self._readonly_webmap_from_js["layers"]
        }
        for wm_layer in self.webmap.layers:
            wm_layer_id = wm_layer["id"]
            if wm_layer_id in js_layers:
                js_layer = js_layers[wm_layer_id]  # js representation of wm layer
                if "renderer" in js_layer and js_layer["renderer"]:
                    renderer = js_layer["renderer"]
                    id = js_layer["id"]
                    self._apply_renderer_to_webmap_layer_id(renderer, id)

    def _apply_renderer_to_webmap_layer_id(self, renderer, id):
        # TODO: find more elegant solution to solve this problem
        index = 0
        for layer in self.webmap._webmapdict["operationalLayers"]:
            if layer["id"] == id:
                wm_layer = self.webmap._webmapdict["operationalLayers"][index]
                if "featureCollection" in wm_layer:
                    wm_layer = wm_layer["featureCollection"]["layers"][0]
                wm_layer["layerDefinition"]["drawingInfo"] = {"renderer": renderer}
            else:
                index += 1

    def _check_for_drawn_layers(self):
        for layer in self._readonly_webmap_from_js["layers"]:
            if ("graphics" in layer) and (len(layer["graphics"]) > 0):
                for graphic in layer["graphics"]:
                    # Infer what type of JS geometry it is
                    if "shape" not in graphic and "geometry" in graphic:
                        geom = Geometry(graphic["geometry"])
                    elif graphic["shape"] == "polyline":
                        geom = Polyline(graphic["geometry"])
                    elif graphic["shape"] == "polygon":
                        geom = Polygon(graphic["geometry"])
                    elif graphic["shape"] == "multipoint":
                        geom = MultiPoint(graphic["geometry"])
                    elif graphic["shape"] == "point":
                        geom = Point(graphic["geometry"])
                    elif "geometry" in graphic:
                        geom = Geometry(graphic["geometry"])
                    else:
                        log.warning(
                            "Graphic unsupported, not adding to webmap."
                            " {}".format(graphic)
                        )
                        continue

                    if not self._check_if_graphic_already_saved(geom):
                        # Create a python object from the JS geometry
                        feat = Feature(geom)
                        fset = FeatureSet([feat])

                        # Add to webmap
                        self.webmap.add_layer(
                            fset,
                            {"title": "Notes from ArcGIS API for Python"},
                        )

    def _check_if_graphic_already_saved(self, geom):
        for layer in self.webmap.layers:
            for fset in layer["featureCollection"]["layers"]:
                for feat in fset["featureSet"]["features"]:
                    wm_geom = Geometry(feat["geometry"])
                    if wm_geom.equals(geom):
                        return True
        return False

    def _save_as_webscene(
        self,
        item_properties,
        thumbnail=None,
        metadata=None,
        owner=None,
        folder=None,
    ):
        self.mode = "3D"
        self._check_item_properties(item_properties)
        item_properties["type"] = "Web Scene"
        if self.webscene_item:
            item_properties["text"] = self.webscene_item.get_data()
        else:
            item_properties["text"] = self._default_webscene_text_property

        self.webscene_item = self.gis.content.add(
            item_properties,
            thumbnail=thumbnail,
            metadata=metadata,
            owner=owner,
            folder=folder,
        )
        self._trigger_webscene_save_to_this_portal_id = self.webscene_item.id
        return self.webscene_item

    def _check_item_properties(self, item_properties):
        if (
            ("title" not in item_properties)
            or ("snippet" not in item_properties)
            or ("tags" not in item_properties)
        ):
            raise RuntimeError(
                "title, snippet and tags are required in " "item_properties dictionary"
            )

    def update(self, mode=None, item_properties=None, thumbnail=None, metadata=None):
        """
        The ``update`` method updates the WebMap/Web Scene item that was used to create the ``MapWidget``
        object. In addition, you can update other item properties, thumbnail
        and metadata.

        .. note::
            If you started out a MapView object from an existing
            webmap/webscene item, use this method to update the
            webmap/webscene item in your with your changes.
            If you started out with a fresh :class:`~arcgis.widgets.MapView` object (without a
            webmap/webscene item), calling this method will raise a
            RuntimeError exception. If you want to save the map widget into a
            new item, call the :attr:`~arcgis.widgets.MapView.save` method instead.
            For ``item_properties``, pass in arguments for only the properties you
            want to be updated. All other properties will be untouched.  For
            example, if you want to update only the item's description, then
            only provide the description argument in item_properties.

        .. note::
            Saving as a ``WebScene`` item only works in a Jupyter environment: the
            map must be visually displayed in the notebook before calling this
            method.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_properties     Optional dictionary. See table below for the keys and values.
        ---------------     --------------------------------------------------------------------
        mode                Optional string. Whether to save this map instance as a 2D WebMap,
                            or a 3D WebScene. Possible strings: ``2D``, ``webmap``, ``3D``, or
                            ``webscene``.
        ---------------     --------------------------------------------------------------------
        thumbnail           Optional string. Either a path or URL to a thumbnail image.
        ---------------     --------------------------------------------------------------------
        metadata            Optional string. Either a path or URL to the metadata.
        ===============     ====================================================================

        *Key:Value Dictionary Options for Parameter item_properties*

        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        typeKeywords       Optional string. Provide a lists all sub-types, see URL 1 below for
                           valid values.
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of
                           strings. Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250
                           characters) of the what the item is.
        -----------------  ---------------------------------------------------------------------
        accessInformation  Optional string. Information on the source of the content.
        -----------------  ---------------------------------------------------------------------
        licenseInfo        Optional string.  Any license information or restrictions regarding
                           the content.
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Locale, country and language information.
        -----------------  ---------------------------------------------------------------------
        access             Optional string. Valid values are private, shared, org, or public.
        -----------------  ---------------------------------------------------------------------
        commentsEnabled    Optional boolean. Default is true, controls whether comments are
                           allowed (true) or not allowed (false).
        =================  =====================================================================

        :return:
           A boolean indicating success (True) or failure (False).

        .. code-block:: python

           USAGE EXAMPLE: Interactively add a new layer and change the basemap of an existing web map.
           italy_streets_item = gis.content.search("Italy streets", "Web Map")[0]
           map1 = MapView(item = italy_streets_item)
           map1.add_layer(Italy_streets2)
           map1.basemap = 'dark-gray-vector'
           map1.update(thumbnail = './new_webmap.png')

        """
        if mode == None:
            mode = self.mode
        if mode == "2D" or "webmap" in mode.lower():
            self.mode = "2D"
            return self._update_as_webmap(
                item_properties=item_properties,
                thumbnail=thumbnail,
                metadata=metadata,
            )
        elif mode == "3D" or "webscene" in mode.lower():
            self.mode = "3D"
            return self._update_as_webscene(
                item_properties=item_properties,
                thumbnail=thumbnail,
                metadata=metadata,
            )

    def _update_as_webmap(self, item_properties, thumbnail, metadata):
        from arcgis.mapping import WebMap

        if not self.webmap.item:
            raise RuntimeError(
                "Webmap Item object missing. You should use "
                "`save()` to save a new web scene item"
            )
        self.mode = "2D"
        if "basemap" in self._readonly_webmap_from_js:
            self.webmap._basemap["baseMapLayers"] = self._readonly_webmap_from_js[
                "basemap"
            ]["baseMapLayers"]
        self.webmap._extent = self.extent
        self._update_webmap_layers_from_js()
        return self.webmap.item.update(
            item_properties=item_properties,
            thumbnail=thumbnail,
            metadata=metadata,
            data=self.webmap._webmapdict,
        )

    def _update_as_webscene(self, item_properties, thumbnail, metadata):
        if not self.webscene_item:
            raise RuntimeError(
                "Webscene Item object missing. You should use "
                "`save()` to save a new web scene item"
            )
        self.mode = "3D"
        result = self.webscene_item.update(
            item_properties=item_properties,
            thumbnail=thumbnail,
            metadata=metadata,
            data=self.webscene_item.get_data(),
        )
        self._trigger_webscene_save_to_this_portal_id = ""
        self._trigger_webscene_save_to_this_portal_id = self.webscene_item.id
        return result

    def export_to_html(
        self,
        path_to_file,
        title="Exported ArcGIS Map Widget",
        credentials_prompt=False,
    ):
        """
        The ``export_to_html`` method takes the current state of the map widget and exports it to a
        standalone HTML file that can be viewed in any web browser.

        By default, only publicly viewable layers will be visible in any
        exported html map. Specify ``credentials_prompt=True`` to have a user
        be prompted for their credentials when opening the HTML page to view
        private content.

        .. warning::
            Follow best security practices when sharing any HTML page that
            prompts a user for a password.

        .. note::
            You cannot successfully authenticate if you open the HTML page in a
            browser locally like file://path/to/file.html. The credentials
            prompt will only properly function if served over a HTTP/HTTPS
            server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        path_to_file           Required string. The path to save the HTML file on disk.
        ------------------     --------------------------------------------------------------------
        title                  Optional string. The HTML title tag used for the HTML file.
        ------------------     --------------------------------------------------------------------
        credentials_prompt     Optional boolean, default ``False``. If set to ``True``, will display a
                               credentials prompt on HTML page load for users to authenticate and
                               view private content.
        ==================     ====================================================================
        """
        # Make sure height is to 100% so exported page loads in fullscreen
        prev_height = self.layout.height
        self.layout.height = "100%"

        # Set the _auth_mode
        prev_auth_mode = self._auth_mode
        if credentials_prompt:
            self._auth_mode = "prompt"
        else:
            self._auth_mode = "anonymous"

        # 'extent' is a property with separate readonly/writeonly fields
        # The below statement makes the readonly match the writeonly
        # This makes sure the current view is exactly what is saved in the
        # model for the exported HTML
        self.extent = self.extent

        # Write the HTML file
        embed_minimal_html(path_to_file, views=[self], title=title)

        # Restore the previous model attributes that were changed
        self._auth_mode = prev_auth_mode
        self.layout.height = prev_height
        return True

    def draw(self, shape, popup=None, symbol=None, attributes=None):
        """
        The ``draw`` method draws a shape on the map widget.

        .. note::
            Anything can be drawn from known :class:`~arcgis.geometry.Geometry` objects, coordinate pairs, and
            :class:`~arcgis.features.FeatureSet` objects.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        shape                  Required object.
                               Known :class:`~arcgis.geometry.Geometry` objects:
                               Shape is one of [``circle``, ``ellipse``, :class:`~arcgis.geometry.Polygon`,
                               :class:`~arcgis.geometry.Polyline`,
                               :class:`~arcgis.geometry.MultiPoint`, :class:`~arcgis.geometry.Point`,
                               ``rectangle``, ``triangle``].

                               Coordinate pair: specify shape as a list of [lat, long]. Eg: [34, -81]

                               FeatureSet: shape can be a :class:`~arcgis.features.FeatureSet` object.

                               Dict object representing a geometry.
        ------------------     --------------------------------------------------------------------
        popup                  Optional dict. Dict containing ``title`` and ``content`` as keys that will be displayed
                               when the shape is clicked. In case of a :class:`~arcgis.features.FeatureSet`,
                               ``title`` and ``content`` are names of
                               attributes of the features in the ``FeatureSet`` instead of actual string values for
                               ``title`` and ``content`` .
        ------------------     --------------------------------------------------------------------
        symbol                 Optional dict. See the `Symbol Objects <http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#//02r3000000n5000000>`_
                               page in the ArcGIS REST API documentation for more information. A
                               default symbol is used if one is not specified.

                               .. note::
                               A helper utility to get the symbol format for several predefined symbols is
                               available at the `Esri symbol selector. <http://esri.github.io/arcgis-python-api/tools/symbol.html>`_
        ------------------     --------------------------------------------------------------------
        attributes             Optional dict. Specify a dict containing name value pairs of fields and field values
                               associated with the graphic.
        ==================     ====================================================================

        .. code-block:: python

            #Usage Example: Drawing two Geometry objects on a map widget.

            >>> from arcgis.gis import GIS, Item
            >>> from arcgis.widgets import MapView
            >>> from arcgis.geometry import Geometry, Polygon

            >>> geom = Geometry({
                            "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                                        [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                                        [-97.06326,32.759]]],
                            "spatialReference" : {"wkid" : 4326}
                            })
            >>> map2 = gis.map("Arlington, Texas")
            >>> map2.draw(shape=geom)
            >>> map2
            <Map Widget Displayed with the drawn Polygons>

        """

        title = (
            attributes["title"]
            if attributes and "title" in attributes
            else "Notebook sketch layer"
        )

        if isinstance(shape, list) and len(shape) == 2:  #  [lat, long] pair
            shape = {
                "x": shape[1],
                "y": shape[0],
                "spatialReference": {"wkid": 4326},
                "type": "point",
            }

        elif isinstance(shape, tuple):  #  (lat, long) pair
            shape = {
                "x": shape[1],
                "y": shape[0],
                "spatialReference": {"wkid": 4326},
                "type": "point",
            }

        elif isinstance(shape, dict) and "location" in shape:  #  geocode loc.
            shape = {
                "x": shape["location"]["x"],
                "y": shape["location"]["y"],
                "spatialReference": {"wkid": 4326},
                "type": "point",
            }

        if isinstance(shape, FeatureSet) or isinstance(shape, dict):
            fset = None
            if isinstance(shape, FeatureSet):
                fset = shape
                self._draw_featureset(fset, popup, symbol)
            elif isinstance(shape, dict):
                if "type" not in shape:
                    if "rings" in shape:
                        shape["type"] = "polygon"
                    elif "paths" in shape:
                        shape["type"] = "polyline"
                    elif "points" in shape:
                        shape["type"] = "multipoint"
                    else:
                        shape["type"] = "point"

                switcher = {
                    "polygon": "esriGeometryPolygon",
                    "polyline": "esriGeometryPolyline",
                    "multipoint": "esriGeometryMultipoint",
                    "point": "esriGeometryPoint",
                }
                geometry_kind = switcher.get(shape["type"])
                if geometry_kind is None:
                    geometry_kind = "esriGeometryNull"

                graphic = {
                    "geometry": shape,
                    "popupTemplate": popup,
                    "symbol": symbol,
                    "attributes": attributes,
                }
                self._add_graphic(graphic)
                f = Feature(shape)
                fset = FeatureSet(
                    [f],
                    geometry_type=geometry_kind,
                    spatial_reference={"wkid": 4326},
                )

            # Now that the `fset` is set, add to webmap
            if popup:
                webmap_popup = {
                    "title": "{" + popup["title"] + "}",
                    "description": "{" + popup["content"] + "}",
                }
            else:
                webmap_popup = None
            wm_options = {
                "popup": webmap_popup,
                "symbol": symbol,
                "attributes": attributes,
                "title": title,
                "extent": self.extent,
            }
            self.webmap.add_layer(fset, wm_options)

        else:  # User passed in a string for interactive draw mode
            if symbol:
                draw_options = {"shape": shape, "symbol": symbol}
            else:
                draw_options = {"shape": shape}
            self._trigger_interactive_draw_mode_for = {}
            self._trigger_interactive_draw_mode_for = draw_options

    def _draw_featureset(self, fset, popup, symbol):
        # FeatureSet needs special case
        graphics = []
        for feature in fset.features:
            graphic = self._get_graphic_from_feature(feature, popup, symbol)
            graphics.append(graphic)
            if self.ready:
                self._add_this_graphic = {}
                self._add_this_graphic = graphic
        self._draw_these_graphics_on_widget_load += tuple(graphics)

    def _get_graphic_from_feature(self, feature, popup, symbol):
        if popup:
            popup_dict = {
                "title": feature.attributes[popup["title"]],
                "content": feature.attributes[popup["content"]],
            }
        else:
            popup_dict = None
        geometry = feature.geometry
        geometry["type"] = feature.geometry_type.lower()
        graphic = {
            "geometry": feature.geometry,
            "popupTemplate": popup_dict,
            "symbol": symbol,
            "attributes": feature.attributes,
        }
        return _make_jsonable_dict(graphic)

    def _add_graphic(self, graphic):
        if self.ready:
            self._add_this_graphic = {}
            self._add_this_graphic = _make_jsonable_dict(graphic)
        else:
            self._draw_these_graphics_on_widget_load += (_make_jsonable_dict(graphic),)

    def clear_graphics(self):
        """
        The ``clear_graphics`` method clear the graphics drawn on the map widget.

        .. note::
            Graphics are shapes drawn using the :attr:`~arcgis.widgets.MapView.draw` method.

        .. code-block:: python

            #Usage Example: Drawing two Geometry objects on a map widget.

            >>> from arcgis.gis import GIS, Item
            >>> from arcgis.widgets import MapView
            >>> from arcgis.geometry import Geometry, Polygon

            >>> geom = Geometry({
                            "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                                        [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                                        [-97.06326,32.759]]],
                            "spatialReference" : {"wkid" : 4326}
                            })
            >>> map2 = gis.map("Arlington, Texas")
            >>> map2.draw(shape=geom)
            >>> map2
            <Map Widget Displayed with the drawn Polygons>
            >>> map2.clear_graphics()
            >>> map2
            <Map Widget Displayed without the drawn Polygons>
        """
        self._layers_to_remove = ("nonexistant_layer_id",)
        # All graphics are saved to a layer with the below id
        self._layers_to_remove = ("graphicsLayerId31195",)

    def on_draw_end(self, callback, remove=False):
        """
        The ``on_draw_end`` method registers a callback to execute when something is drawn.
        The callback will be called with two arguments:
        1. The clicked widget instance
        2. The drawn geometry

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        remove                 Optional boolean. Set to true to remove the callback from the list
                               of callbacks.
        ==================     ====================================================================

        """
        self._draw_end_handlers.register_callback(callback, remove=remove)

    def on_click(self, callback, remove=False):
        """
        The ``on_click`` method registers a callback to execute when the map is clicked.

        .. note::
            The callback will be called with one argument, the clicked widget instance.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        remove                 Optional boolean. Set to true to remove the callback from the list
                               of callbacks.
        ==================     ====================================================================

        """
        self._click_handlers.register_callback(callback, remove=remove)

    def toggle_window_view(self, title="ArcGIS Map", tab_mode=None):
        """
        In a JupyterLab environment, calling ``toggle_window_view`` will separate
        the drawn map widget to a new window next to the open notebook,
        allowing you to move the widget to it, split it, put it in a new tab, etc.
        If the widget is already separated in a new window, calling this
        function will restore the widget to the notebook where it originated
        from.

        .. note::
            See the `JupyterLab guide pages <https://developers.arcgis.com/python/guide/using-the-jupyter-lab-environment/>`_
            for more information.

        .. raw:: html

            <p>Note that this functionality can also be achieved by pressing
            the <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAA
            gCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAA
            AAB3RJTUUH4gcUAAIIxu7mQQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoI
            EdJTVBkLmUHAAABI0lEQVRYw+1Wy3GDMBTch3JLBanA+TQR0gDz4JB0Y1xISkBAA4Eq
            HOeYU6qAzYVkPAyWYSLii/Ymzb63q89qBAQEXBgyNamquYhs5zSw1v72SNOUp3gkd2V
            Z5uP5K1dzkrslqxnzRWQD4MVV4zQw5XguP8uyDckGwBeAm1M10Rrnqqq3fd+3JEVEYh
            c3WkMcQDMM46IoPv7NQJIkdyLS/oiXZXk4VxP5FDfGNEvEz17Cuciy7J7kGwB0XRfXd
            X0Yp+NoZ/waGMQbABzE35ekKfIkDhF5mhJf7Q6o6sOReFwUxd7BzdM0ffR5BALglaQA
            iK21eydZZEsSAFpfBkjy2Rhz7Vr5qimoqurTR4JWeYqDgWDAWwpUNb+ogbn/woCAgL/
            gG9knfWwKBGcvAAAAAElFTkSuQmCC"> icon in the widget UI.</p>

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        title                  What text will display as the widget tab. Default: "ArcGIS Map".
        ------------------     --------------------------------------------------------------------
        tab_mode               The 'tab mode' that this window will open in. Will use this
                               MapView instance's `tab_mode` property if not specified. Possible
                               values: "auto", "split-top", "split-left", "split-right",
                               "split-bottom", "tab-before", "tab-after"
        ==================     ====================================================================
        """

        args = {"title": title}
        if tab_mode:
            args["tab_mode"] = tab_mode
        if not self.ready:
            self._ipython_display_()
            # TODO: Find more elegant solution for this time.sleep hack
            time.sleep(1)
        self._trigger_new_jlab_window_with_args = {}
        self._trigger_new_jlab_window_with_args = args

    def _handle_map_msg(self, _, content, buffers):
        """Handle a msg from the front-end.
        Parameters
        ----------
        content: dict
            Content of the msg."""

        if content.get("event", "") == "mouseclick":
            self._click_handlers(self, content.get("message", None))
        if content.get("event", "") == "draw-end":
            self._draw_end_handlers(self, content.get("message", None))

    def zoom_to_layer(self, item, options={}):
        """
        The ``zoom_to_layer`` method snaps the map to the extent of the provided :class:`~arcgis.gis.Item` object(s).

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   The item at which you want to zoom your map to.
                               This can be a single or a list of :class:`~arcgis.gis.Item`, ``Layer`` , ``DataFrame`` ,
                               :class:`~arcgis.features.FeatureSet`, or :class:`~arcgis.features.FeatureCollection`
                               objects.
        ------------------     --------------------------------------------------------------------
        options                Optional set of arguments.

        ==================     ====================================================================
        """
        target_extent = _get_extent(item)
        target_sr = getattr(self, "extent", {}).get("spatialReference", default_sr)
        if isinstance(target_extent, list):
            target_extent = _flatten_list(target_extent)
            if len(target_extent) > 1:
                target_extent = _get_master_extent(target_extent, target_sr)
            else:
                target_extent = target_extent[0]
        if not (target_extent["spatialReference"] == target_sr):
            target_extent = _reproject_extent(target_extent, target_sr)
        if self.ready:
            self.extent = (
                self.extent
            )  # Sometimes setting extent will not work for the same target extent if we do it multiple times, doing this fixes that issue.
            self.extent = target_extent
        else:
            self._extent = target_extent

    # Start time section

    time_slider = Bool(False).tag(sync=True)
    """
    ``time_slider`` is a string property that determines whether a time slider exists for a map widget.
    If set to `True`, will display a time slider in the widget that will
    allow you to visualize temporal data for an applicable layer added to
    the map. Default: `False`.
    
    See the `TimeSlider <https://developers.arcgis.com/javascript/latest/api-reference/esri-widgets-TimeSlider.html#mode>`_
    page in the ArcGIS REST API page for more info.
    """

    time_mode = Unicode("time-window").tag(sync=True)
    """
    ``time_mode`` is the string used for defining if the temporal data will be displayed
    cumulatively up to a point in time, a single instant in time, or
    within a time range.

    Possible values: "instant", "time-window", "cumulative-from-start",
    "cumulative-from-end". Default: "time-window"

    See the `TimeSlider <https://developers.arcgis.com/javascript/latest/api-reference/esri-widgets-TimeSlider.html#mode>`_
    page in the ArcGIS REST API page for more info.
    """

    _time_info = Dict({}).tag(sync=True)

    _writeonly_start_time = Datetime().tag(sync=True)
    _writeonly_start_time.default_value = dt.datetime(1970, 1, 2)
    _readonly_start_time = Unicode("").tag(sync=True)
    """JS can't send `Date` objects -- ISO string of time"""

    @property
    def start_time(self):
        """
        The ``start_time`` property is a representation of a `datetime.datetime` property.
        If `time_mode` == `"time-window"`,
        represents the lower bound 'thumb' of the time slider. For all other
        `time_mode` values, ``start_time`` represents the single thumb on the time slider.
        """
        date_as_iso = dateutil.parser.parse(self._readonly_start_time)
        date_local = date_as_iso.astimezone()
        return date_local

    @start_time.setter
    def start_time(self, value):
        if not isinstance(value, dt.datetime):
            raise Exception("Value must be of type `datetime.datetime`")
        self._writeonly_start_time = dt.datetime(1970, 1, 2)
        self._writeonly_start_time = value

    _writeonly_end_time = Datetime().tag(sync=True)
    _writeonly_end_time.default_value = dt.datetime(1970, 1, 2)
    _readonly_end_time = Unicode("").tag(sync=True)
    """JS can't send `Date` objects -- ISO string of time"""

    @property
    def end_time(self):
        """
        ``end_time`` is a ``datetime.datetime`` property. If ``time_mode`` == ``time-window``
        ``end_time`` represents the upper bound 'thumb' of the time slider. For all other
        ``time_mode`` values, ``end_time`` is not used."""

        date_as_iso = dateutil.parser.parse(self._readonly_end_time)
        date_local = date_as_iso.astimezone()
        return date_local

    @end_time.setter
    def end_time(self, value):
        if not isinstance(value, dt.datetime):
            raise Exception("Value must be of type `datetime.datetime`")
        self._writeonly_end_time = dt.datetime(1970, 1, 2)
        self._writeonly_end_time = value

    def _update_time_extent_if_applicable(self, item):
        try:
            if hasattr(item, "properties") and hasattr(item.properties, "timeInfo"):
                time_info = item.properties.timeInfo
                if hasattr(time_info, "timeExtent"):
                    start_time = dt.datetime.fromtimestamp(
                        time_info.timeExtent[0] / 1000
                    )
                    end_time = dt.datetime.fromtimestamp(time_info.timeExtent[1] / 1000)
                    kwargs = {}
                    if time_info.defaultTimeInterval:
                        kwargs["interval"] = time_info.defaultTimeInterval
                    if time_info.defaultTimeIntervalUnits:
                        item_units = time_info.defaultTimeIntervalUnits.lower()
                        if "mill" in item_units:
                            kwargs["unit"] = "milliseconds"
                        elif "sec" in item_units:
                            kwargs["unit"] = "seconds"
                        elif "minute" in item_units:
                            kwargs["unit"] = "minutes"
                        elif "hour" in item_units:
                            kwargs["unit"] = "hours"
                        elif "day" in item_units:
                            kwargs["unit"] = "days"
                        elif "week" in item_units:
                            kwargs["unit"] = "weeks"
                        elif "month" in item_units:
                            kwargs["unit"] = "months"
                        elif "year" in item_units:
                            kwargs["unit"] = "years"
                        elif "decad" in item_units:
                            kwargs["unit"] = "decades"
                        elif "centur" in item_units:
                            kwargs["unit"] = "centuries"
                    self.set_time_extent(start_time, end_time, **kwargs)
        except Exception:
            pass

    def set_time_extent(self, start_time, end_time, interval=1, unit="milliseconds"):
        """
        The ``set_time_extent`` is called when `time_slider = True` and is the time extent to display on the time
        slider.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        start_time             Required ``datetime.datetime``. The lower bound of the time extent to
                               display on the time slider.
        ------------------     --------------------------------------------------------------------
        end_time               Required ``datetime.datetime``. The upper bound of the time extent to
                               display on the time slider.
        ------------------     --------------------------------------------------------------------
        interval               Optional number, default `1`. The numerical value of the time
                               extent.
        ------------------     --------------------------------------------------------------------
        unit                   Optional string, default `"milliseconds"`. Temporal units. Possible
                               values: `"milliseconds"`, `"seconds"`, `"minutes"`, `"hours"`,
                               `"days"`, `"weeks"`, `"months"`, `"years"`, `"decades"`,
                               `"centuries"`
        ==================     ====================================================================

        """
        if not (
            isinstance(start_time, dt.datetime) and isinstance(end_time, dt.datetime)
        ):
            raise Exception(
                "`start_time` and `end_time` arguments must be of type `datetime.datetime`"
            )
        self._time_info = {
            "time_extent": [start_time, end_time],
            "interval": interval,
            "unit": unit,
        }

    # end time section

    # start local raster overlay section

    def _add_overlay(self, raster_overlay):
        image_dict = {
            "id": raster_overlay.id,
            "src": raster_overlay.img_url,
            "extent": raster_overlay.extent,
            "opacity": raster_overlay.opacity,
        }
        if self.ready:
            self._overlay_this_image = {}
            self._overlay_this_image = image_dict
        else:
            self._overlay_these_images_on_widget_load += (image_dict,)

    def _remove_overlay(self, overlay_id):
        self._image_overlays_to_remove += (overlay_id,)
        if self.ready:
            self._image_overlays_to_remove = tuple()

    def _isinstance(self, *args, **kwargs):
        return isinstance(*args, **kwargs)

    _image_overlays_to_remove = Tuple(tuple()).tag(sync=True)
    _overlay_this_image = Dict({}).tag(sync=True)
    _overlay_these_images_on_widget_load = Tuple(tuple()).tag(sync=True)

    _synced_mapviews = []
    _mapview_uuid_to_dlinks = {}
    _dlinks = []

    def sync_navigation(self, mapview):
        """
        The ``sync_navigation`` method synchronizes the navigation from this :class:`~arcgis.widgets.MapView` to
        another :class:`~arcgis.widgets.MapView` instance so panning/zooming/navigating in one will update the other.

        .. note::
            Users can sync more than two class:`~arcgis.widgets.MapView` instances together. For example, a user can
            sync MapView A to MapView B, MapView B to MapView C, and MapView C to MapView D together and all will be in
            sync. Thus, driving one of these ``MapView`` objects will make the other ``MapView`` objects follow.

        ==================     ===================================================================
        **Parameter**           **Description**
        ------------------     -------------------------------------------------------------------
        mapview                Either a single :class:`~arcgis.widgets.MapView` instance, or a list of ``MapView``
                               instances to synchronize to.
        ==================     ===================================================================

        .. code-block:: python

            # USAGE EXAMPLE: link the navigation of two maps together
            from ipywidgets import HBox
            map1 = gis.map("Chicago, IL")
            map1.basemap = "gray"
            map2 = gis.map("Chicago, IL")
            map2.basemap = "dark-gray"
            map1.sync_navigation(map2)
            HBox([map1, map2])

        """
        if _is_iterable(mapview):
            for m in mapview:
                self.sync_navigation(m)
        elif not self._isinstance(mapview, MapView):
            raise Exception("Can only link navigation to a `MapView` instance")
        else:
            self._sync_navigation(mapview, ignore_errors=False)
            for m in mapview._synced_mapviews:
                m._sync_navigation(self, ignore_errors=True)
            for m in self._synced_mapviews:
                m._sync_navigation(mapview, ignore_errors=True)

    def _sync_navigation(self, mapview, ignore_errors=False):
        try:
            # Check to make sure this call is valid
            if mapview in self._synced_mapviews or self in mapview._synced_mapviews:
                raise Exception(
                    f"Not syncing MapView {mapview} since it is " "already synced"
                )
            if self == mapview:
                return

            # Edge case for when you link widgets before they are drawn
            if not self.ready:
                self._readonly_extent = self._extent
            if not mapview.ready:
                mapview._readonly_extent = mapview._extent

            # Set references to each other
            self._synced_mapviews.append(mapview)
            mapview._synced_mapviews.append(self)

            # Set up, reference, and apply dlinks for each other,
            self_dlinks = []
            their_dlinks = []
            self_dlinks.append(
                ipywidgets.dlink(
                    (self, "_readonly_extent"),
                    (mapview, "_link_writeonly_extent"),
                )
            )
            their_dlinks.append(
                ipywidgets.dlink(
                    (mapview, "_readonly_extent"),
                    (self, "_link_writeonly_extent"),
                )
            )

            self_dlinks.append(
                ipywidgets.dlink(
                    (self, "_readonly_rotation"),
                    (mapview, "_link_writeonly_rotation"),
                )
            )
            their_dlinks.append(
                ipywidgets.dlink(
                    (mapview, "_readonly_rotation"),
                    (self, "_link_writeonly_rotation"),
                )
            )

            self_dlinks.append(
                ipywidgets.dlink(
                    (self, "_readonly_heading"),
                    (mapview, "_link_writeonly_heading"),
                )
            )
            their_dlinks.append(
                ipywidgets.dlink(
                    (mapview, "_readonly_heading"),
                    (self, "_link_writeonly_heading"),
                )
            )

            self_dlinks.append(
                ipywidgets.dlink(
                    (self, "_readonly_tilt"),
                    (mapview, "_link_writeonly_tilt"),
                )
            )
            their_dlinks.append(
                ipywidgets.dlink(
                    (mapview, "_readonly_tilt"),
                    (self, "_link_writeonly_tilt"),
                )
            )
            self._mapview_uuid_to_dlinks[mapview._uuid] = self_dlinks
            mapview._mapview_uuid_to_dlinks[self._uuid] = their_dlinks
            return True
        except Exception as e:
            if ignore_errors:
                return True
            else:
                raise e

    def unsync_navigation(self, mapview=None):
        """
        The ``unsync_navigation`` method unsynchronizes connections made to other :class:`~arcgis.widgets.MapView`
        instances made via `my_mapview.sync_navigation(other_mapview)`.

        ==================     ===================================================================
        **Parameter**           **Description**
        ------------------     -------------------------------------------------------------------
        mapview                (Optional) Either a single `MapView` instance, or a list of
                               `MapView` instances to unsynchronize. If not specified, will
                               unsynchronize all synced `MapView` instances
        ==================     ===================================================================

        """
        if mapview is None:
            mapview = self._synced_mapviews
        if _is_iterable(mapview):
            for m in mapview:
                self.unsync_navigation(m)
        elif not self._isinstance(mapview, MapView):
            raise Exception("Can only unsync navigation to a `MapView` instance")
        else:
            self._unsync_navigation(mapview, ignore_errors=False)
            for m in mapview._synced_mapviews:
                m._unsync_navigation(self, ignore_errors=True)
            for m in self._synced_mapviews:
                m._unsync_navigation(mapview, ignore_errors=True)

    def _unsync_navigation(self, mapview, ignore_errors=False):
        try:
            if (
                mapview not in self._synced_mapviews
                and self not in mapview._synced_mapviews
            ):
                raise Exception(
                    f"Not unsyncing MapView {mapview} since it " "hasn't been synced"
                )
            if self == mapview:
                return

            for self_dlink in self._mapview_uuid_to_dlinks[mapview._uuid]:
                self_dlink.unlink()
            self._synced_mapviews.remove(mapview)
            del self._mapview_uuid_to_dlinks[mapview._uuid]

            for their_dlink in mapview._mapview_uuid_to_dlinks[self._uuid]:
                their_dlink.unlink()
            mapview._synced_mapviews.remove(self)
            del mapview._mapview_uuid_to_dlinks[self._uuid]
            return True
        except Exception as e:
            if ignore_errors:
                return True
            else:
                raise e
