import os
import logging as _logging
from typing import Any, Optional
import arcgis
from arcgis.gis import GIS


_log = _logging.getLogger(__name__)

_use_async = False


def _convert_colorbrewer(cb_list: list, alpha: float = 1):
    """
    Helper to convert JS array derived from custom colorbrewer palette
    into format used by the renderers. See https://colorbrewer2.org/.
    Accepts HEX or RGB versions.

    Example:
    >>> palette_hex = ['#fee6ce','#fdae6b','#e6550d']
    >>> _convert_colorbrewer(palette_hex)
    [[254, 230, 206, 255], [253, 174, 107, 255], [230, 85, 13, 255]]

    >>> palette_rgb = ['rgb(254,230,206)','rgb(253,174,107)','rgb(230,85,13)']
    >>> _convert_colorbrewer(palette_rgb)
    [[254, 230, 206, 255], [253, 174, 107, 255], [230, 85, 13, 255]]

    """

    new_list = []

    # hex format
    if "#" in cb_list[0]:
        for color in cb_list:
            color = color.strip("#")
            length = len(color)
            new_color = [
                int(color[i : i + length // 3], 16)
                for i in range(0, length, length // 3)
            ]
            new_color.append(alpha * 255)
            new_list.append(new_color)

    # rgb format
    else:
        for color in cb_list:
            color = color[4:-1]
            tokens = color.split(",")
            new_color = [int(token) for token in tokens]
            new_color.append(alpha * 255)
            new_list.append(new_color)

    return new_list


def _format_colors(colors, alpha, cstep=None):
    """
    Helper to format any form of color input into something usable by the
    renderers. Strings will be tokenized into colormap names or converted
    to RGB numbers if they are HEX format. Exported colorbrewer lists will
    be converted to RGB lists, RGB lists will be left untouched. Palettable
    palettes will have their .colors attribute converted from tuples to lists.
    Anything else will default to "jet" colormap.

    Example:
    >>> hex_string = '#fee6ce, #fdae6b, #e6550d'
    >>> _format_colors(hex_string, 1)
    [[254, 230, 206, 255], [253, 174, 107, 255], [230, 85, 13, 255]]

    >>> cmap_string = "Spectral, jet, autumn"
    >>> _format_colors(cmap_string, 1)
    ['Spectral', 'jet', 'autumn']

    >>> import palettable.wesanderson as wa
    >>> wes_test = wa.Aquatic1_5
    >>> _format_colors(wes_test, 1)
    [[52, 36, 25, 255],
    [28, 64, 39, 255],
    [241, 201, 14, 255],
    [102, 88, 153, 255],
    [184, 147, 130, 255]]

    >>> premade_list = [[254, 230, 206, 255], [253, 174, 107, 255], [230, 85, 13, 255]]
    >>> _format_colors(premade_list, 1)
    [[254, 230, 206, 255], [253, 174, 107, 255], [230, 85, 13, 255]]

    """
    if colors is None:
        fmt_colors = "jet"
    if isinstance(colors, list):
        # single RGB + Alpha set
        if len(colors) == 4 and all([isinstance(i, int) for i in colors]):
            fmt_colors = [colors]

        # exported colorbrewer JSON array or list of colormaps
        elif all([isinstance(i, str) for i in colors]):
            if all(["#" in i for i in colors]) or all(["rgb(" in i for i in colors]):
                fmt_colors = _convert_colorbrewer(colors, alpha)
            else:
                fmt_colors = colors

        # list of RGB + Alpha set (already valid)
        elif all(
            [
                isinstance(color, list)
                and len(color) == 4
                and all([isinstance(i, int) for i in color])
                for color in colors
            ]
        ):
            fmt_colors = colors

        # doesn't fit any formats, default it
        else:
            fmt_colors = "jet"

    # palettable palette object
    elif hasattr(colors, "colors"):
        if cstep is not None:
            temp_map = colors.mpl_colormap
            color_tuple = temp_map(cstep, bytes=True)
            fmt_colors = [[int(i) for i in color_tuple]]
        else:
            fmt_colors = []
            for color in colors.colors:
                val_list = [val for val in color]
                val_list.append(alpha * 255)
                fmt_colors.append(val_list)

    # any form of string
    elif isinstance(colors, str):
        fmt_colors = colors.replace(" ", "")
        fmt_colors = fmt_colors.split(",")
        if all(["#" in i for i in fmt_colors]):
            fmt_colors = _convert_colorbrewer(fmt_colors, alpha)

    # default
    else:
        fmt_colors = ["jet"]
    return fmt_colors


def _get_list_value(index, array):
    """
    helper operation to loop a list of values regardless of the index value

    Example:
    >>> a = [111,222,333]
    >>> list_loop(15, a)
    111
    """
    if len(array) == 0:
        return None
    elif index >= 0 and index < len(array):
        return array[index]
    return array[index % len(array)]


def create_colormap(
    color_list: list,
    name: str = "temp_cmap",
    bins: int = 256,
):
    """
    The ``create_colormap`` function is a simple function that allows a
    user to make a matplotlib ``LinearSegmentedColormap`` object based on a
    custom assortment of colors. Accepts a list of RGB/RGB + alpha arrays.
    This function is great for visualizing potential color schemes to be used
    in mapping.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    color_list             Required list. List items must be of the format ``[R, G, B]``, where
                           R, G, and B represent values from 0 to 255, or any other format
                           where the first 3 values represent RGB values in bytes.
    ------------------     --------------------------------------------------------------------
    name                   Optional string. The name of the new colormap item. Defaults to
                           "temp_cmap".
    ------------------     --------------------------------------------------------------------
    bins                   Optional int. The number of bins (or uniquely retrievable colors)
                           the new colormap has. Defaults to 256.
    ==================     ====================================================================
    """
    from matplotlib.colors import LinearSegmentedColormap

    if len(color_list) < 2:
        raise ValueError("List must have multiple colors")

    mpl_colors = []
    for color in color_list:
        conv = (color[0] / 255, color[1] / 255, color[2] / 255)
        mpl_colors.append(conv)

    return LinearSegmentedColormap.from_list(name, mpl_colors, bins)


def export_map(
    web_map_as_json: Optional[dict] = None,
    format: str = """PDF""",
    layout_template: str = """MAP_ONLY""",
    gis: Optional[GIS] = None,
    **kwargs,
):
    """
    The ``export_map`` function takes the state of the :class:`~arcgis.mapping.WebMap` object (for example, included services, layer visibility
    settings, client-side graphics, and so forth) and returns either (a) a page layout or
    (b) a map without page surrounds of the specified area of interest in raster or vector format.
    The input for this function is a piece of text in JavaScript object notation (JSON) format describing the layers,
    graphics, and other settings in the web map. The JSON must be structured according to the WebMap specification
    in the ArcGIS Help.

    .. note::
        The ``export_map`` tool is shipped with ArcGIS Server to support web services for printing, including the
        preconfigured service named ``PrintingTools``.


    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    web_map_as_json        Web Map JSON along with export options. See the
                           `Export Web Map Specifications <https://developers.arcgis.com/rest/services-reference/exportwebmap-specification.htm>`_
                           for more information on structuring this JSON.
    ------------------     --------------------------------------------------------------------
    format                 Format (str). Optional parameter.  The format in which the map image
                           for printing will be delivered. The following strings are accepted:

                           For example:
                                PNG8

                           Choice list:
                                ['PDF', 'PNG32', 'PNG8', 'JPG', 'GIF', 'EPS', 'SVG', 'SVGZ']
    ------------------     --------------------------------------------------------------------
    layout_template        Layout Template (str). Optional parameter.  Either a name of a
                           template from the list or the keyword MAP_ONLY. When MAP_ONLY is chosen
                           or an empty string is passed in, the output map does not contain any
                           page layout surroundings.

                           For example - title, legends, scale bar, and so forth

                           Choice list:

                               | ['A3 Landscape', 'A3 Portrait',
                               | 'A4 Landscape', 'A4 Portrait', 'Letter ANSI A Landscape',
                               | 'Letter ANSI A Portrait', 'Tabloid ANSI B Landscape',
                               | 'Tabloid ANSI B Portrait', 'MAP_ONLY'].

                           You can get the layouts configured with your GIS by calling the :meth:`get_layout_templates <arcgis.mapping.get_layout_templates>` function
    ------------------     --------------------------------------------------------------------
    gis                    The :class:`~arcgis.gis.GIS` to use for printing. Optional
                           parameter. When not specified, the active GIS will be used.
    ==================     ====================================================================

    Returns:
        A dictionary with URL to download the output file.
    """

    from arcgis.geoprocessing import import_toolbox as _import_toolbox
    from arcgis.geoprocessing._tool import _camelCase_to_underscore
    from urllib import parse

    verbose = kwargs.pop("verbose", False)

    if gis is None:
        gis = arcgis.env.active_gis
    params = {
        "web_map_as_json": web_map_as_json,
        "format": format,
        "layout_template": layout_template,
        "gis": gis,
        "future": False,
    }
    params.update(kwargs)

    url = os.path.dirname(gis.properties.helperServices.printTask.url)
    tbx = _import_toolbox(url, gis=gis, verbose=verbose)
    basename = os.path.basename(gis.properties.helperServices.printTask.url)
    basename = _camelCase_to_underscore(parse.unquote_plus(parse.unquote(basename)))

    fn = getattr(tbx, basename)
    return fn(**params)


def get_layout_templates(gis: Optional[GIS] = None):
    """

    The ``get_layout_templates`` method returns the content of the :class:`~arcgis.gis.GIS` object's layout templates.

    .. note::
        The layout templates are formatted as a dictionary.

    .. note::
        See the
        `Get Layout Templates Info Task <https://utility.arcgisonline.com/arcgis/rest/directories/arcgisoutput/Utilities/PrintingTools_GPServer/Utilities_PrintingTools/GetLayoutTemplatesInfo.htm>`_
        for additional help on the ``get_layout_templates`` method.


    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS` object. The ``GIS`` on which ``get_layout_templates`` runs.

                           .. note::
                            If ``gis`` is not specified, the active GIS is used.

    ==================     ====================================================================

    Returns:
       ``output_json`` - The layout templates as Python dictionary


    """
    from arcgis.geoprocessing import DataFile
    from arcgis.geoprocessing._support import _execute_gp_tool

    param_db = {
        "output_json": (str, "Output JSON"),
    }
    return_values = [
        {"name": "output_json", "display_name": "Output JSON", "type": str},
    ]

    if gis is None:
        gis = arcgis.env.active_gis

    url = os.path.dirname(gis.properties.helperServices.printTask.url)
    kwargs = {"gis": gis}
    return _execute_gp_tool(
        gis,
        "Get Layout Templates Info Task",
        kwargs,
        param_db,
        return_values,
        _use_async,
        url,
    )


get_layout_templates.__annotations__ = {"return": str}
