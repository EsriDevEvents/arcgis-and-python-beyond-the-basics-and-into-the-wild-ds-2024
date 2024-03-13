"""

The ``Symbology`` class generates Symbol Types for the :class:`~arcgis.gis.GIS` object.

"""
import json
import arcgis
from arcgis.features import FeatureCollection, FeatureSet
from arcgis.gis import GIS
from arcgis.geometry import _types

###########################################################################
ALLOWED_CMAPS = [
    "Accent",
    "Accent_r",
    "Blues",
    "Blues_r",
    "BrBG",
    "BrBG_r",
    "BuGn",
    "BuGn_r",
    "BuPu",
    "BuPu_r",
    "CMRmap",
    "CMRmap_r",
    "Dark2",
    "Dark2_r",
    "GnBu",
    "GnBu_r",
    "Greens",
    "Greens_r",
    "Greys",
    "Greys_r",
    "OrRd",
    "OrRd_r",
    "Oranges",
    "Oranges_r",
    "PRGn",
    "PRGn_r",
    "Paired",
    "Paired_r",
    "Pastel1",
    "Pastel1_r",
    "Pastel2",
    "Pastel2_r",
    "PiYG",
    "PiYG_r",
    "PuBu",
    "PuBuGn",
    "PuBuGn_r",
    "PuBu_r",
    "PuOr",
    "PuOr_r",
    "PuRd",
    "PuRd_r",
    "Purples",
    "Purples_r",
    "RdBu",
    "RdBu_r",
    "RdGy",
    "RdGy_r",
    "RdPu",
    "RdPu_r",
    "RdYlBu",
    "RdYlBu_r",
    "RdYlGn",
    "RdYlGn_r",
    "Reds",
    "Reds_r",
    "Set1",
    "Set1_r",
    "Set2",
    "Set2_r",
    "Set3",
    "Set3_r",
    "Spectral",
    "Spectral_r",
    "Wistia",
    "Wistia_r",
    "YlGn",
    "YlGnBu",
    "YlGnBu_r",
    "YlGn_r",
    "YlOrBr",
    "YlOrBr_r",
    "YlOrRd",
    "YlOrRd_r",
    "afmhot",
    "afmhot_r",
    "autumn",
    "autumn_r",
    "binary",
    "binary_r",
    "bone",
    "bone_r",
    "brg",
    "brg_r",
    "bwr",
    "bwr_r",
    "cool",
    "cool_r",
    "coolwarm",
    "coolwarm_r",
    "copper",
    "copper_r",
    "cubehelix",
    "cubehelix_r",
    "flag",
    "flag_r",
    "gist_earth",
    "gist_earth_r",
    "gist_gray",
    "gist_gray_r",
    "gist_heat",
    "gist_heat_r",
    "gist_ncar",
    "gist_ncar_r",
    "gist_rainbow",
    "gist_rainbow_r",
    "gist_stern",
    "gist_stern_r",
    "gist_yarg",
    "gist_yarg_r",
    "gnuplot",
    "gnuplot2",
    "gnuplot2_r",
    "gnuplot_r",
    "gray",
    "gray_r",
    "hot",
    "hot_r",
    "hsv",
    "hsv_r",
    "inferno",
    "inferno_r",
    "jet",
    "jet_r",
    "magma",
    "magma_r",
    "nipy_spectral",
    "nipy_spectral_r",
    "ocean",
    "ocean_r",
    "pink",
    "pink_r",
    "plasma",
    "plasma_r",
    "prism",
    "prism_r",
    "rainbow",
    "rainbow_r",
    "seismic",
    "seismic_r",
    "spring",
    "spring_r",
    "summer",
    "summer_r",
    "terrain",
    "terrain_r",
    "viridis",
    "viridis_r",
    "winter",
    "winter_r",
]
###########################################################################
_CMAP_LOOKUP = dict(zip([c.lower() for c in ALLOWED_CMAPS], ALLOWED_CMAPS))
###########################################################################
_SYMBOL_TYPES = {
    "simple": "Simple Point, Line or Polygon Symbol",
    "text": "Text Symbol",
    "picture": "Picture Marker or Fill Symbol",
    "carto": "Cartographic Line Symbol",
}
###########################################################################
LINE_STYLES = {
    "s": "esriSLSSolid",  # default
    "-": "esriSLSDash",
    "-.": "esriSLSDashDot",
    "-..": "esriSLSDashDotDot",
    ".": "esriSLSDot",
    "--": "esriSLSLongDash",
    "--.": "esriSLSLongDashDot",
    "n": "esriSLSNull",
    "s-": "esriSLSShortDash",
    "s-.": "esriSLSShortDashDot",
    "s-..": "esriSLSShortDashDotDot",
    "s.": "esriSLSShortDot",
}
###########################################################################
POINT_STYLES = {
    "o": "esriSMSCircle",  # default
    "+": "esriSMSCross",
    "d": "esriSMSDiamond",
    "s": "esriSMSSquare",
    "x": "esriSMSX",
    # "^" : "esriSMSTriangle" # Does not render in web maps
}
###########################################################################
POLYGON_STYLES = {
    "\\": "esriSFSBackwardDiagonal",
    "/": "esriSFSForwardDiagonal",
    "|": "esriSFSVertical",
    "-": "esriSFSHorizontal",
    "x": "esriSFSDiagonalCross",
    "+": "esriSFSCross",
    "s": "esriSFSSolid",  # default
}
###########################################################################
POLYGON_STYLES_DISPLAY = {
    "\\": "Backward Diagonal",
    "/": "Forward Diagonal",
    "|": "Vertical Bar",
    "-": "Horizontal Bar",
    "x": "Diagonal Cross",
    "+": "Cross",
    "s": "Solid Fill (default)",  # default
}
###########################################################################
POINT_STYLES_DISPLAY = {
    "o": "Circle (default)",  # default
    "+": "Cross",
    "d": "Diamond",
    "s": "Square",
    "x": "X",
    # "^" : "esriSMSTriangle" # Does not render in web maps
}
###########################################################################
LINE_STYLES_DISPLAY = {
    "s": "Solid (default)",  # default
    "-": "Dash",
    "-.": "Dash Dot",
    "-..": "Dash Dot Dot",
    ".": "Dot",
    "--": "Long Dash",
    "--.": "Long Dash Dot",
    "n": "Null",
    "s-": "Short Dash",
    "s-.": "Short Dash Dot",
    "s-..": "Short Dash Dot Dot",
    "s.": "Short Dot",
}
###########################################################################
# Have colormaps separated into categories:
# http://matplotlib.org/examples/color/colormaps_reference.html
cmaps = [
    ("Perceptually Uniform Sequential", ["viridis", "plasma", "inferno", "magma"]),
    (
        "Sequential",
        [
            "Greys",
            "Purples",
            "Blues",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
        ],
    ),
    (
        "Sequential (2)",
        [
            "binary",
            "gist_yarg",
            "gist_gray",
            "gray",
            "bone",
            "pink",
            "spring",
            "summer",
            "autumn",
            "winter",
            "cool",
            "Wistia",
            "hot",
            "afmhot",
            "gist_heat",
            "copper",
        ],
    ),
    (
        "Diverging",
        [
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
        ],
    ),
    (
        "Qualitative",
        [
            "Pastel1",
            "Pastel2",
            "Paired",
            "Accent",
            "Dark2",
            "Set1",
            "Set2",
            "Set3",
            "tab10",
            "tab20",
            "tab20b",
            "tab20c",
        ],
    ),
    (
        "Miscellaneous",
        [
            "flag",
            "prism",
            "ocean",
            "gist_earth",
            "terrain",
            "gist_stern",
            "gnuplot",
            "gnuplot2",
            "CMRmap",
            "cubehelix",
            "brg",
            "hsv",
            "gist_rainbow",
            "rainbow",
            "jet",
            "nipy_spectral",
            "gist_ncar",
        ],
    ),
]


###########################################################################
def _cmap2rgb(colors, step, alpha=1):
    """converts a color map to RGBA list"""
    from matplotlib import cm

    t = getattr(cm, colors)(step, bytes=True)
    t = [int(i) for i in t]
    t[-1] = alpha * 255
    return t


###########################################################################
def display_colormaps(colors=None):
    """
    The ``display_colormaps`` method displays a visual colormaps in order to assist users in selecting a color scheme
    for the data they wish to display on a map, or in a web map.

    .. note::
        ``display_colormaps is a variation of the
        `colormaps reference <http://matplotlib.org/examples/color/colormaps_reference.html>`_ page for `matplotlib`.

    """
    import numpy as np
    import matplotlib.pyplot as plt

    if colors is None:
        cmaps = [("All Color Maps", ALLOWED_CMAPS)]
    elif isinstance(colors, (str)):
        cmaps = [("Selected Color Map", [_CMAP_LOOKUP[colors.lower()]])]
    elif isinstance(colors, (tuple, list)):
        colors = [_CMAP_LOOKUP[color.lower()] for color in colors]
        cmaps = [("Selected Color Map", colors)]
    nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list, nrows):
        """creates the entry for the plot"""
        if nrows == 1:
            nrows = 2
        height = 0.2 * nrows
        fig, axes = plt.subplots(nrows=nrows, figsize=(6, height))
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        axes[0].set_title(cmap_category, fontsize=12)

        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect="auto", cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.0
            fig.text(x_text, y_text, name, va="center", ha="right", fontsize=10)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()

    for cmap_category, cmap_list in cmaps:
        plot_color_gradients(cmap_category, cmap_list, nrows)

    plt.show()


###########################################################################
def show_styles(geometry_type):
    """
    The ``show_styles`` method retrieves the available styles for a given geometry type as a Pandas dataframe.

    :return:
        A Pandas Dataframe
    """
    import pandas as pd

    if geometry_type.lower() in ["point", "multipoint"]:
        c = POINT_STYLES_DISPLAY
    elif geometry_type.lower() in ["polyline", "line"]:
        c = LINE_STYLES_DISPLAY
    elif geometry_type.lower() in ["polygon"]:
        c = POLYGON_STYLES_DISPLAY
    else:
        raise Exception("Invalid geometry type")

    d = [(k, v) for k, v in c.items()]
    return pd.DataFrame(data=d, columns=["MARKER", "ESRI_STYLE"])


###########################################################################
def create_symbol(
    geometry_type, symbol_type=None, symbol_style=None, colors=None, **kwargs
):
    """
    The ``create_symbol`` method generates a Symbol from a given set of parameters.

    ``create_symbol`` creates either a ``Picture``, ``Text``, ``Cartographic``, or ``Simple Symbol``
    based on a given set of parameters.

    .. note::
        Each symbol type has a specific set of
        parameters that are excepted.  There is a simplified input definition similar to `matplotlib`.


    =======================  =========================================================
    **Required Argument**    **Description**
    -----------------------  ---------------------------------------------------------
    geometry_type            required string.  This can be ``point``, ``line``, ``polygon``, or
                             ``multipoint``. It helps to ensure that the symbol created
                             will be supported by that geometry type.
    =======================  =========================================================


    =======================  =========================================================
    **Optional Argument**    **Description**
    -----------------------  ---------------------------------------------------------
    symbol_type              optional string. This is the type of symbol the user
                             needs to create.  Valid inputs are: simple, picture, text,
                             or carto.  The default is simple.
    -----------------------  ---------------------------------------------------------
    symbol_style             optional string. This is the symbology used by the
                             geometry.  For example 's' for a Line geometry is a solid
                             line. And '-' is a dash line.

                             **Point Symbols**

                             + 'o' - Circle (default)
                             + '+' - Cross
                             + 'D' - Diamond
                             + 's' - Square
                             + 'x' - X

                             **Polyline Symbols**

                             + 's' - Solid (default)
                             + '-' - Dash
                             + '-.' - Dash Dot
                             + '-..' - Dash Dot Dot
                             + '.' - Dot
                             + '--' - Long Dash
                             + '--.' - Long Dash Dot
                             + 'n' - Null
                             + 's-' - Short Dash
                             + 's-.' - Short Dash Dot
                             + 's-..' - Short Dash Dot Dot
                             + 's.' - Short Dot

                             **Polygon Symbols**

                             + 's' - Solid Fill (default)
                             + '\' - Backward Diagonal
                             + '/' - Forward Diagonal
                             + '|' - Vertical Bar
                             + '-' - Horizontal Bar
                             + 'x' - Diagonal Cross
                             + '+' - Cross
    -----------------------  ---------------------------------------------------------
    colors                   Optional string or list.  This is the color scheme a user can provide if the exact color is not needed, or a user can provide a list with the color defined as: [red, green blue, alpha]. The values red, green, blue are from 0-255 and alpha is a float value from 0 - 1. The default value is 'jet' color scheme.
    -----------------------  ---------------------------------------------------------
    cstep                    optional integer.  If provided, it's the color location on
                             the color scheme.
    =======================  =========================================================

    **Simple Symbols**

    This is a list of optional parameters that can be given for point, line or
    polygon geometries.

    ====================  =========================================================
    **Parameter**          **Description**
    --------------------  ---------------------------------------------------------
    marker_size           optional float.  Numeric size of the symbol given in
                          points.
    --------------------  ---------------------------------------------------------
    marker_angle          optional float. Numeric value used to rotate the symbol.
                          The symbol is rotated counter-clockwise. For example,
                          The following, angle=-30, in will create a symbol rotated
                          -30 degrees counter-clockwise; that is, 30 degrees
                          clockwise.
    --------------------  ---------------------------------------------------------
    marker_xoffset        Numeric value indicating the offset on the x-axis in points.
    --------------------  ---------------------------------------------------------
    marker_yoffset        Numeric value indicating the offset on the y-axis in points.
    --------------------  ---------------------------------------------------------
    line_width            optional float. Numeric value indicating the width of the line in points
    --------------------  ---------------------------------------------------------
    outline_style         Optional string. For polygon point, and line geometries , a
                          customized outline type can be provided.

                          Allowed Styles:

                          + 's' - Solid (default)
                          + '-' - Dash
                          + '-.' - Dash Dot
                          + '-..' - Dash Dot Dot
                          + '.' - Dot
                          + '--' - Long Dash
                          + '--.' - Long Dash Dot
                          + 'n' - Null
                          + 's-' - Short Dash
                          + 's-.' - Short Dash Dot
                          + 's-..' - Short Dash Dot Dot
                          + 's.' - Short Dot
    --------------------  ---------------------------------------------------------
    outline_color         optional string or list.  This is the same color as the
                          colors property, but specifically applies to the outline_color.
    ====================  =========================================================

    **Picture Symbol**

    This type of symbol only applies to Points, MultiPoints and Polygons.

    ====================  =========================================================
    **Parameter**          **Description**
    --------------------  ---------------------------------------------------------
    marker_angle          Numeric value that defines the number of degrees ranging
                          from 0-360, that a marker symbol is rotated. The rotation
                          is from East in a counter-clockwise direction where East
                          is the 0 axis.
    --------------------  ---------------------------------------------------------
    marker_xoffset        Numeric value indicating the offset on the x-axis in points.
    --------------------  ---------------------------------------------------------
    marker_yoffset        Numeric value indicating the offset on the y-axis in points.
    --------------------  ---------------------------------------------------------
    height                Numeric value used if needing to resize the symbol. Specify a value in points. If images are to be displayed in their original size, leave this blank.
    --------------------  ---------------------------------------------------------
    width                 Numeric value used if needing to resize the symbol. Specify a value in points. If images are to be displayed in their original size, leave this blank.
    --------------------  ---------------------------------------------------------
    url                   String value indicating the URL of the image. The URL should be relative if working with static layers. A full URL should be used for map service dynamic layers. A relative URL can be dereferenced by accessing the map layer image resource or the feature layer image resource.
    --------------------  ---------------------------------------------------------
    image_data            String value indicating the base64 encoded data.
    --------------------  ---------------------------------------------------------
    xscale                Numeric value indicating the scale factor in x direction.
    --------------------  ---------------------------------------------------------
    yscale                Numeric value indicating the scale factor in y direction.
    --------------------  ---------------------------------------------------------
    outline_color         optional string or list.  This is the same color as the
                          colors property, but specifically applies to the outline_color.
    --------------------  ---------------------------------------------------------
    outline_style         Optional string. For polygon point, and line geometries , a
                          customized outline type can be provided.

                          Allowed Styles:

                          + 's' - Solid (default)
                          + '-' - Dash
                          + '-.' - Dash Dot
                          + '-..' - Dash Dot Dot
                          + '.' - Dot
                          + '--' - Long Dash
                          + '--.' - Long Dash Dot
                          + 'n' - Null
                          + 's-' - Short Dash
                          + 's-.' - Short Dash Dot
                          + 's-..' - Short Dash Dot Dot
                          + 's.' - Short Dot
    --------------------  ---------------------------------------------------------
    outline_color         optional string or list.  This is the same color as the
                          colors property, but specifically applies to the outline_color.
    --------------------  ---------------------------------------------------------
    line_width            optional float. Numeric value indicating the width of the line in points
    ====================  =========================================================

    **Text Symbol**

    This type of symbol only applies to Points, MultiPoints and Polygons.

    ====================  =========================================================
    **Parameter**          **Description**
    --------------------  ---------------------------------------------------------
    font_decoration       The text decoration. Must be one of the following values:

                          - line-through
                          - underline
                          - none
    --------------------  ---------------------------------------------------------
    font_family           Optional string. The font family.
    --------------------  ---------------------------------------------------------
    font_size             Optional float. The font size in points.
    --------------------  ---------------------------------------------------------
    font_style            Optional string. The text style.

                          - italic
                          - normal
                          - oblique
    --------------------  ---------------------------------------------------------
    font_weight           Optional string. The text weight.
                          Must be one of the following values:

                          - bold
                          - bolder
                          - lighter
                          - normal
    --------------------  ---------------------------------------------------------
    background_color      optional string/list. Background color is represented as
                          a four-element array or string of a color map.
    --------------------  ---------------------------------------------------------
    halo_color            Optional string/list. Color of the halo around the text.
                          The default is None.
    --------------------  ---------------------------------------------------------
    halo_size             Optional integer/float. The point size of a halo around
                          the text symbol.
    --------------------  ---------------------------------------------------------
    horizontal_alignment  optional string. One of the following string values
                          representing the horizontal alignment of the text.
                          Must be one of the following values:

                          - left
                          - right
                          - center
                          - justify
    --------------------  ---------------------------------------------------------
    kerning               optional boolean. Boolean value indicating whether to
                          adjust the spacing between characters in the text string.
    --------------------  ---------------------------------------------------------
    line_color            optional string/list. Outline color is represented as
                          a four-element array or string of a color map.
    --------------------  ---------------------------------------------------------
    line_width            optional integer/float. Outline size.
    --------------------  ---------------------------------------------------------
    marker_angle          optional int. A numeric value that defines the number of
                          degrees (0 to 360) that a text symbol is rotated. The
                          rotation is from East in a counter-clockwise direction
                          where East is the 0 axis.
    --------------------  ---------------------------------------------------------
    marker_xoffset        optional int/float.Numeric value indicating the offset
                          on the x-axis in points.
    --------------------  ---------------------------------------------------------
    marker_yoffset        optional int/float.Numeric value indicating the offset
                          on the x-axis in points.
    --------------------  ---------------------------------------------------------
    right_to_left         optional boolean. Set to true if using Hebrew or Arabic
                          fonts.
    --------------------  ---------------------------------------------------------
    rotated               optional boolean. Boolean value indicating whether every
                          character in the text string is rotated.
    --------------------  ---------------------------------------------------------
    text                  Required string.  Text Value to display next to geometry.
    --------------------  ---------------------------------------------------------
    vertical_alignment    Optional string. One of the following string values
                          representing the vertical alignment of the text.
                          Must be one of the following values:

                          - top
                          - bottom
                          - middle
                          - baseline
    ====================  =========================================================

    **Cartographic Symbol**

    This type of symbol only applies to line geometries.

    ====================  =========================================================
    **Parameter**          **Description**
    --------------------  ---------------------------------------------------------
    line_width            optional float. Numeric value indicating the width of the line in points
    --------------------  ---------------------------------------------------------
    cap                   Optional string.  The cap style.
    --------------------  ---------------------------------------------------------
    join                  Optional string. The join style.
    --------------------  ---------------------------------------------------------
    miter_limit           Optional string. Size threshold for showing mitered line joins.
    ====================  =========================================================

    :return: Dictionary

    """
    import numpy as np
    import matplotlib.pyplot as plt

    alpha = kwargs.pop("alpha", 1)
    symbol = kwargs.pop("symbol", None)
    if symbol_type is None:
        symbol_type = "simple"

    gtype = geometry_type.upper()
    if colors is None:
        colors = "jet"

    # Get color step, if not specified, pick a random value between 0 and 255.
    cstep = kwargs.pop("cstep", int(np.random.randint(0, 255)))
    if cstep is None:
        cstep = int(np.random.randint(0, 255))
    renderer_type = "simple"

    marker_size = kwargs.pop("marker_size", 8)
    marker_angle = kwargs.pop("marker_angle", 0)
    marker_xoffset = kwargs.pop("marker_xoffset", 0)
    marker_yoffset = kwargs.pop("marker_yoffset", 0)

    line_width = kwargs.pop("line_width", 2)

    outline_style = LINE_STYLES[kwargs.pop("outline_style", "s")]

    # get outline_color if specified, else use a nice mild gray
    outline_color = kwargs.pop("outline_color", [128, 128, 128, 255])

    # get a random color if a colormap is specified
    if isinstance(colors, str):
        colors = list(_cmap2rgb(colors=colors, step=cstep, alpha=alpha))

    # if specific 4 color tuple is specified, go with it
    elif isinstance(colors, (list, tuple)) and len(colors) == 4:
        colors = colors

    # if not, the first color in the 'jet' color map. This is a deep blue
    else:
        colors = list(_cmap2rgb(colors="jet", step=cstep, alpha=alpha))

    if isinstance(outline_color, str):
        outline_cmap = list(
            _cmap2rgb(
                colors=outline_color, step=int(np.random.randint(0, 255)), alpha=alpha
            )
        )
    elif isinstance(outline_color, (tuple, list)) and len(outline_color) == 4:
        outline_cmap = outline_color
    else:
        outline_cmap = list(
            _cmap2rgb(colors="jet", step=int(np.random.randint(0, 255)), alpha=alpha)
        )

    if symbol is not None:
        return symbol

    if symbol_type.lower() == "simple":  # Default Simple Symbol
        if symbol_style is None:
            if gtype == "POINT":
                symbol_type = "esriSMS"
                symbol_style = POINT_STYLES["o"]
            elif gtype in ["POLYLINE", "LINE"]:
                symbol_type = "esriSLS"
                symbol_style = LINE_STYLES["s"]
            elif gtype in ["POLYGON"]:
                symbol_type = "esriSFS"
                symbol_style = POLYGON_STYLES["s"]
            else:
                raise Exception(
                    "Invalid geometry types only points, lines, and polygons can be plotted"
                )
        else:
            if gtype == "POINT":
                symbol_type = "esriSMS"
                symbol_style = POINT_STYLES[symbol_style]
            elif gtype in ["POLYLINE", "LINE"]:
                symbol_type = "esriSLS"
                symbol_style = LINE_STYLES[symbol_style]
            elif gtype in ["POLYGON"]:
                symbol_type = "esriSFS"
                symbol_style = POLYGON_STYLES[symbol_style]
            else:
                raise Exception(
                    "Invalid geometry types only points, lines, and polygons can be plotted"
                )
        # build the symbol
        symbol = dict()
        symbol["type"] = symbol_type
        if isinstance(colors, str):
            colors = list(_cmap2rgb(colors=colors, step=cstep, alpha=alpha))
        elif isinstance(colors, (list, tuple)) and len(colors) == 4:
            colors = colors
        else:
            colors = list(_cmap2rgb(colors="jet", step=cstep, alpha=alpha))
        if isinstance(outline_color, str):
            outline_cmap = list(
                _cmap2rgb(
                    colors=outline_color,
                    step=int(np.random.randint(0, 255)),
                    alpha=alpha,
                )
            )
        elif isinstance(outline_color, (tuple, list)) and len(outline_color) == 4:
            outline_cmap = outline_color
        else:
            outline_cmap = list(
                _cmap2rgb(
                    colors="jet", step=int(np.random.randint(0, 255)), alpha=alpha
                )
            )
        if gtype == "POINT":
            symbol["style"] = symbol_style
            symbol["color"] = colors
            symbol["size"] = marker_size
            symbol["angle"] = marker_angle
            symbol["xoffset"] = marker_xoffset
            symbol["yoffset"] = marker_yoffset
            symbol["outline"] = {
                "style": outline_style,
                "type": "esriSLS",
                "color": outline_cmap,
                "width": line_width,
            }
        elif gtype in ["LINE", "POLYLINE"]:
            symbol["style"] = symbol_style
            symbol["color"] = colors
            symbol["width"] = line_width
        elif gtype in ["POLYGON"]:
            symbol["style"] = symbol_style
            symbol["color"] = colors
            symbol["outline"] = {
                "style": outline_style,
                "type": "esriSLS",
                "color": outline_cmap,
                "width": line_width,
            }
        return symbol
    elif symbol_type == "picture" and geometry_type.lower() in [
        "multipoint",
        "point",
        "polygon",
    ]:
        if geometry_type.lower() in ["point", "multipoint"]:
            symbol = {
                "type": "esriPMS",
                "angle": marker_angle,
                "xoffset": marker_xoffset,
                "yoffset": marker_yoffset,
                "width": kwargs.pop("width", 10),
                "height": kwargs.pop("height", 10),
                "url": kwargs.pop("url", ""),
                "imageData": kwargs.pop("image_data", ""),
            }
            return symbol
        elif geometry_type.lower() == "polygon":
            symbol = {
                "type": "esriPFS",
                "color": colors,
                "xoffset": marker_xoffset,
                "yoffset": marker_yoffset,
                "xscale": kwargs.pop("xscale", None),
                "yscale": kwargs.pop("yscale", None),
                "width": kwargs.pop("width", 10),
                "height": kwargs.pop("height", 10),
                "angle": marker_angle,
                "url": kwargs.pop("url", ""),
                "imageData": kwargs.pop("image_data", None),
                "outline": {
                    "color": outline_cmap,
                    "width": line_width,
                    "type": "esriSLS",
                    "style": outline_style,
                },
            }
            return symbol
    elif symbol_type.lower() == "picture" and geometry_type.lower() in ["line"]:
        raise Exception("Line geometries do not support picture symbology")
    elif symbol_type.lower() == "text" and geometry_type.lower() in ["point"]:
        symbol = {
            "color": colors,
            "type": "esriTS",
            "horizontalAlignment": kwargs.pop("horizontal_alignment", "center"),
            "verticalAlignment": kwargs.pop("vertical_alignment", "middle"),
            "rightToLeft": kwargs.pop("right_to_left", False),
            "backgroundColor": kwargs.pop("background_color", colors),
            "borderLineColor": outline_cmap,
            "borderLineSize": line_width,
            "haloColor": kwargs.pop("halo_color", None),
            "haloSize": kwargs.pop("halo_size", None),
            "angle": marker_angle,
            "xoffset": marker_xoffset,
            "yoffset": marker_yoffset,
            "text": kwargs.pop("text", "N/A"),
            "rotated": kwargs.pop("rotated", False),
            "kerning": kwargs.pop("kerning", True),
            "font": {
                "size": kwargs.pop("font_size", 10),
                "style": kwargs.pop("font_style", "normal"),
                "weight": kwargs.pop("font_weight", "normal"),
                "family": kwargs.pop("font_familty", "serif"),
                "decoration": kwargs.pop("font_decoration", "none"),
            },
        }
        return symbol
    elif symbol_type.lower() == "text" and geometry_type.lower() not in [
        "point",
        "multipoint",
    ]:
        raise Exception("Text symbology is only supported by points and multipoints")
    elif symbol_type.lower() == "carto" and geometry_type in ["line", "polyline"]:
        cap = kwargs.pop("cap", "esriLCButt")
        join = kwargs.pop("join", "esriLJSMiter")
        miter_limit = kwargs.pop("miter_limit", 1)
        symbol = {
            "color": colors,
            "width": line_width,
            "type": "esriCLS",
            "cap": cap,
            "join": join,
            "miter_limit": miter_limit,
        }
    elif symbol_type.lower() == "carto" and geometry_type not in ["line"]:
        raise Exception("Cartographic symbology is only supported for line geomtries")
    else:
        raise Exception("Invalid symbol and geometry type")
    return symbol
