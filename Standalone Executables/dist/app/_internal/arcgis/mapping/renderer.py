"""
Creates renderer dictionaries that can be used to help visualize webmap content
"""
from __future__ import annotations
from typing import Optional, Union

from arcgis._impl.common._utils import chunks
from arcgis.mapping._utils import (
    _get_list_value,
    _format_colors,
    create_colormap,
)
from arcgis.mapping.symbol import create_symbol, _cmap2rgb
from arcgis.auth.tools import LazyLoader
import itertools

np = LazyLoader("numpy")

__all__ = ["generate_renderer"]

RENDERER_TYPES = {
    "d": "dot density",
    "s": "simple",  #
    "u": "unique",  #
    "u-a": "unique",  #
    "h": "heatmap",  #
    "c": "ClassBreaks",  #
    "p": "Predominance",
    "str": "Stretch",  #
    "t": "Temporal",  #
    "v": "vector field",  #
}


class _DotDensity(object):
    """
    Creates the dot density renderer for Polygon Geometries.

    """

    _df = None
    _attributes = None
    _dot_value = None
    _ref_scale = None
    _unit = None
    _blend_dots = None
    _dot_shape = None
    _dot_size = None
    _bg_color = None
    _seed = 1
    _outline = None
    _type = "dotDensity"

    # ----------------------------------------------------------------------
    def __init__(
        self,
        df,
        attributes,
        dot_value,
        ref_scale,
        unit,
        blend_dots=False,
        shape="s",
        size=1,
        background=None,
        seed=1,
    ):
        """initalizer"""
        self._df = df
        self._attributes = attributes
        self._dot_value = dot_value
        self._ref_scale = ref_scale
        self._unit = unit
        self._blend_dots = blend_dots
        self._dot_shape = shape
        self._dot_size = size
        if background is None:
            background = [0, 0, 0, 0]
        else:
            background = self._cmap(color=background, cstep=None, alpha=0.1)
        self._bg_color = background
        self._seed = seed

    # ----------------------------------------------------------------------
    @property
    def data(self):
        """gets the data"""
        return self._df

    # ----------------------------------------------------------------------
    @property
    def renderer_type(self):
        """returns the current renderer type"""
        return self._type

    # ----------------------------------------------------------------------
    @property
    def background(self):
        """
        Get/Set the background color

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required string. Color to set the background to.
        ===============     ====================================================================

        :return: List

        """
        return self._bg_color

    # ----------------------------------------------------------------------
    @property
    def shape(self):
        """
        Get/Set the shape of the dots

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required string.
                            Values: "o" | "+" | "d" | "s" | "x"
        ===============     ====================================================================

        :return:
            The string representing the dot shape
            {
            "o": "Circle",  # default
            "+": "Cross",
            "d": "Diamond",
            "s": "Square",
            "x": "X",
            }
        """
        return self._dot_shape

    # ----------------------------------------------------------------------
    @shape.setter
    def shape(self, value: str):
        """
        Sets the shape of the dots in the renderer.
        """
        POINT_STYLES_DISPLAY = {
            "o": "Circle",  # default
            "+": "Cross",
            "d": "Diamond",
            "s": "Square",
            "x": "X",
        }
        if str(value).lower() in POINT_STYLES_DISPLAY:
            self._dot_shape = POINT_STYLES_DISPLAY[str(value).lower()]

    # ----------------------------------------------------------------------
    @property
    def unit(self):
        """Get/Set the units"""
        return self._unit

    # ----------------------------------------------------------------------
    @unit.setter
    def unit(self, value: str):
        """gets/sets the units"""
        self._unit = value

    # ----------------------------------------------------------------------
    @property
    def size(self):
        """Get/Set the size of the dots"""
        return self._dot_size

    # ----------------------------------------------------------------------
    @property
    def ref_scale(self):
        """
        Get/Set the reference scale

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required int or float.
        ===============     ====================================================================

        :return: Int or float value depicting the current reference scale
        """
        return self._ref_scale

    # ----------------------------------------------------------------------
    @ref_scale.setter
    def ref_scale(self, value: Union[int, float]):
        """ """
        if isinstance(value, (int, float)):
            self._ref_scale = value
        else:
            raise ValueError("Value must be an int or float.")

    # ----------------------------------------------------------------------
    @size.setter
    def size(self, value: Union[int, float]):
        """
        Sets the size of the dots.
        """
        self._dot_size = float(value)

    # ----------------------------------------------------------------------
    @background.setter
    def background(self, value: Union[str, list, tuple]):
        """sets the background color"""
        import random

        if isinstance(value, str):
            cstep = random.randint(0, 255)
            color = value
            alpha = 1
            self._bg_color = _cmap2rgb(color, cstep, alpha)
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            self._bg_color = value
        else:
            raise ValueError("Invalid color specified.")

    # ----------------------------------------------------------------------
    @property
    def attributes(self):
        """
        returns a list of attributes
        """
        if self._attributes is None:
            self._attributes = []
        return self._attributes

    # ----------------------------------------------------------------------
    def _cmap(self, color, cstep=None, alpha=1):
        """processes the colors"""
        import random

        if isinstance(color, (list, tuple)) and len(color) == 4:
            return color
        else:
            if cstep is None:
                cstep = random.randint(0, 255)
            if color is None:
                color = "jet"
            return _cmap2rgb(color, cstep, alpha)
        return

    # ----------------------------------------------------------------------
    def _cmap2rgb(self, colors, step, alpha=1):
        """converts a color map to RGBA list"""
        from matplotlib import cm

        t = getattr(cm, colors)(step, bytes=True)
        t = [int(i) for i in t]
        t[-1] = alpha * 255
        return t

    # ----------------------------------------------------------------------
    def add_attribute(
        self,
        label: str,
        field: str,
        color: Union[str, list[int]],
        cstep: Optional[int] = None,
        alpha: float = 1,
    ):
        """
        Assigns an attribute to the dot density renderer

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        field               Required String.  Name of the dataset field
        ---------------     --------------------------------------------------------------------
        label               Required String.  Descriptive name of the field.
        ---------------     --------------------------------------------------------------------
        color               Required String/List. A integer array consisting of R,G,B,A or the
                            name of a color map.
        ---------------     --------------------------------------------------------------------
        cstep               Optional Int. A position on the color chart between 0-255.
        ---------------     --------------------------------------------------------------------
        alpha               Optional float. A value between 0-1 that determines the symbol opacity.
        ===============     ====================================================================

        :return: True if successful otherwise error message

        """
        mapped_names = [n["field"].lower() for n in self.attributes]
        if field.lower() in mapped_names:
            raise ValueError("Field already assigned to a label.")
        if field in self._df.columns:
            self._attributes.append(
                {"field": field, "label": label, "color": self._cmap(color)}
            )
            return True
        else:
            raise ValueError("Field not found in dataset.")

    # ----------------------------------------------------------------------
    def remove_attribute(self, field: str):
        """
        Removes the attribute to the dot density renderer.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        field               Required String.  Name of the dataset field
        ===============     ====================================================================

        :return:True if successful else False

        """
        mapped_names = [n["field"].lower() for n in self.attributes if "field" in n]
        if field.lower() in mapped_names:
            idx = mapped_names.index(field.lower())
            self._attributes.pop(idx)
            return True
        return False

    # ----------------------------------------------------------------------
    def add_expression(
        self,
        expression: str,
        title: str,
        label: str,
        color: Union[list, str],
    ):
        """
        Adds an arcade expression to the attributes


        """
        self._attributes.append(
            {
                "valueExpression": expression,
                "valueExpressionTitle": title,
                "label": label,
                "color": self._cmap(color),
            }
        )

    # ----------------------------------------------------------------------
    def remove_expression(self, label: str):
        """
        Adds an arcade expression to the attributes


        """
        labels = [n["label"].lower() for n in self.attributes if "label" in n]
        if label.lower() in labels:
            idx = labels.index(label.lower())
            self._attributes.pop(idx)
            return True
        return False

    # ----------------------------------------------------------------------
    @property
    def dot_value(self):
        """
        Get/Set what each dot is worth. This should be an float/integer.
        """
        return self._dot_value

    # ----------------------------------------------------------------------
    @dot_value.setter
    def dot_value(self, value: Union[float, int]):
        """
        Get/Sets what each dot is worth. This should be an float/integer.
        """
        self._dot_value = value

    # ----------------------------------------------------------------------
    @property
    def renderer(self):
        """returns the JSON renderer"""
        r = {
            "type": "dotDensity",
            "outline": "null",
            "attributes": self.attributes,
            "backgroundColor": self.background,
            "blendDots": self._blend_dots,
            "dotShape": self.shape,
            "dotSize": self.size,
            "legendOptions": {"unit": self.unit},
            "referenceDotValue": self.dot_value,
            "referenceScale": self.ref_scale,
            "seed": self._seed,
        }
        return r


def _size_info(field, min_value, max_value, min_size=6, max_size=37.5, unit="unknown"):
    """ """
    return {
        "type": "sizeInfo",
        "field": field,
        "valueUnit": unit,
        "minSize": min_size,
        "maxSize": max_size,
        "minDataValue": min_value,
        "maxDataValue": max_value,
    }


def _color_info(
    field: str, values: list, steps: int = 6, colors: str = "Reds_r"
) -> dict:
    """
    Creates the Color Information Visual Variable from a collection of information.

    """
    d = {"type": "colorInfo", "field": field, "stops": []}
    import numpy as np

    steps = np.linspace(0, 254, steps, endpoint=True, dtype=int).tolist()

    cmaps = [_cmap2rgb(colors, step) for step in steps]
    uvalues = list(set(values))
    sorted(uvalues)
    psteps = uvalues[:: int(len(uvalues) / len(steps))]
    if psteps[-1] != uvalues[-1]:
        psteps.append(uvalues[-1])
        steps = np.linspace(0, 254, len(steps) + 1, endpoint=True, dtype=int).tolist()
        cmaps = [_cmap2rgb(colors, step) for step in steps]

    # value_index = [uvalues[int(s*step_size)] for s in range(steps)]
    for k, v in dict(zip(psteps, cmaps)).items():
        d["stops"].append({"value": k, "color": v, "label": None})
    return d


# --------------------------------------------------------------------------
def _trans_info(data, **kwargs):
    """creates a transparency information for visual variables"""
    ti = None
    if "trans_info_field" in kwargs:
        ti = {}

        ti["field"] = kwargs.pop("trans_info_field", None)
        ti["normalizationField"] = kwargs.pop("trans_norm_field", None)
        stops = kwargs.pop("trans_stops", None)
        if stops is None and data:
            stops = []
            transp = 100 / len(data)
            for d in data:
                stops.append({"stop": {"value": d, "transparency": transp}})
                transp += transp
                del d
            del data
        elif data is None and stops is None:
            raise Exception("Cannot create transparency info.")

        ti["stops"] = stops
        ti["type"] = "transparencyInfo"
        ti["valueExpression"] = kwargs.pop("trans_value_exp", None)
        ti["valueExpressionTitle"] = kwargs.pop("trans_exp_title", None)
    return ti


# --------------------------------------------------------------------------
def _si_creator(**kwargs):
    """creates the size information from key/value pairs"""
    si = {"type": "sizeInfo"}
    if "si_field" in kwargs or "si_minSize" in kwargs or "size_field" in kwargs:
        si["expression"] = kwargs.pop("si_expresion", "view.scale")
        si["field"] = kwargs.pop("si_field", None)
        si["maxDataValue"] = kwargs.pop("si_max_data_value", None)
        si["maxSize"] = kwargs.pop("si_max_size", 20)
        si["minDatavalue"] = kwargs.pop("si_min_data_value", None)
        si["minSize"] = kwargs.pop("si_min_size", 10)

        si["normalizationField"] = kwargs.pop("si_norm_field", None)
        si["stops"] = kwargs.pop("si_stops", None)
        if "si_target" in kwargs:
            si["target"] = kwargs.pop("si_target", None)
        si["valueExpression"] = kwargs.pop("si_expression", None)
        si["valueExpressionTitle"] = kwargs.pop("si_expression_title", None)
        si["valueUnit"] = kwargs.pop("si_value_unit", None)
        return si
    else:
        return None
    return None


# --------------------------------------------------------------------------
def _ri_creator(**kwargs):
    """creates rotation information for visual variables"""
    ri = {"type": "rotationInfo"}
    if "ri_type" in kwargs:
        ri["rotatationType"] = kwargs.pop("ri_type")
        ri["valueExpression"] = kwargs.pop("ri_expression", None)
        ri["valueExpressionTitle"] = kwargs.pop("ri_expression_title", "ri_title")
        ri["field"] = kwargs.pop("ri_field", None)

        return ri
    return None


# --------------------------------------------------------------------------
def _assemble_visual(
    sdf_or_series,
    **symbol_args,
):
    """
    Helper function to put together a certain form of visual variables for
    renderer functions.

    Note: This was logic taken directly out of the old generate_renderer()
    function so it didn't have to be repeated every time in the new, split
    up renderer functions. This is just a temporary helper function that
    will eventually be made obsolete by visual_variables(), but must exist
    to ensure backwards compatibility for the time being.
    """
    vv = []
    if "size_field" in symbol_args:
        size_field = symbol_args.pop("size_field")
        vv.append(
            _size_info(
                field=size_field,
                min_value=sdf_or_series[size_field].min(),
                max_value=sdf_or_series[size_field].max(),
                min_size=symbol_args.pop("min_size", 1),
                max_size=symbol_args.pop("max_size", 24.1),
                unit=symbol_args.pop("size_units", "unknown"),
            )
        )
    if "ci_field" in symbol_args:
        color_field = symbol_args.pop("ci_field")
        vv.append(
            _color_info(
                field=color_field,
                values=sdf_or_series[color_field],
                steps=symbol_args.pop("ci_steps", 3),
                colors=symbol_args.pop("ci_color", "Reds_r"),
            )
        )
    if "opacity_expression" in symbol_args and "opacity_stops" in symbol_args:
        vv.append(
            {
                "type": "transparencyInfo",
                "valueExpression": symbol_args.pop("opacity_expression"),
                "valueExpressionTitle": "Opacity Expression",
                "stops": symbol_args.pop("opacity_stops"),
            }
        )
    return vv


# --------------------------------------------------------------------------
def visual_variables(geometry_type, sdf_or_list, **kwargs):
    """
    The ``visual_variables`` function is used to create visual variables for the :class:`~arcgis.gis.GIS` object.

    ``visual_variables`` allows developers to take a deep dive into developing custom renderer.
    Here a user/developer can create transparency, size information, and other rules to
    improve the overall feel and look of spatial information on a map.

    .. note::
        Each type of information is detailed in the tables below.

    ======================  =========================================================
    **Optional variables**  **Description**
    ----------------------  ---------------------------------------------------------
    trans_info_field        Attribute field used for setting the transparency of a
                            feature if no trans_value_exp is provided.
    ----------------------  ---------------------------------------------------------
    trans_norm_field        Attribute field used to normalize the data.
    ----------------------  ---------------------------------------------------------
    trans_stops             An array of transparency stop objects.
    ----------------------  ---------------------------------------------------------
    trans_value_exp         An Arcade expression evaluating to a number.
    ----------------------  ---------------------------------------------------------
    trans_exp_title         The title identifying and describing the associated
                            Arcade expression as defined in the valueExpression
                            property.
    ======================  =========================================================

    **Size Info Visual Variable**

    The size Info visual variable defines how size is applied to features based on the
    values of a numeric field attribute. The minimum and maximum values of the data
    should be indicated along with their respective size values. You must specify
    minSize and maxSize or stops to construct the size ramp. All features with values
    falling in between the specified min and max data values (or stops) will be scaled
    proportionally between the provided min and max sizes.

    ======================  =========================================================
    **Parameter**           **Description**
    ----------------------  ---------------------------------------------------------
    si_field                Attribute field used for size rendering if no
                            si_expression is provided.
    ----------------------  ---------------------------------------------------------
    si_max_data_value       The maximum data value.
    ----------------------  ---------------------------------------------------------
    si_max_size             Specifies the largest marker size to use at any given map
                            scale. Can be either a fixed number or object, depending
                            on whether the user chose a fixed range or not.
    ----------------------  ---------------------------------------------------------
    si_min_data_value       The minimum data value.
    ----------------------  ---------------------------------------------------------
    si_min_size             Specifies the smallest marker size to use at any given
                            map scale. Can be either a fixed number or object,
                            depending on whether the user chose a fixed range or not.
    ----------------------  ---------------------------------------------------------
    si_norm_field           Attribute field used to normalize the data.
    ----------------------  ---------------------------------------------------------
    si_stops                An array of objects that defines the thematic size ramp
                            in a sequence of data or expression stops.
    ----------------------  ---------------------------------------------------------
    si_target               Only used when sizeInfo is used for polygon outlines.
                            Value of this property must be outline
    ----------------------  ---------------------------------------------------------
    si_expression           An Arcade expression evaluating to a number
    ----------------------  ---------------------------------------------------------
    si_expression_title     the title identifying and describing the associated
                            Arcade expression
    ----------------------  ---------------------------------------------------------
    si_value_unit           A string value indicating the required unit of measurement.
    ======================  =========================================================

    **Rotation Info Visual Variable**

    A rotation variable is a visual variable that defines the rotation of a symbol
    based on a numeric data value returned from a field or expression. This value is
    typically used to rotate symbols that indicate directionality.

    ======================  =========================================================
    **Parameter**          **Description**
    ----------------------  ---------------------------------------------------------
    ri_field                Attribute field used for setting the rotation of a symbol
                            if no ``ri_expression`` is provided.
    ----------------------  ---------------------------------------------------------
    ri_type                 Defines the origin and direction of rotation depending on
                            how the angle of rotation was measured. Possible values
                            are geographic which rotates the symbol from the north in
                            a clockwise direction and arithmetic which rotates the
                            symbol from the east in a counter-clockwise direction.
                            Must be one of the following values:

                            - geographic
                            - arithmetic

    ----------------------  ---------------------------------------------------------
    ri_expression           An Arcade expression evaluating to a number.
    ----------------------  ---------------------------------------------------------
    ri_expression_title     The title identifying and describing the ``ri_expression``
    ======================  =========================================================



    """
    import pandas as pd

    v = []
    if isinstance(sdf_or_list, pd.DataFrame) and "trans_info_field" in kwargs:
        trans_info_field = kwargs["trans_info_field"]
        data = list(sdf_or_list[trans_info_field].unique())
    elif isinstance(sdf_or_list, (tuple, list)):
        data = list(set(sdf_or_list))
    else:
        data = None

    ti = _trans_info(data=data, **kwargs)
    if ti:
        v.append(ti)
    si = _si_creator(**kwargs)
    if si:
        v.append(si)
    ri = _ri_creator(**kwargs)
    if ri:
        v.append(ri)
    return v


# --------------------------------------------------------------------------
def generate_heatmap(
    sdf_or_series=None,
    colors: Optional[Union[str, list[int], object]] = None,
    alpha: Optional[float] = 1,
    blur_radius: Optional[int] = 10,
    field: Optional[str] = None,
    max_intensity: Optional[int] = 10,
    min_intensity: Optional[int] = 0,
    ratio: Optional[float] = 0.01,
    stops: Optional[int] = 3,
    show_none: Optional[bool] = False,
):
    """

    Generates a heatmap renderer. Used in ``spatial.plot()`` and ``generate_renderer()``.

    ======================  =========================================================
    **Parameter**            **Description**
    ----------------------  ---------------------------------------------------------
    sdf_or_series           Optional Pandas Series. The spatial dataset to render.
    ----------------------  ---------------------------------------------------------
    colors                  Optional string/list/object. Color mapping. Can be array
                            of RGB values, the string name of a colormap, the hex
                            string of a color, or a palettable object. Defaults to
                            'jet' colormap.
    ----------------------  ---------------------------------------------------------
    alpha                   Optional float. The 0 to 1 alpha value of the color used
                            to render, if the input for 'colors' is a colorbrewer JS
                            array or palettable object with no cstep (all other forms
                            of color input have alpha accounted for in bytes form).
                            Defaults to 1.
    ----------------------  ---------------------------------------------------------
    blur_radius             Optional int. The radius (in pixels) of the circle over
                            which the majority of each point's value is spread.
                            Defaults to 10.
    ----------------------  ---------------------------------------------------------
    field                   Optional string. This is optional as this renderer can be
                            created if no field is specified. Each feature gets the
                            same value/importance/weight or with a field where each
                            feature is weighted by the field's value.
    ----------------------  ---------------------------------------------------------
    max_intensity           Optional int. The pixel intensity value which is assigned
                            the final color in the color ramp. Defaults to 10.
    ----------------------  ---------------------------------------------------------
    min_intensity           Optional int. The pixel intensity value which is assigned
                            the initial color in the color ramp. Defaults to 0.
    ----------------------  ---------------------------------------------------------
    ratio                   Optional float. A number between 0-1. Describes what
                            portion along the gradient the colorStop is added.
                            Defaults to 0.01
    ----------------------  ---------------------------------------------------------
    stops                   Optional int. The amount of color stops created for the
                            renderer. Default and minimum is 3.
    ----------------------  ---------------------------------------------------------
    show_none               Optional boolean. Determines the alpha value of the base
                            color for the heatmap. Setting this to ``True`` covers an
                            entire map with the base color of the heatmap. Default is
                            ``False``.
    ======================  =========================================================

    :return: A dictionary of the renderer.
    """
    colors = _format_colors(colors, alpha)
    if sdf_or_series is None and field:
        raise ValueError(
            "sdf_or_series must be a Pandas' Series"
            + " or Pandas DataFrame for this type of renderer"
        )

    # check if user specified exact RGB colorstops in 'colors'
    colorStops = []
    if all([isinstance(i, list) for i in colors]) and len(colors) > 2:
        if not show_none:
            colors[0][-1] = 0
        ratios = np.linspace(0, 1, len(colors)).tolist()
        for idx in range(0, len(colors)):
            colorStops.append(
                {
                    "ratio": ratios[idx],
                    "color": colors[idx],
                }
            )

    # check if user specified colormaps as colorstops in 'colors'
    # uses base color from each map
    elif all([isinstance(i, str) for i in colors]) and len(colors) > 2:
        if show_none:
            calpha = alpha
        else:
            calpha = 0
        ratios = np.linspace(0, 1, len(colors)).tolist()
        colorStops.append(
            {
                "ratio": ratios[0],
                "color": _cmap2rgb(colors=colors[0], step=0, alpha=calpha),
            }
        )
        for idx in range(1, len(colors)):
            colorStops.append(
                {
                    "ratio": ratios[idx],
                    "color": _cmap2rgb(colors=colors[idx], step=0, alpha=alpha),
                }
            )

    else:
        r = 0
        if stops < 3:
            stops = 3
        ratios = np.linspace(0, 1, num=stops)
        for idx, cstep in enumerate(np.linspace(0, 255, num=stops, dtype=int).tolist()):
            if r == 0 and show_none == True:
                calpha = alpha
            elif r == 0 and show_none == False:
                calpha = 0
            else:
                calpha = alpha

            colorStops.append(
                {
                    "ratio": ratios[idx],
                    "color": _cmap2rgb(colors=colors[0], step=cstep, alpha=calpha),
                }
            )
            r += ratio
            del cstep
    renderer = {
        "type": "heatmap",
        "field": field,
        "blurRadius": blur_radius,
        "maxPixelIntensity": max_intensity,
        "minPixelIntensity": min_intensity,
        "colorStops": colorStops,
    }
    return renderer


# --------------------------------------------------------------------------
def generate_unique(
    geometry_type: str,
    sdf_or_series=None,
    colors: Optional[Union[str, list[int], object]] = None,
    alpha: Optional[float] = 1,
    **symbol_args,
):
    """
    Generates a unique renderer. Define an attribute field using a ``field1`` argument.
    Accepts symbol arguments used in ``mapping.symbol.create_symbol()``.
    Used in ``spatial.plot()`` and ``generate_renderer()``.

    ======================  =========================================================
    **Parameter**            **Description**
    ----------------------  ---------------------------------------------------------
    geometry_type           Required string. The allowed values are: ``Point``, ``Polyline``,
                            ``Polygon``, or ``Raster``. This required parameter is used to
                            help ensure the requested renderer is valid for the
                            specific type of geometry.

                            =============   ========================================
                            **Geometry**    **Supported Renderer Types**
                            -------------   ----------------------------------------
                            Point           simple, unique, class breaks, heat map,
                                            temporal
                            -------------   ----------------------------------------
                            Polyline        simple, unique, class break
                            -------------   ----------------------------------------
                            Polygon         simple, unique, class break, dot density
                            -------------   ----------------------------------------
                            Raster          stretched
                            =============   ========================================

                            The table above provides a quick summary based on the
                            allowed renderer types based on the geometry.

    ----------------------  ---------------------------------------------------------
    sdf_or_series           Optional Pandas Series. The spatial dataset to render.
    ----------------------  ---------------------------------------------------------
    colors                  Optional string/list/object. Color mapping. Can be array
                            of RGB values, the string name of a colormap, the hex
                            string of a color, or a palettable object. Defaults to
                            'jet' colormap.
    ----------------------  ---------------------------------------------------------
    alpha                   Optional float. The 0 to 1 alpha value of the color used
                            to render, if the input for 'colors' is a colorbrewer JS
                            array or palettable object with no cstep (all other forms
                            of color input have alpha accounted for in bytes form).
                            Defaults to 1.
    ======================  =========================================================

    ======================  =========================================================
    **Optional Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    background_fill_symbol  A symbol used for polygon features as a background if the
                            renderer uses point symbols, e.g. for bivariate types &
                            size rendering. Only applicable to polygon layers.
                            PictureFillSymbols can also be used outside of the Map
                            Viewer for Size and Predominance and Size renderers.
    ----------------------  ---------------------------------------------------------
    default_label           Default label for the default symbol used to draw
                            unspecified values.
    ----------------------  ---------------------------------------------------------
    default_symbol          Symbol used when a value cannot be matched.
    ----------------------  ---------------------------------------------------------
    field1, field2, field3  Attribute field renderer uses to match values.
    ----------------------  ---------------------------------------------------------
    field_delimiter         String inserted between the values if multiple attribute
                            fields are specified.
    ----------------------  ---------------------------------------------------------
    rotation_expression     A constant value or an expression that derives the angle
                            of rotation based on a feature attribute value. When an
                            attribute name is specified, it's enclosed in square
                            brackets. Rotation is set using a visual variable of type
                            rotation info with a specified field or value expression
                            property.
    ----------------------  ---------------------------------------------------------
    rotation_type           String property which controls the origin and direction
                            of rotation. If the rotation type is defined as
                            arithmetic the symbol is rotated from East in a
                            counter-clockwise direction where East is the 0 degree
                            axis. If the rotation type is defined as geographic, the
                            symbol is rotated from North in a clockwise direction
                            where North is the 0 degree axis.
                            Must be one of the following values:

                            + arithmetic
                            + geographic

    ----------------------  ---------------------------------------------------------
    arcade_expression       An Arcade expression evaluating to either a string or a
                            number.
    ----------------------  ---------------------------------------------------------
    arcade_title            The title identifying and describing the associated
                            Arcade expression as defined in the valueExpression
                            property.
    ----------------------  ---------------------------------------------------------
    visual_variables        An array of objects used to set rendering properties.
    ----------------------  ---------------------------------------------------------
    unique_values           Optional list of dictionaries.  If you want to define the
                            unique values listed in the renderer, specify the list
                            using this variable.  The format of each unique value is
                            as follows:


                                | {
                                |     "value" : <value>,
                                |     "label" : <label value>,
                                |     "description" : <optional text description>,
                                |     "symbol" : {...symbol...}
                                | }

    ======================  =========================================================

    :return: A dictionary of the renderer.
    """

    if (
        "size_field" in symbol_args
        or "ci_field" in symbol_args
        or "opacity_expression" in symbol_args
    ):
        vv = _assemble_visual(sdf_or_series, **symbol_args)
    else:
        vv = visual_variables(
            geometry_type=geometry_type,
            sdf_or_list=sdf_or_series,
            **symbol_args,
        )

    if "arcade_expression" not in symbol_args:
        if not hasattr(colors, "mpl_colormap"):
            colors = _format_colors(colors, alpha)
        if sdf_or_series is None:
            raise ValueError(
                "sdf_or_series must be a Pandas' Series"
                + " or Pandas DataFrame for this type of renderer"
            )
        st = symbol_args.pop("symbol_type", None)
        ss = symbol_args.pop("symbol_style", None)
        default_symbol = symbol_args.pop("default_symbol", None)
        if default_symbol is None:
            if isinstance(colors, (list, tuple)):
                ccmap = colors[0]
            else:
                ccmap = colors
            default_symbol = create_symbol(
                geometry_type=geometry_type.lower(),
                symbol_type=st,
                symbol_style=ss,
                colors=ccmap,
                **symbol_args,
            )
        field1 = symbol_args.pop("field1", None)
        if field1 is None:
            raise ValueError(
                "You must provide a single field name to use unique value renderer as field1='columnname'"
            )
        field2 = symbol_args.pop("field2", None)
        field3 = symbol_args.pop("field3", None)
        fields = [field1]
        if field2 is not None:
            fields.append(field2)
        if field3 is not None:
            fields.append(field3)
        if isinstance(fields, str):
            fields = [fields]
        field_delimiter = symbol_args.pop("field_delimiter", ",")
        rotation_expression = symbol_args.pop("rotation_expression", None)
        rotation_type = symbol_args.pop("rotation_type", "arithmetic")

        renderer = {
            "type": "uniqueValue",
            "defaultLabel": symbol_args.pop("default_label", "Other"),
            "defaultSymbol": default_symbol,
            "fieldDelimiter": field_delimiter,
            "rotationExpression": rotation_expression,
            "rotationType": rotation_type,
            "valueExpression": symbol_args.pop("arcade_expression", None),
            "valueExpressionTitle": symbol_args.pop("arcade_title", None),
            "visualVariables": symbol_args.pop("visual_variables", vv),
        }
        unique_values = symbol_args.pop("unique_values", None)
        if unique_values is None:
            c = 1
            for f in fields:
                renderer["field%s" % c] = f
                c += 1
            if len(fields) == 1:
                try:
                    uvals = list(sdf_or_series[fields[0]].unique())
                except:
                    uvals = sdf_or_series[fields[0]].unique().tolist()
            else:
                uvals = (
                    sdf_or_series.groupby(fields)
                    .size()
                    .reset_index()
                    .rename(columns={0: "count"})
                    .drop(columns="count")
                    .tolist()
                )
                uvals2 = []
                for r in uvals:
                    row = []
                    for i in r:
                        row.append(str(i))
                    uvals2.append(",".join(row))
                    del r
                uvals = uvals2
                if len(uvals) > 255:
                    uvals = uvals[:255]
            unique_values = []

            steps = np.linspace(0, 255, len(uvals), dtype=int)

            for idx, uval in enumerate(uvals):
                if hasattr(colors, "mpl_colormap"):
                    temp_map = colors.mpl_colormap
                    color_tuple = temp_map(steps[idx], bytes=True)
                    color = [int(i) for i in color_tuple]
                elif isinstance(colors[0], str):
                    color = _cmap2rgb(colors[0], steps[idx], alpha)
                elif isinstance(colors[0], list):
                    color = _get_list_value(idx, colors)
                unique_values.append(
                    {
                        "value": uval,
                        "label": uval,
                        "description": "",
                        "symbol": create_symbol(
                            geometry_type=geometry_type.lower(),
                            symbol_type=st,
                            symbol_style=ss,
                            colors=color,
                            **symbol_args,
                        ),
                    }
                )
            renderer["uniqueValueInfos"] = unique_values
        else:
            renderer["uniqueValueInfos"] = unique_values
    else:
        if sdf_or_series is None:
            raise ValueError(
                "sdf_or_series must be a Pandas' Series"
                + " or Pandas DataFrame for this type of renderer"
            )
        st = symbol_args.pop("symbol_type", None)
        ss = symbol_args.pop("symbol_style", None)
        default_symbol = symbol_args.pop("default_symbol", None)
        if default_symbol is None:
            if isinstance(colors, (list, tuple)):
                ccmap = colors[0]
            else:
                ccmap = colors
            default_symbol = create_symbol(
                geometry_type=geometry_type.lower(),
                symbol_type=st,
                symbol_style=ss,
                colors=ccmap,
                **symbol_args,
            )
        field_delimiter = symbol_args.pop("field_delimiter", ",")
        rotation_expression = symbol_args.pop("rotation_expression", None)
        rotation_type = symbol_args.pop("rotation_type", "arithmetic")

        renderer = {
            "type": "uniqueValue",
            "defaultLabel": symbol_args.pop("default_label", "Other"),
            "defaultSymbol": default_symbol,
            "fieldDelimiter": field_delimiter,
            "rotationExpression": rotation_expression,
            "rotationType": rotation_type,
            "valueExpression": symbol_args.pop("arcade_expression", None),
            "valueExpressionTitle": symbol_args.pop("arcade_title", None),
            "visualVariables": symbol_args.pop("visual_variables", vv),
        }
        if "unique_values" not in symbol_args:
            raise ValueError("unique_values must be provided if field1 is not given.")
        renderer["uniqueValueInfos"] = symbol_args.pop("unique_values")
    return renderer


# --------------------------------------------------------------------------
def generate_classbreaks(
    sdf_or_series=None,
    geometry_type: Optional[str] = None,
    colors: Optional[Union[str, list[int], object]] = None,
    alpha: Optional[float] = 1,
    **symbol_args,
):
    """
    Generates a classbreaks renderer. Define the attribute field using a ``field`` argument.
    Accepts symbol arguments used in ``mapping.symbol.create_symbol()``.
    Used in ``spatial.plot()`` and ``generate_renderer()``.

    ======================  =========================================================
    **Parameter**            **Description**
    ----------------------  ---------------------------------------------------------
    geometry_type           Required string. The allowed values are: ``Point``, ``Polyline``,
                            ``Polygon``, or ``Raster``. This required parameter is used to
                            help ensure the requested renderer is valid for the
                            specific type of geometry.

                            =============   ========================================
                            **Geometry**    **Supported Renderer Types**
                            -------------   ----------------------------------------
                            Point           simple, unique, class breaks, heat map,
                                            temporal
                            -------------   ----------------------------------------
                            Polyline        simple, unique, class break
                            -------------   ----------------------------------------
                            Polygon         simple, unique, class break, dot density
                            -------------   ----------------------------------------
                            Raster          stretched
                            =============   ========================================

                            The table above provides a quick summary based on the
                            allowed renderer types based on the geometry.

    ----------------------  ---------------------------------------------------------
    sdf_or_series           Optional Pandas Series. The spatial dataset to render.
    ----------------------  ---------------------------------------------------------
    colors                  Optional string/list/object. Color mapping. Can be array
                            of RGB values, the string name of a colormap, the hex
                            string of a color, or a palettable object. Defaults to
                            'jet' colormap.
    ----------------------  ---------------------------------------------------------
    alpha                   Optional float. The 0 to 1 alpha value of the color used
                            to render, if the input for 'colors' is a colorbrewer JS
                            array or palettable object with no cstep (all other forms
                            of color input have alpha accounted for in bytes form).
                            Defaults to 1.
    ======================  =========================================================

    ======================  =========================================================
    **Optional Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    background_fill_symbol  A symbol used for polygon features as a background if the
                            renderer uses point symbols, e.g. for bivariate types &
                            size rendering. Only applicable to polygon layers.
                            PictureFillSymbols can also be used outside of the Map
                            Viewer for Size and Predominance and Size renderers.
    ----------------------  ---------------------------------------------------------
    default_label           Default label for the default symbol used to draw
                            unspecified values.
    ----------------------  ---------------------------------------------------------
    default_symbol          Symbol used when a value cannot be matched.
    ----------------------  ---------------------------------------------------------
    method                  Determines the classification method that was used to
                            generate class breaks.

                            Must be one of the following values:

                            + esriClassifyDefinedInterval
                            + esriClassifyEqualInterval
                            + esriClassifyGeometricalInterval
                            + esriClassifyNaturalBreaks
                            + esriClassifyQuantile
                            + esriClassifyStandardDeviation
                            + esriClassifyManual

    ----------------------  ---------------------------------------------------------
    field                   Attribute field used for renderer.
    ----------------------  ---------------------------------------------------------
    min_value               The minimum numeric data value needed to begin class
                            breaks.
    ----------------------  ---------------------------------------------------------
    normalization_field     Used when normalizationType is field. The string value
                            indicating the attribute field by which the data value is
                            normalized.
    ----------------------  ---------------------------------------------------------
    normalization_total     Used when normalizationType is percent-of-total, this
                            number property contains the total of all data values.
    ----------------------  ---------------------------------------------------------
    normalization_type      Determine how the data was normalized.

                            Must be one of the following values:

                            + esriNormalizeByField
                            + esriNormalizeByLog
                            + esriNormalizeByPercentOfTotal
    ----------------------  ---------------------------------------------------------
    rotation_expression     A constant value or an expression that derives the angle
                            of rotation based on a feature attribute value. When an
                            attribute name is specified, it's enclosed in square
                            brackets.
    ----------------------  ---------------------------------------------------------
    rotation_type           A string property which controls the origin and direction
                            of rotation. If the rotation_type is defined as
                            arithmetic, the symbol is rotated from East in a
                            couter-clockwise direction where East is the 0 degree
                            axis. If the rotationType is defined as geographic, the
                            symbol is rotated from North in a clockwise direction
                            where North is the 0 degree axis.

                            Must be one of the following values:

                            + arithmetic
                            + geographic

    ----------------------  ---------------------------------------------------------
    arcade_expression       An Arcade expression evaluating to a number.
    ----------------------  ---------------------------------------------------------
    arcade_title            The title identifying and describing the associated
                            Arcade expression as defined in the arcade_expression
                            property.
    ----------------------  ---------------------------------------------------------
    visual_variables        An object used to set rendering options.
    ======================  =========================================================

    :return: A dictionary of the renderer.

    """

    if (
        "size_field" in symbol_args
        or "ci_field" in symbol_args
        or "opacity_expression" in symbol_args
    ):
        vv = _assemble_visual(sdf_or_series, **symbol_args)
    else:
        vv = visual_variables(
            geometry_type=geometry_type,
            sdf_or_list=sdf_or_series,
            **symbol_args,
        )

    if sdf_or_series is None:
        raise ValueError(
            "sdf_or_series must be a Pandas' Series"
            + " or Pandas DataFrame for this type of renderer"
        )
    class_count = symbol_args.pop(
        "class_count", 3
    )  # number of classess for class break

    temp_map = None
    if hasattr(colors, "mpl_colormap"):
        temp_map = colors.mpl_colormap
    else:
        colors = _format_colors(colors, alpha)
        if isinstance(colors[0], list) and len(colors) > 1:
            temp_map = create_colormap(colors)
        # possible outcomes are colormap, single rbg array, str colormap name

    try:
        if hasattr(sdf_or_series, "geometry_type"):
            gt = sdf_or_series.geometry_type
        elif (
            hasattr(sdf_or_series, "spatial")
            and sdf_or_series.spatial.name
            and hasattr(sdf_or_series.spatial, "geometry_type")
        ):
            gt = sdf_or_series.spatial.geometry_type[0]
    except:
        raise Exception(
            "geometry_type not found, please ensure DataFrame is spatially enabled."
        )
    renderer = {
        "type": "classBreaks",
        "valueExpression": symbol_args.pop("arcade_expression", None),
        "valueExpressionTitle": symbol_args.pop("arcade_title", None),
        "visualVariables": symbol_args.pop("visual_variables", vv),
        "rotationType": symbol_args.pop("rotation_type", "arithmetic"),
        "rotationExpression": symbol_args.pop("rotation_expression", None),
        "normalizationType": symbol_args.pop("normalization_type", None),
        "normalizationTotal": symbol_args.pop("normalization_total", None),
        "normalizationField": symbol_args.pop("normalization_field", None),
        "minValue": symbol_args.pop("min_value", 0),
        "field": symbol_args.pop("field"),
        "defaultSymbol": symbol_args.pop(
            "default_symbol",
            create_symbol(geometry_type=gt, colors=_format_colors(colors, alpha)[0]),
        ),
        "defaultLabel": symbol_args.pop("default_label", "Other"),
        "classificationMethod": symbol_args.pop("method", None),
        "classBreakInfos": [],
        "backgroundFillSymbol": symbol_args.pop("background_fill_symbol", None),
    }
    minValue = sdf_or_series[renderer["field"]].min()
    maxValue = sdf_or_series[renderer["field"]].max()

    def pairwise(iterable, fillvalue=999):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."

        a, b = itertools.tee(iterable)
        next(b, fillvalue)
        return itertools.zip_longest(a, b)

    # calculate the class breaks from column data
    cbs = []
    breaks = np.linspace(float(minValue), float(maxValue), num=class_count + 1).tolist()
    steps = np.linspace(0, 255, len(breaks), dtype=int)
    ss = symbol_args.pop("symbol_style", None)
    st = symbol_args.pop("symbol_type", None)
    import sys

    for idx, pair in enumerate(pairwise(breaks, fillvalue=sys.maxsize)):
        gt = None
        if pair[1] is None:
            break
        if hasattr(sdf_or_series, "geometry_type"):
            gt = sdf_or_series.geometry_type
        elif (
            hasattr(sdf_or_series, "spatial")
            and sdf_or_series.spatial.name
            and hasattr(sdf_or_series.spatial, "geometry_type")
        ):
            gt = sdf_or_series.spatial.geometry_type[0]
        if temp_map:
            color_tuple = temp_map(steps[idx], bytes=True)
            color = [int(i) for i in color_tuple]
        else:
            color = colors[0]
        cbs.append(
            {
                "classMaxValue": pair[1] or pair[0],
                "label": "%s - %s" % (pair[0], pair[1] or pair[0]),
                "description": "%s - %s" % (pair[0], pair[1] or pair[0]),
                "symbol": create_symbol(
                    geometry_type=gt,
                    symbol_style=ss,
                    symbol_type=st,
                    colors=color,
                    cstep=steps[idx],
                    **symbol_args,
                ),
            }
        )
        del pair
    renderer["classBreakInfos"] = cbs
    for key in [k for k, v in renderer.items() if v is None]:
        del renderer[key]
    return renderer


# --------------------------------------------------------------------------
def generate_simple(
    geometry_type: str,
    sdf_or_series=None,
    label: Optional[str] = None,
    colors: Optional[Union[str, list[int], object]] = None,
    alpha: Optional[float] = 1,
    **symbol_args,
):
    """
    Generates a simple renderer. Accepts certain symbol arguments used in ``mapping.symbol.create_symbol()``.
    Used in ``spatial.plot()`` and ``generate_renderer()``.

    ======================  =========================================================
    **Parameter**            **Description**
    ----------------------  ---------------------------------------------------------
    geometry_type           Required string. The allowed values are: ``Point``, ``Polyline``,
                            ``Polygon``, or ``Raster``. This required parameter is used to
                            help ensure the requested renderer is valid for the
                            specific type of geometry.

                            =============   ========================================
                            **Geometry**    **Supported Renderer Types**
                            -------------   ----------------------------------------
                            Point           simple, unique, class breaks, heat map,
                                            temporal
                            -------------   ----------------------------------------
                            Polyline        simple, unique, class break
                            -------------   ----------------------------------------
                            Polygon         simple, unique, class break, dot density
                            -------------   ----------------------------------------
                            Raster          stretched
                            =============   ========================================

                            The table above provides a quick summary based on the
                            allowed renderer types based on the geometry.

    ----------------------  ---------------------------------------------------------
    sdf_or_series           Optional Pandas Series. The spatial dataset to render.
    ----------------------  ---------------------------------------------------------
    label                   Optional string. Name of the layer in the TOC/Legend.
    ----------------------  ---------------------------------------------------------
    colors                  Optional string/list/object. Color mapping. Can be array
                            of RGB values, the string name of a colormap, the hex
                            string of a color, or a palettable object. Defaults to
                            'jet' colormap.
    ----------------------  ---------------------------------------------------------
    alpha                   Optional float. The 0 to 1 alpha value of the color used
                            to render, if the input for 'colors' is a colorbrewer JS
                            array or palettable object with no cstep (all other forms
                            of color input have alpha accounted for in bytes form).
                            Defaults to 1.
    ======================  =========================================================


    ======================  =========================================================
    **Symbol Argument**     **Description**
    ----------------------  ---------------------------------------------------------
    symbol_type             Optional string. This is the type of symbol the user
                            needs to create.  Valid inputs are: simple, picture, text,
                            or carto.  The default is simple.
    ----------------------  ---------------------------------------------------------
    symbol_style            Optional string. This is the symbology used by the
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
    ----------------------  ---------------------------------------------------------
    description             Description of the renderer.
    ----------------------  ---------------------------------------------------------
    cstep                   Int. If colors argument is the name of a colormap or a
                            palettable object, the colorstep value for the colormap.
                            Range is 0-255.
    ----------------------  ---------------------------------------------------------
    rotation_expression     A constant value or an expression that derives the angle
                            of rotation based on a feature attribute value. When an
                            attribute name is specified, it's enclosed in square
                            brackets.
    ----------------------  ---------------------------------------------------------
    rotation_type           String value which controls the origin and direction of
                            rotation on point features. If the rotationType is
                            defined as arithmetic, the symbol is rotated from East in
                            a counter-clockwise direction where East is the 0 degree
                            axis. If the rotationType is defined as geographic, the
                            symbol is rotated from North in a clockwise direction
                            where North is the 0 degree axis.

                            Must be one of the following values:

                            + arithmetic
                            + geographic

    ----------------------  ---------------------------------------------------------
    visual_variables        An array of objects used to set rendering properties.
    ======================  =========================================================

    :return: A dictionary of the renderer.
    """
    if (
        "size_field" in symbol_args
        or "ci_field" in symbol_args
        or "opacity_expression" in symbol_args
    ):
        vv = _assemble_visual(sdf_or_series, **symbol_args)
    else:
        vv = visual_variables(
            geometry_type=geometry_type,
            sdf_or_list=sdf_or_series,
            **symbol_args,
        )
    cstep = symbol_args.pop("cstep", None)
    """if "cstep" in symbol_args:
        fstep = symbol_args["cstep"]
    else:
        fstep = 0"""
    colors = _format_colors(colors, alpha, cstep)
    symbol = symbol_args.pop("symbol", None)
    if symbol is None:
        symbol = create_symbol(
            geometry_type=geometry_type.lower(),
            symbol_type=symbol_args.pop("symbol_type", None),
            symbol_style=symbol_args.pop("symbol_style", None),
            colors=colors[0],
            cstep=cstep,
            **symbol_args,
        )
    renderer = {
        "type": "simple",
        "label": label,
        "description": symbol_args.pop("description", ""),
        "rotationExpression": symbol_args.pop("rotation_expression", ""),
        "rotationType": symbol_args.pop("rotation_type", "arithmetic"),
        "visualVariables": symbol_args.pop("visual_variables", vv),
        "symbol": symbol,
    }
    return renderer


# --------------------------------------------------------------------------
def generate_renderer(
    geometry_type: str,
    sdf_or_series=None,
    label: Optional[str] = None,
    render_type: Optional[str] = None,
    colors: Optional[Union[str, list[int]]] = None,
    **symbol_args,
):
    """
    Generates the Renderer JSON


    ======================  =========================================================
    **Explicit Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    geometry_type           Required string. The allowed values are: ``Point``, ``Polyline``,
                            ``Polygon``, or ``Raster``. This required parameter is used to
                            help ensure the requested renderer is valid for the
                            specific type of geometry.

                            =============   ========================================
                            **Geometry**    **Supported Renderer Types**
                            -------------   ----------------------------------------
                            Point           simple, unique, class breaks, heat map,
                                            temporal
                            -------------   ----------------------------------------
                            Polyline        simple, unique, class break
                            -------------   ----------------------------------------
                            Polygon         simple, unique, class break, dot density
                            -------------   ----------------------------------------
                            Raster          stretched
                            =============   ========================================

                            The table above provides a quick summary based on the
                            allowed renderer types based on the geometry.

    ----------------------  ---------------------------------------------------------
    sdf_or_series           Optional Pandas Series. The spatial dataset to render.
    ----------------------  ---------------------------------------------------------
    label                   Optional string. Name of the layer in the TOC/Legend
    ----------------------  ---------------------------------------------------------
    render_type             Optional string.  Determines the type of renderer to use
                            for the provided dataset. The default is 's' which is for
                            simple renderers.

                            Allowed values:

                            + 's' - is a simple renderer that uses one symbol only.
                            + 'u' - unique renderer symbolizes features based on one or more matching string attributes.
                            + 'u-a' - unique renderer symbolizes features based on an arcade expression.
                            + 'c' - A class breaks renderer symbolizes based on the value of some numeric attribute.
                            + 'h' - heatmap renders point data into a raster visualization that emphasizes areas of higher density or weighted values.
                            + 'd' - dot density renderer

    ----------------------  ---------------------------------------------------------
    colors                  Optional string/list.  Color mapping.  For simple renderer,
                            just provide a string or single RGB + alpha color array.
                            For a unique renderer, a list of color arrays or colormaps
                            can be given. For heatmaps, list specific RGB colorstops,
                            give the name of a colormap, or provide a list of colormaps.
    ======================  =========================================================

    **Simple Renderer**

    A simple renderer is a renderer that uses one symbol only.

    ======================  =========================================================
    **Optional Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    symbol_type             optional string. This is the type of symbol the user
                            needs to create.  Valid inputs are: simple, picture, text,
                            or carto.  The default is simple.
    ----------------------  ---------------------------------------------------------
    symbol_style            optional string. This is the symbology used by the
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
    ----------------------  ---------------------------------------------------------
    description             Description of the renderer.
    ----------------------  ---------------------------------------------------------
    rotation_expression     A constant value or an expression that derives the angle
                            of rotation based on a feature attribute value. When an
                            attribute name is specified, it's enclosed in square
                            brackets.
    ----------------------  ---------------------------------------------------------
    rotation_type           String value which controls the origin and direction of
                            rotation on point features. If the rotationType is
                            defined as arithmetic, the symbol is rotated from East in
                            a counter-clockwise direction where East is the 0 degree
                            axis. If the rotationType is defined as geographic, the
                            symbol is rotated from North in a clockwise direction
                            where North is the 0 degree axis.

                            Must be one of the following values:

                            + arithmetic
                            + geographic

    ----------------------  ---------------------------------------------------------
    visual_variables        An array of objects used to set rendering properties.
    ======================  =========================================================

    **Heatmap Renderer**

    The HeatmapRenderer renders point data into a raster visualization that emphasizes
    areas of higher density or weighted values.

    ======================  =========================================================
    **Optional Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    blur_radius             The radius (in pixels) of the circle over which the
                            majority of each point's value is spread.
    ----------------------  ---------------------------------------------------------
    field                   This is optional as this renderer can be created if no
                            field is specified. Each feature gets the same
                            value/importance/weight or with a field where each
                            feature is weighted by the field's value.
    ----------------------  ---------------------------------------------------------
    max_intensity           The pixel intensity value which is assigned the final
                            color in the color ramp.
    ----------------------  ---------------------------------------------------------
    min_intensity           The pixel intensity value which is assigned the initial
                            color in the color ramp.
    ----------------------  ---------------------------------------------------------
    ratio                   A number between 0-1. Describes what portion along the
                            gradient the colorStop is added.
    ----------------------  ---------------------------------------------------------
    show_none               Boolean. Determines the alpha value of the base color for
                            the heatmap. Setting this to ``True`` covers an entire map
                            with the base color of the heatmap. Default is ``False``.
    ======================  =========================================================

    **Predominance/Unique Renderer**

    This renderer symbolizes features based on one or more matching string attributes.

    ======================  =========================================================
    **Optional Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    background_fill_symbol  A symbol used for polygon features as a background if the
                            renderer uses point symbols, e.g. for bivariate types &
                            size rendering. Only applicable to polygon layers.
                            PictureFillSymbols can also be used outside of the Map
                            Viewer for Size and Predominance and Size renderers.
    ----------------------  ---------------------------------------------------------
    default_label           Default label for the default symbol used to draw
                            unspecified values.
    ----------------------  ---------------------------------------------------------
    default_symbol          Symbol used when a value cannot be matched.
    ----------------------  ---------------------------------------------------------
    field1, field2, field3  Attribute field renderer uses to match values.
    ----------------------  ---------------------------------------------------------
    field_delimiter         String inserted between the values if multiple attribute
                            fields are specified.
    ----------------------  ---------------------------------------------------------
    rotation_expression     A constant value or an expression that derives the angle
                            of rotation based on a feature attribute value. When an
                            attribute name is specified, it's enclosed in square
                            brackets. Rotation is set using a visual variable of type
                            rotation info with a specified field or value expression
                            property.
    ----------------------  ---------------------------------------------------------
    rotation_type           String property which controls the origin and direction
                            of rotation. If the rotation type is defined as
                            arithmetic the symbol is rotated from East in a
                            counter-clockwise direction where East is the 0 degree
                            axis. If the rotation type is defined as geographic, the
                            symbol is rotated from North in a clockwise direction
                            where North is the 0 degree axis.
                            Must be one of the following values:

                            + arithmetic
                            + geographic

    ----------------------  ---------------------------------------------------------
    arcade_expression       An Arcade expression evaluating to either a string or a
                            number.
    ----------------------  ---------------------------------------------------------
    arcade_title            The title identifying and describing the associated
                            Arcade expression as defined in the valueExpression
                            property.
    ----------------------  ---------------------------------------------------------
    visual_variables        An array of objects used to set rendering properties.
    ----------------------  ---------------------------------------------------------
    unique_values           Optional list of dictionaries.  If you want to define the
                            unique values listed in the renderer, specify the list
                            using this variable.  The format of each unique value is
                            as follows:


                                | {
                                |     "value" : <value>,
                                |     "label" : <label value>,
                                |     "description" : <optional text description>,
                                |     "symbol" : {...symbol...}
                                | }

    ======================  =========================================================

    **Class Breaks Renderer**

    A class breaks renderer symbolizes based on the value of some numeric attribute.

    ======================  =========================================================
    **Optional Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    background_fill_symbol  A symbol used for polygon features as a background if the
                            renderer uses point symbols, e.g. for bivariate types &
                            size rendering. Only applicable to polygon layers.
                            PictureFillSymbols can also be used outside of the Map
                            Viewer for Size and Predominance and Size renderers.
    ----------------------  ---------------------------------------------------------
    default_label           Default label for the default symbol used to draw
                            unspecified values.
    ----------------------  ---------------------------------------------------------
    default_symbol          Symbol used when a value cannot be matched.
    ----------------------  ---------------------------------------------------------
    method                  Determines the classification method that was used to
                            generate class breaks.

                            Must be one of the following values:

                            + esriClassifyDefinedInterval
                            + esriClassifyEqualInterval
                            + esriClassifyGeometricalInterval
                            + esriClassifyNaturalBreaks
                            + esriClassifyQuantile
                            + esriClassifyStandardDeviation
                            + esriClassifyManual

    ----------------------  ---------------------------------------------------------
    field                   Attribute field used for renderer.
    ----------------------  ---------------------------------------------------------
    min_value               The minimum numeric data value needed to begin class
                            breaks.
    ----------------------  ---------------------------------------------------------
    normalization_field     Used when normalizationType is field. The string value
                            indicating the attribute field by which the data value is
                            normalized.
    ----------------------  ---------------------------------------------------------
    normalization_total     Used when normalizationType is percent-of-total, this
                            number property contains the total of all data values.
    ----------------------  ---------------------------------------------------------
    normalization_type      Determine how the data was normalized.

                            Must be one of the following values:

                            + esriNormalizeByField
                            + esriNormalizeByLog
                            + esriNormalizeByPercentOfTotal
    ----------------------  ---------------------------------------------------------
    rotation_expression     A constant value or an expression that derives the angle
                            of rotation based on a feature attribute value. When an
                            attribute name is specified, it's enclosed in square
                            brackets.
    ----------------------  ---------------------------------------------------------
    rotation_type           A string property which controls the origin and direction
                            of rotation. If the rotation_type is defined as
                            arithmetic, the symbol is rotated from East in a
                            couter-clockwise direction where East is the 0 degree
                            axis. If the rotationType is defined as geographic, the
                            symbol is rotated from North in a clockwise direction
                            where North is the 0 degree axis.

                            Must be one of the following values:

                            + arithmetic
                            + geographic

    ----------------------  ---------------------------------------------------------
    arcade_expression       An Arcade expression evaluating to a number.
    ----------------------  ---------------------------------------------------------
    arcade_title            The title identifying and describing the associated
                            Arcade expression as defined in the arcade_expression
                            property.
    ----------------------  ---------------------------------------------------------
    visual_variables        An object used to set rendering options.
    ======================  =========================================================


    **Dot Density Renderer**

    A class breaks renderer symbolizes based on the value of some numeric attribute.

    ======================  =========================================================
    **Optional Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    attributes              Required List. The fields, labels and colors to add to
                            the web map.  The list consists of dictionarys with the
                            following keys:

                            ===============     ====================================================================
                            **Parameter**        **Description**
                            ---------------     --------------------------------------------------------------------
                            field               Required String.  Name of the dataset field
                            ---------------     --------------------------------------------------------------------
                            label               Required String.  Descriptive name of the field.
                            ---------------     --------------------------------------------------------------------
                            color               Required List. A integer array consisting of R,G,B,A values
                            ===============     ====================================================================

                            If the field name is not in the SeDF, then an error will be raised on renderering.

    ----------------------  ---------------------------------------------------------
    dot_value               Required Float. The unit value of what 1 dot equals.
    ----------------------  ---------------------------------------------------------
    ref_scale               Required Int. The reference scale of the dots.
    ----------------------  ---------------------------------------------------------
    unit                    Required string.  A label of the unit which each dot
                            means.
    ----------------------  ---------------------------------------------------------
    blend_dots              Optional boolean.  Allows for the dots to overlap.
    ----------------------  ---------------------------------------------------------
    size                    Optional float. The size of the dot on the density map.
    ----------------------  ---------------------------------------------------------
    background              Optional List.  A color background as a list of [r,g,b,a]
                            values.  The default is no background [0,0,0,0].
    ======================  =========================================================


    :return: A dictionary of the renderer.

    """
    import numpy as np

    if "alpha" in symbol_args:
        alpha = symbol_args["alpha"]
    else:
        alpha = 1

    renderer = None
    vv = []
    if "size_field" in symbol_args:
        size_field = symbol_args.pop("size_field")
        vv.append(
            _size_info(
                field=size_field,
                min_value=sdf_or_series[size_field].min(),
                max_value=sdf_or_series[size_field].max(),
                min_size=symbol_args.pop("min_size", 1),
                max_size=symbol_args.pop("max_size", 24.1),
                unit=symbol_args.pop("size_units", "unknown"),
            )
        )
    if "ci_field" in symbol_args:
        color_field = symbol_args.pop("ci_field")
        vv.append(
            _color_info(
                field=color_field,
                values=sdf_or_series[color_field],
                steps=symbol_args.pop("ci_steps", 3),
                colors=symbol_args.pop("ci_color", "Reds_r"),
            )
        )
    if "opacity_expression" in symbol_args and "opacity_stops" in symbol_args:
        vv.append(
            {
                "type": "transparencyInfo",
                "valueExpression": symbol_args.pop("opacity_expression"),
                "valueExpressionTitle": "Opacity Expression",
                "stops": symbol_args.pop("opacity_stops"),
            }
        )
    if render_type is None and str(geometry_type).lower() != "raster":
        render_type = "s"
    elif render_type is None and str(geometry_type).lower() == "raster":
        render_type = "str"
    elif render_type.lower() not in RENDERER_TYPES:
        raise Exception("Invalid Renderer type.")
    else:
        render_type = render_type.lower()
    if render_type == "d" and geometry_type == "polygon":
        import pandas as pd

        if isinstance(sdf_or_series, pd.DataFrame) == False:
            raise Exception("DotDensity only works for Polygon DataFrames")

        attributes = symbol_args.pop("attributes")
        dot_value = symbol_args.pop("dot_value")
        ref_scale = symbol_args.pop("ref_scale")
        unit = symbol_args.pop("unit")
        blend_dots = symbol_args.pop("blend_dots", False)
        shape = symbol_args.pop("shape", "s")
        size = symbol_args.pop("size", 1)
        background = symbol_args.pop("background", None)
        seed = 1
        return _DotDensity(
            df=sdf_or_series,
            attributes=attributes,
            dot_value=dot_value,
            ref_scale=ref_scale,
            unit=unit,
            blend_dots=blend_dots,
            shape=shape,
            size=size,
            background=background,
            seed=seed,
        ).renderer
    elif render_type == "d" and geometry_type != "polygon":
        raise Exception("Dot Density is only supported by polygons")
    if render_type == "s":
        renderer = generate_simple(
            geometry_type=geometry_type,
            sdf_or_series=sdf_or_series,
            label=label,
            symbol_type=symbol_args.pop("symbol_type", None),
            symbol_style=symbol_args.pop("symbol_style", None),
            colors=colors,
            alpha=alpha,
            visual_variables=vv,
            **symbol_args,
        )
        return renderer
    elif render_type.lower() == "h":
        renderer = generate_heatmap(
            sdf_or_series=sdf_or_series,
            colors=colors,
            alpha=alpha,
            blur_radius=symbol_args.pop("blur_radius", 10),
            field=symbol_args.pop("field", None),
            max_intensity=symbol_args.pop("max_intensity", 10),
            min_intensity=symbol_args.pop("min_intensity", 0),
            ratio=symbol_args.pop("ratio", 0.01),
            stops=symbol_args.pop("stops", 3),
            show_none=symbol_args.pop("show_none", False),
        )
        return renderer
    elif render_type in ["u", "p"] and "field1" in symbol_args:
        renderer = generate_unique(
            geometry_type=geometry_type,
            sdf_or_series=sdf_or_series,
            symbol_type=symbol_args.pop("symbol_type", None),
            symbol_style=symbol_args.pop("symbol_style", None),
            colors=colors,
            alpha=alpha,
            visual_variables=vv,
            **symbol_args,
        )
        return renderer
    elif (
        render_type in ["u-a", "u"]
        and "field1" not in symbol_args
        and "arcade_expression" in symbol_args
    ):
        if sdf_or_series is None:
            raise ValueError(
                "sdf_or_series must be a Pandas' Series"
                + " or Pandas DataFrame for this type of renderer"
            )
        st = symbol_args.pop("symbol_type", None)
        ss = symbol_args.pop("symbol_style", None)
        default_symbol = symbol_args.pop("default_symbol", None)
        if default_symbol is None:
            if isinstance(colors, (list, tuple)):
                ccmap = colors[0]
            else:
                ccmap = colors
            default_symbol = create_symbol(
                geometry_type=geometry_type.lower(),
                symbol_type=st,
                symbol_style=ss,
                colors=ccmap,
                **symbol_args,
            )
        field_delimiter = symbol_args.pop("field_delimiter", ",")
        rotation_expression = symbol_args.pop("rotation_expression", None)
        rotation_type = symbol_args.pop("rotation_type", "arithmetic")

        renderer = {
            "type": "uniqueValue",
            "defaultLabel": symbol_args.pop("default_label", "Other"),
            "defaultSymbol": default_symbol,
            "fieldDelimiter": field_delimiter,
            "rotationExpression": rotation_expression,
            "rotationType": rotation_type,
            "valueExpression": symbol_args.pop("arcade_expression", None),
            "valueExpressionTitle": symbol_args.pop("arcade_title", None),
            "visualVariables": symbol_args.pop("visual_variables", vv),
        }
        if "unique_values" not in symbol_args:
            raise ValueError("unique_values must be provided if field1 is not given.")
        renderer["uniqueValueInfos"] = symbol_args.pop("unique_values")
    elif render_type == "v":
        renderer = {
            "type": "vectorField",
            "visualVariables": symbol_args.pop("visual_variables", vv),
            "style": symbol_args.pop("style"),
            "rotationType": symbol_args.pop("rotation_type", "arithmetic"),
            "flowRepresentation": symbol_args.pop("flow", "flow_from"),
            "attributeField": symbol_args.pop("attribute_field", None),
        }
    elif render_type == "c":
        renderer = generate_classbreaks(
            geometry_type=geometry_type,
            sdf_or_series=sdf_or_series,
            symbol_type=symbol_args.pop("symbol_type", None),
            symbol_style=symbol_args.pop("symbol_style", None),
            colors=colors,
            alpha=alpha,
            visual_variables=vv,
            **symbol_args,
        )
        return renderer
    elif render_type == "str":
        renderer = {
            "computeGamma": symbol_args.pop("compute_gamma", True),
            "dra": symbol_args.pop("dra", None),
            "gamma": symbol_args.pop("gamma", None),
            "max": symbol_args.pop("max_value", None),
            "min": symbol_args.pop("min_value", None),
            "maxPercent": symbol_args.pop("max_percent", None),
            "minPercent": symbol_args.pop("min_percent", None),
            "numberOfStandardDeviations": symbol_args.pop("std", None),
            "sigmoidStrengthLevel": symbol_args.pop("sigmoid", None),
            "statistics": symbol_args.pop("statistics", None),
            "stretchType": symbol_args.pop("type", "none"),
            "type": "rasterStretch",
            "useGamma": symbol_args.pop("use_gamma", False),
        }
        return renderer
    elif render_type == "t":
        renderer = {
            "type": "temporal",
            "latestObservationRenderer": symbol_args.pop(
                "latest_observation",
                generate_simple(
                    geometry_type="point",
                    label="Latest",
                    render_type="s",
                    colors=colors,
                    sdf_or_series=sdf_or_series,
                    **symbol_args,
                ),
            ),
            "observationRenderer": symbol_args.pop(
                "observation",
                generate_simple(
                    geometry_type="point",
                    label="Observation",
                    render_type="s",
                    colors=colors,
                    sdf_or_series=sdf_or_series,
                    **symbol_args,
                ),
            ),
            "trackRenderer": symbol_args.pop(
                "track",
                generate_simple(
                    geometry_type="polyline",
                    label="Track",
                    render_type="s",
                    colors=colors,
                    sdf_or_series=sdf_or_series,
                    **symbol_args,
                ),
            ),
        }
        return renderer
    return renderer
