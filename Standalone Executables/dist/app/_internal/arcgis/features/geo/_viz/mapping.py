"""
Mapping Holds the Plot function for creating a FeatureCollection JSON plus the render options
"""
import json
from typing import Optional, Union
import pandas as pd
import arcgis
from arcgis.mapping.renderer import (
    generate_classbreaks,
    generate_heatmap,
    generate_simple,
    generate_unique,
)
from arcgis.mapping.symbol import create_symbol, display_colormaps, show_styles
from arcgis.mapping.renderer import (
    generate_renderer,
    generate_classbreaks,
    generate_heatmap,
    generate_simple,
    generate_unique,
)
from arcgis.widgets import MapView

CLASSIFICATIONS = {
    "simple": {"renderer_type": "u"},  # simple
    "quantiles": {"renderer_type": "u"},  # quantiles
    "Quantiles": {"renderer_type": "u"},
}

CLASS_CMAPS = {
    "classless": "Greys",
    "unique_values": "Paired",
    "quantiles": "hot_r",
    "fisher_jenks": "hot_r",
    "equal_interval": "hot_r",
    "hot_spot": "hot_r",
}


RENDERER_TYPES = {
    "s": "simple",  ##
    "u": "unique",  ##
    "h": "heatmap",  ##
    "c": "ClassBreaks",  ##
    # "p" : 'Predominance',
    "str": "Stretch",  #
    "t": "Temporal",  #
    "v": "vector field",  #
}


def plot(
    df,
    map_widget: Optional[Union[arcgis.mapping.WebMap, MapView]] = None,
    name: Optional[str] = None,
    renderer_type: Optional[str] = None,
    symbol_type: Optional[str] = None,
    symbol_style: Optional[str] = None,
    col: Optional[Union[str, list]] = None,
    colors: Optional[Union[str, list, object]] = "jet",
    alpha: float = 1,
    **kwargs,
):
    """

    Plot draws the data on a web map. The user can describe in simple terms how to
    renderer spatial data using symbol.  To make the process simplier a palette
    for which colors are drawn from can be used instead of explicit colors.


    ======================  =========================================================
    **Explicit Argument**   **Description**
    ----------------------  ---------------------------------------------------------
    df                      Required Spatially Enabled DataFrame or GeoSeries. This is the data
                            to map.
    ----------------------  ---------------------------------------------------------
    map_widget              Optional WebMap object. This is the map to display the
                            data on.
    ----------------------  ---------------------------------------------------------
    name                    Optional string. The name to assign as a title of the map widget.
    ----------------------  ---------------------------------------------------------
    renderer_type           Optional string.  Determines the type of renderer to use
                            for the provided dataset. The default is 's' which is for
                            simple renderers.

                            Allowed values:

                            + 's' - is a simple renderer that uses one symbol only.
                            + 'u' - unique renderer symbolizes features based on one
                                    or more matching string attributes.
                            + 'c' - A class breaks renderer symbolizes based on the
                                    value of some numeric attribute.
                            + 'h' - heatmap renders point data into a raster
                                    visualization that emphasizes areas of higher
                                    density or weighted values.
    ----------------------  ---------------------------------------------------------
    symbol_type             Optional string. This is the type of symbol the user
                            needs to create.  Valid inputs are: simple, picture, text,
                            or carto.  The default is simple.
    ----------------------  ---------------------------------------------------------
    symbol_style            Optional string. This is the symbology used by the
                            geometry.  For example 's' for a Line geometry is a solid
                            line. And '-' is a dash line.

                            Allowed symbol types based on geometries:

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
    col                     Optional string/list. Field or fields used for heatmap,
                            class breaks, or unique renderers.
    ----------------------  ---------------------------------------------------------
    colors                  Optional string. The colormap, RGB array, or list of
                            either that determines the symbol color(s) for the data.
                            The default cmap is 'jet'. To get a visual representation
                            of the allowed color maps, use the **display_colormaps**
                            method.
    ----------------------  ---------------------------------------------------------
    alpha                   Optional float.  This is a value between 0 and 1 with 1
                            being the default value.  The alpha sets the transparancy
                            of the renderer when applicable.
    ======================  =========================================================

    The kwargs parameter accepts all parameters of the create_symbol method and the
    create_renderer method.


    """
    renderer = kwargs.pop("renderer", None)

    if not hasattr(df, "spatial") and not hasattr(df, "geom"):
        raise ValueError("DataFrame or Series must be spatially enabled.")

    if renderer_type is None and renderer is None and df.spatial.renderer:
        renderer = json.loads(df.spatial.renderer.json)

    if isinstance(df, pd.Series) and df.dtype.name == "geometry":
        fid = df.index.tolist()
        sdf = pd.DataFrame(data=fid, columns=["OID"])
        sdf["SHAPE"] = df
        return plot(
            df=sdf,
            map_widget=map_widget,
            name=name,
            renderer_type=renderer_type,
            symbol_type=symbol_type,
            symbol_style=symbol_style,
            col=col,
            colors=colors,
            alpha=alpha,
            **kwargs,
        )
    r = None
    if isinstance(col, str):
        col = [col]
    map_exists = True
    if symbol_type is None:
        symbol_type = "simple"
    if name is None:
        import uuid

        name = uuid.uuid4().hex[:7]
    if map_widget is None:
        map_exists = False
        map_widget = MapView()
    import string

    trantab = str.maketrans(string.punctuation, "_" * len(string.punctuation))
    col_new = [col.translate(trantab) for col in df.columns]
    col_old = df.columns.tolist()
    df.columns = col_new
    fc = df.spatial.to_feature_collection(name=name)
    df.columns = col_old
    gt = [el for el in df.spatial.geometry_type if el is not None]
    if renderer_type is None and not renderer is None:
        fc.layer["layerDefinition"]["drawingInfo"]["renderer"] = renderer
        if map_exists:
            map_widget.add_layer(fc, options={"title": name})
        else:
            map_widget.add_layer(fc, options={"title": name})
            return map_widget
        return
    elif renderer_type in [None, "s"]:
        renderer_type = "s"  # simple (default)
        r = generate_simple(
            geometry_type=gt[0].lower(),
            sdf_or_series=df,
            label=name,
            symbol_type=symbol_type,
            symbol_style=symbol_style,
            colors=colors,
            alpha=alpha,
            **kwargs,
        )
        fc.layer["layerDefinition"]["drawingInfo"]["renderer"] = r
    elif isinstance(col, str) and col not in df.columns:
        raise ValueError("Columns %s does not exist." % col)
    elif (
        isinstance(col, (tuple, list, str))
        and all([c in df.columns for c in col]) == True
        and renderer_type in ["u", "c"]
    ):
        if isinstance(col, str):
            col = [col]
        idx = 1
        if renderer_type == "u":
            for c in col:
                kwargs["field%s" % idx] = c
                idx += 1
            r = generate_unique(
                geometry_type=kwargs.pop("geometry_type", gt[0].lower()),
                sdf_or_series=df,
                symbol_type=symbol_type,
                symbol_style=symbol_style,
                colors=colors,
                alpha=alpha,
                **kwargs,
            )
        elif renderer_type == "c":
            kwargs["field"] = col[0]
            r = generate_classbreaks(
                geometry_type=kwargs.pop("geometry_type", gt[0].lower()),
                sdf_or_series=df,
                symbol_type=symbol_type,
                symbol_style=symbol_style,
                colors=colors,
                alpha=alpha,
                **kwargs,
            )
        fc.layer["layerDefinition"]["drawingInfo"]["renderer"] = r
    elif renderer_type in ["u", "u-a"]:
        r = generate_unique(
            geometry_type=kwargs.pop("geometry_type", gt[0].lower()),
            sdf_or_series=df,
            symbol_type=symbol_type,
            symbol_style=symbol_style,
            colors=colors,
            alpha=alpha,
            **kwargs,
        )
        fc.layer["layerDefinition"]["drawingInfo"]["renderer"] = r
    elif renderer_type == "h":
        r = generate_heatmap(
            sdf_or_series=df,
            colors=colors,
            alpha=alpha,
            blur_radius=kwargs.pop("blur_radius", 10),
            field=kwargs.pop("field", None),
            max_intensity=kwargs.pop("max_intensity", 10),
            min_intensity=kwargs.pop("min_intensity", 0),
            ratio=kwargs.pop("ratio", 0.01),
            stops=kwargs.pop("stops", 3),
            show_none=kwargs.pop("show_none", False),
        )
        fc.layer["layerDefinition"]["drawingInfo"]["renderer"] = r
    elif renderer_type == "str":
        r = generate_renderer(
            geometry_type=kwargs.pop("geometry_type", gt[0].lower()),
            sdf_or_series=df,
            label=name,
            symbol_type=None,
            symbol_style=None,
            render_type=renderer_type,
            colors=colors,
            alpha=alpha,
            **kwargs,
        )
        fc.layer["layerDefinition"]["drawingInfo"]["renderer"] = r
    elif renderer_type == "t":
        r = generate_renderer(
            geometry_type=kwargs.pop("geometry_type", gt[0].lower()),
            sdf_or_series=df,
            label=name,
            symbol_type=None,
            symbol_style=None,
            render_type=renderer_type,
            colors=colors,
            alpha=alpha,
            **kwargs,
        )
        fc.layer["layerDefinition"]["drawingInfo"]["renderer"] = r

    if map_exists:
        map_widget.add_layer(fc, options={"title": name})
    else:
        map_widget.add_layer(fc, options={"title": name})
        return map_widget
