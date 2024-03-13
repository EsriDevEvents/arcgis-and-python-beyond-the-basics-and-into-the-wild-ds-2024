import importlib.util
from typing import Union, Optional

from arcgis.features.geo import _is_geoenabled
from arcgis.env import active_gis
from arcgis.geometry import find_transformation, SpatialReference, Point
from arcgis.gis import GIS
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

arcpy_avail = True if importlib.util.find_spec("arcpy") else False

if arcpy_avail:
    import arcpy


def change_spatial_reference(
    input_dataframe: pd.DataFrame,
    output_spatial_reference: Union[int, SpatialReference] = 4326,
    input_spatial_reference: Optional[Union[int, SpatialReference]] = None,
    transformation_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Change the spatial reference of the input Spatially Enabled Dataframe to a desired output spatial reference,
        applying a transformation if needed if to the geographic coordinate system changes.
    Args:
        input_dataframe: Valid Spatially Enabled DataFrame
        output_spatial_reference: Optional - Desired output Spatial Reference. Default is
            4326 (WGS84).
        input_spatial_reference: Optional - Only necessary if the Spatial Reference is not
            properly defined for the input data geometry.
        transformation_name: Optional - Transformation name to be used, if needed, to
            convert between spatial references. If not explicitly provided, this will be
            inferred based on the spatial reference of the input data and desired output
            spatial reference.
    Returns: Spatially Enabled DataFrame in the desired output spatial reference.
    """

    # ensure the input spatially enabled dataframe is valid
    assert _is_geoenabled(input_dataframe), "The DataFrame does not appear to be valid."

    # if a spatial reference is set for the dataframe, just use it
    if input_dataframe.spatial.sr is not None:
        in_sr = input_dataframe.spatial.sr

    # if a spatial reference is explicitly provided, but the data does not have one set, use the one provided
    elif input_spatial_reference is not None:
        # check the input
        assert isinstance(input_spatial_reference, int) or isinstance(
            input_spatial_reference, SpatialReference
        ), (
            f"input_spatial_reference must be either an int referencing a wkid or a SpatialReference object, "
            f"not {type(input_spatial_reference)}."
        )

        if isinstance(input_spatial_reference, int):
            in_sr = SpatialReference(input_spatial_reference)
        else:
            in_sr = input_spatial_reference

    # if the spatial reference is not set, common for data coming from geojson, check if values are in lat/lon
    # range, and if so, go with WGS84, as this is likely the case if in this range
    else:
        # get the bounding values for the data
        x_min, y_min, x_max, y_max = input_dataframe.spatial.full_extent

        # check the range of the values, if in longitude and latitude range
        wgs_range = (
            True
            if (x_min > -181 and y_min > -91 and x_max < 181 and y_max < 91)
            else False
        )
        assert wgs_range, (
            "Input data for changing the spatial reference must have a spatial reference set, or one "
            "must be provided."
        )

        # if the values are in range, run with it
        in_sr = SpatialReference(4326)

    # ensure the output spatial reference is a SpatialReference object instance
    if isinstance(output_spatial_reference, SpatialReference):
        out_sr = output_spatial_reference
    else:
        out_sr = SpatialReference(output_spatial_reference)

    # copy the input spatially enabled dataframe since the project function changes the dataframe in place
    out_df = input_dataframe.copy()
    out_df.spatial.set_geometry(input_dataframe.spatial.name)

    # if a transformation was not explicitly provided, see if one is needed
    if transformation_name is None:
        # variable for saving the transformations if needed
        trns_lst = []

        # if arcpy is available, use it to find the transformation
        if arcpy_avail:
            # get any necessary transformations using arcpy, which returns only a list of transformation names
            trns_lst = arcpy.ListTransformations(in_sr.as_arcpy, out_sr.as_arcpy)

        # otherwise we will have to use the geometry rest endpoint to find transformations
        elif transformation_name is None:
            # explicitly ensure find_transformations has a gis instance
            gis = active_gis if active_gis else GIS()

            # get any transformations, if needed due to changing geographic spatial reference, as a list of dicts
            trns_lst = find_transformation(in_sr, out_sr, gis=gis)["transformations"]

        # if a transformation was not explicitly provided and one was discovered to be needed above, get it
        if len(trns_lst) > 0:
            transformation_name = trns_lst[0]

    # project to new spatial reference using the apply method since the geoaccessor project method is not working
    # reliably and only if necessary if the spatial reference is being changed
    if in_sr.wkid != out_sr.wkid:
        out_df[input_dataframe.spatial.name] = out_df[
            input_dataframe.spatial.name
        ].apply(
            lambda geom: geom.project_as(
                out_sr, transformation_name=transformation_name
            )
        )

    # ensure the spatial column is set
    if not _is_geoenabled(out_df):
        out_df.spatial.set_geometry(input_dataframe.spatial.name)

    return out_df


def _get_weighted_centroid_geometry(
    sub_df: pd.DataFrame,
    weighting_column: str,
    sptl_ref: SpatialReference,
    geom_col: str = "SHAPE",
) -> Point:
    """
    Helper function to calculate centroid coordinates.

    Args:
        sub_df: DataFrame to calculate weighted coordinates for.
        weighting_column: Column to be used for weighting.
        sptl_ref: Spatial reference for the output geometry.
        geom_col: Optional - Geometry column, defaults to 'SHAPE'.

    Returns: Tuple of weighted centroid Geometry objects.

    """
    # check if the weighting sum will be naught
    wgt_sum = sub_df[weighting_column].sum()

    # if there is a weighting sum, use it
    if wgt_sum > 0:
        wgt_x = np.average(
            sub_df[geom_col].apply(lambda geom: geom.centroid[0]),
            weights=sub_df[weighting_column],
        )
        wgt_y = np.average(
            sub_df[geom_col].apply(lambda geom: geom.centroid[1]),
            weights=sub_df[weighting_column],
        )

    # if there is not a weighting sum, just get the average centroid
    else:
        wgt_x = np.average(sub_df[geom_col].apply(lambda geom: geom.centroid[0]))
        wgt_y = np.average(sub_df[geom_col].apply(lambda geom: geom.centroid[1]))

    # create a point geometry at the weighted centroid
    wgt_geom = Point({"x": wgt_x, "y": wgt_y, "spatialReference": sptl_ref})

    return wgt_geom


def get_weighted_centroid(
    input_dataframe: pd.DataFrame, grouping_column: str, weighting_column: str
) -> pd.DataFrame:
    """
    Get a spatially enabled dataframe of weighted centroids identified by a grouping
        column.

    Args:
        input_dataframe: Spatially enabled DataFrame with a column identifying unique
            features to be grouped together.
        grouping_column: Column with values for grouping.
        weighting_column: Column with scalar value for weighting.

    Returns: Spatially enabled DataFrame of centroids with the index as values from
        the grouping column.

    """
    # check the input dataframe
    msg_valdf = "A valid Spatially Enabled DataFrame must be provided."
    assert isinstance(input_dataframe, pd.DataFrame), msg_valdf
    assert _is_geoenabled(input_dataframe), msg_valdf

    # ensure the input columns are in the dataframe
    in_cols = input_dataframe.columns
    for col in [grouping_column, weighting_column]:
        msg_col = f'{col} does not appear to be in the DataFrame columns [{", ".join(in_cols)}]'
        assert col in in_cols, msg_col

    # ensure the weighting column is a numeric column
    assert is_numeric_dtype(input_dataframe[weighting_column]), (
        f"The weighting column ({weighting_column}) must be "
        f"numeric, and it appears to be "
        f"{input_dataframe[weighting_column].dtype()}."
    )

    # get the geometry column name
    geom_col = [
        c for c in in_cols if input_dataframe[c].dtype.name.lower() == "geometry"
    ][0]

    # get the input spatial reference
    sptl_ref = input_dataframe.spatial.sr

    # calculate weighted centroids
    centroid_df = (
        input_dataframe.groupby(grouping_column)
        .apply(
            lambda sub_df: _get_weighted_centroid_geometry(
                sub_df, weighting_column, sptl_ref, geom_col
            )
        )
        .to_frame("SHAPE")
    )

    # reset the geometry to ensure updates recognized
    centroid_df.spatial.set_geometry("SHAPE")

    return centroid_df
