from distutils.version import LooseVersion
import json, logging
import warnings

from pandas import DataFrame
import numpy as np
from arcgis import __version__
from arcgis.geometry import Geometry, SpatialReference

import pyarrow
import pyarrow.feather as feather
import pyarrow.parquet as parquet
from arcgis.geometry._types import _check_geometry_engine
from arcgis.features.geo._array import GeoArray

_logging = logging.getLogger()

_HASARCPY, _HASSHAPELY = _check_geometry_engine()

_METADATA_VERSION = "0.4.0"
# reference: https://github.com/geopandas/geo-arrow-spec

# Metadata structure:
# {
#     "geo": {
#         "columns": {
#             "<name>": {
#                 "crs": "<WKT or None: REQUIRED>",
#                 "encoding": "WKB",
#                 "geometry_tye" : "<OGC Types>",
#                 "bbox" : [xmin, ymin, xmax, ymax]
#             }
#         },
#         "creator": {
#             "library": "arcgis",
#             "version": "<arcgis.__version__>"
#         }
#         "primary_column": "<str: REQUIRED>",
#         "schema_version": "<METADATA_VERSION>"
#     }
# }


def _create_metadata(df):
    """Create and encode geo metadata dict.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """
    _gt_lu = {
        "point": "Point",
        "polyline": "MultiLineString",
        "polygon": "MultiPolygon",
        "multipoint": "MultiPoint",
        "unknown": "Unknown",
        "extent": "Unknown",
    }
    # Construct metadata for each geometry
    column_metadata = {}
    for col in df.columns[df.dtypes == "geometry"]:
        # series = df[col]

        if _HASARCPY:
            sr = SpatialReference(df.spatial.sr).as_arcpy.exportToString()
        else:
            sr = f"ESPG:{df.spatial.sr.get('wkid', 4326)}"
        gt = [_gt_lu[g.lower()] for g in df.spatial.geometry_type]
        if len(gt) == 1:
            gt = gt[0]
        column_metadata[col] = {
            "crs": sr,  # df.spatial.sr[0],#series.crs.to_wkt() if series.crs else None,
            "encoding": "WKB",
            "bbox": df.spatial.full_extent,
            "geometry_type": gt,
        }

    return {
        "primary_column": df.spatial.name,
        "columns": column_metadata,
        "schema_version": _METADATA_VERSION,
        "creator": {"library": "arcgis", "version": __version__},
    }


def _encode_metadata(metadata):
    """Encode metadata dict to UTF-8 JSON string

    Parameters
    ----------
    metadata : dict

    Returns
    -------
    UTF-8 encoded JSON string
    """
    return json.dumps(metadata).encode("utf-8")


def _decode_metadata(metadata_str):
    """Decode a UTF-8 encoded JSON string to dict

    Parameters
    ----------
    metadata_str : string (UTF-8 encoded)

    Returns
    -------
    dict
    """
    if metadata_str is None:
        return None

    return json.loads(metadata_str.decode("utf-8"))


def _validate_dataframe(df):
    """Validate that the Spatially Enabled DataFrame conforms to requirements for writing
    to Parquet format.

    Raises `ValueError` if the Spatially Enabled DataFrame is not valid.

    copied from `pandas.io.parquet`

    Parameters
    ----------
    df : pd.DataFrame
    """

    if not isinstance(df, DataFrame):
        raise ValueError("Writing to Parquet/Feather only supports IO with DataFrames")

    # must have value column names (strings only)
    if df.columns.inferred_type not in {"string", "unicode", "empty"}:
        raise ValueError("Writing to Parquet/Feather requires string column names")

    # index level names must be strings
    valid_names = all(
        isinstance(name, str) for name in df.index.names if name is not None
    )
    if not valid_names:
        raise ValueError("Index level names must be strings")


def _validate_metadata(metadata):
    """Validate geo metadata.
    Must not be empty, and must contain the structure specified above.

    Raises ValueError if metadata is not valid.

    Parameters
    ----------
    metadata : dict
    """

    if not metadata:
        raise ValueError("Missing or malformed geo metadata in Parquet/Feather file")

    required_keys = ("primary_column", "columns")
    for key in required_keys:
        if metadata.get(key, None) is None:
            raise ValueError(
                "'geo' metadata in Parquet/Feather file is missing required key: "
                "'{key}'".format(key=key)
            )

    if not isinstance(metadata["columns"], dict):
        raise ValueError("'columns' in 'geo' metadata must be a dict")

    # Validate that geometry columns have required metadata and values
    required_col_keys = ("crs", "encoding")
    for col, column_metadata in metadata["columns"].items():
        for key in required_col_keys:
            if key not in column_metadata:
                raise ValueError(
                    "'geo' metadata in Parquet/Feather file is missing required key "
                    "'{key}' for column '{col}'".format(key=key, col=col)
                )

        if column_metadata["encoding"] != "WKB":
            raise ValueError("Only WKB geometry encoding is supported")


def _sedf_to_arrow(df, index=None):
    """
    Helper function with main, shared logic for to_parquet/to_feather.
    """
    from pyarrow import Table

    _validate_dataframe(df)

    # create geo metadata before altering incoming data frame
    geo_metadata = _create_metadata(df)

    table = Table.from_pandas(df, preserve_index=index)

    # Store geopandas specific file-level metadata
    # This must be done AFTER creating the table or it is not persisted
    metadata = table.schema.metadata
    metadata.update({b"geo": _encode_metadata(geo_metadata)})
    fin = table.replace_schema_metadata(metadata)

    return fin


def _to_parquet(
    df: "DataFrame", path: str, index: bool = None, compression: str = "gzip", **kwargs
) -> str:
    """
    Write a Spatially Enabled DataFrame to the Parquet format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow'.

    WARNING: this is an initial implementation of Parquet file support and
    associated metadata.  This is tracking version 0.1.0 of the metadata
    specification at:
    https://github.com/geopandas/geo-arrow-spec

    .. versionadded:: 1.9

    Parameters
    ----------
    path : str, path object
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    kwargs
        Additional keyword arguments passed to pyarrow.parquet.write_table().
    """
    table = _sedf_to_arrow(df, index=index)
    parquet.write_table(table, path, compression=compression, **kwargs)
    return path


def _to_feather(df, path, index=None, compression=None, **kwargs) -> str:
    """
    Write a Spatially Enabled DataFrame to the Feather format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow' >= 0.17.

    WARNING: this is an initial implementation of Feather file support and
    associated metadata.  This is tracking version 0.1.0 of the metadata
    specification at:
    https://github.com/geopandas/geo-arrow-spec

    .. versionadded:: 1.9

    Parameters
    ----------
    path : str, path object
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    compression : {'zstd', 'lz4', 'uncompressed'}, optional
        Name of the compression to use. Use ``"uncompressed"`` for no
        compression. By default uses LZ4 if available, otherwise uncompressed.
    kwargs
        Additional keyword arguments passed to pyarrow.feather.write_feather().
    """

    if pyarrow.__version__ < LooseVersion("0.17.0"):
        raise ImportError("pyarrow >= 0.17 required for Feather support")

    table = _sedf_to_arrow(df, index=index)
    feather.write_feather(table, path, compression=compression, **kwargs)
    return path


def _arrow_to_sedf(table) -> "pandas.DataFrame":
    """
    Helper function with main, shared logic for read_parquet/read_feather.
    """
    df = table.to_pandas()

    metadata = table.schema.metadata
    if b"geo" not in metadata:
        raise ValueError(
            """Missing geo metadata in Parquet/Feather file.
            Use pandas.read_parquet/read_feather() instead."""
        )

    try:
        metadata = _decode_metadata(metadata.get(b"geo", b""))

    except (TypeError, json.decoder.JSONDecodeError):
        raise ValueError("Missing or malformed geo metadata in Parquet/Feather file")

    _validate_metadata(metadata)

    # Find all geometry columns that were read from the file.  May
    # be a subset if 'columns' parameter is used.
    geometry_columns = df.columns.intersection(metadata["columns"])

    if not len(geometry_columns):
        raise ValueError(
            """No geometry columns are included in the columns read from
            the Parquet/Feather file.  To read this file without geometry columns,
            use pandas.read_parquet/read_feather() instead."""
        )

    geometry = metadata["primary_column"]

    # Missing geometry likely indicates a subset of columns was read;
    # promote the first available geometry to the primary geometry.
    if len(geometry_columns) and geometry not in geometry_columns:
        geometry = geometry_columns[0]

        # if there are multiple non-primary geometry columns, raise a warning
        if len(geometry_columns) > 1:
            _logging.warning(
                "Multiple non-primary geometry columns read from Parquet/Feather "
                "file. The first column read was promoted to the primary geometry."
            )

    # Convert the WKB columns that are present back to geometry.

    if _HASARCPY:
        for col in geometry_columns:
            array = np.empty(len(table.column(geometry)), "O")
            array[:] = [Geometry(i.as_py()) for i in table.column(geometry)]
            df[col] = array
            df.spatial.set_geometry(geometry)
            for key in metadata["columns"].keys():
                data = metadata["columns"][key]
                if key == metadata["primary_column"]:
                    if (
                        "crs" in data
                        and data["crs"]
                        and data["crs"].find("ESPG:") == -1
                    ):
                        df.spatial.sr = {"wkt": data["crs"]}
                    elif (
                        "crs" in data and data["crs"] and data["crs"].find("ESPG:") > -1
                    ):
                        df.spatial.sr = {"wkid": int(data["crs"].split(":")[0])}
                    else:
                        df.spatial.sr = {"wkid": 4326}
                else:
                    df.spatial.name = key
                    if (
                        "crs" in data
                        and data["crs"]
                        and data["crs"].find("ESPG:") == -1
                    ):
                        df.spatial.sr = {"wkt": data["crs"]}
                    elif (
                        "crs" in data and data["crs"] and data["crs"].find("ESPG:") > -1
                    ):
                        df.spatial.sr = {"wkid": int(data["crs"].split(":")[0])}
                    else:
                        df.spatial.sr = {"wkid": 4326}
            if df.spatial.name != geometry:
                df.spatial.set_geometry(geometry)
    else:
        from geomet import wkb
        from geomet import esri as _esri

        for col in geometry_columns:
            array = np.empty(len(table.column(geometry)), "O")
            array[:] = [
                Geometry(_esri.dumps(wkb.loads(i.as_py())))
                for i in table.column(geometry)
            ]
            df[col] = array
            df.spatial.set_geometry(geometry)
    return df


def _read_parquet(path, columns=None, **kwargs):
    """
    Load a Parquet object from the file path, returning a Spatially Enabled DataFrame.

    You can read a subset of columns in the file using the ``columns`` parameter.
    However, the structure of the returned Spatially Enabled DataFrame will depend on which
    columns you read:

    * if no geometry columns are read, this will raise a ``ValueError`` - you
      should use the pandas `read_parquet` method instead.
    * if the primary geometry column saved to this file is not included in
      columns, the first available geometry column will be set as the geometry
      column of the returned Spatially Enabled DataFrame.

    Requires 'pyarrow'.

    .. versionadded:: 1.9

    Parameters
    ----------
    path : str, path object
    columns : list-like of strings, default=None
        If not None, only these columns will be read from the file.  If
        the primary geometry column is not included, the first secondary
        geometry read from the file will be set as the geometry column
        of the returned Spatially Enabled DataFrame.  If no geometry columns are present,
        a ``ValueError`` will be raised.
    **kwargs
        Any additional kwargs passed to pyarrow.parquet.read_table().

    Returns
    -------
    Spatially Enabled DataFrame

    Examples
    --------
    >>> df = pd.DataFrame.spatial.read_parquet("data.parquet")  # doctest: +SKIP

    Specifying columns to read:

    >>> df = pd.DataFrame.spatial.read_parquet(
    ...     "data.parquet",
    ...     columns=["geometry", "pop_est"]
    ... )  # doctest: +SKIP
    """
    kwargs["use_pandas_metadata"] = True
    table = parquet.read_table(path, columns=columns, **kwargs)

    return _arrow_to_sedf(table)


def _read_feather(path, columns=None, **kwargs):
    """
    Load a Feather object from the file path, returning a Spatially Enabled DataFrame.

    You can read a subset of columns in the file using the ``columns`` parameter.
    However, the structure of the returned Spatially Enabled DataFrame will depend on which
    columns you read:

    * if no geometry columns are read, this will raise a ``ValueError`` - you
      should use the pandas `read_feather` method instead.
    * if the primary geometry column saved to this file is not included in
      columns, the first available geometry column will be set as the geometry
      column of the returned Spatially Enabled DataFrame.

    Requires 'pyarrow' >= 0.17.

    .. versionadded:: 1.9

    Parameters
    ----------
    path : str, path object
    columns : list-like of strings, default=None
        If not None, only these columns will be read from the file.  If
        the primary geometry column is not included, the first secondary
        geometry read from the file will be set as the geometry column
        of the returned Spatially Enabled DataFrame.  If no geometry columns are present,
        a ``ValueError`` will be raised.
    **kwargs
        Any additional kwargs passed to pyarrow.feather.read_table().

    Returns
    -------
    Spatially Enabled DataFrame

    Examples
    --------
    >>> df = pd.DataFrame.spatial.read_feather("data.feather")  # doctest: +SKIP

    Specifying columns to read:

    >>> df = pd.DataFrame.spatial.read_feather(
    ...     "data.feather",
    ...     columns=["geometry", "pop_est"]
    ... )  # doctest: +SKIP
    """

    if pyarrow.__version__ < LooseVersion("0.17.0"):
        raise ImportError("pyarrow >= 0.17 required for Feather support")

    table = feather.read_table(path, columns=columns, **kwargs)
    return _arrow_to_sedf(table)
