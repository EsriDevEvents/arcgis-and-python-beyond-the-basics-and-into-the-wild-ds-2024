"""
IO operations for Feature Classes
"""
from arcgis.auth.tools import LazyLoader
import io
import os
import uuid
import copy
from pathlib import Path, PurePath
import logging
import datetime
import ujson as _ujson
import numpy as np
import pandas as pd
from contextlib import closing
import zipfile
from arcgis.geometry import Geometry

arcgis = LazyLoader("arcgis")
try:
    arcpy = LazyLoader("arcpy", strict=True)
    HASARCPY = True
except:
    HASARCPY = False

try:
    fiona = LazyLoader("fiona", strict=True)

    HASFIONA = True
except:
    HASFIONA = False

try:
    import shapefile

    HASPYSHP = True
    SHPVERSION = [int(i) for i in shapefile.__version__.split(".")]
except:
    HASPYSHP = False

_logging = logging.getLogger(__name__)


def _fc2pandas_dtypes(describe: dict) -> dict:
    """
    returns a dtypes to cast the final dataframe to the proper datatypes (if possible).

    :returns: Dict[str, Any]
    """
    if describe is None:
        return None
    if [float(i) for i in pd.__version__.split(".")] < [1, 0, 0]:
        _lu_types = {
            "OID": np.int64,
            "SmallInteger": np.int32,
            "Integer": np.int32,
            "Single": float,
            "Double": float,
            "String": "<U",
            "Blob": "O",
            "Guid": "<U38",
            "Raster": "O",
            "Date": "<M8[us]",
        }
    else:
        from numpy import dtype as _dtype

        _lu_types = {
            "OID": pd.Int64Dtype(),
            "SmallInteger": pd.Int32Dtype(),
            "Integer": pd.Int32Dtype(),
            "Single": float,
            "Double": float,
            "String": pd.StringDtype(),
            "Blob": "O",
            "Guid": pd.StringDtype(),
            "Raster": "O",
            "Date": "<M8[us]",
            "BigInteger": pd.Int64Dtype(),
            "DateOnly": _dtype("<M8[s]"),
            "TimeOnly": _dtype("<m8[s]"),
            "TimestampOffset": _dtype("<M8[us]"),
        }
    dtypes = None
    if "fields" in describe:
        dtypes = {}
        for field in describe["fields"]:
            # if field.type.lower() in ["TEXT", "string"]:
            #    dtypes[field.name] = f"{_lu_types['String']}{field.length}"
            if field.type in _lu_types.keys():
                dtypes[field.name] = _lu_types[field.type]
            elif field.type.lower() == "geometry":
                pass  # Skip value
            else:
                dtypes[field.name] = object
    return dtypes


# --------------------------------------------------------------------------
def _infer_type(df, col):
    """
    internal function used to get the datatypes for the feature class if
    the dataframe's _field_reference is NULL or there is a column that does
    not have a dtype assigned to it.

    Input:
     dataframe - Spatially Enabled DataFrame object
    Ouput:
      field type name
    """
    nn = df[col].notnull()
    nn = list(df[nn].index)
    if len(nn) > 0:
        val = df[col][nn[0]]
        if isinstance(val, str):
            return "TEXT"
        elif isinstance(val, tuple([int] + [np.int32])):
            return "INTEGER"
        elif isinstance(val, (float, np.int64)):
            return "FLOAT"
        elif isinstance(val, datetime):
            return "DATE"
    return "TEXT"


# --------------------------------------------------------------------------
def _geojson_to_esrijson(geojson):
    """converts the geojson spec to esri json spec"""
    if geojson["type"] in ["Polygon", "MultiPolygon"]:
        return {
            "rings": geojson["coordinates"],
            "spatialReference": {"wkid": 4326},
        }
    elif geojson["type"] == "Point":
        return {
            "x": geojson["coordinates"][0],
            "y": geojson["coordinates"][1],
            "spatialReference": {"wkid": 4326},
        }
    elif geojson["type"] == "MultiPoint":
        return {
            "points": geojson["coordinates"],
            "spatialReference": {"wkid": 4326},
        }
    elif geojson["type"] in ["LineString"]:  # , 'MultiLineString']:
        return {
            "paths": [[list(gj) for gj in geojson["coordinates"]]],
            "spatialReference": {"wkid": 4326},
        }
    elif geojson["type"] in ["MultiLineString"]:
        coords = []
        for pts in geojson["coordinates"]:
            coords.append(list([list(pt) for pt in pts]))
        return {"paths": coords, "spatialReference": {"wkid": 4326}}
    return geojson


# --------------------------------------------------------------------------
def _geometry_to_geojson(geom):
    """converts the esri json spec to geojson"""
    if "rings" in geom and len(geom["rings"]) == 1:
        return {"type": "Polygon", "coordinates": geom["rings"]}
    elif "rings" in geom and len(geom["rings"]) > 1:
        return {"type": "MultiPolygon", "coordinates": geom["rings"]}
    elif geom["type"] == "Point":
        return {"coordinates": [geom["x"], geom["y"]], "type": "Point"}
    elif geom["type"] == "MultiPoint":
        return {"coordinates": geom["points"], "type": "MultiPoint"}
    elif geom["type"].lower() == "polyline" and len(geom["paths"]) <= 1:
        return {"coordinates": geom["paths"], "type": "LineString"}
    elif geom["type"].lower() == "polyline" and len(geom["paths"]) > 1:
        return {"coordinates": geom["paths"], "type": "MultiLineString"}
    return geom


# --------------------------------------------------------------------------
def _from_xy(df, x_column, y_column, sr=None, z_column=None, m_column=None, **kwargs):
    """
    Takes an X/Y Column and Creates a Point Geometry from it. Can handle Z
    and M Columns as well.

    **kwargs**

    :oid_field: - Optional String value that can be specified to preven the int64 from turning into int32.


    """
    oid_field = kwargs.pop("oid_field", None)
    from arcgis.geometry import SpatialReference, Point
    from arcgis.features.geo._array import GeoArray

    def _xy_to_geometry(x, y, sr, z=None, m=None):
        """converts x/y coordinates to Point object"""
        if z and m:
            return Point({"spatialReference": sr, "x": x, "y": y, "z": z, "m": m})
        elif z and not m:
            return Point({"spatialReference": sr, "x": x, "y": y, "z": z})
        elif m and not z:
            return Point({"spatialReference": sr, "x": x, "y": y, "m": m})
        else:
            return Point({"spatialReference": sr, "x": x, "y": y})

    if sr is None:
        sr = SpatialReference({"wkid": 4326})
    if not isinstance(sr, SpatialReference):
        if isinstance(sr, dict):
            sr = SpatialReference(sr)
        elif isinstance(sr, int):
            sr = SpatialReference({"wkid": sr})
        elif isinstance(sr, str):
            sr = SpatialReference({"wkt": sr})
    geoms = []
    v_func = np.vectorize(_xy_to_geometry, otypes="O")
    ags_geom = np.empty(len(df), dtype="O")

    if z_column and m_column:
        ags_geom[:] = v_func(
            df[x_column].values,
            df[y_column].values,
            sr,
            df[z_column].values,
            df[m_column].values,
        )
    elif z_column and not m_column:
        ags_geom[:] = v_func(
            df[x_column].values,
            df[y_column].values,
            sr,
            df[z_column].values,
            None,
        )
    elif m_column and not z_column:
        ags_geom[:] = v_func(
            df[x_column].values,
            df[y_column].values,
            sr,
            None,
            df[m_column].values,
        )
    else:
        ags_geom[:] = v_func(df[x_column].values, df[y_column].values, sr)
    oid_fields = ["oid", "fid", "objectid"]
    if oid_field:
        oid_fields.append(oid_field.lower())
    df = df.astype(
        {
            col: "int32"
            for col in df.select_dtypes("int64").columns
            if not col.lower() in oid_fields
        }
    )
    df["SHAPE"] = GeoArray(ags_geom)
    df.spatial.name
    for i in range(len(df)):
        try:
            shape = df.iloc[i]["SHAPE"]
        except:
            shape = df.loc[i]["SHAPE"]
        if "EMPTY" in shape.WKT:
            df.iat[i, df.columns.get_loc("SHAPE")] = None
    return df


def _ensure_path_string(input_path):
    """Provide hander to facilitate file path inputs to be Path object instances."""
    return str(input_path) if isinstance(input_path, Path) else input_path


def from_url(url: str) -> list:
    """
    Loads a `shapefile` from a URL endpoint into a spatially enabled dataframe.

    ===========================     ====================================================================
    **Parameter**                    **Description**
    ---------------------------     --------------------------------------------------------------------
    url                             Required String. The web location of the compressed shapefile.
    ===========================     ====================================================================

    :return: List[pd.DataFrame]

    """

    if HASPYSHP == False:
        raise Exception("pyshp is required to read hosted shapefiles.")
    import requests

    r = requests.get(url)
    with closing(r), zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        datasets = {}
        for member in archive.infolist():
            f_name_in_zip = member.filename
            filename = f_name_in_zip.split(".")[0]
            ends_with = os.path.splitext(member.filename)[1]
            if filename in datasets:
                if ends_with.lower() in [".shp", ".dbf", ".prj", ".shx"]:
                    datasets[filename][ends_with.replace(".", "")] = member
            else:
                datasets[filename] = {}
                if ends_with.lower() in [".shp", ".dbf", ".prj", ".shx"]:
                    datasets[filename][ends_with.replace(".", "")] = member
        # build readers
        readers = []
        for key in datasets.keys():
            if list(datasets[key].keys()) >= ["shp", "dbf"]:
                shx = None
                if datasets[key].get("shx", None):
                    shx = io.BytesIO()
                    shx.write(archive.read(datasets[key].get("shx", None).filename))

                shp = io.BytesIO()  #
                shp.write(archive.read(datasets[key].get("shp", None).filename))
                dbf = io.BytesIO()
                dbf.write(archive.read(datasets[key].get("dbf", None).filename))
                readers.append(shapefile.Reader(shp=shp, shx=shx, dbf=dbf))
        # construct SeDF from URL based datasets
        sdfs = []
        for reader in readers:
            records = []
            fields = [field[0] for field in reader.fields if field[0] != "DeletionFlag"]
            for idx, r in enumerate(reader.shapeRecords()):
                atr = dict(zip(fields, r.record))
                g = r.shape.__geo_interface__
                geom = Geometry(g)
                atr["SHAPE"] = geom
                records.append(atr)
                del atr
                del r, g
                del geom
            sdf = pd.DataFrame(records)
            sdf.spatial.set_geometry("SHAPE")
            sdf["OBJECTID"] = range(sdf.shape[0])
            sdf.reset_index(inplace=True)
            sdf.spatial._meta.source = url
            sdfs.append(sdf)
            del reader
        return sdfs


# --------------------------------------------------------------------------
def read_feather(
    path, spatial_column="SHAPE", columns=None, use_threads: bool = True
) -> pd.DataFrame:
    """
    Load a feather-format object from the file path.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.feather``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handler (e.g. via builtin ``open`` function)
        or ``StringIO``.
    spatial_column : str, Name of the geospatial column. The default is `SHAPE`.
       .. versionadded:: v1.8.2 of ArcGIS API for Python
    columns : sequence, default None
        If not provided, all columns are read.

        .. versionadded:: v1.8.2 of ArcGIS API for Python
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads.

       .. versionadded:: v1.8.2 of ArcGIS API for Python

    Returns
    -------
    type of object stored in file
    """
    sdf = pd.read_feather(path=path, columns=columns, use_threads=use_threads)
    if spatial_column and spatial_column in sdf.columns:
        sdf.spatial.set_geometry(spatial_column)
        sdf.spatial.name
    return sdf


# --------------------------------------------------------------------------
def from_table(filename, **kwargs):
    """
    Allows a user to read from a non-spatial table

    **Note: ArcPy is Required for this method**

    ===============     ====================================================
    **Parameter**        **Description**
    ---------------     ----------------------------------------------------
    filename            Required string or pathlib.Path. The path to the
                        table.
    ===============     ====================================================

    **Keyword Arguments**

    ===============     ====================================================
    **Parameter**        **Description**
    ---------------     ----------------------------------------------------
    fields              Optional List/Tuple. A list (or tuple) of field
                        names. For a single field, you can use a string
                        instead of a list of strings.

                        Use an asterisk (*) instead of a list of fields if
                        you want to access all fields from the input table
                        (raster and BLOB fields are excluded). However, for
                        faster performance and reliable field order, it is
                        recommended that the list of fields be narrowed to
                        only those that are actually needed.

                        Geometry, raster, and BLOB fields are not supported.

    ---------------     ----------------------------------------------------
    where               Optional String. An optional expression that limits
                        the records returned.
    ---------------     ----------------------------------------------------
    skip_nulls          Optional Boolean. This controls whether records
                        using nulls are skipped. Default is True.
    ---------------     ----------------------------------------------------
    null_value          Optional String/Integer/Float. Replaces null values
                        from the input with a new value.
    ===============     ====================================================

    :return: pd.DataFrame

    """

    filename = _ensure_path_string(filename)

    if HASARCPY and not filename.lower().endswith(".dbf"):
        import arcpy

        where = kwargs.pop("where", None)
        fields = kwargs.pop("fields", "*")
        skip_nulls = kwargs.pop("skip_nulls", True)
        null_value = kwargs.pop("null_value", None)
        arr = arcpy.da.TableToNumPyArray(
            in_table=filename,
            field_names=fields,
            where_clause=where,
            skip_nulls=skip_nulls,
            null_value=null_value,
        )
        return pd.DataFrame(arr)
    elif HASARCPY and filename.lower().endswith(".dbf"):
        import arcpy

        with arcpy.da.SearchCursor(
            in_table=filename,
            field_names=kwargs.pop("fields", "*"),
            where_clause=kwargs.pop("where", None),
        ) as scur:
            array = [row for row in scur]
            df = pd.DataFrame(array, columns=scur.fields)
            try:
                return df.convert_dtypes()
            except:
                return df
        return None
    elif filename.lower().endswith(".dbf"):
        import shapefile

        with open(filename, "rb") as f:
            reader = shapefile.Reader(dbf=f)
            return pd.DataFrame([record.as_dict() for record in reader.iterRecords()])
    elif filename.lower().find(".csv") > -1:
        return pd.read_csv(filename)

    return


# --------------------------------------------------------------------------
def to_table(geo, location, overwrite=True, sanitize_columns=False):
    """
    Exports a geo enabled dataframe to a table.

    .. note::
        Null integer values will be changed to 0 when using shapely instead
        of ArcPy due to shapely conventions.
        With ArcPy null integer values will remain null.

    ===========================     ====================================================================
    **Parameter**                    **Description**
    ---------------------------     --------------------------------------------------------------------
    location                        Required string. The output of the table.
    ---------------------------     --------------------------------------------------------------------
    overwrite                       Optional Boolean.  If True and if the table exists, it will be
                                    deleted and overwritten.  This is default.  If False, the table and
                                    the table exists, and exception will be raised.
    ---------------------------     --------------------------------------------------------------------
    sanitize_columns                Optional Boolean. If True, column names will be converted to
                                    string, invalid characters removed and other checks will be
                                    performed. The default is False.
    ===========================     ====================================================================

    :return: String
    """
    old_column, old_index = None, None
    if sanitize_columns:
        old_column = geo._data.columns.tolist()
        old_index = copy.deepcopy(geo._data.index)
        _sanitize_column_names(geo, inplace=True)
    out_location = os.path.dirname(location)
    fc_name = os.path.basename(location)
    df = geo._data.copy().convert_dtypes()
    df[df.select_dtypes(np.number).columns.tolist()] = df[
        df.select_dtypes(np.number).columns.tolist()
    ].replace({pd.NA: None})
    df[df.select_dtypes(pd.StringDtype()).columns.tolist()] = df[
        df.select_dtypes(pd.StringDtype()).columns.tolist()
    ].replace(pd.NA, "")

    if location.lower().find(".csv") > -1:
        geo._data.to_csv(location)
        if not old_column is None:
            geo._data.columns = old_column
        if not old_index is None:
            geo._data.index = old_index
        return location
    elif HASARCPY:
        import arcpy

        columns = df.columns.tolist()
        join_dummy = "AEIOUYAJC81Z"
        try:
            columns.pop(columns.index(df.spatial.name))
        except:
            pass
        dtypes = [(join_dummy, np.int64)]
        if overwrite and arcpy.Exists(location):
            arcpy.Delete_management(location)
        elif overwrite == False and arcpy.Exists(location):
            raise ValueError(
                ("overwrite set to False, Cannot " "overwrite the table. ")
            )
        fc = arcpy.CreateTable_management(out_path=out_location, out_name=fc_name)[0]
        # 2. Add the Fields and Data Types
        #
        oidfld = arcpy.da.Describe(fc)["OIDFieldName"]
        for col in columns[:]:
            if (col.lower() == oidfld.lower()) or (
                col.lower() in ["fid", "oid", "objectid"] and location.endswith(".dbf")
            ):
                pass
            elif col.lower() in ["fid", "oid", "objectid"]:
                dtypes.append((col, np.int32))
            elif df[col].dtype.name == "datetime64[s]":
                dtypes.append((col, "<m8[us]"))
            elif df[col].dtype.name.find("datetime") > -1:
                dtypes.append((col, "<M8[us]"))
                df[col] = df[col].dt.to_pydatetime()
            elif df[col].dtype.name.find("timedelta") > -1:
                dtypes.append((col, float))
                df[col] = df[col].dt.total_seconds() * 1000
            elif df[col].dtype.name == "object":
                try:
                    u = type(df[col][df[col].first_valid_index()])
                except:
                    u = pd.unique(df[col].apply(type)).tolist()[0]
                if u is None:
                    dtypes.append((col, "<U254"))
                elif u == datetime.time:
                    dtypes.append((col, "<m8[us]"))
                elif u == type(None):
                    dtypes.append((col, "<U254"))
                elif issubclass(u, str):
                    mlen = df[col].str.len().max()
                    dtypes.append((col, "<U%s" % int(mlen)))
                else:
                    try:
                        dtypes.append((col, type(df[col][0])))
                    except:
                        dtypes.append((col, "<U254"))
            elif df[col].dtype.name == "string":
                try:
                    u = type(df[col][df[col].first_valid_index()])
                except:
                    u = pd.unique(df[col].apply(type)).tolist()[0]
                if u is None:
                    dtypes.append((col, "<U254"))
                elif u == datetime.time:
                    dtypes.append((col, "<m8[us]"))
                elif issubclass(u, str):
                    mlen = df[col].str.len().max()
                    if int(mlen) == 0:
                        mlen = 1
                    dtypes.append((col, "<U%s" % int(mlen)))
                else:
                    try:
                        dtypes.append((col, type(df[col][0])))
                    except:
                        dtypes.append((col, "<U254"))
            elif df[col].dtype.name in ["int64", "Int64"]:
                dtypes.append((col, "<i8"))
            elif df[col].dtype.name == "bool":
                dtypes.append((col, np.int32))
            elif df[col].dtype.name == "boolean":
                dtypes.append((col, np.int32))
            elif isinstance(df[col].dtype, pd.CategoricalDtype):
                dtype = df[col].dtype
                if dtype.categories.dtype.name == "object":
                    try:
                        msize = max(dtype.categories.str.len())
                    except:
                        msize = 254
                    dtypes.append((col, "<U%s" % msize))
                elif dtype.categories.dtype.name.find("datetime") > -1:
                    dtypes.append((col, "<M8[us]"))
                else:
                    dtypes.append((col, dtype.categories.dtype))
            else:
                dtypes.append((col, df[col].dtype.type))

        array = np.array([], np.dtype(dtypes))
        arcpy.da.ExtendTable(fc, oidfld, array, join_dummy, append_only=False)
        # 3. Insert the Data
        #
        fields = arcpy.ListFields(fc)
        icols = [
            fld.name
            for fld in fields
            if fld.type not in ["OID", "Geometry", "FID"] and fld.name in df.columns
        ]
        dfcols = [
            fld.name
            for fld in fields
            if fld.type not in ["OID", "Geometry", "FID"] and fld.name in df.columns
        ]
        with arcpy.da.InsertCursor(fc, icols) as irows:
            bool_fld_idx = [
                irows.fields.index(col)
                for col in df.select_dtypes(pd.BooleanDtype()).columns.tolist()
            ]
            dt_fld_idx = [
                irows.fields.index(col)
                for col in df.columns
                if df[col].dtype.name.startswith("datetime")
            ]
            if len(bool_fld_idx) > 0:
                df = df.replace({pd.NA: None})
            if len(dt_fld_idx) > 0:
                df = df.replace({pd.NaT: None})
            for idx, row in df[dfcols].iterrows():
                try:
                    row = row.tolist()
                    if len(dt_fld_idx) > 0:
                        for idx in dt_fld_idx:
                            if row[idx]:
                                row[idx] = row[idx].to_pydatetime()
                    if len(bool_fld_idx) > 0:
                        for idx in bool_fld_idx:
                            if row[idx]:
                                row[idx] = int(row[idx])
                    irows.insertRow(row)

                except:
                    _logging.warn("row %s could not be inserted." % idx)
        if not old_column is None:
            geo._data.columns = old_column
        if not old_index is None:
            geo._data.index = old_index
        return fc
    if not old_column is None:
        geo._data.columns = old_column
    if not old_index is None:
        geo._data.index = old_index
    return


# --------------------------------------------------------------------------
def from_featureclass(filename, **kwargs):
    """
    Returns a GeoDataFrame (Spatially Enabled Pandas DataFrame) from a feature class.

    ===========================     ====================================================================
    **Parameter**                    **Description**
    ---------------------------     --------------------------------------------------------------------
    filename                        Required string or pathlib.Path. Full path to the feature class or URL (shapefiles only).
    ===========================     ====================================================================

    *Optional parameters when ArcPy library is available in the current environment*:
    ===========================     ====================================================================
    **Key**                         **Value**
    ---------------------------     --------------------------------------------------------------------
    sql_clause                      sql clause to parse data down. To learn more see
                                    [ArcPy Search Cursor](https://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm)
    ---------------------------     --------------------------------------------------------------------
    where_clause                    where statement. To learn more see [ArcPy SQL reference](https://pro.arcgis.com/en/pro-app/help/mapping/navigation/sql-reference-for-elements-used-in-query-expressions.htm)
    ---------------------------     --------------------------------------------------------------------
    fields                          list of strings specifying the field names.
    ---------------------------     --------------------------------------------------------------------
    spatial_filter                  A `Geometry` object that will filter the results.  This requires
                                    `arcpy` to work.
    ---------------------------     --------------------------------------------------------------------
    sr                              A Spatial reference to project (or tranform) output GeoDataFrame to.
                                    This requires `arcpy` to work.
    ---------------------------     --------------------------------------------------------------------
    datum_transformation            Used in combination with 'sr' parameter. if the spatial reference of
                                    output GeoDataFrame and input data do not share the same datum,
                                    an appropriate datum transformation should be specified.
                                    To Learn more see [Geographic datum transformations](https://pro.arcgis.com/en/pro-app/help/mapping/properties/geographic-coordinate-system-transformation.htm)
                                    This requires `arcpy` to work.
    ===========================     ====================================================================

    :return: pandas.core.frame.DataFrame

    """
    from arcgis.geometry import _types
    import json

    if HASARCPY and (
        isinstance(filename, (arcpy._mp.Layer))
        or type(filename).__name__.find("arcpy") > -1
    ):
        filename = filename
    else:
        filename = _ensure_path_string(filename)
        if not isinstance(filename, (str, Path, PurePath)):
            raise ValueError(
                f"filename must be a `str`, `Path`, or `PurePath`, not {type(filename)}"
            )
        if filename.find("http://") > -1 or filename.find("https://") > -1:
            res = from_url(url=filename)
            if len(res) == 1:
                return res[0]
            return res
    if HASARCPY:
        sql_clause = kwargs.pop("sql_clause", (None, None))
        where_clause = kwargs.pop("where_clause", None)
        fields = kwargs.pop("fields", None)
        sr = kwargs.pop("sr", None)
        spatial_filter = kwargs.pop("spatial_filter", None)
        geom = None
        try:
            desc = arcpy.da.Describe(filename)
            area_field = desc.pop("areaFieldName", None)
            length_field = desc.pop("lengthFieldName", None)
        except:  # for older versions of arcpy
            desc = arcpy.Describe(filename)
            desc = {"fields": desc.fields, "shapeType": desc.shapeType}
            area_field = getattr(desc, "areaFieldName", None)
            length_field = getattr(desc, "lengthFieldName", None)
        pandas_dtypes = _fc2pandas_dtypes(desc)

        if spatial_filter:
            _sf_lu = {
                "esriSpatialRelIntersects": "INTERSECT",
                "esriSpatialRelContains": "CONTAINS",
                "esriSpatialRelCrosses": "CROSSED_BY_THE_OUTLINE_OF",
                "esriSpatialRelEnvelopeIntersects": "INTERSECT",
                "esriSpatialRelIndexIntersects": "INTERSECT",
                "esriSpatialRelOverlaps": "INTERSECT",
                "esriSpatialRelTouches": "BOUNDARY_TOUCHES",
                "esriSpatialRelWithin": "WITHIN",
            }
            relto = _sf_lu[spatial_filter["spatialRel"]]
            geom = spatial_filter["geometry"]
            if hasattr(geom, "polygon"):
                geom = geom.polygon
            geom = geom.as_arcpy
            flname = "a" + uuid.uuid4().hex[:6]
            filename = arcpy.management.MakeFeatureLayer(
                filename, out_layer=flname, where_clause=where_clause
            )[0]
            arcpy.management.SelectLayerByLocation(
                filename, overlap_type=relto, select_features=geom
            )[0]

        shape_name = desc["shapeType"]
        if fields is None:
            fields = [
                fld.name
                for fld in desc["fields"]
                if fld.type not in ["Geometry"]
                and fld.name not in [area_field, length_field]
            ]
        cursor_fields = fields + ["SHAPE@JSON"]
        df_fields = fields + ["SHAPE"]
        count = 0
        dfs = []
        shape_field_idx = cursor_fields.index("SHAPE@JSON")
        with arcpy.da.SearchCursor(
            filename,
            field_names=cursor_fields,
            where_clause=where_clause,
            sql_clause=sql_clause,
            spatial_reference=sr,
            datum_transformation=kwargs.get("datum_transformation", None),
        ) as rows:
            srows = []
            for row in rows:
                srows.append(row)
                if len(srows) == 25000:
                    dfs.append(pd.DataFrame(srows, columns=df_fields))
                    srows = []
            if len(srows):
                dfs.append(pd.DataFrame(srows, columns=df_fields))
                srows = []
            del srows
        if len(dfs) > 0:
            df = pd.concat(dfs)
            df = df.reset_index(drop=True)
        elif len(dfs) == 1:
            df = dfs[0]
        else:
            df = pd.DataFrame([], columns=df_fields)
        q = df.SHAPE.notnull()
        none_q = ~df.SHAPE.notnull()
        gt = desc["shapeType"].lower()
        geoms = {
            "point": _types.Point,
            "polygon": _types.Polygon,
            "polyline": _types.Polyline,
            "multipoint": _types.MultiPoint,
            "envelope": _types.Envelope,
            "geometry": _types.Geometry,
        }
        import json

        df.SHAPE = df.SHAPE[q].apply(_ujson.loads).apply(geoms[gt])
        df.loc[none_q, "SHAPE"] = None
        df.spatial.set_geometry("SHAPE")
        df.spatial._meta.source = filename
        try:
            for key, value in pandas_dtypes.items():
                try:
                    df[key] = df[key].astype(value)  # , errors='ignore')
                except:
                    from numpy import dtype as _dtype

                    if value == _dtype("<m8[s]"):
                        df[key] = pd.to_datetime(df[key], format="%H:%M:%S").dt.time
                    elif value == _dtype("<M8[us]"):
                        df[key] = pd.to_datetime(df[key], utc=True)

            return df.convert_dtypes()
        except:
            return df.convert_dtypes()
    elif HASARCPY == False and HASPYSHP == True and filename.lower().find(".shp") > -1:
        geoms = []
        records = []
        reader = shapefile.Reader(filename)
        fields = [field[0] for field in reader.fields if field[0] != "DeletionFlag"]
        for r in reader.shapeRecords():
            atr = dict(zip(fields, r.record))
            g = r.shape.__geo_interface__
            g = _geojson_to_esrijson(g)
            geom = _types.Geometry(g)
            atr["SHAPE"] = geom
            records.append(atr)
            del atr
            del r, g
            del geom
        sdf = pd.DataFrame(records)
        sdf.spatial.set_geometry("SHAPE")
        sdf["OBJECTID"] = range(sdf.shape[0])
        sdf.reset_index(inplace=True)
        sdf.spatial._meta.source = filename
        return sdf
    elif (
        HASARCPY == False
        and HASFIONA == True
        and (
            filename.lower().find(".shp") > -1
            or os.path.dirname(filename).lower().find(".gdb") > -1
        )
    ):
        is_gdb = os.path.dirname(filename).lower().find(".gdb") > -1
        if is_gdb:
            # Remove deprecation warning.
            fiona_env = fiona.drivers
            if hasattr(fiona, "Env"):
                fiona_env = fiona.Env

            with fiona_env():
                from arcgis.geometry import _types

                fp = os.path.dirname(filename)
                fn = os.path.basename(filename)
                geoms = []
                atts = []
                with fiona.open(fp, layer=fn) as source:
                    meta = source.meta
                    cols = list(source.schema["properties"].keys())

                    # Get the CRS
                    try:
                        wkid = source.crs["init"].split(":")[1]
                    except:
                        wkid = 4326

                    sr = _types.SpatialReference({"wkid": int(wkid)})

                    for idx, row in source.items():
                        g = _types.Geometry(row["geometry"])
                        geoms.append(g)
                        atts.append(list(row["properties"].values()))
                        del idx, row
                    df = pd.DataFrame(data=atts, columns=cols)
                    df.spatial.set_geometry(geoms)
                    df.spatial.sr = sr
                    df.spatial._meta.source = filename
                    return df
        else:
            with fiona.drivers():
                from arcgis.geometry import _types

                geoms = []
                atts = []
                with fiona.open(filename) as source:
                    meta = source.meta
                    cols = list(source.schema["properties"].keys())
                    for idx, row in source.items():
                        geoms.append(_types.Geometry(row["geometry"]))
                        atts.append(list(row["properties"].values()))
                        del idx, row
                    df = pd.DataFrame(data=atts, columns=cols)
                    df.spatial.set_geometry(geoms)
                    df.spatial._meta.source = filename
                    return df
    else:
        if os.path.dirname(filename).lower().find(".gdb") > -1:
            message = """
            Cannot Open Geodatabase without Arcpy or Fiona
            \nPlease switch to Arcpy for full support or install fiona by this command `conda install fiona`
            """.strip()
            print(message)
            raise Exception("Failed to import Feature Class from Geodatabase specified")
        else:
            raise Exception(
                "Unsupported Data Format or Invalid Feature Class specified"
            )
    # return


# --------------------------------------------------------------------------
import functools


@functools.lru_cache(maxsize=255)
def _examine_meta(meta):
    gt_lu: dict[str, str] = {
        "esriGeometryPoint": "point",
        "esriGeometryLine": "polyline",
        "esriGeometryPolyline": "polyline",
        "esriGeometryPath": "polyline",
        "esriGeometryPolygon": "polygon",
        "esriGeometryMultiPatch": "polygon",
        "esriGeometryMultipoint": "multipoint",
    }
    sr = arcgis.geometry.SpatialReference({"wkid": 4326})
    if isinstance(meta.source, arcgis.features.FeatureLayer):
        try:
            sr = arcgis.geometry.SpatialReference(
                meta.source.properties["extent"]["spatialReference"]
            )
        except:
            sr = arcgis.geometry.SpatialReference({"wkid": 4326})

        return (gt_lu[meta.source.properties.geometryType], sr)
    return "point", sr


# --------------------------------------------------------------------------
def to_featureclass(
    geo,
    location,
    overwrite=True,
    validate=False,
    sanitize_columns=True,
    has_m=True,
    has_z=False,
):
    """
    Exports the DataFrame to a Feature class.

    .. note::
        Null integer values will be changed to 0 when using shapely instead
        of ArcPy due to shapely conventions.
        With ArcPy null integer values will remain null.

    ===============     ====================================================
    **Parameter**        **Description**
    ---------------     ----------------------------------------------------
    location            Required string. This is the output location for the
                        feature class. This should be the path and feature
                        class name.
    ---------------     ----------------------------------------------------
    overwrite           Optional Boolean. If overwrite is true, existing
                        data will be deleted and replaced with the spatial
                        dataframe.
    ---------------     ----------------------------------------------------
    validate            Optional Boolean. If True, the export will check if
                        all the geometry objects are correct upon export.
    ---------------     ----------------------------------------------------
    sanitize_columns    Optional Boolean. If True, column names will be
                        converted to string, invalid characters removed and
                        other checks will be performed. The default is False.
    ---------------     ----------------------------------------------------
    ham_m               Optional Boolean to indicate if data has linear
                        referencing (m) values. Default is False.
    ---------------     ----------------------------------------------------
    has_z               Optional Boolean to indicate if data has elevation
                        (z) values. Default is False.
    ===============     ====================================================


    :return: A string

    """
    out_location = os.path.dirname(location)

    fc_name = os.path.basename(location)
    df = geo._data.copy().convert_dtypes()
    old_idx = df.index
    df.reset_index(drop=True, inplace=True)
    if geo.name is None:
        raise ValueError("DataFrame must have geometry set.")
    if validate and geo.validate(strict=True) == False:
        raise ValueError(
            ("Mixed geometry types detected, " "cannot export to feature class.")
        )
    # deep copy of original columns to reassign them in finally of arcpy statement
    original_columns = copy.deepcopy(df.columns.tolist())
    geometry_name = geo.name or None
    if geometry_name and geometry_name != "SHAPE":
        new_columns = geo._data.columns.tolist()
        new_columns[new_columns.index(geometry_name)] = "SHAPE"
        df.columns = new_columns
        df.spatial._name = "SHAPE"
    # sanitize
    if sanitize_columns:
        # logic
        _sanitize_column_names(df.spatial, inplace=True)

    columns = df.columns.tolist()
    for col in columns[:]:
        if not isinstance(col, str):
            df.rename(columns={col: str(col)}, inplace=True)
            col = str(col)
    df[df.select_dtypes(np.number).columns.tolist()] = df[
        df.select_dtypes(np.number).columns.tolist()
    ].replace({pd.NA: None})
    df[df.select_dtypes(pd.StringDtype()).columns.tolist()] = df[
        df.select_dtypes(pd.StringDtype()).columns.tolist()
    ].replace(pd.NA, "")

    if HASARCPY:
        try:
            # 1. Create the Save Feature Class
            #
            columns = df.columns.tolist()
            join_dummy = "AEIOUYAJC81Z"
            columns.pop(columns.index(df.spatial.name))
            dtypes = [(join_dummy, np.int64)]
            if overwrite and arcpy.Exists(location):
                arcpy.Delete_management(location)
            elif overwrite == False and arcpy.Exists(location):
                raise ValueError(
                    ("overwrite set to False, Cannot " "overwrite the table. ")
                )

            notnull = df[df.spatial.name].notnull()
            idx = df[df.spatial.name][notnull].first_valid_index()
            if idx is None:
                gt, sr = _examine_meta(df.spatial._meta)
            else:
                sr = arcgis.geometry.SpatialReference(
                    df[df.spatial.name][idx]["spatialReference"]
                )
                gt = df[df.spatial.name][idx].geometry_type.upper()

            null_geom = {
                "point": pd.io.json.dumps(
                    {"x": None, "y": None, "spatialReference": sr}
                ),
                "polyline": pd.io.json.dumps({"paths": [], "spatialReference": sr}),
                "polygon": pd.io.json.dumps({"rings": [], "spatialReference": sr}),
                "multipoint": pd.io.json.dumps({"points": [], "spatialReference": sr}),
            }

            null_geom = null_geom[gt.lower()]
            sr = sr.as_arcpy
            if has_m == True:
                has_m = "ENABLED"
            else:
                has_m = None

            if has_z == True:
                has_z = "ENABLED"
            else:
                has_z = None

            fc = arcpy.CreateFeatureclass_management(
                out_location,
                spatial_reference=sr,
                geometry_type=gt,
                out_name=fc_name,
                has_m=has_m,
                has_z=has_z,
            )[0]

            # 2. Add the Fields and Data Types
            oidfld = arcpy.da.Describe(fc)["OIDFieldName"]
            for col in columns[:]:
                if col.lower() in ["fid", "oid", "objectid"]:
                    dtypes.append((col, np.int32))
                elif df[col].dtype.name == "datetime64[s]":
                    dtypes.append((col, "<m8[us]"))
                elif df[col].dtype.name.startswith("datetime"):
                    dtypes.append((col, "<M8[us]"))
                elif df[col].dtype.name.find("timedelta") > -1:
                    dtypes.append((col, float))
                    df[col] = df[col].dt.total_seconds() * 1000
                elif df[col].dtype.name == "object":
                    try:
                        u = type(df[col][df[col].first_valid_index()])
                    except:
                        u = pd.unique(df[col].apply(type)).tolist()[0]
                    if u is None:
                        dtypes.append((col, "<U254"))
                    elif u == datetime.time:
                        dtypes.append((col, "<m8[us]"))
                    elif issubclass(u, str):
                        mlen = df[col].str.len().max()
                        dtypes.append((col, "<U%s" % int(mlen)))
                    elif u is datetime.datetime:
                        dtypes.append((col, "<M8[us]"))
                    else:
                        try:
                            if df[col][idx] is None:
                                dtypes.append((col, "<U254"))
                            else:
                                dtypes.append((col, type(df[col][idx])))
                        except:
                            dtypes.append((col, "<U254"))
                elif df[col].dtype.name in ["int64", "Int64"]:
                    # Enterprise 11.1 and less do not accept Int64. Need to make float
                    gis = arcgis.env.active_gis
                    if (
                        gis is not None
                        and gis._is_agol == False
                        and gis.version <= [10, 3]
                    ):
                        dtypes.append((col, np.float64))
                    else:
                        dtypes.append((col, "<i8"))
                elif df[col].dtype.name == "bool":
                    dtypes.append((col, np.int32))
                elif df[col].dtype.name == "boolean":
                    dtypes.append((col, np.int32))
                elif isinstance(df[col].dtype, pd.CategoricalDtype):
                    dtype = df[col].dtype
                    if dtype.categories.dtype.name == "object":
                        try:
                            msize = max(dtype.categories.str.len())
                        except:
                            msize = 254
                        dtypes.append((col, "<U%s" % msize))
                    elif dtype.categories.dtype.name.find("datetime") > -1:
                        dtypes.append((col, "<M8[us]"))
                    else:
                        dtypes.append((col, dtype.categories.dtype))
                else:
                    if (
                        df[col].dtype.name == "object"
                        and df[col].first_valid_index()
                        and isinstance(df[col][idx], datetime.datetime)
                    ):
                        dtypes.append((col, "<M8[us]"))
                    elif df[col].dtype.type == str:
                        mlen = df[col].str.len().max()
                        if pd.isna(mlen):
                            mlen = 254
                        elif mlen == 0:
                            mlen = 254
                        dtypes.append((col, "<U%s" % int(mlen)))
                    else:
                        dtypes.append((col, df[col].dtype.type))
            from arcgis._impl.common._utils import chunks as _chunks

            smaller_dtypes = [[dtypes[0]] + flds for flds in _chunks(dtypes[1:], 10)]
            smaller_array = [np.array([], np.dtype(d)) for d in smaller_dtypes]
            for array in smaller_array:
                try:
                    arcpy.da.ExtendTable(
                        fc, oidfld, array, join_dummy, append_only=False
                    )
                except Exception as e:
                    print(e)

            # 3. Insert the Data
            if len(df) == 0:
                return fc
            fields = arcpy.ListFields(fc)
            icols = [
                fld.name
                for fld in fields
                if fld.type not in ["OID", "Geometry"] and fld.name in df.columns
            ] + ["SHAPE@JSON"]
            dfcols = [
                fld.name
                for fld in fields
                if fld.type not in ["OID", "Geometry"] and fld.name in df.columns
            ] + [df.spatial.name]

            with arcpy.da.InsertCursor(fc, icols) as irows:
                dt_fld_idx = [
                    irows.fields.index(col)
                    for col in df.columns
                    if df[col].dtype.name.startswith("datetime")
                ]
                bool_fld_idx = [
                    irows.fields.index(col)
                    for col in df.select_dtypes(pd.BooleanDtype()).columns.tolist()
                ]
                if len(bool_fld_idx) > 0:
                    df = df.replace({pd.NA: None})
                if len(dt_fld_idx) > 0:
                    df = df.replace({pd.NaT: None})

                def _insert_row(row):
                    row[-1] = pd.io.json.dumps(row[-1])
                    for idx in bool_fld_idx:
                        if isinstance(row[idx], (int, bool)):
                            row[idx] = int(row[idx])
                        else:
                            row[idx] = None
                    try:
                        irows.insertRow(row)
                    except Exception as e:
                        _logging.warn(
                            f"Could not insert the row because of error message: {e}. Recheck your data."
                        )

                q = df[df.spatial.name].isna()
                df.loc[q, "SHAPE"] = null_geom  # set null values to proper JSON
                replace_mappings = {
                    pd.NA: None,
                    np.nan: None,
                    np.NaN: None,
                    np.NAN: None,
                    pd.NaT: None,
                }
                np.apply_along_axis(
                    _insert_row,
                    1,
                    df.replace(replace_mappings)[dfcols].values,
                )

                df.loc[q, "SHAPE"] = None  # reset null values
        except ValueError as ve:
            df.columns = original_columns
            df.set_index(old_idx)
            fc = None
            raise
        except Exception as e:
            # something failed in try so reset columns to original columns
            # return empty item
            fc = None
            raise e
        finally:
            df.columns = original_columns
            df.set_index(old_idx)
        return fc
    elif HASPYSHP:
        if fc_name.endswith(".shp") == False:
            fc_name = "%s.shp" % fc_name
        if SHPVERSION < [2]:
            res = _pyshp_to_shapefile(df=df, out_path=out_location, out_name=fc_name)
            df.set_index(old_idx)
            return res
        else:
            res = _pyshp2(df=df, out_path=out_location, out_name=fc_name)
            df.set_index(old_idx)
            return res
    elif HASARCPY == False and HASPYSHP == False:
        raise Exception(
            (
                "Cannot Export the data without ArcPy or PyShp modules."
                " Please install them and try again."
            )
        )
    else:
        df.set_index(old_idx)
        return None


# --------------------------------------------------------------------------
def _pyshp_to_shapefile(df, out_path, out_name):
    """
    Saves a Spatially Enabled DataFrame to a Shapefile using pyshp

    :Parameters:
     :df: spatial dataframe
     :out_path: folder location to save the data
     :out_name: name of the shapefile
    :Output:
     path to the shapefile or None if pyshp isn't installed or
     spatial dataframe does not have a geometry column.
    """
    from arcgis.geometry._types import Geometry

    if HASPYSHP:
        GEOMTYPELOOKUP = {
            "Polygon": shapefile.POLYGON,
            "Point": shapefile.POINT,
            "Polyline": shapefile.POLYLINE,
            "null": shapefile.NULL,
        }
        if os.path.isdir(out_path) == False:
            os.makedirs(out_path)
        out_fc = os.path.join(out_path, out_name)
        if out_fc.lower().endswith(".shp") == False:
            out_fc += ".shp"
        geom_field = df.spatial.name
        if geom_field is None:
            return
        geom_type = "null"
        idx = df[geom_field].first_valid_index()
        if idx > -1:
            geom_type = df.loc[idx][geom_field].type
        with shapefile.Writer(
            GEOMTYPELOOKUP[geom_type], dbf=out_fc + ".dbf"
        ) as shpfile:
            shpfile.autoBalance = 1
            dfields = []
            cfields = []
            for c in df.columns:
                idx = df[c].first_valid_index() or df.index.tolist()[0]
                if idx > -1:
                    if isinstance(df[c].loc[idx], Geometry):
                        geom_field = (c, "GEOMETRY")
                    else:
                        cfields.append(c)
                        if isinstance(df[c].loc[idx], (str)):
                            shpfile.field(name=c, size=255)
                        elif isinstance(df[c].loc[idx], (int)):
                            shpfile.field(name=c, fieldType="N", size=5)
                        elif isinstance(df[c].loc[idx], (int, np.int32)):
                            shpfile.field(name=c, fieldType="N", size=10)
                        elif isinstance(df[c].loc[idx], (float, np.float64, np.int64)):
                            shpfile.field(name=c, fieldType="F", size=19, decimal=11)
                        elif (
                            isinstance(
                                df[c].loc[idx],
                                (datetime.datetime, np.datetime64),
                            )
                            or df[c].dtype.name.find("datetime") > -1
                        ):
                            shpfile.field(name=c, fieldType="D", size=8)
                            dfields.append(c)
                        elif isinstance(df[c].loc[idx], (bool)):
                            shpfile.field(name=c, fieldType="L", size=1)
                del c
                del idx
            for idx, row in df.iterrows():
                geom = row[df.spatial._name]
                if geom.type == "Polygon":
                    shpfile.poly(geom["rings"])
                elif geom.type == "Polyline":
                    shpfile.line(geom["paths"])
                elif geom.type == "Point":
                    shpfile.point(x=geom.x, y=geom.y)
                else:
                    shpfile.null()
                row = row[cfields].tolist()
                for fld in dfields:
                    idx = df[cfields].columns.tolist().index(fld)
                    if row[idx]:
                        if isinstance(row[idx].to_pydatetime(), (type(pd.NaT))):
                            row[idx] = None
                        else:
                            row[idx] = row[idx].to_pydatetime()
                shpfile.record(*row)
                del idx
                del row
                del geom

            # create the PRJ file
            try:
                from urllib import request

                wkid = df.spatial.sr["wkid"]
                if wkid == 102100:
                    wkid = 3857
                prj_filename = out_fc.replace(".shp", ".prj")

                url = "http://epsg.io/{}.esriwkt".format(wkid)

                opener = request.build_opener()
                opener.addheaders = [("User-Agent", "geosaurus")]
                resp = opener.open(url)

                wkt = resp.read().decode("utf-8")
                if len(wkt) > 0:
                    prj = open(prj_filename, "w")
                    prj.write(wkt)
                    prj.close()
            except:
                # Unable to write PRJ file.
                pass
            shpfile.close()
            del shpfile
            return out_fc
    return None


# --------------------------------------------------------------------------
def _pyshp2(df, out_path, out_name):
    """
    Saves a Spatially Enabled DataFrame to a Shapefile using pyshp v2.0

    :Parameters:
     :df: spatial dataframe
     :out_path: folder location to save the data
     :out_name: name of the shapefile
    :Output:
     path to the shapefile or None if pyshp isn't installed or
     spatial dataframe does not have a geometry column.
    """
    from arcgis.geometry._types import Geometry

    if HASPYSHP:
        GEOMTYPELOOKUP = {
            "Polygon": shapefile.POLYGON,
            "Point": shapefile.POINT,
            "Polyline": shapefile.POLYLINE,
            "null": shapefile.NULL,
        }

        if os.path.isdir(out_path) == False:
            os.makedirs(out_path)
        out_fc = os.path.join(out_path, out_name)
        if out_fc.lower().endswith(".shp") == False:
            out_fc += ".shp"
        geom_field = df.spatial.name
        if geom_field is None:
            return
        geom_type = "null"
        idx = df[geom_field].first_valid_index()
        if idx > -1:
            geom_type = df.loc[idx][geom_field].type
        shpfile = shapefile.Writer(
            target=out_fc,
            shapeType=GEOMTYPELOOKUP[geom_type],
            autoBalance=True,
        )

        # Start writing to shapefile
        dfields = []
        cfields = []
        for c in df.columns:
            idx = df[c].first_valid_index() or df.index.tolist()[0]
            if idx > -1:
                if isinstance(df[c].loc[idx], Geometry):
                    geom_field = (c, "GEOMETRY")
                    geom_column = c
                    # Since geometry is present, handle None type geometry occurrence
                    query_index = _handle_none_type_geometry(df, geom_type, geom_column)
                else:
                    cfields.append(c)
                    if isinstance(df[c].loc[idx], (str)):
                        shpfile.field(name=c, size=255)
                    elif isinstance(df[c].loc[idx], (int)):
                        shpfile.field(name=c, fieldType="N", size=5)
                    elif isinstance(df[c].loc[idx], np.int32):
                        shpfile.field(name=c, fieldType="N", size=10)
                    elif isinstance(df[c].loc[idx], (float, np.float64, np.int64)):
                        shpfile.field(name=c, fieldType="F", size=19, decimal=11)
                    elif (
                        isinstance(
                            df[c].loc[idx],
                            (datetime.datetime, np.datetime64),
                        )
                        or df[c].dtype.name.find("datetime") > -1
                    ):
                        shpfile.field(name=c, fieldType="D", size=8)
                        dfields.append(c)
                    elif isinstance(df[c].loc[idx], (bool)):
                        shpfile.field(name=c, fieldType="L", size=1)
            del c
            del idx

        for idx, row in df.iterrows():
            geom = row[df.spatial.name]
            if geom.type == "Polygon":
                shpfile.poly(geom["rings"])
            elif geom.type == "Polyline":
                shpfile.line(geom["paths"])
            elif geom.type == "Point":
                shpfile.point(x=geom.x, y=geom.y)
            else:
                shpfile.null()
            row = row[cfields].tolist()
            for fld in dfields:
                idx = df[cfields].columns.tolist().index(fld)
                if row[idx]:
                    if isinstance(row[idx].to_pydatetime(), (type(pd.NaT))):
                        row[idx] = None
                    else:
                        row[idx] = row[idx].to_pydatetime()
            for idx, value in enumerate(row):
                if value is np.nan:
                    row[idx] = None
            shpfile.record(*row)
            del idx
            del row
            del geom
        shpfile.close()

        # create the PRJ file
        try:
            from urllib import request

            wkid = df.spatial.sr["wkid"]
            if wkid == 102100:
                wkid = 3857
            prj_filename = out_fc.replace(".shp", ".prj")

            url = "http://epsg.io/{}.esriwkt".format(wkid)

            opener = request.build_opener()
            opener.addheaders = [("User-Agent", "geosaurus")]
            resp = opener.open(url)

            wkt = resp.read().decode("utf-8")
            if len(wkt) > 0:
                prj = open(prj_filename, "w")
                prj.write(wkt)
                prj.close()
        except:
            # Unable to write PRJ file.
            pass

        # Change back null columns to None
        for index, row in query_index.items():
            if row is True:
                df.iat[index, df.columns.get_loc(geom_column)] = None

        del shpfile
        return out_fc
    return None


def _handle_none_type_geometry(df, geom_type, geom_column):
    # Handle none type geometry occurrence

    # bool to see if empty
    query = df[geom_column].isnull()
    df_view = df[query]
    empty = df_view.empty

    if empty is False:
        for idx, row in df_view.iterrows():
            if df.loc[idx][geom_column] is None:
                if geom_type == "Point":
                    df.iat[idx, df.columns.get_loc(geom_column)] = Geometry(
                        {
                            "x": np.NAN,
                            "y": np.NAN,
                            "spatialReference": df.spatial.sr,
                        }
                    )
                elif geom_type == "Poyline":
                    df.iat[idx, df.columns.get_loc(geom_column)] = Geometry(
                        {"paths": []}
                    ).WKT
                elif geom_type == "Polygon":
                    df.iat[idx, df.columns.get_loc(geom_column)] = Geometry(
                        {"rings": []}
                    ).WKT
    return query


def _sanitize_column_names(
    geo,
    remove_special_char=True,
    rename_duplicates=True,
    inplace=False,
    use_snake_case=True,
):
    """
    Implementation for pd.DataFrame.spatial.sanitize_column_names()
    """
    original_col_names = list(geo._data.columns)

    # convert to string
    new_col_names = [str(x) for x in original_col_names]

    # use snake case
    if use_snake_case:
        import re

        for ind, val in enumerate(new_col_names):
            # skip reserved cols
            if val == geo.name:
                continue
            # replace Pascal and camel case using RE
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", val)
            name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
            # remove leading spaces
            name = name.lstrip(" ")
            # replace spaces with _
            name = name.replace(" ", "_")
            # clean up too many _
            name = re.sub("_+", "_", name)
            new_col_names[ind] = name

    # remove special characters
    if remove_special_char:
        for ind, val in enumerate(new_col_names):
            name = "".join(i for i in val if i.isalnum() or "_" in i)

            # remove numeral prefixes
            for ind2, element in enumerate(name):
                if element.isdigit():
                    continue
                else:
                    name = name[ind2:]
                    break
            new_col_names[ind] = name

    # fill empty column names
    for ind, val in enumerate(new_col_names):
        if val == "":
            new_col_names[ind] = "column"

    # rename duplicates
    if rename_duplicates:
        for ind, val in enumerate(new_col_names):
            if val == geo.name:
                pass
            if new_col_names.count(val) > 1:
                counter = 1
                new_name = val + str(counter)  # adds a integer suffix to column name
                while new_col_names.count(new_name) > 0:
                    counter += 1
                    new_name = val + str(
                        counter
                    )  # if a column with the suffix exists, increment suffix
                new_col_names[ind] = new_name

    for idx, name in enumerate(new_col_names):
        if name.startswith("_"):
            new_col_names[idx] = name[1:]

    # if inplace
    if inplace:
        geo._data.columns = new_col_names
    else:
        # return a new dataframe
        df = geo._data.copy()
        df.spatial.name
        df.columns = new_col_names
        return df
    return True
