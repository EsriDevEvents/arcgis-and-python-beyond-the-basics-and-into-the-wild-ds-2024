import os
import glob
import uuid
import importlib
from functools import lru_cache
import numpy as np
import pandas as pd
from distutils.version import LooseVersion
import dask
from dask.dataframe import Series, from_pandas
import dask.dataframe as dd
from dask.dataframe.core import get_parallel_type, make_meta
from dask.base import normalize_token
from dask.dataframe.extensions import (
    make_array_nonempty,
    make_scalar,
    register_dataframe_accessor,
    register_series_accessor,
)

DASK_2021_06_0 = str(dask.__version__) >= LooseVersion("2021.06.0")

if DASK_2021_06_0:
    from dask.dataframe.dispatch import make_meta_dispatch
    from dask.dataframe.backends import (
        _nonempty_index,
        meta_nonempty_dataframe,
        meta_nonempty,
    )
else:
    from dask.dataframe.core import make_meta as make_meta_dispatch
    from dask.dataframe.utils import (
        _nonempty_index,
        meta_nonempty_dataframe,
        meta_nonempty,
    )

from arcgis.geometry import BaseGeometry, Geometry
from arcgis.features import FeatureCollection
from arcgis._impl.common._isd import InsensitiveDict
from arcgis._impl.common._mixins import PropertyMap
from ._array import GeoType, GeoArray
from ._io.fileops import to_featureclass, from_featureclass
from ._index._dqtree import DaskSpatialIndex
from ._viz._dmapping import dask_plot


# -------------------------------------------------------------------------
@lru_cache(maxsize=200)
def _default_geometry():
    return Geometry({"x": 0, "y": 0, "spatialReference": {"wkid": 4326}})


# -------------------------------------------------------------------------
def _isna(value):
    """
    Check if scalar value is NA-like (None or np.nan).
    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    if value is None:
        return True
    elif isinstance(value, float) and np.isnan(value):
        return True
    else:
        return False
    # -------------------------------------------------------------------------


def _from_geometry(data: list) -> "GeoArray":
    """
    Convert a list or array of dict/geometries objects to a GeoArray.
    """

    n = len(data)

    out = []

    for idx in range(n):
        geom = data[idx]
        if isinstance(geom, BaseGeometry):
            out.append(geom)
        elif hasattr(geom, "__geo_interface__"):
            geom = Geometry(geom.__geo_interface__)
            out.append(geom)
        elif _isna(geom):
            out.append(None)
        else:
            raise TypeError("Input must be valid geometry objects: {0}".format(geom))

    aout = np.empty(n, dtype=object)
    aout[:] = out
    return GeoArray(aout)


# -------------------------------------------------------------------------
@register_dataframe_accessor("spatial")
class GeoDaskSpatialAccessor:
    """
    The Geospatial dask dataframe accessor used to perform dataset specific operations.
    """

    _sr = None
    _data = None
    _name = None
    _sindex = None
    _renderer = None

    # ----------------------------------------------------------------------
    def __init__(self, obj):
        self._data = obj
        self._fn_attr = lambda a, op: getattr(a, op)
        self._fn_method = lambda a, op, **kwargs: getattr(a, op)(**kwargs)
        self._sfn_method = lambda a, sop, **kwargs: getattr(a, sop)(**kwargs)

    # ----------------------------------------------------------------------
    @property
    def __feature_set__(self):
        """returns a dictionary representation of an Esri FeatureSet"""
        row_pieces = self._data.map_partitions(
            lambda part: part.spatial.__feature_set__
        ).compute()

        if len(row_pieces) == 1:
            return row_pieces[0]
        else:
            fs = row_pieces[0]
            [fs["features"].extend(pieces["features"]) for pieces in row_pieces[1:]]
            return fs

    # ----------------------------------------------------------------------
    @dask.delayed
    def to_feature_collection(
        self, name=None, drawing_info=None, extent=None, global_id_field=None
    ):
        """
        Converts a Spatially Enabled Dask DataFrame to a Feature Collection

        =====================  ===============================================================
        **optional argument**  **Description**
        ---------------------  ---------------------------------------------------------------
        name                   optional string. Name of the Feature Collection
        ---------------------  ---------------------------------------------------------------
        drawing_info           Optional dictionary. This is the rendering information for a
                               Feature Collection.  Rendering information is a dictionary with
                               the symbology, labelling and other properties defined.  See:
                               https://developers.arcgis.com/documentation/common-data-types/renderer-objects.htm
        ---------------------  ---------------------------------------------------------------
        extent                 Optional dictionary.  If desired, a custom extent can be
                               provided to set where the map starts up when showing the data.
                               The default is the full extent of the dataset in the Spatial
                               DataFrame.
        ---------------------  ---------------------------------------------------------------
        global_id_field        Optional string. The Global ID field of the dataset.
        =====================  ===============================================================

        :returns: FeatureCollection object
        """

        import string
        import random

        fs = self.__feature_set__
        if name is None:
            name = random.choice(string.ascii_letters) + uuid.uuid4().hex[:5]
        if extent is None:
            ext = self.full_extent
            extent = {
                "xmin": ext[0],
                "ymin": ext[1],
                "xmax": ext[2],
                "ymax": ext[3],
                "spatialReference": fs["spatialReference"],
            }

        for fld in fs["fields"]:
            if fld["name"].lower() == fs["objectIdFieldName"].lower():
                fld["editable"] = False
                fld["sqlType"] = "sqlTypeOther"
                fld["domain"] = None
                fld["defaultValue"] = None
                fld["nullable"] = False
            else:
                fld["editable"] = True
                fld["sqlType"] = "sqlTypeOther"
                fld["domain"] = None
                fld["defaultValue"] = None
                fld["nullable"] = True
        if drawing_info is None:
            import json

            di = {"renderer": json.loads(self._data.spatial.renderer.json)}
        else:
            di = drawing_info
        layer = {
            "layerDefinition": {
                "currentVersion": 10.7,
                "id": 0,
                "name": name,
                "type": "Feature Layer",
                "displayField": "",
                "description": "",
                "copyrightText": "",
                "defaultVisibility": True,
                "relationships": [],
                "isDataVersioned": False,
                "supportsAppend": True,
                "supportsCalculate": True,
                "supportsASyncCalculate": True,
                "supportsTruncate": False,
                "supportsAttachmentsByUploadId": True,
                "supportsAttachmentsResizing": True,
                "supportsRollbackOnFailureParameter": True,
                "supportsStatistics": True,
                "supportsExceedsLimitStatistics": True,
                "supportsAdvancedQueries": True,
                "supportsValidateSql": True,
                "supportsCoordinatesQuantization": True,
                "supportsFieldDescriptionProperty": True,
                "supportsQuantizationEditMode": True,
                "supportsApplyEditsWithGlobalIds": False,
                "supportsMultiScaleGeometry": True,
                "supportsReturningQueryGeometry": True,
                "hasGeometryProperties": True,
                "advancedQueryCapabilities": {
                    "supportsPagination": True,
                    "supportsPaginationOnAggregatedQueries": True,
                    "supportsQueryRelatedPagination": True,
                    "supportsQueryWithDistance": True,
                    "supportsReturningQueryExtent": True,
                    "supportsStatistics": True,
                    "supportsOrderBy": True,
                    "supportsDistinct": True,
                    "supportsQueryWithResultType": True,
                    "supportsSqlExpression": True,
                    "supportsAdvancedQueryRelated": True,
                    "supportsCountDistinct": True,
                    "supportsReturningGeometryCentroid": True,
                    "supportsReturningGeometryProperties": True,
                    "supportsQueryWithDatumTransformation": True,
                    "supportsHavingClause": True,
                    "supportsOutFieldSQLExpression": True,
                    "supportsMaxRecordCountFactor": True,
                    "supportsTopFeaturesQuery": True,
                    "supportsDisjointSpatialRel": True,
                    "supportsQueryWithCacheHint": True,
                },
                "useStandardizedQueries": False,
                "geometryType": fs["geometryType"],
                "minScale": 0,
                "maxScale": 0,
                "extent": extent,
                "drawingInfo": di,
                "allowGeometryUpdates": True,
                "hasAttachments": False,
                "htmlPopupType": "esriServerHTMLPopupTypeNone",
                "hasM": False,
                "hasZ": False,
                "objectIdField": fs["objectIdFieldName"] or "OBJECTID",
                "globalIdField": "",
                "typeIdField": "",
                "fields": fs["fields"],
                "types": [],
                "supportedQueryFormats": "JSON, geoJSON",
                "hasStaticData": True,
                "maxRecordCount": 32000,
                "standardMaxRecordCount": 4000,
                "tileMaxRecordCount": 4000,
                "maxRecordCountFactor": 1,
                "capabilities": "Query",
            },
            "featureSet": {
                "features": fs["features"],
                "geometryType": fs["geometryType"],
            },
        }
        if global_id_field is not None:
            layer["layerDefinition"]["globalIdField"] = global_id_field
        return FeatureCollection(layer)

    # ----------------------------------------------------------------------
    @property
    def name(self):
        """returns the name of the geometry column"""
        if self._name is None:
            try:
                cols = [c.lower() for c in self._data.columns]
                if any(self._data.dtypes == "geometry"):
                    name = self._data.dtypes[self._data.dtypes == "geometry"].index[0]
                    self._name = name
                    return name
                elif "shape" in cols:
                    idx = cols.index("shape")
                    self._name = self._data.columns[idx]
            except:
                raise Exception("Spatial column not defined, please use `set_geometry`")
        return self._name

    # ----------------------------------------------------------------------
    @property
    def area(self):
        """
        Returns the total area of the dataframe

        :returns: float

        >>> df.spatial.area
        143.23427

        """
        return self._data[self.name].geom.area.sum()

    # ----------------------------------------------------------------------
    @property
    def bbox(self):
        """
        Returns the total length of the dataframe

        :returns: Polygon

        >>> df.spatial.bbox
        {'rings' : [[[1,2], [2,3], [3,3],....]], 'spatialReference' {'wkid': 4326}}
        """
        parts = self._data.map_partitions(
            lambda part: self._fn_attr(part.spatial, "bbox")
        ).compute()
        if len(parts) == 1:
            df = pd.DataFrame(parts[0]["rings"][0], columns=["x", "y"])
            sr = parts[0]["spatialReference"]
        else:
            main_part = parts[0]
            sr = parts[0]["spatialReference"]
            for pt in parts[1:]:
                main_part["rings"][0].extend(pt["rings"][0])
            df = pd.DataFrame(main_part["rings"][0], columns=["x", "y"])
        xmin, ymin, xmax, ymax = (
            df.x.min(),
            df.y.min(),
            df.x.max(),
            df.y.max(),
        )
        if isinstance(sr, list) and len(sr) > 0:
            sr = sr[0]
        if xmin == xmax:
            xmin -= 0.001
            xmax += 0.001
        if ymin == ymax:
            ymin -= 0.001
            ymax += 0.001
        return Geometry(
            {
                "rings": [
                    [
                        [xmin, ymin],
                        [xmin, ymax],
                        [xmax, ymax],
                        [xmax, ymin],
                        [xmin, ymin],
                    ]
                ],
                "spatialReference": dict(sr),
            }
        )

    # ----------------------------------------------------------------------
    @property
    def centroid(self):
        """
        Returns the centroid of the dataframe

        :returns: Geometry

        >>> df.spatial.centroid
        (-14.23427, 39)

        """

        df = pd.DataFrame(
            self._data[self.name].geom.centroid.compute().tolist(),
            columns=["x", "y"],
        )
        return df.x.mean(), df.y.mean()

    # ----------------------------------------------------------------------
    @property
    def full_extent(self):
        """
        Returns the extent of the dataframe

        :returns: tuple

        >>> df.spatial.full_extent
        (-118, 32, -97, 33)

        """
        ge = self._data[self.name].geom.extent.compute()
        q = ge.notnull()
        data = ge[q].tolist()
        array = np.array(data)
        return (
            float(array[:, 0][array[:, 0] != None].min()),
            float(array[:, 1][array[:, 1] != None].min()),
            float(array[:, 2][array[:, 2] != None].max()),
            float(array[:, 3][array[:, 3] != None].max()),
        )

    # ----------------------------------------------------------------------
    @property
    @lru_cache(maxsize=10)
    def geometry_type(self):
        """
        Returns a list Geometry Types for the DataFrame
        """
        q = self._data.index < 10
        return (
            self._data.loc[q, [self.name]][  # source dataframe
                self.name
            ]  # gets the SHAPE column and records
            .geom.geometry_type.unique()  # geometry accessor  # method to call  # get unique values
            .compute()  # run the command
            .tolist()  # convert to list
        )  # return results

    # ----------------------------------------------------------------------
    def relationship(self, other, op, relation=None):
        """
        This method allows for dataframe to dataframe compairson using
        spatial relationships.  The return is a pd.DataFrame that meet the
        operations' requirements.

        =========================    =========================================================
        **Parameter**                 **Description**
        -------------------------    ---------------------------------------------------------
        other                        Required Spatially Enabled DataFrame. The geometry to
                                     perform the operation from.
        -------------------------    ---------------------------------------------------------
        op                           Optional String. The spatial operation to perform.  The
                                     allowed value are: contains,crosses,disjoint,equals,
                                     overlaps,touches, or within.

                                     - contains - Indicates if the base geometry contains the comparison geometry.
                                     - crosses -  Indicates if the two geometries intersect in a geometry of a lesser shape type.
                                     - disjoint - Indicates if the base and comparison geometries share no points in common.
                                     - equals - Indicates if the base and comparison geometries are of the same shape type and define the same set of points in the plane. This is a 2D comparison only; M and Z values are ignored.
                                     - overlaps - Indicates if the intersection of the two geometries has the same shape type as one of the input geometries and is not equivalent to either of the input geometries.
                                     - touches - Indicates if the boundaries of the geometries intersect.
                                     - within - Indicates if the base geometry is within the comparison geometry.
                                     - intersect - Intdicates if the base geometry has an intersection of the other geometry.
        -------------------------    ---------------------------------------------------------
        relation                     Optional String.  The spatial relationship type.  The
                                     allowed values are: BOUNDARY, CLEMENTINI, and PROPER.

                                     + BOUNDARY - Relationship has no restrictions for interiors or boundaries.
                                     + CLEMENTINI - Interiors of geometries must intersect. This is the default.
                                     + PROPER - Boundaries of geometries must not intersect.

                                     This only applies to contains,
        =========================    =========================================================

        :returns: Delayed DataFrame


        """
        results = self._data.map_partitions(
            lambda part: self._fn_method(
                part.spatial,
                "relationship",
                **{"other": other, "relation": relation, "op": op},
            )
        ).compute()
        return results

    # ----------------------------------------------------------------------
    def select(self, other: "Geometry"):
        """
        This operation performs a dataset wide **selection** by geometric
        intersection. A geometry or another Spatially enabled DataFrame
        can be given and `select` will return all rows that intersect that
        input geometry.  The `select` operation uses a spatial index to
        complete the task, so if it is not built before the first run, the
        function will build a quadtree index on the fly.

        **requires ArcPy or Shapely**

        :returns: pd.DataFrame (spatially enabled)

        """
        results = self._data.map_partitions(
            lambda part: self._fn_method(part.spatial, "select", **{"other": other})
        )
        return results

    # ----------------------------------------------------------------------
    def overlay(self, sdf, op="union"):
        """
        Performs spatial operation operations on two spatially enabled dataframes.

        **requires ArcPy or Shapely**

        =========================    =========================================================
        **Parameter**                 **Description**
        -------------------------    ---------------------------------------------------------
        sdf                          Required Spatially Enabled DataFrame. The geometry to
                                     perform the operation from.
        -------------------------    ---------------------------------------------------------
        op                           Optional String. The spatial operation to perform.  The
                                     allowed value are: union, erase, identity, intersection.
                                     `union` is the default operation.
        =========================    =========================================================

        :returns: Spatially enabled DataFrame (pd.DataFrame)

        """
        from ._tools._dask_overlay import overlay_dask

        if isinstance(sdf, pd.DataFrame):
            meta = pd.concat([sdf[0:0], self._data._meta])

        elif isinstance(sdf, dd.DataFrame):
            meta = pd.concat([sdf._meta, self._data._meta])

        # joined_meta = pd.concat([ddf2._meta, ddf1._meta])
        op = str(op).lower()
        if self._data.spatial.geometry_type != sdf.spatial.geometry_type:
            raise ValueError(
                ("Spatially enabled DataFrame must " "be the same geometry type.")
            )

        if (
            op == "symmetric_difference"
            and self._data.spatial.geometry_type != ["polygon"]
            and sdf.spatial.geometry_type != ["polygon"]
        ):
            raise ValueError(
                ("symmetric_difference is only supported for " "polygon geometries.")
            )

        def fn(part):
            return overlay_dask(part, sdf, op)

        results = self._data.map_partitions(fn, meta=meta, enforce_metadata=False)
        return results

    # ----------------------------------------------------------------------
    def join(
        self,
        right_df,
        how="inner",
        op="intersects",
        left_tag="left",
        right_tag="right",
    ):
        """
        Joins the current DataFrame to another spatially enabled dataframes based
        on spatial location based.

        .. note::
            requires the SEDF to be in the same coordinate system


        ======================    =========================================================
        **Parameter**              **Description**
        ----------------------    ---------------------------------------------------------
        right_df                  Required pd.DataFrame. Spatially enabled dataframe to join.
        ----------------------    ---------------------------------------------------------
        how                       Required string. The type of join:

                                    + `left` - use keys from current dataframe and retains only current geometry column
                                    + `right` - use keys from right_df; retain only right_df geometry column
                                    + `inner` - use intersection of keys from both dfs and retain only current geometry column

        ----------------------    ---------------------------------------------------------
        op                        Required string. The operation to use to perform the join.
                                  The default is `intersects`.

                                  supported perations: `intersects`, `within`, and `contains`
        ----------------------    ---------------------------------------------------------
        left_tag                  Optional String. If the same column is in the left and
                                  right dataframe, this will append that string value to
                                  the field.
        ----------------------    ---------------------------------------------------------
        right_tag                 Optional String. If the same column is in the left and
                                  right dataframe, this will append that string value to
                                  the field.
        ======================    =========================================================

        :returns:
          Delayed Dask DataFrame
        """
        if isinstance(right_df, pd.DataFrame):
            meta = pd.concat([right_df[0:0], self._data._meta])

        elif isinstance(right_df, dd.DataFrame):
            meta = pd.concat([right_df._meta, self._data._meta])

        results = self._data.map_partitions(
            lambda part: self._sfn_method(
                part.spatial,
                "join",
                **{
                    "right_df": right_df,
                    "how": how,
                    "op": op,
                    "left_tag": left_tag,
                    "right_tag": right_tag,
                },
            ),
            meta=meta,
            enforce_metadata=False,
        )
        return results

    # ----------------------------------------------------------------------
    @property
    def has_z(self):
        """
        Returns a boolean that determines if the datasets have `Z` values

        :returns: Boolean
        """
        return self._data[self.name].geom.has_z.all()

    # ----------------------------------------------------------------------
    @property
    def has_m(self):
        """
        Returns a boolean that determines if the datasets have `Z` values

        :returns: Boolean
        """
        return self._data[self.name].geom.has_m.all()

    # ----------------------------------------------------------------------
    @property
    def length(self):
        """
        Returns the total length of the dataframe

        :returns: float

        >>> df.spatial.length
        1.23427

        """
        return self._data[self.name].geom.length.sum()

    # ----------------------------------------------------------------------
    def plot(self, map_widget: "MapView" = None, renderer: dict = None):
        """Displays the Dask DataFrame on a Map Widget"""
        return dask_plot(
            df=self._data,  # dask dataframe
            map_widget=map_widget,  # map view
            renderer=renderer or self.renderer,  # display information
        )

    # ----------------------------------------------------------------------
    def project(self, spatial_reference, transformation_name=None):
        """
        Reprojects the who dataset into a new spatial reference. This is an inplace operation meaning
        that it will update the defined geometry column from the `set_geometry`.

        **This requires ArcPy or pyproj v4**

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required SpatialReference. The new spatial reference. This can be a
                                 SpatialReference object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Optional String. The geotransformation name.
        ====================     ====================================================================

        :returns: boolean
        """
        _HASARCPY, _HASSHAPELY, _HASPYPROJ = self._check_geometry_engine()
        try:
            if isinstance(spatial_reference, (int, str)) and _HASARCPY:
                import arcpy

                spatial_reference = arcpy.SpatialReference(spatial_reference)
                vals = (
                    self._data[self.name]
                    .geom.project_as(
                        **{
                            "spatial_reference": spatial_reference,
                            "transformation_name": transformation_name,
                        }
                    )
                    .compute()
                )
                self._data[self.name] = vals
                return True
            elif isinstance(spatial_reference, (int, str)) and _HASPYPROJ:
                vals = (
                    self._data[self.name]
                    .geom.project_as(
                        **{
                            "spatial_reference": spatial_reference,
                            "transformation_name": transformation_name,
                        }
                    )
                    .compute()
                )
                self._data[self.name] = vals
                return True
            else:
                return False
        except Exception as e:
            raise Exception(e)

    # ----------------------------------------------------------------------
    @lru_cache(maxsize=255)
    def _check_geometry_engine(self):
        _HASARCPY = True
        _HASSHAPELY = True
        _HASPYPROJ = True
        try:
            i = importlib.util.find_spec("arcpy")
            if i is None:
                raise ImportError("Cannot find arcpy.")
        except ImportError:
            _HASARCPY = False
        try:
            i = importlib.util.find_spec("shapely")
            if i is None:
                raise ImportError("Cannot find shapely.")
        except ImportError:
            _HASSHAPELY = False
        try:
            i = importlib.util.find_spec("pyproj")
            if i is None:
                raise ImportError("Cannot find pyproj.")
        except ImportError:
            _HASPYPROJ = False
        return _HASARCPY, _HASSHAPELY, _HASPYPROJ

    # ----------------------------------------------------------------------
    def sindex(self, reset=False) -> "DaskSpatialIndex":
        """
        Returns a dask specialized spatial index
        """

        if reset:
            self._sindex = None
        if self._sindex is None:
            results = (
                self._data.map_partitions(
                    lambda part: self._fn_method(
                        part.spatial, "sindex", **{"stype": "quadtree"}
                    )
                )
                .compute()
                .tolist()
            )
            self._sindex = DaskSpatialIndex(results)
        return self._sindex

    # ----------------------------------------------------------------------
    @property
    def renderer(self) -> "InsensitiveDict":
        """
        Define the renderer for the Dask DF.  If none is given, then the value is reset.

        :returns: InsensitiveDict
        """
        if self._renderer is None:
            self._renderer = self._build_renderer()
        return self._renderer

    # ----------------------------------------------------------------------
    @renderer.setter
    def renderer(self, renderer):
        """
        Define the renderer for the Dask DF.  If none is given, then the value is reset.

        :returns: InsensitiveDict
        """

        if renderer is None:
            renderer = self._build_renderer()
        if isinstance(renderer, dict):
            renderer = InsensitiveDict.from_dict(renderer)
        elif isinstance(renderer, PropertyMap):
            renderer = InsensitiveDict.from_dict(dict(renderer))
        elif isinstance(renderer, InsensitiveDict):
            pass
        else:
            raise ValueError("renderer must be a dictionary type.")
        self._renderer = renderer

    # ----------------------------------------------------------------------
    def set_geometry(self, col, sr=None, inplace=True):
        """Assigns the geometry column by name or by list

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        col                    Required string, Pandas Series, GeoArray, list or tuple. If a string, this
                               is the name of the column containing the geometry. If a Pandas Series
                               GeoArray, list or tuple, it is an iterable of Geometry objects.
        ------------------     --------------------------------------------------------------------
        sr                     Optional integer or spatial reference of the geometries described in
                               the first parameter. If the geometry objects already have the spatial
                               reference defined, this is not necessary. If the spatial reference for
                               the geometry objects is NOT define, it will default to WGS84 (wkid 4326).
        ------------------     --------------------------------------------------------------------
        inplace                Optional bool. Whether or not to modify the dataframe in place, or return
                               a new dataframe. If True, nothing is returned and the dataframe is modified
                               in place. If False, a new dataframe is returned with the geometry set.
                               Defaults to True.
        ==================     ====================================================================

        :return:
            Spatially Enabled DataFrame or None
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.spatial,
                "set_geometry",
                **{"col": col, "sr": sr, "inplace": inplace},
            )
        )

    # ----------------------------------------------------------------------
    @property
    def sr(self):
        """returns the spatial references of the dataframe"""
        b = self._data.index.partitions[0]
        res = self._data.loc[b.to_series().compute().nlargest(5).tolist(), [self.name]][
            self.name
        ].geom.spatial_reference.unique()
        return res

    # ----------------------------------------------------------------------
    @staticmethod
    def from_featureclass(location: str, npartitions: int, **kwargs):
        """
        Returns a Spatially enabled `pandas.DataFrame` from a feature class.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string or pathlib.Path. Full path to the feature class
        ---------------------------     --------------------------------------------------------------------
        npartitions                     Required int. The number of partitions of the index to create. Note
                                        that depending on the size and index of the dataframe, the output
                                        may have fewer partitions than requested.
        ===========================     ====================================================================

        *Optional parameters when ArcPy library is available in the current environment*:

        ===========================     ====================================================================
        **Optional Argument**           **Description**
        ---------------------------     --------------------------------------------------------------------
        sql_clause                      sql clause to parse data down. To learn more see
                                        `ArcPy Search Cursor <https://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm>`_
        ---------------------------     --------------------------------------------------------------------
        where_clause                    where statement. To learn more see `ArcPy SQL reference <https://pro.arcgis.com/en/pro-app/help/mapping/navigation/sql-reference-for-elements-used-in-query-expressions.htm>`_
        ---------------------------     --------------------------------------------------------------------
        fields                          list of strings specifying the field names.
        ---------------------------     --------------------------------------------------------------------
        spatial_filter                  A `Geometry` object that will filter the results.  This requires
                                        `arcpy` to work.
        ===========================     ====================================================================

        :returns: pandas.core.frame.DataFrame
        """
        return from_pandas(
            data=from_featureclass(filename=location, **kwargs),
            npartitions=npartitions,
        )

    # ----------------------------------------------------------------------
    def to_featureclass(
        self,
        location,
        overwrite=True,
        has_z=None,
        has_m=None,
        sanitize_columns=True,
    ):
        """
        Exports a dask dataframe to a feature class.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string. The output of the table.
        ---------------------------     --------------------------------------------------------------------
        overwrite                       Optional Boolean.  If True and if the feature class exists, it will be
                                        deleted and overwritten.  This is default.  If False, the feature class
                                        and the feature class exists, and exception will be raised.
        ---------------------------     --------------------------------------------------------------------
        has_z                           Optional Boolean.  If True, the dataset will be forced to have Z
                                        based geometries.  If a geometry is missing a Z value when true, a
                                        RuntimeError will be raised.  When False, the API will not use the
                                        Z value.
        ---------------------------     --------------------------------------------------------------------
        has_m                           Optional Boolean.  If True, the dataset will be forced to have M
                                        based geometries.  If a geometry is missing a M value when true, a
                                        RuntimeError will be raised. When False, the API will not use the
                                        M value.
        ---------------------------     --------------------------------------------------------------------
        sanitize_columns                Optional Boolean. If True, column names will be converted to string,
                                        invalid characters removed and other checks will be performed. The
                                        default is True.
        ===========================     ====================================================================

        :returns: String

        """
        if location and not str(os.path.dirname(location)).lower() in [
            "memory",
            "in_memory",
        ]:
            location = os.path.abspath(path=location)
        df = self._data.compute().convert_dtypes()
        df.spatial.name
        return to_featureclass(
            geo=df.spatial,
            location=location,
            overwrite=overwrite,
            has_z=has_z,
            sanitize_columns=sanitize_columns,
            has_m=has_m,
        )

    # ----------------------------------------------------------------------
    def to_parquet(
        self,
        folder: str,
        index: bool = None,
        compression: str = "gzip",
        **kwargs,
    ):
        """
        Exports Each Dask DataFrame partition to a parquet file

        :returns: string (folder path)

        """
        if os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        results = []
        for i in range(self._data.npartitions):
            fp = os.path.join(f"{folder}", "part.%i.parquet" % (i))
            res = (
                self._data.get_partition(i)
                .compute()
                .spatial.to_parquet(fp, index=index, compression=compression, **kwargs)
            )
            results.append(res)
        return results

    # ----------------------------------------------------------------------
    @staticmethod
    def from_parquet(folder: str, ext: str = None):
        dfs = []
        from ._io._arrow import _read_parquet

        if ext is None:
            ext = "parquet"
        for fn in glob.glob(pathname=f"{folder}/*.{ext}"):
            dfs.append(_read_parquet(fn))
        df = pd.concat(dfs, ignore_index=True)
        df.reset_index(drop=True)
        return dd.from_pandas(df, 10)

    # ----------------------------------------------------------------------
    def _build_renderer(self):
        """sets the default symbology"""
        gt = self.geometry_type[0]
        base_renderer = {
            "labelingInfo": None,
            "label": "",
            "description": "",
            "type": "simple",
            "symbol": None,
        }
        if gt.lower() in ["point", "multipoint"]:
            base_renderer["symbol"] = {
                "color": [0, 128, 0, 128],
                "size": 18,
                "angle": 0,
                "xoffset": 0,
                "yoffset": 0,
                "type": "esriSMS",
                "style": "esriSMSCircle",
                "outline": {
                    "color": [0, 128, 0, 255],
                    "width": 1,
                    "type": "esriSLS",
                    "style": "esriSLSSolid",
                },
            }

        elif gt.lower() == "polyline":
            base_renderer["symbol"] = {
                "type": "esriSLS",
                "style": "esriSLSSolid",
                "color": [0, 128, 0, 128],
                "width": 1,
            }
        elif gt.lower() == "polygon":
            base_renderer["symbol"] = {
                "type": "esriSFS",
                "style": "esriSFSSolid",
                "color": [0, 128, 0, 128],
                "outline": {
                    "type": "esriSLS",
                    "style": "esriSLSSolid",
                    "color": [110, 110, 110, 255],
                    "width": 1,
                },
            }
        return InsensitiveDict(base_renderer)


###########################################################################
@register_series_accessor("geom")
class GeoDaskSeriesAccessor:
    """
    The Geospatial series accessor used to perform column specific operations.
    """

    _accessor_name = None
    _data = None
    _index = None
    _name = None

    # ----------------------------------------------------------------------
    def __init__(self, series: Series, *args, **kwargs):
        """initializer"""
        self._accessor_name = "geom"
        if series.dtype != "geometry":
            raise ValueError("'geometry' only works on Geometry dtypes.")
        self._data = series
        self._fn_attr = lambda a, op: getattr(a, op)
        self._fn_method = lambda a, op, **kwargs: getattr(a, op)(**kwargs)

    ##---------------------------------------------------------------------
    ##   Accessor Properties
    ##---------------------------------------------------------------------
    @property
    def area(self):
        """
        Returns the features area

        :returns: float in a series
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "area"),
            meta=pd.Series(dtype=float),
        )

    # ----------------------------------------------------------------------
    @property
    def as_arcpy(self):
        """
        Returns the features as ArcPy Geometry

        :returns: arcpy.Geometry in a series
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "as_arcpy"),
            meta=pd.Series(dtype=object),
        )

    # ----------------------------------------------------------------------
    @property
    def as_shapely(self):
        """
        Returns the features as Shapely Geometry

        :returns: shapely.Geometry in a series
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "as_shapely")
        )

    # ----------------------------------------------------------------------
    @property
    def centroid(self):
        """
        Returns the feature's centroid

        :returns: tuple (x,y) in series
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "centroid")
        )

    # ----------------------------------------------------------------------
    @property
    def extent(self):
        """
        Returns the feature's extent

        :returns: tuple (xmin,ymin,xmax,ymax) in series
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "extent")
        )

    # ----------------------------------------------------------------------
    @property
    def first_point(self):
        """
        Returns the feature's first point

        :returns: Geometry
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "first_point")
        )

    # ----------------------------------------------------------------------
    @property
    def geoextent(self):
        """
        A returns the geometry's extents

        :returns: Series of Floats
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "geoextent")
        )

    # ----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """
        returns the geometry types

        :returns: Series of strings
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "geometry_type"),
            meta=pd.Series(dtype=str),
        )

    # ----------------------------------------------------------------------
    @property
    def hull_rectangle(self):
        """
        A space-delimited string of the coordinate pairs of the convex hull

        :returns: Series of strings
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "hull_rectangle")
        )

    # ----------------------------------------------------------------------
    @property
    def has_z(self):
        """
        Determines if the geometry has a Z value

        :returns: Series of Boolean
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "has_z"),
            meta=pd.Series(dtype=bool),
        )

    # ----------------------------------------------------------------------
    @property
    def has_m(self):
        """
        Determines if the geometry has a M value

        :returns: Series of Boolean
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "has_m"),
            meta=pd.Series(dtype=bool),
        )

    # ----------------------------------------------------------------------
    @property
    def is_empty(self):
        """
        Returns True/False if feature is empty

        :returns: Series of Booleans
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "is_empty"),
            meta=pd.Series(dtype=bool),
        )

    # ----------------------------------------------------------------------
    @property
    def is_multipart(self):
        """
        Returns True/False if features has multiple parts

        :returns: Series of Booleans
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "is_multipart"),
            meta=pd.Series(dtype=bool),
        )

    # ----------------------------------------------------------------------
    @property
    def is_valid(self):
        """
        Returns True/False if features geometry is valid

        :returns: Series of Booleans
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "is_valid"),
            meta=pd.Series(dtype=bool),
        )

    # ----------------------------------------------------------------------
    @property
    def JSON(self):
        """
        Returns JSON string  of Geometry

        :returns: Series of strings
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "JSON"),
            meta=pd.Series(dtype=str),
        )

    # ----------------------------------------------------------------------
    @property
    def label_point(self):
        """
        Returns the geometry point for the optimal label location

        :returns: Series of Geometries
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "label_point")
        )

    # ----------------------------------------------------------------------
    @property
    def last_point(self):
        """
        Returns the Geometry of the last point in a feature.

        :returns: Series of Geometry
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "last_point")
        )

    # ----------------------------------------------------------------------
    @property
    def length(self):
        """
        Returns the length of the features

        :returns: Series of float
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "length"),
            meta=pd.Series(dtype=float),
        )

    # ----------------------------------------------------------------------
    @property
    def length3D(self):
        """
        Returns the length of the features

        :returns: Series of float
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "length3D"),
            meta=pd.Series(dtype=float),
        )

    # ----------------------------------------------------------------------
    @property
    def part_count(self):
        """
        Returns the number of parts in a feature's geometry

        :returns: Series of Integer
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "part_count"),
            meta=pd.Series(dtype=int),
        )

    # ----------------------------------------------------------------------
    @property
    def point_count(self):
        """
        Returns the number of points in a feature's geometry

        :returns: Series of Integer
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "point_count"),
            meta=pd.Series(dtype=int),
        )

    # ----------------------------------------------------------------------
    @property
    def spatial_reference(self):
        """
        Returns the Spatial Reference of the Geometry

        :returns: Series of SpatialReference
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "spatial_reference")
        )

    # ----------------------------------------------------------------------
    @property
    def true_centroid(self):
        """
        Returns the true centroid of the Geometry

        :returns: Series of Points
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "true_centroid")
        )

    # ----------------------------------------------------------------------
    @property
    def WKB(self):
        """
        Returns the Geometry as WKB

        :returns: Series of Bytes
        """
        return self._data.map_partitions(lambda part: self._fn_attr(part.geom, "WKB"))

    # ----------------------------------------------------------------------
    @property
    def WKT(self):
        """
        Returns the Geometry as WKT

        :returns: Series of String
        """
        return self._data.map_partitions(
            lambda part: self._fn_attr(part.geom, "WKT"),
            meta=pd.Series(dtype=str),
        )

    ##---------------------------------------------------------------------
    ##  Accessor Geometry Method
    ##---------------------------------------------------------------------
    def vectorize(self, normalize=False):
        """
        Converts the Geometry to `numpy array` where the following holds true:

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        normalize           Optional Boolean.  If **True**, the values are normalized using a Min/Max Scalar.
        ===============     ====================================================================

        **Output index position of vectorized geometry**

        ==================     ====================================================================
        **Index Position**           **Description**
        ------------------     --------------------------------------------------------------------
        0                      Float. X Coordinate of the Geometry
        ------------------     --------------------------------------------------------------------
        1                      Float. Y Coordinate of the Geometry
        ------------------     --------------------------------------------------------------------
        2                      Integer. The values can be 0 or 1. Where 1 indicates the start of the a geometry.
        ------------------     --------------------------------------------------------------------
        3                      Integer. The values can be 0 or 1. Where 1 indicates Non endpoint or start point of geometry.
        ------------------     --------------------------------------------------------------------
        4                      Integer. The values can be 0 or 1. Where 1 indicates the endpoint of the a geometry.
        ------------------     --------------------------------------------------------------------
        5                      Integer. The values can be 0 or 1. Where 1 indicates the bounding geometery.
        ==================     ====================================================================

        **Geometry Output Example**

        >>> polygon.vectorize(normalize=False)
        np.array([[-2, 2, 1, 0, 0, 1],
                  [2, 2, 0, 1, 0, 1],
                  [1, 5, 0, 0, 1, 1]])


        This output describes a Polygon Geometry in the shape of a triangle with `normalize=False`

        :returns: Series of np.ndarray objects

        """
        pass

    def angle_distance_to(self, second_geometry: "Geometry", method: str = "GEODESIC"):
        """
        Returns a tuple of angle and distance to another point using a
        measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required Geometry.  A arcgis.Geometry object.
        ---------------     --------------------------------------------------------------------
        method              Optional String. PLANAR measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC, and
                            LOXODROME measurement types may be chosen as
                            an alternative, if desired.
        ===============     ====================================================================

        :returns: a tuple of angle and distance to another point using a measurement type.
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "angle_distance_to",
                **{"second_geometry": second_geometry, "method": method},
            )
        )

    # ----------------------------------------------------------------------
    def boundary(self):
        """
        Constructs the boundary of the geometry.

        :returns: arcgis.geometry.Polyline
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(part.geom, "boundary")
        )

    # ----------------------------------------------------------------------
    def buffer(self, distance: float):
        """
        Constructs a polygon at a specified distance from the geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        distance            Required float. The buffer distance. The buffer distance is in the
                            same units as the geometry that is being buffered.
                            A negative distance can only be specified against a polygon geometry.
        ===============     ====================================================================

        :returns: arcgis.geometry.Polygon
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(part.geom, "buffer", **{"distance": distance})
        )

    # ----------------------------------------------------------------------
    def clip(self, envelope: tuple):
        """
        Constructs the intersection of the geometry and the specified extent.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        envelope            required tuple. The tuple must have (XMin, YMin, XMax, YMax) each value
                            represents the lower left bound and upper right bound of the extent.
        ===============     ====================================================================

        :returns: output geometry clipped to extent

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(part.geom, "clip", **{"envelope": envelope})
        )

    # ----------------------------------------------------------------------
    def contains(self, second_geometry: "Geometry", relation: str = None):
        """
        Indicates if the base geometry contains the comparison geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional string. The spatial relationship type.

                            + BOUNDARY - Relationship has no restrictions for interiors or boundaries.
                            + CLEMENTINI - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            + PROPER - Boundaries of geometries must not intersect.
        ===============     ====================================================================

        :returns: boolean
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "contains",
                **{"second_geometry": second_geometry, "relation": relation},
            )
        )

    # ----------------------------------------------------------------------
    def convex_hull(self):
        """
        Constructs the geometry that is the minimal bounding polygon such
        that all outer angles are convex.
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(part.geom, "convex_hull")
        )

    # ----------------------------------------------------------------------
    def crosses(self, second_geometry: "Geometry"):
        """
        Indicates if the two geometries intersect in a geometry of a lesser
        shape type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "crosses", **{"second_geometry": second_geometry}
            )
        )

    # ----------------------------------------------------------------------
    def cut(self, cutter: "Polyline"):
        """
        Splits this geometry into a part left of the cutting polyline, and
        a part right of it.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        cutter              Required Polyline. The cuttin polyline geometry
        ===============     ====================================================================

        :returns: a list of two geometries

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(part.geom, "cut", **{"cutter": cutter})
        )

    # ----------------------------------------------------------------------
    def densify(self, method: str, distance: float, deviation: float):
        """
        Creates a new geometry with added vertices

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. The type of densification, DISTANCE, ANGLE, or GEODESIC
        ---------------     --------------------------------------------------------------------
        distance            Required float. The maximum distance between vertices. The actual
                            distance between vertices will usually be less than the maximum
                            distance as new vertices will be evenly distributed along the
                            original segment. If using a type of DISTANCE or ANGLE, the
                            distance is measured in the units of the geometry's spatial
                            reference. If using a type of GEODESIC, the distance is measured
                            in meters.
        ---------------     --------------------------------------------------------------------
        deviation           Required float. Densify uses straight lines to approximate curves.
                            You use deviation to control the accuracy of this approximation.
                            The deviation is the maximum distance between the new segment and
                            the original curve. The smaller its value, the more segments will
                            be required to approximate the curve.
        ===============     ====================================================================

        :returns: arcgis.geometry.Geometry

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "densify",
                **{
                    "method": method,
                    "distance": distance,
                    "deviation": deviation,
                },
            )
        )

    # ----------------------------------------------------------------------
    def difference(self, second_geometry: "Geometry"):
        """
        Constructs the geometry that is composed only of the region unique
        to the base geometry but not part of the other geometry. The
        following illustration shows the results when the red polygon is the
        source geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: arcgis.geometry.Geometry

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "difference",
                **{"second_geometry": second_geometry},
            )
        )

    # ----------------------------------------------------------------------
    def disjoint(self, second_geometry: "Geometry"):
        """
        Indicates if the base and comparison geometries share no points in
        common.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "disjoin", **{"second_geometry": second_geometry}
            )
        )

    # ----------------------------------------------------------------------
    def distance_to(self, second_geometry: "Geometry"):
        """
        Returns the minimum distance between two geometries. If the
        geometries intersect, the minimum distance is 0.
        Both geometries must have the same projection.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: float

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "distance_to",
                **{"second_geometry": second_geometry},
            )
        )

    # ----------------------------------------------------------------------
    def equals(self, second_geometry: "Geometry"):
        """
        Indicates if the base and comparison geometries are of the same
        shape type and define the same set of points in the plane. This is
        a 2D comparison only; M and Z values are ignored.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean


        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "equals", **{"second_geometry": second_geometry}
            )
        )

    # ----------------------------------------------------------------------
    def generalize(self, max_offset: float):
        """
        Creates a new simplified geometry using a specified maximum offset
        tolerance.  This only works on Polylines and Polygons.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        max_offset          Required float. The maximum offset tolerance.
        ===============     ====================================================================

        :returns: arcgis.geometry.Geometry

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "generalize", **{"max_offset": max_offset}
            )
        )

    # ----------------------------------------------------------------------
    def get_area(self, method: str, units: str = None):
        """
        Returns the area of the feature using a measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. PLANAR measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). GEODESIC,
                            GREAT_ELLIPTIC, LOXODROME, and PRESERVE_SHAPE measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Optional String. Areal unit of measure keywords: ACRES | ARES | HECTARES
                            | SQUARECENTIMETERS | SQUAREDECIMETERS | SQUAREINCHES | SQUAREFEET
                            | SQUAREKILOMETERS | SQUAREMETERS | SQUAREMILES |
                            SQUAREMILLIMETERS | SQUAREYARDS
        ===============     ====================================================================

        :returns: float

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "get_area", **{"method": method, "units": units}
            )
        )

    # ----------------------------------------------------------------------
    def get_length(self, method: str, units: str):
        """
        Returns the length of the feature using a measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. PLANAR measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). GEODESIC,
                            GREAT_ELLIPTIC, LOXODROME, and PRESERVE_SHAPE measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Required String. Linear unit of measure keywords: CENTIMETERS |
                            DECIMETERS | FEET | INCHES | KILOMETERS | METERS | MILES |
                            MILLIMETERS | NAUTICALMILES | YARDS
        ===============     ====================================================================

        :returns: float

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "get_length", **{"method": method, "units": units}
            )
        )

    # ----------------------------------------------------------------------
    def get_part(self, index: int = None):
        """
        Returns an array of point objects for a particular part of geometry
        or an array containing a number of arrays, one for each part.

        **requires arcpy**

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        index               Required Integer. The index position of the geometry.
        ===============     ====================================================================

        :return: arcpy.Array

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(part.geom, "get_part", **{"index": index})
        )

    # ----------------------------------------------------------------------
    def intersect(self, second_geometry: "Geometry", dimension: int = 1):
        """
        Constructs a geometry that is the geometric intersection of the two
        input geometries. Different dimension values can be used to create
        different shape types. The intersection of two geometries of the
        same shape type is a geometry containing only the regions of overlap
        between the original geometries.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        dimension           Required Integer. The topological dimension (shape type) of the
                            resulting geometry.

                            + 1  -A zero-dimensional geometry (point or multipoint).
                            + 2  -A one-dimensional geometry (polyline).
                            + 4  -A two-dimensional geometry (polygon).

        ===============     ====================================================================

        :returns: boolean

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "intersect",
                **{
                    "second_geometry": second_geometry,
                    "dimension": dimension,
                },
            )
        )

    # ----------------------------------------------------------------------
    def measure_on_line(self, second_geometry: "Geometry", as_percentage: bool = False):
        """
        Returns a measure from the start point of this line to the in_point.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional Boolean. If False, the measure will be returned as a
                            distance; if True, the measure will be returned as a percentage.
        ===============     ====================================================================

        :return: float

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "measure_on_line",
                **{
                    "second_geometry": second_geometry,
                    "as_percentage": as_percentage,
                },
            )
        )

    # ----------------------------------------------------------------------
    def overlaps(self, second_geometry: "Geometry"):
        """
        Indicates if the intersection of the two geometries has the same
        shape type as one of the input geometries and is not equivalent to
        either of the input geometries.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "overlaps", **{"second_geometry": second_geometry}
            )
        )

    # ----------------------------------------------------------------------
    def point_from_angle_and_distance(
        self, angle: float, distance: float, method: str = "GEODESCIC"
    ):
        """
        Returns a point at a given angle and distance in degrees and meters
        using the specified measurement type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        angle               Required Float. The angle in degrees to the returned point.
        ---------------     --------------------------------------------------------------------
        distance            Required Float. The distance in meters to the returned point.
        ---------------     --------------------------------------------------------------------
        method              Optional String. PLANAR measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC,
                            LOXODROME, and PRESERVE_SHAPE measurement types may be chosen as
                            an alternative, if desired.
        ===============     ====================================================================

        :return: arcgis.geometry.Geometry


        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "point_from_angle_and_distance",
                **{"angle": angle, "distance": distance, "method": method},
            )
        )

    # ----------------------------------------------------------------------
    def position_along_line(self, value: float, use_percentage: bool = False):
        """
        Returns a point on a line at a specified distance from the beginning
        of the line.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required Float. The distance along the line.
        ---------------     --------------------------------------------------------------------
        use_percentage      Optional Boolean. The distance may be specified as a fixed unit
                            of measure or a ratio of the length of the line. If True, value
                            is used as a percentage; if False, value is used as a distance.
                            For percentages, the value should be expressed as a double from
                            0.0 (0%) to 1.0 (100%).
        ===============     ====================================================================

        :return: Geometry

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "position_along_line",
                **{"value": value, "use_percentage": use_percentage},
            )
        )

    # ----------------------------------------------------------------------
    def project_as(
        self,
        spatial_reference: "SpatialReference",
        transformation_name: str = None,
    ):
        """
        Projects a geometry and optionally applies a geotransformation.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required SpatialReference. The new spatial reference. This can be a
                                 SpatialReference object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Required String. The geotransformation name.
        ====================     ====================================================================

        :returns: arcgis.geometry.Geometry
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "project_as",
                **{
                    "spatial_reference": spatial_reference,
                    "transformation_name": transformation_name,
                },
            )
        )

    # ----------------------------------------------------------------------
    def query_point_and_distance(
        self, second_geometry: "Geometry", use_percentage: bool = False
    ):
        """
        Finds the point on the polyline nearest to the in_point and the
        distance between those points. Also returns information about the
        side of the line the in_point is on as well as the distance along
        the line where the nearest point occurs.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional boolean - if False, the measure will be returned as
                            distance, True, measure will be a percentage
        ===============     ====================================================================

        :return: tuple

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "query_point_and_distance",
                **{
                    "second_geometry": second_geometry,
                    "use_percentage": use_percentage,
                },
            )
        )

    # ----------------------------------------------------------------------
    def segment_along_line(
        self,
        start_measure: float,
        end_measure: float,
        use_percentage: bool = False,
    ):
        """
        Returns a Polyline between start and end measures. Similar to
        Polyline.positionAlongLine but will return a polyline segment between
        two points on the polyline instead of a single point.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        start_measure       Required Float. The starting distance from the beginning of the line.
        ---------------     --------------------------------------------------------------------
        end_measure         Required Float. The ending distance from the beginning of the line.
        ---------------     --------------------------------------------------------------------
        use_percentage      Optional Boolean. The start and end measures may be specified as
                            fixed units or as a ratio.
                            If True, start_measure and end_measure are used as a percentage; if
                            False, start_measure and end_measure are used as a distance. For
                            percentages, the measures should be expressed as a double from 0.0
                            (0 percent) to 1.0 (100 percent).
        ===============     ====================================================================

        :returns: Geometry

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "segment_along_line",
                **{
                    "start_measure": start_measure,
                    "end_measure": end_measure,
                    "use_percentage": use_percentage,
                },
            )
        )

    # ----------------------------------------------------------------------
    def snap_to_line(self, second_geometry: "Geometry"):
        """
        Returns a new point based on in_point snapped to this geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "snap_to_line",
                **{"second_geometry": second_geometry},
            )
        )

    # ----------------------------------------------------------------------
    def symmetric_difference(self, second_geometry: "Geometry"):
        """
        Constructs the geometry that is the union of two geometries minus the
        instersection of those geometries.

        The two input geometries must be the same shape type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "symmetric_difference",
                **{"second_geometry": second_geometry},
            )
        )

    # ----------------------------------------------------------------------
    def touches(self, second_geometry: "Geometry"):
        """
        Indicates if the boundaries of the geometries intersect.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "touches", **{"second_geometry": second_geometry}
            )
        )

    # ----------------------------------------------------------------------
    def union(self, second_geometry: "Geometry"):
        """
        Constructs the geometry that is the set-theoretic union of the input
        geometries.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry
        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom, "union", **{"second_geometry": second_geometry}
            )
        )

    # ----------------------------------------------------------------------
    def within(self, second_geometry: "Geometry", relation: str = None):
        """
        Indicates if the base geometry is within the comparison geometry.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional String. The spatial relationship type.

                            - BOUNDARY  - Relationship has no restrictions for interiors or boundaries.
                            - CLEMENTINI  - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            - PROPER  - Boundaries of geometries must not intersect.

        ===============     ====================================================================

        :return: boolean

        """
        return self._data.map_partitions(
            lambda part: self._fn_method(
                part.geom,
                "within",
                **{"second_geometry": second_geometry, "relation": relation},
            )
        )


# -------------------------------------------------------------------------
@make_array_nonempty.register(GeoType)
def _(dtype):
    a = np.array([_default_geometry(), _default_geometry()], dtype=GeoType)
    return _from_geometry(a)


# -------------------------------------------------------------------------
@make_scalar.register(GeoType.type)
def _(x):
    return _default_geometry()


# -------------------------------------------------------------------------
@make_meta_dispatch.register(Geometry)
def make_meta_arcgis_geometry(x, index=None):
    return x


@normalize_token.register(GeoArray)
def tokenize_geometryarray(x):
    return uuid.uuid4().hex
