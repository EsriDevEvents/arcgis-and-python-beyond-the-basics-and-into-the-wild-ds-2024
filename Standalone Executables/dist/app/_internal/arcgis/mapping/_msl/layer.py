from __future__ import annotations
from datetime import datetime
import json
import os
from string import digits
from functools import lru_cache

from re import search
from typing import Any, Optional, Union

from arcgis._impl.common import _utils
from arcgis._impl.common._filters import (
    StatisticFilter,
    TimeFilter,
    GeometryFilter,
)
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._utils import _date_handler, chunks

from arcgis.features.feature import FeatureSet
from arcgis.geometry import SpatialReference
from arcgis.gis import Item, Layer
from arcgis.mapping import MapImageLayer


###########################################################################
class MapFeatureLayer(Layer):
    """
    The ``MapFeatureLayer`` class represents Map Feature Layers.
    Map Feature Layers can be added to and visualized using maps.

    Map Feature Layers are created by publishing feature data to a :class:`~arcgis.gis.GIS`, and are exposed as a
    broader resource (:class:`~arcgis.gis.Item`) in the ``GIS``.
    `MapFeatureLayer` objects can be obtained through the layers attribute on map image service Items in the ``GIS``.
    """

    _metadatamanager = None
    _renderer = None
    _storage = None
    _dynamic_layer = None
    _attachments = None
    _time_filter = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, container=None, dynamic_layer=None):
        """
        Constructs a map feature layer given a feature layer URL
        :param url: layer url
        :param gis: optional, the GIS that this layer belongs to. Required for secure map feature layers.
        :param container: optional, the MapImageLayer to which this layer belongs
        :param dynamic_layer: optional dictionary. If the layer is given a dynamic layer definition, this will be added to functions.
        """
        if gis is None:
            import arcgis

            gis = arcgis.env.active_gis
        if str(url).lower().endswith("/"):
            url = url[:-1]
        super(MapFeatureLayer, self).__init__(url, gis)

        self._attachments = None
        self._dynamic_layer = dynamic_layer
        self._time_filter = None

    # ----------------------------------------------------------------------
    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": "FeatureLayer", "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": "FeatureLayer", "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    @lru_cache(maxsize=10)
    def attachements(self):
        """
        The ``attachements`` property provides a manager to work with attachments if the ``MapFeatureLayer``
        supports this functionality.
        """
        if (
            "supportsQueryAttachments" in self.properties
            and self.properties["supportsQueryAttachments"]
            and self._attachments is None
        ):
            from arcgis.features.managers import AttachmentManager

            self._attachments = AttachmentManager(self)
        return self._attachments

    # ----------------------------------------------------------------------
    @property
    def time_filter(self):
        """
        Starting at Enterprise 10.7.1+, instead of querying time-enabled
        map service layers or time-enabled feature service layers, a
        time filter can be set using the ``time_filter`` property.
        Time can be filtered as Python `datetime <https://docs.python.org/3/library/datetime.html#datetime-objects>`_,
        objects or strings representing Unix epoch values in milliseconds.
        An extent can be specified by separating the start and stop values
        comma.

        .. code-block:: python

            >>> import datetime as dt

            >>> map_feature_lyr.time_filter = [dt.datetime(2021, 1, 1), dt.datetime(2022, 1, 10)]

        """
        return self._time_filter

    # ----------------------------------------------------------------------
    @time_filter.setter
    def time_filter(self, value: Union[datetime, list[datetime], list[str]]):
        """
        See main ``time_filter`` property docstring
        """
        import datetime as _dt

        v = []
        if isinstance(value, _dt.datetime):
            self._time_filter = f"{int(value.timestamp() * 1000)}"  # means single time
        elif isinstance(value, (tuple, list)):
            for idx, d in enumerate(value):
                if idx > 1:
                    break
                if isinstance(d, _dt.datetime):
                    v.append(f"{int(value.timestamp() * 1000)}")
                elif isinstance(d, str):
                    v.append(d)
                elif d is None:
                    v.append("null")
            self._time_filter = ",".join(v)
        elif isinstance(value, str):
            self._time_filter = value
        elif value is None:
            self._time_filter = None
        else:
            raise Exception("Invalid datetime filter")

    # ----------------------------------------------------------------------
    @property
    def renderer(self):
        """
        Get/Set the Renderer of the Map Feature Layer.

        .. note::
            The ``renderer`` property overrides the default symbology when displaying it on a
            :class:`~arcgis.mapping.WebMap`.

        :return:
            ``InsensitiveDict``: A case-insensitive ``dict`` like object used to update and alter JSON
            A varients of a case-less dictionary that allows for dot and bracket notation.

        """
        from arcgis._impl.common._isd import InsensitiveDict

        if self._renderer is None and "drawingInfo" in self.properties:
            self._renderer = InsensitiveDict(dict(self.properties.drawingInfo.renderer))
        return self._renderer

    # ----------------------------------------------------------------------
    @renderer.setter
    def renderer(self, value):
        """
        Get/Set the Renderer of the Map Feature Layer.  This overrides the default symbology when displaying it on a webmap.

        :return:
            ```InsensitiveDict```: A case-insensitive ``dict`` like object used to update and alter JSON
            A varients of a case-less dictionary that allows for dot and bracket notation.

        """
        from arcgis._impl.common._isd import InsensitiveDict

        if isinstance(value, (dict, PropertyMap)):
            self._renderer = InsensitiveDict(dict(value))
        elif value is None:
            self._renderer = None
        elif not isinstance(value, InsensitiveDict):
            raise ValueError("Invalid renderer type.")
        self._refresh = value

    # ----------------------------------------------------------------------
    @classmethod
    def fromitem(cls, item: Item, layer_id: int = 0):
        """
        The ``fromitem`` method creates a :class:`~arcgis.mapping.MapFeatureLayer` from a GIS :class:`~arcgis.gis.Item`.


        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        item                                     Required :class:`~arcgis.gis.Item` object. The type of item should be
                                                 a :class:`~arcgis.mapping.MapServiceLayer` object.
        ------------------------------------     --------------------------------------------------------------------
        layer_id                                 Optional integer. The id of the layer in the Map Service's Layer.
                                                 The default is 0.
        ====================================     ====================================================================

        :return:
            A :class:`~arcgis.mapping.MapFeatureLayer` object

        .. code-block:: python

            # USAGE EXAMPLE

            >>> from arcgis.mapping import MapImageLayer, MapFeatureLayer
            >>> from arcgis.gis import GIS

            # connect to your GIS and get the web map item
            >>> gis = GIS(url, username, password)

            >>> map_image_item = gis.content.get("2aaddab96684405880d27f5261125061")
            >>> map_feature_layer = MapFeatureLayer.fromitem(item = map_image_item,
                                                             layer_id = 2)
            >>> print(f"{map_feature_layer.properties.name:30}{type(map_feature_layer)}")
            <State Boundaries              <class 'arcgis.mapping._msl.layer.MapFeatureLayer'>>

        """
        from arcgis.mapping import MapImageLayer

        return MapImageLayer.fromitem(item).layers[layer_id]

    # ----------------------------------------------------------------------
    @property
    def container(self):
        """
        The ``container`` property represents the :class:`~arcgis.mapping.MapImageLayer` to which this layer belongs.
        """
        if self._storage is None:
            self._storage = MapImageLayer(
                url=self._url.rstrip(digits)[:-1], gis=self._gis
            )
        return self._storage

    # ----------------------------------------------------------------------
    def export_attachments(self, output_folder: str, label_field: Optional[str] = None):
        """
        The ``export_attachments`` method exports attachments from the map feature layer in ``Imagenet`` format using
        the ``output_label_field``.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        output_folder                            Required String. Output folder path where the attachments will be stored.
        ------------------------------------     --------------------------------------------------------------------
        label_field                              Optional. Field which contains the label/category of each feature.
                                                 If None, a default folder is created.
        ====================================     ====================================================================

        :return:
            A path to the exported attachments
        """
        import pandas
        import urllib
        import hashlib

        if not self.properties["hasAttachments"]:
            raise Exception("Map Feature Layer doesn't have any attachments.")

        if not os.path.exists(output_folder):
            raise Exception("Invalid output folder path.")

        object_attachments_mapping = {}

        object_id_field = self.properties["objectIdField"]

        dataframe_merged = pandas.merge(
            self.query().sdf,
            self._attachments.search(as_df=True),
            left_on=object_id_field,
            right_on="PARENTOBJECTID",
        )

        token = self._con.token

        internal_folder = os.path.join(output_folder, "images")
        if not os.path.exists(internal_folder):
            os.mkdir(internal_folder)

        folder = "images"
        for row in dataframe_merged.iterrows():
            if label_field is not None:
                folder = row[1][label_field]

            path = os.path.join(internal_folder, folder)

            if not os.path.exists(path):
                os.mkdir(path)

            if token is not None:
                url = "{}/{}/attachments/{}?token={}".format(
                    self.url,
                    row[1][object_id_field],
                    row[1]["ID"],
                    self._con.token,
                )
            else:
                url = "{}/{}/attachments/{}".format(
                    self.url, row[1][object_id_field], row[1]["ID"]
                )

            if not object_attachments_mapping.get(row[1][object_id_field]):
                object_attachments_mapping[row[1][object_id_field]] = []

            content = urllib.request.urlopen(url).read()

            md5_hash = hashlib.md5(content).hexdigest()
            attachment_path = os.path.join(path, f"{md5_hash}.jpg")

            object_attachments_mapping[row[1][object_id_field]].append(
                os.path.join("images", os.path.join(folder, f"{md5_hash}.jpg"))
            )

            if os.path.exists(attachment_path):
                continue
            file = open(attachment_path, "wb")
            file.write(content)
            file.close()

        mapping_path = os.path.join(output_folder, "mapping.txt")
        file = open(mapping_path, "w")
        file.write(json.dumps(object_attachments_mapping))
        file.close()

    # ----------------------------------------------------------------------
    def generate_renderer(
        self, definition: dict[str, Any], where: Optional[str] = None
    ):
        """
        The ``generate_renderer`` operation groups data using the supplied definition
        (classification definition) and an optional where clause. The
        result is a renderer object. Use ``baseSymbol`` and ``colorRamp`` to define
        the symbols assigned to each class.

        .. note::
            If the operation is performed
            on a table, the result is a renderer object containing the data
            classes and no symbols.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        definition            Required dict. The definition using the renderer that is generated.
                              Use either class breaks or unique value classification definitions.
                              See the
                              `classification definitions <https://resources.arcgis.com/en/help/rest/apiref/ms_classification.html>`_
                              page in the ArcGIS REST API documentation for more information.
        -----------------     --------------------------------------------------------------------
        where                 Optional string. A where clause for which the data needs to be
                              classified. Any legal SQL where clause operating on the fields in
                              the dynamic layer/table is allowed.
        =================     ====================================================================

        :return: dictionary

        """
        if self._dynamic_layer:
            url = "%s/generateRenderer" % self._url.split("?")[0]
        else:
            url = "%s/generateRenderer" % self._url
        params = {"f": "json", "classificationDef": definition}
        if where:
            params["where"] = where
        if self._dynamic_layer is not None:
            params["layer"] = self._dynamic_layer
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def _add_attachment(self, oid, file_path):
        """
        Adds an attachment to a feature service

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        oid                   Required string/integer. OBJECTID value to add attachment to.
        -----------------     --------------------------------------------------------------------
        file_path             Required string. Location of the file to attach.
        =================     ====================================================================

        :return: dictionary

        """
        if (os.path.getsize(file_path) >> 20) <= 9:
            params = {"f": "json"}
            if self._dynamic_layer:
                attach_url = self._url.split("?")[0] + "/%s/addAttachment" % oid
                params["layer"] = self._dynamic_layer
            else:
                attach_url = self._url + "/%s/addAttachment" % oid
            files = {"attachment": file_path}
            res = self._con.post(
                path=attach_url,
                postdata=params,
                files=files,
                token=self._token,
            )
            return res
        else:
            params = {"f": "json"}
            container = self.container
            itemid = container.upload(file_path)
            if self._dynamic_layer:
                attach_url = self._url.split("?")[0] + "/%s/addAttachment" % oid
                params["layer"] = self._dynamic_layer
            else:
                attach_url = self._url + "/%s/addAttachment" % oid
            params["uploadId"] = itemid
            res = self._con.post(attach_url, params)
            if res["addAttachmentResult"]["success"] == True:
                container._delete_upload(itemid)
            return res

    # ----------------------------------------------------------------------
    def _delete_attachment(self, oid, attachment_id):
        """
        Removes an attachment from a feature service feature

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        oid                   Required string/integer. OBJECTID value to add attachment to.
        -----------------     --------------------------------------------------------------------
        attachment_id         Required integer. Id of the attachment to erase.
        =================     ====================================================================

        :return: dictionary
        """
        params = {"f": "json", "attachmentIds": "%s" % attachment_id}
        if self._dynamic_layer:
            url = self._url.split("?")[0] + "/%s/deleteAttachments" % oid
            params["layer"] = self._dynamic_layer
        else:
            url = self._url + "/%s/deleteAttachments" % oid
        return self._con.post(url, params, token=self._token)

    # ----------------------------------------------------------------------
    def _update_attachment(self, oid, attachment_id, file_path):
        """
        Updates an existing attachment with a new file

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        oid                   Required string/integer. OBJECTID value to add attachment to.
        -----------------     --------------------------------------------------------------------
        attachment_id         Required integer. Id of the attachment to erase.
        -----------------     --------------------------------------------------------------------
        file_path             Required string. Path to new attachment
        =================     ====================================================================

        :return: dictionary

        """
        params = {"f": "json", "attachmentId": "%s" % attachment_id}
        files = {"attachment": file_path}
        if self._dynamic_layer is not None:
            url = self.url.split("?")[0] + f"/{oid}/updateAttachment"
            params["layer"] = self._dynamic_layer
        else:
            url = self._url + f"/{oid}/updateAttachment"
        res = self._con.post(path=url, postdata=params, files=files, token=self._token)
        return res

    # ----------------------------------------------------------------------
    def _list_attachments(self, oid):
        """list attachments for a given OBJECT ID"""

        params = {"f": "json"}
        if self._dynamic_layer is not None:
            url = self.url.split("?")[0] + "/%s/attachments" % oid
            params["layer"] = self._dynamic_layer
        else:
            url = self._url + "/%s/attachments" % oid
        return self._con.get(path=url, params=params, token=self._token)

    # ----------------------------------------------------------------------
    def get_unique_values(self, attribute: str, query_string: str = "1=1"):
        """
        The ``get_unique_values`` method retrieves a list of unique values for a given attribute.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        attribute                           Required string. The map feature layer attribute to query.
        -------------------------------     --------------------------------------------------------------------
        query_string                        Optional string. SQL Query that will be used to filter attributes
                                            before unique values are returned.
        ===============================     ====================================================================

        :return:
            A List

        .. code-block:: python

            # USAGE EXAMPLE

            >>> from arcgis.mapping import MapImageLayer, MapFeatureLayer
            >>> from arcgis.gis import GIS

            # connect to your GIS and get the web map item
            >>> gis = GIS(url, username, password)

            >>> map_image_item = gis.content.get("2aaddab96684405880d27f5261125061")
            >>> map_feature_layer = MapFeatureLayer.fromitem(item = map_image_item,
                                                             layer_id = 2)
            >>> unique_values = map_feature_layer.get_unique_values(attribute ="Name",
                                                    query_string ="name_2 like '%K%'")
            >>> type(unique_values)
            <List>
        """

        result = self.query(
            query_string,
            return_geometry=False,
            out_fields=attribute,
            return_distinct_values=True,
        )
        return [feature.attributes[attribute] for feature in result.features]

    # ----------------------------------------------------------------------
    def query(
        self,
        where: str = "1=1",
        text: Optional[str] = None,  # new
        out_fields: Union[str, list[str]] = "*",
        time_filter: Optional[
            Union[list[int], list[datetime], dict[str, datetime]]
        ] = None,
        geometry_filter: Optional[GeometryFilter] = None,
        return_geometry: bool = True,
        return_count_only: bool = False,
        return_ids_only: bool = False,
        return_distinct_values: bool = False,
        return_extent_only: bool = False,
        group_by_fields_for_statistics: Optional[str] = None,
        statistic_filter: Optional[StatisticFilter] = None,
        result_offset: Optional[int] = None,
        result_record_count: Optional[int] = None,
        object_ids: Optional[str] = None,
        distance: Optional[int] = None,
        units: Optional[str] = None,
        max_allowable_offset: Optional[float] = None,
        out_sr: Optional[int] = None,
        geometry_precision: Optional[int] = None,
        gdb_version: Optional[str] = None,
        order_by_fields: Optional[str] = None,
        out_statistics: Optional[list[dict[str, Any]]] = None,
        return_z: bool = False,
        return_m: bool = False,
        multipatch_option=None,
        quantization_parameters: Optional[dict[str, Any]] = None,
        return_centroid: bool = False,
        return_all_records: bool = True,
        result_type: Optional[str] = None,
        historic_moment: Optional[Union[int, datetime]] = None,
        sql_format: Optional[str] = None,
        return_true_curves: bool = False,
        return_exceeded_limit_features: Optional[bool] = None,
        as_df: bool = False,
        datum_transformation: Optional[Union[int, dict[str, Any]]] = None,
        range_values: Optional[dict[str, Any]] = None,
        parameter_values: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        The ``query`` method queries a map feature layer based on a sql statement.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        where                               Optional string. The default is 1=1. The selection sql statement.
        -------------------------------     --------------------------------------------------------------------
        text                                Optional String. A literal search text. If the layer has a display
                                            field associated with it, the server searches for this text in this
                                            field.
        -------------------------------     --------------------------------------------------------------------
        out_fields                          Optional List of field names to return. Field names can be specified
                                            either as a List of field names or as a comma separated string.
                                            The default is "*", which returns all the fields.
        -------------------------------     --------------------------------------------------------------------
        object_ids                          Optional string. The object IDs of this layer or table to be queried.
                                            The object ID values should be a comma-separated string.
        -------------------------------     --------------------------------------------------------------------
        distance                            Optional integer. The buffer distance for the input geometries.
                                            The distance unit is specified by units. For example, if the
                                            distance is 100, the query geometry is a point, units is set to
                                            meters, and all points within 100 meters of the point are returned.
        -------------------------------     --------------------------------------------------------------------
        units                               Optional string. The unit for calculating the buffer distance. If
                                            unit is not specified, the unit is derived from the geometry spatial
                                            reference. If the geometry spatial reference is not specified, the
                                            unit is derived from the feature service data spatial reference.
                                            This parameter only applies if `supportsQueryWithDistance` is
                                            `true`.

                                            Value options:
                                                    ``esriSRUnit_Meter`` | ``esriSRUnit_StatuteMile`` |
                                                    ``esriSRUnit_Foot`` | ``esriSRUnit_Kilometer`` |
                                                    ``esriSRUnit_NauticalMile`` | ``esriSRUnit_USNauticalMile``
        -------------------------------     --------------------------------------------------------------------
        time_filter                         Optional list of `startTime` and `endTime` values.
                                            :Syntax:

                                            .. code-block:: python

                                                >>> time_filter=[<startTime>, <endTime>]

                                            .. note::
                                                Specified as ``datetime.date``, ``datetime.datetime`` or
                                                ``timestamp`` in milliseconds
        -------------------------------     --------------------------------------------------------------------
        geometry_filter                     Optional :class:`filter <arcgis.geometry.filters>` object. Allows for
                                            the information to be filtered on spatial relationship with another
                                            geometry.
        -------------------------------     --------------------------------------------------------------------
        max_allowable_offset                Optional float. This option can be used to specify the
                                            `max_allowable_offset` to be used for generalizing geometries
                                            returned by the query operation in the units of `out_sr`. If
                                            `out_sr`  is not specified, the value is in units of the spatial
                                            reference of the layer.
        -------------------------------     --------------------------------------------------------------------
        out_sr                              Optional Integer. The WKID for the spatial reference of the returned
                                            geometry.
        -------------------------------     --------------------------------------------------------------------
        geometry_precision                  Optional Integer. This option can be used to specify the number of
                                            decimal places in the response geometries returned by the query
                                            operation.
                                            This applies to X and Y values only (not m or z-values).
        -------------------------------     --------------------------------------------------------------------
        gdb_version                         Optional string. The geodatabase version to query. This parameter
                                            applies only if the `isDataVersioned` property of the layer is true.
                                            If not specified, the query will apply to the published map's
                                            version.
        -------------------------------     --------------------------------------------------------------------
        return_geometry                     Optional boolean. If `true`, geometry is returned with the query.
                                            Default is `true`.
        -------------------------------     --------------------------------------------------------------------
        return_distinct_values              Optional boolean.  If `True`, it returns distinct values based on
                                            fields specified in `out_fields`. This parameter applies only if the
                                            `supportsAdvancedQueries` property of the layer is true.
        -------------------------------     --------------------------------------------------------------------
        return_ids_only                     Optional boolean. Default is `False`.  If `True`, the response only
                                            includes an array of object IDs. Otherwise, the response is a
                                            :class:`~arcgis.features.FeatureSet`.
        -------------------------------     --------------------------------------------------------------------
        return_count_only                   Optional boolean. If `True`, the response only includes the count
                                            of features/records satisfying the query. Otherwise, the response is
                                            a :class:`~arcgis.features.FeatureSet`. The default is `False`. This
                                            option supersedes the `returns_ids_only` parameter. If
                                            ``returnCountOnly = True`` , the response will return both the count
                                            and the extent.
        -------------------------------     --------------------------------------------------------------------
        return_extent_only                  Optional boolean. If `True`, the response only includes the extent
                                            of the features satisying the query. If `returnCountOnly=true`, the
                                            response will return both the count and the extent. The default is
                                            `False`. This parameter applies only if the
                                            `supportsReturningQueryExtent` property of the layer is `true`.
        -------------------------------     --------------------------------------------------------------------
        order_by_fields                     Optional string. One or more field names by which to order the
                                            results. Use ``ASC`` or ``DESC`` for ascending
                                            or descending, respectively, following every field to be ordered:

                                            .. code-block:: python

                                                >>> order_by_fields = "STATE_NAME ASC, RACE DESC, GENDER ASC"

        -------------------------------     --------------------------------------------------------------------
        group_by_fields_for_statistics      Optional string. One or more field names on which to group results
                                            for calculating the statistics.

                                            .. code-block:: python

                                                >>> group_by_fields_for_statiscits = "STATE_NAME, GENDER"
        -------------------------------     --------------------------------------------------------------------
        out_statistics                      Optional List. The definitions for one or more field-based
                                            statistics to be calculated.

                                            :Syntax:

                                            .. code-block:: python

                                                >>> out_statistics = [
                                                                        {
                                                                          "statisticType": "<count | sum | min | max | avg | stddev | var>",
                                                                          "onStatisticField": "Field1",
                                                                          "outStatisticFieldName": "Out_Field_Name1"
                                                                        },
                                                                        {
                                                                          "statisticType": "<count | sum | min | max | avg | stddev | var>",
                                                                          "onStatisticField": "Field2",
                                                                          "outStatisticFieldName": "Out_Field_Name2"
                                                                        }
                                                                     ]
        -------------------------------     --------------------------------------------------------------------
        return_z                            Optional boolean. If `True`, Z values are included in the results if
                                            the features have Z values. Otherwise, Z values are not returned.
                                            The default is `False`.
        -------------------------------     --------------------------------------------------------------------
        return_m                            Optional boolean. If `True`, M values are included in the results if
                                            the features have M values. Otherwise, M values are not returned.
                                            The default is `False`.
        -------------------------------     --------------------------------------------------------------------
        multipatch_option                   Optional x/y footprint. This option dictates how the geometry of
                                            a multipatch feature will be returned.
        -------------------------------     --------------------------------------------------------------------
        result_offset                       Optional integer. This option can be used for fetching query results
                                            by skipping the specified number of records and starting from the
                                            next record (that is, `resultOffset + ith` value). This option is
                                            ignored if `return_all_records` is `True` (i.e. by default).
        -------------------------------     --------------------------------------------------------------------
        result_record_count                 Optional integer. This option can be used for fetching query results
                                            up to the `result_record_count` specified. When `result_offset` is
                                            specified but this parameter is not, the map service defaults it to
                                            `max_record_count`. The maximum value for this parameter is the value
                                            of the layer's `maxRecordCount` property. This option is ignored if
                                            `return_all_records` is True (i.e. by default).
        -------------------------------     --------------------------------------------------------------------
        quantization_parameters             Optional dict. Used to project the geometry onto a virtual grid,
                                            likely representing pixels on the screen.
        -------------------------------     --------------------------------------------------------------------
        return_centroid                     Optional boolean. Used to return the geometry centroid associated
                                            with each feature returned. If `True`, the result includes the
                                            geometry centroid. The default is `False`.
        -------------------------------     --------------------------------------------------------------------
        return_all_records                  Optional boolean. When `True`, the query operation will call the
                                            service until all records that satisfy the `where_clause` are
                                            returned.

                                            .. note::
                                                `result_offset` and `result_record_count` will be
                                                ignored if set to `True`. If `return_count_only`, `return_ids_only`,
                                                or `return_extent_only` are `True`, this parameter is ignored.
        -------------------------------     --------------------------------------------------------------------
        result_type                         Optional string. Controls the number of features returned by the
                                            operation.
                                            Options: ``None`` | ``standard`` | ``tile``

                                            .. note::
                                                See `Query (Feature Service/Layer) <https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer-.htm>`_
                                                for full explanation.
        -------------------------------     --------------------------------------------------------------------
        historic_moment                     Optional integer. The historic moment to query. This parameter
                                            applies only if the layer is archiving enabled and the
                                            `supportsQueryWithHistoricMoment` property is set to `true`. This
                                            property is provided in the layer's
                                            :attr:`~arcgis.features.FeatureLayer.properties` resource. If
                                            not specified, the query will apply to the current features.
        -------------------------------     --------------------------------------------------------------------
        sql_format                          Optional string.  The `sql_format` parameter can be either standard
                                            SQL92 or it can use the native SQL of the underlying
                                            datastore. The default is `None`, which means it depends on the
                                            `useStandardizedQuery` layer property.
                                            Values: ``None`` | ``standard`` | ``native``
        -------------------------------     --------------------------------------------------------------------
        return_true_curves                  Optional boolean. When set to `True`, returns true curves in output
                                            geometries. When set to `False`, curves are converted to densified
                                            polylines or polygons.
        -------------------------------     --------------------------------------------------------------------
        return_exceeded_limit_features      Optional boolean. Optional parameter which is true by default. When
                                            set to true, features are returned even when the results include
                                            the `exceededTransferLimit: True` property.

                                            When set to `False` and querying with `resultType = tile`, features
                                            are not returned when the results include
                                            `exceededTransferLimit: True`. This allows a client to find the
                                            resolution in which the transfer limit is no longer exceeded without
                                            making multiple calls.
        -------------------------------     --------------------------------------------------------------------
        as_df                               Optional boolean.  If `True`, the results are returned as a
                                            `DataFrame` instead of a :class:`~arcgis.features.FeatureSet`.
        -------------------------------     --------------------------------------------------------------------
        datum_transformation                Optional Integer/Dictionary.  This parameter applies a datum transformation while
                                            projecting geometries in the results when out_sr is different than the layer's spatial
                                            reference. When specifying transformations, you need to think about which datum
                                            transformation best projects the layer (not the feature service) to the `outSR` and
                                            `sourceSpatialReference` property in the layer properties. For a list of valid datum
                                            transformation ID values ad well-known text strings, see `Coordinate systems and
                                            transformations <https://developers.arcgis.com/net/latest/wpf/guide/coordinate-systems-and-transformations.htm>`_.
                                            For more information on datum transformations, please see the transformation
                                            parameter in the `Project operation <https://developers.arcgis.com/rest/services-reference/project.htm>`_.

                                            Example:


                                            ===========     ===================================
                                            Inputs          Description
                                            -----------     -----------------------------------
                                            WKID            Integer.

                                                            .. code-block:: python

                                                                >>> datum_transformation=4326

                                            -----------     -----------------------------------
                                            WKT             Dict.

                                                            .. code-block:: python

                                                                >>> datum_transformation = {"wkt": "<WKT>"}

                                            -----------     -----------------------------------
                                            Composite       Dict.

                                                            .. code-block:: python

                                                                >>> datum_transformation = {"geoTransforms" : [
                                                                                                               {"wkid" : "<id>",
                                                                                                                "forward" : True | False},
                                                                                                               {"wkt" : "WKT",
                                                                                                                "forward" : True: False}
                                                                                                              ]
                                                                                           }

                                            ===========     ===================================
        -------------------------------     --------------------------------------------------------------------
        range_values                        Optional List. Allows you to filter features from the layer that are
                                            within the specified range instant or extent.

                                            .. code-block:: python

                                                >>> range_values = [
                                                                    {
                                                                     "name": "range name" ,
                                                                     # single value or a value-range
                                                                     "value": <value> or [ <value1>, <value2> ]

                                                                    },
                                                                    {
                                                                     "name": "range name 2",
                                                                     "value": <value> or  [ <value3>, <value4> ]
                                                                    }
                                                                   ]


                                            .. note::

                                                `None` is allowed in value-range case to indicate infinity

                                                .. code-block:: python

                                                    # all features with values <= 1500
                                                    >>> range_values = [
                                                                        {"name" : "range name",
                                                                         "value" : [None, 1500]}
                                                                       ]

                                                    # all features with values >= 1000
                                                    >>> range_values = [
                                                                        {"name" : "range name",
                                                                         "value" : [1000, None]}
                                                                       ]

        -------------------------------     --------------------------------------------------------------------
        parameter_values                    Optional Dict. Allows you to filter the layers by specifying
                                            value(s) to an array of pre-authored parameterized filters for those
                                            layers. When value is not specified for any parameter in a request,
                                            the default value, that is assigned during authoring time, gets used
                                            instead.

                                            When a `parameterInfo` allows multiple values, you must pass them in
                                            an array.

                                            .. note::
                                                Check `parameterValues` at the `Query (Map Service/Layer) <https://developers.arcgis.com/rest/services-reference/enterprise/query-map-service-layer-.htm#GUID-403AC0F3-4B48-45BD-B473-E52E790FD296>`_
                                                for details on parameterized filters.
        -------------------------------     --------------------------------------------------------------------
        kwargs                              Optional dict. Optional parameters that can be passed to the Query
                                            function.  This will allow users to pass additional parameters not
                                            explicitly implemented on the function. A complete list of functions
                                            available is documented  at `Query (Feature Service/Layer) <https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer-.htm>`_.
        ===============================     ====================================================================

        :return: A :class:`~arcgis.features.FeatureSet` containing the features matching the query unless another
        return type is specified, such as ``count``.

        .. code-block:: python

            # USAGE EXAMPLE

            >>> from arcgis.mapping import MapImageLayer, MapFeatureLayer
            >>> from arcgis.gis import GIS

            # connect to your GIS and get the web map item
            >>> gis = GIS(url, username, password)

            >>> map_image_item = gis.content.get("2aaddab96684405880d27f5261125061")
            >>> map_feature_layer = MapFeatureLayer.fromitem(item = map_image_item,
                                                             layer_id = 2)
            >>> query_count = map_feature_layer.query(where "1=1",
                                        text = "Hurricane Data",
                                        units = "esriSRUnit_Meter",
                                        return_count_only = True,
                                        out_statistics = [
                                                            {
                                                            "statisticType": "count",
                                                            "onStatisticField": "Field1",
                                                            "outStatisticFieldName": "Out_Field_Name1"
                                                            },
                                                            {
                                                            "statisticType": "avg",
                                                            "onStatisticField": "Field2",
                                                            "outStatisticFieldName": "Out_Field_Name2"
                                                            }
                                                        ],
                                        range_values= [
                                                {
                                                  "name": "range name",
                                                  "value": [None, 1500]
                                                  },
                                                  {
                                                    "name": "range name 2",
                                                    "value":[1000, None]
                                                  }
                                                }
                                            ]
                                        )
            >>> query_count
            <149>
        """
        as_raw = as_df
        if self._dynamic_layer is None:
            url = self._url + "/query"
        else:
            url = "%s/query" % self._url.split("?")[0]

        params = {"f": "json"}
        if self._dynamic_layer is not None:
            params["layer"] = self._dynamic_layer
        if result_type is not None:
            params["resultType"] = result_type
        if historic_moment is not None:
            params["historicMoment"] = historic_moment
        if sql_format is not None:
            params["sqlFormat"] = sql_format
        if return_true_curves is not None:
            params["returnTrueCurves"] = return_true_curves
        if return_exceeded_limit_features is not None:
            params["returnExceededLimitFeatures"] = return_exceeded_limit_features
        params["where"] = where
        params["returnGeometry"] = return_geometry
        params["returnDistinctValues"] = return_distinct_values
        params["returnCentroid"] = return_centroid
        params["returnCountOnly"] = return_count_only
        params["returnExtentOnly"] = return_extent_only
        params["returnIdsOnly"] = return_ids_only
        params["returnZ"] = return_z
        params["returnM"] = return_m
        if parameter_values:
            params["parameterValues"]
        if range_values:
            params["rangeValues"] = range_values
        if not datum_transformation is None:
            params["datumTransformation"] = datum_transformation

        # convert out_fields to a comma separated string
        if isinstance(out_fields, (list, tuple)):
            out_fields = ",".join(out_fields)

        if out_fields != "*" and not return_distinct_values:
            try:
                # Check if object id field is in out_fields.
                # If it isn't, add it
                object_id_field = [
                    x.name
                    for x in self.properties.fields
                    if x.type == "esriFieldTypeOID"
                ][0]
                if object_id_field not in out_fields.split(","):
                    out_fields = object_id_field + "," + out_fields
            except (IndexError, AttributeError):
                pass
        params["outFields"] = out_fields
        if return_count_only or return_extent_only or return_ids_only:
            return_all_records = False
        if result_record_count and not return_all_records:
            params["resultRecordCount"] = result_record_count
        if result_offset and not return_all_records:
            params["resultOffset"] = result_offset
        if quantization_parameters:
            params["quantizationParameters"] = quantization_parameters
        if multipatch_option:
            params["multipatchOption"] = multipatch_option
        if order_by_fields:
            params["orderByFields"] = order_by_fields
        if group_by_fields_for_statistics:
            params["groupByFieldsForStatistics"] = group_by_fields_for_statistics
        if statistic_filter and isinstance(statistic_filter, StatisticFilter):
            params["outStatistics"] = statistic_filter.filter
        if out_statistics:
            params["outStatistics"] = out_statistics
        if text:
            params["text"] = text
        if out_sr:
            params["outSR"] = out_sr
        if max_allowable_offset:
            params["maxAllowableOffset"] = max_allowable_offset
        if gdb_version:
            params["gdbVersion"] = gdb_version
        if geometry_precision:
            params["geometryPrecision"] = geometry_precision
        if object_ids:
            params["objectIds"] = object_ids
        if distance:
            params["distance"] = distance
        if units:
            params["units"] = units

        if time_filter is None and self.time_filter:
            params["time"] = self.time_filter
        elif time_filter is not None:
            if type(time_filter) is list:
                starttime = _date_handler(time_filter[0])
                endtime = _date_handler(time_filter[1])
                if starttime is None:
                    starttime = "null"
                if endtime is None:
                    endtime = "null"
                params["time"] = "%s,%s" % (starttime, endtime)
            elif isinstance(time_filter, dict):
                for key, val in time_filter.items():
                    params[key] = val
            else:
                params["time"] = _date_handler(time_filter)

        if geometry_filter and isinstance(geometry_filter, GeometryFilter):
            for key, val in geometry_filter.filter:
                params[key] = val
        elif geometry_filter and isinstance(geometry_filter, dict):
            for key, val in geometry_filter.items():
                params[key] = val
        if len(kwargs) > 0:
            for key, val in kwargs.items():
                if (
                    key
                    in (
                        "returnCountOnly",
                        "returnExtentOnly",
                        "returnIdsOnly",
                    )
                    and val
                ):
                    # If these keys are passed in as kwargs instead of parameters, set return_all_records
                    return_all_records = False
                params[key] = val
                del key, val

        if not return_all_records or "outStatistics" in params:
            if as_df:
                return self._query_df(url, params)
            return self._query(url, params, raw=as_raw)

        params["returnCountOnly"] = True
        record_count = self._query(url, params, raw=as_raw)
        if "maxRecordCount" in self.properties:
            max_records = self.properties["maxRecordCount"]
        else:
            max_records = 1000

        supports_pagination = True
        if (
            "advancedQueryCapabilities" not in self.properties
            or "supportsPagination" not in self.properties["advancedQueryCapabilities"]
            or not self.properties["advancedQueryCapabilities"]["supportsPagination"]
        ):
            supports_pagination = False

        params["returnCountOnly"] = False
        if record_count == 0 and as_df:
            from arcgis.features.geo._array import GeoArray
            import numpy as np
            import pandas as pd

            _fld_lu = {
                "esriFieldTypeSmallInteger": np.int32,
                "esriFieldTypeInteger": np.int64,
                "esriFieldTypeSingle": float,
                "esriFieldTypeDouble": float,
                "esriFieldTypeFloat": float,
                "esriFieldTypeString": str,
                "esriFieldTypeDate": np.datetime64,
                "esriFieldTypeOID": np.int64,
                "esriFieldTypeGeometry": object,
                "esriFieldTypeBlob": object,
                "esriFieldTypeRaster": object,
                "esriFieldTypeGUID": str,
                "esriFieldTypeGlobalID": str,
                "esriFieldTypeXML": object,
            }
            columns = {}
            for fld in self.properties.fields:
                fld = dict(fld)
                columns[fld["name"]] = _fld_lu[fld["type"]]
            if (
                "geometryType" in self.properties
                and not self.properties.geometryType is None
            ):
                columns["SHAPE"] = object
            df = pd.DataFrame([], columns=columns.keys()).astype(columns, True)
            if "SHAPE" in df.columns:
                df["SHAPE"] = GeoArray([])
                df.spatial.set_geometry("SHAPE")
                df.spatial.renderer = self.renderer
                df.spatial._meta.source = self
            return df
        elif record_count <= max_records:
            if supports_pagination and record_count > 0:
                params["resultRecordCount"] = record_count
            if as_df:
                import pandas as pd

                df = self._query_df(url, params)
                dt_fields = [
                    fld["name"]
                    for fld in self.properties.fields
                    if fld["type"] == "esriFieldTypeDate"
                ]
                if "SHAPE" in df.columns:
                    df.spatial.set_geometry("SHAPE")
                    df.spatial.renderer = self.renderer
                    df.spatial._meta.source = self
                for fld in dt_fields:
                    try:
                        if fld in df.columns:
                            df[fld] = pd.to_datetime(
                                df[fld] / 1000,
                                unit="s",
                            )
                    except:
                        if fld in df.columns:
                            df[fld] = pd.to_datetime(
                                df[fld],
                            )
                return df

            return self._query(url, params, raw=as_raw)

        result = None
        i = 0
        count = 0
        df = None
        dfs = []
        if not supports_pagination:
            params["returnIdsOnly"] = True
            oid_info = self._query(url, params, raw=as_raw)
            params["returnIdsOnly"] = False
            for ids in chunks(oid_info["objectIds"], max_records):
                ids = [str(i) for i in ids]
                sql = "%s in (%s)" % (
                    oid_info["objectIdFieldName"],
                    ",".join(ids),
                )
                params["where"] = sql
                if not as_df:
                    records = self._query(url, params, raw=as_raw)
                    if result:
                        if "features" in result:
                            result["features"].append(records["features"])
                        else:
                            result.features.extend(records.features)
                    else:
                        result = records
                else:
                    df = self._query_df(url, params)
                    dfs.append(df)
        else:
            while True:
                params["resultRecordCount"] = max_records
                params["resultOffset"] = max_records * i
                if not as_df:
                    records = self._query(url, params, raw=as_raw)

                    if result:
                        if "features" in result:
                            result["features"].append(records["features"])
                        else:
                            result.features.extend(records.features)
                    else:
                        result = records

                    if len(records.features) < max_records:
                        break
                else:
                    df = self._query_df(url, params)
                    count += len(df)
                    dfs.append(df)
                    if count == record_count:
                        break
                i += 1
        if as_df:
            import pandas as pd

            dt_fields = [
                fld["name"]
                for fld in self.properties.fields
                if fld["type"] == "esriFieldTypeDate"
            ]
            if len(dfs) == 1:
                df = dfs[0]
            else:
                df = pd.concat(dfs, sort=True)
                df.reset_index(drop=True, inplace=True)
            if "SHAPE" in df.columns:
                df.spatial.set_geometry("SHAPE")
                df.spatial.renderer = self.renderer
                df.spatial._meta.source = self
            for fld in dt_fields:
                if fld in df.columns:
                    try:
                        df[fld] = pd.to_datetime(
                            df[fld] / 1000,
                            unit="s",
                        )
                    except:
                        df[fld] = pd.to_datetime(
                            df[fld],
                            errors="coerce",
                        )
            return df
        return result

    # ----------------------------------------------------------------------
    def query_related_records(
        self,
        object_ids: str,
        relationship_id: str,
        out_fields: Union[str, list[str]] = "*",
        definition_expression: Optional[str] = None,
        return_geometry: bool = True,
        max_allowable_offset: Optional[float] = None,
        geometry_precision: Optional[int] = None,
        out_wkid: Optional[int] = None,
        gdb_version: Optional[str] = None,
        return_z: bool = False,
        return_m: bool = False,
        historic_moment: Optional[Union[int, datetime]] = None,
        return_true_curve: bool = False,
    ):
        """
        The ``query_related_records`` operation is performed on a :class:`~arcgis.mapping.MapFeatureLayer`
        resource. The result of this operation are :class:`~arcgis.features.FeatureSet` objects grouped
        by source layer/table object IDs. Each :class:`~arcgis.features.FeatureSet` contains
        :class:`~arcgis.features.Feature` objects including the values for the fields requested by
        the user.

        .. note::
            For related layers, if you request geometry
            information, the geometry of each feature is also returned in
            the feature set. For related tables, the feature set does not
            include geometries.

        .. note::
            See the :attr:`~arcgis.mapping.MapFeatureLayer.query` method for more information.


        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        object_ids                 Required string. The object IDs of the table/layer to be queried
        ----------------------     --------------------------------------------------------------------
        relationship_id            Required string. The ID of the relationship to be queried.
        ----------------------     --------------------------------------------------------------------
        out_fields                 Required string. the list of fields from the related table/layer
                                   to be included in the returned feature set. This list is a comma
                                   delimited list of field names. If you specify the shape field in the
                                   list of return fields, it is ignored. To request geometry, set
                                   return_geometry to true. You can also specify the wildcard "*" as
                                   the value of this parameter. In this case, the results will include
                                   all the field values.
        ----------------------     --------------------------------------------------------------------
        definition_expression      Optional string. The definition expression to be applied to the
                                   related table/layer. From the list of objectIds, only those records
                                   that conform to this expression are queried for related records.
        ----------------------     --------------------------------------------------------------------
        return_geometry            Optional boolean. If true, the feature set includes the geometry
                                   associated with each feature. The default is true.
        ----------------------     --------------------------------------------------------------------
        max_allowable_offset       Optional float. This option can be used to specify the
                                   max_allowable_offset to be used for generalizing geometries returned
                                   by the query operation. The max_allowable_offset is in the units of
                                   the outSR. If out_wkid is not specified, then max_allowable_offset
                                   is assumed to be in the unit of the spatial reference of the map.
        ----------------------     --------------------------------------------------------------------
        geometry_precision         Optional integer. This option can be used to specify the number of
                                   decimal places in the response geometries.
        ----------------------     --------------------------------------------------------------------
        out_wkid                   Optional Integer. The spatial reference of the returned geometry.
        ----------------------     --------------------------------------------------------------------
        gdb_version                Optional string. The geodatabase version to query. This parameter
                                   applies only if the isDataVersioned property of the layer queried is
                                   true.
        ----------------------     --------------------------------------------------------------------
        return_z                   Optional boolean. If true, Z values are included in the results if
                                   the features have Z values. Otherwise, Z values are not returned.
                                   The default is false.
        ----------------------     --------------------------------------------------------------------
        return_m                   Optional boolean. If true, M values are included in the results if
                                   the features have M values. Otherwise, M values are not returned.
                                   The default is false.
        ----------------------     --------------------------------------------------------------------
        historic_moment            Optional Integer/datetime. The historic moment to query. This parameter
                                   applies only if the supportsQueryWithHistoricMoment property of the
                                   layers being queried is set to true. This setting is provided in the
                                   layer resource.

                                   If historic_moment is not specified, the query will apply to the
                                   current features.

                                   Syntax:
                                        historic_moment=<Epoch time in milliseconds>
        ----------------------     --------------------------------------------------------------------
        return_true_curves         Optional boolean. Optional parameter that is false by default. When
                                   set to true, returns true curves in output geometries; otherwise,
                                   curves are converted to densified polylines or polygons.
        ======================     ====================================================================


        :return: dict


        """
        params = {
            "f": "json",
            "objectIds": object_ids,
            "relationshipId": relationship_id,
            "outFields": out_fields,
            "returnGeometry": return_geometry,
            "returnM": return_m,
            "returnZ": return_z,
        }
        if historic_moment:
            if hasattr(historic_moment, "timestamp"):
                historic_moment = int(historic_moment.timestamp() * 1000)
            params["historicMoment"] = historic_moment
        if return_true_curve:
            params["returnTrueCurves"] = return_true_curve
        if self._dynamic_layer is not None:
            params["layer"] = self._dynamic_layer
        if gdb_version is not None:
            params["gdbVersion"] = gdb_version
        if definition_expression is not None:
            params["definitionExpression"] = definition_expression
        if out_wkid is not None and isinstance(out_wkid, SpatialReference):
            params["outSR"] = out_wkid
        elif out_wkid is not None and isinstance(out_wkid, dict):
            params["outSR"] = out_wkid
        if max_allowable_offset is not None:
            params["maxAllowableOffset"] = max_allowable_offset
        if geometry_precision is not None:
            params["geometryPrecision"] = geometry_precision
        if self._dynamic_layer is None:
            qrr_url = self._url + "/queryRelatedRecords"
        else:
            qrr_url = "%s/queryRelatedRecords" % self._url.split("?")[0]

        return self._con.post(path=qrr_url, postdata=params, token=self._token)

    # ----------------------------------------------------------------------
    def get_html_popup(self, oid: str):
        """
        The ``get_html_popup`` resource provides details about the HTML pop-up
        authored by the user using ArcGIS Pro or ArcGIS Desktop.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        oid                 Optional string. Object id of the feature to get the HTML popup.
        ===============     ====================================================================


        :return:
            A string

        """
        if self.properties.htmlPopupType != "esriServerHTMLPopupTypeNone":
            pop_url = self._url + "/%s/htmlPopup" % oid
            params = {"f": "json"}

            return self._con.get(path=pop_url, params=params, token=self._token)
        return ""

    # ----------------------------------------------------------------------
    def _status_via_url(self, con, url, params):
        """
        performs the asynchronous check to see if the operation finishes
        """
        status_allowed = [
            "Pending",
            "InProgress",
            "Completed",
            "Failed ImportChanges",
            "ExportChanges",
            "ExportingData",
            "ExportingSnapshot",
            "ExportAttachments",
            "ImportAttachments",
            "ProvisioningReplica",
            "UnRegisteringReplica",
            "CompletedWithErrors",
        ]
        status = con.get(url, params)
        while status["status"] in status_allowed and status["status"] != "Completed":
            if status["status"] == "Completed":
                return status
            elif status["status"] == "CompletedWithErrors":
                break
            elif "fail" in status["status"].lower():
                break
            elif "error" in status["status"].lower():
                break
            status = con.get(url, params)
        return status

    # ----------------------------------------------------------------------
    def _query(self, url, params, raw=False):
        """returns results of query"""
        try:
            result = self._con.post(path=url, postdata=params, token=self._token)
            if "exceededTransferLimit" in result:
                while (
                    "exceededTransferLimit" in result
                    and result["exceededTransferLimit"] == True
                ):
                    params["resultRecordCount"] = params["resultRecordCount"] * 2
                    result = self._con.post(
                        path=url, postdata=params, token=self._token
                    )

        except Exception as queryException:
            error_list = [
                "Error performing query operation",
                "HTTP Error 504: GATEWAY_TIMEOUT",
            ]
            if any(ele in queryException.__str__() for ele in error_list):
                # half the max record count
                max_record = (
                    int(params["resultRecordCount"])
                    if "resultRecordCount" in params
                    else 1000
                )
                offset = int(params["resultOffset"]) if "resultOffset" in params else 0
                # reduce this number to 125 if you still sees 500/504 error
                if max_record < 250:
                    # when max_record is lower than 250, but still getting error 500 or 504, just exit with exception
                    raise queryException
                else:
                    max_rec = int((max_record + 1) / 2)
                    i = 0
                    result = None
                    while max_rec * i < max_record:
                        params["resultRecordCount"] = (
                            max_rec
                            if max_rec * (i + 1) <= max_record
                            else (max_record - max_rec * i)
                        )
                        params["resultOffset"] = offset + max_rec * i
                        try:
                            records = self._query(url, params, raw=True)
                            if result:
                                for feature in records["features"]:
                                    result["features"].append(feature)
                            else:
                                result = records
                            i += 1
                        except Exception as queryException2:
                            raise queryException2

            else:
                raise queryException

        def is_true(x):
            if isinstance(x, bool) and x:
                return True
            elif isinstance(x, str) and x.lower() == "true":
                return True
            else:
                return False

        if "error" in result:
            raise ValueError(result)
        if "returnCountOnly" in params and is_true(params["returnCountOnly"]):
            return result["count"]
        elif "returnIdsOnly" in params and is_true(params["returnIdsOnly"]):
            return result
        elif "extent" in result:
            return result
        elif is_true(raw):
            return result
        else:
            return FeatureSet.from_dict(result)

    # ----------------------------------------------------------------------
    def _query_df(self, url, params):
        """returns results of a query as a pd.DataFrame"""
        import pandas as pd
        from arcgis.features import GeoAccessor, GeoSeriesAccessor
        import numpy as np

        if [float(i) for i in pd.__version__.split(".")] < [1, 0, 0]:
            _fld_lu = {
                "esriFieldTypeSmallInteger": np.int32,
                "esriFieldTypeInteger": np.int64,
                "esriFieldTypeSingle": float,
                "esriFieldTypeDouble": float,
                "esriFieldTypeFloat": float,
                "esriFieldTypeString": str,
                "esriFieldTypeDate": pd.datetime,
                "esriFieldTypeOID": np.int64,
                "esriFieldTypeGeometry": object,
                "esriFieldTypeBlob": object,
                "esriFieldTypeRaster": object,
                "esriFieldTypeGUID": str,
                "esriFieldTypeGlobalID": str,
                "esriFieldTypeXML": object,
            }
        else:
            from datetime import datetime as _datetime

            _fld_lu = {
                "esriFieldTypeSmallInteger": np.int32,
                "esriFieldTypeInteger": np.int64,
                "esriFieldTypeSingle": float,
                "esriFieldTypeDouble": float,
                "esriFieldTypeFloat": float,
                "esriFieldTypeString": str,
                "esriFieldTypeDate": _datetime,
                "esriFieldTypeOID": np.int64,
                "esriFieldTypeGeometry": object,
                "esriFieldTypeBlob": object,
                "esriFieldTypeRaster": object,
                "esriFieldTypeGUID": str,
                "esriFieldTypeGlobalID": str,
                "esriFieldTypeXML": object,
            }

        def feature_to_row(feature, sr):
            """:return: a feature from a dict"""
            from arcgis.geometry import Geometry

            geom = feature["geometry"] if "geometry" in feature else None
            attribs = feature["attributes"] if "attributes" in feature else {}
            if "centroid" in feature:
                if attribs is None:
                    attribs = {"centroid": feature["centroid"]}
                elif "centroid" in attribs:
                    import uuid

                    fld = "centroid_" + uuid.uuid4().hex[:2]
                    attribs[fld] = feature["centroid"]
                else:
                    attribs["centroid"] = feature["centroid"]
            if geom:
                if "spatialReference" not in geom:
                    geom["spatialReference"] = sr
                attribs["SHAPE"] = Geometry(geom)
            return attribs

        # ------------------------------------------------------------------
        try:
            featureset_dict = self._con.post(url, params, token=self._token)
        except Exception as queryException:
            error_list = [
                "Error performing query operation",
                "HTTP Error 504: GATEWAY_TIMEOUT",
            ]
            if any(ele in queryException.__str__() for ele in error_list):
                # half the max record count
                max_record = (
                    int(params["resultRecordCount"])
                    if "resultRecordCount" in params
                    else 1000
                )
                offset = int(params["resultOffset"]) if "resultOffset" in params else 0
                # reduce this number to 125 if you still sees 500/504 error
                if max_record < 250:
                    # when max_record is lower than 250, but still getting error 500 or 504, just exit with exception
                    raise queryException
                else:
                    max_rec = int((max_record + 1) / 2)
                    i = 0
                    featureset_dict = None
                    while max_rec * i < max_record:
                        params["resultRecordCount"] = (
                            max_rec
                            if max_rec * (i + 1) <= max_record
                            else (max_record - max_rec * i)
                        )
                        params["resultOffset"] = offset + max_rec * i
                        try:
                            records = self._query(url, params, raw=True)
                            if featureset_dict is not None:
                                for feature in records["features"]:
                                    featureset_dict["features"].append(feature)
                            else:
                                featureset_dict = records
                            i += 1
                        except Exception as queryException2:
                            raise queryException2

            else:
                raise queryException

        if len(featureset_dict["features"]) == 0:
            return pd.DataFrame([])
        sr = None
        if "spatialReference" in featureset_dict:
            sr = featureset_dict["spatialReference"]

        df = None
        dtypes = None
        geom = None
        names = None
        dfields = []
        rows = [feature_to_row(row, sr) for row in featureset_dict["features"]]
        if len(rows) == 0:
            return None
        df = pd.DataFrame.from_records(data=rows)
        if "fields" in featureset_dict:
            dtypes = {}
            names = []
            fields = featureset_dict["fields"]
            for fld in fields:
                if fld["type"] != "esriFieldTypeGeometry":
                    dtypes[fld["name"]] = _fld_lu[fld["type"]]
                    names.append(fld["name"])
                if fld["type"] == "esriFieldTypeDate":
                    dfields.append(fld["name"])
        if "SHAPE" in featureset_dict:
            df.spatial.set_geometry("SHAPE")
        if len(dfields) > 0:
            df[dfields] = df[dfields].apply(pd.to_datetime, unit="ms")
        return df


###########################################################################
class MapRasterLayer(MapFeatureLayer):
    """
    The ``MapRasterLayer`` class represents a geo-referenced image hosted in a ``Map Service``.
    """

    @property
    def _lyr_dict(self):
        url = self.url

        if "lods" in self.container.properties:
            lyr_dict = {"type": "ArcGISTiledMapServiceLayer", "url": url}

        else:
            lyr_dict = {"type": type(self.container).__name__, "url": url}

        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        if "lods" in self.container.properties:
            lyr_dict = {
                "type": "ArcGISTiledMapServiceLayer",
                "url": self.container.url,
            }

        else:
            lyr_dict = {
                "type": type(self.container).__name__,
                "url": self.container.url,
            }

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict


###########################################################################
class MapTable(MapFeatureLayer):
    """
    The ``MapTable`` class represents entity classes with uniform properties.

    .. note::
        In addition to working with entities with ``location`` as
        features, the :class:`~arcgis.gis.GIS` can also work with non-spatial entities as rows in tables.

    Working with tables is similar to working with a :class:`~arcgis.mapping.MapFeatureLayer`, except that the rows
    (:class:`~arcgis.features.Feature`) in a table do not have a geometry, and tables ignore any geometry related
    operation.
    """

    @classmethod
    def fromitem(cls, item: Item, table_id: int = 0):
        """
        The ``fromitem`` method creates a :class:`~arcgis.mapping.MapTable` from a GIS :class:`~arcgis.gis.Item`.


        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        item                                     Required :class:`~arcgis.gis.Item` object. The type of item should be
                                                 a :class:`~arcgis.mapping.MapImageService` object.
        ------------------------------------     --------------------------------------------------------------------
        layer_id                                 Optional integer. The id of the layer in the Map Service's Layer.
                                                 The default is 0.
        ====================================     ====================================================================

        :return:
            A :class:`~arcgis.mapping.MapTable` object

        .. code-block:: python

            # USAGE EXAMPLE

            >>> from arcgis.mapping import MapImageLayer, MapTable
            >>> from arcgis.gis import GIS

            # connect to your GIS and get the web map item
            >>> gis = GIS(url, username, password)

            >>> map_image_item = gis.content.get("2aaddab96684405880d27f5261125061")
            >>> map_table = MapFeatureLayer.fromitem(item = map_image_item,
                                                             layer_id = 2)
            >>> print(f"{map_table.properties.name:30}{type(map_table)}")
            <State Boundaries              <class 'arcgis.mapping.MapTable'>>
        """
        return item.tables[table_id]

    # ----------------------------------------------------------------------
    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": "FeatureLayer", "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": "FeatureLayer", "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    # ----------------------------------------------------------------------
    def query(
        self,
        where: str = "1=1",
        out_fields: Union[str, list[str]] = "*",
        time_filter: Optional[
            Union[datetime, list[datetime], list[str], dict[datetime]]
        ] = None,
        return_count_only: bool = False,
        return_ids_only: bool = False,
        return_distinct_values: bool = False,
        group_by_fields_for_statistics: Optional[str] = None,
        statistic_filter: Optional[StatisticFilter] = None,
        result_offset: Optional[int] = None,
        result_record_count: Optional[int] = None,
        object_ids: Optional[str] = None,
        gdb_version: Optional[str] = None,
        order_by_fields: Optional[str] = None,
        out_statistics: Optional[str[dict]] = None,
        return_all_records: bool = True,
        historic_moment: Optional[Union[int, datetime]] = None,
        sql_format: Optional[str] = None,
        return_exceeded_limit_features: Optional[bool] = None,
        as_df: bool = False,
        range_values: Optional[list[dict[str, Any]]] = None,
        parameter_values: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        The ``query`` method queries a Table Layer based on a set of criteria from a sql statement.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        where                               Optional string. The default is 1=1. The selection sql statement.
        -------------------------------     --------------------------------------------------------------------
        out_fields                          Optional List of field names to return. Field names can be specified
                                            either as a List of field names or as a comma separated string.
                                            The default is "*", which returns all the fields.
        -------------------------------     --------------------------------------------------------------------
        object_ids                          Optional string. The object IDs of this layer or table to be queried.
                                            The object ID values should be a comma-separated string.
        -------------------------------     --------------------------------------------------------------------
        time_filter                         Optional list. The format is of [<startTime>, <endTime>] using
                                            datetime.date, datetime.datetime or timestamp in milliseconds.

                                            .. code-block:: python

                                                >>> time_filter=[<startTime>, <endTime>]

                                            Specified as ``datetime.date``, ``datetime.datetime`` or
                                            ``timestamp`` in milliseconds.

                                            .. code-block:: python

                                                >>> import datetime as dt

                                                >>> time_filter = [dt.datetime(2022, 1, 1), dt.dateime(2022, 1, 12)]

        -------------------------------     --------------------------------------------------------------------
        gdb_version                         Optional string. The geodatabase version to query. This parameter
                                            applies only if the `isDataVersioned` property of the layer is
                                            `true`. If this is not specified, the query will apply to the
                                            published map's version.
        -------------------------------     --------------------------------------------------------------------
        return_geometry                     Optional boolean. If `True`, geometry is returned with the query.
                                            Default is `True`.
        -------------------------------     --------------------------------------------------------------------
        return_distinct_values              Optional boolean.  If `True`, it returns distinct values based on
                                            the fields specified in `out_fields`. This parameter applies only if
                                            the `supportsAdvancedQueries` property of the layer is `true`.
        -------------------------------     --------------------------------------------------------------------
        return_ids_only                     Optional boolean. Default is False.  If `True`, the response only
                                            includes an array of object IDs. Otherwise, the response is a
                                            :class:`~arcgis.features.FeatureSet`.
        -------------------------------     --------------------------------------------------------------------
        return_count_only                   Optional boolean. If `True`, the response only includes the count
                                            (number of features/records) that would be returned by a query.
                                            Otherwise, the response is a :class:`~arcgis.features.FeatureSet`.
                                            The default is `False`. This option supersedes the
                                            `return_ids_only` parameter. If `return_count_only = True`, the
                                            response will return both the count and the extent.
        -------------------------------     --------------------------------------------------------------------
         order_by_fields                    Optional string. One or more field names by which to order the
                                            results. Use ``ASC`` or ``DESC`` for ascending
                                            or descending, respectively, following every field to be ordered:

                                            .. code-block:: python

                                                >>> order_by_fields = "STATE_NAME ASC, RACE DESC, GENDER ASC"

        -------------------------------     --------------------------------------------------------------------
        group_by_fields_for_statistics      Optional string. One or more field names on which to group results
                                            for calculating the statistics.

                                            .. code-block:: python

                                                >>> group_by_fields_for_statiscits = "STATE_NAME, GENDER"

        -------------------------------     --------------------------------------------------------------------
        out_statistics                      Optional string. The definitions for one or more field-based
                                            statistics to be calculated.

                                            :Syntax:

                                            .. code-block:: python

                                                >>> out_statistics = [
                                                                        {
                                                                          "statisticType": "<count | sum | min | max | avg | stddev | var>",
                                                                          "onStatisticField": "Field1",
                                                                          "outStatisticFieldName": "Out_Field_Name1"
                                                                        },{
                                                                           "statisticType": "<count | sum | min | max | avg | stddev | var>",
                                                                           "onStatisticField": "Field2",
                                                                           "outStatisticFieldName": "Out_Field_Name2"
                                                                          }
                                                                    ]
        -------------------------------     --------------------------------------------------------------------
        result_offset                       Optional integer. This option can be used for fetching query results
                                            by skipping the specified number of records and starting from the
                                            next record (that is, `result_offset + ith`). This option is ignored
                                            if `return_all_records` is `True` (i.e. by default).
        -------------------------------     --------------------------------------------------------------------
        result_record_count                 Optional integer. This option can be used for fetching query results
                                            up to the `result_record_count` specified. When `result_offset` is
                                            specified but this parameter is not, the map service defaults it to
                                            `max_record_count`. The maximum value for this parameter is the value
                                            of the layer's `maxRecordCount` property. This option is ignored if
                                            `return_all_records` is `True` (i.e. by default).
        -------------------------------     --------------------------------------------------------------------
        return_all_records                  Optional boolean. When `True`, the query operation will call the
                                            service until all records that satisfy the `where_clause` are
                                            returned. Note: `result_offset` and `result_record_count` will be
                                            ignored if `return_all_records` is True. Also, if
                                            `return_count_only`, `return_ids_only`, or `return_extent_only` are
                                            `True`, this parameter will be ignored.
        -------------------------------     --------------------------------------------------------------------
        historic_moment                     Optional integer. The historic moment to query. This parameter
                                            applies only if the layer is archiving enabled and the
                                            `supportsQueryWithHistoricMoment` property is set to `true`. This
                                            property is provided in the layer resource.

                                            .. note::
                                                See `Query (Feature Service/Layer) <https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer-.htm>`_
                                                for full explanation of layer properties. Use :attr:`~arcgis.features.FeatureLayer.properties`
                                                to examine layer properties.

                                            If `historic_moment` is not specified, the query will apply to the
                                            current features.
        -------------------------------     --------------------------------------------------------------------
        sql_format                          Optional string.  The `sql_format` parameter can be either standard
                                            SQL92 or it can use the native SQL of the underlying
                                            datastore. The default is none which means the sql_format
                                            depends on the `useStandardizedQuery` parameter.
                                            Values: ``none`` | ``standard`` | ``native``
        -------------------------------     --------------------------------------------------------------------
        return_exceeded_limit_features      Optional boolean. Optional parameter which is `true` by default.
                                            When set to `true`, features are returned even when the results
                                            include the `exceededTransferLimit: true` property.

                                            When set to false and querying with `resultType = 'tile'`, features
                                            are not returned when the results include
                                            `exceededTransferLimit: True`. This allows a client to find the
                                            resolution in which the transfer limit is no longer exceeded without
                                            making multiple calls.
        -------------------------------     --------------------------------------------------------------------
        as_df                               Optional boolean.  If `True`, the results are returned as a
                                            `DataFrame` instead of a :class:`~arcgis.features.FeatureSet`.
        -------------------------------     --------------------------------------------------------------------
        range_values                        Optional List. Allows you to filter features from the layer that are
                                            within the specified range instant or extent.

                                            :Syntax:

                                            .. code-block:: python

                                                >>> range_values =     [
                                                                        {
                                                                          "name": "range name",
                                                                          "value": <value> or [ <value1>, <value2> ]
                                                                          },
                                                                          {
                                                                            "name": "range name 2",
                                                                            "value": <value> or  [ <value3>, <value4>]
                                                                          }
                                                                        }
                                                                       ]

                                            .. note::

                                                None is allowed in value-range case -- that means infinity

                                                .. code-block:: python

                                                    # all features with values <= 1500

                                                    >>> range_values = {"name" : "range name",
                                                                         "value :[None, 1500]}

                                                    # all features with values >= 1000

                                                    >>> range_values = {"name" : "range name",
                                                                        "value" : [1000, None]}

        -------------------------------     --------------------------------------------------------------------
        parameter_values                    Optional Dict. Allows you to filter the features layers by specifying
                                            value(s) to an array of pre-authored parameterized filters for those
                                            layers. When value is not specified for any parameter in a request,
                                            the default value, that is assigned during authoring time, gets used
                                            instead.

                                            When `parameterInfo` allows multiple values, you must pass them in
                                            an array.

                                            Note: Check `parameterInfos` at the layer
                                            :attr:`properties <arcgis.features.FeatureLayer.properties>` for
                                            the available parameterized filters, their default values and
                                            expected data type.
        -------------------------------     --------------------------------------------------------------------
        kwargs                              Optional dict. Optional parameters that can be passed to the Query
                                            function.  This will allow users to pass additional parameters not
                                            explicitly implemented on the function. A complete list of possible
                                            parameters is documented at `Query (Map Service/Layer) <https://developers.arcgis.com/rest/services-reference/enterprise/query-map-service-layer-.htm>`_
        ===============================     ====================================================================

        :return:
            A :class:`~arcgis.features.FeatureSet` or Panda's DataFrame containing the :class:`~arcgis.features.Feature`
            objects matching the query, unless another return type is specified, such as ``count``

        .. code-block:: python

            # USAGE EXAMPLE

            >>> from arcgis.mapping import MapImageLayer, MapFeatureLayer
            >>> from arcgis.gis import GIS

            # connect to your GIS and get the web map item
            >>> gis = GIS(url, username, password)

            >>> map_image_item = gis.content.get("2aaddab96684405880d27f5261125061")
            >>> map_feature_layer = MapFeatureLayer.fromitem(item = map_image_item,
                                                             layer_id = 2)
            >>> query_count = map_feature_layer.query(where "1=1",
                                        text = "Hurricane Data",
                                        units = "esriSRUnit_Meter",
                                        return_count_only = True,
                                        out_statistics = [
                                                            {
                                                            "statisticType": "count",
                                                            "onStatisticField": "Field1",
                                                            "outStatisticFieldName": "Out_Field_Name1"
                                                            },
                                                            {
                                                            "statisticType": "avg",
                                                            "onStatisticField": "Field2",
                                                            "outStatisticFieldName": "Out_Field_Name2"
                                                            }
                                                        ],
                                        range_values= [
                                                {
                                                  "name": "range name",
                                                  "value": [None, 1500]
                                                  },
                                                  {
                                                    "name": "range name 2",
                                                    "value":[1000, None]
                                                  }
                                                }
                                            ]
                                        )
            >>> query_count
            <149>
        """
        as_raw = as_df
        if self._dynamic_layer is None:
            url = self._url + "/query"
        else:
            url = "%s/query" % self._url.split("?")[0]

        params = {"f": "json"}
        if self._dynamic_layer is not None:
            params["layer"] = self._dynamic_layer
        if historic_moment is not None:
            params["historicMoment"] = historic_moment
        if sql_format is not None:
            params["sqlFormat"] = sql_format
        if return_exceeded_limit_features is not None:
            params["returnExceededLimitFeatures"] = return_exceeded_limit_features
        params["where"] = where
        params["returnDistinctValues"] = return_distinct_values
        params["returnCountOnly"] = return_count_only
        params["returnIdsOnly"] = return_ids_only

        # convert out_fields to a comma separated string
        if isinstance(out_fields, (list, tuple)):
            out_fields = ",".join(out_fields)

        if out_fields != "*" and not return_distinct_values:
            try:
                # Check if object id field is in out_fields.
                # If it isn't, add it
                object_id_field = [
                    x.name
                    for x in self.properties.fields
                    if x.type == "esriFieldTypeOID"
                ][0]
                if object_id_field not in out_fields.split(","):
                    out_fields = object_id_field + "," + out_fields
            except (IndexError, AttributeError):
                pass
        params["outFields"] = out_fields
        if return_count_only or return_ids_only:
            return_all_records = False
        if result_record_count and not return_all_records:
            params["resultRecordCount"] = result_record_count
        if result_offset and not return_all_records:
            params["resultOffset"] = result_offset
        if order_by_fields:
            params["orderByFields"] = order_by_fields
        if group_by_fields_for_statistics:
            params["groupByFieldsForStatistics"] = group_by_fields_for_statistics
        if statistic_filter and isinstance(statistic_filter, StatisticFilter):
            params["outStatistics"] = statistic_filter.filter
        if out_statistics:
            params["outStatistics"] = out_statistics
        if gdb_version:
            params["gdbVersion"] = gdb_version
        if object_ids:
            params["objectIds"] = object_ids

        if time_filter is None and self.time_filter:
            params["time"] = self.time_filter
        elif time_filter is not None:
            if type(time_filter) is list:
                starttime = _date_handler(time_filter[0])
                endtime = _date_handler(time_filter[1])
                if starttime is None:
                    starttime = "null"
                if endtime is None:
                    endtime = "null"
                params["time"] = "%s,%s" % (starttime, endtime)
            elif isinstance(time_filter, dict):
                for key, val in time_filter.items():
                    params[key] = val
            else:
                params["time"] = _date_handler(time_filter)

        if len(kwargs) > 0:
            for key, val in kwargs.items():
                if key in ("returnCountOnly", "returnIdsOnly") and val:
                    # If these keys are passed in as kwargs instead of parameters, set return_all_records
                    return_all_records = False
                params[key] = val
                del key, val

        if not return_all_records or "outStatistics" in params:
            if as_df:
                return self._query_df(url, params)
            return self._query(url, params, raw=as_raw)

        params["returnCountOnly"] = True
        record_count = self._query(url, params, raw=as_raw)
        if "maxRecordCount" in self.properties:
            max_records = self.properties["maxRecordCount"]
        else:
            max_records = 1000

        supports_pagination = True
        if (
            "advancedQueryCapabilities" not in self.properties
            or "supportsPagination" not in self.properties["advancedQueryCapabilities"]
            or not self.properties["advancedQueryCapabilities"]["supportsPagination"]
        ):
            supports_pagination = False

        params["returnCountOnly"] = False
        if record_count == 0 and as_df:
            from arcgis.features.geo._array import GeoArray
            import numpy as np
            import pandas as pd

            _fld_lu = {
                "esriFieldTypeSmallInteger": np.int32,
                "esriFieldTypeInteger": np.int64,
                "esriFieldTypeSingle": float,
                "esriFieldTypeDouble": float,
                "esriFieldTypeFloat": float,
                "esriFieldTypeString": str,
                "esriFieldTypeDate": np.datetime64,
                "esriFieldTypeOID": np.int64,
                "esriFieldTypeGeometry": object,
                "esriFieldTypeBlob": object,
                "esriFieldTypeRaster": object,
                "esriFieldTypeGUID": str,
                "esriFieldTypeGlobalID": str,
                "esriFieldTypeXML": object,
            }
            columns = {}
            for fld in self.properties.fields:
                fld = dict(fld)
                columns[fld["name"]] = _fld_lu[fld["type"]]
            if (
                "geometryType" in self.properties
                and not self.properties.geometryType is None
            ):
                columns["SHAPE"] = object
            df = pd.DataFrame([], columns=columns.keys()).astype(columns, True)
            if "SHAPE" in df.columns:
                df["SHAPE"] = GeoArray([])
                df.spatial.set_geometry("SHAPE")
                df.spatial.renderer = self.renderer
                df.spatial._meta.source = self
            return df
        elif record_count <= max_records:
            if supports_pagination and record_count > 0:
                params["resultRecordCount"] = record_count
            if as_df:
                import pandas as pd

                df = self._query_df(url, params)
                dt_fields = [
                    fld["name"]
                    for fld in self.properties.fields
                    if fld["type"] == "esriFieldTypeDate"
                ]
                if "SHAPE" in df.columns:
                    df.spatial.set_geometry("SHAPE")
                    df.spatial.renderer = self.renderer
                    df.spatial._meta.source = self
                for fld in dt_fields:
                    try:
                        if fld in df.columns:
                            df[fld] = pd.to_datetime(
                                df[fld] / 1000,
                                unit="s",
                            )
                    except:
                        if fld in df.columns:
                            df[fld] = pd.to_datetime(df[fld])
                return df

            return self._query(url, params, raw=as_raw)

        result = None
        i = 0
        count = 0
        df = None
        dfs = []
        if not supports_pagination:
            params["returnIdsOnly"] = True
            oid_info = self._query(url, params, raw=as_raw)
            params["returnIdsOnly"] = False
            for ids in chunks(oid_info["objectIds"], max_records):
                ids = [str(i) for i in ids]
                sql = "%s in (%s)" % (
                    oid_info["objectIdFieldName"],
                    ",".join(ids),
                )
                params["where"] = sql
                if not as_df:
                    records = self._query(url, params, raw=as_raw)
                    if result:
                        if "features" in result:
                            result["features"].append(records["features"])
                        else:
                            result.features.extend(records.features)
                    else:
                        result = records
                else:
                    df = self._query_df(url, params)
                    dfs.append(df)
        else:
            while True:
                params["resultRecordCount"] = max_records
                params["resultOffset"] = max_records * i
                if not as_df:
                    records = self._query(url, params, raw=as_raw)

                    if result:
                        if "features" in result:
                            result["features"].append(records["features"])
                        else:
                            result.features.extend(records.features)
                    else:
                        result = records

                    if len(records.features) < max_records:
                        break
                else:
                    df = self._query_df(url, params)
                    count += len(df)
                    dfs.append(df)
                    if count == record_count:
                        break
                i += 1
        if as_df:
            import pandas as pd

            dt_fields = [
                fld["name"]
                for fld in self.properties.fields
                if fld["type"] == "esriFieldTypeDate"
            ]
            if len(dfs) == 1:
                df = dfs[0]
            else:
                df = pd.concat(dfs, sort=True)
                df.reset_index(drop=True, inplace=True)
            if "SHAPE" in df.columns:
                df.spatial.set_geometry("SHAPE")
                df.spatial.renderer = self.renderer
                df.spatial._meta.source = self
            for fld in dt_fields:
                try:
                    df[fld] = pd.to_datetime(df[fld] / 1000, unit="s")
                except:
                    df[fld] = pd.to_datetime(df[fld])
            return df
        return result


###########################################################################
class _MSILayerFactory(type):
    """
    Factory that generates the Map Service Layers

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /MapServer/<index>
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`  object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a Map Service Layer object

        from arcgis.mapping import SceneLayer
        ms_layer = MapServiceLayer(url='https://your_portal.com/arcgis/rest/services/service_name/MapServer/0')

        type(ms_layer)
        >> arcgis.mapping._types.MapTable

        print(s_layer.properties.name)
        >> 'pipe_properties'
    """

    def __call__(cls, url, gis=None, container=None, dynamic_layer=None):
        lyr = Layer(url=url, gis=gis)
        props = lyr.properties
        if "type" in props and props.type.lower() == "table":
            return MapTable(
                url=url,
                gis=gis,
                container=container,
                dynamic_layer=dynamic_layer,
            )
        elif "type" in props and props.type.lower() == "raster layer":
            return MapRasterLayer(
                url=url,
                gis=gis,
                container=container,
                dynamic_layer=dynamic_layer,
            )
        elif "type" in props and props.type.lower() == "feature layer":
            return MapFeatureLayer(
                url=url,
                gis=gis,
                container=container,
                dynamic_layer=dynamic_layer,
            )
        return lyr


###########################################################################
class MapServiceLayer(Layer, metaclass=_MSILayerFactory):
    """
    The ``MapServiceLayer`` class is a factory that generates the Map Service Layers.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /MapServer/<index>
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS` object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a Map Service Layer object

        from arcgis.mapping import MapServiceLayer
        ms_layer = MapServiceLayer(url='https://your_portal.com/arcgis/rest/services/service_name/MapServer/0')

        type(ms_layer)
        >> arcgis.mapping._types.MapTable

        print(ms_layer.properties.name)
        >> 'pipe_properties'

    """

    def __init__(self, url, gis=None, container=None, dynamic_layer=None):
        """
        Constructs a Map Services Layer given a URL and GIS
        """
        super(MapServiceLayer, self).__init__(
            url=url,
            gis=gis,
            container=container,
            dynamic_layer=dynamic_layer,
        )
