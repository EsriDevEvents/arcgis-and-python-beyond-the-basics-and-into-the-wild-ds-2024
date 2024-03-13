"""
Feature Layers and Tables provide the primary interface for working with features in a GIS.

Users create, import, export, analyze, edit, and visualize features, i.e. entities in space as feature layers.

A FeatureLayerCollection is a collection of feature layers and tables, with the associated relationships among the entities.
"""
from __future__ import annotations
from datetime import datetime
import json
import os
from re import S, search
import time
import concurrent.futures
from typing import Any, Optional, Union
from arcgis._impl.common import _utils
from arcgis._impl.common._filters import (
    StatisticFilter,
    GeometryFilter,
    TimeFilter,
    LayerDefinitionFilter,
)
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._utils import _date_handler, chunks
from arcgis.features._async import EditFeatureJob

from .managers import (
    AttachmentManager,
    SyncManager,
    FeatureLayerCollectionManager,
    FeatureLayerManager,
)
from .feature import Feature, FeatureSet
from arcgis.gis import Item, Layer, _GISResource
from arcgis.geometry import Geometry, SpatialReference


class FeatureLayer(Layer):
    """
    The ``FeatureLayer`` class is the primary concept for working with :class:`~arcgis.features.Feature` objects
    in a :class:`~arcgis.gis.GIS`.

    :class:`~arcgis.gis.User` objects create, import, export, analyze, edit, and visualize features,
    i.e. entities in space as feature layers.

    ``Feature layers`` can be added to and visualized using maps. They act as inputs to and outputs from feature
    analysis tools.

    Feature layers are created by publishing feature data to a GIS, and are exposed as a broader resource
    (:class:`~arcgis.gis.Item`) in the
    GIS. Feature layer objects can be obtained through the layers attribute on feature layer Items in the GIS.
    """

    _metadatamanager = None
    _renderer = None

    def __init__(self, url, gis=None, container=None, dynamic_layer=None):
        """
        Constructs a feature layer given a feature layer URL
        :param url: feature layer url
        :param gis: optional, the GIS that this layer belongs to. Required for secure feature layers.
        :param container: optional, the feature layer collection to which this layer belongs
        :param dynamic_layer: optional dictionary. If the layer is given a dynamic layer definition, this will be added to functions.
        """
        if gis is None:
            import arcgis

            gis = arcgis.env.active_gis
        if str(url).lower().endswith("/"):
            url = url[:-1]
        super(FeatureLayer, self).__init__(url, gis)
        self._storage = container
        self._dynamic_layer = dynamic_layer
        self.attachments = AttachmentManager(self)
        self._time_filter = None

    @property
    def field_groups(self) -> dict[str, Any]:
        """
        Returns the defined list of field groups for a given layer.

        :returns: dict[str,Any]
        """
        url: str = f"{self._url}/fieldGroups"
        params: dict[str, Any] = {"f": "json"}
        try:
            return self._con.get(url, params=params)
        except:
            return {}

    @property
    def contingent_values(self) -> dict[str, Any]:
        """
        Returns the define contingent values for the given layer.
        :returns: Dict[str,Any]
        """
        url: str = f"{self._url}/contingentValues"
        params: dict[str, Any] = {"f": "json"}
        try:
            return self._con.get(url, params=params)
        except:
            return {}

    @property
    def time_filter(self):
        """
        The ``time_filter`` method is used to set a time filter instead of querying time-enabled map
        service layers or time-enabled feature service layers, a time filter
        can be specified. Time can be filtered as a single instant or by
        separating the two ends of a time extent with a comma.

        .. note::
            The ``time_filter`` method is supported starting at Enterprise 10.7.1+.

        ================     =================================================
        **Input**            **Description**
        ----------------     -------------------------------------------------
        value                Required Datetime/List Datetime. This is a single
                             or list of start/stop date.
        ================     =================================================

        :return:
            A string of datetime values as milliseconds from epoch

        """
        return self._time_filter

    @time_filter.setter
    def time_filter(self, value: Optional[Union[datetime, list[datetime]]]):
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

    @property
    def renderer(self):
        """
        Get/Set the Renderer of the Feature Layer.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        value                   Required dict.
        ==================      ====================================================================

        .. note::
            When set, this overrides the default symbology when displaying it on a webmap.

        :return:
            ```InsensitiveDict```: A case-insensitive ``dict`` like object used to update and alter JSON
            A varients of a case-less dictionary that allows for dot and bracket notation.

        """
        from arcgis._impl.common._isd import InsensitiveDict

        if self._renderer is None and "drawingInfo" in self.properties:
            self._renderer = InsensitiveDict(dict(self.properties.drawingInfo.renderer))
        return self._renderer

    @renderer.setter
    def renderer(self, value: Optional[dict]):
        """
        See main ``renderer`` property docstring
        """
        from arcgis._impl.common._isd import InsensitiveDict

        if isinstance(value, (dict, PropertyMap)):
            self._renderer = InsensitiveDict(dict(value))
        elif value is None:
            self._renderer = None
        elif not isinstance(value, InsensitiveDict):
            raise ValueError("Invalid renderer type.")
        else:
            self._renderer = value

    @classmethod
    def fromitem(cls, item: Item, layer_id: int = 0):
        """
        The ``fromitem`` method creates a :class:`~arcgis.features.FeatureLayer` from an :class:`~arcgis.gis.Item`
        object.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        item                                Required :class:`~arcgis.gis.Item` object. The type of item should be a
                                            ``Feature Service`` that represents a :class:`~arcgis.features.FeatureLayerCollection`
        -------------------------------     --------------------------------------------------------------------
        layer_id                            Required Integer. the id of the layer in feature layer collection (feature service).
                                            The default for ``layer_id`` is 0.
        ===============================     ====================================================================

        :return:
            A :class:`~arcgis.features.FeatureSet` object


        .. code-block:: python

            # Usage Example

            >>> from arcgis.features import FeatureLayer

            >>> gis = GIS("pro")
            >>> buck = gis.content.search("owner:"+ gis.users.me.username)
            >>> buck_1 =buck[1]
            >>> buck_1.type
            'Feature Service'
            >>> new_layer= FeatureLayer.fromitem(item = buck_1)
            >>> type(new_layer)
            <class 'arcgis.features.layer.FeatureLayer'>

        """
        return FeatureLayerCollection.fromitem(item).layers[layer_id]

    @property
    def manager(self):
        """
        The ``manager`` property is a helper object to manage the :class:`~arcgis.features.FeatureLayer`, such as
        updating its definition.

        :return:
            A :class:`~arcgis.features.managers.FeatureLayerManager`

        .. code-block:: python

            # Usage Example

            >>> manager = FeatureLayer.manager
        """
        url = self._url
        res = search("/rest/", url).span()
        add_text = "admin/"
        part1 = url[: res[1]]
        part2 = url[res[1] :]
        admin_url = "%s%s%s" % (part1, add_text, part2)

        res = FeatureLayerManager(admin_url, self._gis)
        return res

    @property
    def metadata(self):
        """
        Get the Feature Layer's metadata.

        .. note::
            If metadata is disabled on the GIS or the
            layer does not support metadata, ``None`` will be returned.

        :return: String of the metadata, if any

        """
        if "hasMetadata" in self.properties:
            try:
                return self._download_metadata()
            except:
                return None
        return None

    # ----------------------------------------------------------------------
    def _download_metadata(self, save_folder=None):
        """
        Downloads the metadata.xml to local disk

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        save_folder           Optional String. A save location to download the metadata XML file.
        =================     ====================================================================

        :return: String
        """
        import tempfile

        if save_folder is None:
            save_folder = tempfile.gettempdir()
        url = "%s%s" % (self.url, "/metadata")
        params = {"f": "json", "format": "default"}

        return self._con.get(
            url, params, out_folder=save_folder, file_name="metadata.xml"
        )

    # ----------------------------------------------------------------------
    def update_metadata(self, file_path: str):
        """
        The ``update_metadata`` updates a :class:`~arcgis.features.FeatureLayer` metadata from an xml file.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        file_path             Required String.  The path to the .xml file that contains the metadata.
        =================     ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        """
        if "hasMetadata" not in self.properties:
            return None

        if (
            os.path.isfile(file_path) == False
            or os.path.splitext(file_path)[1].lower() != ".xml"
        ):
            raise ValueError("file_path must be a XML file.")

        url = "%s%s" % (self.url, "/metadata/update")
        with open(file_path, "r") as reader:
            text = reader.read()

            params = {
                "f": "json",
                "metadata": text,
                "metadataUploadId": "",
                "metadataItemId": "",
                "metadataUploadFormat": "xml",
            }
            res = self._con.post(url, params)
            if "statusUrl" in res:
                return self._status_metadata(res["statusUrl"])
        return False

    # ----------------------------------------------------------------------
    def _status_metadata(self, url):
        """checks the update status"""
        res = self._con.get(url, {"f": "json"})
        if res["status"].lower() == "completed":
            return True
        while res["status"].lower() != "completed":
            time.sleep(1)
            res = self._con.get(url, {"f": "json"})
            if res["status"].lower() == "completed":
                return True
            elif res["status"].lower() == "failed":
                return False
        return False

    @property
    def container(self):
        """
        Get/Set the :class:`~arcgis.features.FeatureLayerCollection` to which this
        layer belongs.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        value                   Required :class:`~arcgis.features.FeatureLayerCollection`.
        ==================      ====================================================================

        :return:
            The Feature Layer Collection where the layer is stored
        """
        return self._storage

    @container.setter
    def container(self, value: Optional[FeatureLayerCollection]):
        """
        See main ``container`` property docstring
        """
        self._storage = value

    def export_attachments(self, output_folder: str, label_field: Optional[str] = None):
        """
        Exports attachments from the :class:`~arcgis.features.FeatureLayer` in Imagenet
        format using the ``output_label_field``.

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        output_folder                            Required string. Output folder where the attachments will be stored.
                                                 If None, a default folder is created
        ------------------------------------     --------------------------------------------------------------------
        label_field                              Optional string. Field which contains the label/category of each feature.
        ====================================     ====================================================================

        :return:
            Nothing is returned from this method
        """
        import pandas
        import urllib
        import hashlib

        if not self.properties["hasAttachments"]:
            raise Exception("Feature Layer doesn't have any attachments.")

        if not os.path.exists(output_folder):
            raise Exception("Invalid output folder path.")

        object_attachments_mapping = {}

        object_id_field = self.properties["objectIdField"]

        dataframe_merged = pandas.merge(
            self.query().sdf,
            self.attachments.search(as_df=True),
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
    def generate_renderer(self, definition: dict, where: Optional[str] = None):
        """
        Groups data using the supplied definition (classification definition) and an optional where clause. The
        result is a renderer object.

        .. note::
            Use baseSymbol and colorRamp to define
            the symbols assigned to each class. If the operation is performed
            on a table, the result is a renderer object containing the data
            classes and no symbols.

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        definition            Required dict. The definition using the renderer that is generated.
                              Use either class breaks or unique value classification definitions.
                              See `Classification Objects <https://developers.arcgis.com/documentation/common-data-types/classification-objects.htm>`_ for additional details.
        -----------------     --------------------------------------------------------------------
        where                 Optional string. A where clause for which the data needs to be
                              classified. Any legal SQL where clause operating on the fields in
                              the dynamic layer/table is allowed.
        =================     ====================================================================

        :return:
            A JSON Dictionary

        .. code-block:: python

            # Example Usage
            FeatureLayer.generate_renderer(
                definition = {"type":"uniqueValueDef",
                              "uniqueValueFields":["Has_Pool"],
                              "fieldDelimiter": ",",
                              "baseSymbol":{
                                  "type": "esriSFS",
                                  "style": "esriSLSSolid",
                                  "width":2
                                  },
                                "colorRamp":{
                                    "type":"algorithmic",
                                    "fromColor":[115,76,0,255],
                                    "toColor":[255,25,86,255],
                                    "algorithm": "esriHSVAlgorithm"
                                    }
                            },
                where = "POP2000 > 350000"
                )

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

    def _add_attachment(
        self,
        oid,
        file_path,
        keywords=None,
        return_moment=False,
        version=None,
    ):
        """
        Adds an attachment to a feature service

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        oid                   Required string/integer. OBJECTID value to add attachment to.
        -----------------     --------------------------------------------------------------------
        file_path             Required string. Location of the file to attach.
        -----------------     --------------------------------------------------------------------
        keywords              Optional string. Sets a text value that is stored as the keywords
                              value for the attachment. If the attachments have keywords enabled and
                              the layer also includes the attachmentFields property, you can use
                              it to understand properties like keywords field length.
        -----------------     --------------------------------------------------------------------
        return_moment         Optional bool. Specify whether the response will report the time
                              attachments were added. If True, the server will return the time
                              in the response's `editMoment` key. The default is False.
        =================     ====================================================================

        :return: A JSON Dictionary indicating 'success' or 'error'

        """
        if (
            os.path.getsize(file_path) < 10e6
        ):  # (os.path.getsize(file_path) >> 20) <= 9:
            params = {
                "f": "json",
                "gdbVersion": version,
                "returnEditMoment": return_moment,
            }
            if self._gis.version > [7, 3] and keywords:
                params["keywords"] = keywords
            if self._dynamic_layer:
                attach_url = self._url.split("?")[0] + "/%s/addAttachment" % oid
                params["layer"] = self._dynamic_layer
            else:
                attach_url = self._url + "/%s/addAttachment" % oid
            files = {"attachment": file_path}
            res = self._con.post(path=attach_url, postdata=params, files=files)
            return res
        else:
            params = {
                "f": "json",
                "gdbVersion": version,
                "returnEditMoment": return_moment,
            }
            if self._gis.version > [7, 3] and keywords:
                params["keywords"] = keywords
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
    def _delete_attachment(
        self,
        oid,
        attachment_id,
        return_moment=False,
        rollback_on_failure=True,
        version=None,
    ):
        """
        Removes an attachment from a feature service feature

        ===================     ====================================================================
        **Parameter**            **Description**
        -------------------     --------------------------------------------------------------------
        oid                     Required string/integer. OBJECTID value to add attachment to.
        -------------------     --------------------------------------------------------------------
        attachment_id           Required string. Ids of the attachment to erase.
        -------------------     --------------------------------------------------------------------
        return_moment           Optional boolean. Specify whether the response will report the time
                                attachments were deleted. If True, the server will report the time
                                in the response's `editMoment` key. The default value is False.
        -------------------     --------------------------------------------------------------------
        rollback_on_failure     Optional boolean. Specifies whether the edits should be applied
                                only if all submitted edits succeed. If False, the server will apply
                                the edits that succeed even if some of the submitted edits fail.
                                If True, the server will apply the edits only if all edits succeed.
                                The default value is true.
        ===================     ====================================================================

        :return: dictionary
        """
        params = {
            "f": "json",
            "attachmentIds": attachment_id,
            "gbdVersion": version,
            "returnEditMoment": return_moment,
            "rollbackOnFailure": rollback_on_failure,
        }
        if self._dynamic_layer:
            url = self._url.split("?")[0] + "/%s/deleteAttachments" % oid
            params["layer"] = self._dynamic_layer
        else:
            url = self._url + "/%s/deleteAttachments" % oid
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def _update_attachment(
        self,
        oid,
        attachment_id,
        file_path,
        return_moment=False,
        version=None,
    ):
        """
        Updates an existing attachment with a new file

        =================     ====================================================================
        **Parameter**          **Description**
        -----------------     --------------------------------------------------------------------
        oid                   Required string. OBJECTID value to add attachment to.
        -----------------     --------------------------------------------------------------------
        attachment_id         Required string. Id of the attachment to erase.
        -----------------     --------------------------------------------------------------------
        file_path             Required string. Path to new attachment
        -----------------     --------------------------------------------------------------------
        return_moment         Optional boolean. Specify whether the response will report the time
                              attachments were deleted. If True, the server will report the time
                              in the response's `editMoment` key. The default value is False.
        =================     ====================================================================

        :return: dictionary

        """
        params = {
            "f": "json",
            "attachmentId": attachment_id,
            "returnEditMoment": return_moment,
            "gbdVersion": version,
        }
        files = {"attachment": file_path}
        if self._dynamic_layer is not None:
            url = self.url.split("?")[0] + f"/{oid}/updateAttachment"
            params["layer"] = self._dynamic_layer
        else:
            url = self._url + f"/{oid}/updateAttachment"
        res = self._con.post(path=url, postdata=params, files=files)
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
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def get_unique_values(self, attribute: str, query_string: str = "1=1"):
        """
        Retrieves a list of unique values for a given attribute in the
        :class:`~arcgis.features.FeatureLayer`.


        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        attribute                           Required string. The feature layer attribute to query.
        -------------------------------     --------------------------------------------------------------------
        query_string                        Optional string. SQL Query that will be used to filter attributes
                                            before unique values are returned.
                                            ex. "name_2 like '%K%'"
        ===============================     ====================================================================

        :return:
            A list of unique values

        .. code-block:: python

            # Usage Example with only a "where" sql statement

            >>> from arcgis.features import FeatureLayer

            >>> gis = GIS("pro")
            >>> buck = gis.content.search("owner:"+ gis.users.me.username)
            >>> buck_1 =buck[1]
            >>> lay = buck_1.layers[0]
            >>> layer = lay.get_unique_values(attribute = "COUNTY")
            >>> layer
            ['PITKIN', 'PLATTE', 'TWIN FALLS']

        """

        result = self.query(
            query_string,
            return_geometry=False,
            out_fields=attribute,
            return_distinct_values=True,
        )
        return [feature.attributes[attribute] for feature in result.features]

    # ----------------------------------------------------------------------
    def query_date_bins(
        self,
        bin_field: str | datetime,
        bin_specs: dict,
        out_statistics: list[dict[str, Any]],
        time_filter: Optional[TimeFilter] = None,
        geometry_filter: Optional[GeometryFilter | dict] = None,
        bin_order: Optional[str] = None,
        where: Optional[str] = None,
        return_centroid: Optional[bool] = False,
        in_sr: Optional[dict[str, Any] | int] = None,
        out_sr: Optional[dict[str, Any] | int] = None,
        spatial_rel: Optional[str] = None,
        quantization_params: Optional[dict[str, Any]] = None,
        result_offset: Optional[int] = None,
        result_record_count: Optional[int] = None,
        return_exceeded_limit_features: Optional[bool] = False,
    ):
        """
        The ``query_date_bins`` operation is performed on a :class:`~arcgis.features.FeatureLayer`.
        This operation returns a histogram of features divided into bins based on a date field.
        The response can include statistical aggregations for each bin, such as a count or
        sum, and may also include the aggregate geometries (in other words, centroid) for
        point layers.

        The parameters define the bins, the aggregate information returned, and the included
        features. Bins are defined using the bin parameter. The ``out_statistics`` and
        ``return_centroid`` parameters define the information each bin will provide. Included
        features can be specified by providing a ``time`` extent, ``where`` condition, and a
        spatial filter, similar to a query operation.

        The contents of the ``bin_specs`` parameter provide flexibility for defining bin
        boundaries. The ``bin_specs`` parameter's ``unit`` property defines the time width of each
        bin, such as one year, quarter, month, day, or hour. Fixed bins can use multiple units for
        these time widths. The ``result_offset`` property defines an offset within that time unit.
        For example, if your bin unit is ``day``, and you want bin boundaries to go from noon to
        noon on the next day, the offset would be 12 hours.

        Features can be manipulated with the ``time_filter``, ``where``, and ``geometry_filter``
        parameters. By default, the result will expand to fit the feature's earliest and latest
        point of time. The ``time_filter`` parameter defines a fixed starting point and ending
        point of the features based on the field used in binField. The ``where`` and
        ``geometry_filter`` parameters allow additional filters to be put on the data.

        This operation is only supported on feature services using a spatiotemporal data
        store. As well, the service property ``supportsQueryDateBins`` must be set to true.

        To use pagination with aggregated queries on hosted feature services in ArcGIS
        Enterprise, the ``supportsPaginationOnAggregatedQueries`` property must be ``true`` on
        the layer. Hosted feature services using a spatiotemporal data store do not currently
        support pagination on aggregated queries.

        ==============================     ====================================================================
        **Parameter**                       **Description**
        ------------------------------     --------------------------------------------------------------------
        bin_field                          Required String. The date field used to determine which bin each
                                           feature falls into.
        ------------------------------     --------------------------------------------------------------------
        bin_specs                          Required Dict. A dictionary that describes the characteristics of
                                           bins, such as the size of the bin and its starting position. The
                                           size of each bin is determined by the number of time units denoted
                                           by the ``number`` and ``unit`` properties.

                                           The starting position of the bin is the earliest moment in the
                                           specified unit. For example, each year begins at midnight of January
                                           1. An offset inside the bin parameter can provide an offset to the
                                           starting position of the bin. This can contain a positive or
                                           negative integer value.

                                           A bin can take two forms: either a calendar bin or a fixed bin. A
                                           calendar bin is aware of calendar-specific adjustments, such as
                                           daylight saving time and leap seconds. Fixed bins are, by contrast,
                                           always a specific unit of measurement (for example, 60 seconds in a
                                           minute, 24 hours in a day) regardless of where the date and time of
                                           the bin starts. For this reason, some calendar-specific units are
                                           only supported as calendar bins.

                                           .. code-block:: python

                                                # Calendar bin

                                                >>> bin_specs= {"calendarBin":
                                                                  {"unit": "year",
                                                                    "timezone": "US/Arizona",
                                                                    "offset": {
                                                                        "number": 5,
                                                                        "unit": "hour"}
                                                                  }
                                                               }

                                                # Fixed bin

                                                >>> bin_specs= {"fixedBin":
                                                                 {
                                                                  "number": 12,
                                                                  "unit": "hour",
                                                                  "offset": {
                                                                    "number": 5,
                                                                    "unit": "hour"}
                                                                 }
                                                               }
        ------------------------------     --------------------------------------------------------------------
        out_statistics                     Required List of Dicts. The definitions for one or more field-based
                                           statistics to be calculated:

                                           .. code-block:: python

                                               {
                                                "statisticType": "<count | sum | min | max | avg | stddev | var>",
                                                "onStatisticField": "Field1",
                                                "outStatisticFieldName": "Out_Field_Name1"
                                               }
        ------------------------------     --------------------------------------------------------------------
        time_filter                        Optional list. The format is of [<startTime>, <endTime>] using
                                           datetime.date, datetime.datetime or timestamp in milliseconds.
        ------------------------------     --------------------------------------------------------------------
        geometry_filter                    Optional from :attr:`~arcgis.geometry.filters`. Allows for the
                                           information to be filtered on spatial relationship with another
                                           geometry.
        ------------------------------     --------------------------------------------------------------------
        bin_order                          Optional String. Either "ASC" or "DESC". Determines whether results
                                           are returned in ascending or descending order. Default is ascending.
        ------------------------------     --------------------------------------------------------------------
        where                              Optional String. A WHERE clause for the query filter. SQL '92 WHERE
                                           clause syntax on the fields in the layer is supported for most data
                                           sources.
        ------------------------------     --------------------------------------------------------------------
        return_centroid                    Optional Boolean. Returns the geometry centroid associated with all
                                           the features in the bin. If true, the result includes the geometry
                                           centroid. The default is false. This parameter is only supported on
                                           point data.
        ------------------------------     --------------------------------------------------------------------
        in_sr                              Optional Integer. The WKID for the spatial reference of the input
                                           geometry.
        ------------------------------     --------------------------------------------------------------------
        out_sr                             Optional Integer. The WKID for the spatial reference of the returned
                                           geometry.
        ------------------------------     --------------------------------------------------------------------
        spatial_rel                        Optional String. The spatial relationship to be applied to the input
                                           geometry while performing the query. The supported spatial
                                           relationships include intersects, contains, envelop intersects,
                                           within, and so on. The default spatial relationship is intersects
                                           (``esriSpatialRelIntersects``). Other options are
                                           ``esriSpatialRelContains``, ``esriSpatialRelCrosses``,
                                           ``esriSpatialRelEnvelopeIntersects``,
                                           ``esriSpatialRelIndexIntersects``, ``esriSpatialRelOverlaps``,
                                           ``esriSpatialRelTouches``, and ``esriSpatialRelWithin``.
        ------------------------------     --------------------------------------------------------------------
        quantization_params                Optional Dict. Used to project the geometry onto a virtual grid,
                                           likely representing pixels on the screen.

                                           .. code-block:: python

                                                # upperLeft origin position

                                                {"mode": "view",
                                                 "originPosition": "upperLeft",
                                                 "tolerance": 1.0583354500042335,
                                                 "extent": {
                                                     "type": "extent",
                                                     "xmin": -18341377.47954369,
                                                     "ymin": 2979920.6113554947,
                                                     "xmax": -7546517.393554582,
                                                     "ymax": 11203512.89298139,
                                                     "spatialReference": {
                                                         "wkid": 102100,
                                                         "latestWkid": 3857}
                                                     }
                                                 }

                                                # lowerLeft origin position

                                                {"mode": "view",
                                                 "originPosition": "lowerLeft",
                                                 "tolerance": 1.0583354500042335,
                                                 "extent": {
                                                    "type": "extent",
                                                    "xmin": -18341377.47954369,
                                                    "ymin": 2979920.6113554947,
                                                    "xmax": -7546517.393554582,
                                                    "ymax": 11203512.89298139,
                                                    "spatialReference": {
                                                        "wkid": 102100,
                                                        "latestWkid": 3857}
                                                    }
                                                }

                                           See `Quantization parameters JSON properties <https://developers.arcgis.com/rest/services-reference/enterprise/query-date-bins-fsl.htm>`_
                                           for details on format of this parameter.

                                           .. note::
                                                This parameter only applies if the layer's
                                                ``supportsCoordinateQuantization`` property is ``true``.
        ------------------------------     --------------------------------------------------------------------
        result_offset                      Optional Int. This parameter fetches query results by skipping the
                                           specified number of records and starting from the next record. The
                                           default is 0.

                                           Note:
                                                This parameter only applies if the layer's
                                                ``supportsPagination`` property is ``true``.
        ------------------------------     --------------------------------------------------------------------
        result_record_count                Optional Int. This parameter fetches query results up to the value
                                           specified. When ``result_offset`` is specified, but this parameter
                                           is not, the map service defaults to the layer's ``maxRecordCount``
                                           property. The maximum value for this parameter is the value of the
                                           ``maxRecordCount`` property. The minimum value entered for this
                                           parameter cannot be below 1.

                                           Note:
                                                This parameter only applies if the layer's
                                                ``supportsPagination`` property is ``true``.
        ------------------------------     --------------------------------------------------------------------
        return_exceeded_limit_features     Optional Boolean. When set to ``True``, features are returned even
                                           when the results include ``"exceededTransferLimit": true``. This
                                           allows a client to find the resolution in which the transfer limit
                                           is no longer exceeded withou making multiple calls. The default
                                           value is ``False``.
        ==============================     ====================================================================

        :return:
            A Dict containing the resulting features and fields.

        .. code-block:: python

            # Usage Example

            >>> flyr_item = gis.content.search("*", "Feature Layer")[0]
            >>> flyr = flyr_item.layers[0]

            >>> qy_result = flyr.query_date_bins(bin_field="boundary",
                                                 bin_specs={"calendarBin":
                                                              {"unit":"day",
                                                               "timezone": "America/Los_Angeles",
                                                               "offset": {"number": 8,
                                                                          "unit": "hour"}
                                                              }
                                                            },
                                                 out_statistics=[{"statisticType": "count",
                                                                  "onStatisticField": "objectid",
                                                                  "outStatisticFieldName": "item_sold_count"},
                                                                 {"statisticType": "avg",
                                                                 "onStatisticField": "price",
                                                                 "outStatisticFieldName": "avg_daily_revenue "}],
                                                 time=[1609516800000, 1612195199999])
            >>> qy_result
               {
                "features": [
                  {
                    "attributes": {
                      "boundary": 1609516800000,
                      "avg_daily_revenue": 300.40,
                      "item_sold_count": 79
                    }
                  },
                  {
                    "attributes": {
                      "boundary": 1612108800000,
                      "avg_daily_revenue": null,
                      "item_sold_count": 0
                    }
                  }
                ],
                "fields": [
                  {
                    "name": "boundary",
                    "type": "esriFieldTypeDate"
                  },
                  {
                    "name": "item_sold_count",
                    "alias": "item_sold_count",
                    "type": "esriFieldTypeInteger"
                  },
                  {
                    "name": "avg_daily_revenue",
                    "alias": "avg_daily_revenue",
                    "type": "esriFieldTypeDouble"
                  }
                ],
                "exceededTransferLimit": false
              }


        """

        qdb_url = self._url + "/queryDateBins"
        params = {
            "binField": bin_field,
            "bin": bin_specs,
            "f": "json",
        }

        layer_props = dict(self.properties)

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

        if bin_order:
            params["binOrder"] = bin_order
        if out_statistics:
            params["outStatistics"] = out_statistics
        if where:
            params["where"] = where
        if return_centroid:
            params["returnCentroid"] = return_centroid
        if in_sr:
            params["inSR"] = in_sr
        if out_sr:
            params["outSR"] = out_sr
        if spatial_rel:
            if spatial_rel in layer_props["supportedSpatialRelationships"]:
                params["spatialRel"] = spatial_rel
            else:
                print(
                    "Designated spatial_rel not supported. Defaulting to esriSpacialRelIntersects..."
                )
        if quantization_params:
            if layer_props["supportsCoordinatesQuantization"]:
                params["quantizationParameters"] = quantization_params
            else:
                print(
                    "Coordinate quantization is not enabled for this layer. Ignoring quantization_params..."
                )
        if result_offset:
            if layer_props["advancedQueryCapabilities"]["supportsPagination"]:
                params["resultOffset"] = result_offset
            else:
                print(
                    "Query pagination is not enabled for this layer. Ignoring result_offset..."
                )
        if result_record_count:
            if layer_props["advancedQueryCapabilities"]["supportsPagination"]:
                params["resultRecordCount"] = result_record_count
            else:
                print(
                    "Query pagination is not enabled for this layer. Ignoring result_record_count..."
                )
        if return_exceeded_limit_features:
            params["returnExceededLimitedFeatures"] = return_exceeded_limit_features

        result = self._con.post(qdb_url, params)
        return result

    # ----------------------------------------------------------------------
    def query_top_features(
        self,
        top_filter: Optional[dict[str, str]] = None,
        where: Optional[str] = None,
        objectids: Optional[list[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        geometry_filter: Optional[GeometryFilter] = None,
        out_fields: str = "*",
        return_geometry: bool = True,
        return_centroid: bool = False,
        max_allowable_offset: Optional[float] = None,
        out_sr: Optional[Union[str, dict[str, int]]] = None,
        geometry_precision: Optional[int] = None,
        return_ids_only: bool = False,
        return_extents_only: bool = False,
        order_by_field: Optional[str] = None,
        return_z: bool = False,
        return_m: bool = False,
        result_type: Optional[str] = None,
        as_df: bool = True,
    ):
        """
        The ``query_top_features`` is performed on a :class:`~arcgis.features.FeatureLayer`. This operation returns a
        feature set or spatially enabled dataframe based on the top features by order within a group. For example, when
        querying counties in the United States, you want to return the top five counties by population in
        each state. To do this, you can use ``query_top_features`` to group by state name, order by desc on
        the population and return the first five rows from each group (state).

        The ``top_filter`` parameter is used to set the group by, order by, and count criteria used in
        generating the result. The operation also has many of the same parameters (for example, where
        and geometry) as the layer query operation. However, unlike the layer query operation,
        ``query_top_feaures`` does not support parameters such as outStatistics and its related parameters
        or return distinct values. Consult the ``advancedQueryCapabilities`` layer property for more details.

        If the feature layer collection supports the `query_top_feaures` operation, it will include
        `"supportsTopFeaturesQuery": True`, in the ``advancedQueryCapabilities`` layer property.

        .. note::
            See the :attr:`~arcgis.features.FeatureLayer.query` method for a similar function.

        ================================     ====================================================================
        **Parameter**                         **Description**
        --------------------------------     --------------------------------------------------------------------
        top_filter                           Required Dict. The `top_filter` define the aggregation of the data.

                                               - groupByFields define the field or fields used to aggregate
                                                your data.
                                               - topCount defines the number of features returned from the top
                                                features query and is a numeric value.
                                               - orderByFields defines the order in which the top features will
                                                be returned. orderByFields can be specified in
                                                either ascending (asc) or descending (desc)
                                                order, ascending being the default.

                                             Example: {"groupByFields": "worker", "topCount": 1,
                                                       "orderByFields": "employeeNumber"}
        --------------------------------     --------------------------------------------------------------------
        where	                             Optional String. A WHERE clause for the query filter. SQL '92 WHERE
                                             clause syntax on the fields in the layer is supported for most data
                                             sources.
        --------------------------------     --------------------------------------------------------------------
        objectids	                         Optional List. The object IDs of the layer or table to be queried.
        --------------------------------     --------------------------------------------------------------------
        start_time                           Optional Datetime. The starting time to query for.
        --------------------------------     --------------------------------------------------------------------
        end_time                             Optional Datetime. The end date to query for.
        --------------------------------     --------------------------------------------------------------------
        geometry_filter                      Optional from arcgis.geometry.filter. Allows for the information to
                                             be filtered on spatial relationship with another geometry.
        --------------------------------     --------------------------------------------------------------------
        out_fields                           Optional String. The list of fields to include in the return results.
        --------------------------------     --------------------------------------------------------------------
        return_geometry                      Optional Boolean. If False, the query will not return geometries.
                                             The default is True.
        --------------------------------     --------------------------------------------------------------------
        return_centroid                      Optional Boolean. If True, the centroid of the geometry will be
                                             added to the output.
        --------------------------------     --------------------------------------------------------------------
        max_allowable_offset                 Optional float. This option can be used to specify the
                                             max_allowable_offset to be used for generalizing geometries returned
                                             by the query operation.
                                             The max_allowable_offset is in the units of out_sr. If out_sr is not
                                             specified, max_allowable_offset is assumed to be in the unit of the
                                             spatial reference of the layer.
        --------------------------------     --------------------------------------------------------------------
        out_sr                               Optional Integer. The WKID for the spatial reference of the returned
                                             geometry.
        --------------------------------     --------------------------------------------------------------------
        geometry_precision                   Optional Integer. This option can be used to specify the number of
                                             decimal places in the response geometries returned by the query
                                             operation.
                                             This applies to X and Y values only (not m or z-values).
        --------------------------------     --------------------------------------------------------------------
        return_ids_only                      Optional boolean. Default is False.  If true, the response only
                                             includes an array of object IDs. Otherwise, the response is a
                                             feature set.
        --------------------------------     --------------------------------------------------------------------
        return_extent_only                   Optional boolean. If true, the response only includes the extent of
                                             the features that would be returned by the query. If
                                             returnCountOnly=true, the response will return both the count and
                                             the extent.
                                             The default is false. This parameter applies only if the
                                             supportsReturningQueryExtent property of the layer is true.
        --------------------------------     --------------------------------------------------------------------
        order_by_field                       Optional Str. Optional string. One or more field names on which the
                                             features/records need to be ordered. Use ASC or DESC for ascending
                                             or descending, respectively, following every field to control the
                                             ordering.
                                             example: STATE_NAME ASC, RACE DESC, GENDER
        --------------------------------     --------------------------------------------------------------------
        return_z                             Optional boolean. If true, Z values are included in the results if
                                             the features have Z values. Otherwise, Z values are not returned.
                                             The default is False.
        --------------------------------     --------------------------------------------------------------------
        return_m                             Optional boolean. If true, M values are included in the results if
                                             the features have M values. Otherwise, M values are not returned.
                                             The default is false.
        --------------------------------     --------------------------------------------------------------------
        result_type                          Optional String. The result_type can be used to control the number
                                             of features returned by the query operation.
                                             Values: none | standard | tile
        --------------------------------     --------------------------------------------------------------------
        as_df                                Optional Boolean. If False, the result is returned as a FeatureSet.
                                             If True (default) the result is returned as a spatially enabled dataframe.
        ================================     ====================================================================


        :return: Default is a pd.DataFrame, but when ```as_df=False``` returns a :class:`~arcgis.feature.FeatureSet`.
                  If ```return_count_only=True```, the return type is Integer.
                  If ```return_ids_only=True```, a list of value is returned.


        """
        import datetime as _datetime

        return_count_only = False
        params = {
            "f": "json",
        }
        params["returnCentroid"] = return_centroid
        if where:
            params["where"] = where
        else:
            params["where"] = "1=1"
        if objectids and isinstance(objectids, (list, tuple)):
            params["objectIds"] = ",".join([str(obj) for obj in objectids])
        elif objectids and isinstance(objectids, str):
            params["objectIds"] = objectids
        if start_time and isinstance(start_time, _datetime.datetime):
            start_time = str(int(start_time.timestamp() * 1000))
        if end_time and isinstance(start_time, _datetime.datetime):
            end_time = str(int(start_time.timestamp() * 1000))
        if start_time and not end_time:
            params["time"] = "%s, null" % start_time
        elif end_time and not start_time:
            params["time"] = "null, %s" % end_time
        elif start_time and end_time:
            params["time"] = "%s, %s" % (start_time, end_time)
        if geometry_filter and isinstance(geometry_filter, GeometryFilter):
            for key, val in geometry_filter.filter:
                params[key] = val
        elif geometry_filter and isinstance(geometry_filter, dict):
            for key, val in geometry_filter.items():
                params[key] = val
        if top_filter:
            params["topFilter"] = top_filter
        if out_fields and isinstance(out_fields, (list, tuple)):
            params["outFields"] = ",".join(out_fields)
        elif out_fields and isinstance(out_fields, str):
            params["outFields"] = out_fields
        else:
            params["outFields"] = "*"
        if return_geometry == False:
            params["returnGeometry"] = False
        elif return_geometry == True:
            params["returnGeometry"] = True
        if max_allowable_offset:
            params["maxAllowableOffset"] = max_allowable_offset
        if geometry_precision:
            params["geometryPrecision"] = geometry_precision
        if out_sr:
            params["outSR"] = out_sr
        if return_ids_only:
            params["returnIdsOnly"] = return_ids_only
        if return_count_only:
            params["returnCountOnly"] = return_count_only
        if return_z:
            params["returnZ"] = return_z
        if return_m:
            params["returnM"] = return_m
        if result_type:
            params["resultType"] = result_type
        else:
            params["resultType"] = "none"
        if order_by_field:
            params["orderByFields"] = order_by_field
        url = self._url + "/queryTopFeatures"
        if as_df and return_count_only == False and return_ids_only == False:
            return self._query_df(url, params)
        elif as_df == False and return_count_only == False and return_ids_only == False:
            res = self._con.post(url, params)
            return FeatureSet.from_dict(res)
        elif return_count_only:
            res = self._con.post(url, params)
            return res
        elif return_ids_only:
            res = self._con.post(url, params)
            return res
        return None

    def _qa_worker(self, url, params):
        """Processes the job, gets the status and returns the results"""

        count = self.query(where=params.get("where", "1=1"), return_count_only=True)
        if "maxRecordCount" in self.properties:
            max_records = self.properties["maxRecordCount"]
        else:
            max_records = 1000

        jobs = {}
        failed = {}
        retry_count = 0
        records = []
        parts = []
        df = None
        if count > max_records:
            oid_info = self.query(
                where=params.get("where", "1=1"),
                geometry_filter=params.get("geometry_filter", None),
                time_filter=params.get("time_filter", None),
                return_ids_only=True,
            )
            for ids in chunks(oid_info["objectIds"], max_records):
                ids = [str(i) for i in ids]
                sql = "%s in (%s)" % (
                    oid_info["objectIdFieldName"],
                    ",".join(ids),
                )
                params["where"] = sql
                jobs[sql] = self._con.post(url, params)
            for where, submit_job in jobs.items():
                if "statusUrl" in submit_job:
                    jobs[where] = self._status_via_url(
                        self._con, submit_job["statusUrl"], {"f": "json"}
                    )
            for where, download_json in jobs.items():
                if "resultUrl" in download_json:
                    jobs[where] = self._con.get(
                        download_json["resultUrl"], {"f": "json"}
                    )
            for where, json_file in jobs.items():
                if isinstance(json_file, str) and os.path.isfile(json_file):
                    with open(json_file, "r") as reader:
                        feature_dict = json.loads(reader.read())
                        parts.append(feature_dict)
                    os.remove(json_file)
                else:
                    if isinstance(json_file, str):
                        feature_dict = json.loads(json_file)
                    else:
                        feature_dict = json_file
                    parts.append(feature_dict)
                del where, json_file

        else:
            submit_job = self._con.post(url, params)
            if "statusUrl" in submit_job:
                status_job = self._status_via_url(
                    self._con, submit_job["statusUrl"], {"f": "json"}
                )
            else:
                raise Exception(f"Job Failed: {submit_job}")
            if "resultUrl" in status_job:
                download_json = self._con.get(status_job["resultUrl"], {"f": "json"})
            else:
                raise Exception(f"Job Failed: {status_job}")
            if isinstance(download_json, str) and os.path.isfile(download_json):
                with open(download_json, "r") as reader:
                    feature_dict = json.loads(reader.read())
                    parts.append(feature_dict)
                os.remove(download_json)
            else:
                if isinstance(download_json, str):
                    feature_dict = json.loads(download_json)
                else:
                    feature_dict = download_json
                parts.append(feature_dict)
        # process the parts into a Spatially Enabled DataFrame
        #
        import pandas as pd

        def _process_result(featureset_dict):
            """converts the Dictionary to an SeDF"""
            import pandas as pd
            import arcgis
            import numpy as np
            from datetime import datetime as _datetime

            _fld_lu = {
                "esriFieldTypeSmallInteger": pd.Int32Dtype(),
                "esriFieldTypeInteger": pd.Int32Dtype(),
                "esriFieldTypeSingle": pd.Float64Dtype(),
                "esriFieldTypeDouble": pd.Float64Dtype(),
                "esriFieldTypeFloat": pd.Float64Dtype(),
                "esriFieldTypeString": pd.StringDtype(),
                "esriFieldTypeDate": _datetime,
                "esriFieldTypeOID": pd.Int64Dtype(),
                "esriFieldTypeGeometry": object,
                "esriFieldTypeBlob": object,
                "esriFieldTypeRaster": object,
                "esriFieldTypeGUID": pd.StringDtype(),
                "esriFieldTypeGlobalID": pd.StringDtype(),
                "esriFieldTypeXML": object,
                "esriFieldTypeTimeOnly": _datetime,
                "esriFieldTypeDateOnly": _datetime,
                "esriFieldTypeTimestampOffset": _datetime,
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
                    if fld["type"] in [
                        "esriFieldTypeDate",
                        "esriFieldTypeDateOnly",
                        "esriFieldTypeTimestampOffset",
                    ]:
                        dfields.append(fld["name"])
            if "SHAPE" in df:
                df.spatial.set_geometry("SHAPE")
            if len(dfields) > 0:
                df[dfields] = df[dfields].apply(pd.to_datetime, unit="ms")
            return df

        if len(parts) == 1:
            return _process_result(featureset_dict=parts[0])
        elif len(parts) == 0:
            return pd.DataFrame([])
        else:
            results = pd.concat([_process_result(df) for df in parts]).reset_index(
                drop=True
            )
            return results

    # ----------------------------------------------------------------------
    def query_analytics(
        self,
        out_analytics: list[dict],  #
        where: str = "1=1",  #
        out_fields: Union[str, list[str]] = "*",  #
        analytic_where: Optional[str] = None,  #
        geometry_filter: Optional[GeometryFilter] = None,  #
        out_sr: Optional[Union[dict[str, int], str]] = None,  #
        return_geometry: bool = True,
        order_by: Optional[str] = None,
        result_type: Optional[str] = None,
        cache_hint: Optional[str] = None,
        result_offset: Optional[int] = None,
        result_record_count: Optional[int] = None,
        quantization_param: Optional[dict[str, Any]] = None,
        sql_format: Optional[str] = None,
        future: bool = True,
        **kwargs,
    ):
        """
        The ``query_analytics`` exposes the standard ``SQL`` windows functions that compute
        aggregate and ranking values based on a group of rows called window
        partition. The window function is applied to the rows after the
        partitioning and ordering of the rows. ``query_analytics`` defines a
        window or user-specified set of rows within a query result set.
        ``query_analytics`` can be used to compute aggregated values such as moving
        averages, cumulative aggregates, or running totals.

        .. note::
            See the :attr:`~arcgis.features.FeatureLayer.query` method for a similar function.

        **SQL Windows Function**

        A window function performs a calculation across a set of rows (SQL partition
        or window) that are related to the current row. Unlike regular aggregate
        functions, use of a window function does not return single output row. The
        rows retain their separate identities with each calculation appended to the
        rows as a new field value. The window function can access more than just
        the current row of the query result.

        ``query_analytics`` currently supports the following windows functions:
             - Aggregate functions
             - Analytic functions
             - Ranking functions

        **Aggregate Functions**

        Aggregate functions are deterministic function that perform a calculation on
        a set of values and return a single value. They are used in the select list
        with optional HAVING clause. GROUP BY clause can also be used to calculate
        the aggregation on categories of rows. ``query_analytics`` can be used to
        calculate the aggregation on a specific range of value. Supported aggregate
        functions are:
             - Min
             - Max
             - Sum
             - Count
             - AVG
             - STDDEV
             - VAR

        **Analytic Functions**

        Several analytic functions available now in all SQL vendors to compute an
        aggregate value based on a group of rows or windows partition. Unlike
        aggregation functions, analytic functions can return single or multiple rows
        for each group.
             - CUM_DIST
             - FIRST_VALUE
             - LAST_VALUE
             - LEAD
             - LAG
             - PERCENTILE_DISC
             - PERCENTILE_CONT
             - PERCENT_RANK

        **Ranking Functions**

        Ranking functions return a ranking value for each row in a partition. Depending
        on the function that is used, some rows might receive the same value as other rows.

             - RANK
             - NTILE
             - DENSE_RANK
             - ROW_NUMBER


        **Partitioning**

        Partitions are extremely useful when you need to calculate the same metric over
        different group of rows. It is very powerful and has many potential usages. For
        example, you can add partition by to your window specification to look at
        different groups of rows individually.

        ``partitionBy`` clause divides the query result set into partitions and the sql
        window function is applied to each partition.
        The 'partitionBy' clause normally refers to the column by which the result is
        partitioned. 'partitionBy' can also be a value expression (column expression or
        function) that references any of the selected columns (not aliases).



        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        out_analytics                       Required List. A set of analytics to calculate on the Feature Layer.

                                            The definitions for one or more field-based or expression analytics
                                            to be computed. This parameter is supported only on layers/tables that
                                            return `true` for *supportsAnalytics* property.

                                            .. note::
                                                If `outAnalyticFieldName` is empty or missing, the server assigns
                                                a field name to the returned analytic field.

                                            The argument should be a list of dictionaries that define analystics.
                                            An analytic definition specifies:

                                            * the type of analytic - key: `analyticType`
                                            * the field or expression on which it is to be computed - key: `onAnalyticField`
                                            * the resulting output field name -key: `outAnalyticFieldName`
                                            * the analytic specifications - `analysticParameters`

                                            See `Overview <https://developers.arcgis.com/rest/services-reference/enterprise/query-analytic.htm#GUID-1713C237-B155-4CFE-8470-FEB3255B7C60>`_
                                            for details.

                                            .. code-block:: python

                                                # Dictionary structure and options for this parameter

                                                [
                                                  {
                                                    "analyticType": "<COUNT | SUM | MIN | MAX | AVG | STDDEV | VAR | FIRST_VALUE, LAST_VALUE, LAG, LEAD, PERCENTILE_CONT, PERCENTILE_DISC, PERCENT_RANK, RANK, NTILE, DENSE_RANK, EXPRESSION>",
                                                    "onAnalyticField": "Field1",
                                                    "outAnalyticFieldName": "Out_Field_Name1",
                                                    "analyticParameters": {
                                                         "orderBy": "<orderBy expression",
                                                         "value": <double value>,// percentile value
                                                         "partitionBy": "<field name or expression>",
                                                         "offset": <integer>, // used by LAG/LEAD
                                                         "windowFrame": {
                                                            "type": "ROWS" | "RANGE",
                                                            "extent": {
                                                               "extentType": "PRECEDING" | "BOUNDARY",
                                                               "PRECEDING": {
                                                                  "type": <"UNBOUNDED" |
                                                                          "NUMERIC_CONSTANT" |
                                                                           "CURRENT_ROW">
                                                                   "value": <numeric constant value>
                                                                }
                                                                "BOUNDARY": {
                                                                 "start": "UNBOUNDED_PRECEDING",
                                                                          "NUMERIC_PRECEDING",
                                                                           "CURRENT_ROW",
                                                                 "startValue": <numeric constant value>,
                                                                 "end": <"UNBOUNDED_FOLLOWING" |
                                                                         "NUMERIC_FOLLOWING" |
                                                                         "CURRENT_ROW",
                                                                 "endValue": <numeric constant value>
                                                                }
                                                              }
                                                            }
                                                         }
                                                    }
                                                  }
                                                ]


                                            .. code-block:: python

                                                # Usage Example:

                                                >>> out_analytics =
                                                        [{"analyticType": "FIRST_VALUE",
                                                          "onAnalyticField": "POP1990",
                                                          "analyticParameters": {
                                                                                 "orderBy": "POP1990",
                                                                                 "partitionBy": "state_name"
                                                                                },
                                                          "outAnalyticFieldName": "FirstValue"}]
        -------------------------------     --------------------------------------------------------------------
        where                               Optional string. The default is 1=1. The selection sql statement.
        -------------------------------     --------------------------------------------------------------------
        out_fields                          Optional List of field names to return. Field names can be specified
                                            either as a List of field names or as a comma separated string.
                                            The default is "*", which returns all the fields.
        -------------------------------     --------------------------------------------------------------------
        analytic_where                      Optional String. A where clause for the query filter that applies to
                                            the result set of applying the source where clause and all other params.
        -------------------------------     --------------------------------------------------------------------
        geometry_filter                     Optional from arcgis.geometry.filter. Allows for the information to
                                            be filtered on spatial relationship with another geometry.
        -------------------------------     --------------------------------------------------------------------
        out_sr                              Optional Integer. The WKID for the spatial reference of the returned
                                            geometry.
        -------------------------------     --------------------------------------------------------------------
        return_geometry                     Optional boolean. If true, geometry is returned with the query.
                                            Default is true.
        -------------------------------     --------------------------------------------------------------------
        order_by                            Optional string. One or more field names on which the
                                            features/records need to be ordered. Use ASC or DESC for ascending
                                            or descending, respectively, following every field to control the
                                            ordering.
                                            example: STATE_NAME ASC, RACE DESC, GENDER
        -------------------------------     --------------------------------------------------------------------
        result_type                         Optional string. The result_type parameter can be used to control
                                            the number of features returned by the query operation.
                                            Values: None | standard | tile
        -------------------------------     --------------------------------------------------------------------
        cache_hint                          Optional Boolean. If you are performing the same query multiple times,
                                            a user can ask the server to cache the call to obtain the results
                                            quicker.  The default is `False`.
        -------------------------------     --------------------------------------------------------------------
        result_offset                       Optional integer. This option can be used for fetching query results
                                            by skipping the specified number of records and starting from the
                                            next record (that is, resultOffset + 1th).
        -------------------------------     --------------------------------------------------------------------
        result_record_count                 Optional integer. This option can be used for fetching query results
                                            up to the result_record_count specified. When result_offset is
                                            specified but this parameter is not, the map service defaults it to
                                            max_record_count. The maximum value for this parameter is the value
                                            of the layer's max_record_count property.
        -------------------------------     --------------------------------------------------------------------
        quantization_parameters             Optional dict. Used to project the geometry onto a virtual grid,
                                            likely representing pixels on the screen.
        -------------------------------     --------------------------------------------------------------------
        sql_format                          Optional string.  The sql_format parameter can be either standard
                                            SQL92 standard or it can use the native SQL of the underlying
                                            datastore native. The default is none which means the sql_format
                                            depends on useStandardizedQuery parameter.
                                            Values: `none | standard | native`
        -------------------------------     --------------------------------------------------------------------
        future                              Optional Boolean. This determines if a `Future` object is returned
                                            (True) the method returns the results directly (False).
        ===============================     ====================================================================


        :return:
            A Pandas DataFrame (pd.DataFrame)

        """

        if self._gis._portal.is_arcgisonline == False:
            raise Exception(
                "`query_analytics` is only supported on ArcGIS Online Hosted Feature Layers."
            )

        url = self._url + "/queryAnalytic"
        params = {"f": "json", "dataFormat": "json"}
        if where:
            params["where"] = where
        if analytic_where:
            params["analyticWhere"] = analytic_where
        if geometry_filter and isinstance(geometry_filter, GeometryFilter):
            for key, val in geometry_filter.filter:
                params[key] = val
        elif geometry_filter and isinstance(geometry_filter, dict):
            for key, val in geometry_filter.items():
                params[key] = val
        if out_sr:
            params["outSR"] = out_sr
        if out_fields:
            params["outFields"] = out_fields
        if out_analytics:
            params["outAnalytics"] = out_analytics
        if order_by:
            params["orderByFields"] = order_by
        if result_type:
            params["resultType"] = result_type
        if not cache_hint is None:
            params["cacheHint"] = cache_hint
        if result_offset:
            params["resultOffset"] = result_offset
        if result_record_count:
            params["resultRecordCount"] = result_record_count
        if quantization_param:
            params["quantizationParameters"] = quantization_param
        if future:
            params["async"] = future
        if sql_format:
            params["sql_format"] = sql_format
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                params[k] = v
        params["async"] = True
        executor = concurrent.futures.ThreadPoolExecutor(1)
        future_job = executor.submit(self._qa_worker, **{"url": url, "params": params})
        executor.shutdown(False)

        if future == False:
            res = future_job.result()
            del executor
            return res
        return future_job

    # ----------------------------------------------------------------------
    def query(
        self,
        where: str = "1=1",
        out_fields: Union[str, list[str]] = "*",
        time_filter: Optional[list[datetime]] = None,
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
        object_ids: Optional[list[str]] = None,
        distance: Optional[int] = None,
        units: Optional[str] = None,
        max_allowable_offset: Optional[int] = None,
        out_sr: Optional[Union[dict[str, int], str]] = None,
        geometry_precision: Optional[int] = None,
        gdb_version: Optional[str] = None,
        order_by_fields: Optional[str] = None,
        out_statistics: Optional[list[dict[str, Any]]] = None,
        return_z: bool = False,
        return_m: bool = False,
        multipatch_option: tuple = None,
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
        **kwargs,
    ):
        """
        The ``query`` method queries a :class:`~arcgis.features.FeatureLayer` based on a ``sql`` statement.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        where                               Optional string. The default is 1=1. The selection sql statement.
        -------------------------------     --------------------------------------------------------------------
        out_fields                          Optional List of field names to return. Field names can be specified
                                            either as a List of field names or as a comma separated string.
                                            The default is "*", which returns all the fields.

                                            .. note::
                                                If specifying `return_count_only`, `return_id_only`, or `return_extent_only`
                                                as True, do not specify this parameter in order to avoid errors.
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
                                            This parameter only applies if supportsQueryWithDistance is true.
                                            Values: `esriSRUnit_Meter | esriSRUnit_StatuteMile |
                                                    esriSRUnit_Foot | esriSRUnit_Kilometer |
                                                    esriSRUnit_NauticalMile | esriSRUnit_USNauticalMile`
        -------------------------------     --------------------------------------------------------------------
        time_filter                         Optional list. The format is of [<startTime>, <endTime>] using
                                            datetime.date, datetime.datetime or timestamp in milliseconds.
                                            Syntax: time_filter=[<startTime>, <endTime>] ; specified as
                                                    datetime.date, datetime.datetime or timestamp in
                                                    milliseconds
        -------------------------------     --------------------------------------------------------------------
        geometry_filter                     Optional from :attr:`~arcgis.geometry.filters`. Allows for the information to
                                            be filtered on spatial relationship with another geometry.
        -------------------------------     --------------------------------------------------------------------
        max_allowable_offset                Optional float. This option can be used to specify the
                                            max_allowable_offset to be used for generalizing geometries returned
                                            by the query operation.
                                            The max_allowable_offset is in the units of out_sr. If out_sr is not
                                            specified, max_allowable_offset is assumed to be in the unit of the
                                            spatial reference of the layer.
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
                                            applies only if the isDataVersioned property of the layer is true.
                                            If this is not specified, the query will apply to the published
                                            map's version.
        -------------------------------     --------------------------------------------------------------------
        return_geometry                     Optional boolean. If true, geometry is returned with the query.
                                            Default is true.
        -------------------------------     --------------------------------------------------------------------
        return_distinct_values              Optional boolean.  If true, it returns distinct values based on the
                                            fields specified in out_fields. This parameter applies only if the
                                            supportsAdvancedQueries property of the layer is true.
        -------------------------------     --------------------------------------------------------------------
        return_ids_only                     Optional boolean. Default is False.  If true, the response only
                                            includes an array of object IDs. Otherwise, the response is a
                                            feature set.
        -------------------------------     --------------------------------------------------------------------
        return_count_only                   Optional boolean. If true, the response only includes the count
                                            (number of features/records) that would be returned by a query.
                                            Otherwise, the response is a feature set. The default is false. This
                                            option supersedes the returnIdsOnly parameter. If
                                            returnCountOnly = true, the response will return both the count and
                                            the extent.
        -------------------------------     --------------------------------------------------------------------
        return_extent_only                  Optional boolean. If true, the response only includes the extent of
                                            the features that would be returned by the query. If
                                            returnCountOnly=true, the response will return both the count and
                                            the extent.
                                            The default is false. This parameter applies only if the
                                            supportsReturningQueryExtent property of the layer is true.
        -------------------------------     --------------------------------------------------------------------
        order_by_fields                     Optional string. One or more field names on which the
                                            features/records need to be ordered. Use ASC or DESC for ascending
                                            or descending, respectively, following every field to control the
                                            ordering.
                                            example: STATE_NAME ASC, RACE DESC, GENDER

                                            .. note::
                                                If specifying `return_count_only`, `return_id_only`, or `return_extent_only`
                                                as True, do not specify this parameter in order to avoid errors.
        -------------------------------     --------------------------------------------------------------------
        group_by_fields_for_statistics      Optional string. One or more field names on which the values need to
                                            be grouped for calculating the statistics.
                                            example: STATE_NAME, GENDER
        -------------------------------     --------------------------------------------------------------------
        out_statistics                      Optional list of dictionaries. The definitions for one or more field-based
                                            statistics to be calculated.

                                            Syntax:

                                            [
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
        statistic_filter                    Optional ``StatisticFilter`` instance. The definitions for one or more field-based
                                            statistics can be added, e.g. statisticType, onStatisticField, or
                                            outStatisticFieldName.

                                            Syntax:

                                            sf = StatisticFilter()
                                            sf.add(statisticType="count", onStatisticField="1", outStatisticFieldName="total")
                                            sf.filter
        -------------------------------     --------------------------------------------------------------------
        return_z                            Optional boolean. If true, Z values are included in the results if
                                            the features have Z values. Otherwise, Z values are not returned.
                                            The default is False.
        -------------------------------     --------------------------------------------------------------------
        return_m                            Optional boolean. If true, M values are included in the results if
                                            the features have M values. Otherwise, M values are not returned.
                                            The default is false.
        -------------------------------     --------------------------------------------------------------------
        multipatch_option                   Optional x/y footprint. This option dictates how the geometry of
                                            a multipatch feature will be returned.
        -------------------------------     --------------------------------------------------------------------
        result_offset                       Optional integer. This option can be used for fetching query results
                                            by skipping the specified number of records and starting from the
                                            next record (that is, resultOffset + 1th). This option is ignored
                                            if return_all_records is True (i.e. by default).
        -------------------------------     --------------------------------------------------------------------
        result_record_count                 Optional integer. This option can be used for fetching query results
                                            up to the result_record_count specified. When result_offset is
                                            specified but this parameter is not, the map service defaults it to
                                            max_record_count. The maximum value for this parameter is the value
                                            of the layer's max_record_count property. This option is ignored if
                                            return_all_records is True (i.e. by default).
        -------------------------------     --------------------------------------------------------------------
        quantization_parameters             Optional dict. Used to project the geometry onto a virtual grid,
                                            likely representing pixels on the screen.
        -------------------------------     --------------------------------------------------------------------
        return_centroid                     Optional boolean. Used to return the geometry centroid associated
                                            with each feature returned. If true, the result includes the geometry
                                            centroid. The default is false.
        -------------------------------     --------------------------------------------------------------------
        return_all_records                  Optional boolean. When True, the query operation will call the
                                            service until all records that satisfy the where_clause are
                                            returned. Note: result_offset and result_record_count will be
                                            ignored if return_all_records is True. Also, if return_count_only,
                                            return_ids_only, or return_extent_only are True, this parameter
                                            will be ignored.
        -------------------------------     --------------------------------------------------------------------
        result_type                         Optional string. The result_type parameter can be used to control
                                            the number of features returned by the query operation.
                                            Values: None | standard | tile
        -------------------------------     --------------------------------------------------------------------
        historic_moment                     Optional integer. The historic moment to query. This parameter
                                            applies only if the layer is archiving enabled and the
                                            supportsQueryWithHistoricMoment property is set to true. This
                                            property is provided in the layer resource.

                                            If historic_moment is not specified, the query will apply to the
                                            current features.
        -------------------------------     --------------------------------------------------------------------
        sql_format                          Optional string.  The sql_format parameter can be either standard
                                            SQL92 standard or it can use the native SQL of the underlying
                                            datastore native. The default is none which means the sql_format
                                            depends on useStandardizedQuery parameter.
                                            Values: none | standard | native
        -------------------------------     --------------------------------------------------------------------
        return_true_curves                  Optional boolean. When set to true, returns true curves in output
                                            geometries. When set to false, curves are converted to densified
                                            polylines or polygons.
        -------------------------------     --------------------------------------------------------------------
        return_exceeded_limit_features      Optional boolean. Optional parameter which is true by default. When
                                            set to true, features are returned even when the results include
                                            'exceededTransferLimit': True.

                                            When set to false and querying with resultType = tile features are
                                            not returned when the results include 'exceededTransferLimit': True.
                                            This allows a client to find the resolution in which the transfer
                                            limit is no longer exceeded without making multiple calls.
        -------------------------------     --------------------------------------------------------------------
        as_df                               Optional boolean.  If True, the results are returned as a DataFrame
                                            instead of a FeatureSet.
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

                                            **Examples**


                                                ===========     ===================================
                                                Inputs          Description
                                                -----------     -----------------------------------
                                                WKID            Integer. Ex: datum_transformation=4326
                                                -----------     -----------------------------------
                                                WKT             Dict. Ex: datum_transformation={"wkt": "<WKT>"}
                                                -----------     -----------------------------------
                                                Composite       Dict. Ex: datum_transformation=```{'geoTransforms':[{'wkid':<id>,'forward':<true|false>},{'wkt':'<WKT>','forward':<True|False>}]}```
                                                ===========     ===================================


        -------------------------------     --------------------------------------------------------------------
        kwargs                              Optional dict. Optional parameters that can be passed to the Query
                                            function.  This will allow users to pass additional parameters not
                                            explicitly implemented on the function. A complete list of functions
                                            available is documented on the Query REST API.
        ===============================     ====================================================================

        :return:
            A :class:`~arcgis.features.FeatureSet` containing the features matching the query unless another return type
            is specified, such as ``return_count_only``, ``return_extent_only``, or ``return_ids_only``.

        .. code-block:: python

            # Usage Example with only a "where" sql statement

            >>> feat_set = feature_layer.query(where = "OBJECTID= 1")
            >>> type(feat_set)
            <arcgis.Features.FeatureSet>
            >>> feat_set[0]
            <Feature 1>

        .. code-block:: python

            # Usage Example of an advanced query returning the object IDs instead of Features

            >>> id_set = feature_layer.query(where = "OBJECTID1",
                                               out_fields = ["FieldName1, FieldName2"],
                                               distance = 100,
                                               units = 'esriSRUnit_Meter',
                                               return_ids_only = True)

            >>> type(id_set)
            <Array>
            >>> id_set[0]
            <"Item_id1">

        .. code-block:: python

            # Usage Example of an advanced query returning the number of features in the query

            >>> search_count = feature_layer.query(where = "OBJECTID1",
                                               out_fields = ["FieldName1, FieldName2"],
                                               distance = 100,
                                               units = 'esriSRUnit_Meter',
                                               return_count_only = True)

            >>> type(search_count)
            <Integer>
            >>> search_count
            <149>

        .. code-block:: python

            # Usage Example with "out_statistics" parameter

            >>> stats = [{
                    'onStatisticField': "1",
                    'outStatisticFieldName': "total",
                    'statisticType': "count"
                }]
            >>> feature_layer.query(out_statistics=stats, as_df=True) # returns a DataFrame containting total count

        .. code-block:: python

            # Usage Example with "StatisticFilter" parameter

            >>> from arcgis._impl.common._filters import StatisticFilter
            >>> sf1 = StatisticFilter()
            >>> sf1.add(statisticType="count", onStatisticField="1", outStatisticFieldName="total")
            >>> sf1.filter # This is to print the filter content
            >>> feature_layer.query(statistic_filter=sf1, as_df=True) # returns a DataFrame containing total count


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
            # we cannot assume that because return_all_records is False it means we specified something else
            if return_count_only or return_extent_only or return_ids_only:
                # Remove to avoid missing when wanting counts only
                if "orderByFields" in params:
                    del params["orderByFields"]
            if as_df:
                return self._query_df(url, params)
            return self._query(url, params, raw=as_raw)

        params["returnCountOnly"] = True
        # need to make edits to out fields if more than one to avoid server error. Split and use only first
        out_fields = params["outFields"]
        params["outFields"] = params["outFields"].split(",")[0]
        if where == "1=1":
            if "objectIdField" in self.properties:
                params["where"] = f"{self.properties.objectIdField} > 0"
            record_count = self._query(url, params, raw=as_raw)
            params["where"] = "1=1"
        else:
            record_count = self._query(url, params, raw=as_raw)
        if "maxRecordCount" in self.properties:
            max_records = self.properties["maxRecordCount"]
        else:
            max_records = 1000
        # reassign to original
        params["outFields"] = out_fields
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
                "esriFieldTypeSmallInteger": pd.Int32Dtype(),
                "esriFieldTypeInteger": pd.Int32Dtype(),
                "esriFieldTypeSingle": pd.Float64Dtype(),
                "esriFieldTypeDouble": pd.Float64Dtype(),
                "esriFieldTypeFloat": pd.Float64Dtype(),
                "esriFieldTypeString": pd.StringDtype(),
                "esriFieldTypeDate": "datetime64[ns]",  # np.datetime64,
                "esriFieldTypeOID": pd.Int64Dtype(),
                "esriFieldTypeGeometry": object,
                "esriFieldTypeBlob": object,
                "esriFieldTypeRaster": object,
                "esriFieldTypeGUID": pd.StringDtype(),
                "esriFieldTypeGlobalID": pd.StringDtype(),
                "esriFieldTypeXML": object,
                "esriFieldTypeTimeOnly": object,
                "esriFieldTypeDateOnly": object,
                "esriFieldTypeTimestampOffset": object,
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
            if return_geometry == False:
                columns.pop("SHAPE", None)
            df = pd.DataFrame([], columns=columns.keys()).astype(columns, True)
            if out_fields != "*":
                df = df[out_fields.split(",")].copy()

            if "SHAPE" in df.columns:
                df["SHAPE"] = GeoArray([])
                df.spatial.set_geometry("SHAPE")
                df.spatial.renderer = self.renderer
                df.spatial._meta.source = self
            return df
        elif record_count <= max_records:
            if (
                supports_pagination
                and record_count > 0
                and return_distinct_values == False
            ):
                params["resultRecordCount"] = record_count
            if as_df:
                import pandas as pd

                df = self._query_df(url, params)
                dt_fields = [
                    fld["name"]
                    for fld in self.properties.fields
                    if fld["type"]
                    in [
                        "esriFieldTypeDate",
                        "esriFieldTypeDateOnly",
                        "esriFieldTypeTimestampOffset",
                    ]
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
                if fld["type"]
                in [
                    "esriFieldTypeDate",
                    "esriFieldTypeDateOnly",
                    "esriFieldTypeTimestampOffset",
                ]
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
    def validate_sql(self, sql: str, sql_type: str = "where"):
        """
        The ``validate_sql`` operation validates an ``SQL-92`` expression or WHERE
        clause.
        The ``validate_sql`` operation ensures that an ``SQL-92`` expression, such
        as one written by a user through a user interface, is correct
        before performing another operation that uses the expression.

        .. note::
            For example, ``validateSQL`` can be used to validate information that is
            subsequently passed in as part of the where parameter of the calculate operation.

        ``validate_sql`` also prevents SQL injection. In addition, all table
        and field names used in the SQL expression or WHERE clause are
        validated to ensure they are valid tables and fields.


        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        sql                                 Required String. The SQL expression of WHERE clause to validate.
                                            Example: "Population > 300000"
        -------------------------------     --------------------------------------------------------------------
        sql_type                            Optional String. Three SQL types are supported in validate_sql
                                                - ``where (default)`` - Represents the custom WHERE clause the user
                                                  can compose when querying a layer or using calculate.
                                                - ``expression`` - Represents an SQL-92 expression. Currently,
                                                  expression is used as a default value expression when adding a
                                                  new field or using the calculate API.
                                                - ``statement`` - Represents the full SQL-92 statement that can be
                                                  passed directly to the database. No current ArcGIS REST API
                                                  resource or operation supports using the full SQL-92 SELECT
                                                  statement directly. It has been added to the validateSQL for
                                                  completeness.
                                                  Values: `where | expression | statement`
        ===============================     ====================================================================

        :return:
            A JSON Dictionary indicating 'success' or 'error'
        """
        params = {"f": "json"}
        if not isinstance(sql, str):
            raise ValueError("sql must be a string")
        else:
            params["sql"] = sql
        if sql_type.lower() not in ["where", "expression", "statement"]:
            raise ValueError(
                "sql_type must have value of: where, expression or statement"
            )
        else:
            params["sqlType"] = sql_type
        sql_type = sql_type.lower()
        url = self._url + "/validateSQL"
        return self._con.post(
            path=url,
            postdata=params,
        )

    # ----------------------------------------------------------------------
    def query_related_records(
        self,
        object_ids: str,
        relationship_id: str,
        out_fields: str = "*",
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
        The ``query_related_records`` operation is performed on a :class:`~arcgis.features.FeatureLayer`
        resource. The result of this operation are feature sets grouped
        by source layer/table object IDs. Each feature set contains
        Feature objects including the values for the fields requested by
        the user. For related layers, if you request geometry
        information, the geometry of each feature is also returned in
        the feature set. For related tables, the feature set does not
        include geometries.

        .. note::
            See the :attr:`~arcgis.features.FeatureLayer.query` method for a similar function.

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

                                   Syntax: historic_moment=<Epoch time in milliseconds>
        ----------------------     --------------------------------------------------------------------
        return_true_curves         Optional boolean. Optional parameter that is false by default. When
                                   set to true, returns true curves in output geometries; otherwise,
                                   curves are converted to densified :class:`~arcgis.geometry.Polyline` or
                                   :class:`~arcgis.features.Polygon` objects.
        ======================     ====================================================================


        :return: Dictionary of the query results

        .. code-block:: python

            # Usage Example:

            # Query returning the related records for a feature with objectid value of 2,
            # returning the values in the 6 attribute fields defined in the `field_string`
            # variable:

            >>> field_string = "objectid,attribute,system_name,subsystem_name,class_name,water_regime_name"
            >>> rel_records = feat_lyr.query_related_records(object_ids = "2",
                                                             relationship_id = 0,
                                                             out_fields = field_string,
                                                             return_geometry=True)

            >>> list(rel_records.keys())
            ['fields', 'relatedRecordGroups']

            >>> rel_records["relatedRecordGroups"]
            [{'objectId': 2,
              'relatedRecords': [{'attributes': {'objectid': 686,
                 'attribute': 'L1UBHh',
                 'system_name': 'Lacustrine',
                 'subsystem_name': 'Limnetic',
                 'class_name': 'Unconsolidated Bottom',
                 'water_regime_name': 'Permanently Flooded'}}]}]
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

        return self._con.post(path=qrr_url, postdata=params)

    # ----------------------------------------------------------------------
    def get_html_popup(self, oid: Optional[str]):
        """
        The ``get_html_popup`` method provides details about the HTML pop-up
        authored by the :class:`~arcgis.gis.User` using ArcGIS Pro or ArcGIS Desktop.

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

            return self._con.get(path=pop_url, params=params)
        return ""

    # ----------------------------------------------------------------------
    def append(
        self,
        item_id: Optional[str] = None,
        upload_format: str = "featureCollection",
        source_table_name: Optional[str] = None,
        field_mappings: Optional[list[dict[str, str]]] = None,
        edits: Optional[dict] = None,
        source_info: Optional[dict] = None,
        upsert: bool = False,
        skip_updates: bool = False,
        use_globalids: bool = False,
        update_geometry: bool = True,
        append_fields: Optional[list[str]] = None,
        rollback: bool = False,
        skip_inserts: Optional[bool] = None,
        upsert_matching_field: Optional[str] = None,
        upload_id: Optional[str] = None,
        *,
        return_messages: Optional[bool] = None,
        future: bool = False,
    ):
        """
        The ``append`` method is used to update an existing hosted :class:`~arcgis.features.FeatureLayer` object.
        See the `Append (Feature Service/Layer) <https://developers.arcgis.com/rest/services-reference/append-feature-service-layer-.htm>`_
        page in the ArcGIS REST API documentation for more information.

        .. note::
            The ``append`` method is only available in ArcGIS Online and ArcGIS Enterprise 10.8.1+

        ========================   ====================================================================
        **Parameter**               **Description**
        ------------------------   --------------------------------------------------------------------
        item_id                    Optional string. The ID for the Portal item that contains the source
                                   file.
                                   Used in conjunction with editsUploadFormat.
        ------------------------   --------------------------------------------------------------------
        upload_format              Required string. The source append data format. The default is
                                   featureCollection.
                                   Values: 'sqlite' | 'shapefile' | 'filegdb' | 'featureCollection' |
                                   'geojson' | 'csv' | 'excel'
        ------------------------   --------------------------------------------------------------------
        source_table_name          Required string. Required even when the source data contains only
                                   one table, e.g., for file geodatabase.

                                   .. code-block:: python

                                       # Example usage:
                                       source_table_name=  "Building"
        ------------------------   --------------------------------------------------------------------
        field_mappings             Optional list. Used to map source data to a destination layer.
                                   Syntax: field_mappings=[{"name" : <"targetName">,
                                                           "sourceName" : < "sourceName">}, ...]
                                   .. code-block:: python

                                       # Example usage:
                                       field_mappings=[{"name" : "CountyID",
                                                       "sourceName" : "GEOID10"}]
        ------------------------   --------------------------------------------------------------------
        edits                      Optional dictionary. Only feature collection json is supported. Append
                                   supports all format through the upload_id or item_id.
        ------------------------   --------------------------------------------------------------------
        source_info                Optional dictionary. This is only needed when appending data from
                                   excel or csv. The appendSourceInfo can be the publishing parameter
                                   returned from analyze the csv or excel file.
        ------------------------   --------------------------------------------------------------------
        upsert                     Optional boolean. Optional parameter specifying whether the edits
                                   needs to be applied as updates if the feature already exists.
                                   Default is false.
        ------------------------   --------------------------------------------------------------------
        skip_updates               Optional boolean. Parameter is used only when upsert is true.
        ------------------------   --------------------------------------------------------------------
        use_globalids              Optional boolean. Specifying whether upsert needs to use GlobalId
                                   when matching features.
        ------------------------   --------------------------------------------------------------------
        update_geometry            Optional boolean. The parameter is used only when upsert is true.
                                   Skip updating the geometry and update only the attributes for
                                   existing features if they match source features by objectId or
                                   globalId.(as specified by useGlobalIds parameter).
        ------------------------   --------------------------------------------------------------------
        append_fields              Optional list. The list of destination fields to append to. This is
                                   supported when upsert=true or false.

                                   .. code-block:: python

                                       #Values:
                                       ["fieldName1", "fieldName2",....]
        ------------------------   --------------------------------------------------------------------
        rollback                   Optional boolean. Optional parameter specifying whether the upsert
                                   edits needs to be rolled back in case of failure. Default is false.
        ------------------------   --------------------------------------------------------------------
        skip_inserts               Used only when upsert is true. Used to skip inserts if the value is
                                   true. The default value is false.
        ------------------------   --------------------------------------------------------------------
        upsert_matching_field      Optional string. The layer field to be used when matching features
                                   with upsert. ObjectId, GlobalId, and any other field that has a
                                   unique index can be used with upsert.
                                   This parameter overrides use_globalids; e.g., specifying
                                   upsert_matching_field will be used even if you specify
                                   use_globalids = True.
                                   Example: upsert_matching_field="MyfieldWithUniqueIndex"
        ------------------------   --------------------------------------------------------------------
        upload_id                  Optional string. The itemID field from an
                                   :func:`~FeatureLayerCollection.upload` response, corresponding with
                                   the `appendUploadId` REST API argument. This argument should not be
                                   used along side the `item_id` argument.
        ------------------------   --------------------------------------------------------------------
        return_messages            Optional Boolean.  When set to `True`, the messages returned from
                                   the append will be returned. If `False`, the response messages will
                                   not be returned.  This alters the output to be a tuple consisting of
                                   a (Boolean, Dictionary).
        ------------------------   --------------------------------------------------------------------
        future                     Optional boolean. If True, a future object will be returned and the process
                                   will not wait for the task to complete. The default is False, which means wait for results.
        ========================   ====================================================================

        :return:
            A boolean indicating success (True), or failure (False). When ``return_messages`` is True, the
            response messages will be return in addition to the boolean as a `tuple`.
            If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

        .. code-block:: python

            # Usage Example

            >>> feature_layer.append(source_table_name= "Building",
                                    field_mappings=[{"name" : "CountyID",
                                                    "sourceName" : "GEOID10"}],
                                    upsert = True,
                                    append_fields = ["fieldName1", "fieldName2",...., fieldname22],
                                    return_messages = False)
            <True>




        """
        import copy

        if (
            hasattr(self._gis, "_portal") and self._gis._portal.is_logged_in == False
        ) or (hasattr(self._gis, "is_logged_in") and self._gis.is_logged_in == False):
            raise Exception("Authentication required to perform append.")
        if self.properties.supportsAppend == False:
            raise Exception(
                "Append is not supported on this layer, please "
                + "update service definition capabilities."
            )

        params = {
            "f": "json",
            "sourceTableName": source_table_name,
            "fieldMappings": field_mappings,
            "edits": edits,
            "appendSourceInfo": source_info,
            "upsert": upsert,
            "skipUpdates": skip_updates,
            "useGlobalIds": use_globalids,
            "updateGeometry": update_geometry,
            "appendFields": append_fields,
            "appendUploadId": upload_id,
            "appendItemId": item_id,
            "appendUploadFormat": upload_format,
            "rollbackOnFailure": rollback,
        }
        if (
            self._gis
            and hasattr(self._gis, "_con")
            and self._gis._con.token
            and hasattr(self._gis, "_portal")
            and self._gis._portal.is_arcgisonline == False
        ):
            params["token"] = self._gis._con.token
        if not upsert_matching_field is None:
            params["upsertMatchingField"] = upsert_matching_field
        if not skip_inserts is None:
            params["skipInserts"] = skip_inserts
        upload_formats = (
            """sqlite,shapefile,filegdb,featureCollection,geojson,csv,excel""".split(
                ","
            )
        )
        if upload_format not in upload_formats:
            raise ValueError("Invalid upload format: %s." % upload_format)
        cparams = copy.copy(params)
        for k, v in cparams.items():
            if v is None:
                params.pop(k)
            del k, v
        url = self._url + "/append"
        del cparams
        res = self._con.post(path=url, postdata=params)
        if future:
            executor = concurrent.futures.ThreadPoolExecutor(1)
            future = executor.submit(self._check_append_status, *(res, return_messages))
            executor.shutdown(False)
            return future
        return self._check_append_status(res, return_messages)

    # ----------------------------------------------------------------------
    def _check_append_status(self, res, return_messages):
        """checks the append status"""
        n = 1
        if "statusUrl" in res:
            time.sleep(1)
            surl = res["statusUrl"]
            sres = self._con.get(path=surl, params={"f": "json"})
            while sres["status"].lower() != "completed":
                sres = self._con.get(path=surl, params={"f": "json"})
                if sres["status"].lower() in "failed":
                    if return_messages:
                        return (False, sres)
                    return False
                if n >= 40:
                    n = 40
                time.sleep(0.5 * n)
                n += 1
            if return_messages:
                return (True, sres)
            else:
                return True
        return res

    # ----------------------------------------------------------------------
    def delete_features(
        self,
        deletes: Optional[str] = None,
        where: Optional[str] = None,
        geometry_filter: Optional[GeometryFilter] = None,
        gdb_version: Optional[str] = None,
        rollback_on_failure: bool = True,
        return_delete_results: bool = True,
        future: bool = False,
    ):
        """
        Deletes features in a :class:`~arcgis.features.FeatureLayer` or
        :class:`~arcgis.features.Table`

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        deletes                    Optional string. A comma separated string of OIDs to remove from the
                                   service.
        ----------------------     --------------------------------------------------------------------
        where                      Optional string.  A where clause for the query filter. Any legal SQL
                                   where clause operating on the fields in the layer is allowed.
                                   Features conforming to the specified where clause will be deleted.
        ----------------------     --------------------------------------------------------------------
        geometry_filter            Optional :class:`~arcgis.geometry.filters.SpatialFilter`. A spatial filter from
                                   arcgis.geometry.filters module to filter results by a spatial
                                   relationship with another geometry.
        ----------------------     --------------------------------------------------------------------
        gdb_version                Optional string. A ``Geodatabase`` version to apply the edits.
        ----------------------     --------------------------------------------------------------------
        rollback_on_failure        Optional boolean. Optional parameter to specify if the edits should
                                   be applied only if all submitted edits succeed. If false, the server
                                   will apply the edits that succeed even if some of the submitted
                                   edits fail. If true, the server will apply the edits only if all
                                   edits succeed. The default value is true.
        ----------------------     --------------------------------------------------------------------
        return_delete_results      Optional Boolean. Optional parameter that indicates whether a result
                                   is returned per deleted row when the deleteFeatures operation is run.
                                   The default is true.
        ----------------------     --------------------------------------------------------------------
        future                     Optional boolean. If True, a future object will be returned and the process
                                   will not wait for the task to complete. The default is False, which means wait for results.
        ======================     ====================================================================

        :return:
            A dictionary if future=False (default), else If ``future = True``,
            then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

        .. code-block:: python

            # Usage Example with only a "where" sql statement

            >>> from arcgis.features import FeatureLayer

            >>> gis = GIS("pro")
            >>> buck = gis.content.search("owner:"+ gis.users.me.username)
            >>> buck_1 =buck[1]
            >>> lay = buck_1.layers[0]

            >>> la_df = lay.delete_features(where = "OBJECTID > 15")
            >>> la_df
            {'deleteResults': [
            {'objectId': 1, 'uniqueId': 5, 'globalId': None, 'success': True},
            {'objectId': 2, 'uniqueId': 5, 'globalId': None, 'success': True},
            {'objectId': 3, 'uniqueId': 5, 'globalId': None, 'success': True},
            {'objectId': 4, 'uniqueId': 5, 'globalId': None, 'success': True},
            {'objectId': 5, 'uniqueId': 5, 'globalId': None, 'success': True},
            {'objectId': 6, 'uniqueId': 6, 'globalId': None, 'success': True},
            {'objectId': 7, 'uniqueId': 7, 'globalId': None, 'success': True},
            {'objectId': 8, 'uniqueId': 8, 'globalId': None, 'success': True},
            {'objectId': 9, 'uniqueId': 9, 'globalId': None, 'success': True},
            {'objectId': 10, 'uniqueId': 10, 'globalId': None, 'success': True},
            {'objectId': 11, 'uniqueId': 11, 'globalId': None, 'success': True},
            {'objectId': 12, 'uniqueId': 12, 'globalId': None, 'success': True},
            {'objectId': 13, 'uniqueId': 13, 'globalId': None, 'success': True},
            {'objectId': 14, 'uniqueId': 14, 'globalId': None, 'success': True},
            {'objectId': 15, 'uniqueId': 15, 'globalId': None, 'success': True}]}


        """
        delete_url = self._url + "/deleteFeatures"
        params = {
            "f": "json",
            "rollbackOnFailure": rollback_on_failure,
            "returnDeleteResults": return_delete_results,
        }
        if gdb_version is not None:
            params["gdbVersion"] = gdb_version

        if deletes is not None and isinstance(deletes, str):
            params["objectIds"] = deletes
        elif deletes is not None and isinstance(deletes, PropertyMap):
            print(
                "pass in delete, unable to convert PropertyMap to string list of OIDs"
            )

        elif deletes is not None and isinstance(deletes, FeatureSet):
            params["objectIds"] = ",".join(
                [
                    str(feat.get_value(field_name=deletes.object_id_field_name))
                    for feat in deletes.features
                ]
            )

        if where is not None:
            params["where"] = where

        if geometry_filter is not None and isinstance(geometry_filter, GeometryFilter):
            for key, val in geometry_filter.filter:
                params[key] = val
        elif geometry_filter is not None and isinstance(geometry_filter, dict):
            for key, val in geometry_filter.items():
                params[key] = val

        if (
            "objectIds" not in params
            and "where" not in params
            and "geometry" not in params
        ):
            print("Parameters not valid for delete_features")
            return None
        if future is False:
            return self._con.post(path=delete_url, postdata=params)
        else:
            params["async"] = True
            import concurrent.futures

            executor = concurrent.futures.ThreadPoolExecutor(1)
            res = self._con.post(path=delete_url, postdata=params)
            time.sleep(2)
            future = executor.submit(
                self._status_via_url,
                *(self._con, res["statusUrl"], {"f": "json"}, True),
            )
            executor.shutdown(False)

            return future

    @property
    def estimates(self) -> dict[str, Any]:
        """
        Returns up-to-date approximations of layer information, such as row count
        and extent. Layers that support this property will include
        `infoInEstimates` information in the layer's :attr:`~arcgis.features.FeatureLayer.properties`.

        Currently available with ArcGIS Online and Enterprise 10.9.1+

        :returns: Dict[str, Any]

        """
        if self._gis.version >= [9, 2] or self._gis._is_agol:
            if "infoInEstimates" in self.properties:
                url = self._url + "/getEstimates"
                params = {"f": "json"}
                return self._con.get(url, params)
        return {}

    # ----------------------------------------------------------------------
    def _status_via_url(self, con, url, params, ignore_error=False):
        """
        performs the asynchronous check to see if the operation finishes
        """
        status_allowed = [
            v.lower()
            for v in [
                "Executing",
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
        ]
        time.sleep(0.5)
        status = con.get(url, params, ignore_error_key=ignore_error)
        if not "status" in status and ignore_error:
            return status
        while (
            status["status"].lower() in status_allowed
            and status["status"].lower() != "completed"
        ):
            if status["status"].lower() == "completed":
                return status
            elif status["status"].lower() == "completedwitherrors":
                break
            elif "fail" in status["status"].lower():
                break
            elif "error" in status["status"].lower():
                break
            status = con.get(url, params, ignore_error_key=ignore_error)
        return status

    # ----------------------------------------------------------------------
    def edit_features(
        self,
        adds: Optional[list[FeatureSet]] = None,
        updates: Optional[list[FeatureSet]] = None,
        deletes: Optional[list[FeatureSet]] = None,
        gdb_version: Optional[str] = None,
        use_global_ids: bool = False,
        rollback_on_failure: bool = True,
        return_edit_moment: bool = False,
        attachments: Optional[dict[str, list[Any]]] = None,
        true_curve_client: bool = False,
        session_id: Optional[str] = None,
        use_previous_moment: bool = False,
        datum_transformation: Optional[Union[int, dict[str, Any]]] = None,
        future: bool = False,
    ):
        """
        Adds, updates, and deletes features to the
        associated :class:`~arcgis.features.FeatureLayer` or :class:`~arcgis.features.Table` in a single call.


        .. note::
            When making large number (250+ records at once) of edits,
            :attr:`~arcgis.features.FeatureLayer.append` should be used over ``edit_features`` to improve
            performance and ensure service stability.


        =====================   ======================================================================================
        **Inputs**              **Description**
        ---------------------   --------------------------------------------------------------------------------------
        adds                    Optional :class:`~arcgis.features.FeatureSet`/List. The array of features to be added.
        ---------------------   --------------------------------------------------------------------------------------
        updates                 Optional :class:`~arcgis.features.FeatureSet`/List. The array of features to be updated.
        ---------------------   --------------------------------------------------------------------------------------
        deletes                 Optional :class:`~arcgis.features.FeatureSet`/List. string of OIDs to remove from service
        ---------------------   --------------------------------------------------------------------------------------
        use_global_ids          Optional boolean. Instead of referencing the default Object ID field, the service
                                will look at a GUID field to track changes. This means the GUIDs will be passed
                                instead of OIDs for delete, update or add features.
        ---------------------   --------------------------------------------------------------------------------------
        gdb_version             Optional boolean. `Geodatabase` version to apply the edits.
        ---------------------   --------------------------------------------------------------------------------------
        rollback_on_failure     Optional boolean. Optional parameter to specify if the edits should be applied only
                                if all submitted edits succeed. If false, the server will apply the edits that succeed
                                even if some of the submitted edits fail. If true, the server will apply the edits
                                only if all edits succeed. The default value is true.
        ---------------------   --------------------------------------------------------------------------------------
        return_edit_moment      Optional boolean. Introduced at 10.5, only applicable with ArcGIS Server services
                                only. Specifies whether the response will report the time edits were applied. If set
                                to true, the server will return the time in the response's editMoment key. The default
                                value is false.
        ---------------------   --------------------------------------------------------------------------------------
        attachments             Optional Dict. This parameter adds, updates, or deletes attachments. It applies only
                                when the `use_global_ids` parameter is set to true. For adds, the globalIds of the
                                attachments provided by the client are preserved. When useGlobalIds is true, updates
                                and deletes are identified by each feature or attachment globalId, rather than their
                                objectId or attachmentId. This parameter requires the layer's
                                supportsApplyEditsWithGlobalIds property to be true.

                                Attachments to be added or updated can use either pre-uploaded data or base 64
                                encoded data.

                                **Inputs**

                                    ========     ================================
                                    Inputs       Description
                                    --------     --------------------------------
                                    adds         List of attachments to add.
                                    --------     --------------------------------
                                    updates      List of attachements to update
                                    --------     --------------------------------
                                    deletes      List of attachments to delete
                                    ========     ================================

                                See the `Apply Edits to a Feature Service layer <https://developers.arcgis.com/rest/services-reference/apply-edits-feature-service-layer-.htm>`_
                                in the ArcGIS REST API for more information.
        ---------------------   --------------------------------------------------------------------------------------
        true_curve_client       Optional boolean. Introduced at 10.5. Indicates to the server whether the client is
                                true curve capable. When set to true, this indicates to the server that true curve
                                geometries should be downloaded and that geometries containing true curves should be
                                consumed by the map service without densifying it. When set to false, this indicates
                                to the server that the client is not true curves capable. The default value is false.
        ---------------------   --------------------------------------------------------------------------------------
        session_id              Optional String. Introduced at 10.6. The `session_id` is a GUID value that clients
                                establish at the beginning and use throughout the edit session. The sessonID ensures
                                isolation during the edit session. The `session_id` parameter is set by a client
                                during long transaction editing on a branch version.
        ---------------------   --------------------------------------------------------------------------------------
        use_previous_moment     Optional Boolean. Introduced at 10.6. The `use_previous_moment` parameter is used to
                                apply the edits with the same edit moment as the previous set of edits. This allows an
                                editor to apply single block of edits partially, complete another task and then
                                complete the block of edits. This parameter is set by a client during long transaction
                                editing on a branch version.

                                When set to true, the edits are applied with the same edit moment as the previous set
                                of edits. When set to false or not set (default) the edits are applied with a new
                                edit moment.

        ---------------------   --------------------------------------------------------------------------------------
        datum_transformation    Optional Integer/Dictionary.  This parameter applies a datum transformation while
                                projecting geometries in the results when out_sr is different than the layer's spatial
                                reference. When specifying transformations, you need to think about which datum
                                transformation best projects the layer (not the feature service) to the `outSR` and
                                `sourceSpatialReference` property in the layer properties. For a list of valid datum
                                transformation ID values ad well-known text strings, see `Using spatial references <https://developers.arcgis.com/rest/services-reference/enterprise/using-spatial-references.htm>`_.
                                For more information on datum transformations please see the transformation
                                parameter in the `Project operation <https://developers.arcgis.com/rest/services-reference/project.htm>`_ documentation.

                                **Examples**

                                    ===========     ===================================
                                    Inputs          Description
                                    -----------     -----------------------------------
                                    WKID            Integer. Ex: datum_transformation=4326
                                    -----------     -----------------------------------
                                    WKT             Dict. Ex: datum_transformation={"wkt": "<WKT>"}
                                    -----------     -----------------------------------
                                    Composite       Dict. Ex: datum_transformation=```{'geoTransforms':[{'wkid':<id>,'forward':<true|false>},{'wkt':'<WKT>','forward':<True|False>}]}```
                                    ===========     ===================================

        ---------------------   --------------------------------------------------------------------------------------
        future                  Optional Boolean.  If the `FeatureLayer` has `supportsAsyncApplyEdits` set
                                to `True`, then edits can be applied asynchronously. If True, a future object will be returned and the process
                                will not wait for the task to complete. The default is False, which means wait for results.
        =====================   ======================================================================================

        :return:
            A dictionary by default, or If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

        .. code-block:: python

            # Usage Example 1:

            feature = [
            {
                'attributes': {
                    'ObjectId': 1,
                    'UpdateDate': datetime.datetime.now(),
                }
            }]
            lyr.edit_features(updates=feature)

        .. code-block:: python

            # Usage Example 2:

            adds = {"geometry": {"x": 500, "y": 500, "spatialReference":
                                {"wkid": 102100, "latestWkid": 3857}},
                    "attributes": {"ADMIN_NAME": "Fake Location"}
                    }
            lyr.edit_features(adds=[adds])

        .. code-block:: python

            # Usage Example 3:

            lyr.edit_features(deletes=[2542])

        """
        try:
            import pandas as pd
            from arcgis.features.geo import _is_geoenabled

            HAS_PANDAS = True
        except:
            HAS_PANDAS = False
        if (
            future
            and "advancedEditingCapabilities" in self.properties
            and "supportsAsyncApplyEdits"
            in self.properties["advancedEditingCapabilities"]
        ):
            future = self.properties["advancedEditingCapabilities"][
                "supportsAsyncApplyEdits"
            ]
        else:
            future = False
        if adds is None:
            adds = []
        if updates is None:
            updates = []
        edit_url = self._url + "/applyEdits"
        params = {
            "f": "json",
            "useGlobalIds": use_global_ids,
            "rollbackOnFailure": rollback_on_failure,
        }
        if gdb_version is not None:
            params["gdbVersion"] = gdb_version
        if HAS_PANDAS and isinstance(adds, pd.DataFrame) and _is_geoenabled(adds):
            cols = [
                c for c in adds.columns.tolist() if c.lower() not in ["objectid", "fid"]
            ]
            params["adds"] = json.dumps(
                adds[cols].spatial.__feature_set__["features"],
                default=_date_handler,
            )
        elif (
            HAS_PANDAS
            and isinstance(adds, pd.DataFrame)
            and _is_geoenabled(adds) == False
        ):
            # we have a regular panadas dataframe
            cols = [
                c for c in adds.columns.tolist() if c.lower() not in ["objectid", "fid"]
            ]
            params["adds"] = json.dumps(
                [{"attributes": row} for row in adds[cols].to_dict("records")],
                default=_date_handler,
            )
        elif isinstance(adds, FeatureSet):
            params["adds"] = json.dumps(
                [f.as_dict for f in adds.features], default=_date_handler
            )

        elif len(adds) > 0:
            if isinstance(adds[0], dict):
                params["adds"] = json.dumps([f for f in adds], default=_date_handler)
            elif isinstance(adds[0], PropertyMap):
                params["adds"] = json.dumps(
                    [dict(f) for f in adds], default=_date_handler
                )
            elif isinstance(adds[0], Feature):

                def _handle_feature(f):
                    d = f.as_dict
                    if f.attributes is None:
                        d["attributes"] = {}
                    return d

                params["adds"] = json.dumps(
                    [_handle_feature(f) for f in adds], default=_date_handler
                )
            else:
                print("pass in features as list of Features, dicts or PropertyMap")
        if isinstance(updates, FeatureSet):
            params["updates"] = json.dumps(
                [f.as_dict for f in updates.features], default=_date_handler
            )
        elif (
            HAS_PANDAS and isinstance(updates, pd.DataFrame) and _is_geoenabled(updates)
        ):
            params["updates"] = json.dumps(
                updates.spatial.__feature_set__["features"],
                default=_date_handler,
            )
        elif (
            HAS_PANDAS
            and isinstance(updates, pd.DataFrame)
            and _is_geoenabled(updates) == False
        ):
            # we have a regular panadas dataframe
            cols = [
                c
                for c in updates.columns.tolist()
                if c.lower() not in ["objectid", "fid"]
            ]
            params["updates"] = json.dumps(
                [{"attributes": row} for row in updates[cols].to_dict("records")],
                default=_date_handler,
            )
        elif len(updates) > 0:
            if isinstance(updates[0], dict):
                params["updates"] = json.dumps(
                    [f for f in updates], default=_date_handler
                )
            elif isinstance(updates[0], PropertyMap):
                params["updates"] = json.dumps(
                    [dict(f) for f in updates], default=_date_handler
                )
            elif isinstance(updates[0], Feature):
                params["updates"] = json.dumps(
                    [f.as_dict for f in updates], default=_date_handler
                )
            else:
                print("pass in features as list of Features, dicts or PropertyMap")
        if deletes is not None and isinstance(deletes, str):
            params["deletes"] = deletes
        elif deletes is not None and isinstance(deletes, PropertyMap):
            print(
                "pass in delete, unable to convert PropertyMap to string list of OIDs"
            )
        elif deletes is not None and isinstance(deletes, pd.DataFrame):
            cols = [
                c for c in deletes.columns.tolist() if c.lower() in ["objectid", "fid"]
            ]
            if len(cols) > 0:
                params["deletes"] = ",".join([str(d) for d in deletes[cols[0]]])
            else:
                raise Exception("Could not find ObjectId or FID field.")
        elif deletes is not None and isinstance(deletes, FeatureSet):
            field_name = None
            if deletes.object_id_field_name:
                field_name = deletes.object_id_field_name
            elif self.properties.objectIdField in deletes.fields:
                field_name = self.properties.objectIdField
            else:
                print("deletes FeatureSet must have object_id_field_name parameter set")

            if field_name:
                params["deletes"] = ",".join(
                    [
                        str(feat.get_value(field_name=field_name))
                        for feat in deletes.features
                    ]
                )
        elif isinstance(deletes, (list, tuple)):
            params["deletes"] = ",".join([str(d) for d in deletes])
        if not return_edit_moment is None:
            params["returnEditMoment"] = return_edit_moment
        if not attachments is None and isinstance(attachments, dict):
            params["attachments"] = attachments
        if not true_curve_client is None:
            params["trueCurveClient"] = true_curve_client
        if not use_previous_moment is None:
            params["usePreviousEditMoment"] = use_previous_moment
        if not datum_transformation is None:
            params["datumTransformation"] = datum_transformation
        if session_id and isinstance(session_id, str):
            params["sessionID"] = session_id
        if (
            "deletes" not in params
            and "updates" not in params
            and "adds" not in params
            and "attachments" not in params
        ):
            print("Parameters not valid for edit_features")
            return None
        try:
            if future:
                params["async"] = True
                executor = concurrent.futures.ThreadPoolExecutor(1)
                res = self._con.post_multipart(path=edit_url, postdata=params)
                future = executor.submit(
                    self._status_via_url,
                    *(self._con, res["statusUrl"], {"f": "json"}, True),
                )
                executor.shutdown(False)

                return EditFeatureJob(future, self._con)
                # return future
            return self._con.post_multipart(path=edit_url, postdata=params)
        except Exception as e:
            if str(e).lower().find("Invalid Token".lower()) > -1:
                params.pop("token", None)
                return self._con.post_multipart(path=edit_url, postdata=params)
            else:
                raise

    # ----------------------------------------------------------------------
    def calculate(
        self,
        where: str,
        calc_expression: list[dict[str, Any]],
        sql_format: str = "standard",
        version: Optional[str] = None,
        sessionid: Optional[str] = None,
        return_edit_moment: Optional[bool] = None,
        future: bool = False,
    ):
        """
        The ``calculate`` operation is performed on a :class:`~arcgis.features.FeatureLayer`
        resource. ``calculate`` updates the values of one or more fields in an
        existing feature service layer based on SQL expressions or scalar
        values. The ``calculate`` operation can only be used if the
        ``supportsCalculate`` property of the layer is `True`.
        Neither the Shape field nor system fields can be updated using
        ``calculate``. System fields include ``ObjectId`` and ``GlobalId``.

        =====================   ====================================================
        **Inputs**              **Description**
        ---------------------   ----------------------------------------------------
        where                   Required String. A where clause can be used to limit
                                the updated records. Any legal SQL where clause
                                operating on the fields in the layer is allowed.
        ---------------------   ----------------------------------------------------
        calc_expression         Required List. The array of field/value info objects
                                that contain the field or fields to update and their
                                scalar values or SQL expression.  Allowed types are
                                dictionary and list.  List must be a list of
                                dictionary objects.

                                Calculation Format is as follows:

                                    `{"field" : "<field name>",  "value" : "<value>"}`

        ---------------------   ----------------------------------------------------
        sql_format              Optional String. The SQL format for the
                                calc_expression. It can be either standard SQL92
                                (standard) or native SQL (native). The default is
                                standard.

                                Values: `standard`, `native`
        ---------------------   ----------------------------------------------------
        version                 Optional String. The geodatabase version to apply
                                the edits.
        ---------------------   ----------------------------------------------------
        sessionid               Optional String. A parameter which is set by a
                                client during long transaction editing on a branch
                                version. The sessionid is a GUID value that clients
                                establish at the beginning and use throughout the
                                edit session.
                                The sessonid ensures isolation during the edit
                                session. This parameter applies only if the
                                `isDataBranchVersioned` property of the layer is
                                true.
        ---------------------   ----------------------------------------------------
        return_edit_moment      Optional Boolean. This parameter specifies whether
                                the response will report the time edits were
                                applied. If true, the server will return the time
                                edits were applied in the response's edit moment
                                key. This parameter applies only if the
                                `isDataBranchVersioned` property of the layer is
                                true.
        ---------------------   ----------------------------------------------------
        future                  Optional boolean. If True, a future object will be
                                returned and the process
                                will not wait for the task to complete. The default is
                                False, which means wait for results.

                                **This applies to 10.8+ only**

        =====================   ====================================================

        :return:
            A dictionary with the following format:
             {
             'updatedFeatureCount': 1,
             'success': True
             }

            If ``future = True``, then the result is a :class:`~concurrent.futures.Future` object. Call ``result()`` to get the response.

        .. code-block:: python

            # Usage Example 1:

            print(fl.calculate(where="OBJECTID < 2",
                               calc_expression={"field": "ZONE", "value" : "R1"}))

        .. code-block:: python

            # Usage Example 2:

            print(fl.calculate(where="OBJECTID < 2001",
                               calc_expression={"field": "A",  "sqlExpression" : "B*3"}))


        """
        url = self._url + "/calculate"
        params = {
            "f": "json",
            "where": where,
        }
        if isinstance(calc_expression, dict):
            params["calcExpression"] = json.dumps(
                [calc_expression], default=_date_handler
            )
        elif isinstance(calc_expression, list):
            params["calcExpression"] = json.dumps(
                calc_expression, default=_date_handler
            )
        if sql_format.lower() in ["native", "standard"]:
            params["sqlFormat"] = sql_format.lower()
        else:
            params["sqlFormat"] = "standard"
        if version:
            params["gdbVersion"] = version
        if sessionid:
            params["sessionID"] = sessionid
        if isinstance(return_edit_moment, bool):
            params["returnEditMoment"] = return_edit_moment
        if (
            "supportsASyncCalculate" in self.properties
            and self.properties.supportsASyncCalculate
            and future
        ):
            params["async"] = True
            executor = concurrent.futures.ThreadPoolExecutor(1)
            res = self._con.post(
                path=url,
                postdata=params,
            )
            future = executor.submit(
                self._status_via_url,
                *(self._con, res["statusUrl"], {"f": "json"}),
            )
            executor.shutdown(False)
            return future
        return self._con.post(
            path=url,
            postdata=params,
        )

    # ----------------------------------------------------------------------
    def _query(self, url, params, raw=False, **kwargs):
        """returns results of query"""
        try:
            result = self._con.post(
                path=url,
                postdata=params,
            )
        except Exception as queryException:
            error_list = [
                "Error performing query operation",
                "HTTP Error 504: GATEWAY_TIMEOUT",
            ]
            if queryException.args[0].lower().find("invalid token") > -1:
                params.pop("token", None)
                return self._query(url, params, raw=False)
            elif any(ele in queryException.__str__() for ele in error_list):
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
    def _query_df(self, url, params, **kwargs):
        """returns results of a query as a pd.DataFrame"""
        import pandas as pd
        import numpy as np

        if [float(i) for i in pd.__version__.split(".")] < [1, 0, 0]:
            _fld_lu = {
                "esriFieldTypeSmallInteger": np.int32,
                "esriFieldTypeInteger": np.int32,
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
                "esriFieldTypeTimeOnly": pd.datetime,
                "esriFieldTypeDateOnly": pd.datetime,
                "esriFieldTypeTimestampOffset": pd.datetime,
            }
        else:
            from datetime import datetime as _datetime

            _fld_lu = {
                "esriFieldTypeSmallInteger": pd.Int32Dtype(),
                "esriFieldTypeInteger": pd.Int32Dtype(),
                "esriFieldTypeSingle": pd.Float64Dtype(),
                "esriFieldTypeDouble": pd.Float64Dtype(),
                "esriFieldTypeFloat": pd.Float64Dtype(),
                "esriFieldTypeString": pd.StringDtype(),
                "esriFieldTypeDate": object,
                "esriFieldTypeOID": pd.Int64Dtype(),
                "esriFieldTypeGeometry": object,
                "esriFieldTypeBlob": object,
                "esriFieldTypeRaster": object,
                "esriFieldTypeGUID": pd.StringDtype(),
                "esriFieldTypeGlobalID": pd.StringDtype(),
                "esriFieldTypeXML": object,
                "esriFieldTypeTimeOnly": pd.StringDtype(),
                "esriFieldTypeDateOnly": object,
                "esriFieldTypeTimestampOffset": object,
                "esriFieldTypeBigInteger": pd.Int64Dtype(),
            }

        def feature_to_row(feature, sr):
            """:return: a feature from a dict"""
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
            featureset_dict = self._con.post(url, params)
        except Exception as queryException:
            error_list = [
                "Error performing query operation",
                "HTTP Error 504: GATEWAY_TIMEOUT",
            ]
            if queryException.args[0].lower().find("invalid token") > -1:
                params.pop("token", None)
                return self._query_df(url, params, raw=False)
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
        if "SHAPE" in df.columns:
            df.loc[df.SHAPE.isna(), "SHAPE"] = None
        if "fields" in featureset_dict:
            dtypes = {}
            names = []
            fields = featureset_dict["fields"]
            for fld in fields:
                if fld["type"] != "esriFieldTypeGeometry":
                    dtypes[fld["name"]] = _fld_lu[fld["type"]]
                    names.append(fld["name"])
                if fld["type"] in [
                    "esriFieldTypeDate",
                    #
                    "esriFieldTypeDateOnly",
                    "esriFieldTypeTimestampOffset",
                ]:
                    dfields.append(fld["name"])
        if dtypes:
            df = df.astype(dtypes)

        if "SHAPE" in featureset_dict:
            df.spatial.set_geometry("SHAPE")
        if len(dfields) > 0:
            for fld in [fld for fld in dfields if fld in df.columns]:
                try:
                    df[fld] = pd.to_datetime(
                        df[fld] / 1000,
                        errors="coerce",
                        unit="s",
                    )
                except:
                    df[fld] = pd.to_datetime(
                        df[fld],
                        errors="coerce",
                    )
        return df


class Table(FeatureLayer):
    """
    ``Table`` objects represent entity classes with uniform properties. In addition to working with
    "entities with location" as :class:`~arcgis.features.Feature` objects, the :class:`~arcgis.gis.GIS` can also work
    with non-spatial entities as rows in tables.

    .. note::
        Working with tables is similar to working with :class:`~arcgis.features.FeatureLayer`objects, except that the
        rows (Features) in a table do not have a geometry, and tables ignore any geometry related operation.
    """

    @classmethod
    def fromitem(cls, item: Item, table_id: int = 0):
        """
        The ``fromitem`` method creates a :class:`~arcgis.features.Table` from a :class:`~arcgis.gis.Item` object.
        The table_id is the id of the table in :class:`~arcgis.features.FeatureLayerCollection` (feature service).

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        item                                Required :class:`~arcgis.gis.Item` object. The type of item should be a
                                            ``Feature Service`` that represents a
                                            :class:`~arcgis.features.FeatureLayerCollection`
        -------------------------------     --------------------------------------------------------------------
        table_id                            Required Integer. The id of the layer in feature layer collection
                                            (feature service).
                                            The default for ``table`` is 0.
        ===============================     ====================================================================

        :return:
            A :class:`~arcgis.features.Table` object
        """
        return item.tables[table_id]

    def query(
        self,
        where: str = "1=1",
        out_fields: Union[str, list[str]] = "*",
        time_filter: list[datetime] = None,
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
        out_statistics: Optional[list[dict[str, Any]]] = None,
        return_all_records: bool = True,
        historic_moment: Optional[Union[int, datetime]] = None,
        sql_format: Optional[str] = None,
        return_exceeded_limit_features: Optional[bool] = None,
        as_df: bool = False,
        having: Optional[str] = None,
        **kwargs,
    ):
        """
        The ``query`` method queries a :class:`~arcgis.features.Table` Layer based on a set of criteria.

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
                                            Syntax: time_filter=[<startTime>, <endTime>] ; specified as
                                                    datetime.date, datetime.datetime or timestamp in
                                                    milliseconds
        -------------------------------     --------------------------------------------------------------------
        gdb_version                         Optional string. The geodatabase version to query. This parameter
                                            applies only if the isDataVersioned property of the layer is true.
                                            If this is not specified, the query will apply to the published
                                            map's version.
        -------------------------------     --------------------------------------------------------------------
        return_geometry                     Optional boolean. If true, geometry is returned with the query.
                                            Default is true.
        -------------------------------     --------------------------------------------------------------------
        return_distinct_values              Optional boolean.  If true, it returns distinct values based on the
                                            fields specified in out_fields. This parameter applies only if the
                                            supportsAdvancedQueries property of the layer is true.
        -------------------------------     --------------------------------------------------------------------
        return_ids_only                     Optional boolean. Default is False.  If true, the response only
                                            includes an array of object IDs. Otherwise, the response is a
                                            feature set.
        -------------------------------     --------------------------------------------------------------------
        return_count_only                   Optional boolean. If true, the response only includes the count
                                            (number of features/records) that would be returned by a query.
                                            Otherwise, the response is a feature set. The default is false. This
                                            option supersedes the returnIdsOnly parameter. If
                                            returnCountOnly = true, the response will return both the count and
                                            the extent.
        -------------------------------     --------------------------------------------------------------------
        order_by_fields                     Optional string. One or more field names on which the
                                            features/records need to be ordered. Use ASC or DESC for ascending
                                            or descending, respectively, following every field to control the
                                            ordering.
                                            example: STATE_NAME ASC, RACE DESC, GENDER
        -------------------------------     --------------------------------------------------------------------
        group_by_fields_for_statistics      Optional string. One or more field names on which the values need to
                                            be grouped for calculating the statistics.
                                            example: STATE_NAME, GENDER
        -------------------------------     --------------------------------------------------------------------
        out_statistics                      Optional string. The definitions for one or more field-based
                                            statistics to be calculated.

                                            Syntax:

                                            [
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
        result_offset                       Optional integer. This option can be used for fetching query results
                                            by skipping the specified number of records and starting from the
                                            next record (that is, resultOffset + 1th). This option is ignored
                                            if return_all_records is True (i.e. by default).
        -------------------------------     --------------------------------------------------------------------
        result_record_count                 Optional integer. This option can be used for fetching query results
                                            up to the result_record_count specified. When result_offset is
                                            specified but this parameter is not, the map service defaults it to
                                            max_record_count. The maximum value for this parameter is the value
                                            of the layer's max_record_count property. This option is ignored if
                                            return_all_records is True (i.e. by default).
        -------------------------------     --------------------------------------------------------------------
        return_all_records                  Optional boolean. When True, the query operation will call the
                                            service until all records that satisfy the where_clause are
                                            returned. Note: result_offset and result_record_count will be
                                            ignored if return_all_records is True. Also, if return_count_only,
                                            return_ids_only, or return_extent_only are True, this parameter
                                            will be ignored.
        -------------------------------     --------------------------------------------------------------------
        historic_moment                     Optional integer. The historic moment to query. This parameter
                                            applies only if the layer is archiving enabled and the
                                            supportsQueryWithHistoricMoment property is set to true. This
                                            property is provided in the layer resource.

                                            If historic_moment is not specified, the query will apply to the
                                            current features.
        -------------------------------     --------------------------------------------------------------------
        sql_format                          Optional string.  The sql_format parameter can be either standard
                                            SQL92 standard or it can use the native SQL of the underlying
                                            datastore native. The default is none which means the sql_format
                                            depends on useStandardizedQuery parameter.
                                            Values: none | standard | native
        -------------------------------     --------------------------------------------------------------------
        return_exceeded_limit_features      Optional boolean. Optional parameter which is true by default. When
                                            set to true, features are returned even when the results include
                                            'exceededTransferLimit': True.

                                            When set to false and querying with resultType = tile features are
                                            not returned when the results include 'exceededTransferLimit': True.
                                            This allows a client to find the resolution in which the transfer
                                            limit is no longer exceeded without making multiple calls.
        -------------------------------     --------------------------------------------------------------------
        as_df                               Optional boolean.  If True, the results are returned as a DataFrame
                                            instead of a FeatureSet.
        -------------------------------     --------------------------------------------------------------------
        kwargs                              Optional dict. Optional parameters that can be passed to the Query
                                            function.  This will allow users to pass additional parameters not
                                            explicitly implemented on the function. A complete list of functions
                                            available is documented on the Query REST API.
        ===============================     ====================================================================

        :return:
            A :class:`~arcgis.features.FeatureSet` object or, if ```as_df=True```, a Panda's DataFrame
            containing the features matching the query unless another return type
            is specified, such as ``return_count_only``

        .. code-block:: python

            # Usage Example with only a "where" sql statement

            >>> feat_set = feature_layer.query(where = "OBJECTID1")
            >>> type(feat_set)
            <arcgis.Features.FeatureSet>
            >>> feat_set[0]
            <Feature 1>

        .. code-block:: python

            # Usage Example of an advanced query returning the object IDs instead of Features

            >>> id_set = feature_layer.query(where = "OBJECTID1",
                                               out_fields = ["FieldName1, FieldName2"],
                                               distance = 100,
                                               units = 'esriSRUnit_Meter',
                                               return_ids_only = True)

            >>> type(id_set)
            <Array>
            >>> id_set[0]
            <"Item_id1">

        .. code-block:: python

            # Usage Example of an advanced query returning the number of features in the query

            >>> search_count = feature_layer.query(where = "OBJECTID1",
                                               out_fields = ["FieldName1, FieldName2"],
                                               distance = 100,
                                               units = 'esriSRUnit_Meter',
                                               return_count_only = True)

            >>> type(search_count)
            <Integer>
            >>> search_count
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
        if where == "1=1":
            if "objectIdField" in self.properties:
                params["where"] = f"{self.properties.objectIdField} > 0"
            else:
                fields = [
                    field["name"]
                    for field in self.properties.fields
                    if field["type"] == "esriFieldTypeOID"
                ]
                params["where"] = f"{fields[0]} > 0"
            record_count = self._query(url, params, raw=as_raw)
            params["where"] = "1=1"
        else:
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
                "esriFieldTypeSmallInteger": pd.Int32Dtype(),
                "esriFieldTypeInteger": pd.Int32Dtype(),
                "esriFieldTypeSingle": pd.Float64Dtype(),
                "esriFieldTypeDouble": pd.Float64Dtype(),
                "esriFieldTypeFloat": pd.Float64Dtype(),
                "esriFieldTypeString": pd.StringDtype(),
                "esriFieldTypeDate": np.datetime64,
                "esriFieldTypeOID": pd.Int64Dtype(),
                "esriFieldTypeGeometry": object,
                "esriFieldTypeBlob": object,
                "esriFieldTypeRaster": object,
                "esriFieldTypeGUID": pd.StringDtype(),
                "esriFieldTypeGlobalID": pd.StringDtype(),
                "esriFieldTypeXML": object,
                "esriFieldTypeTimeOnly": object,
                "esriFieldTypeDateOnly": object,
                "esriFieldTypeTimestampOffset": object,
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
            if (
                supports_pagination
                and record_count > 0
                and return_distinct_values == False
            ):
                params["resultRecordCount"] = record_count
            if as_df:
                import pandas as pd

                df = self._query_df(url, params)
                dt_fields = [
                    fld["name"]
                    for fld in self.properties.fields
                    if fld["type"]
                    in [
                        "esriFieldTypeDate",
                        "esriFieldTypeDateOnly",
                        "esriFieldTypeTimestampOffset",
                    ]
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
                if fld["type"]
                in [
                    "esriFieldTypeDate",
                    "esriFieldTypeDateOnly",
                    "esriFieldTypeTimestampOffset",
                ]
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
                    df[fld] = pd.to_datetime(
                        df[fld],
                    )
            return df
        return result


class FeatureLayerCollection(_GISResource):
    """
    A ``FeatureLayerCollection`` is a collection of :class:`~arcgis.features.FeatureLayer` and
    :class:`~arcgis.features.Table`, with the associated relationships among the entities.

    In a web GIS, a feature layer collection is exposed as a feature service with multiple feature layers.

    Instances of ``FeatureLayerCollection`` can be obtained from feature service Items in the GIS using
    :attr:`~arcgis.features.FeatureLayerCollection.fromitem`, from feature service endpoints using the constructor,
    or by accessing the ``dataset`` attribute of :class:`~arcgis.features.FeatureLayer` objects.

    ``FeatureLayerCollection``s can be configured and managed using their `manager` helper object.

    If the dataset supports the sync operation, the `replicas` helper object allows management and synchronization of
    replicas for disconnected editing of the feature layer collection.

    .. note::
        You can use the ``layers`` and ``tables`` property to get to the individual layers and tables in this
        feature layer collection.
    """

    _vermgr = None

    def __init__(self, url, gis=None):
        super(FeatureLayerCollection, self).__init__(url, gis)

        try:
            if self.properties.syncEnabled:
                self.replicas = SyncManager(self)
        except AttributeError:
            pass

        self._populate_layers()
        self._admin = None
        try:
            from arcgis.gis.server._service._adminfactory import (
                AdminServiceGen,
            )

            self.service = AdminServiceGen(service=self, gis=gis)
        except:
            pass

    def _populate_layers(self):
        """
        populates the layers and tables for this feature service
        """
        layers = []
        tables = []

        for lyr in self.properties.layers:
            lyr = FeatureLayer(self.url + "/" + str(lyr.id), self._gis, self)
            layers.append(lyr)

        for lyr in self.properties.tables:
            lyr = Table(self.url + "/" + str(lyr.id), self._gis, self)
            tables.append(lyr)

        self.layers = layers
        self.tables = tables

    @property
    def manager(self):
        """
        A helper object to manage the :class:`~arcgis.features.FeatureLayerCollection`,
        for example updating its definition.

        :return:
            A :class:`~arcgis.features.FeatureLayerCollectionManager` object
        """
        if self._admin is None:
            url = self._url
            res = search("/rest/", url).span()
            add_text = "admin/"
            part1 = url[: res[1]]
            part2 = url[res[1] :]
            admin_url = "%s%s%s" % (part1, add_text, part2)

            self._admin = FeatureLayerCollectionManager(admin_url, self._gis, self)
        return self._admin

    @property
    def relationships(self):
        """
        Gets relationship information for
        the layers and tables in the :class:`~arcgis.features.FeatureLayerCollection` object.

        The relationships resource includes information about relationship
        rules from the back-end relationship classes, in addition to the
        relationship information already found in the individual :class:`~arcgis.features.FeatureLayer` and
        :class:`~arcgis.features.Table`.

        Feature layer collections that support the relationships resource
        will have the "supportsRelationshipsResource": true property on
        their properties.

        :return: List of Dictionaries

        """
        if (
            "supportsRelationshipsResource" in self.properties
            and self.properties["supportsRelationshipsResource"]
        ):
            url = self._url + "/relationships"
            params = {"f": "json"}
            res = self._con.get(url, params)
            if "relationships" in res:
                return res["relationships"]
            return res
        return []

    @property
    def versions(self):
        """
        Creates a ``VersionManager`` to create, update and use versions on a
        :class:`~arcgis.features.FeatureLayerCollection`.

        .. note::
            If versioning is not enabled on the service, None is returned.
        """
        if (
            "hasVersionedData" in self.properties
            and self.properties.hasVersionedData == True
        ):
            if self._vermgr is None:
                from ._version import VersionManager
                import os

                url = os.path.dirname(self.url) + "/VersionManagementServer"
                self._vermgr = VersionManager(url=url, gis=self._gis)
            return self._vermgr
        return None

    # ----------------------------------------------------------------------
    def query_domains(self, layers: Union[tuple, list[int]]):
        """
        Returns full domain information for the domains
        referenced by the layers in the :class:`~arcgis.features.FeatureLayerCollection`. This
        operation is performed on a feature layer collection. The operation
        takes an array of layer IDs and returns the set of domains referenced
        by the layers.

        .. note::
            See the :attr:`~arcgis.features.FeatureLayerCollection.query` method for a similar function.

        ================================     ====================================================================
        **Parameter**                         **Description**
        --------------------------------     --------------------------------------------------------------------
        layers                               Required List.  An array of layers. The set of domains to return is
                                             based on the domains referenced by these layers. Example: [1,2,3,4]
        ================================     ====================================================================

        :return:
            List of dictionaries

        """
        if (
            "supportsQueryDomains" in self.properties
            and self.properties["supportsQueryDomains"]
        ):
            if not isinstance(layers, (tuple, list)):
                raise ValueError("The layer variable must be a list.")
            url = "{base}/queryDomains".format(base=self._url)
            params = {"f": "json"}
            params["layers"] = layers
            res = self._con.post(url, params)
            if "domains" in res:
                return res["domains"]
            return res
        return []

    # ----------------------------------------------------------------------
    def extract_changes(
        self,
        layers: list[int],
        servergen: list[int] = None,
        layer_servergen: list[dict[str, Any]] = None,
        queries: Optional[dict[str, Any]] = None,
        geometry: Optional[Union[Geometry, dict[str, int]]] = None,
        geometry_type: Optional[str] = None,
        in_sr: Optional[Union[dict[str, Any], int]] = None,
        version: Optional[str] = None,
        return_inserts: bool = False,
        return_updates: bool = False,
        return_deletes: bool = False,
        return_ids_only: bool = False,
        return_extent_only: bool = False,
        return_attachments: bool = False,
        attachments_by_url: bool = False,
        data_format: str = "json",
        change_extent_grid_cell: Optional[str] = None,
        return_geometry_updates: Optional[bool] = None,
        fields_to_compare: list | None = None,
        out_sr: int | None = None,
    ):
        """
        A change tracking mechanism for applications. Applications can use ``extract_changes`` to
        query changes that have been made to the layers and tables in the service.

        .. note::
            For Enterprise geodatabase based feature services published
            from ArcGIS Pro 2.2 or higher, the ``ChangeTracking`` capability
            requires all layers and tables to be either archive enabled or
            branch versioned and have globalid columns.

        Change tracking can also be enabled for ArcGIS Online hosted feature services. If all layers
        and tables in the service have the ChangeTracking capability, the
        ``extract_changes`` operation can be used to get changes.

        ================================     ====================================================================
        **Parameter**                         **Description**
        --------------------------------     --------------------------------------------------------------------
        layers                               Required List.  The list of layers (by index value) and tables to include in the
                                             output.
        --------------------------------     --------------------------------------------------------------------
        servergen                            Required List (when layer_servergen not present). Introduced at 11.0.
                                             This parameter sets the servergens to apply to all layers included in
                                             the layers parameter. Either a single generation, or a pair of
                                             generations, can be used as values for this parameter. If a single
                                             servergen value is provided, all changes that have happened since
                                             that generation are returned. If a pair of serverGen values are
                                             provided, changes that have happened between the first generation
                                             (the minimum value) and the second generation (the maximum value)
                                             are returned. If providing two generations, the first value in the
                                             pair is expected to be the smaller of the two values.
                                             Support for this parameter is indicated when the service-level
                                             'supportServerGens' property, under 'extractChangesCapabilities', is
                                             set as 'True'. This operation requires either 'serverGens' or
                                             'layerServerGens' be submitted with the request.

                                             .. code-block:: python

                                                # Usage Example:

                                                servergen= [10500,11000]
        --------------------------------     --------------------------------------------------------------------
        layer_servergen                      Required List (when servergen not present). The servergen numbers allow a client to specify the last
                                             layer generation numbers (a Unix epoch time value in milliseconds) for the
                                             changes received from the server. All changes made after this value will be
                                             returned.

                                                + ``minServerGen``: It is the min generation of the server data changes.
                                                  Clients with layerServerGens that is less than minServerGen cannot
                                                  extract changes and would need to make a full server/layers query
                                                  instead of extracting changes.
                                                + ``serverGen``: It is the current server generation number of the
                                                  changes. Every changed feature has a version or a generation number
                                                  that is changed every time the feature is updated.

                                             Syntax:
                                                 servergen= [{"id": <layerId1>, "serverGen": <genNum1>}, {"id": <layerId2>, "serverGen": <genNum2>}]

                                             The ``id`` value for the layer is the index of the layer from the :attr:`layers`
                                             attribute on the :class:`~arcgis.features.FeatureLayerCollection`. The ``serverGen`` value is a Unix epoch timestamp value in milliseconds.

                                             .. code-block:: python

                                                # Usage Example:

                                                layer_servergen= [{"id": 0, "serverGen": 10500},
                                                                  {"id": 1, "serverGen": 1100},
                                                                  {"id": 2, "serverGen": 1200}]
        --------------------------------     --------------------------------------------------------------------
        queries                              Optional Dictionary. In addition to the layers and geometry
                                             parameters, the `queries` parameter can be used to further define
                                             what changes to return. This parameter allows you to set query
                                             properties on a per-layer or per-table basis. If a layer's ID is
                                             present in the layers parameter and missing from layer `queries`,
                                             it's changed features that intersect with the filter geometry are
                                             returned.

                                             The properties include the following:

                                                + ``where`` - Defines an attribute query for a layer or table. The
                                                  default is no where clause.
                                                + ``useGeometry`` - Determines whether or not to apply the geometry
                                                  for the layer. The default is true. If set to false, features
                                                  from the layer that intersect the geometry are not added.
                                                + ``includeRelated`` - Determines whether or not to add related
                                                  rows. The default is true. The value true is honored only
                                                  for queryOption=none. This is only applicable if your data
                                                  has relationship classes. Relationships are only processed
                                                  in a forward direction from origin to destination.
                                                + ``queryOption`` - Defines whether or how filters will be applied
                                                  to a layer. The queryOption was added in 10.2. See the
                                                  `Compatibility notes <https://developers.arcgis.com/rest/services-reference/sync-compatibility-notes.htm>`_ topic for more information.
                                                  Valid values are ``None``, ``useFilter``, or ``all``. See also the
                                                  ``layerQueries`` column in the Request Parameters table in the `Extract Changes (Feature Service) help <https://developers.arcgis.com/rest/services-reference/extract-changes-feature-service-.htm>`_
                                                  for details and code samples.

                                                * When the value is none, no feature are returned based on where and filter geometry.
                                                * If ``includeRelated`` is false, no features are returned.
                                                * If ``includeRelated`` is true, features in this layer (that are related to the features in other layers in the replica) are returned.
                                                * When the value is ``useFilter``, features that satisfy filtering based on geometry and ``where`` are returned. The value of ``includeRelated`` is ignored.

                                             .. code-block:: python

                                                # Usage Example:

                                                queries={Layer_or_tableID1:{"where":"attribute query",
                                                                            "useGeometry": true | false,
                                                                            "includeRelated": true | false},
                                                         Layer_or_tableID2: {.}}
        --------------------------------     --------------------------------------------------------------------
        geometry                             Optional :class:`~arcgis.geometry.Geometry`/:class:`~arcgis.geometry.Extent`.
                                             The geometry to apply as the spatial filter for the changes. All the changed
                                             features in layers intersecting this geometry will be returned. The structure
                                             of the geometry is the same as the structure of the `JSON geometry objects <https://developers.arcgis.com/documentation/common-data-types/geometry-objects.htm>`_
                                             returned by the ArcGIS REST API. In addition to the JSON structures,
                                             for envelopes and points you can specify the geometry with a simpler
                                             comma-separated syntax.
        --------------------------------     --------------------------------------------------------------------
        geometry_type                        Optional String. The type of geometry specified by the geometry
                                             parameter. The geometry type can be an envelope, point, line or
                                             polygon. The default geometry type is an envelope.

                                             Values: ``esriGeometryPoint``, ``esriGeometryMultipoint``, ``esriGeometryPolyline``, ``esriGeometryPolygon``, ``esriGeometryEnvelope``
        --------------------------------     --------------------------------------------------------------------
        in_sr                                Optional Integer. The spatial reference of the input geometry.
        --------------------------------     --------------------------------------------------------------------
        out_sr                               Optional Integer/String. The output spatial reference of the
                                             returned changes.
        --------------------------------     --------------------------------------------------------------------
        version                              Optional String. If branch versioning is enabled, a user can specify
                                             the branch version name to extract changes from.
        --------------------------------     --------------------------------------------------------------------
        return_inserts                       Optional Boolean.  If true, newly inserted features will be
                                             returned. The default is false.
        --------------------------------     --------------------------------------------------------------------
        return_updates                       Optional Boolean. If true, updated features will be returned. The
                                             default is false.
        --------------------------------     --------------------------------------------------------------------
        return_deletes                       Optional Boolean. If true, deleted features will be returned. The
                                             default is false.
        --------------------------------     --------------------------------------------------------------------
        return_ids_only                      Optional Boolean. If true, the response includes an array of object
                                             IDs only. The default is false.
        --------------------------------     --------------------------------------------------------------------
        return_attachments                   Optional Boolean.  If true, attachments changes are returned in the
                                             response. Otherwise, attachments are not included. The default is
                                             false. This parameter is only applicable if the feature service has
                                             attachments.
        --------------------------------     --------------------------------------------------------------------
        attachments_by_url                   Optional Boolean.  If true, a reference to a URL will be provided
                                             for each attachment returned. Otherwise, attachments are embedded in
                                             the response. The default is true.
        --------------------------------     --------------------------------------------------------------------
        data_format                          Optional String. The format of the changes returned in the response.
                                             The default is json. Values: sqllite or json
        --------------------------------     --------------------------------------------------------------------
        change_extent_grid_cell              Optional String. To optimize localizing changes extent, the value
                                             medium is an 8x8 grid that bound the changes extent. Used only when
                                             `return_extent_only` is true. The default is none.
                                             Values: None, large, medium, or small
        --------------------------------     --------------------------------------------------------------------
        return_geometry_updates              Optional Boolean. If true, the response includes a
                                             'hasGeometryUpdates' property set as true for each layer with
                                             updates that have geometry changes. The default is false.

                                             If a layer's edits include only inserts, deletes, or updates to
                                             fields other than geometry, hasGeometryUpdates is not set or is
                                             returned as false. When a layer has multiple rows with updates,
                                             only one needs to include a geometry changes for
                                             `hasGeometryUpdates` to be set as true.
        --------------------------------     --------------------------------------------------------------------
        fields_to_compare                    Optional List. Introduced at 11.0. This parameter allows you to
                                             determine if any array of fields has been updated. The accepted
                                             values for this parameter is a fields array that include the fields
                                             you want to evaluate. The response includes a fieldUpdates array,
                                             which includes rows that contain any updates made to the specified
                                             fields. If no updates were made to any fields, the fieldUpdates
                                             array is empty.
        ================================     ====================================================================

        :return:
            A dictionary containing the layerServerGens and an array of edits


        .. code-block:: python

           #Usage Example for extracting all changes to a feaature layer in a particular version since the time the Feature Layer was created.

           from arcgis.gis import GIS
           from arcgis.features import FeatureLayerCollection

           >>> gis = GIS(<url>, <username>, <password>)

           # Search for the Feature Service item
           >>> fl_item = gis.content.search('title:"my_feature_layer" type:"Feature Layer"')[0]
           >>> created_time = fl_item.created

           # Get the Feature Service url
           >>> fs=gis.content.search('title:"my_feature_layer" type:"Feature"')[0].url

           # Instantiate the a FeatureLayerCollection from the url
           >>> flc=FeatureLayerCollection(fs, gis)

           # Extract the changes for the version
           >>> extracted_changes=flc.extract_changes(layers=[0],
                                      servergen=[{"id": 0, "serverGen": created_time}],
                                      version="<version_owner>.<version_name>",
                                      return_ids_only=True,
                                      return_inserts=True,
                                      return_updates=True,
                                      return_deletes=True,
                                      data_format="json")

           >>> extracted_changes

           {'layerServerGens': [{'id': 0, 'serverGen': 1600713614620}],
            'edits': [{'id': 0,
              'objectIds': {'adds': [], 'updates': [194], 'deletes': []}}]}
        """
        if servergen is None and layer_servergen is None:
            raise ValueError("Please provide a servergen or layer_servergen")
        url = "%s/extractChanges" % self._url
        params = {
            "f": "json",
            "layerQueries": queries or "",
            "layers": layers,  # ",".join([str(lyr) for lyr in layers]),
            "geometry": geometry or "",
            "outSR": out_sr or "",
            "geometryType": geometry_type or "esriGeometryEnvelope",
            "inSR": in_sr or "",
            "gdbVersion": version or "",
            "returnInserts": return_inserts,
            "returnUpdates": return_updates,
            "returnDeletes": return_deletes,
            "returnDeletedFeatures": return_deletes,
            "returnIdsOnly": return_ids_only,
            "returnExtentOnly": return_extent_only,
            "returnAttachments": return_attachments,
            "returnAttachmentsDatabyURL": attachments_by_url,
            "dataFormat": data_format,
            "serverGens": servergen or "",
            "layerServerGens": layer_servergen or "",
            "changesExtentGridCell": change_extent_grid_cell,
            "fieldsToCompare": None or "",
            "async": True,
        }
        if not fields_to_compare is None:
            params["fieldsToCompare"] = {"fields": fields_to_compare}
        else:
            del params["fieldsToCompare"]
        if not return_geometry_updates is None:
            params["returnHasGeometryUpdates"] = return_geometry_updates
        res = self._con.post(url, params)
        if "statusUrl" in res:
            surl = res["statusUrl"]
            params = {"f": "json"}
            res = self._con.get(surl, params)
            while res["status"].lower() != "completed":
                res = self._con.get(surl, params)
                status = res["status"]
                if status.lower() == "completed":
                    res = self._con.get(res["resultUrl"])
                    break
                elif status.lower() == "failed":
                    return None
                else:
                    time.sleep(0.5)
            if "status" in res and res["status"].lower() == "completed":
                res = self._con.get(res["resultUrl"])
        if (
            isinstance(res, str)
            and os.path.isfile(res)
            and str(data_format).lower() == "json"
        ):
            with open(res, "r") as reader:
                return json.loads(reader.read())
        return res

    def query(
        self,
        layer_defs_filter: Optional[LayerDefinitionFilter] = None,
        geometry_filter: Optional[GeometryFilter] = None,
        time_filter: Optional[TimeFilter] = None,
        return_geometry: bool = True,
        return_ids_only: bool = False,
        return_count_only: bool = False,
        return_z: bool = False,
        return_m: bool = False,
        out_sr: Optional[int] = None,
    ):
        """
         Queries the current :class:`~arcgis.features.FeatureLayerCollection` based on ``sql``
         statement.

        ===============================     ====================================================================
        **Parameter**                        **Description**
        -------------------------------     --------------------------------------------------------------------
        time_filter                         Optional list. The format is of `[<startTime>, <endTime>]` using
                                            datetime.date, datetime.datetime or timestamp in milliseconds.
                                            Syntax: time_filter=[<startTime>, <endTime>] ; specified as
                                                    datetime.date, datetime.datetime or timestamp in
                                                    milliseconds
        -------------------------------     --------------------------------------------------------------------
        geometry_filter                     Optional from arcgis.geometry.filter. Allows for the information to
                                            be filtered on spatial relationship with another geometry.
        -------------------------------     --------------------------------------------------------------------
        layer_defs_filter                   Optional Layer Definition Filter.
        -------------------------------     --------------------------------------------------------------------
        return_geometry                     Optional boolean. If true, geometry is returned with the query.
                                            Default is true.
        -------------------------------     --------------------------------------------------------------------
        return_ids_only                     Optional boolean. Default is False.  If true, the response only
                                            includes an array of object IDs. Otherwise, the response is a
                                            feature set.
        -------------------------------     --------------------------------------------------------------------
        return_count_only                   Optional boolean. If true, the response only includes the count
                                            (number of features/records) that would be returned by a query.
                                            Otherwise, the response is a feature set. The default is false. This
                                            option supersedes the returnIdsOnly parameter. If
                                            returnCountOnly = true, the response will return both the count and
                                            the extent.
        -------------------------------     --------------------------------------------------------------------
        return_z                            Optional boolean. If true, Z values are included in the results if
                                            the features have Z values. Otherwise, Z values are not returned.
                                            The default is False.
        -------------------------------     --------------------------------------------------------------------
        return_m                            Optional boolean. If true, M values are included in the results if
                                            the features have M values. Otherwise, M values are not returned.
                                            The default is false.
        -------------------------------     --------------------------------------------------------------------
        out_sr                              Optional Integer. The ``WKID`` for the spatial reference of the returned
                                            geometry.
        ===============================     ====================================================================

        :return:
            A :class:`~arcgis.features.FeatureSet` of the queried Feature Layer Collection unless
            ``return_count_only`` or ``return_ids_only`` is True.

        """
        qurl = self._url + "/query"
        params = {
            "f": "json",
            "returnGeometry": return_geometry,
            "returnIdsOnly": return_ids_only,
            "returnCountOnly": return_count_only,
            "returnZ": return_z,
            "returnM": return_m,
        }
        if layer_defs_filter is not None and isinstance(layer_defs_filter, dict):
            params["layerDefs"] = layer_defs_filter
        elif layer_defs_filter is not None and isinstance(layer_defs_filter, dict):
            pass
        if geometry_filter is not None and isinstance(geometry_filter, dict):
            params["geometryType"] = geometry_filter["geometryType"]
            params["spatialRel"] = geometry_filter["spatialRel"]
            params["geometry"] = geometry_filter["geometry"]
            if "inSR" in geometry_filter:
                params["inSR"] = geometry_filter["inSR"]

        if out_sr is not None and isinstance(out_sr, SpatialReference):
            params["outSR"] = out_sr
        elif out_sr is not None and isinstance(out_sr, dict):
            params["outSR"] = out_sr
        if time_filter is not None and isinstance(time_filter, dict):
            params["time"] = time_filter
        results = self._con.get(path=qurl, params=params)
        if "error" in results:
            raise ValueError(results)
        if not return_count_only and not return_ids_only:
            return results
        else:
            return FeatureSet.from_dict(results)

    # ----------------------------------------------------------------------
    def query_data_elements(self, layers: list) -> dict:
        """
        The `query_data_elements` provides access to valuable information
        for datasets exposed through a feature service such as a feature
        layer, a table or a utility network layer. The response is
        dependent on the type of layer that is queried.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        layers                     Required list. Array of layerIds for which to get the data elements.
        ======================     ====================================================================

        :returns: dict

        """
        if (
            "supportsQueryDataElements" in self.properties
            and self.properties.supportsQueryDataElements
        ):
            url = f"{self._url}/queryDataElements"
            params = {"f": "json", "layers": layers}
            return self._con.get(url, params)
        return []

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
    ):
        """
        The ``query_related_records`` operation is performed on a :class:`~arcgis.features.FeatureLayerCollection`
        resource. The result of this operation are feature sets grouped
        by source :class:`~arcgis.features.FeatureLayer`/:class:`~arcgis.features.Table` object IDs.
        Each feature set contains :class:`~arcgis.features.Feature` objects including the values for the fields
        requested by the :class:`~arcgis.gis.User`. For related layers, if you request geometry
        information, the geometry of each feature is also returned in
        the feature set. For related tables, the feature set does not
        include geometries.

        .. note::
            See the :attr:`~arcgis.features.FeatureLayerCollection.query` method for a similar function.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        object_ids                 Optional string. the object IDs of the table/layer to be queried.
        ----------------------     --------------------------------------------------------------------
        relationship_id            Optional string. The ID of the relationship to be queried.
        ----------------------     --------------------------------------------------------------------
        out_fields                 Optional string.the list of fields from the related table/layer
                                   to be included in the returned feature set. This list is a comma
                                   delimited list of field names. If you specify the shape field in the
                                   list of return fields, it is ignored. To request geometry, set
                                   return_geometry to true. You can also specify the wildcard "*" as the
                                   value of this parameter. In this case, the results will include all
                                   the field values.
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
                                   the outSR. If outSR is not specified, then max_allowable_offset is
                                   assumed to be in the unit of the spatial reference of the map.
        ----------------------     --------------------------------------------------------------------
        geometry_precision         Optional integer. This option can be used to specify the number of
                                   decimal places in the response geometries.
        ----------------------     --------------------------------------------------------------------
        out_wkid                   Optional integer. The spatial reference of the returned geometry.
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
        ======================     ====================================================================


        :return: Dictionary of query results

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
        qrr_url = self._url + "/queryRelatedRecords"
        res = self._con.get(path=qrr_url, params=params)
        return res

    # ----------------------------------------------------------------------
    @property
    def _replicas(self):
        """returns all the replicas for a feature service"""
        params = {
            "f": "json",
        }
        url = self._url + "/replicas"
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def _unregister_replica(self, replica_id):
        """
        Removes a replica from a feature service

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        replica_id          Optional string. The replica_id returned by the feature service when
                            the replica was created.
        ===============     ====================================================================


        :return: boolean

        """
        params = {"f": "json", "replicaID": replica_id}
        url = self._url + "/unRegisterReplica"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def _replica_info(self, replica_id):
        """
        The replica info resources lists replica metadata for a specific replica.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        replica_id          Optional string. The replica_id returned by the feature service when
                            the replica was created.
        ===============     ====================================================================

        :return: dict

        """
        params = {"f": "json"}
        url = self._url + "/replicas/" + replica_id
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def _create_replica(
        self,
        replica_name,
        layers,
        layer_queries=None,
        geometry_filter=None,
        replica_sr=None,
        transport_type="esriTransportTypeUrl",
        return_attachments=False,
        return_attachments_data_by_url=False,
        asynchronous=False,
        sync_direction=None,
        target_type="client",
        attachments_sync_direction="none",
        sync_model="none",
        data_format="json",
        replica_options=None,
        wait=False,
        out_path=None,
        transformations=None,
        time_reference_unknown_client=None,
    ):
        """
        The createReplica operation is performed on a feature service
        resource. This operation creates the replica between the feature
        service and a client based on a client-supplied replica definition.
        It requires the Sync capability. See Sync overview for more
        information on sync. The response for createReplica includes
        replicaID, server generation number, and data similar to the
        response from the feature service query operation.
        The createReplica operation returns a response of type
        esriReplicaResponseTypeData, as the response has data for the
        layers in the replica. If the operation is called to register
        existing data by using replicaOptions, the response type will be
        esriReplicaResponseTypeInfo, and the response will not contain data
        for the layers in the replica.

        =============================   ====================================================================
        **Parameter**                    **Description**
        -----------------------------   --------------------------------------------------------------------
        replicaName                     Optional string. The name of the replica
        -----------------------------   --------------------------------------------------------------------
        layers                          The layers to export
        -----------------------------   --------------------------------------------------------------------
        layer_queries                   In addition to the layers and geometry parameters, the layerQueries
                                        parameter can be used to further define what is replicated. This
                                        parameter allows you to set properties on a per layer or per table
                                        basis. Only the properties for the layers and tables that you want
                                        changed from the default are required.

                                        #Example:
                                        layerQueries = {"0":{"queryOption": "useFilter", "useGeometry": true,
                                                        "where": "requires_inspection = Yes"}}
        -----------------------------   --------------------------------------------------------------------
        geometry_filter                 Spatial filter from arcgis.geometry.filters module to filter results by a
                                        spatial relationship with another geometry.
                                        Only intersections are currently supported
        -----------------------------   --------------------------------------------------------------------
        return_attachments              Optional boolean. If true, attachments are added to the replica and returned in the
                                        response. Otherwise, attachments are not included.
        -----------------------------   --------------------------------------------------------------------
        return_attachment_databy_url    If true, a reference to a URL will be provided for each
                                        attachment returned from createReplica. Otherwise,
                                        attachments are embedded in the response.
        -----------------------------   --------------------------------------------------------------------
        replica_sr                      The spatial reference of the replica geometry
        -----------------------------   --------------------------------------------------------------------
        transport_type                  The transportType represents the response format. If the
                                        transportType is esriTransportTypeUrl, the JSON response is contained in a file,
                                        and the URL link to the file is returned. Otherwise, the JSON object is returned
                                        directly. The default is esriTransportTypeUrl.
                                        If async is true, the results will always be returned as if transportType is
                                        esriTransportTypeUrl. If dataFormat is sqlite, the transportFormat will always be
                                        esriTransportTypeUrl regardless of how the parameter is set.

                                        Values: esriTransportTypeUrl | esriTransportTypeEmbedded
        -----------------------------   --------------------------------------------------------------------
        attachments_sync_direction      Client can specify the attachmentsSyncDirection when
                                        creating a replica. AttachmentsSyncDirection is currently a createReplica property
                                        and cannot be overridden during sync.

                                        Values: none, upload, bidirectional
        -----------------------------   --------------------------------------------------------------------
        asynchronous                    If true, the request is processed as an asynchronous job, and a URL is
                                        returned that a client can visit to check the status of the job. See the topic on
                                        asynchronous usage for more information. The default is false.
        -----------------------------   --------------------------------------------------------------------
        sync_model                      Client can specify the attachmentsSyncDirection when creating a replica.
                                        AttachmentsSyncDirection is currently a createReplica property and cannot be
                                        overridden during sync.
        -----------------------------   --------------------------------------------------------------------
        data_format                     The format of the replica geodatabase returned in the response. The
                                        default is json.

                                        Values: filegdb, json, sqlite, shapefile
        -----------------------------   --------------------------------------------------------------------
        target_type                     This option was added at 10.5.1. Can be set to either server or client.
                                        If not set, the default is client.A targetType of client will generate a replica that
                                        matches those generated in pre-10.5.1 releases. These are designed to support syncing
                                        with lightweight mobile clients and have a single generation number (serverGen or
                                        replicaServerGen).
                                        A targetType of server generates a replica that supports syncing in one direction
                                        between 2 feature services running on servers or between an ArcGIS Server feature
                                        service and an ArcGIS Online feature service. When the targetType is server, the replica
                                        information includes a second generation number. This second generation number is called
                                        replicaServerSibGen for perReplica types and serverSibGen for perLayer types.
                                        target_type server replicas generated with dataFormat SQLite can be published as new
                                        services in another ArcGIS Online organization or in ArcGIS Enterprise. When published,
                                        a replica is generated on these new services with a matching replicaID and a
                                        replicaServerSibGen or serverSibGens. The replicaServerSibGen or serverSibGens values
                                        can be used as the replicaServerGen or serverGen values when calling synchronize replica
                                        on the source service to get the latest changes. These changes can then be imported into
                                        the new service using the synchronizeReplica operation. When calling synchronizeReplica
                                        on the new service to import the changes, be sure to pass the new replicaServerGen or
                                        serverGen from the source service as the replicaServerSibGen or serverSibGen. This will
                                        update the replica metadata appropriately such that it can be used in the next sync.

                                        Values: server, client
        -----------------------------   --------------------------------------------------------------------
        sync_direction                  Defaults to bidirectional when the targetType is client and download
                                        when the targetType is server. If set, only bidirectional is supported when
                                        targetType is client. If set, only upload or download are supported when targetType is
                                        server.
                                        A syncDirection of bidirectional matches the functionality from replicas generated in
                                        pre-10.5.1 releases and allows upload and download of edits. It is only supported
                                        when targetType is client.
                                        When targetType is server, only a one way sync is supported thus only upload or
                                        download are valid options.
                                        A syncDirection of upload means that the synchronizeReplica operation allows only sync
                                        with an upload direction. Use this option to allow the upload of edits from the source
                                        service.
                                        A syncDirection of download means that the synchronizeReplica operation allows only sync
                                        with a download direction. Use this option to allow the download of edits to provide to
                                        the source service.
        -----------------------------   --------------------------------------------------------------------
        replica_options                 This parameter instructs the createReplica operation to create a
                                        new replica based on an existing replica definition (refReplicaId). It can be used
                                        to specify parameters for registration of existing data for sync. The operation
                                        will create a replica but will not return data. The responseType returned in the
                                        createReplica response will be esriReplicaResponseTypeInfo.
        -----------------------------   --------------------------------------------------------------------
        wait                            If async, wait to pause the process until the async operation is completed.
        -----------------------------   --------------------------------------------------------------------
        out_path                        Folder path to save the file
        -----------------------------   --------------------------------------------------------------------
        transformations                 Optional List. Introduced at 10.8. This parameter applies a datum
                                        transformation on each layer when the spatial reference used in
                                        geometry is different than the layer's spatial reference.
        -----------------------------   --------------------------------------------------------------------
        time_reference_unknown_client   Setting timeReferenceUnknownClient as true indicates that the client is
                                        capable of working with data values that are not in UTC. If its not set
                                        to true, and the service layer's datesInUnknownTimeZone property is true,
                                        then an error is returned. The default is false

                                        Its possible to define a service's time zone of date fields as unknown.
                                        Setting the time zone as unknown means that date values will be returned
                                        as-is from the database, rather than as date values in UTC. Non-hosted feature
                                        services can be set to use an unknown time zone using ArcGIS Server Manager.
                                        Setting the time zones to unknown also sets the datesInUnknownTimeZone layer property
                                        as true. Currently, hosted feature services do not support this setting.
                                        This setting does not apply to editor tracking date fields which are
                                        stored and returned in UTC even when the time zone is set to unknown.

                                        Most clients released prior to ArcGIS Enterprise 10.9 will not be able
                                        to work with feature services that have an unknown time setting.
                                        The timeReferenceUnknownClient parameter prevents these clients from working
                                        with the service in order to avoid problems..
                                        Setting this parameter to true indicates that the client is capable of working with
                                        unknown date values that are not in UTC.
        =============================   ====================================================================

        :return: The created replica

        """
        if (
            not self.properties.syncEnabled
            and "Extract" not in self.properties.capabilities
        ):
            return None
        url = self._url + "/createReplica"
        dataformat = ["filegdb", "json", "sqlite", "shapefile"]
        params = {
            "f": "json",
            "replicaName": replica_name,
            "returnAttachments": json.dumps(return_attachments),
            "returnAttachmentsDatabyUrl": json.dumps(return_attachments_data_by_url),
            "async": json.dumps(asynchronous),
            "syncModel": sync_model,
            "layers": layers,
            "targetType": target_type,
        }
        if transformations:
            params["datumTransformations"] = transformations
        if attachments_sync_direction:
            params["attachmentsSyncDirection"] = attachments_sync_direction
        if sync_direction:
            params["syncDirection"] = sync_direction
        if data_format.lower() in dataformat:
            params["dataFormat"] = data_format.lower()
        else:
            raise Exception("Invalid dataFormat")
        if layer_queries is not None:
            params["layerQueries"] = layer_queries
        if geometry_filter is not None and isinstance(geometry_filter, dict):
            params["geometry"] = geometry_filter["geometry"]
            params["geometryType"] = geometry_filter["geometryType"]
            if "inSR" in geometry_filter:
                params["inSR"] = geometry_filter["inSR"]
        if replica_sr is not None:
            params["replicaSR"] = replica_sr
        if replica_options is not None:
            params["replicaOptions"] = replica_options
        if transport_type is not None:
            params["transportType"] = transport_type
        # parameter added at version 10.9
        if self._gis.version >= [8, 4]:
            params["timeReferenceUnknownClient"] = time_reference_unknown_client
        if asynchronous:
            if wait:
                export_job = self._con.post(path=url, postdata=params)
                status = self._replica_status(url=export_job["statusUrl"])
                while status["status"] not in (
                    "Completed",
                    "CompletedWithErrors",
                ):
                    if status["status"] == "Failed":
                        return status
                    # wait before checking again
                    time.sleep(2)
                    status = self._replica_status(url=export_job["statusUrl"])

                res = status

            else:
                res = self._con.post(path=url, postdata=params)
        else:
            res = self._con.post(path=url, postdata=params)

        if out_path is not None and os.path.isdir(out_path):
            dl_url = None
            if "resultUrl" in res:
                dl_url = res["resultUrl"]
            elif "responseUrl" in res:
                dl_url = res["responseUrl"]

            if dl_url is not None:
                return self._con.get(
                    path=dl_url,
                    file_name=dl_url.split("/")[-1],
                    out_folder=out_path,
                    try_json=False,
                )

            else:
                return res
        elif res is not None:
            return res
        return None

    # ----------------------------------------------------------------------
    def _cleanup_change_tracking(
        self,
        layers,
        retention_period,
        period_unit="days",
        min_server_gen=None,
        replica_id=None,
        future=False,
    ):
        """



        :return: Boolean

        """
        url = "{url}/cleanupChangeTracking".format(url=self._url)
        url = url.replace("/rest/services/", "/rest/admin/services/")
        params = {
            "f": "json",
            "layers": layers,
            "retentionPeriod": retention_period,
            "retentionPeriodUnits": period_unit,
        }
        if min_server_gen:
            params["minServerGen"] = min_server_gen
        if replica_id:
            params["replicaId"] = replica_id
        if future:
            params["async"] = future
            res = self._con.post(url, params)
            if "statusUrl" in res:
                import concurrent.futures

                executor = concurrent.futures.ThreadPoolExecutor(1)
                res = self._con.post(path=url, postdata=params)
                future = executor.submit(
                    self._status_via_url,
                    *(self._con, res["statusUrl"], {"f": "json"}),
                )
                executor.shutdown(False)
                return future
            return res
        else:
            res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

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
        while not status["status"] in status_allowed:
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
    def _synchronize_replica(
        self,
        replica_id,
        transport_type="esriTransportTypeUrl",
        replica_server_gen=None,
        replica_servers_sib_gen=None,
        return_ids_for_adds=False,
        edits=None,
        return_attachment_databy_url=False,
        asynchronous=False,
        sync_direction=None,
        sync_layers="perReplica",
        edits_upload_id=None,
        edits_upload_format=None,
        data_format="json",
        rollback_on_failure=True,
        close_replica=False,
        out_path=None,
    ):
        """
        The synchronizeReplica operation is performed on a feature service resource. This operation
        synchronizes changes between the feature service and a client based on the replicaID
        provided by the client. Requires the sync capability. See Sync overview for more information
        on sync.
        The client obtains the replicaID by first calling the _create_replica operation.
        Synchronize applies the client's data changes by importing them into the server's
        geodatabase. It then exports the changes from the server geodatabase that have taken place
        since the last time the client got the data from the server. Edits can be supplied in the
        edits parameter, or, alternatively, by using the editsUploadId and editUploadFormat to
        identify a file containing the edits that were previously uploaded using the upload_item
        operation.
        The response for this operation includes the replicaID, new replica generation number, or
        the layer's generation numbers. The response has edits or layers according to the
        syncDirection/syncLayers. Presence of layers and edits in the response is indicated by the
        responseType.
        If the responseType is esriReplicaResponseTypeEdits or esriReplicaResponseTypeEditsAndData,
        the result of this operation can include arrays of edit results for each layer/table edited
        as specified in edits. Each edit result identifies a single feature on a layer or table and
        indicates if the edits were successful or not. If an edit is not successful, the edit result
        also includes an error code and an error description.
        If syncModel is perReplica and syncDirection is download or bidirectional, the
        _synchronize_replica operation's response will have edits. If syncDirection is snapshot, the
        response will have replacement data.
        If syncModel is perLayer, and syncLayers have syncDirection as download or bidirectional,
        the response will have edits. If syncLayers have syncDirection as download or bidirectional
        for some layers and snapshot for some other layers, the response will have edits and data.
        If syncDirection for all the layers is snapshot, the response will have replacement data.
        When syncModel is perReplica, the createReplica and synchronizeReplica operations' responses
        contain replicaServerGen. When syncModel is perLayer, the createReplica and
        synchronizeReplica operations' responses contain layerServerGens.
        You can provide arguments to the synchronizeReplica operation as defined in the parameters
        table below.

        ===============                 ====================================================================
        **Parameter**                    **Description**
        ---------------                 --------------------------------------------------------------------
        replica_id                      The ID of the replica you want to synchronize.
        ---------------                 --------------------------------------------------------------------
        transport_type
        ---------------                 --------------------------------------------------------------------
        replica_server_gen              Is a generation number that allows the server to keep track of what
                                        changes have already been synchronized. A new replicaServerGen is sent with the response
                                        to the synchronizeReplica operation. Clients should persist this value and use it with the
                                        next synchronizeReplica call.
                                        It applies to replicas with syncModel = perReplica.
                                        For replicas with syncModel = perLayer, layer generation numbers are specified using
                                        parameter: syncLayers; and replicaServerSibGen is not needed.
        ---------------                 --------------------------------------------------------------------
        return_ids_for_adds             If true, the objectIDs and globalIDs of features added during the
                                        synchronize will be returned to the client in the addResults sections of the response.
                                        Otherwise, the IDs are not returned. The default is false.

                                        Values: true | false
        ---------------                 --------------------------------------------------------------------
        edits                           The edits the client wants to apply to the service. Alternatively, the
                                        edits_upload_ID and editsUploadFormat can be used to specify the edits in a delta file.
                                        The edits are described using an array where an element in the array includes:
                                        - The layer or table ID
                                        - The feature or row edits to apply listed as inserts, updates, and deletes
                                        - The attachments to apply listed as inserts, updates, and deletes
                                        For features, adds and updates are specified as feature objects that include geometry and
                                        attributes.
                                        Deletes can be specified using globalIDs for features and attachments.
                                        For attachments, updates and adds are specified using the following set of properties for
                                        each attachment. If embedding the attachment, set the data property; otherwise, set the url
                                        property. All other properties are required:
                                        - globalid - The globalID of the attachment that is to be added or updated.
                                        - parentGlobalid - The globalID of the feature associated with the attachment.
                                        - contentType - Describes the file type of the attachment (for example, image/jpeg).
                                        - name - The file name (for example, hydrant.jpg).
                                        - data - The base 64 encoded data if embedding the data. Only required if the attachment
                                            is embedded.
                                        - url - The location where the service will upload the attachment file (for example,
                                            http://machinename/arcgisuploads/Hydrant.jpg). Only required if the attachment is not
                                            embedded.
        ---------------                 --------------------------------------------------------------------
        return_attachment_databy_url    If true, a reference to a URL will be provided for each
                                        attachment returned from synchronizeReplica. Otherwise, attachments are embedded in the
                                        response. The default is true. Applies only if attachments are included in the replica.
        ---------------                 --------------------------------------------------------------------
        asynchronous                    If true, the request is processed as an asynchronous job and a URL is
                                        returned that a client can visit to check the status of the job. See the topic on
                                        asynchronous usage for more information. The default is false.
        ---------------                 --------------------------------------------------------------------
        sync_direction                  Determines whether to upload, download, or upload and download on sync. By
                                        default, a replica is synchronized bi-directionally. Only applicable when
                                        syncModel = perReplica. If syncModel = perLayer, sync direction is specified using
                                        syncLayers.

                                        Values: download | upload | bidirectional | snapshot

                                        - download-The changes that have taken place on the server since last download are
                                            returned. Client does not need to send any changes. If the changes are sent, service
                                            will ignore them.
                                        - upload-The changes submitted in the edits or editsUploadID/editsUploadFormatt
                                            parameters are applied, and no changes are downloaded from the server.
                                        - bidirectional-The changes submitted in the edits or editsUploadID/editsUploadFormat
                                            parameters are applied, and changes on the server are downloaded. This is the default
                                            value.
                                        - snapshot-The current state of the features is downloaded from the server. If any edits
                                            are specified, they will be ignored.
        ---------------                 --------------------------------------------------------------------
        sync_layers                     Allows a client to specify layer-level generation numbers for a sync
                                        operation. It can also be used to specify sync directions at layer-level. This parameter
                                        is needed for replicas with syncModel = perLayer. It is ignored for replicas with
                                        syncModel = perReplica.
                                        serverGen is required for layers with syncDirection = bidirectional or download.
                                        serverSibGen is needed only for replicas where the targetType = server. For replicas with
                                        syncModel = perLayer, the serverSibGen serves the same purpose at the layer level as the
                                        replicaServerSibGen does in the case of syncModel = perReplica. See the
                                        replicaServerSibGen parameter for more information.
                                        If a sync operation has both the syncDirection and syncLayersparameters, and the replica's
                                        syncModel is perLayer, the layers that do not have syncDirection values will use the value
                                        of the syncDirection parameter. If the syncDirection parameter is not specified, the
                                        default value of bidirectional is used.

                                        Values: download | upload | bidirectional | snapshot
        ---------------                 --------------------------------------------------------------------
        edits_upload_id                 The ID for the uploaded item that contains the edits the client wants to
                                        apply to the service. Used in conjunction with editsUploadFormat.
        ---------------                 --------------------------------------------------------------------
        edits_upload_format             The data format of the uploaded data reference in edit_upload_id.
                                        data_format="json"
        ---------------                 --------------------------------------------------------------------
        data_format                     The format of the replica geodatabase returned in the response. The
                                        default is json.

                                        Values: filegdb, json, sqlite, shapefile
        ---------------                 --------------------------------------------------------------------
        rollback_on_failure             Determines the behavior when there are errors while importing edits
                                        on the server during synchronization. This only applies in cases where edits are being
                                        uploaded to the server (syncDirection = upload or bidirectional). See the
                                        RollbackOnFailure and Sync Models topic for more details.
                                        When true, if an error occurs while importing edits on the server, all edits are rolled
                                        back (not applied), and the operation returns an error in the response. Use this setting
                                        when the edits are such that you will either want all or none applied.
                                        When false, if an error occurs while importing an edit on the server, the import process
                                        skips the edit and continues. All edits that were skipped are returned in the edits
                                        results with information describing why the edits were skipped.
        ---------------                 --------------------------------------------------------------------
        close_replica                   If true, the replica will be unregistered when the synchronize completes.
                                        This is the same as calling synchronize and then calling unregisterReplica. Otherwise, the
                                        replica can continue to be synchronized. The default is false.
        ---------------                 --------------------------------------------------------------------
        out_path                        Folder path to save the file
        ===============                 ====================================================================

        :returns:
        """

        url = "{url}/synchronizeReplica".format(url=self._url)
        params = {
            "f": "json",
            "replicaID": replica_id,
        }

        if transport_type is not None:
            params["transportType"] = transport_type
        if edits is not None:
            params["edits"] = edits
        if replica_server_gen is not None:
            params["replicaServerGen"] = replica_server_gen
        if return_ids_for_adds is not None:
            params["returnIdsForAdds"] = return_ids_for_adds
        if return_attachment_databy_url is not None:
            params["returnAttachmentDatabyURL"] = return_attachment_databy_url
        if asynchronous is not None:
            params["async"] = asynchronous
        if sync_direction is not None:
            params["syncDirection"] = sync_direction
        if sync_layers is not None:
            params["syncLayers"] = sync_layers
        if edits_upload_format is not None:
            params["editsUploadFormat"] = edits_upload_format
        if edits_upload_id is not None:
            params["editsUploadID"] = edits_upload_id
        if data_format is not None:
            params["dataFormat"] = data_format
        # if edits_upload_id:
        #    params['dataFormat'] = edits_upload_id
        if rollback_on_failure is not None:
            params["rollbackOnFailure"] = rollback_on_failure
        if close_replica:
            params["closeReplica"] = close_replica
        if replica_servers_sib_gen:
            params["replicaServerSibGen"] = replica_servers_sib_gen
        res = self._con.post(path=url, postdata=params)
        if out_path is not None and os.path.isdir(out_path):
            dl_url = None
            if "resultUrl" in res:
                dl_url = res["resultUrl"]
            elif "responseUrl" in res:
                dl_url = res["responseUrl"]
            elif "URL" in res:
                dl_url = res["URL"]
            if dl_url is not None:
                return self._con.get(
                    path=dl_url,
                    file_name=dl_url.split("/")[-1],
                    out_folder=out_path,
                    try_json=False,
                )
            else:
                return res
        return res

    # ----------------------------------------------------------------------
    def _replica_status(self, url):
        """gets the replica status when exported async set to True"""
        params = {"f": "json"}
        if url.lower().endswith("/status") == False:
            return self._con.get(path=url, params=params)
        else:
            url += "/status"
            return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def upload(
        self,
        path: Optional[str],
        description: Optional[str] = None,
        upload_size: Optional[int] = None,
    ):
        """
        The ``upload`` method uploads a new item to the server.

        .. note::
            Once the operation is completed successfully, item id of the uploaded item is returned.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        path                Optional string. Filepath of the file to upload.
        ---------------     --------------------------------------------------------------------
        description         Optional string. Descriptive text for the uploaded item.
        ---------------     --------------------------------------------------------------------
        upload_size         Optional Integer. For large uploads, a user can specify the upload
                            size of each part.  The default is 1mb.
        ===============     ====================================================================

        :return: Item id of uploaded item

        """
        if os.path.getsize(path) < 10e6:
            url = self._url + "/uploads/upload"
            params = {
                "f": "json",
                "filename": os.path.basename(path),
                "overwrite": True,
            }
            files = {}
            files["file"] = path
            if description:
                params["description"] = description
            res = self._con.post(path=url, postdata=params, files=files)
            if "error" in res:
                raise Exception(res)
            else:
                return res["item"]["itemID"]
        else:
            if upload_size is None:
                upload_size = 1e6
            file_path = path
            item_id = self._register_upload(file_path)
            self._upload_by_parts(item_id, file_path, size=upload_size)
            return self._commit_upload(item_id)

    # ----------------------------------------------------------------------
    def _register_upload(self, file_path):
        """returns the itemid for the upload by parts logic"""
        r_url = "%s/uploads/register" % self._url
        params = {"f": "json", "itemName": os.path.basename(file_path)}
        reg_res = self._con.post(r_url, params)
        if "item" in reg_res and "itemID" in reg_res["item"]:
            return reg_res["item"]["itemID"]
        return None

    # ----------------------------------------------------------------------
    def _upload_by_parts(self, item_id, file_path, size=1e6):
        """loads a file for attachmens by parts"""
        import mmap, tempfile

        size = int(size)
        b_url = "%s/uploads/%s" % (self._url, item_id)
        upload_part_url = "%s/uploadPart" % b_url
        params = {"f": "json"}
        with open(file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            steps = int(os.fstat(f.fileno()).st_size / size)
            if os.fstat(f.fileno()).st_size % size > 0:
                steps += 1
            for i in range(steps):
                files = {}
                tempFile = os.path.join(tempfile.gettempdir(), "split.part%s" % i)
                if os.path.isfile(tempFile):
                    os.remove(tempFile)
                with open(tempFile, "wb") as writer:
                    writer.write(mm.read(int(size)))
                    writer.flush()
                    writer.close()
                del writer
                files["file"] = tempFile
                params["partId"] = i + 1
                res = self._con.post(upload_part_url, postdata=params, files=files)
                if "error" in res:
                    raise Exception(res)
                os.remove(tempFile)
                del files
            del mm
        return True

    # ----------------------------------------------------------------------
    def _commit_upload(self, item_id):
        """commits an upload by parts upload"""
        b_url = "%s/uploads/%s" % (self._url, item_id)
        commit_part_url = "%s/commit" % b_url
        params = {"f": "json", "parts": self._uploaded_parts(itemid=item_id)}
        res = self._con.post(commit_part_url, params)
        if "error" in res:
            raise Exception(res)
        else:
            return res["item"]["itemID"]

    # ----------------------------------------------------------------------
    def _delete_upload(self, item_id):
        """commits an upload by parts upload"""
        b_url = "%s/uploads/%s" % (self._url, item_id)
        delete_part_url = "%s/delete" % b_url
        params = {
            "f": "json",
        }
        res = self._con.post(delete_part_url, params)
        if "error" in res:
            raise Exception(res)
        else:
            return res

    # ----------------------------------------------------------------------
    def _uploaded_parts(self, itemid):
        """
        returns the parts uploaded for a given item

        ==================   ==============================================
        Arguments           Description
        ------------------   ----------------------------------------------
        itemid               required string. Id of the uploaded by parts item.
        ==================   ==============================================

        """
        url = self._url + "/uploads/%s/parts" % itemid
        params = {"f": "json"}
        res = self._con.get(url, params)
        return ",".join(res["parts"])
